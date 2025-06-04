use crate::schema::json_schema_for;
use action_log::ActionLog;
use anyhow::{Result, anyhow};
use assistant_tool::{Tool, ToolResult};
use gpui::{AnyWindowHandle, App, Entity, Task};
use language::ToOffset;
use language_model::{LanguageModel, LanguageModelRequest, LanguageModelToolSchemaFormat};
use project::Project;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{fmt::Write, path::Path, sync::Arc};
use ui::IconName;
use util::markdown::MarkdownInlineCode;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SymbolDefinitionToolInput {
    /// The path of the file containing the symbol.
    ///
    /// This path should never be absolute, and the first component
    /// of the path should always be a root directory in a project.
    ///
    /// <example>
    /// If the project has the following root directories:
    ///
    /// - frontend
    /// - backend
    ///
    /// If you want to find a symbol in `main.rs` in `backend/src`, you should use the path `backend/src/main.rs`.
    /// </example>
    pub path: String,

    /// The line number where the symbol is located (1-based).
    pub line: u32,

    /// The column number where the symbol is located (1-based).
    pub column: u32,
}

pub struct SymbolDefinitionTool;

impl Tool for SymbolDefinitionTool {
    fn name(&self) -> String {
        "symbol_definition".into()
    }

    fn needs_confirmation(&self, _: &serde_json::Value, _: &Entity<Project>, _: &App) -> bool {
        false
    }

    fn may_perform_edits(&self) -> bool {
        false
    }

    fn description(&self) -> String {
        include_str!("./symbol_definition_tool/description.md").into()
    }

    fn icon(&self) -> IconName {
        IconName::MagnifyingGlass
    }

    fn input_schema(&self, format: LanguageModelToolSchemaFormat) -> Result<serde_json::Value> {
        json_schema_for::<SymbolDefinitionToolInput>(format)
    }

    fn ui_text(&self, input: &serde_json::Value) -> String {
        if let Ok(input) = serde_json::from_value::<SymbolDefinitionToolInput>(input.clone()) {
            format!(
                "Find definition at {} line {} column {}",
                MarkdownInlineCode(&input.path),
                input.line,
                input.column
            )
        } else {
            "Find symbol definition".to_string()
        }
    }

    fn run(
        self: Arc<Self>,
        input: serde_json::Value,
        _request: Arc<LanguageModelRequest>,
        project: Entity<Project>,
        action_log: Entity<ActionLog>,
        _model: Arc<dyn LanguageModel>,
        _window: Option<AnyWindowHandle>,
        cx: &mut App,
    ) -> ToolResult {
        let input: SymbolDefinitionToolInput = match serde_json::from_value(input) {
            Ok(input) => input,
            Err(e) => {
                return Task::ready(Err(anyhow!("Invalid input: {}", e))).into();
            }
        };

        // Validate line and column are positive
        if input.line == 0 || input.column == 0 {
            return Task::ready(Err(anyhow!("Line and column must be positive (1-based)"))).into();
        }

        let Some(project_path) = project.read(cx).find_project_path(&input.path, cx) else {
            return Task::ready(Err(anyhow!(
                "Could not find path {} in project",
                input.path
            )))
            .into();
        };

        let buffer = project.update(cx, |project, cx| project.open_buffer(project_path, cx));

        cx.spawn(async move |cx| {
            let buffer = buffer.await?;

            // Convert 1-based line/column to 0-based point
            let point = {
                let snapshot = buffer.read_with(cx, |buffer, _cx| buffer.snapshot())?;
                let line_index = (input.line as usize).saturating_sub(1);
                let column_index = (input.column as usize).saturating_sub(1);

                // Validate line is within bounds
                let max_line = snapshot.max_point().row;
                if line_index > max_line as usize {
                    return Err(anyhow!(
                        "Line {} is out of bounds (file has {} lines)",
                        input.line,
                        max_line + 1
                    ));
                }

                // Get the line and validate column is within bounds
                let line_len = snapshot.line_len(line_index as u32) as usize;
                if column_index > line_len {
                    return Err(anyhow!(
                        "Column {} is out of bounds (line {} has {} characters)",
                        input.column,
                        input.line,
                        line_len
                    ));
                }

                language::Point::new(line_index as u32, column_index as u32)
            };

            // Get the anchor for the position
            let anchor = buffer.read_with(cx, |buffer, _cx| buffer.anchor_before(point))?;

            // Request definition from the project
            let definitions =
                project.update(cx, |project, cx| project.definitions(&buffer, anchor, cx))?;

            let definitions = definitions.await?;

            action_log.update(cx, |_log, _cx| {
                // Log action - currently no public API for adding custom logs
            })?;

            // Format the results
            let mut output = String::new();

            match definitions {
                None => {
                    output
                        .push_str("No definition found for the symbol at the specified location.");
                }
                Some(ref defs) if defs.is_empty() => {
                    output
                        .push_str("No definition found for the symbol at the specified location.");
                }
                Some(definitions) => {
                    for (idx, location_link) in definitions.iter().enumerate() {
                        if definitions.len() > 1 {
                            writeln!(&mut output, "Definition {}:", idx + 1)?;
                        }

                        let target = &location_link.target;
                        let (path, worktree_id) = target.buffer.read_with(cx, |buffer, cx| {
                            let file = buffer.file();
                            let path = file
                                .map(|f| f.path().clone())
                                .unwrap_or_else(|| Arc::from(Path::new("(untitled)")));
                            let worktree_id = file.map(|f| f.worktree_id(cx));
                            (path, worktree_id)
                        })?;

                        // Format the path relative to project root if possible
                        let relative_path = if let Some(worktree_id) = worktree_id {
                            project.read_with(cx, |project, cx| {
                                project
                                    .worktree_for_id(worktree_id, cx)
                                    .map(|worktree| {
                                        let root_name = worktree.read(cx).root_name();
                                        Path::new(root_name)
                                            .join(&path)
                                            .to_string_lossy()
                                            .to_string()
                                    })
                                    .unwrap_or_else(|| path.to_string_lossy().to_string())
                            })?
                        } else {
                            path.to_string_lossy().to_string()
                        };

                        // Get the range in the target buffer
                        let range = target.buffer.read_with(cx, |buffer, _cx| {
                            let start = target.range.start.to_offset(buffer);
                            let end = target.range.end.to_offset(buffer);
                            (buffer.offset_to_point(start), buffer.offset_to_point(end))
                        })?;

                        writeln!(
                            &mut output,
                            "File: {}\nLocation: line {} column {} to line {} column {}",
                            relative_path,
                            range.0.row + 1,
                            range.0.column + 1,
                            range.1.row + 1,
                            range.1.column + 1
                        )?;

                        // If there's an origin (where the symbol was referenced), show it too
                        if let Some(origin) = &location_link.origin {
                            let (origin_path, origin_worktree_id) =
                                origin.buffer.read_with(cx, |buffer, cx| {
                                    let file = buffer.file();
                                    let path = file
                                        .map(|f| f.path().clone())
                                        .unwrap_or_else(|| Arc::from(Path::new("(untitled)")));
                                    let worktree_id = file.map(|f| f.worktree_id(cx));
                                    (path, worktree_id)
                                })?;

                            let origin_relative_path = if let Some(worktree_id) = origin_worktree_id
                            {
                                project.read_with(cx, |project, cx| {
                                    project
                                        .worktree_for_id(worktree_id, cx)
                                        .map(|worktree| {
                                            let root_name = worktree.read(cx).root_name();
                                            Path::new(root_name)
                                                .join(&origin_path)
                                                .to_string_lossy()
                                                .to_string()
                                        })
                                        .unwrap_or_else(|| {
                                            origin_path.to_string_lossy().to_string()
                                        })
                                })?
                            } else {
                                origin_path.to_string_lossy().to_string()
                            };

                            let origin_range = origin.buffer.read_with(cx, |buffer, _cx| {
                                let start = origin.range.start.to_offset(buffer);
                                let end = origin.range.end.to_offset(buffer);
                                (buffer.offset_to_point(start), buffer.offset_to_point(end))
                            })?;

                            writeln!(
                                &mut output,
                                "Referenced from: {} at line {} column {}",
                                origin_relative_path,
                                origin_range.0.row + 1,
                                origin_range.0.column + 1
                            )?;
                        }

                        if definitions.len() > 1 && idx < definitions.len() - 1 {
                            writeln!(&mut output)?;
                        }
                    }
                }
            }

            Ok(output.into())
        })
        .into()
    }
}
