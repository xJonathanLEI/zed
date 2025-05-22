use crate::schema::json_schema_for;
use action_log::ActionLog;
use anyhow::{Result, anyhow};
use assistant_tool::{Tool, ToolResult};
use gpui::{AnyWindowHandle, App, AppContext, Entity, Task};
use language_model::{LanguageModel, LanguageModelRequest, LanguageModelToolSchemaFormat};
use project::Project;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};
use ui::IconName;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SleepToolInput {
    /// The number of milliseconds to sleep for.
    ///
    /// <example>
    /// To sleep for 5 seconds use a value of `5000`
    /// </example>
    pub milliseconds: u64,
}

pub struct SleepTool;

impl Tool for SleepTool {
    fn name(&self) -> String {
        "sleep".into()
    }

    fn needs_confirmation(&self, _: &serde_json::Value, _: &Entity<Project>, _: &App) -> bool {
        false
    }

    fn may_perform_edits(&self) -> bool {
        false
    }

    fn description(&self) -> String {
        include_str!("./sleep_tool/description.md").into()
    }

    fn icon(&self) -> IconName {
        IconName::CountdownTimer
    }

    fn input_schema(&self, format: LanguageModelToolSchemaFormat) -> Result<serde_json::Value> {
        json_schema_for::<SleepToolInput>(format)
    }

    fn ui_text(&self, input: &serde_json::Value) -> String {
        match serde_json::from_value::<SleepToolInput>(input.clone()) {
            Ok(input) => {
                format!("Sleep for {} milliseconds", input.milliseconds)
            }
            Err(_) => "Sleep".to_string(),
        }
    }

    fn run(
        self: Arc<Self>,
        input: serde_json::Value,
        _request: Arc<LanguageModelRequest>,
        _project: Entity<Project>,
        _action_log: Entity<ActionLog>,
        _model: Arc<dyn LanguageModel>,
        _window: Option<AnyWindowHandle>,
        cx: &mut App,
    ) -> ToolResult {
        let input = match serde_json::from_value::<SleepToolInput>(input) {
            Ok(input) => input,
            Err(err) => return Task::ready(Err(anyhow!(err))).into(),
        };

        cx.background_spawn(async move {
            std::thread::sleep(Duration::from_millis(input.milliseconds));
            Ok(format!("Slept for {} milliseconds", input.milliseconds).into())
        })
        .into()
    }
}
