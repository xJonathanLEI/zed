//! # Changed Files Picker
//!
//! This module provides a picker interface for quickly navigating and opening files
//! that have been modified in the current Git repository. It displays all changed files
//! (modified, added, deleted, renamed, etc.) with appropriate status icons and allows
//! fuzzy searching through the list.
//!
//! ## Usage
//!
//! The picker can be opened with the keybinding:
//! - macOS: `cmd-alt-g f`
//! - Linux: `alt-g f`
//!
//! Once open, you can:
//! - Type to fuzzy search through changed files
//! - Use arrow keys to navigate the list
//! - Press Enter to open the selected file
//! - Press Escape to close the picker

use fuzzy::{StringMatch, StringMatchCandidate};
use git::status::{FileStatus, StatusCode, TrackedStatus};
use gpui::{
    App, Context, DismissEvent, Entity, EventEmitter, FocusHandle, Focusable, ParentElement,
    Render, Task, WeakEntity, Window, rems,
};
use picker::{Picker, PickerDelegate};
use project::{Project, ProjectPath};
use std::sync::Arc;
use ui::{Color, HighlightedLabel, Icon, IconName, Label, ListItem, ListItemSpacing, prelude::*};
use util::ResultExt;
use workspace::{ModalView, Workspace};

// Action to toggle the changed files picker.
gpui::actions!(changed_file_finder, [Toggle]);

pub fn init(cx: &mut App) {
    cx.observe_new(ChangedFilesPicker::register).detach();
}

/// A modal picker that displays all changed files in the current Git repository.
///
/// This picker shows files with their Git status (modified, added, deleted, etc.)
/// and allows users to quickly navigate to any changed file in their project.
pub struct ChangedFilesPicker {
    picker: Entity<Picker<ChangedFilesPickerDelegate>>,
}

impl ChangedFilesPicker {
    fn register(workspace: &mut Workspace, _: Option<&mut Window>, _: &mut Context<Workspace>) {
        workspace.register_action(|workspace, _: &Toggle, window, cx| {
            let weak_workspace = workspace.weak_handle();
            let project = workspace.project().clone();
            workspace.toggle_modal(window, cx, |window, cx| {
                let delegate = ChangedFilesPickerDelegate::new(
                    cx.entity().downgrade(),
                    weak_workspace,
                    project,
                    cx,
                );
                ChangedFilesPicker::new(delegate, window, cx)
            });
        });
    }

    fn new(
        delegate: ChangedFilesPickerDelegate,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let picker = cx.new(|cx| Picker::uniform_list(delegate, window, cx));
        Self { picker }
    }
}

impl EventEmitter<DismissEvent> for ChangedFilesPicker {}

impl ModalView for ChangedFilesPicker {}

impl Focusable for ChangedFilesPicker {
    fn focus_handle(&self, cx: &App) -> FocusHandle {
        self.picker.focus_handle(cx)
    }
}

impl Render for ChangedFilesPicker {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        v_flex().w(rems(34.)).child(self.picker.clone())
    }
}

/// The delegate that handles the picker's behavior and data.
///
/// This struct manages the list of changed files, handles filtering based on
/// user input, and processes file selection.
pub struct ChangedFilesPickerDelegate {
    changed_files_picker: WeakEntity<ChangedFilesPicker>,
    workspace: WeakEntity<Workspace>,
    entries: Vec<ChangedFileEntry>,
    matches: Vec<StringMatch>,
    selected_index: usize,
}

/// Represents a single changed file entry in the picker.
///
/// Contains the file's project path and its Git status information.
#[derive(Clone, Debug)]
struct ChangedFileEntry {
    project_path: ProjectPath,
    status: FileStatus,
}

impl ChangedFilesPickerDelegate {
    pub fn new(
        changed_files_picker: WeakEntity<ChangedFilesPicker>,
        workspace: WeakEntity<Workspace>,
        project: Entity<Project>,
        cx: &mut App,
    ) -> Self {
        let entries = Self::collect_changed_files(&project, cx);

        let mut delegate = Self {
            changed_files_picker,
            workspace,
            matches: Vec::new(),
            selected_index: 0,
            entries,
        };

        let _ = delegate.update_matches("".to_string(), cx);
        delegate
    }

    fn collect_changed_files(project: &Entity<Project>, cx: &App) -> Vec<ChangedFileEntry> {
        let mut entries = Vec::new();
        let project = project.read(cx);

        let git_store = project.git_store();
        let git_store = git_store.read(cx);

        for repository in git_store.repositories().values() {
            let repo = repository.read(cx);

            for entry in repo.cached_status() {
                if let Some(project_path) = repo.repo_path_to_project_path(&entry.repo_path, cx) {
                    entries.push(ChangedFileEntry {
                        project_path,
                        status: entry.status,
                    });
                }
            }
        }

        // Sort by path for consistent ordering
        entries.sort_by(|a, b| a.project_path.path.cmp(&b.project_path.path));
        entries
    }

    fn update_matches(&mut self, query: String, cx: &App) -> Task<()> {
        let query = query.trim().to_string();

        if query.is_empty() {
            self.matches = self
                .entries
                .iter()
                .enumerate()
                .map(|(index, _)| StringMatch {
                    candidate_id: index,
                    string: String::new(),
                    positions: Vec::new(),
                    score: 0.0,
                })
                .collect();
        } else {
            let candidates = self
                .entries
                .iter()
                .enumerate()
                .map(|(id, entry)| {
                    let path_string = entry.project_path.path.to_string_lossy().into_owned();
                    let char_bag = path_string.chars().collect();
                    StringMatchCandidate {
                        id,
                        string: path_string,
                        char_bag,
                    }
                })
                .collect::<Vec<_>>();

            let executor = cx.background_executor().clone();
            self.matches = cx.background_executor().block(fuzzy::match_strings(
                &candidates,
                &query,
                false,
                true,
                100,
                &Default::default(),
                executor,
            ));
        }

        self.selected_index = self
            .selected_index
            .min(self.matches.len().saturating_sub(1));

        Task::ready(())
    }

    /// Returns the appropriate icon and color for a given file status.
    ///
    /// Maps Git file statuses to UI icons:
    /// - Added/Untracked files: Plus icon with Created color
    /// - Modified files: Pencil icon with Modified color
    /// - Deleted files: Trash icon with Deleted color
    /// - Renamed/Copied files: Arrow icon with Modified color
    /// - Conflicted files: Warning icon with Conflict color
    fn status_icon(status: &FileStatus) -> Option<(IconName, Color)> {
        use git::status::{StatusCode, TrackedStatus};

        match status {
            FileStatus::Untracked => Some((IconName::Plus, Color::Created)),
            FileStatus::Ignored => None,
            FileStatus::Unmerged(_) => Some((IconName::Warning, Color::Conflict)),
            FileStatus::Tracked(TrackedStatus {
                index_status,
                worktree_status,
            }) => {
                // Prefer worktree status over index status for display
                let code = if *worktree_status != StatusCode::Unmodified {
                    worktree_status
                } else {
                    index_status
                };

                match code {
                    StatusCode::Added => Some((IconName::Plus, Color::Created)),
                    StatusCode::Modified | StatusCode::TypeChanged => {
                        Some((IconName::Pencil, Color::Modified))
                    }
                    StatusCode::Deleted => Some((IconName::Trash, Color::Deleted)),
                    StatusCode::Renamed | StatusCode::Copied => {
                        Some((IconName::ArrowRight, Color::Modified))
                    }
                    StatusCode::Unmodified => None,
                }
            }
        }
    }
}

impl PickerDelegate for ChangedFilesPickerDelegate {
    type ListItem = ListItem;

    fn match_count(&self) -> usize {
        self.matches.len()
    }

    fn selected_index(&self) -> usize {
        self.selected_index
    }

    fn set_selected_index(&mut self, ix: usize, _: &mut Window, _: &mut Context<Picker<Self>>) {
        self.selected_index = ix;
    }

    fn placeholder_text(&self, _: &mut Window, _: &mut App) -> Arc<str> {
        "Search changed files...".into()
    }

    fn update_matches(
        &mut self,
        query: String,
        _: &mut Window,
        cx: &mut Context<Picker<Self>>,
    ) -> Task<()> {
        self.update_matches(query, cx)
    }

    fn confirm(&mut self, _: bool, window: &mut Window, cx: &mut Context<Picker<Self>>) {
        let selected_match = match self.matches.get(self.selected_index) {
            Some(m) => m,
            None => return,
        };

        let entry = &self.entries[selected_match.candidate_id];

        if matches!(
            entry.status,
            FileStatus::Tracked(TrackedStatus {
                worktree_status: StatusCode::Deleted,
                ..
            }) | FileStatus::Tracked(TrackedStatus {
                index_status: StatusCode::Deleted,
                worktree_status: StatusCode::Unmodified
            })
        ) {
            return;
        }

        let Some(workspace) = self.workspace.upgrade() else {
            return;
        };

        let project_path = entry.project_path.clone();
        let changed_files_picker = self.changed_files_picker.clone();

        let open_task = workspace.update(cx, |workspace, cx| {
            workspace.open_path(project_path, None, true, window, cx)
        });

        cx.spawn_in(window, async move |_, cx| {
            open_task.await.ok()?;
            changed_files_picker
                .update(cx, |_, cx| cx.emit(DismissEvent))
                .ok()?;
            Some(())
        })
        .detach();
    }

    fn dismissed(&mut self, _: &mut Window, cx: &mut Context<Picker<Self>>) {
        self.changed_files_picker
            .update(cx, |_, cx| cx.emit(DismissEvent))
            .log_err();
    }

    fn render_match(
        &self,
        ix: usize,
        selected: bool,
        _: &mut Window,
        _: &mut Context<Picker<Self>>,
    ) -> Option<Self::ListItem> {
        let hit = self.matches.get(ix)?;
        let entry = &self.entries[hit.candidate_id];

        let path_string = entry.project_path.path.to_string_lossy();
        let (file_name, directory) = if let Some(slash_idx) = path_string.rfind('/') {
            (
                &path_string[slash_idx + 1..],
                Some(&path_string[..slash_idx]),
            )
        } else {
            (path_string.as_ref(), None)
        };

        let mut item = ListItem::new(ix)
            .inset(true)
            .spacing(ListItemSpacing::Sparse);

        if selected {
            item = item.toggle_state(true);
        }

        if let Some((icon, color)) = Self::status_icon(&entry.status) {
            item = item.start_slot(Icon::new(icon).size(IconSize::Small).color(color));
        }

        // Calculate positions for filename and directory separately
        let filename_offset = path_string.len() - file_name.len();

        // Separate positions into filename and directory positions
        let mut filename_positions = Vec::new();
        let mut directory_positions = Vec::new();

        for &pos in &hit.positions {
            if pos >= filename_offset {
                filename_positions.push(pos - filename_offset);
            } else {
                directory_positions.push(pos);
            }
        }

        // Always show filename and directory separately, like the file finder
        let file_name_element = if filename_positions.is_empty() {
            Label::new(file_name.to_string()).into_any_element()
        } else {
            HighlightedLabel::new(file_name.to_string(), filename_positions).into_any_element()
        };

        let directory_element = directory.map(|dir| {
            if directory_positions.is_empty() {
                Label::new(dir.to_string())
                    .size(LabelSize::Small)
                    .color(Color::Muted)
                    .into_any_element()
            } else {
                HighlightedLabel::new(dir.to_string(), directory_positions)
                    .size(LabelSize::Small)
                    .color(Color::Muted)
                    .into_any_element()
            }
        });

        item = item.child(
            h_flex()
                .gap_2()
                .child(file_name_element)
                .when_some(directory_element, |this, element| this.child(element)),
        );

        Some(item)
    }
}
