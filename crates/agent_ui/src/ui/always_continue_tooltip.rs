// Note: The UI integration for this tooltip was removed during refactoring,
// but the backend always_continue functionality remains in agent/src/thread.rs
#![allow(dead_code)]

use crate::ToggleAlwaysContinue;
use gpui::{Context, FontWeight, IntoElement, Render, Window};
use ui::{KeyBinding, prelude::*, tooltip_container};

pub struct AlwaysContinueTooltip {
    selected: bool,
}

impl AlwaysContinueTooltip {
    pub fn new() -> Self {
        Self { selected: false }
    }

    pub fn selected(mut self, selected: bool) -> Self {
        self.selected = selected;
        self
    }
}

impl Render for AlwaysContinueTooltip {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let (icon, color) = if self.selected {
            (IconName::RotateCw, Color::Success)
        } else {
            (IconName::RotateCw, Color::Default)
        };

        let turned_on = h_flex()
            .h_4()
            .px_1()
            .py_0p5()
            .rounded_md()
            .bg(cx.theme().status().success_background)
            .child(
                Label::new("ON")
                    .size(LabelSize::XSmall)
                    .weight(FontWeight::BOLD)
                    .color(Color::Success),
            );

        let title = h_flex()
            .gap_1()
            .child(Icon::new(icon).color(color).size(IconSize::Small))
            .child(Label::new("Always Continue").weight(FontWeight::SEMIBOLD))
            .when(self.selected, |title| title.child(turned_on));

        let keybinding = KeyBinding::for_action(&ToggleAlwaysContinue, window, cx)
            .map(|kb| kb.size(rems_from_px(12.)));

        let description = if self.selected {
            "Automatically continue when tool use limit is reached."
        } else {
            "Enable automatic continuation when tool use limit is reached."
        };

        tooltip_container(window, cx, move |this, _, _| {
            this.child(
                v_flex()
                    .gap_1p5()
                    .py_1()
                    .px_1()
                    .child(
                        h_flex()
                            .gap_3()
                            .justify_between()
                            .child(title)
                            .children(keybinding),
                    )
                    .child(
                        h_flex().max_w_96().gap_2().items_start().child(
                            Label::new(description)
                                .size(LabelSize::Small)
                                .color(Color::Muted),
                        ),
                    ),
            )
        })
    }
}
