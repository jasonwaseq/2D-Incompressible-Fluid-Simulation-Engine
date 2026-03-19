use glam::Vec2;

use crate::sim::forces::SimCommand;

#[derive(Debug, Clone)]
pub struct MouseState {
    pub cursor_position: Vec2,
    pub previous_cursor_position: Option<Vec2>,
    pub left_button_down: bool,
}

impl Default for MouseState {
    fn default() -> Self {
        Self {
            cursor_position: Vec2::ZERO,
            previous_cursor_position: None,
            left_button_down: false,
        }
    }
}

impl MouseState {
    pub fn cursor_position(&self) -> Vec2 {
        self.cursor_position
    }

    pub fn set_cursor_position(&mut self, position: Vec2) {
        if self.left_button_down {
            self.previous_cursor_position = Some(self.cursor_position);
        }

        self.cursor_position = position;
    }

    pub fn set_left_button_down(&mut self, is_down: bool) {
        self.left_button_down = is_down;

        if !is_down {
            self.previous_cursor_position = None;
        }
    }

    pub fn build_drag_commands(
        &mut self,
        brush_radius: f32,
        dye_amount: f32,
        force_scale: f32,
    ) -> Vec<SimCommand> {
        if !self.left_button_down {
            return Vec::new();
        }

        let previous = self
            .previous_cursor_position
            .unwrap_or(self.cursor_position);
        let delta = (self.cursor_position - previous) * force_scale;
        self.previous_cursor_position = Some(self.cursor_position);

        vec![
            SimCommand::AddDye {
                position: self.cursor_position,
                amount: dye_amount,
                radius: brush_radius,
            },
            SimCommand::AddForce {
                position: self.cursor_position,
                delta,
                radius: brush_radius,
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec2;

    use super::MouseState;
    use crate::sim::forces::SimCommand;

    #[test]
    fn drag_builds_force_and_dye_commands() {
        let mut mouse = MouseState::default();
        mouse.set_cursor_position(Vec2::new(10.0, 20.0));
        mouse.set_left_button_down(true);
        mouse.set_cursor_position(Vec2::new(12.0, 21.0));

        let commands = mouse.build_drag_commands(5.0, 2.0, 10.0);

        assert_eq!(commands.len(), 2);
        match &commands[0] {
            SimCommand::AddDye {
                position,
                amount,
                radius,
            } => {
                assert_eq!(*position, Vec2::new(12.0, 21.0));
                assert_eq!(*amount, 2.0);
                assert_eq!(*radius, 5.0);
            }
            _ => panic!("expected dye command"),
        }
    }
}
