use glam::Vec2;

use super::state::SimulationState;

#[derive(Debug, Clone)]
pub enum SimCommand {
    AddForce {
        position: Vec2,
        delta: Vec2,
        radius: f32,
    },
    AddDye {
        position: Vec2,
        amount: f32,
        radius: f32,
    },
    Clear,
}

pub fn apply_commands(state: &mut SimulationState, commands: &[SimCommand]) {
    for command in commands {
        match command {
            SimCommand::AddForce {
                position,
                delta,
                radius,
            } => apply_force_impulse(state, *position, *delta, *radius),
            SimCommand::AddDye {
                position,
                amount,
                radius,
            } => add_dye(state, *position, *amount, *radius),
            SimCommand::Clear => state.clear(),
        }
    }
}

fn apply_force_impulse(state: &mut SimulationState, position: Vec2, delta: Vec2, radius: f32) {
    if radius <= 0.0 {
        return;
    }

    let grid = state.grid;
    let inv_radius = 1.0 / radius;

    for j in 0..grid.ny {
        for i in 0..=grid.nx {
            let sample_position = grid.u_face_position(i, j);
            let weight = radial_falloff(sample_position, position, inv_radius);
            if weight > 0.0 {
                let value = state.velocity.u.get_face(i, j) + delta.x * weight;
                state.velocity.u.set_face(i, j, value);
            }
        }
    }

    for j in 0..=grid.ny {
        for i in 0..grid.nx {
            let sample_position = grid.v_face_position(i, j);
            let weight = radial_falloff(sample_position, position, inv_radius);
            if weight > 0.0 {
                let value = state.velocity.v.get_face(i, j) + delta.y * weight;
                state.velocity.v.set_face(i, j, value);
            }
        }
    }
}

fn add_dye(state: &mut SimulationState, position: Vec2, amount: f32, radius: f32) {
    if radius <= 0.0 || amount == 0.0 {
        return;
    }

    let grid = state.grid;
    let inv_radius = 1.0 / radius;

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let sample_position = grid.cell_center(i, j);
            let weight = radial_falloff(sample_position, position, inv_radius);
            if weight > 0.0 {
                let value = state.density.get_cell(i, j) + amount * weight;
                state.density.set_cell(i, j, value);
            }
        }
    }
}

fn radial_falloff(sample_position: Vec2, center: Vec2, inv_radius: f32) -> f32 {
    let normalized_distance = (sample_position - center).length() * inv_radius;
    if normalized_distance >= 1.0 {
        0.0
    } else {
        let weight = 1.0 - normalized_distance;
        weight * weight
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec2;

    use super::{apply_commands, SimCommand};
    use crate::sim::grid::GridSize;
    use crate::sim::state::SimulationState;

    #[test]
    fn clear_command_resets_state() {
        let grid = GridSize::new(8, 6, 1.0).expect("grid should be valid");
        let mut state = SimulationState::new(grid);
        state.density.set_cell(3, 3, 1.0);

        apply_commands(&mut state, &[SimCommand::Clear]);

        assert_eq!(state.density.get_cell(3, 3), 0.0);
    }

    #[test]
    fn dye_and_force_commands_inject_values_near_cursor() {
        let grid = GridSize::new(8, 6, 1.0).expect("grid should be valid");
        let mut state = SimulationState::new(grid);

        apply_commands(
            &mut state,
            &[
                SimCommand::AddDye {
                    position: Vec2::new(4.0, 3.0),
                    amount: 2.0,
                    radius: 2.0,
                },
                SimCommand::AddForce {
                    position: Vec2::new(4.0, 3.0),
                    delta: Vec2::new(1.0, -0.5),
                    radius: 2.0,
                },
            ],
        );

        assert!(state.density.get_cell(3, 2) > 0.0);
        assert!(state.velocity.u.get_face(4, 2) > 0.0);
        assert!(state.velocity.v.get_face(3, 3) < 0.0);
    }
}
