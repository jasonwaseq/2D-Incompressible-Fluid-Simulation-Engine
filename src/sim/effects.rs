use glam::Vec2;

use crate::config::SimulationConfig;

use super::boundary::{apply_velocity_boundary, VelocityBoundary};
use super::field::{MacVelocity, ScalarField};
use super::state::SimulationState;

pub trait SolverEffect: std::fmt::Debug + Send + Sync {
    fn name(&self) -> &'static str;
    fn apply(&self, state: &mut SimulationState, dt: f32, velocity_boundary: VelocityBoundary);
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BuoyancyEffect {
    pub strength: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VorticityConfinementEffect {
    pub strength: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BuiltinEffect {
    Buoyancy(BuoyancyEffect),
    VorticityConfinement(VorticityConfinementEffect),
}

impl BuiltinEffect {
    pub fn from_config(config: &SimulationConfig) -> Vec<Self> {
        let mut effects = Vec::with_capacity(2);

        if config.buoyancy > 0.0 {
            effects.push(Self::Buoyancy(BuoyancyEffect {
                strength: config.buoyancy,
            }));
        }

        if config.vorticity_confinement > 0.0 {
            effects.push(Self::VorticityConfinement(VorticityConfinementEffect {
                strength: config.vorticity_confinement,
            }));
        }

        effects
    }
}

impl SolverEffect for BuiltinEffect {
    fn name(&self) -> &'static str {
        match self {
            Self::Buoyancy(_) => "buoyancy",
            Self::VorticityConfinement(_) => "vorticity-confinement",
        }
    }

    fn apply(&self, state: &mut SimulationState, dt: f32, velocity_boundary: VelocityBoundary) {
        match self {
            Self::Buoyancy(effect) => {
                apply_buoyancy(&mut state.velocity, &state.density, dt, effect.strength);
            }
            Self::VorticityConfinement(effect) => {
                apply_vorticity_confinement(
                    &mut state.velocity,
                    &mut state.vorticity,
                    dt,
                    effect.strength,
                );
            }
        }

        apply_velocity_boundary(&mut state.velocity, velocity_boundary);
    }
}

pub fn compute_vorticity(velocity: &MacVelocity, vorticity: &mut ScalarField) {
    assert_eq!(velocity.u.grid(), vorticity.grid(), "velocity and vorticity grids must match");

    let grid = velocity.u.grid();
    let inv_2h = 0.5 / grid.cell_size;
    vorticity.fill(0.0);

    for j in 0..grid.ny {
        let j_minus = j.saturating_sub(1);
        let j_plus = (j + 1).min(grid.ny - 1);

        for i in 0..grid.nx {
            let i_minus = i.saturating_sub(1);
            let i_plus = (i + 1).min(grid.nx - 1);

            let left = velocity.cell_center_velocity(i_minus, j);
            let right = velocity.cell_center_velocity(i_plus, j);
            let up = velocity.cell_center_velocity(i, j_minus);
            let down = velocity.cell_center_velocity(i, j_plus);

            let dv_dx = (right.y - left.y) * inv_2h;
            let du_dy = (down.x - up.x) * inv_2h;
            vorticity.set_cell(i, j, dv_dx - du_dy);
        }
    }
}

pub fn max_abs_vorticity(vorticity: &ScalarField) -> f32 {
    let grid = vorticity.grid();
    let row_stride = grid.scalar_row_stride();
    let data = vorticity.as_slice();
    let mut max_value = 0.0_f32;

    for j in 0..grid.ny {
        let row_start = (j + 1) * row_stride + 1;
        for value in &data[row_start..row_start + grid.nx] {
            max_value = max_value.max(value.abs());
        }
    }

    max_value
}

pub fn apply_buoyancy(
    velocity: &mut MacVelocity,
    density: &ScalarField,
    dt: f32,
    strength: f32,
) {
    assert_eq!(velocity.u.grid(), density.grid(), "velocity and density grids must match");

    if strength <= 0.0 {
        return;
    }

    let grid = density.grid();

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let local_density = density.get_cell(i, j).max(0.0);
            if local_density <= 0.0 {
                continue;
            }

            // Screen-space y grows downward, so negative v lifts smoke upward.
            let impulse = -0.5 * dt * strength * local_density;
            velocity.v.set_face(i, j, velocity.v.get_face(i, j) + impulse);
            velocity.v.set_face(i, j + 1, velocity.v.get_face(i, j + 1) + impulse);
        }
    }
}

pub fn apply_vorticity_confinement(
    velocity: &mut MacVelocity,
    vorticity: &mut ScalarField,
    dt: f32,
    strength: f32,
) {
    assert_eq!(velocity.u.grid(), vorticity.grid(), "velocity and vorticity grids must match");

    if strength <= 0.0 {
        return;
    }

    compute_vorticity(velocity, vorticity);

    let grid = vorticity.grid();
    let inv_2h = 0.5 / grid.cell_size;
    let strength = strength * grid.cell_size;

    for j in 0..grid.ny {
        let j_minus = j.saturating_sub(1);
        let j_plus = (j + 1).min(grid.ny - 1);

        for i in 0..grid.nx {
            let i_minus = i.saturating_sub(1);
            let i_plus = (i + 1).min(grid.nx - 1);

            let abs_x = (vorticity.get_cell(i_plus, j).abs() - vorticity.get_cell(i_minus, j).abs())
                * inv_2h;
            let abs_y = (vorticity.get_cell(i, j_plus).abs() - vorticity.get_cell(i, j_minus).abs())
                * inv_2h;

            let gradient = Vec2::new(abs_x, abs_y);
            let magnitude = gradient.length();
            if magnitude <= 1.0e-6 {
                continue;
            }

            let normal = gradient / magnitude;
            let omega = vorticity.get_cell(i, j);
            let force = Vec2::new(normal.y * omega, -normal.x * omega) * strength;
            let u_impulse = 0.5 * dt * force.x;
            let v_impulse = 0.5 * dt * force.y;

            velocity
                .u
                .set_face(i, j, velocity.u.get_face(i, j) + u_impulse);
            velocity
                .u
                .set_face(i + 1, j, velocity.u.get_face(i + 1, j) + u_impulse);
            velocity
                .v
                .set_face(i, j, velocity.v.get_face(i, j) + v_impulse);
            velocity
                .v
                .set_face(i, j + 1, velocity.v.get_face(i, j + 1) + v_impulse);
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use glam::Vec2;

    use super::{
        apply_buoyancy, apply_vorticity_confinement, compute_vorticity, max_abs_vorticity,
        BuiltinEffect, SolverEffect,
    };
    use crate::config::SimulationConfig;
    use crate::sim::boundary::VelocityBoundary;
    use crate::sim::field::{MacVelocity, ScalarField};
    use crate::sim::grid::GridSize;
    use crate::sim::state::SimulationState;

    #[test]
    fn builtin_effects_follow_configuration() {
        let mut config = SimulationConfig::default();
        config.buoyancy = 1.25;
        config.vorticity_confinement = 3.5;

        let effects = BuiltinEffect::from_config(&config);

        assert_eq!(effects.len(), 2);
        assert_eq!(effects[0].name(), "buoyancy");
        assert_eq!(effects[1].name(), "vorticity-confinement");
    }

    #[test]
    fn vorticity_matches_affine_solid_body_rotation() {
        let grid = GridSize::new(10, 10, 1.0).expect("grid should be valid");
        let center = Vec2::new(5.0, 5.0);
        let angular_speed = 0.5;
        let mut velocity = MacVelocity::zeros(grid);
        let mut vorticity = ScalarField::zeros(grid);

        for j in 0..grid.ny {
            for i in 0..=grid.nx {
                let position = grid.u_face_position(i, j);
                velocity
                    .u
                    .set_face(i, j, -angular_speed * (position.y - center.y));
            }
        }

        for j in 0..=grid.ny {
            for i in 0..grid.nx {
                let position = grid.v_face_position(i, j);
                velocity
                    .v
                    .set_face(i, j, angular_speed * (position.x - center.x));
            }
        }

        compute_vorticity(&velocity, &mut vorticity);

        assert_relative_eq!(vorticity.get_cell(5, 5), 2.0 * angular_speed, epsilon = 1.0e-5);
        assert_relative_eq!(max_abs_vorticity(&vorticity), 2.0 * angular_speed, epsilon = 1.0e-5);
    }

    #[test]
    fn buoyancy_adds_upward_vertical_impulse() {
        let grid = GridSize::new(6, 6, 1.0).expect("grid should be valid");
        let mut velocity = MacVelocity::zeros(grid);
        let mut density = ScalarField::zeros(grid);
        density.set_cell(2, 3, 2.0);

        apply_buoyancy(&mut velocity, &density, 0.5, 1.0);

        assert!(velocity.v.get_face(2, 3) < 0.0);
        assert!(velocity.v.get_face(2, 4) < 0.0);
    }

    #[test]
    fn vorticity_confinement_adds_finite_force_near_localized_swirl() {
        let grid = GridSize::new(12, 12, 1.0).expect("grid should be valid");
        let center = Vec2::new(6.0, 6.0);
        let radius_scale = 9.0;
        let mut velocity = MacVelocity::zeros(grid);
        let mut vorticity = ScalarField::zeros(grid);

        for j in 0..grid.ny {
            for i in 0..=grid.nx {
                let position = grid.u_face_position(i, j) - center;
                let weight = (-position.length_squared() / radius_scale).exp();
                velocity.u.set_face(i, j, -position.y * weight);
            }
        }

        for j in 0..=grid.ny {
            for i in 0..grid.nx {
                let position = grid.v_face_position(i, j) - center;
                let weight = (-position.length_squared() / radius_scale).exp();
                velocity.v.set_face(i, j, position.x * weight);
            }
        }

        let before = velocity.max_speed();
        apply_vorticity_confinement(&mut velocity, &mut vorticity, 0.25, 2.0);
        let after = velocity.max_speed();

        assert!(after.is_finite());
        assert!(max_abs_vorticity(&vorticity).is_finite());
        assert!(after > before);
    }

    #[test]
    fn builtin_effect_application_updates_state_in_place() {
        let grid = GridSize::new(8, 8, 1.0).expect("grid should be valid");
        let mut state = SimulationState::new(grid);
        state.density.set_cell(4, 4, 1.0);

        BuiltinEffect::Buoyancy(super::BuoyancyEffect { strength: 2.0 }).apply(
            &mut state,
            0.5,
            VelocityBoundary::NoSlipBox,
        );

        assert!(state.velocity.v.get_face(4, 4) <= 0.0);
    }
}
