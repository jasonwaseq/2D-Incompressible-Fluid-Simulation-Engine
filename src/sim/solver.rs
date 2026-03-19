use super::forces::SimCommand;
use super::state::SimulationState;
use crate::config::SimulationConfig;
#[cfg(any(debug_assertions, feature = "sanity-checks"))]
use crate::math::operators::all_finite;
use crate::util::timer::FrameTimer;

use super::advection::{advect_scalar, advect_velocity};
use super::boundary::{apply_scalar_boundary, apply_velocity_boundary, ScalarBoundary, VelocityBoundary};
use super::diffusion::{diffuse_scalar, diffuse_velocity};
use super::effects::{compute_vorticity, max_abs_vorticity, BuiltinEffect, SolverEffect};
use super::forces::apply_commands;
use super::pressure::project_velocity;

#[derive(Debug, Clone)]
pub struct FluidSolver {
    config: SimulationConfig,
    effects: Vec<BuiltinEffect>,
    velocity_boundary: VelocityBoundary,
    scalar_boundary: ScalarBoundary,
}

impl FluidSolver {
    pub fn new(config: SimulationConfig) -> Self {
        Self {
            effects: BuiltinEffect::from_config(&config),
            config,
            velocity_boundary: VelocityBoundary::NoSlipBox,
            scalar_boundary: ScalarBoundary::ZeroGradientBox,
        }
    }

    pub fn config(&self) -> &SimulationConfig {
        &self.config
    }

    pub fn dt(&self) -> f32 {
        self.config.dt
    }

    pub fn set_dt(&mut self, dt: f32) {
        assert!(dt.is_finite() && dt > 0.0, "dt must be finite and positive");
        self.config.dt = dt;
    }

    pub fn step(&mut self, state: &mut SimulationState, commands: &[SimCommand]) {
        assert_eq!(state.grid.nx, self.config.grid_width, "grid width must match solver config");
        assert_eq!(state.grid.ny, self.config.grid_height, "grid height must match solver config");
        assert!(
            (state.grid.cell_size - self.config.cell_size).abs() < f32::EPSILON,
            "cell size must match solver config"
        );

        let timer = FrameTimer::start();
        apply_commands(state, commands);
        debug_validate_stage("post-commands", state);

        for effect in &self.effects {
            effect.apply(state, self.config.dt, self.velocity_boundary);
            debug_validate_stage(effect.name(), state);
        }

        apply_velocity_boundary(&mut state.velocity, self.velocity_boundary);
        apply_scalar_boundary(&mut state.density, self.scalar_boundary);
        debug_validate_stage("post-boundaries", state);

        diffuse_velocity(
            &state.velocity,
            self.config.viscosity,
            self.config.dt,
            self.config.solver_iterations,
            self.velocity_boundary,
            &mut state.scratch.velocity0,
        );
        std::mem::swap(&mut state.velocity, &mut state.scratch.velocity0);
        debug_validate_stage("post-diffuse-velocity", state);

        project_velocity(
            &mut state.velocity,
            &mut state.pressure,
            &mut state.divergence,
            self.config.solver_iterations,
            self.velocity_boundary,
            self.scalar_boundary,
        );
        debug_validate_stage("post-project-1", state);

        advect_velocity(&state.velocity, self.config.dt, &mut state.scratch.velocity0);
        std::mem::swap(&mut state.velocity, &mut state.scratch.velocity0);
        apply_velocity_boundary(&mut state.velocity, self.velocity_boundary);
        debug_validate_stage("post-advect-velocity", state);

        let max_divergence = project_velocity(
            &mut state.velocity,
            &mut state.pressure,
            &mut state.divergence,
            self.config.solver_iterations,
            self.velocity_boundary,
            self.scalar_boundary,
        );
        debug_validate_stage("post-project-2", state);

        diffuse_scalar(
            &state.density,
            self.config.diffusion,
            self.config.dt,
            self.config.solver_iterations,
            self.scalar_boundary,
            &mut state.scratch.scalar0,
        );
        std::mem::swap(&mut state.density, &mut state.scratch.scalar0);
        debug_validate_stage("post-diffuse-density", state);

        advect_scalar(
            &state.density,
            &state.velocity,
            self.config.dt,
            &mut state.scratch.scalar0,
        );
        std::mem::swap(&mut state.density, &mut state.scratch.scalar0);
        apply_scalar_boundary(&mut state.density, self.scalar_boundary);
        debug_validate_stage("post-advect-density", state);

        compute_vorticity(&state.velocity, &mut state.vorticity);
        debug_validate_stage("post-vorticity", state);

        state.stats.max_divergence = max_divergence;
        state.stats.max_vorticity = max_abs_vorticity(&state.vorticity);
        state.stats.cfl = estimate_cfl(&state.velocity, self.config.dt);
        state.stats.step_ms = timer.elapsed().as_secs_f64() as f32 * 1000.0;
        state.stats.pressure_iterations = self.config.solver_iterations;

        debug_validate_stage("post-stats", state);
    }
}

fn estimate_cfl(velocity: &super::field::MacVelocity, dt: f32) -> f32 {
    let grid = velocity.u.grid();
    let max_u = velocity
        .u
        .as_slice()
        .iter()
        .fold(0.0_f32, |acc, value| acc.max(value.abs()));
    let max_v = velocity
        .v
        .as_slice()
        .iter()
        .fold(0.0_f32, |acc, value| acc.max(value.abs()));

    dt * max_u.max(max_v) / grid.cell_size
}

#[inline(always)]
fn debug_validate_stage(stage: &str, state: &SimulationState) {
    #[cfg(any(debug_assertions, feature = "sanity-checks"))]
    {
        debug_validate_stage_impl(stage, state);
    }

    #[cfg(not(any(debug_assertions, feature = "sanity-checks")))]
    {
        let _ = stage;
        let _ = state;
    }
}

#[cfg(any(debug_assertions, feature = "sanity-checks"))]
fn debug_validate_stage_impl(stage: &str, state: &SimulationState) {
    debug_assert!(
        all_finite(state.velocity.u.as_slice()),
        "non-finite x-face velocity values detected after {stage}"
    );
    debug_assert!(
        all_finite(state.velocity.v.as_slice()),
        "non-finite y-face velocity values detected after {stage}"
    );
    debug_assert!(
        all_finite(state.density.as_slice()),
        "non-finite density values detected after {stage}"
    );
    debug_assert!(
        all_finite(state.pressure.as_slice()),
        "non-finite pressure values detected after {stage}"
    );
    debug_assert!(
        all_finite(state.divergence.as_slice()),
        "non-finite divergence values detected after {stage}"
    );
    debug_assert!(
        all_finite(state.vorticity.as_slice()),
        "non-finite vorticity values detected after {stage}"
    );
}

#[cfg(test)]
mod tests {
    use glam::Vec2;

    use super::FluidSolver;
    use crate::config::SimulationConfig;
    use crate::sim::forces::SimCommand;
    use crate::sim::grid::GridSize;
    use crate::sim::pressure::max_abs_divergence;
    use crate::sim::state::SimulationState;

    #[test]
    fn zero_state_remains_stable_when_stepped() {
        let config = SimulationConfig::default();
        let grid = GridSize::new(config.grid_width, config.grid_height, config.cell_size)
            .expect("grid should be valid");
        let mut solver = FluidSolver::new(config);
        let mut state = SimulationState::new(grid);

        for _ in 0..5 {
            solver.step(&mut state, &[]);
        }

        assert!(state.stats.max_divergence <= 1.0e-6);
    }

    #[test]
    fn injected_force_creates_motion() {
        let config = SimulationConfig::default();
        let grid = GridSize::new(config.grid_width, config.grid_height, config.cell_size)
            .expect("grid should be valid");
        let mut solver = FluidSolver::new(config);
        let mut state = SimulationState::new(grid);

        solver.step(
            &mut state,
            &[SimCommand::AddForce {
                position: Vec2::new(80.0, 45.0),
                delta: Vec2::new(2.0, 0.0),
                radius: 6.0,
            }],
        );

        let max_speed = state
            .velocity
            .u
            .as_slice()
            .iter()
            .chain(state.velocity.v.as_slice().iter())
            .fold(0.0_f32, |acc, value| acc.max(value.abs()));

        assert!(max_speed > 0.0);
    }

    #[test]
    fn dye_injection_survives_step() {
        let config = SimulationConfig::default();
        let grid = GridSize::new(config.grid_width, config.grid_height, config.cell_size)
            .expect("grid should be valid");
        let mut solver = FluidSolver::new(config);
        let mut state = SimulationState::new(grid);

        solver.step(
            &mut state,
            &[SimCommand::AddDye {
                position: Vec2::new(80.0, 45.0),
                amount: 2.0,
                radius: 6.0,
            }],
        );

        let max_density = state
            .density
            .as_slice()
            .iter()
            .fold(0.0_f32, |acc, value| acc.max(*value));

        assert!(max_density > 0.0);
    }

    #[test]
    fn solver_updates_finite_diagnostics_each_step() {
        let config = SimulationConfig::default();
        let expected_iterations = config.solver_iterations;
        let grid = GridSize::new(config.grid_width, config.grid_height, config.cell_size)
            .expect("grid should be valid");
        let mut solver = FluidSolver::new(config);
        let mut state = SimulationState::new(grid);

        solver.step(
            &mut state,
            &[SimCommand::AddForce {
                position: Vec2::new(80.0, 45.0),
                delta: Vec2::new(2.0, 1.0),
                radius: 5.0,
            }],
        );

        assert!(state.stats.max_divergence.is_finite());
        assert!(state.stats.max_vorticity.is_finite());
        assert!(state.stats.cfl.is_finite());
        assert!(state.stats.step_ms.is_finite());
        assert!(state.stats.step_ms >= 0.0);
        assert_eq!(state.stats.pressure_iterations, expected_iterations);
        assert_eq!(
            state.stats.max_divergence,
            max_abs_divergence(&state.divergence)
        );
    }

    #[test]
    fn configured_buoyancy_generates_upward_velocity() {
        let mut config = SimulationConfig::default();
        config.buoyancy = 2.0;
        let grid = GridSize::new(config.grid_width, config.grid_height, config.cell_size)
            .expect("grid should be valid");
        let mut solver = FluidSolver::new(config);
        let mut state = SimulationState::new(grid);

        solver.step(
            &mut state,
            &[SimCommand::AddDye {
                position: Vec2::new(80.0, 45.0),
                amount: 4.0,
                radius: 4.0,
            }],
        );

        let min_v = state
            .velocity
            .v
            .as_slice()
            .iter()
            .fold(0.0_f32, |acc, value| acc.min(*value));

        assert!(min_v < 0.0);
    }
}
