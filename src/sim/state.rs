use super::field::{MacVelocity, ScalarField, SolidMask};
use super::grid::GridSize;

#[derive(Debug, Clone, Default)]
pub struct SimulationStats {
    pub max_divergence: f32,
    pub max_vorticity: f32,
    pub cfl: f32,
    pub step_ms: f32,
    pub pressure_iterations: usize,
}

#[derive(Debug, Clone)]
pub struct SimulationScratch {
    pub scalar0: ScalarField,
    pub scalar1: ScalarField,
    pub velocity0: MacVelocity,
    pub velocity1: MacVelocity,
}

impl SimulationScratch {
    pub fn new(grid: GridSize) -> Self {
        Self {
            scalar0: ScalarField::zeros(grid),
            scalar1: ScalarField::zeros(grid),
            velocity0: MacVelocity::zeros(grid),
            velocity1: MacVelocity::zeros(grid),
        }
    }

    pub fn clear(&mut self) {
        self.scalar0.fill(0.0);
        self.scalar1.fill(0.0);
        self.velocity0.fill(0.0);
        self.velocity1.fill(0.0);
    }
}

#[derive(Debug, Clone)]
pub struct SimulationState {
    pub grid: GridSize,
    pub velocity: MacVelocity,
    pub density: ScalarField,
    pub pressure: ScalarField,
    pub divergence: ScalarField,
    pub vorticity: ScalarField,
    pub solids: SolidMask,
    pub scratch: SimulationScratch,
    pub stats: SimulationStats,
}

impl SimulationState {
    pub fn new(grid: GridSize) -> Self {
        Self {
            grid,
            velocity: MacVelocity::zeros(grid),
            density: ScalarField::zeros(grid),
            pressure: ScalarField::zeros(grid),
            divergence: ScalarField::zeros(grid),
            vorticity: ScalarField::zeros(grid),
            solids: SolidMask::empty(grid),
            scratch: SimulationScratch::new(grid),
            stats: SimulationStats::default(),
        }
    }

    pub fn clear(&mut self) {
        self.velocity.fill(0.0);
        self.density.fill(0.0);
        self.pressure.fill(0.0);
        self.divergence.fill(0.0);
        self.vorticity.fill(0.0);
        self.solids.fill(false);
        self.scratch.clear();
        self.stats = SimulationStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::SimulationState;
    use crate::sim::grid::GridSize;

    #[test]
    fn clear_reuses_existing_allocations() {
        let grid = GridSize::new(8, 6, 1.0).expect("grid should be valid");
        let mut state = SimulationState::new(grid);

        state.density.set_cell(3, 2, 9.0);
        state.velocity.u.set_face(4, 2, 1.5);
        state.solids.set_solid_cell(0, 0, true);

        let density_ptr = state.density.as_ptr();
        let vorticity_ptr = state.vorticity.as_ptr();
        let u_ptr = state.velocity.u.as_ptr();
        let v_ptr = state.velocity.v.as_ptr();

        state.clear();

        assert_eq!(density_ptr, state.density.as_ptr());
        assert_eq!(vorticity_ptr, state.vorticity.as_ptr());
        assert_eq!(u_ptr, state.velocity.u.as_ptr());
        assert_eq!(v_ptr, state.velocity.v.as_ptr());
        assert_eq!(state.density.get_cell(3, 2), 0.0);
        assert_eq!(state.velocity.u.get_face(4, 2), 0.0);
        assert!(!state.solids.is_solid_cell(0, 0));
    }
}
