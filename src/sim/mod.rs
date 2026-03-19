pub mod advection;
pub mod boundary;
pub mod diffusion;
pub mod effects;
pub mod field;
pub mod forces;
pub mod gpu;
pub mod grid;
pub mod obstacles;
pub mod pressure;
pub mod runtime;
pub mod solver;
pub mod state;

pub use effects::{
    compute_vorticity, max_abs_vorticity, BuiltinEffect, BuoyancyEffect, SolverEffect,
    VorticityConfinementEffect,
};
pub use field::{FaceFieldX, FaceFieldY, MacVelocity, ScalarField, SolidMask};
pub use gpu::{GpuBackendError, GpuFluidBackend};
pub use grid::{FieldShape, GridSize, GHOST_LAYERS};
pub use obstacles::{clear_obstacles, CircleObstacle, ObstaclePrimitive};
pub use pressure::{
    compute_divergence, max_abs_divergence, project_velocity, GaussSeidelPressureSolver,
    PcgPressureSolver, PressureSolver, PressureSolverRuntime, ProjectionStats,
};
pub use runtime::SimulationRuntime;
pub use solver::FluidSolver;
pub use state::{SimulationScratch, SimulationState, SimulationStats};
