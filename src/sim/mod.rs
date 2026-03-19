pub mod advection;
pub mod boundary;
pub mod diffusion;
pub mod effects;
pub mod field;
pub mod forces;
pub mod grid;
pub mod pressure;
pub mod solver;
pub mod state;

pub use field::{FaceFieldX, FaceFieldY, MacVelocity, ScalarField, SolidMask};
pub use grid::{FieldShape, GridSize, GHOST_LAYERS};
pub use effects::{
    compute_vorticity, max_abs_vorticity, BuiltinEffect, BuoyancyEffect, SolverEffect,
    VorticityConfinementEffect,
};
pub use solver::FluidSolver;
pub use state::{SimulationScratch, SimulationState, SimulationStats};
