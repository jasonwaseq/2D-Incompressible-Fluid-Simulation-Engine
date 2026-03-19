use crate::config::SimulationConfig;

use super::forces::SimCommand;
use super::gpu::{GpuBackendError, GpuFluidBackend};
use super::grid::GridSize;
use super::solver::FluidSolver;
use super::state::SimulationState;

pub enum SimulationRuntime {
    Cpu {
        solver: FluidSolver,
        state: SimulationState,
    },
    Gpu(GpuFluidBackend),
}

impl SimulationRuntime {
    pub fn new(config: SimulationConfig, grid: GridSize) -> Result<Self, GpuBackendError> {
        match config.backend {
            crate::config::SimulationBackendKind::Cpu => Ok(Self::Cpu {
                solver: FluidSolver::new(config),
                state: SimulationState::new(grid),
            }),
            crate::config::SimulationBackendKind::Gpu => {
                Ok(Self::Gpu(GpuFluidBackend::new(config, grid)?))
            }
        }
    }

    pub fn state(&self) -> &SimulationState {
        match self {
            Self::Cpu { state, .. } => state,
            Self::Gpu(backend) => backend.state(),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Cpu { .. } => "cpu",
            Self::Gpu(_) => "gpu",
        }
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Gpu(_))
    }

    pub fn as_gpu(&self) -> Option<&GpuFluidBackend> {
        match self {
            Self::Gpu(backend) => Some(backend),
            Self::Cpu { .. } => None,
        }
    }

    pub fn as_gpu_mut(&mut self) -> Option<&mut GpuFluidBackend> {
        match self {
            Self::Gpu(backend) => Some(backend),
            Self::Cpu { .. } => None,
        }
    }

    pub fn clear(&mut self) -> Result<(), GpuBackendError> {
        match self {
            Self::Cpu { state, .. } => {
                state.clear();
                Ok(())
            }
            Self::Gpu(backend) => backend.clear(),
        }
    }

    pub fn set_dt(&mut self, dt: f32) -> Result<(), GpuBackendError> {
        match self {
            Self::Cpu { solver, .. } => {
                solver.set_dt(dt);
                Ok(())
            }
            Self::Gpu(backend) => {
                backend.set_dt(dt);
                Ok(())
            }
        }
    }

    pub fn step(&mut self, commands: &[SimCommand]) -> Result<(), GpuBackendError> {
        match self {
            Self::Cpu { solver, state } => {
                solver.step(state, commands);
                Ok(())
            }
            Self::Gpu(backend) => backend.step(commands),
        }
    }
}
