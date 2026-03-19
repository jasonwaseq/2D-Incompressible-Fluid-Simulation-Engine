use std::error::Error;
use std::fmt::{Display, Formatter};

use clap::{Args, Parser, ValueEnum};

#[derive(Debug, Clone, Parser)]
#[command(
    name = "fluid-sim-2d",
    version,
    about = "A modular 2D incompressible fluid simulation engine."
)]
pub struct AppConfig {
    #[command(flatten)]
    pub simulation: SimulationConfig,

    #[command(flatten)]
    pub render: RenderConfig,

    #[command(flatten)]
    pub input: InputConfig,

    #[arg(long, default_value_t = false)]
    pub start_paused: bool,

    #[arg(long, default_value_t = 0.25)]
    pub max_frame_time: f32,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            simulation: SimulationConfig::default(),
            render: RenderConfig::default(),
            input: InputConfig::default(),
            start_paused: false,
            max_frame_time: 0.25,
        }
    }
}

impl AppConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.simulation.validate()?;
        self.render.validate()?;
        self.input.validate()?;

        if !(self.max_frame_time.is_finite() && self.max_frame_time > 0.0) {
            return Err(ConfigError::NonPositiveValue {
                field: "max_frame_time",
                value: self.max_frame_time,
            });
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Args)]
pub struct SimulationConfig {
    #[arg(long = "grid-width", default_value_t = 160)]
    pub grid_width: usize,

    #[arg(long = "grid-height", default_value_t = 90)]
    pub grid_height: usize,

    #[arg(long, default_value_t = 1.0)]
    pub cell_size: f32,

    #[arg(long, default_value_t = 1.0 / 60.0)]
    pub dt: f32,

    #[arg(long, default_value_t = 0.0001)]
    pub viscosity: f32,

    #[arg(long, default_value_t = 0.0)]
    pub diffusion: f32,

    #[arg(long = "iterations", default_value_t = 40)]
    pub solver_iterations: usize,

    #[arg(long, value_enum, default_value_t = SimulationBackendKind::Cpu)]
    pub backend: SimulationBackendKind,

    #[arg(long, value_enum, default_value_t = PressureSolverKind::GaussSeidel)]
    pub pressure_solver: PressureSolverKind,

    #[arg(long = "pressure-tolerance", default_value_t = 1.0e-3)]
    pub pressure_tolerance: f32,

    #[arg(long, value_enum, default_value_t = ScalarAdvectionKind::SemiLagrangian)]
    pub scalar_advection: ScalarAdvectionKind,

    #[arg(long, default_value_t = 0.0)]
    pub buoyancy: f32,

    #[arg(long = "vorticity-confinement", default_value_t = 0.0)]
    pub vorticity_confinement: f32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            grid_width: 160,
            grid_height: 90,
            cell_size: 1.0,
            dt: 1.0 / 60.0,
            viscosity: 0.0001,
            diffusion: 0.0,
            solver_iterations: 40,
            backend: SimulationBackendKind::Cpu,
            pressure_solver: PressureSolverKind::GaussSeidel,
            pressure_tolerance: 1.0e-3,
            scalar_advection: ScalarAdvectionKind::SemiLagrangian,
            buoyancy: 0.0,
            vorticity_confinement: 0.0,
        }
    }
}

impl SimulationConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.grid_width == 0 {
            return Err(ConfigError::ZeroDimension("grid_width"));
        }

        if self.grid_height == 0 {
            return Err(ConfigError::ZeroDimension("grid_height"));
        }

        if !(self.cell_size.is_finite() && self.cell_size > 0.0) {
            return Err(ConfigError::NonPositiveValue {
                field: "cell_size",
                value: self.cell_size,
            });
        }

        if !(self.dt.is_finite() && self.dt > 0.0) {
            return Err(ConfigError::NonPositiveValue {
                field: "dt",
                value: self.dt,
            });
        }

        if !self.viscosity.is_finite() || self.viscosity < 0.0 {
            return Err(ConfigError::NegativeValue {
                field: "viscosity",
                value: self.viscosity,
            });
        }

        if !self.diffusion.is_finite() || self.diffusion < 0.0 {
            return Err(ConfigError::NegativeValue {
                field: "diffusion",
                value: self.diffusion,
            });
        }

        if self.solver_iterations == 0 {
            return Err(ConfigError::ZeroDimension("solver_iterations"));
        }

        if !(self.pressure_tolerance.is_finite() && self.pressure_tolerance > 0.0) {
            return Err(ConfigError::NonPositiveValue {
                field: "pressure_tolerance",
                value: self.pressure_tolerance,
            });
        }

        if !self.buoyancy.is_finite() || self.buoyancy < 0.0 {
            return Err(ConfigError::NegativeValue {
                field: "buoyancy",
                value: self.buoyancy,
            });
        }

        if !self.vorticity_confinement.is_finite() || self.vorticity_confinement < 0.0 {
            return Err(ConfigError::NegativeValue {
                field: "vorticity_confinement",
                value: self.vorticity_confinement,
            });
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SimulationBackendKind {
    Cpu,
    Gpu,
}

impl Default for SimulationBackendKind {
    fn default() -> Self {
        Self::Cpu
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum VisualizationMode {
    Density,
    VelocityMagnitude,
    Pressure,
    Divergence,
    Vorticity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum PressureSolverKind {
    GaussSeidel,
    Pcg,
}

impl Default for PressureSolverKind {
    fn default() -> Self {
        Self::GaussSeidel
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ScalarAdvectionKind {
    SemiLagrangian,
    MacCormack,
}

impl Default for ScalarAdvectionKind {
    fn default() -> Self {
        Self::SemiLagrangian
    }
}

impl Default for VisualizationMode {
    fn default() -> Self {
        Self::Density
    }
}

#[derive(Debug, Clone, Args)]
pub struct RenderConfig {
    #[arg(long, default_value_t = 6)]
    pub window_scale: u32,

    #[arg(long, value_enum, default_value_t = VisualizationMode::Density)]
    pub visualization_mode: VisualizationMode,

    #[arg(long, default_value_t = false)]
    pub show_velocity_vectors: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            window_scale: 6,
            visualization_mode: VisualizationMode::Density,
            show_velocity_vectors: false,
        }
    }
}

impl RenderConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.window_scale == 0 {
            return Err(ConfigError::ZeroDimension("window_scale"));
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Args)]
pub struct InputConfig {
    #[arg(long, default_value_t = 6.0)]
    pub brush_radius: f32,

    #[arg(long, default_value_t = 250.0)]
    pub force_scale: f32,

    #[arg(long, default_value_t = 1.0)]
    pub dye_amount: f32,
}

impl Default for InputConfig {
    fn default() -> Self {
        Self {
            brush_radius: 6.0,
            force_scale: 250.0,
            dye_amount: 1.0,
        }
    }
}

impl InputConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !(self.brush_radius.is_finite() && self.brush_radius > 0.0) {
            return Err(ConfigError::NonPositiveValue {
                field: "brush_radius",
                value: self.brush_radius,
            });
        }

        if !(self.force_scale.is_finite() && self.force_scale >= 0.0) {
            return Err(ConfigError::NegativeValue {
                field: "force_scale",
                value: self.force_scale,
            });
        }

        if !(self.dye_amount.is_finite() && self.dye_amount >= 0.0) {
            return Err(ConfigError::NegativeValue {
                field: "dye_amount",
                value: self.dye_amount,
            });
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConfigError {
    ZeroDimension(&'static str),
    NonPositiveValue { field: &'static str, value: f32 },
    NegativeValue { field: &'static str, value: f32 },
}

impl Display for ConfigError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension(field) => write!(f, "{field} must be greater than zero"),
            Self::NonPositiveValue { field, value } => {
                write!(f, "{field} must be > 0.0, got {value}")
            }
            Self::NegativeValue { field, value } => {
                write!(f, "{field} must be >= 0.0, got {value}")
            }
        }
    }
}

impl Error for ConfigError {}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{
        AppConfig, ConfigError, PressureSolverKind, ScalarAdvectionKind, SimulationBackendKind,
        VisualizationMode,
    };

    #[test]
    fn default_config_is_valid() {
        let config = AppConfig::default();
        config.validate().expect("default config should be valid");
    }

    #[test]
    fn clap_parses_visualization_mode() {
        let config = AppConfig::parse_from([
            "fluid-sim-2d",
            "--visualization-mode",
            "vorticity",
            "--backend",
            "gpu",
            "--pressure-solver",
            "pcg",
            "--scalar-advection",
            "mac-cormack",
            "--grid-width",
            "320",
            "--buoyancy",
            "1.5",
        ]);

        assert_eq!(
            config.render.visualization_mode,
            VisualizationMode::Vorticity
        );
        assert_eq!(config.simulation.backend, SimulationBackendKind::Gpu);
        assert_eq!(config.simulation.pressure_solver, PressureSolverKind::Pcg);
        assert_eq!(
            config.simulation.scalar_advection,
            ScalarAdvectionKind::MacCormack
        );
        assert_eq!(config.simulation.grid_width, 320);
        assert_eq!(config.simulation.buoyancy, 1.5);
    }

    #[test]
    fn simulation_config_rejects_negative_effect_strengths() {
        let mut config = AppConfig::default();
        config.simulation.vorticity_confinement = -1.0;

        let error = config
            .validate()
            .expect_err("negative effect strength should fail validation");
        assert_eq!(
            error,
            ConfigError::NegativeValue {
                field: "vorticity_confinement",
                value: -1.0,
            }
        );
    }

    #[test]
    fn simulation_config_rejects_non_positive_pressure_tolerance() {
        let mut config = AppConfig::default();
        config.simulation.pressure_tolerance = 0.0;

        let error = config
            .validate()
            .expect_err("non-positive pressure tolerance should fail");
        assert_eq!(
            error,
            ConfigError::NonPositiveValue {
                field: "pressure_tolerance",
                value: 0.0,
            }
        );
    }
}
