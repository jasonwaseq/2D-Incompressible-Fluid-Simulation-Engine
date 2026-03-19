#![forbid(unsafe_code)]

pub mod app;
pub mod config;
pub mod input;
pub mod math;
pub mod render;
pub mod sim;
pub mod util;

pub use app::{App, AppError, AppFrameReport};
pub use config::{AppConfig, ConfigError, InputConfig, RenderConfig, SimulationConfig, VisualizationMode};
