#![forbid(unsafe_code)]

use clap::Parser;
use fluid_sim_2d::{App, AppConfig, AppError};

fn main() -> Result<(), AppError> {
    env_logger::init();

    let config = AppConfig::parse();
    let app = App::new(config)?;
    app.run()
}
