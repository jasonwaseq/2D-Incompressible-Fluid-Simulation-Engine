use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::Vec2;
use log::{info, warn};
use pixels::{Pixels, SurfaceTexture};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize};
use winit::error::{EventLoopError, OsError};
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::config::{AppConfig, ConfigError, VisualizationMode};
use crate::input::MouseState;
use crate::render::Renderer;
use crate::sim::forces::SimCommand;
use crate::sim::grid::GridError;
use crate::sim::{FluidSolver, GridSize, SimulationState};
use crate::util::timer::{FixedStepClock, TimerError};

pub type AppResult<T> = Result<T, AppError>;

#[derive(Debug)]
pub enum AppError {
    InvalidConfiguration(ConfigError),
    InvalidGrid(GridError),
    InvalidTimer(TimerError),
    EventLoop(EventLoopError),
    WindowCreation(OsError),
    Pixels(pixels::Error),
    Texture(pixels::TextureError),
}

impl Display for AppError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration(err) => write!(f, "invalid configuration: {err}"),
            Self::InvalidGrid(err) => write!(f, "invalid grid: {err}"),
            Self::InvalidTimer(err) => write!(f, "timer setup failed: {err}"),
            Self::EventLoop(err) => write!(f, "event loop failed: {err}"),
            Self::WindowCreation(err) => write!(f, "window creation failed: {err}"),
            Self::Pixels(err) => write!(f, "pixels renderer failed: {err}"),
            Self::Texture(err) => write!(f, "pixels texture update failed: {err}"),
        }
    }
}

impl Error for AppError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidConfiguration(err) => Some(err),
            Self::InvalidGrid(err) => Some(err),
            Self::InvalidTimer(err) => Some(err),
            Self::EventLoop(err) => Some(err),
            Self::WindowCreation(err) => Some(err),
            Self::Pixels(err) => Some(err),
            Self::Texture(err) => Some(err),
        }
    }
}

impl From<ConfigError> for AppError {
    fn from(value: ConfigError) -> Self {
        Self::InvalidConfiguration(value)
    }
}

impl From<TimerError> for AppError {
    fn from(value: TimerError) -> Self {
        Self::InvalidTimer(value)
    }
}

impl From<GridError> for AppError {
    fn from(value: GridError) -> Self {
        Self::InvalidGrid(value)
    }
}

impl From<EventLoopError> for AppError {
    fn from(value: EventLoopError) -> Self {
        Self::EventLoop(value)
    }
}

impl From<OsError> for AppError {
    fn from(value: OsError) -> Self {
        Self::WindowCreation(value)
    }
}

impl From<pixels::Error> for AppError {
    fn from(value: pixels::Error) -> Self {
        Self::Pixels(value)
    }
}

impl From<pixels::TextureError> for AppError {
    fn from(value: pixels::TextureError) -> Self {
        Self::Texture(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AppFrameReport {
    pub simulated_steps: u32,
    pub paused: bool,
    pub accumulator_seconds: f32,
    pub fixed_dt_seconds: f32,
    pub frame_time_ms: f32,
}

#[derive(Debug)]
pub struct App {
    config: AppConfig,
    clock: FixedStepClock,
    solver: FluidSolver,
    state: SimulationState,
    last_frame_ms: f32,
    paused: bool,
    single_step_requested: bool,
    reset_requested: bool,
}

impl App {
    pub fn new(config: AppConfig) -> AppResult<Self> {
        config.validate()?;
        let grid = GridSize::new(
            config.simulation.grid_width,
            config.simulation.grid_height,
            config.simulation.cell_size,
        )?;

        let clock = FixedStepClock::new(
            Duration::from_secs_f32(config.simulation.dt),
            Duration::from_secs_f32(config.max_frame_time),
        )?;

        Ok(Self {
            paused: config.start_paused,
            solver: FluidSolver::new(config.simulation.clone()),
            state: SimulationState::new(grid),
            last_frame_ms: 0.0,
            config,
            clock,
            single_step_requested: false,
            reset_requested: false,
        })
    }

    pub fn config(&self) -> &AppConfig {
        &self.config
    }

    pub fn paused(&self) -> bool {
        self.paused
    }

    pub fn state(&self) -> &SimulationState {
        &self.state
    }

    pub fn visualization_mode(&self) -> VisualizationMode {
        self.config.render.visualization_mode
    }

    pub fn set_visualization_mode(&mut self, mode: VisualizationMode) {
        self.config.render.visualization_mode = mode;
    }

    pub fn show_velocity_vectors(&self) -> bool {
        self.config.render.show_velocity_vectors
    }

    pub fn toggle_velocity_vectors(&mut self) {
        self.config.render.show_velocity_vectors = !self.config.render.show_velocity_vectors;
    }

    pub fn force_scale(&self) -> f32 {
        self.config.input.force_scale
    }

    pub fn dt(&self) -> f32 {
        self.config.simulation.dt
    }

    pub fn last_frame_ms(&self) -> f32 {
        self.last_frame_ms
    }

    pub fn input_brush_radius(&self) -> f32 {
        self.config.input.brush_radius
    }

    pub fn input_dye_amount(&self) -> f32 {
        self.config.input.dye_amount
    }

    pub fn pause(&mut self) {
        self.paused = true;
        self.clock.clear_accumulator();
    }

    pub fn resume(&mut self) {
        self.paused = false;
    }

    pub fn toggle_pause(&mut self) {
        if self.paused {
            self.resume();
        } else {
            self.pause();
        }
    }

    pub fn request_single_step(&mut self) {
        self.single_step_requested = true;
    }

    pub fn request_reset(&mut self) {
        self.reset_requested = true;
        self.clock.clear_accumulator();
    }

    pub fn take_reset_requested(&mut self) -> bool {
        std::mem::take(&mut self.reset_requested)
    }

    pub fn adjust_force_scale(&mut self, scale: f32) {
        let updated = (self.config.input.force_scale * scale).clamp(1.0, 5_000.0);
        self.config.input.force_scale = updated;
    }

    pub fn adjust_dt(&mut self, scale: f32) -> AppResult<()> {
        let dt = (self.config.simulation.dt * scale).clamp(1.0 / 240.0, 0.1);
        self.config.simulation.dt = dt;
        self.solver.set_dt(dt);
        self.clock.set_step(Duration::from_secs_f32(dt))?;
        self.clock.clear_accumulator();
        Ok(())
    }

    pub fn simulation_position_from_pixel(&self, pixel_x: usize, pixel_y: usize) -> Vec2 {
        let h = self.state.grid.cell_size;
        Vec2::new((pixel_x as f32 + 0.5) * h, (pixel_y as f32 + 0.5) * h)
    }

    pub fn update(&mut self, real_dt: Duration) -> AppFrameReport {
        self.simulate_frame(real_dt, &[])
    }

    pub fn simulate_frame(&mut self, real_dt: Duration, commands: &[SimCommand]) -> AppFrameReport {
        self.last_frame_ms = real_dt.as_secs_f32() * 1000.0;

        if self.take_reset_requested() {
            self.state.clear();
        }

        let simulated_steps = if self.paused {
            self.clock.clear_accumulator();

            if self.single_step_requested {
                self.single_step_requested = false;
                1
            } else {
                0
            }
        } else {
            self.clock.accumulate_and_consume(real_dt)
        };

        for step_index in 0..simulated_steps {
            let step_commands = if step_index == 0 { commands } else { &[] };
            self.solver.step(&mut self.state, step_commands);
        }

        AppFrameReport {
            simulated_steps,
            paused: self.paused,
            accumulator_seconds: self.clock.accumulator().as_secs_f32(),
            fixed_dt_seconds: self.clock.step().as_secs_f32(),
            frame_time_ms: self.last_frame_ms,
        }
    }

    pub fn window_title(&self) -> String {
        let mode = Renderer::mode_label(self.visualization_mode());
        let run_state = if self.paused { "Paused" } else { "Running" };
        let vectors = if self.show_velocity_vectors() { "vectors:on" } else { "vectors:off" };
        format!(
            "Fluid Sim 2D | {mode} | {run_state} | dt={:.4} | frame={:.2}ms | sim={:.2}ms | cfl={:.3} | div={:.3e} | vort={:.3e} | iters={} | force={:.1} | {vectors}",
            self.dt(),
            self.last_frame_ms(),
            self.state.stats.step_ms,
            self.state.stats.cfl,
            self.state.stats.max_divergence,
            self.state.stats.max_vorticity,
            self.state.stats.pressure_iterations,
            self.force_scale(),
        )
    }

    pub fn run(self) -> AppResult<()> {
        info!("Launching interactive fluid-engine app.");
        info!("Simulation config: {:?}", self.config.simulation);
        info!("Render config: {:?}", self.config.render);
        info!("Input config: {:?}", self.config.input);

        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);

        let mut runtime = WindowedApp::new(self);
        let run_result = event_loop.run_app(&mut runtime);

        if let Some(err) = runtime.take_fatal_error() {
            return Err(err);
        }

        run_result?;
        Ok(())
    }
}

#[derive(Debug)]
struct WindowedApp {
    app: App,
    renderer: Renderer,
    mouse: MouseState,
    window: Option<Arc<Window>>,
    pixels: Option<Pixels<'static>>,
    last_redraw_at: Option<Instant>,
    last_stats_log_at: Option<Instant>,
    fatal_error: Option<AppError>,
}

impl WindowedApp {
    fn new(app: App) -> Self {
        let renderer = Renderer::new(app.show_velocity_vectors());

        Self {
            app,
            renderer,
            mouse: MouseState::default(),
            window: None,
            pixels: None,
            last_redraw_at: None,
            last_stats_log_at: None,
            fatal_error: None,
        }
    }

    fn take_fatal_error(&mut self) -> Option<AppError> {
        self.fatal_error.take()
    }

    fn set_fatal_error(&mut self, event_loop: &ActiveEventLoop, error: AppError) {
        self.fatal_error = Some(error);
        event_loop.exit();
    }

    fn create_window_and_pixels(&mut self, event_loop: &ActiveEventLoop) -> AppResult<()> {
        let grid = self.app.state.grid;
        let scale = self.app.config.render.window_scale as f64;

        let window_attributes = Window::default_attributes()
            .with_title(self.app.window_title())
            .with_inner_size(LogicalSize::new(grid.nx as f64 * scale, grid.ny as f64 * scale))
            .with_min_inner_size(LogicalSize::new(grid.nx as f64, grid.ny as f64));

        let window = Arc::new(event_loop.create_window(window_attributes)?);
        let size = window.inner_size();
        let surface_texture = SurfaceTexture::new(size.width.max(1), size.height.max(1), window.clone());
        let pixels = Pixels::new(grid.nx as u32, grid.ny as u32, surface_texture)?;

        self.window = Some(window);
        self.pixels = Some(pixels);
        self.last_redraw_at = Some(Instant::now());
        Ok(())
    }

    fn handle_resize(&mut self, size: PhysicalSize<u32>) -> AppResult<()> {
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        if let Some(pixels) = self.pixels.as_mut() {
            pixels.resize_surface(size.width, size.height)?;
        }

        Ok(())
    }

    fn update_cursor_from_window_position(&mut self, position: PhysicalPosition<f64>) {
        let Some(pixels) = self.pixels.as_ref() else {
            return;
        };

        let physical_position = (position.x as f32, position.y as f32);
        let pixel_position = pixels
            .window_pos_to_pixel(physical_position)
            .unwrap_or_else(|outside| pixels.clamp_pixel_pos(outside));

        let simulation_position =
            self.app
                .simulation_position_from_pixel(pixel_position.0, pixel_position.1);
        self.mouse.set_cursor_position(simulation_position);
    }

    fn handle_keycode(&mut self, event_loop: &ActiveEventLoop, code: KeyCode) -> AppResult<()> {
        match code {
            KeyCode::Escape => event_loop.exit(),
            KeyCode::Space => self.app.toggle_pause(),
            KeyCode::KeyN => self.app.request_single_step(),
            KeyCode::KeyR => self.app.request_reset(),
            KeyCode::Digit1 => self.app.set_visualization_mode(VisualizationMode::Density),
            KeyCode::Digit2 => self
                .app
                .set_visualization_mode(VisualizationMode::VelocityMagnitude),
            KeyCode::Digit3 => self.app.set_visualization_mode(VisualizationMode::Pressure),
            KeyCode::Digit4 => self.app.set_visualization_mode(VisualizationMode::Divergence),
            KeyCode::Digit5 => self.app.set_visualization_mode(VisualizationMode::Vorticity),
            KeyCode::BracketLeft => self.app.adjust_force_scale(0.9),
            KeyCode::BracketRight => self.app.adjust_force_scale(1.1),
            KeyCode::Minus => self.app.adjust_dt(0.9)?,
            KeyCode::Equal => self.app.adjust_dt(1.1)?,
            KeyCode::KeyV => {
                self.app.toggle_velocity_vectors();
                self.renderer
                    .set_show_velocity_vectors(self.app.show_velocity_vectors());
            }
            _ => {}
        }

        if let Some(window) = self.window.as_ref() {
            window.set_title(&self.app.window_title());
        }

        Ok(())
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) -> AppResult<()> {
        if self.window.is_none() {
            return Ok(());
        }

        let now = Instant::now();
        let real_dt = self
            .last_redraw_at
            .replace(now)
            .map(|previous| now.saturating_duration_since(previous))
            .unwrap_or_else(|| Duration::from_secs_f32(self.app.dt()));

        let commands = self.mouse.build_drag_commands(
            self.app.input_brush_radius(),
            self.app.input_dye_amount(),
            self.app.force_scale(),
        );
        let report = self.app.simulate_frame(real_dt, &commands);
        self.log_runtime_stats_if_due(&report);

        let Some(pixels) = self.pixels.as_mut() else {
            return Ok(());
        };

        self.renderer
            .draw(self.app.state(), self.app.visualization_mode(), pixels.frame_mut());

        let window = self
            .window
            .as_ref()
            .expect("window should remain available during redraw");
        window.set_title(&self.app.window_title());
        window.pre_present_notify();

        if let Err(err) = pixels.render() {
            self.set_fatal_error(event_loop, err.into());
        }

        Ok(())
    }

    fn log_runtime_stats_if_due(&mut self, report: &AppFrameReport) {
        let now = Instant::now();
        let should_log = self
            .last_stats_log_at
            .map(|previous| now.saturating_duration_since(previous) >= Duration::from_secs(1))
            .unwrap_or(true);

        if !should_log {
            return;
        }

        self.last_stats_log_at = Some(now);

        let stats = &self.app.state.stats;
        info!(
            "mode={} paused={} frame_ms={:.2} sim_ms={:.2} steps={} cfl={:.3} max_div={:.3e} max_vort={:.3e} iters={}",
            Renderer::mode_label(self.app.visualization_mode()),
            report.paused,
            report.frame_time_ms,
            stats.step_ms,
            report.simulated_steps,
            stats.cfl,
            stats.max_divergence,
            stats.max_vorticity,
            stats.pressure_iterations,
        );

        if report.simulated_steps > 0 && stats.cfl > 1.0 {
            warn!(
                "CFL exceeded 1.0 (cfl={:.3}); expect extra semi-Lagrangian dissipation or unstable-looking transport.",
                stats.cfl
            );
        }

        if report.simulated_steps > 0 && stats.max_divergence > 1.0e-2 {
            warn!(
                "Projection residual is still high (max_div={:.3e}); pressure iterations may be insufficient.",
                stats.max_divergence
            );
        }
    }
}

impl ApplicationHandler for WindowedApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        if let Err(err) = self.create_window_and_pixels(event_loop) {
            self.set_fatal_error(event_loop, err);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.as_ref() else {
            return;
        };

        if window.id() != window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Err(err) = self.handle_resize(size) {
                    self.set_fatal_error(event_loop, err);
                }
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                if let Err(err) = self.handle_resize(window.inner_size()) {
                    self.set_fatal_error(event_loop, err);
                }
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && !event.repeat =>
            {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if let Err(err) = self.handle_keycode(event_loop, code) {
                        self.set_fatal_error(event_loop, err);
                    }
                }
            }
            WindowEvent::MouseInput { button, state, .. } => {
                if button == winit::event::MouseButton::Left {
                    self.mouse.set_left_button_down(state == ElementState::Pressed);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.update_cursor_from_window_position(position);
            }
            WindowEvent::CursorLeft { .. } => {
                self.mouse.set_left_button_down(false);
            }
            WindowEvent::RedrawRequested => {
                if let Err(err) = self.redraw(event_loop) {
                    self.set_fatal_error(event_loop, err);
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        warn!("Exiting fluid simulation app.");
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use glam::Vec2;

    use crate::config::AppConfig;
    use crate::sim::forces::SimCommand;

    use super::App;

    #[test]
    fn paused_app_only_steps_when_requested() {
        let mut config = AppConfig::default();
        config.start_paused = true;

        let mut app = App::new(config).expect("app should construct");

        let report = app.update(Duration::from_secs_f32(0.5));
        assert_eq!(report.simulated_steps, 0);
        assert!(report.paused);

        app.request_single_step();
        let report = app.update(Duration::from_secs_f32(0.5));
        assert_eq!(report.simulated_steps, 1);
    }

    #[test]
    fn running_app_consumes_fixed_steps() {
        let mut config = AppConfig::default();
        config.simulation.dt = 0.01;
        let mut app = App::new(config).expect("app should construct");

        let report = app.update(Duration::from_secs_f32(0.050));
        assert_eq!(report.simulated_steps, 5);
        assert!((report.frame_time_ms - 50.0).abs() <= 1.0e-3);
        assert_eq!(app.last_frame_ms(), report.frame_time_ms);
    }

    #[test]
    fn app_simulation_frame_applies_commands_to_state() {
        let config = AppConfig::default();
        let mut app = App::new(config).expect("app should construct");

        app.simulate_frame(
            Duration::from_secs_f32(1.0 / 60.0),
            &[SimCommand::AddDye {
                position: Vec2::new(80.0, 45.0),
                amount: 2.0,
                radius: 6.0,
            }],
        );

        let max_density = app
            .state()
            .density
            .as_slice()
            .iter()
            .fold(0.0_f32, |acc, value| acc.max(*value));

        assert!(max_density > 0.0);
    }

    #[test]
    fn dt_adjustment_updates_clock_and_solver() {
        let config = AppConfig::default();
        let mut app = App::new(config).expect("app should construct");

        app.adjust_dt(1.1).expect("dt adjustment should succeed");

        assert!(app.dt() > 1.0 / 60.0);
    }

    #[test]
    fn pixel_coordinates_map_to_simulation_positions() {
        let config = AppConfig::default();
        let app = App::new(config).expect("app should construct");

        assert_eq!(app.simulation_position_from_pixel(0, 0), Vec2::new(0.5, 0.5));
        assert_eq!(app.simulation_position_from_pixel(3, 5), Vec2::new(3.5, 5.5));
    }

    #[test]
    fn window_title_exposes_runtime_diagnostics() {
        let config = AppConfig::default();
        let mut app = App::new(config).expect("app should construct");

        app.update(Duration::from_secs_f32(1.0 / 60.0));
        let title = app.window_title();

        assert!(title.contains("frame="));
        assert!(title.contains("sim="));
        assert!(title.contains("cfl="));
        assert!(title.contains("vort="));
        assert!(title.contains("iters="));
    }
}
