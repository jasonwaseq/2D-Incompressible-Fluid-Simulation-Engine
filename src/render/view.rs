use glam::Vec2;

use crate::config::VisualizationMode;
use crate::render::colormap::{fire, grayscale, rgba, signed_blue_red};
use crate::sim::state::SimulationState;

#[derive(Debug, Clone)]
pub struct Renderer {
    show_velocity_vectors: bool,
    vector_stride: usize,
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new(false)
    }
}

impl Renderer {
    pub fn new(show_velocity_vectors: bool) -> Self {
        Self {
            show_velocity_vectors,
            vector_stride: 10,
        }
    }

    pub fn mode_label(mode: VisualizationMode) -> &'static str {
        match mode {
            VisualizationMode::Density => "Density",
            VisualizationMode::VelocityMagnitude => "Velocity",
            VisualizationMode::Pressure => "Pressure",
            VisualizationMode::Divergence => "Divergence",
            VisualizationMode::Vorticity => "Vorticity",
        }
    }

    pub fn show_velocity_vectors(&self) -> bool {
        self.show_velocity_vectors
    }

    pub fn set_show_velocity_vectors(&mut self, show_velocity_vectors: bool) {
        self.show_velocity_vectors = show_velocity_vectors;
    }

    pub fn draw(
        &mut self,
        state: &SimulationState,
        mode: VisualizationMode,
        frame: &mut [u8],
    ) {
        let grid = state.grid;
        assert_eq!(
            frame.len(),
            grid.nx * grid.ny * 4,
            "frame size must match the scalar render target dimensions"
        );

        let density_scale = interior_max(state.density.as_slice(), grid.nx, grid.ny, 1.0);
        let pressure_scale = interior_max_abs(state.pressure.as_slice(), grid.nx, grid.ny, 1.0e-5);
        let divergence_scale =
            interior_max_abs(state.divergence.as_slice(), grid.nx, grid.ny, 1.0e-6);
        let vorticity_scale =
            interior_max_abs(state.vorticity.as_slice(), grid.nx, grid.ny, 1.0e-6);
        let velocity_scale = state.velocity.max_speed().max(1.0e-6);

        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let rgba = match mode {
                    VisualizationMode::Density => fire(state.density.get_cell(i, j) / density_scale),
                    VisualizationMode::VelocityMagnitude => {
                        let speed = state.velocity.cell_center_velocity(i, j).length();
                        grayscale(speed / velocity_scale)
                    }
                    VisualizationMode::Pressure => {
                        signed_blue_red(state.pressure.get_cell(i, j) / pressure_scale)
                    }
                    VisualizationMode::Divergence => {
                        signed_blue_red(state.divergence.get_cell(i, j) / divergence_scale)
                    }
                    VisualizationMode::Vorticity => {
                        signed_blue_red(state.vorticity.get_cell(i, j) / vorticity_scale)
                    }
                };

                put_pixel(frame, grid.nx, i, j, rgba);
            }
        }

        if self.show_velocity_vectors {
            self.draw_velocity_vectors(state, frame);
        }
    }

    fn draw_velocity_vectors(&self, state: &SimulationState, frame: &mut [u8]) {
        let grid = state.grid;
        let max_speed = state.velocity.max_speed();
        if max_speed <= 1.0e-6 {
            return;
        }

        for j in (0..grid.ny).step_by(self.vector_stride) {
            for i in (0..grid.nx).step_by(self.vector_stride) {
                let velocity = state.velocity.cell_center_velocity(i, j);
                let speed = velocity.length();
                if speed <= 1.0e-6 {
                    continue;
                }

                let start = Vec2::new(i as f32 + 0.5, j as f32 + 0.5);
                let length = 0.45 * self.vector_stride as f32 * (speed / max_speed).clamp(0.0, 1.0);
                let end = start + velocity.normalize() * length;

                draw_line(frame, grid.nx, grid.ny, start, end, rgba(255, 255, 255));
            }
        }
    }
}

fn interior_max(values: &[f32], nx: usize, ny: usize, min_span: f32) -> f32 {
    let row_stride = nx + 2;
    let mut max_value = 0.0_f32;

    for j in 0..ny {
        let row_start = (j + 1) * row_stride + 1;
        let row_end = row_start + nx;
        for value in &values[row_start..row_end] {
            max_value = max_value.max(*value);
        }
    }

    max_value.max(min_span)
}

fn interior_max_abs(values: &[f32], nx: usize, ny: usize, min_span: f32) -> f32 {
    let row_stride = nx + 2;
    let mut max_value = 0.0_f32;

    for j in 0..ny {
        let row_start = (j + 1) * row_stride + 1;
        let row_end = row_start + nx;
        for value in &values[row_start..row_end] {
            max_value = max_value.max(value.abs());
        }
    }

    max_value.max(min_span)
}

fn put_pixel(frame: &mut [u8], width: usize, x: usize, y: usize, color: [u8; 4]) {
    let index = (x + y * width) * 4;
    frame[index..index + 4].copy_from_slice(&color);
}

fn draw_line(frame: &mut [u8], width: usize, height: usize, start: Vec2, end: Vec2, color: [u8; 4]) {
    let delta = end - start;
    let steps = delta.abs().max_element().ceil().max(1.0) as usize;

    for step in 0..=steps {
        let t = step as f32 / steps as f32;
        let point = start + delta * t;
        let x = point.x.round() as isize;
        let y = point.y.round() as isize;

        if x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height {
            put_pixel(frame, width, x as usize, y as usize, color);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Renderer;
    use crate::config::VisualizationMode;
    use crate::sim::grid::GridSize;
    use crate::sim::state::SimulationState;

    #[test]
    fn mode_labels_match_expected_debug_names() {
        assert_eq!(Renderer::mode_label(VisualizationMode::Density), "Density");
        assert_eq!(
            Renderer::mode_label(VisualizationMode::VelocityMagnitude),
            "Velocity"
        );
        assert_eq!(Renderer::mode_label(VisualizationMode::Pressure), "Pressure");
        assert_eq!(
            Renderer::mode_label(VisualizationMode::Divergence),
            "Divergence"
        );
        assert_eq!(Renderer::mode_label(VisualizationMode::Vorticity), "Vorticity");
    }

    #[test]
    fn density_view_draws_non_black_pixels_when_density_is_present() {
        let grid = GridSize::new(8, 6, 1.0).expect("grid should be valid");
        let mut state = SimulationState::new(grid);
        let mut renderer = Renderer::new(false);
        let mut frame = vec![0; grid.nx * grid.ny * 4];

        state.density.set_cell(3, 2, 1.0);
        renderer.draw(&state, VisualizationMode::Density, &mut frame);

        assert!(frame.iter().any(|component| *component != 0));
    }

    #[test]
    fn vector_overlay_draws_white_samples_for_nonzero_velocity() {
        let grid = GridSize::new(12, 12, 1.0).expect("grid should be valid");
        let mut state = SimulationState::new(grid);
        let mut renderer = Renderer::new(true);
        let mut frame = vec![0; grid.nx * grid.ny * 4];

        for j in 0..grid.ny {
            for i in 0..=grid.nx {
                state.velocity.u.set_face(i, j, 1.0);
            }
        }

        renderer.draw(&state, VisualizationMode::VelocityMagnitude, &mut frame);

        assert!(frame.chunks_exact(4).any(|pixel| pixel[0] == 255 && pixel[1] == 255 && pixel[2] == 255));
    }
}
