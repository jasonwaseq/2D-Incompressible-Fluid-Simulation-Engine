use std::fmt::{Display, Formatter};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use wgpu::util::DeviceExt;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::window::Window;

use crate::config::VisualizationMode;
use crate::sim::{GpuFluidBackend, GridSize, SimulationState};

#[derive(Debug)]
pub enum GpuSurfaceRendererError {
    CreateSurface(wgpu::CreateSurfaceError),
    UnsupportedSurface,
    NoSurfaceFormats,
    Surface(wgpu::SurfaceError),
}

impl Display for GpuSurfaceRendererError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CreateSurface(err) => write!(f, "failed to create GPU surface: {err}"),
            Self::UnsupportedSurface => write!(
                f,
                "the selected GPU adapter cannot present to this window surface"
            ),
            Self::NoSurfaceFormats => write!(f, "the window surface reported no supported formats"),
            Self::Surface(err) => write!(f, "GPU surface error: {err}"),
        }
    }
}

impl std::error::Error for GpuSurfaceRendererError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CreateSurface(err) => Some(err),
            Self::Surface(err) => Some(err),
            _ => None,
        }
    }
}

impl From<wgpu::CreateSurfaceError> for GpuSurfaceRendererError {
    fn from(value: wgpu::CreateSurfaceError) -> Self {
        Self::CreateSurface(value)
    }
}

impl From<wgpu::SurfaceError> for GpuSurfaceRendererError {
    fn from(value: wgpu::SurfaceError) -> Self {
        Self::Surface(value)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuRenderParams {
    surface_and_grid: [u32; 4],
    mode_and_flags: [u32; 4],
    primary_scales: [f32; 4],
    secondary_scales: [f32; 4],
}

impl GpuRenderParams {
    fn new(size: PhysicalSize<u32>, state: &SimulationState, mode: VisualizationMode) -> Self {
        let grid = state.grid;
        Self {
            surface_and_grid: [
                size.width.max(1),
                size.height.max(1),
                grid.nx as u32,
                grid.ny as u32,
            ],
            mode_and_flags: [visualization_mode_id(mode), 0, 0, 0],
            primary_scales: [
                interior_max(state.density.as_slice(), grid.nx, grid.ny, 1.0),
                interior_max_abs(state.pressure.as_slice(), grid.nx, grid.ny, 1.0e-5),
                interior_max_abs(state.divergence.as_slice(), grid.nx, grid.ny, 1.0e-6),
                interior_max_abs(state.vorticity.as_slice(), grid.nx, grid.ny, 1.0e-6),
            ],
            secondary_scales: [state.velocity.max_speed().max(1.0e-6), 0.0, 0.0, 0.0],
        }
    }
}

pub struct GpuSurfaceRenderer {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    params_buffer: wgpu::Buffer,
    params_bind_group: wgpu::BindGroup,
    fields_bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
}

impl GpuSurfaceRenderer {
    pub fn new(
        window: Arc<Window>,
        backend: &GpuFluidBackend,
    ) -> Result<Self, GpuSurfaceRendererError> {
        let surface = backend.instance().create_surface(window.clone())?;
        if !backend.adapter().is_surface_supported(&surface) {
            return Err(GpuSurfaceRendererError::UnsupportedSurface);
        }

        let size = window.inner_size();
        let caps = surface.get_capabilities(backend.adapter());
        let format = caps
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .or_else(|| caps.formats.first().copied())
            .ok_or(GpuSurfaceRendererError::NoSurfaceFormats)?;
        let present_mode = caps
            .present_modes
            .iter()
            .copied()
            .find(|mode| *mode == wgpu::PresentMode::Mailbox)
            .unwrap_or(caps.present_modes[0]);
        let alpha_mode = caps.alpha_modes[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            desired_maximum_frame_latency: 1,
            alpha_mode,
            view_formats: vec![],
        };
        surface.configure(backend.device(), &config);

        let params = GpuRenderParams::new(size, backend.state(), VisualizationMode::Density);
        let params_buffer =
            backend
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("fluid-sim-gpu-render-params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let params_layout =
            backend
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("fluid-sim-gpu-render-params-layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
        let params_bind_group = backend
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fluid-sim-gpu-render-params-bind-group"),
                layout: &params_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                }],
            });

        let fields_layout =
            backend
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("fluid-sim-gpu-render-fields-layout"),
                    entries: &[
                        storage_entry(0),
                        storage_entry(1),
                        storage_entry(2),
                        storage_entry(3),
                        storage_entry(4),
                        storage_entry(5),
                        storage_entry(6),
                    ],
                });
        let fields_bind_group = backend
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fluid-sim-gpu-render-fields-bind-group"),
                layout: &fields_layout,
                entries: &[
                    bind_entry(0, backend.density_buffer()),
                    bind_entry(1, backend.pressure_buffer()),
                    bind_entry(2, backend.divergence_buffer()),
                    bind_entry(3, backend.u_buffer()),
                    bind_entry(4, backend.v_buffer()),
                    bind_entry(5, backend.vorticity_buffer()),
                    bind_entry(6, backend.obstacle_buffer()),
                ],
            });

        let shader = backend
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fluid-sim-gpu-render-shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("gpu_view.wgsl").into()),
            });

        let pipeline_layout =
            backend
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("fluid-sim-gpu-render-layout"),
                    bind_group_layouts: &[&params_layout, &fields_layout],
                    push_constant_ranges: &[],
                });
        let pipeline = backend
            .device()
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("fluid-sim-gpu-render-pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

        Ok(Self {
            surface,
            config,
            params_buffer,
            params_bind_group,
            fields_bind_group,
            pipeline,
        })
    }

    pub fn resize(&mut self, backend: &GpuFluidBackend, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }

        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(backend.device(), &self.config);
    }

    pub fn draw(
        &mut self,
        backend: &GpuFluidBackend,
        mode: VisualizationMode,
    ) -> Result<(), GpuSurfaceRendererError> {
        backend.queue().write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&GpuRenderParams::new(
                PhysicalSize::new(self.config.width, self.config.height),
                backend.state(),
                mode,
            )),
        );

        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Timeout) => return Ok(()),
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                self.surface.configure(backend.device(), &self.config);
                return Ok(());
            }
            Err(err) => return Err(err.into()),
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder =
            backend
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fluid-sim-gpu-render-encoder"),
                });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fluid-sim-gpu-render-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.params_bind_group, &[]);
            pass.set_bind_group(1, &self.fields_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        backend.queue().submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }

    pub fn window_position_to_simulation(
        &self,
        position: PhysicalPosition<f64>,
        grid: GridSize,
    ) -> Vec2 {
        let (origin_x, origin_y, viewport_width, viewport_height) = viewport_rect(
            self.config.width as f32,
            self.config.height as f32,
            grid.nx as f32,
            grid.ny as f32,
        );

        let x = ((position.x as f32 - origin_x) / viewport_width).clamp(0.0, 1.0);
        let y = ((position.y as f32 - origin_y) / viewport_height).clamp(0.0, 1.0);
        Vec2::new(
            x * grid.nx as f32 * grid.cell_size,
            y * grid.ny as f32 * grid.cell_size,
        )
    }
}

fn bind_entry<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn visualization_mode_id(mode: VisualizationMode) -> u32 {
    match mode {
        VisualizationMode::Density => 0,
        VisualizationMode::VelocityMagnitude => 1,
        VisualizationMode::Pressure => 2,
        VisualizationMode::Divergence => 3,
        VisualizationMode::Vorticity => 4,
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

fn viewport_rect(
    surface_width: f32,
    surface_height: f32,
    grid_width: f32,
    grid_height: f32,
) -> (f32, f32, f32, f32) {
    let surface_aspect = surface_width / surface_height.max(1.0);
    let grid_aspect = grid_width / grid_height.max(1.0);

    if surface_aspect > grid_aspect {
        let viewport_height = surface_height.max(1.0);
        let viewport_width = viewport_height * grid_aspect;
        let origin_x = 0.5 * (surface_width - viewport_width);
        (origin_x, 0.0, viewport_width.max(1.0), viewport_height)
    } else {
        let viewport_width = surface_width.max(1.0);
        let viewport_height = viewport_width / grid_aspect;
        let origin_y = 0.5 * (surface_height - viewport_height);
        (0.0, origin_y, viewport_width, viewport_height.max(1.0))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::{viewport_rect, GpuRenderParams};

    #[test]
    fn viewport_rect_letterboxes_wider_windows() {
        let (origin_x, origin_y, width, height) = viewport_rect(1920.0, 1080.0, 4.0, 3.0);
        assert!(origin_x > 0.0);
        assert_abs_diff_eq!(origin_y, 0.0);
        assert_abs_diff_eq!(height, 1080.0);
        assert!(width < 1920.0);
    }

    #[test]
    fn viewport_rect_letterboxes_taller_windows() {
        let (origin_x, origin_y, width, height) = viewport_rect(900.0, 1200.0, 16.0, 9.0);
        assert_abs_diff_eq!(origin_x, 0.0);
        assert!(origin_y > 0.0);
        assert_abs_diff_eq!(width, 900.0);
        assert!(height < 1200.0);
    }

    #[test]
    fn render_params_uniform_uses_explicit_16_byte_lanes() {
        assert_eq!(std::mem::size_of::<GpuRenderParams>(), 64);
    }
}
