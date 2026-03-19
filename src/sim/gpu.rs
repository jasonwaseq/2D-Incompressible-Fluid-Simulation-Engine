use std::fmt::{Display, Formatter};
use std::sync::mpsc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use log::warn;
use wgpu::util::DeviceExt;

use crate::config::{ScalarAdvectionKind, SimulationConfig};

use super::effects::max_abs_vorticity;
use super::field::MacVelocity;
use super::forces::SimCommand;
use super::grid::GridSize;
use super::pressure::max_abs_divergence;
use super::state::SimulationState;

const WORKGROUP_SIZE: u32 = 8;
const MAX_GPU_COMMANDS: usize = 64;
const CMD_DYE: u32 = 1;
const CMD_FORCE: u32 = 2;

#[derive(Debug)]
pub enum GpuBackendError {
    NoAdapter,
    RequestDevice(wgpu::RequestDeviceError),
    Map(String),
}

impl Display for GpuBackendError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "no compatible GPU adapter found for compute backend"),
            Self::RequestDevice(err) => write!(f, "failed to create GPU device: {err}"),
            Self::Map(message) => write!(f, "GPU readback failed: {message}"),
        }
    }
}

impl std::error::Error for GpuBackendError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::RequestDevice(err) => Some(err),
            _ => None,
        }
    }
}

impl From<wgpu::RequestDeviceError> for GpuBackendError {
    fn from(value: wgpu::RequestDeviceError) -> Self {
        Self::RequestDevice(value)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSimParams {
    nx: u32,
    ny: u32,
    scalar_width: u32,
    scalar_height: u32,
    u_width: u32,
    u_height: u32,
    v_width: u32,
    v_height: u32,
    cell_size: f32,
    dt: f32,
    viscosity: f32,
    diffusion: f32,
    command_count: u32,
    pressure_phase: u32,
    pressure_anchor_x: u32,
    pressure_anchor_y: u32,
}

impl GpuSimParams {
    fn new(
        config: &SimulationConfig,
        grid: GridSize,
        command_count: u32,
        pressure_phase: u32,
        pressure_anchor: Option<(u32, u32)>,
    ) -> Self {
        Self {
            nx: grid.nx as u32,
            ny: grid.ny as u32,
            scalar_width: grid.scalar_shape().width as u32,
            scalar_height: grid.scalar_shape().height as u32,
            u_width: grid.u_shape().width as u32,
            u_height: grid.u_shape().height as u32,
            v_width: grid.v_shape().width as u32,
            v_height: grid.v_shape().height as u32,
            cell_size: grid.cell_size,
            dt: config.dt,
            viscosity: config.viscosity,
            diffusion: config.diffusion,
            command_count,
            pressure_phase,
            pressure_anchor_x: pressure_anchor.map(|(x, _)| x).unwrap_or(u32::MAX),
            pressure_anchor_y: pressure_anchor.map(|(_, y)| y).unwrap_or(u32::MAX),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCommandRecord {
    kind: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    position: [f32; 2],
    delta: [f32; 2],
    amount: f32,
    radius: f32,
    _pad3: [f32; 2],
}

struct GpuBuffers {
    density0: wgpu::Buffer,
    density1: wgpu::Buffer,
    density2: wgpu::Buffer,
    u0: wgpu::Buffer,
    u1: wgpu::Buffer,
    u2: wgpu::Buffer,
    v0: wgpu::Buffer,
    v1: wgpu::Buffer,
    v2: wgpu::Buffer,
    pressure0: wgpu::Buffer,
    divergence: wgpu::Buffer,
    vorticity: wgpu::Buffer,
    obstacles: wgpu::Buffer,
    scalar_bytes: u64,
    u_bytes: u64,
    v_bytes: u64,
}

struct GpuStagingBuffers {
    density: wgpu::Buffer,
    pressure: wgpu::Buffer,
    divergence: wgpu::Buffer,
    vorticity: wgpu::Buffer,
    u: wgpu::Buffer,
    v: wgpu::Buffer,
}

struct GpuPipelines {
    apply_scalar_commands: wgpu::ComputePipeline,
    apply_u_commands: wgpu::ComputePipeline,
    apply_v_commands: wgpu::ComputePipeline,
    apply_scalar_boundary: wgpu::ComputePipeline,
    apply_u_boundary: wgpu::ComputePipeline,
    apply_v_boundary: wgpu::ComputePipeline,
    advect_scalar: wgpu::ComputePipeline,
    maccormack_correct: wgpu::ComputePipeline,
    advect_u: wgpu::ComputePipeline,
    advect_v: wgpu::ComputePipeline,
    diffuse_scalar_jacobi: wgpu::ComputePipeline,
    diffuse_u_jacobi: wgpu::ComputePipeline,
    diffuse_v_jacobi: wgpu::ComputePipeline,
    compute_divergence: wgpu::ComputePipeline,
    pressure_red_black: wgpu::ComputePipeline,
    project_u: wgpu::ComputePipeline,
    project_v: wgpu::ComputePipeline,
    compute_vorticity: wgpu::ComputePipeline,
}

pub struct GpuFluidBackend {
    config: SimulationConfig,
    state: SimulationState,
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    params: GpuSimParams,
    params_bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    fields_layout: wgpu::BindGroupLayout,
    command_buffer: wgpu::Buffer,
    dummy_buffer: wgpu::Buffer,
    buffers: GpuBuffers,
    staging: GpuStagingBuffers,
    pipelines: GpuPipelines,
    readback_interval_steps: u32,
    steps_since_readback: u32,
}
impl GpuFluidBackend {
    pub fn new(config: SimulationConfig, grid: GridSize) -> Result<Self, GpuBackendError> {
        if config.pressure_solver != crate::config::PressureSolverKind::GaussSeidel {
            warn!(
                "GPU backend currently uses a red/black Gauss-Seidel pressure iteration regardless of --pressure-solver; requested {:?} will be ignored.",
                config.pressure_solver
            );
        }

        if config.buoyancy > 0.0 || config.vorticity_confinement > 0.0 {
            warn!(
                "GPU backend currently focuses on advection/projection core passes; buoyancy and vorticity confinement are not applied on the GPU path yet."
            );
        }

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or(GpuBackendError::NoAdapter)?;

        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("fluid-sim-gpu-device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
            },
            None,
        ))?;

        let params = GpuSimParams::new(&config, grid, 0, 0, None);
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fluid-sim-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fluid-sim-params-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fluid-sim-params-bind-group"),
            layout: &params_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            }],
        });

        let fields_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fluid-sim-fields-layout"),
            entries: &[
                storage_entry(0, false),
                storage_entry(1, false),
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, true),
                storage_entry(7, true),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fluid-sim-compute-layout"),
            bind_group_layouts: &[&params_layout, &fields_layout],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fluid-sim-compute-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu.wgsl").into()),
        });

        let buffers = GpuBuffers::new(&device, grid);
        let staging = GpuStagingBuffers::new(&device, &buffers);
        let dummy_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fluid-sim-dummy-storage"),
            contents: bytemuck::cast_slice(&[0.0f32]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let command_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fluid-sim-command-buffer"),
            size: (MAX_GPU_COMMANDS * std::mem::size_of::<GpuCommandRecord>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipelines = GpuPipelines {
            apply_scalar_commands: create_compute_pipeline(
                &device,
                &pipeline_layout,
                &shader,
                "apply_scalar_commands",
            ),
            apply_u_commands: create_compute_pipeline(
                &device,
                &pipeline_layout,
                &shader,
                "apply_u_commands",
            ),
            apply_v_commands: create_compute_pipeline(
                &device,
                &pipeline_layout,
                &shader,
                "apply_v_commands",
            ),
            apply_scalar_boundary: create_compute_pipeline(
                &device,
                &pipeline_layout,
                &shader,
                "apply_scalar_boundary",
            ),
            apply_u_boundary: create_compute_pipeline(
                &device,
                &pipeline_layout,
                &shader,
                "apply_u_boundary",
            ),
            apply_v_boundary: create_compute_pipeline(
                &device,
                &pipeline_layout,
                &shader,
                "apply_v_boundary",
            ),
            advect_scalar: create_compute_pipeline(
                &device,
                &pipeline_layout,
                &shader,
                "advect_scalar",
            ),
            maccormack_correct: create_compute_pipeline(
                &device,
                &pipeline_layout,
                &shader,
                "maccormack_correct",
            ),
            advect_u: create_compute_pipeline(&device, &pipeline_layout, &shader, "advect_u"),
            advect_v: create_compute_pipeline(&device, &pipeline_layout, &shader, "advect_v"),
            diffuse_scalar_jacobi: create_compute_pipeline(
                &device,
                &pipeline_layout,
                &shader,
                "diffuse_scalar_jacobi",
            ),
            diffuse_u_jacobi: create_compute_pipeline(
                &device,
                &pipeline_layout,
                &shader,
                "diffuse_u_jacobi",
            ),
            diffuse_v_jacobi: create_compute_pipeline(
                &device,
                &pipeline_layout,
                &shader,
                "diffuse_v_jacobi",
            ),
            compute_divergence: create_compute_pipeline(
                &device,
                &pipeline_layout,
                &shader,
                "compute_divergence",
            ),
            pressure_red_black: create_compute_pipeline(
                &device,
                &pipeline_layout,
                &shader,
                "pressure_red_black",
            ),
            project_u: create_compute_pipeline(&device, &pipeline_layout, &shader, "project_u"),
            project_v: create_compute_pipeline(&device, &pipeline_layout, &shader, "project_v"),
            compute_vorticity: create_compute_pipeline(
                &device,
                &pipeline_layout,
                &shader,
                "compute_vorticity",
            ),
        };

        let mut backend = Self {
            config,
            state: SimulationState::new(grid),
            instance,
            adapter,
            device,
            queue,
            params,
            params_bind_group,
            params_buffer,
            fields_layout,
            command_buffer,
            dummy_buffer,
            buffers,
            staging,
            pipelines,
            readback_interval_steps: 1,
            steps_since_readback: 0,
        };

        backend.clear_gpu_only();
        Ok(backend)
    }

    pub fn state(&self) -> &SimulationState {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut SimulationState {
        &mut self.state
    }

    pub fn instance(&self) -> &wgpu::Instance {
        &self.instance
    }

    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn grid(&self) -> GridSize {
        self.state.grid
    }

    pub fn density_buffer(&self) -> &wgpu::Buffer {
        &self.buffers.density0
    }

    pub fn pressure_buffer(&self) -> &wgpu::Buffer {
        &self.buffers.pressure0
    }

    pub fn divergence_buffer(&self) -> &wgpu::Buffer {
        &self.buffers.divergence
    }

    pub fn u_buffer(&self) -> &wgpu::Buffer {
        &self.buffers.u0
    }

    pub fn v_buffer(&self) -> &wgpu::Buffer {
        &self.buffers.v0
    }

    pub fn obstacle_buffer(&self) -> &wgpu::Buffer {
        &self.buffers.obstacles
    }

    pub fn vorticity_buffer(&self) -> &wgpu::Buffer {
        &self.buffers.vorticity
    }

    pub fn set_readback_interval_steps(&mut self, interval: u32) {
        self.readback_interval_steps = interval.max(1);
        self.steps_since_readback = 0;
    }

    pub fn set_dt(&mut self, dt: f32) {
        self.config.dt = dt;
    }

    pub fn clear(&mut self) -> Result<(), GpuBackendError> {
        self.clear_gpu_only();
        self.state.clear();
        Ok(())
    }
    pub fn step(&mut self, commands: &[SimCommand]) -> Result<(), GpuBackendError> {
        let frame_start = Instant::now();

        if commands
            .iter()
            .any(|command| matches!(command, SimCommand::Clear))
        {
            self.clear_gpu_only();
            self.state.clear();
        }

        let command_records = self.encode_commands(commands);
        self.upload_obstacles();
        self.write_params(command_records.len() as u32, 0);
        if !command_records.is_empty() {
            self.queue.write_buffer(
                &self.command_buffer,
                0,
                bytemuck::cast_slice(&command_records),
            );
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fluid-sim-step-encoder"),
            });

        if !command_records.is_empty() {
            self.dispatch_pipeline(
                &mut encoder,
                &self.pipelines.apply_scalar_commands,
                self.slots([
                    Some(&self.buffers.density0),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ]),
                self.state.grid.nx as u32,
                self.state.grid.ny as u32,
            );
            self.dispatch_pipeline(
                &mut encoder,
                &self.pipelines.apply_u_commands,
                self.slots([Some(&self.buffers.u0), None, None, None, None, None, None]),
                (self.state.grid.nx + 1) as u32,
                self.state.grid.ny as u32,
            );
            self.dispatch_pipeline(
                &mut encoder,
                &self.pipelines.apply_v_commands,
                self.slots([Some(&self.buffers.v0), None, None, None, None, None, None]),
                self.state.grid.nx as u32,
                (self.state.grid.ny + 1) as u32,
            );
        }

        self.apply_scalar_boundary(&mut encoder, &self.buffers.density0);
        self.apply_velocity_boundary(&mut encoder, &self.buffers.u0, &self.buffers.v0);

        if self.config.viscosity > 0.0 {
            self.copy_buffer(
                &mut encoder,
                &self.buffers.u0,
                &self.buffers.u1,
                self.buffers.u_bytes,
            );
            self.copy_buffer(
                &mut encoder,
                &self.buffers.v0,
                &self.buffers.v1,
                self.buffers.v_bytes,
            );

            let mut u_old = &self.buffers.u1;
            let mut u_new = &self.buffers.u2;
            let mut v_old = &self.buffers.v1;
            let mut v_new = &self.buffers.v2;
            for _ in 0..self.config.solver_iterations {
                self.dispatch_pipeline(
                    &mut encoder,
                    &self.pipelines.diffuse_u_jacobi,
                    self.slots([
                        Some(&self.buffers.u0),
                        Some(u_old),
                        Some(u_new),
                        None,
                        None,
                        None,
                        None,
                    ]),
                    (self.state.grid.nx + 1) as u32,
                    self.state.grid.ny as u32,
                );
                self.apply_u_boundary(&mut encoder, u_new);
                std::mem::swap(&mut u_old, &mut u_new);

                self.dispatch_pipeline(
                    &mut encoder,
                    &self.pipelines.diffuse_v_jacobi,
                    self.slots([
                        Some(&self.buffers.v0),
                        Some(v_old),
                        Some(v_new),
                        None,
                        None,
                        None,
                        None,
                    ]),
                    self.state.grid.nx as u32,
                    (self.state.grid.ny + 1) as u32,
                );
                self.apply_v_boundary(&mut encoder, v_new);
                std::mem::swap(&mut v_old, &mut v_new);
            }

            self.copy_buffer(&mut encoder, u_old, &self.buffers.u0, self.buffers.u_bytes);
            self.copy_buffer(&mut encoder, v_old, &self.buffers.v0, self.buffers.v_bytes);
        }

        self.project(&mut encoder);

        self.dispatch_pipeline(
            &mut encoder,
            &self.pipelines.advect_u,
            self.slots([
                Some(&self.buffers.u0),
                Some(&self.buffers.v0),
                Some(&self.buffers.u1),
                None,
                None,
                None,
                None,
            ]),
            (self.state.grid.nx + 1) as u32,
            self.state.grid.ny as u32,
        );
        self.dispatch_pipeline(
            &mut encoder,
            &self.pipelines.advect_v,
            self.slots([
                Some(&self.buffers.u0),
                Some(&self.buffers.v0),
                Some(&self.buffers.v1),
                None,
                None,
                None,
                None,
            ]),
            self.state.grid.nx as u32,
            (self.state.grid.ny + 1) as u32,
        );
        self.apply_velocity_boundary(&mut encoder, &self.buffers.u1, &self.buffers.v1);
        self.copy_buffer(
            &mut encoder,
            &self.buffers.u1,
            &self.buffers.u0,
            self.buffers.u_bytes,
        );
        self.copy_buffer(
            &mut encoder,
            &self.buffers.v1,
            &self.buffers.v0,
            self.buffers.v_bytes,
        );

        self.project(&mut encoder);

        if self.config.diffusion > 0.0 {
            self.copy_buffer(
                &mut encoder,
                &self.buffers.density0,
                &self.buffers.density1,
                self.buffers.scalar_bytes,
            );

            let mut old = &self.buffers.density1;
            let mut new = &self.buffers.density2;
            for _ in 0..self.config.solver_iterations {
                self.dispatch_pipeline(
                    &mut encoder,
                    &self.pipelines.diffuse_scalar_jacobi,
                    self.slots([
                        Some(&self.buffers.density0),
                        Some(old),
                        Some(new),
                        None,
                        None,
                        None,
                        None,
                    ]),
                    self.state.grid.nx as u32,
                    self.state.grid.ny as u32,
                );
                self.apply_scalar_boundary(&mut encoder, new);
                std::mem::swap(&mut old, &mut new);
            }

            self.copy_buffer(
                &mut encoder,
                old,
                &self.buffers.density0,
                self.buffers.scalar_bytes,
            );
        }

        match self.config.scalar_advection {
            ScalarAdvectionKind::SemiLagrangian => {
                self.dispatch_pipeline(
                    &mut encoder,
                    &self.pipelines.advect_scalar,
                    self.slots([
                        Some(&self.buffers.density0),
                        Some(&self.buffers.u0),
                        Some(&self.buffers.v0),
                        Some(&self.buffers.density1),
                        None,
                        None,
                        None,
                    ]),
                    self.state.grid.nx as u32,
                    self.state.grid.ny as u32,
                );
                self.apply_scalar_boundary(&mut encoder, &self.buffers.density1);
                self.copy_buffer(
                    &mut encoder,
                    &self.buffers.density1,
                    &self.buffers.density0,
                    self.buffers.scalar_bytes,
                );
            }
            ScalarAdvectionKind::MacCormack => {
                self.dispatch_pipeline(
                    &mut encoder,
                    &self.pipelines.advect_scalar,
                    self.slots([
                        Some(&self.buffers.density0),
                        Some(&self.buffers.u0),
                        Some(&self.buffers.v0),
                        Some(&self.buffers.density1),
                        None,
                        None,
                        None,
                    ]),
                    self.state.grid.nx as u32,
                    self.state.grid.ny as u32,
                );
                self.apply_scalar_boundary(&mut encoder, &self.buffers.density1);

                let original_dt = self.config.dt;
                self.config.dt = -original_dt;
                self.write_params(command_records.len() as u32, 0);
                self.dispatch_pipeline(
                    &mut encoder,
                    &self.pipelines.advect_scalar,
                    self.slots([
                        Some(&self.buffers.density1),
                        Some(&self.buffers.u0),
                        Some(&self.buffers.v0),
                        Some(&self.buffers.density2),
                        None,
                        None,
                        None,
                    ]),
                    self.state.grid.nx as u32,
                    self.state.grid.ny as u32,
                );
                self.apply_scalar_boundary(&mut encoder, &self.buffers.density2);
                self.config.dt = original_dt;
                self.write_params(command_records.len() as u32, 0);

                self.dispatch_pipeline(
                    &mut encoder,
                    &self.pipelines.maccormack_correct,
                    self.slots([
                        Some(&self.buffers.density0),
                        Some(&self.buffers.density1),
                        Some(&self.buffers.density2),
                        Some(&self.buffers.u0),
                        Some(&self.buffers.v0),
                        Some(&self.buffers.divergence),
                        None,
                    ]),
                    self.state.grid.nx as u32,
                    self.state.grid.ny as u32,
                );
                self.apply_scalar_boundary(&mut encoder, &self.buffers.divergence);
                self.copy_buffer(
                    &mut encoder,
                    &self.buffers.divergence,
                    &self.buffers.density0,
                    self.buffers.scalar_bytes,
                );
            }
        }

        self.dispatch_pipeline(
            &mut encoder,
            &self.pipelines.compute_vorticity,
            self.slots([
                Some(&self.buffers.u0),
                Some(&self.buffers.v0),
                Some(&self.buffers.vorticity),
                None,
                None,
                None,
                None,
            ]),
            self.state.grid.nx as u32,
            self.state.grid.ny as u32,
        );

        self.queue.submit(Some(encoder.finish()));
        self.steps_since_readback += 1;
        let should_sync = self.steps_since_readback >= self.readback_interval_steps;
        if should_sync {
            self.sync_state_from_gpu()?;
            self.steps_since_readback = 0;
        }

        if should_sync {
            self.state.stats.max_divergence =
                max_abs_divergence(&self.state.divergence, &self.state.solids);
            self.state.stats.max_vorticity = max_abs_vorticity(&self.state.vorticity);
            self.state.stats.cfl = estimate_cfl(&self.state.velocity, self.config.dt);
        }
        self.state.stats.step_ms = frame_start.elapsed().as_secs_f32() * 1000.0;
        self.state.stats.pressure_iterations = self.config.solver_iterations * 2;

        Ok(())
    }

    fn project(&self, encoder: &mut wgpu::CommandEncoder) {
        self.apply_velocity_boundary(encoder, &self.buffers.u0, &self.buffers.v0);
        self.dispatch_pipeline(
            encoder,
            &self.pipelines.compute_divergence,
            self.slots([
                Some(&self.buffers.u0),
                Some(&self.buffers.v0),
                Some(&self.buffers.divergence),
                None,
                None,
                None,
                None,
            ]),
            self.state.grid.nx as u32,
            self.state.grid.ny as u32,
        );
        self.clear_scalar_buffer(encoder, &self.buffers.pressure0);
        for _ in 0..self.config.solver_iterations {
            self.queue.write_buffer(
                &self.params_buffer,
                0,
                bytemuck::bytes_of(&GpuSimParams::new(
                    &self.config,
                    self.state.grid,
                    self.params.command_count,
                    0,
                    first_fluid_cell(&self.state),
                )),
            );
            self.dispatch_pipeline(
                encoder,
                &self.pipelines.pressure_red_black,
                self.slots([
                    Some(&self.buffers.divergence),
                    Some(&self.buffers.pressure0),
                    Some(&self.buffers.pressure0),
                    None,
                    None,
                    None,
                    None,
                ]),
                self.state.grid.nx as u32,
                self.state.grid.ny as u32,
            );
            self.apply_scalar_boundary(encoder, &self.buffers.pressure0);

            self.queue.write_buffer(
                &self.params_buffer,
                0,
                bytemuck::bytes_of(&GpuSimParams::new(
                    &self.config,
                    self.state.grid,
                    self.params.command_count,
                    1,
                    first_fluid_cell(&self.state),
                )),
            );
            self.dispatch_pipeline(
                encoder,
                &self.pipelines.pressure_red_black,
                self.slots([
                    Some(&self.buffers.divergence),
                    Some(&self.buffers.pressure0),
                    Some(&self.buffers.pressure0),
                    None,
                    None,
                    None,
                    None,
                ]),
                self.state.grid.nx as u32,
                self.state.grid.ny as u32,
            );
            self.apply_scalar_boundary(encoder, &self.buffers.pressure0);
        }

        self.dispatch_pipeline(
            encoder,
            &self.pipelines.project_u,
            self.slots([
                Some(&self.buffers.pressure0),
                Some(&self.buffers.u0),
                Some(&self.buffers.u1),
                None,
                None,
                None,
                None,
            ]),
            (self.state.grid.nx + 1) as u32,
            self.state.grid.ny as u32,
        );
        self.dispatch_pipeline(
            encoder,
            &self.pipelines.project_v,
            self.slots([
                Some(&self.buffers.pressure0),
                Some(&self.buffers.v0),
                Some(&self.buffers.v1),
                None,
                None,
                None,
                None,
            ]),
            self.state.grid.nx as u32,
            (self.state.grid.ny + 1) as u32,
        );
        self.apply_velocity_boundary(encoder, &self.buffers.u1, &self.buffers.v1);
        self.copy_buffer(
            encoder,
            &self.buffers.u1,
            &self.buffers.u0,
            self.buffers.u_bytes,
        );
        self.copy_buffer(
            encoder,
            &self.buffers.v1,
            &self.buffers.v0,
            self.buffers.v_bytes,
        );
        self.dispatch_pipeline(
            encoder,
            &self.pipelines.compute_divergence,
            self.slots([
                Some(&self.buffers.u0),
                Some(&self.buffers.v0),
                Some(&self.buffers.divergence),
                None,
                None,
                None,
                None,
            ]),
            self.state.grid.nx as u32,
            self.state.grid.ny as u32,
        );
    }
    fn sync_state_from_gpu(&mut self) -> Result<(), GpuBackendError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fluid-sim-readback-encoder"),
            });
        self.copy_buffer(
            &mut encoder,
            &self.buffers.density0,
            &self.staging.density,
            self.buffers.scalar_bytes,
        );
        self.copy_buffer(
            &mut encoder,
            &self.buffers.pressure0,
            &self.staging.pressure,
            self.buffers.scalar_bytes,
        );
        self.copy_buffer(
            &mut encoder,
            &self.buffers.divergence,
            &self.staging.divergence,
            self.buffers.scalar_bytes,
        );
        self.copy_buffer(
            &mut encoder,
            &self.buffers.vorticity,
            &self.staging.vorticity,
            self.buffers.scalar_bytes,
        );
        self.copy_buffer(
            &mut encoder,
            &self.buffers.u0,
            &self.staging.u,
            self.buffers.u_bytes,
        );
        self.copy_buffer(
            &mut encoder,
            &self.buffers.v0,
            &self.staging.v,
            self.buffers.v_bytes,
        );
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        readback_f32_buffer(
            &self.device,
            &self.staging.density,
            self.state.density.as_mut_slice(),
            "density",
        )?;
        readback_f32_buffer(
            &self.device,
            &self.staging.pressure,
            self.state.pressure.as_mut_slice(),
            "pressure",
        )?;
        readback_f32_buffer(
            &self.device,
            &self.staging.divergence,
            self.state.divergence.as_mut_slice(),
            "divergence",
        )?;
        readback_f32_buffer(
            &self.device,
            &self.staging.vorticity,
            self.state.vorticity.as_mut_slice(),
            "vorticity",
        )?;
        readback_f32_buffer(
            &self.device,
            &self.staging.u,
            self.state.velocity.u.as_mut_slice(),
            "u-velocity",
        )?;
        readback_f32_buffer(
            &self.device,
            &self.staging.v,
            self.state.velocity.v.as_mut_slice(),
            "v-velocity",
        )?;
        Ok(())
    }

    fn write_params(&mut self, command_count: u32, pressure_phase: u32) {
        let anchor = first_fluid_cell(&self.state);
        self.params = GpuSimParams::new(
            &self.config,
            self.state.grid,
            command_count,
            pressure_phase,
            anchor,
        );
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
    }

    fn upload_obstacles(&self) {
        let shape = self.state.grid.scalar_shape();
        let mut obstacle_data = vec![[0.0_f32; 4]; shape.len()];

        for raw_j in 0..shape.height {
            for raw_i in 0..shape.width {
                let index = raw_i + raw_j * shape.width;
                let velocity = self.state.solids.raw_velocity(raw_i, raw_j);
                obstacle_data[index] = [
                    if self.state.solids.is_solid_raw(raw_i, raw_j) {
                        1.0
                    } else {
                        0.0
                    },
                    velocity.x,
                    velocity.y,
                    0.0,
                ];
            }
        }

        self.queue.write_buffer(
            &self.buffers.obstacles,
            0,
            bytemuck::cast_slice(&obstacle_data),
        );
    }

    fn encode_commands(&self, commands: &[SimCommand]) -> Vec<GpuCommandRecord> {
        let mut records = Vec::with_capacity(commands.len().min(MAX_GPU_COMMANDS));

        for command in commands {
            match command {
                SimCommand::Clear => records.clear(),
                SimCommand::AddDye {
                    position,
                    amount,
                    radius,
                } => records.push(GpuCommandRecord {
                    kind: CMD_DYE,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                    position: position.to_array(),
                    delta: [0.0, 0.0],
                    amount: *amount,
                    radius: *radius,
                    _pad3: [0.0, 0.0],
                }),
                SimCommand::AddForce {
                    position,
                    delta,
                    radius,
                } => records.push(GpuCommandRecord {
                    kind: CMD_FORCE,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                    position: position.to_array(),
                    delta: delta.to_array(),
                    amount: 0.0,
                    radius: *radius,
                    _pad3: [0.0, 0.0],
                }),
            }

            if records.len() == MAX_GPU_COMMANDS {
                warn!(
                    "GPU backend command buffer is full; truncating commands at {MAX_GPU_COMMANDS} entries."
                );
                break;
            }
        }

        records
    }

    fn apply_scalar_boundary(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::Buffer) {
        self.dispatch_pipeline(
            encoder,
            &self.pipelines.apply_scalar_boundary,
            self.slots([Some(target), None, None, None, None, None, None]),
            self.params.scalar_width,
            self.params.scalar_height,
        );
    }

    fn apply_u_boundary(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::Buffer) {
        self.dispatch_pipeline(
            encoder,
            &self.pipelines.apply_u_boundary,
            self.slots([Some(target), None, None, None, None, None, None]),
            self.params.u_width,
            self.params.u_height,
        );
    }

    fn apply_v_boundary(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::Buffer) {
        self.dispatch_pipeline(
            encoder,
            &self.pipelines.apply_v_boundary,
            self.slots([Some(target), None, None, None, None, None, None]),
            self.params.v_width,
            self.params.v_height,
        );
    }

    fn apply_velocity_boundary(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        u_target: &wgpu::Buffer,
        v_target: &wgpu::Buffer,
    ) {
        self.apply_u_boundary(encoder, u_target);
        self.apply_v_boundary(encoder, v_target);
    }

    fn clear_scalar_buffer(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::Buffer) {
        encoder.clear_buffer(target, 0, None);
    }

    fn clear_gpu_only(&mut self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fluid-sim-clear-encoder"),
            });

        for buffer in [
            &self.buffers.density0,
            &self.buffers.density1,
            &self.buffers.density2,
            &self.buffers.u0,
            &self.buffers.u1,
            &self.buffers.u2,
            &self.buffers.v0,
            &self.buffers.v1,
            &self.buffers.v2,
            &self.buffers.pressure0,
            &self.buffers.divergence,
            &self.buffers.vorticity,
            &self.buffers.obstacles,
        ] {
            encoder.clear_buffer(buffer, 0, None);
        }

        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    fn dispatch_pipeline(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        slots: [&wgpu::Buffer; 7],
        width: u32,
        height: u32,
    ) {
        let fields_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fluid-sim-fields-bind-group"),
            layout: &self.fields_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: slots[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: slots[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: slots[2].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: slots[3].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: slots[4].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: slots[5].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: slots[6].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.command_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fluid-sim-compute-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.params_bind_group, &[]);
        pass.set_bind_group(1, &fields_bind_group, &[]);
        pass.dispatch_workgroups(
            div_ceil(width, WORKGROUP_SIZE),
            div_ceil(height, WORKGROUP_SIZE),
            1,
        );
    }

    fn slots<'a>(&'a self, overrides: [Option<&'a wgpu::Buffer>; 7]) -> [&'a wgpu::Buffer; 7] {
        let dummy = &self.dummy_buffer;
        [
            overrides[0].unwrap_or(dummy),
            overrides[1].unwrap_or(dummy),
            overrides[2].unwrap_or(dummy),
            overrides[3].unwrap_or(dummy),
            overrides[4].unwrap_or(dummy),
            overrides[5].unwrap_or(dummy),
            overrides[6].unwrap_or(&self.buffers.obstacles),
        ]
    }

    fn copy_buffer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        destination: &wgpu::Buffer,
        size: u64,
    ) {
        encoder.copy_buffer_to_buffer(source, 0, destination, 0, size);
    }
}

impl GpuBuffers {
    fn new(device: &wgpu::Device, grid: GridSize) -> Self {
        let scalar_bytes = (grid.scalar_shape().len() * std::mem::size_of::<f32>()) as u64;
        let u_bytes = (grid.u_shape().len() * std::mem::size_of::<f32>()) as u64;
        let v_bytes = (grid.v_shape().len() * std::mem::size_of::<f32>()) as u64;
        let obstacle_bytes = (grid.scalar_shape().len() * std::mem::size_of::<[f32; 4]>()) as u64;

        Self {
            density0: create_storage_buffer(device, "density0", scalar_bytes),
            density1: create_storage_buffer(device, "density1", scalar_bytes),
            density2: create_storage_buffer(device, "density2", scalar_bytes),
            u0: create_storage_buffer(device, "u0", u_bytes),
            u1: create_storage_buffer(device, "u1", u_bytes),
            u2: create_storage_buffer(device, "u2", u_bytes),
            v0: create_storage_buffer(device, "v0", v_bytes),
            v1: create_storage_buffer(device, "v1", v_bytes),
            v2: create_storage_buffer(device, "v2", v_bytes),
            pressure0: create_storage_buffer(device, "pressure0", scalar_bytes),
            divergence: create_storage_buffer(device, "divergence", scalar_bytes),
            vorticity: create_storage_buffer(device, "vorticity", scalar_bytes),
            obstacles: create_storage_buffer(device, "obstacles", obstacle_bytes),
            scalar_bytes,
            u_bytes,
            v_bytes,
        }
    }
}

impl GpuStagingBuffers {
    fn new(device: &wgpu::Device, buffers: &GpuBuffers) -> Self {
        Self {
            density: create_staging_buffer(device, "density-readback", buffers.scalar_bytes),
            pressure: create_staging_buffer(device, "pressure-readback", buffers.scalar_bytes),
            divergence: create_staging_buffer(device, "divergence-readback", buffers.scalar_bytes),
            vorticity: create_staging_buffer(device, "vorticity-readback", buffers.scalar_bytes),
            u: create_staging_buffer(device, "u-readback", buffers.u_bytes),
            v: create_staging_buffer(device, "v-readback", buffers.v_bytes),
        }
    }
}
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: Some(layout),
        module: shader,
        entry_point,
    })
}

fn create_storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_staging_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn div_ceil(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

fn readback_f32_buffer(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    destination: &mut [f32],
    label: &str,
) -> Result<(), GpuBackendError> {
    let slice = staging.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);

    match receiver.recv() {
        Ok(Ok(())) => {
            let mapped = slice.get_mapped_range();
            for (value, bytes) in destination.iter_mut().zip(mapped.chunks_exact(4)) {
                *value = f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            }
            drop(mapped);
            staging.unmap();
            Ok(())
        }
        Ok(Err(err)) => Err(GpuBackendError::Map(format!("{label}: {err}"))),
        Err(err) => Err(GpuBackendError::Map(format!("{label}: {err}"))),
    }
}

fn estimate_cfl(velocity: &MacVelocity, dt: f32) -> f32 {
    let grid = velocity.u.grid();
    let max_u = velocity
        .u
        .as_slice()
        .iter()
        .fold(0.0_f32, |acc, value| acc.max(value.abs()));
    let max_v = velocity
        .v
        .as_slice()
        .iter()
        .fold(0.0_f32, |acc, value| acc.max(value.abs()));

    dt * max_u.max(max_v) / grid.cell_size
}

fn first_fluid_cell(state: &SimulationState) -> Option<(u32, u32)> {
    let grid = state.grid;
    for j in 0..grid.ny {
        for i in 0..grid.nx {
            if state.solids.is_fluid_cell(i, j) {
                return Some((i as u32, j as u32));
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use glam::Vec2;

    use super::{GpuBackendError, GpuFluidBackend};
    use crate::config::{SimulationBackendKind, SimulationConfig};
    use crate::sim::forces::SimCommand;
    use crate::sim::grid::GridSize;

    #[test]
    fn gpu_backend_constructs_and_steps_if_adapter_is_available() {
        let mut config = SimulationConfig::default();
        config.backend = SimulationBackendKind::Gpu;

        let grid = GridSize::new(32, 24, 1.0).expect("grid should be valid");
        match GpuFluidBackend::new(config, grid) {
            Ok(mut backend) => {
                backend
                    .step(&[
                        SimCommand::AddDye {
                            position: Vec2::new(16.0, 12.0),
                            amount: 1.0,
                            radius: 4.0,
                        },
                        SimCommand::AddForce {
                            position: Vec2::new(16.0, 12.0),
                            delta: Vec2::new(8.0, -4.0),
                            radius: 4.0,
                        },
                    ])
                    .expect("GPU backend should step");

                let max_density = backend
                    .state()
                    .density
                    .as_slice()
                    .iter()
                    .fold(0.0_f32, |acc, value| acc.max(*value));
                assert!(max_density >= 0.0);
            }
            Err(GpuBackendError::NoAdapter) => {}
            Err(err) => panic!("GPU backend smoke test failed: {err}"),
        }
    }
}
