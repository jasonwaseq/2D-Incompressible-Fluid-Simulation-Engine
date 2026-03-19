# Fluid Sim 2D

A modular 2D incompressible fluid simulation engine in Rust.

This project is a CPU-first Eulerian fluid solver built around a MAC
(staggered) grid, a fixed-step simulation loop, and a lightweight interactive
viewer using `winit` + `pixels`. It now also includes a separate `wgpu`
compute backend that keeps the core simulation fields resident on the GPU and
can render directly from GPU buffers without a per-frame CPU readback in the
main window path.

## Current Features

- 2D incompressible flow on a MAC grid
- Semi-Lagrangian advection
- Implicit diffusion with Gauss-Seidel relaxation
- Pressure projection for divergence-free velocity
- Selectable pressure solvers:
  - Gauss-Seidel
  - PCG
- Configurable pressure tolerance so iterative solves can stop early
- Interactive dye and momentum injection with the mouse
- Active obstacle handling with moving-wall boundary velocities
- Lower-dissipation scalar transport with MacCormack advection
- Default-on parallel CPU stencil passes with `rayon`
- Selectable simulation backends:
  - CPU solver
  - GPU compute solver with `wgpu`
- Real-time visualization modes:
  - density
  - velocity magnitude
  - pressure
  - divergence
  - vorticity
- Optional advanced effects:
  - buoyancy
  - vorticity confinement
- Runtime diagnostics:
  - CFL estimate
  - max divergence
  - max vorticity
  - solver timing
- Unit tests and a simple benchmark harness

## Why A MAC Grid

The solver uses a staggered-grid layout:

- scalar fields at cell centers
- `u` velocity on vertical faces
- `v` velocity on horizontal faces

This avoids a lot of the pressure-velocity decoupling problems that show up in
naive collocated layouts and makes incompressibility enforcement much more
robust for a first serious implementation.

## Project Layout

```text
src/
  main.rs
  lib.rs
  app.rs
  config.rs
  sim/
    advection.rs
    boundary.rs
    diffusion.rs
    effects.rs
    field.rs
    forces.rs
    gpu.rs
    gpu.wgsl
    grid.rs
    pressure.rs
    runtime.rs
    solver.rs
    state.rs
  render/
    colormap.rs
    gpu_view.rs
    gpu_view.wgsl
    mod.rs
    view.rs
  input/
    mod.rs
    mouse.rs
  math/
    interp.rs
    mod.rs
    operators.rs
  util/
    mod.rs
    timer.rs
benches/
  solver_bench.rs
docs/
  advanced_extensions.md
```

## Build And Run

Requirements:

- Rust stable
- a desktop environment capable of opening a `winit` window

Run the interactive app:

```powershell
cargo run --release
```

Run the GPU compute backend:

```powershell
cargo run --release -- --backend gpu
```

Run with the stronger pressure solver and lower-dissipation dye transport:

```powershell
cargo run --release -- --pressure-solver pcg --scalar-advection mac-cormack
```

Tune pressure convergence explicitly:

```powershell
cargo run --release -- --pressure-solver gauss-seidel --pressure-tolerance 5e-4
```

Run with advanced effects enabled:

```powershell
cargo run --release -- --buoyancy 1.5 --vorticity-confinement 2.0
```

Start directly in vorticity view:

```powershell
cargo run --release -- --visualization-mode vorticity --vorticity-confinement 2.0
```

Show the CLI options:

```powershell
cargo run -- --help
```

## Controls

- `Left Mouse Drag`: inject dye and momentum
- `Space`: pause / resume
- `N`: single-step one frame
- `R`: clear the simulation
- `1`: density view
- `2`: velocity magnitude view
- `3`: pressure view
- `4`: divergence view
- `5`: vorticity view
- `V`: toggle velocity vectors
- `[` / `]`: decrease / increase force scale
- `-` / `=`: decrease / increase timestep
- `Esc`: quit

## Testing And Verification

Run the unit tests:

```powershell
cargo test
```

Run tests with runtime sanity checks forced on:

```powershell
cargo test --features sanity-checks
```

Run the benchmark harness:

```powershell
cargo bench --bench solver_bench
```

The test suite covers:

- indexing and ghost-cell layout
- bilinear interpolation
- advection invariants
- diffusion behavior
- divergence computation
- pressure projection
- buoyancy and vorticity effect behavior

## Numerical Model

Each simulation step follows the classic graphics-oriented incompressible flow
pipeline:

1. apply user forces and configured solver effects
2. diffuse velocity
3. project velocity
4. advect velocity
5. project velocity again
6. diffuse density
7. advect density
8. refresh diagnostics and debug fields

This is a stable, practical baseline intended for interactive experimentation.
It is deliberately more focused on robustness and clarity than on minimal
numerical dissipation.

The current implementation uses:

- Gauss-Seidel as the default interactive pressure solver
- PCG as a stronger selectable alternative when you want lower divergence and
  can afford more CPU time
- configurable pressure tolerances so iterative solves can exit early once
  they have converged enough for the chosen mode
- MacCormack for scalar transport when explicitly enabled
- semi-Lagrangian transport for velocity, which remains the more stable choice
  for the current engine
- default-on multicore CPU parallelism for advection, divergence, vorticity,
  and PCG stencil-style passes
- an optional `wgpu` compute backend for GPU-resident density, velocity,
  pressure, divergence, vorticity, and obstacle buffers, with compute passes
  for:
  - command injection
  - advection
  - diffusion
  - divergence
  - red/black pressure iteration
  - projection
  - vorticity
  - direct GPU presentation for the main render path

## Performance Notes

This repository is still correctness-first, but it now has a real GPU path.

The current CPU path includes:

- flat contiguous storage
- buffer reuse across steps
- raw-slice inner loops for diffusion and pressure
- selectable pressure solvers behind a clean interface
- early-exit pressure convergence control
- parallel stencil-style passes behind the default `parallel` feature
- lower-dissipation MacCormack scalar transport
- a benchmark target for tracking regressions

The GPU path now moves both the expensive solver stages and the main window
presentation path onto the GPU. Today that means:

- simulation fields are stored on the GPU
- the main interactive renderer reads GPU-resident simulation buffers directly
- CPU readback is no longer required every frame and is now used only for
  periodic diagnostics/state mirroring
- obstacle-aware boundaries are handled on the GPU, including moving-wall
  obstacle velocities
- box boundaries are supported
- the core incompressible pipeline runs as compute passes

Current GPU-path limitations:

- buoyancy and vorticity-confinement settings are currently CPU-only features
- the `--pressure-solver` flag is still CPU-focused; the GPU path currently
  uses its own red/black Gauss-Seidel pressure iteration
- velocity-vector overlays are still CPU-renderer functionality; the GPU path
  focuses on the scalar debug views and velocity-magnitude view

There is still headroom. The most natural next steps from here are:

- multigrid or a stronger GPU preconditioner for the pressure solve
- lower-dissipation velocity advection once we are comfortable with the
  stability tradeoff
- removing or reducing the remaining periodic diagnostic readbacks

## Roadmap

See [docs/advanced_extensions.md](docs/advanced_extensions.md) for the current
extension roadmap, including:

- solid and moving obstacles
- alternative pressure solvers
- lower-dissipation transport
- multithreading
- GPU compute
- 3D / PIC-FLIP / SPH exploration paths
