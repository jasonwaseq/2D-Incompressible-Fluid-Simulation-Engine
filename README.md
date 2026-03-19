# Fluid Sim 2D

A modular 2D incompressible fluid simulation engine in Rust.

This project is a CPU-first Eulerian fluid solver built around a MAC
(staggered) grid, a fixed-step simulation loop, and a lightweight interactive
viewer using `winit` + `pixels`. The codebase is organized to be readable,
testable, and extensible before chasing GPU or multithreaded complexity.

## Current Features

- 2D incompressible flow on a MAC grid
- Semi-Lagrangian advection
- Implicit diffusion with Gauss-Seidel relaxation
- Pressure projection for divergence-free velocity
- Interactive dye and momentum injection with the mouse
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
    grid.rs
    pressure.rs
    solver.rs
    state.rs
  render/
    colormap.rs
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

## Performance Notes

This repository is still a correctness-first CPU implementation.

Phase 9 added:

- flat contiguous storage
- buffer reuse across steps
- raw-slice inner loops for diffusion and pressure
- a benchmark target for tracking regressions

If you want to push performance further, the most natural next steps are:

- better pressure solvers
- parallel stencil passes
- lower-dissipation advection
- a GPU compute backend

## Roadmap

See [docs/advanced_extensions.md](docs/advanced_extensions.md) for the current
extension roadmap, including:

- solid and moving obstacles
- alternative pressure solvers
- lower-dissipation transport
- multithreading
- GPU compute
- 3D / PIC-FLIP / SPH exploration paths

## License

Licensed under either of:

- Apache License, Version 2.0, see [LICENSE-APACHE](LICENSE-APACHE)
- MIT license, see [LICENSE-MIT](LICENSE-MIT)

at your option.
