# Advanced Extensions Roadmap

The engine now includes:

- buoyancy
- vorticity confinement
- active solid and moving-obstacle boundary handling
- selectable CPU pressure solvers
- a GPU compute backend with direct GPU presentation for the main window path

That pipeline is the intended insertion point for future force-like effects that
operate on `SimulationState` without contaminating rendering or input code.

## Next Solver Features

- Stronger GPU pressure solvers:
  Keep the current red/black Gauss-Seidel GPU path as the baseline and explore
  multigrid or better preconditioned iterative methods on the GPU.

- Lower-dissipation velocity advection:
  Extend beyond scalar MacCormack and explore BFECC or MacCormack-style
  velocity transport with careful limiter and stability work.

- Multithreading:
  Parallelize read-mostly passes first:
  advection, divergence, vorticity, and visualization reductions.

- GPU diagnostics:
  Reduce or eliminate the remaining periodic GPU-to-CPU diagnostic readbacks by
  moving reductions and debug summaries onto the GPU.

## Longer-Term Research Paths

- 3D staggered-grid extension
- PIC/FLIP hybrid particle-grid coupling
- SPH comparison build for free-surface and splash-heavy cases

## Verification Guidance

When adding new effects or solvers:

- keep them stage-local
- add unit tests on synthetic fields before visual testing
- keep the CPU path deterministic and readable
- compare divergence and timing against the current baseline benchmarks
