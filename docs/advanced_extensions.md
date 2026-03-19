# Advanced Extensions Roadmap

This phase adds a solver-effect pipeline plus two concrete effects:

- buoyancy
- vorticity confinement

That pipeline is the intended insertion point for future force-like effects that
operate on `SimulationState` without contaminating rendering or input code.

## Next Solver Features

- Solid obstacles:
  Extend `SolidMask` from a passive field into active boundary-aware obstacle
  handling for advection, projection, and wall velocities.

- Moving obstacles:
  Add per-cell obstacle velocities and incorporate them into wall boundary terms
  during pressure projection.

- Pressure solver variants:
  Split the current pressure path behind a dedicated pressure-solver interface.
  Keep Gauss-Seidel as the baseline and add PCG or multigrid as alternatives.

- Lower-dissipation advection:
  Add MacCormack or BFECC as optional advection modes for dye and velocity.

- Multithreading:
  Parallelize read-mostly passes first:
  advection, divergence, vorticity, and visualization reductions.

- GPU backend:
  Preserve the CPU solver as the reference implementation and mirror the same
  state layout/stage ordering in a future compute backend.

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
