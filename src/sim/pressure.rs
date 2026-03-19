use crate::config::PressureSolverKind;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::boundary::{
    apply_scalar_boundary_with_solids, apply_velocity_boundary_with_solids, ScalarBoundary,
    VelocityBoundary,
};
use super::field::{MacVelocity, ScalarField, SolidMask};
use super::state::SimulationScratch;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectionStats {
    pub max_divergence: f32,
    pub iterations_used: usize,
}

pub trait PressureSolver: std::fmt::Debug + Send + Sync {
    fn kind(&self) -> PressureSolverKind;

    fn solve(
        &mut self,
        divergence: &ScalarField,
        solids: &SolidMask,
        iterations: usize,
        boundary: ScalarBoundary,
        scratch: &mut SimulationScratch,
        pressure: &mut ScalarField,
    ) -> usize;
}

#[derive(Debug, Clone)]
pub struct GaussSeidelPressureSolver {
    tolerance: f32,
}

#[derive(Debug, Clone)]
pub struct PcgPressureSolver {
    tolerance: f32,
}

impl Default for PcgPressureSolver {
    fn default() -> Self {
        Self { tolerance: 1.0e-3 }
    }
}

impl Default for GaussSeidelPressureSolver {
    fn default() -> Self {
        Self { tolerance: 1.0e-3 }
    }
}

#[derive(Debug, Clone)]
pub enum PressureSolverRuntime {
    GaussSeidel(GaussSeidelPressureSolver),
    Pcg(PcgPressureSolver),
}

impl PressureSolverRuntime {
    pub fn new(kind: PressureSolverKind, tolerance: f32) -> Self {
        match kind {
            PressureSolverKind::GaussSeidel => {
                Self::GaussSeidel(GaussSeidelPressureSolver { tolerance })
            }
            PressureSolverKind::Pcg => Self::Pcg(PcgPressureSolver { tolerance }),
        }
    }
}

impl PressureSolver for PressureSolverRuntime {
    fn kind(&self) -> PressureSolverKind {
        match self {
            Self::GaussSeidel(_) => PressureSolverKind::GaussSeidel,
            Self::Pcg(_) => PressureSolverKind::Pcg,
        }
    }

    fn solve(
        &mut self,
        divergence: &ScalarField,
        solids: &SolidMask,
        iterations: usize,
        boundary: ScalarBoundary,
        scratch: &mut SimulationScratch,
        pressure: &mut ScalarField,
    ) -> usize {
        match self {
            Self::GaussSeidel(solver) => {
                solver.solve(divergence, solids, iterations, boundary, scratch, pressure)
            }
            Self::Pcg(solver) => {
                solver.solve(divergence, solids, iterations, boundary, scratch, pressure)
            }
        }
    }
}

impl PressureSolver for GaussSeidelPressureSolver {
    fn kind(&self) -> PressureSolverKind {
        PressureSolverKind::GaussSeidel
    }

    fn solve(
        &mut self,
        divergence: &ScalarField,
        solids: &SolidMask,
        iterations: usize,
        boundary: ScalarBoundary,
        _scratch: &mut SimulationScratch,
        pressure: &mut ScalarField,
    ) -> usize {
        solve_pressure_gauss_seidel(
            divergence,
            solids,
            iterations,
            self.tolerance,
            boundary,
            pressure,
        )
    }
}

impl PressureSolver for PcgPressureSolver {
    fn kind(&self) -> PressureSolverKind {
        PressureSolverKind::Pcg
    }

    fn solve(
        &mut self,
        divergence: &ScalarField,
        solids: &SolidMask,
        iterations: usize,
        boundary: ScalarBoundary,
        scratch: &mut SimulationScratch,
        pressure: &mut ScalarField,
    ) -> usize {
        solve_pressure_pcg(
            divergence,
            solids,
            iterations,
            self.tolerance,
            boundary,
            scratch,
            pressure,
        )
    }
}

pub fn compute_divergence(
    velocity: &MacVelocity,
    solids: &SolidMask,
    divergence: &mut ScalarField,
) {
    assert_eq!(
        velocity.u.grid(),
        divergence.grid(),
        "velocity and divergence grids must match"
    );
    assert_eq!(
        velocity.u.grid(),
        solids.grid(),
        "velocity and solid mask grids must match"
    );

    let grid = divergence.grid();
    let inv_h = 1.0 / grid.cell_size;
    #[cfg(feature = "parallel")]
    let row_stride = grid.scalar_row_stride();

    #[cfg(feature = "parallel")]
    {
        divergence
            .as_mut_slice()
            .par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(raw_j, row)| {
                if raw_j == 0 || raw_j == grid.ny + 1 {
                    return;
                }

                let j = raw_j - 1;
                for i in 0..grid.nx {
                    row[i + 1] = if solids.is_solid_cell(i, j) {
                        0.0
                    } else {
                        let flux_x = velocity.u.get_face(i + 1, j) - velocity.u.get_face(i, j);
                        let flux_y = velocity.v.get_face(i, j + 1) - velocity.v.get_face(i, j);
                        (flux_x + flux_y) * inv_h
                    };
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    divergence.fill(0.0);

    #[cfg(not(feature = "parallel"))]
    for j in 0..grid.ny {
        for i in 0..grid.nx {
            if solids.is_solid_cell(i, j) {
                divergence.set_cell(i, j, 0.0);
                continue;
            }

            let flux_x = velocity.u.get_face(i + 1, j) - velocity.u.get_face(i, j);
            let flux_y = velocity.v.get_face(i, j + 1) - velocity.v.get_face(i, j);
            divergence.set_cell(i, j, (flux_x + flux_y) * inv_h);
        }
    }
}

pub fn subtract_pressure_gradient(
    velocity: &mut MacVelocity,
    pressure: &ScalarField,
    solids: &SolidMask,
    boundary: VelocityBoundary,
) {
    assert_eq!(
        velocity.u.grid(),
        pressure.grid(),
        "velocity and pressure grids must match"
    );
    assert_eq!(
        velocity.u.grid(),
        solids.grid(),
        "velocity and solid mask grids must match"
    );

    let grid = pressure.grid();
    let inv_h = 1.0 / grid.cell_size;

    for j in 0..grid.ny {
        for i in 1..grid.nx {
            if solids.is_fluid_cell(i - 1, j) && solids.is_fluid_cell(i, j) {
                let gradient = (pressure.get_cell(i, j) - pressure.get_cell(i - 1, j)) * inv_h;
                let corrected = velocity.u.get_face(i, j) - gradient;
                velocity.u.set_face(i, j, corrected);
            }
        }
    }

    for j in 1..grid.ny {
        for i in 0..grid.nx {
            if solids.is_fluid_cell(i, j - 1) && solids.is_fluid_cell(i, j) {
                let gradient = (pressure.get_cell(i, j) - pressure.get_cell(i, j - 1)) * inv_h;
                let corrected = velocity.v.get_face(i, j) - gradient;
                velocity.v.set_face(i, j, corrected);
            }
        }
    }

    apply_velocity_boundary_with_solids(velocity, boundary, solids);
}

pub fn max_abs_divergence(divergence: &ScalarField, solids: &SolidMask) -> f32 {
    let grid = divergence.grid();
    #[cfg(feature = "parallel")]
    {
        return (0..grid.ny)
            .into_par_iter()
            .map(|j| {
                let mut row_max = 0.0_f32;
                for i in 0..grid.nx {
                    if solids.is_fluid_cell(i, j) {
                        row_max = row_max.max(divergence.get_cell(i, j).abs());
                    }
                }
                row_max
            })
            .reduce(|| 0.0_f32, f32::max);
    }

    #[cfg(not(feature = "parallel"))]
    let mut max_value = 0.0_f32;

    #[cfg(not(feature = "parallel"))]
    for j in 0..grid.ny {
        for i in 0..grid.nx {
            if solids.is_fluid_cell(i, j) {
                max_value = max_value.max(divergence.get_cell(i, j).abs());
            }
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        max_value
    }
}

pub fn project_velocity(
    velocity: &mut MacVelocity,
    pressure: &mut ScalarField,
    divergence: &mut ScalarField,
    solids: &SolidMask,
    iterations: usize,
    velocity_boundary: VelocityBoundary,
    scalar_boundary: ScalarBoundary,
    scratch: &mut SimulationScratch,
    pressure_solver: &mut PressureSolverRuntime,
) -> ProjectionStats {
    apply_velocity_boundary_with_solids(velocity, velocity_boundary, solids);
    compute_divergence(velocity, solids, divergence);

    let iterations_used = pressure_solver.solve(
        divergence,
        solids,
        iterations,
        scalar_boundary,
        scratch,
        pressure,
    );

    subtract_pressure_gradient(velocity, pressure, solids, velocity_boundary);
    compute_divergence(velocity, solids, divergence);

    ProjectionStats {
        max_divergence: max_abs_divergence(divergence, solids),
        iterations_used,
    }
}

fn solve_pressure_gauss_seidel(
    divergence: &ScalarField,
    solids: &SolidMask,
    iterations: usize,
    tolerance: f32,
    boundary: ScalarBoundary,
    pressure: &mut ScalarField,
) -> usize {
    assert_eq!(
        divergence.grid(),
        pressure.grid(),
        "divergence and pressure grids must match"
    );
    assert_eq!(
        divergence.grid(),
        solids.grid(),
        "divergence and solid mask grids must match"
    );
    assert!(
        iterations > 0,
        "pressure iterations must be greater than zero"
    );

    let grid = pressure.grid();
    let h2 = grid.cell_size * grid.cell_size;
    let anchor = first_fluid_cell(solids);

    pressure.fill(0.0);

    for iteration in 0..iterations {
        let mut max_delta = 0.0_f32;

        for j in 0..grid.ny {
            for i in 0..grid.nx {
                if solids.is_solid_cell(i, j) || Some((i, j)) == anchor {
                    pressure.set_cell(i, j, 0.0);
                    continue;
                }

                let (diag, neighbor_sum) = fluid_neighbor_sum(pressure, solids, i, j);
                if diag == 0.0 {
                    pressure.set_cell(i, j, 0.0);
                    continue;
                }

                let previous = pressure.get_cell(i, j);
                let updated = (neighbor_sum - h2 * divergence.get_cell(i, j)) / diag;
                max_delta = max_delta.max((updated - previous).abs());
                pressure.set_cell(i, j, updated);
            }
        }

        apply_scalar_boundary_with_solids(pressure, boundary, solids);

        if max_delta <= tolerance {
            return iteration + 1;
        }
    }

    iterations
}

fn solve_pressure_pcg(
    divergence: &ScalarField,
    solids: &SolidMask,
    iterations: usize,
    tolerance: f32,
    boundary: ScalarBoundary,
    scratch: &mut SimulationScratch,
    pressure: &mut ScalarField,
) -> usize {
    assert_eq!(
        divergence.grid(),
        pressure.grid(),
        "divergence and pressure grids must match"
    );
    assert_eq!(
        divergence.grid(),
        solids.grid(),
        "divergence and solid mask grids must match"
    );
    assert!(
        iterations > 0,
        "pressure iterations must be greater than zero"
    );

    let grid = pressure.grid();
    let h2 = grid.cell_size * grid.cell_size;
    let anchor = first_fluid_cell(solids);
    let residual = &mut scratch.scalar1;
    let preconditioned = &mut scratch.scalar2;
    let direction = &mut scratch.scalar3;
    let operator = &mut scratch.scalar4;

    pressure.fill(0.0);
    residual.fill(0.0);
    preconditioned.fill(0.0);
    direction.fill(0.0);
    operator.fill(0.0);

    #[cfg(feature = "parallel")]
    {
        let row_stride = grid.scalar_row_stride();
        residual
            .as_mut_slice()
            .par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(raw_j, row)| {
                if raw_j == 0 || raw_j == grid.ny + 1 {
                    return;
                }

                let j = raw_j - 1;
                for i in 0..grid.nx {
                    if solids.is_fluid_cell(i, j) && Some((i, j)) != anchor {
                        row[i + 1] = -h2 * divergence.get_cell(i, j);
                    }
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                if solids.is_fluid_cell(i, j) && Some((i, j)) != anchor {
                    residual.set_cell(i, j, -h2 * divergence.get_cell(i, j));
                }
            }
        }
    }

    apply_diagonal_preconditioner(residual, solids, anchor, preconditioned);
    direction.copy_from(preconditioned);

    let mut rz_old = dot_fluid(residual, preconditioned, solids, anchor);
    if max_abs_divergence(residual, solids) <= tolerance {
        apply_scalar_boundary_with_solids(pressure, boundary, solids);
        return 0;
    }

    for iteration in 0..iterations {
        apply_poisson_operator(direction, solids, anchor, operator);
        let denom = dot_fluid(direction, operator, solids, anchor);
        if denom.abs() <= 1.0e-12 {
            break;
        }

        let alpha = rz_old / denom;
        axpy_fluid(pressure, direction, alpha, solids, anchor);
        axpy_fluid(residual, operator, -alpha, solids, anchor);

        if max_abs_divergence(residual, solids) <= tolerance {
            apply_scalar_boundary_with_solids(pressure, boundary, solids);
            return iteration + 1;
        }

        apply_diagonal_preconditioner(residual, solids, anchor, preconditioned);
        let rz_new = dot_fluid(residual, preconditioned, solids, anchor);
        if rz_old.abs() <= 1.0e-20 {
            break;
        }

        let beta = rz_new / rz_old;
        combine_search_direction(direction, preconditioned, beta, solids, anchor);
        rz_old = rz_new;
    }

    apply_scalar_boundary_with_solids(pressure, boundary, solids);
    iterations
}

fn apply_poisson_operator(
    input: &ScalarField,
    solids: &SolidMask,
    anchor: Option<(usize, usize)>,
    output: &mut ScalarField,
) {
    let grid = input.grid();
    #[cfg(feature = "parallel")]
    let row_stride = grid.scalar_row_stride();
    output.fill(0.0);

    #[cfg(feature = "parallel")]
    {
        output
            .as_mut_slice()
            .par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(raw_j, row)| {
                if raw_j == 0 || raw_j == grid.ny + 1 {
                    return;
                }

                let j = raw_j - 1;
                for i in 0..grid.nx {
                    row[i + 1] = if solids.is_solid_cell(i, j) || Some((i, j)) == anchor {
                        0.0
                    } else {
                        let (diag, neighbor_sum) = fluid_neighbor_sum(input, solids, i, j);
                        diag * input.get_cell(i, j) - neighbor_sum
                    };
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    for j in 0..grid.ny {
        for i in 0..grid.nx {
            if solids.is_solid_cell(i, j) || Some((i, j)) == anchor {
                output.set_cell(i, j, 0.0);
                continue;
            }

            let (diag, neighbor_sum) = fluid_neighbor_sum(input, solids, i, j);
            output.set_cell(i, j, diag * input.get_cell(i, j) - neighbor_sum);
        }
    }
}

fn apply_diagonal_preconditioner(
    residual: &ScalarField,
    solids: &SolidMask,
    anchor: Option<(usize, usize)>,
    output: &mut ScalarField,
) {
    let grid = residual.grid();
    #[cfg(feature = "parallel")]
    let row_stride = grid.scalar_row_stride();
    output.fill(0.0);

    #[cfg(feature = "parallel")]
    {
        output
            .as_mut_slice()
            .par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(raw_j, row)| {
                if raw_j == 0 || raw_j == grid.ny + 1 {
                    return;
                }

                let j = raw_j - 1;
                for i in 0..grid.nx {
                    row[i + 1] = if solids.is_solid_cell(i, j) || Some((i, j)) == anchor {
                        0.0
                    } else {
                        let diag = fluid_neighbor_count(solids, i, j);
                        if diag > 0.0 {
                            residual.get_cell(i, j) / diag
                        } else {
                            0.0
                        }
                    };
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    for j in 0..grid.ny {
        for i in 0..grid.nx {
            if solids.is_solid_cell(i, j) || Some((i, j)) == anchor {
                output.set_cell(i, j, 0.0);
                continue;
            }

            let diag = fluid_neighbor_count(solids, i, j);
            if diag > 0.0 {
                output.set_cell(i, j, residual.get_cell(i, j) / diag);
            }
        }
    }
}

fn dot_fluid(
    left: &ScalarField,
    right: &ScalarField,
    solids: &SolidMask,
    anchor: Option<(usize, usize)>,
) -> f32 {
    let grid = left.grid();
    #[cfg(feature = "parallel")]
    {
        return (0..grid.ny)
            .into_par_iter()
            .map(|j| {
                let mut sum = 0.0_f32;
                for i in 0..grid.nx {
                    if solids.is_fluid_cell(i, j) && Some((i, j)) != anchor {
                        sum += left.get_cell(i, j) * right.get_cell(i, j);
                    }
                }
                sum
            })
            .sum();
    }

    #[cfg(not(feature = "parallel"))]
    let mut sum = 0.0_f32;

    #[cfg(not(feature = "parallel"))]
    for j in 0..grid.ny {
        for i in 0..grid.nx {
            if solids.is_fluid_cell(i, j) && Some((i, j)) != anchor {
                sum += left.get_cell(i, j) * right.get_cell(i, j);
            }
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        sum
    }
}

fn axpy_fluid(
    destination: &mut ScalarField,
    source: &ScalarField,
    scale: f32,
    solids: &SolidMask,
    anchor: Option<(usize, usize)>,
) {
    let grid = destination.grid();
    #[cfg(feature = "parallel")]
    let row_stride = grid.scalar_row_stride();

    #[cfg(feature = "parallel")]
    {
        destination
            .as_mut_slice()
            .par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(raw_j, row)| {
                if raw_j == 0 || raw_j == grid.ny + 1 {
                    return;
                }

                let j = raw_j - 1;
                for i in 0..grid.nx {
                    if solids.is_fluid_cell(i, j) && Some((i, j)) != anchor {
                        row[i + 1] += scale * source.get_cell(i, j);
                    }
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    for j in 0..grid.ny {
        for i in 0..grid.nx {
            if solids.is_fluid_cell(i, j) && Some((i, j)) != anchor {
                destination.set_cell(
                    i,
                    j,
                    destination.get_cell(i, j) + scale * source.get_cell(i, j),
                );
            }
        }
    }
}

fn combine_search_direction(
    direction: &mut ScalarField,
    preconditioned: &ScalarField,
    beta: f32,
    solids: &SolidMask,
    anchor: Option<(usize, usize)>,
) {
    let grid = direction.grid();
    #[cfg(feature = "parallel")]
    let row_stride = grid.scalar_row_stride();

    #[cfg(feature = "parallel")]
    {
        direction
            .as_mut_slice()
            .par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(raw_j, row)| {
                if raw_j == 0 || raw_j == grid.ny + 1 {
                    return;
                }

                let j = raw_j - 1;
                for i in 0..grid.nx {
                    if solids.is_fluid_cell(i, j) && Some((i, j)) != anchor {
                        row[i + 1] = preconditioned.get_cell(i, j) + beta * row[i + 1];
                    }
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    for j in 0..grid.ny {
        for i in 0..grid.nx {
            if solids.is_fluid_cell(i, j) && Some((i, j)) != anchor {
                let updated = preconditioned.get_cell(i, j) + beta * direction.get_cell(i, j);
                direction.set_cell(i, j, updated);
            }
        }
    }
}

fn fluid_neighbor_sum(field: &ScalarField, solids: &SolidMask, i: usize, j: usize) -> (f32, f32) {
    let mut diag = 0.0_f32;
    let mut sum = 0.0_f32;

    if i > 0 && solids.is_fluid_cell(i - 1, j) {
        diag += 1.0;
        sum += field.get_cell(i - 1, j);
    }

    if i + 1 < field.grid().nx && solids.is_fluid_cell(i + 1, j) {
        diag += 1.0;
        sum += field.get_cell(i + 1, j);
    }

    if j > 0 && solids.is_fluid_cell(i, j - 1) {
        diag += 1.0;
        sum += field.get_cell(i, j - 1);
    }

    if j + 1 < field.grid().ny && solids.is_fluid_cell(i, j + 1) {
        diag += 1.0;
        sum += field.get_cell(i, j + 1);
    }

    (diag, sum)
}

fn fluid_neighbor_count(solids: &SolidMask, i: usize, j: usize) -> f32 {
    let grid = solids.grid();
    let mut diag = 0.0_f32;

    if i > 0 && solids.is_fluid_cell(i - 1, j) {
        diag += 1.0;
    }

    if i + 1 < grid.nx && solids.is_fluid_cell(i + 1, j) {
        diag += 1.0;
    }

    if j > 0 && solids.is_fluid_cell(i, j - 1) {
        diag += 1.0;
    }

    if j + 1 < grid.ny && solids.is_fluid_cell(i, j + 1) {
        diag += 1.0;
    }

    diag
}

fn first_fluid_cell(solids: &SolidMask) -> Option<(usize, usize)> {
    let grid = solids.grid();
    for j in 0..grid.ny {
        for i in 0..grid.nx {
            if solids.is_fluid_cell(i, j) {
                return Some((i, j));
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{
        compute_divergence, max_abs_divergence, project_velocity, PressureSolver,
        PressureSolverRuntime,
    };
    use crate::config::PressureSolverKind;
    use crate::sim::boundary::{ScalarBoundary, VelocityBoundary};
    use crate::sim::field::{MacVelocity, ScalarField, SolidMask};
    use crate::sim::grid::GridSize;
    use crate::sim::state::SimulationScratch;

    #[test]
    fn divergence_operator_matches_known_flux_difference() {
        let grid = GridSize::new(3, 2, 1.0).expect("grid should be valid");
        let mut velocity = MacVelocity::zeros(grid);
        let mut divergence = ScalarField::zeros(grid);
        let solids = SolidMask::empty(grid);

        velocity.u.set_face(1, 0, 1.0);
        velocity.u.set_face(2, 0, 3.0);
        velocity.v.set_face(1, 2, 3.0);

        compute_divergence(&velocity, &solids, &mut divergence);

        assert_eq!(divergence.get_cell(1, 0), 2.0);
        assert_eq!(divergence.get_cell(1, 1), 3.0);
    }

    #[test]
    fn gauss_seidel_projection_reduces_divergence() {
        let grid = GridSize::new(16, 16, 1.0).expect("grid should be valid");
        let mut velocity = MacVelocity::zeros(grid);
        let mut pressure = ScalarField::zeros(grid);
        let mut divergence = ScalarField::zeros(grid);
        let solids = SolidMask::empty(grid);
        let mut scratch = SimulationScratch::new(grid);
        let mut pressure_solver =
            PressureSolverRuntime::new(PressureSolverKind::GaussSeidel, 1.0e-4);

        for j in 0..grid.ny {
            for i in 0..=grid.nx {
                velocity
                    .u
                    .set_face(i, j, if i < grid.nx / 2 { 1.0 } else { -1.0 });
            }
        }

        compute_divergence(&velocity, &solids, &mut divergence);
        let before = max_abs_divergence(&divergence, &solids);

        let after = project_velocity(
            &mut velocity,
            &mut pressure,
            &mut divergence,
            &solids,
            160,
            VelocityBoundary::NoSlipBox,
            ScalarBoundary::ZeroGradientBox,
            &mut scratch,
            &mut pressure_solver,
        );

        assert!(after.max_divergence < before);
    }

    #[test]
    fn pcg_solver_reduces_divergence_by_more_than_an_order_of_magnitude() {
        let grid = GridSize::new(16, 16, 1.0).expect("grid should be valid");
        let mut velocity = MacVelocity::zeros(grid);
        let mut pressure = ScalarField::zeros(grid);
        let mut divergence = ScalarField::zeros(grid);
        let solids = SolidMask::empty(grid);
        let mut scratch = SimulationScratch::new(grid);
        let mut pressure_solver = PressureSolverRuntime::new(PressureSolverKind::Pcg, 1.0e-4);

        for j in 0..grid.ny {
            for i in 0..=grid.nx {
                velocity
                    .u
                    .set_face(i, j, if i < grid.nx / 2 { 1.0 } else { -1.0 });
            }
        }

        compute_divergence(&velocity, &solids, &mut divergence);
        let before = max_abs_divergence(&divergence, &solids);

        let stats = project_velocity(
            &mut velocity,
            &mut pressure,
            &mut divergence,
            &solids,
            80,
            VelocityBoundary::NoSlipBox,
            ScalarBoundary::ZeroGradientBox,
            &mut scratch,
            &mut pressure_solver,
        );

        assert_eq!(pressure_solver.kind(), PressureSolverKind::Pcg);
        assert!(stats.max_divergence < before * 0.1);
        assert!(stats.iterations_used <= 80);
    }

    #[test]
    fn solid_cells_do_not_report_divergence() {
        let grid = GridSize::new(6, 6, 1.0).expect("grid should be valid");
        let velocity = MacVelocity::zeros(grid);
        let mut divergence = ScalarField::zeros(grid);
        let mut solids = SolidMask::empty(grid);
        solids.set_solid_cell(2, 2, true);

        compute_divergence(&velocity, &solids, &mut divergence);

        assert_eq!(divergence.get_cell(2, 2), 0.0);
    }

    #[test]
    fn zero_velocity_stays_zero_after_projection() {
        let grid = GridSize::new(8, 6, 1.0).expect("grid should be valid");
        let mut velocity = MacVelocity::zeros(grid);
        let mut pressure = ScalarField::zeros(grid);
        let mut divergence = ScalarField::zeros(grid);
        let solids = SolidMask::empty(grid);
        let mut scratch = SimulationScratch::new(grid);
        let mut pressure_solver =
            PressureSolverRuntime::new(PressureSolverKind::GaussSeidel, 1.0e-4);

        let stats = project_velocity(
            &mut velocity,
            &mut pressure,
            &mut divergence,
            &solids,
            20,
            VelocityBoundary::NoSlipBox,
            ScalarBoundary::ZeroGradientBox,
            &mut scratch,
            &mut pressure_solver,
        );

        assert_eq!(stats.max_divergence, 0.0);
        for value in velocity.u.as_slice() {
            assert_eq!(*value, 0.0);
        }
        for value in velocity.v.as_slice() {
            assert_eq!(*value, 0.0);
        }
    }
}
