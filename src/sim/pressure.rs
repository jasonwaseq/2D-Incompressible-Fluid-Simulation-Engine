use super::boundary::{apply_scalar_boundary, apply_velocity_boundary, ScalarBoundary, VelocityBoundary};
use super::field::{MacVelocity, ScalarField};

pub fn compute_divergence(velocity: &MacVelocity, divergence: &mut ScalarField) {
    assert_eq!(velocity.u.grid(), divergence.grid(), "velocity and divergence grids must match");

    let grid = divergence.grid();
    let scalar_stride = grid.scalar_row_stride();
    let u_stride = grid.u_row_stride();
    let v_stride = grid.v_row_stride();
    let u_data = velocity.u.as_slice();
    let v_data = velocity.v.as_slice();
    let divergence_data = divergence.as_mut_slice();
    let inv_h = 1.0 / grid.cell_size;
    divergence_data.fill(0.0);

    for j in 0..grid.ny {
        let scalar_row = (j + 1) * scalar_stride;
        let u_row = (j + 1) * u_stride;
        let v_row = (j + 1) * v_stride;
        for i in 0..grid.nx {
            let scalar_index = scalar_row + i + 1;
            let u_left = u_data[u_row + i + 1];
            let u_right = u_data[u_row + i + 2];
            let v_bottom = v_data[v_row + i + 1];
            let v_top = v_data[v_row + v_stride + i + 1];
            divergence_data[scalar_index] = (u_right - u_left + v_top - v_bottom) * inv_h;
        }
    }
}

pub fn solve_pressure(
    divergence: &ScalarField,
    iterations: usize,
    boundary: ScalarBoundary,
    pressure: &mut ScalarField,
) {
    assert_eq!(divergence.grid(), pressure.grid(), "divergence and pressure grids must match");
    assert!(iterations > 0, "pressure iterations must be greater than zero");

    let grid = pressure.grid();
    let row_stride = grid.scalar_row_stride();
    let divergence_data = divergence.as_slice();
    pressure.as_mut_slice().fill(0.0);
    let h2 = grid.cell_size * grid.cell_size;

    for _ in 0..iterations {
        {
            let pressure_data = pressure.as_mut_slice();

            for j in 1..=grid.ny {
                let row_start = j * row_stride + 1;
                for i in 1..=grid.nx {
                    let index = row_start + i - 1;
                    let relaxed = pressure_data[index - 1]
                        + pressure_data[index + 1]
                        + pressure_data[index - row_stride]
                        + pressure_data[index + row_stride]
                        - h2 * divergence_data[index];

                    pressure_data[index] = 0.25 * relaxed;
                }
            }
        }

        apply_scalar_boundary(pressure, boundary);
    }
}

pub fn subtract_pressure_gradient(
    velocity: &mut MacVelocity,
    pressure: &ScalarField,
    boundary: VelocityBoundary,
) {
    assert_eq!(velocity.u.grid(), pressure.grid(), "velocity and pressure grids must match");

    let grid = pressure.grid();
    let pressure_data = pressure.as_slice();
    let u_stride = grid.u_row_stride();
    let v_stride = grid.v_row_stride();
    let p_stride = grid.scalar_row_stride();
    let u_data = velocity.u.as_mut_slice();
    let v_data = velocity.v.as_mut_slice();
    let inv_h = 1.0 / grid.cell_size;

    for j in 0..grid.ny {
        let u_row = (j + 1) * u_stride;
        let p_row = (j + 1) * p_stride;
        for i in 1..grid.nx {
            let pressure_index = p_row + i;
            let u_index = u_row + i + 1;
            let gradient = (pressure_data[pressure_index + 1] - pressure_data[pressure_index]) * inv_h;
            u_data[u_index] -= gradient;
        }
    }

    for j in 1..grid.ny {
        let v_row = (j + 1) * v_stride;
        let p_row = (j + 1) * p_stride;
        let p_prev_row = p_row - p_stride;
        for i in 0..grid.nx {
            let pressure_index = p_row + i + 1;
            let v_index = v_row + i + 1;
            let gradient = (pressure_data[pressure_index] - pressure_data[p_prev_row + i + 1]) * inv_h;
            v_data[v_index] -= gradient;
        }
    }

    apply_velocity_boundary(velocity, boundary);
}

pub fn max_abs_divergence(divergence: &ScalarField) -> f32 {
    let grid = divergence.grid();
    let row_stride = grid.scalar_row_stride();
    let data = divergence.as_slice();
    let mut max_value = 0.0_f32;

    for j in 0..grid.ny {
        let row_start = (j + 1) * row_stride + 1;
        for value in &data[row_start..row_start + grid.nx] {
            max_value = max_value.max(value.abs());
        }
    }

    max_value
}

pub fn project_velocity(
    velocity: &mut MacVelocity,
    pressure: &mut ScalarField,
    divergence: &mut ScalarField,
    iterations: usize,
    velocity_boundary: VelocityBoundary,
    scalar_boundary: ScalarBoundary,
) -> f32 {
    apply_velocity_boundary(velocity, velocity_boundary);
    compute_divergence(velocity, divergence);
    solve_pressure(divergence, iterations, scalar_boundary, pressure);
    subtract_pressure_gradient(velocity, pressure, velocity_boundary);
    compute_divergence(velocity, divergence);
    max_abs_divergence(divergence)
}

#[cfg(test)]
mod tests {
    use super::{compute_divergence, max_abs_divergence, project_velocity};
    use crate::sim::boundary::{ScalarBoundary, VelocityBoundary};
    use crate::sim::field::{MacVelocity, ScalarField};
    use crate::sim::grid::GridSize;

    #[test]
    fn divergence_operator_matches_known_flux_difference() {
        let grid = GridSize::new(3, 2, 1.0).expect("grid should be valid");
        let mut velocity = MacVelocity::zeros(grid);
        let mut divergence = ScalarField::zeros(grid);

        velocity.u.set_face(1, 0, 1.0);
        velocity.u.set_face(2, 0, 3.0);
        velocity.v.set_face(1, 2, 3.0);

        compute_divergence(&velocity, &mut divergence);

        assert_eq!(divergence.get_cell(1, 0), 2.0);
        assert_eq!(divergence.get_cell(1, 1), 3.0);
    }

    #[test]
    fn projection_reduces_divergence_by_more_than_an_order_of_magnitude() {
        let grid = GridSize::new(16, 16, 1.0).expect("grid should be valid");
        let mut velocity = MacVelocity::zeros(grid);
        let mut pressure = ScalarField::zeros(grid);
        let mut divergence = ScalarField::zeros(grid);

        for j in 0..grid.ny {
            for i in 0..=grid.nx {
                velocity.u.set_face(i, j, if i < grid.nx / 2 { 1.0 } else { -1.0 });
            }
        }

        compute_divergence(&velocity, &mut divergence);
        let before = max_abs_divergence(&divergence);

        let after = project_velocity(
            &mut velocity,
            &mut pressure,
            &mut divergence,
            80,
            VelocityBoundary::NoSlipBox,
            ScalarBoundary::ZeroGradientBox,
        );

        assert!(after < before * 0.1, "expected after={after} to be much smaller than before={before}");
    }

    #[test]
    fn zero_velocity_stays_zero_after_projection() {
        let grid = GridSize::new(8, 6, 1.0).expect("grid should be valid");
        let mut velocity = MacVelocity::zeros(grid);
        let mut pressure = ScalarField::zeros(grid);
        let mut divergence = ScalarField::zeros(grid);

        let after = project_velocity(
            &mut velocity,
            &mut pressure,
            &mut divergence,
            20,
            VelocityBoundary::NoSlipBox,
            ScalarBoundary::ZeroGradientBox,
        );

        assert_eq!(after, 0.0);
        for value in velocity.u.as_slice() {
            assert_eq!(*value, 0.0);
        }
        for value in velocity.v.as_slice() {
            assert_eq!(*value, 0.0);
        }
    }
}
