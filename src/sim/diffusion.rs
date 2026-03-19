use super::boundary::{apply_scalar_boundary, apply_velocity_boundary, ScalarBoundary, VelocityBoundary};
use super::field::{MacVelocity, ScalarField};

pub fn diffuse_scalar(
    source: &ScalarField,
    diffusivity: f32,
    dt: f32,
    iterations: usize,
    boundary: ScalarBoundary,
    destination: &mut ScalarField,
) {
    assert_eq!(source.grid(), destination.grid(), "scalar field grids must match");
    assert!(iterations > 0, "diffusion iterations must be greater than zero");

    destination.copy_from(source);

    if diffusivity <= 0.0 {
        apply_scalar_boundary(destination, boundary);
        return;
    }

    let grid = source.grid();
    let row_stride = grid.scalar_row_stride();
    let source_data = source.as_slice();
    let alpha = diffusivity * dt / (grid.cell_size * grid.cell_size);
    let denom = 1.0 + 4.0 * alpha;

    for _ in 0..iterations {
        {
            let destination_data = destination.as_mut_slice();

            for j in 1..=grid.ny {
                let row_start = j * row_stride + 1;
                for i in 1..=grid.nx {
                    let index = row_start + i - 1;
                    let relaxed = source_data[index]
                        + alpha
                            * (destination_data[index - 1]
                                + destination_data[index + 1]
                                + destination_data[index - row_stride]
                                + destination_data[index + row_stride]);

                    destination_data[index] = relaxed / denom;
                }
            }
        }

        apply_scalar_boundary(destination, boundary);
    }
}

pub fn diffuse_velocity(
    source: &MacVelocity,
    viscosity: f32,
    dt: f32,
    iterations: usize,
    boundary: VelocityBoundary,
    destination: &mut MacVelocity,
) {
    assert_eq!(source.u.grid(), destination.u.grid(), "velocity grids must match");
    assert!(iterations > 0, "diffusion iterations must be greater than zero");

    destination.copy_from(source);

    if viscosity <= 0.0 {
        apply_velocity_boundary(destination, boundary);
        return;
    }

    let grid = source.u.grid();
    let u_row_stride = grid.u_row_stride();
    let v_row_stride = grid.v_row_stride();
    let source_u = source.u.as_slice();
    let source_v = source.v.as_slice();
    let alpha = viscosity * dt / (grid.cell_size * grid.cell_size);
    let denom = 1.0 + 4.0 * alpha;

    for _ in 0..iterations {
        {
            let destination_u = destination.u.as_mut_slice();

            for j in 1..=grid.ny {
                let row_start = j * u_row_stride + 1;
                for i in 1..=grid.nx + 1 {
                    let index = row_start + i - 1;
                    let relaxed = source_u[index]
                        + alpha
                            * (destination_u[index - 1]
                                + destination_u[index + 1]
                                + destination_u[index - u_row_stride]
                                + destination_u[index + u_row_stride]);

                    destination_u[index] = relaxed / denom;
                }
            }
        }

        {
            let destination_v = destination.v.as_mut_slice();

            for j in 1..=grid.ny + 1 {
                let row_start = j * v_row_stride + 1;
                for i in 1..=grid.nx {
                    let index = row_start + i - 1;
                    let relaxed = source_v[index]
                        + alpha
                            * (destination_v[index - 1]
                                + destination_v[index + 1]
                                + destination_v[index - v_row_stride]
                                + destination_v[index + v_row_stride]);

                    destination_v[index] = relaxed / denom;
                }
            }
        }

        apply_velocity_boundary(destination, boundary);
    }
}

#[cfg(test)]
mod tests {
    use super::{diffuse_scalar, diffuse_velocity};
    use crate::sim::boundary::{ScalarBoundary, VelocityBoundary};
    use crate::sim::field::{MacVelocity, ScalarField};
    use crate::sim::grid::GridSize;

    #[test]
    fn scalar_diffusion_reduces_peak_values() {
        let grid = GridSize::new(9, 9, 1.0).expect("grid should be valid");
        let mut source = ScalarField::zeros(grid);
        let mut destination = ScalarField::zeros(grid);
        source.set_cell(4, 4, 1.0);

        diffuse_scalar(
            &source,
            0.1,
            1.0,
            40,
            ScalarBoundary::ZeroGradientBox,
            &mut destination,
        );

        assert!(destination.get_cell(4, 4) < 1.0);
        assert!(destination.get_cell(4, 4) > 0.0);
    }

    #[test]
    fn uniform_scalar_field_stays_uniform() {
        let grid = GridSize::new(5, 4, 1.0).expect("grid should be valid");
        let source = ScalarField::new_filled(grid, 2.0);
        let mut destination = ScalarField::zeros(grid);

        diffuse_scalar(
            &source,
            0.25,
            1.0,
            30,
            ScalarBoundary::ZeroGradientBox,
            &mut destination,
        );

        for j in 0..grid.ny {
            for i in 0..grid.nx {
                assert!((destination.get_cell(i, j) - 2.0).abs() < 1.0e-5);
            }
        }
    }

    #[test]
    fn velocity_diffusion_preserves_wall_normal_zero_boundary() {
        let grid = GridSize::new(6, 4, 1.0).expect("grid should be valid");
        let mut source = MacVelocity::zeros(grid);
        let mut destination = MacVelocity::zeros(grid);

        for j in 0..grid.ny {
            for i in 0..=grid.nx {
                source.u.set_face(i, j, 1.0);
            }
        }

        diffuse_velocity(
            &source,
            0.05,
            1.0,
            30,
            VelocityBoundary::NoSlipBox,
            &mut destination,
        );

        for j in 0..grid.ny {
            assert_eq!(destination.u.get_face(0, j), 0.0);
            assert_eq!(destination.u.get_face(grid.nx, j), 0.0);
        }
    }
}
