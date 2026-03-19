use glam::Vec2;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::math::interp::bilerp;

use super::field::{MacVelocity, ScalarField};

#[inline(always)]
pub fn sample_scalar(field: &ScalarField, position: Vec2) -> f32 {
    let grid = field.grid();
    let position = grid.clamp_scalar_position(position);
    let inv_h = 1.0 / grid.cell_size;

    let gx = position.x * inv_h - 0.5;
    let gy = position.y * inv_h - 0.5;

    let i0 = gx.floor() as usize;
    let j0 = gy.floor() as usize;
    let i1 = (i0 + 1).min(grid.nx - 1);
    let j1 = (j0 + 1).min(grid.ny - 1);

    let tx = gx - i0 as f32;
    let ty = gy - j0 as f32;

    bilerp(
        field.get_cell(i0, j0),
        field.get_cell(i1, j0),
        field.get_cell(i0, j1),
        field.get_cell(i1, j1),
        tx,
        ty,
    )
}

fn sample_scalar_bounds(field: &ScalarField, position: Vec2) -> (f32, f32) {
    let grid = field.grid();
    let position = grid.clamp_scalar_position(position);
    let inv_h = 1.0 / grid.cell_size;

    let gx = position.x * inv_h - 0.5;
    let gy = position.y * inv_h - 0.5;

    let i0 = gx.floor() as usize;
    let j0 = gy.floor() as usize;
    let i1 = (i0 + 1).min(grid.nx - 1);
    let j1 = (j0 + 1).min(grid.ny - 1);

    let samples = [
        field.get_cell(i0, j0),
        field.get_cell(i1, j0),
        field.get_cell(i0, j1),
        field.get_cell(i1, j1),
    ];

    let mut min_value = samples[0];
    let mut max_value = samples[0];
    for value in samples.into_iter().skip(1) {
        min_value = min_value.min(value);
        max_value = max_value.max(value);
    }

    (min_value, max_value)
}

#[inline(always)]
pub fn sample_u(velocity: &MacVelocity, position: Vec2) -> f32 {
    let grid = velocity.u.grid();
    let position = grid.clamp_u_position(position);
    let inv_h = 1.0 / grid.cell_size;

    let gx = position.x * inv_h;
    let gy = position.y * inv_h - 0.5;

    let i0 = gx.floor() as usize;
    let j0 = gy.floor() as usize;
    let i1 = (i0 + 1).min(grid.nx);
    let j1 = (j0 + 1).min(grid.ny - 1);

    let tx = gx - i0 as f32;
    let ty = gy - j0 as f32;

    bilerp(
        velocity.u.get_face(i0, j0),
        velocity.u.get_face(i1, j0),
        velocity.u.get_face(i0, j1),
        velocity.u.get_face(i1, j1),
        tx,
        ty,
    )
}

#[inline(always)]
pub fn sample_v(velocity: &MacVelocity, position: Vec2) -> f32 {
    let grid = velocity.v.grid();
    let position = grid.clamp_v_position(position);
    let inv_h = 1.0 / grid.cell_size;

    let gx = position.x * inv_h - 0.5;
    let gy = position.y * inv_h;

    let i0 = gx.floor() as usize;
    let j0 = gy.floor() as usize;
    let i1 = (i0 + 1).min(grid.nx - 1);
    let j1 = (j0 + 1).min(grid.ny);

    let tx = gx - i0 as f32;
    let ty = gy - j0 as f32;

    bilerp(
        velocity.v.get_face(i0, j0),
        velocity.v.get_face(i1, j0),
        velocity.v.get_face(i0, j1),
        velocity.v.get_face(i1, j1),
        tx,
        ty,
    )
}

#[inline(always)]
pub fn sample_velocity(velocity: &MacVelocity, position: Vec2) -> Vec2 {
    Vec2::new(sample_u(velocity, position), sample_v(velocity, position))
}

pub fn advect_scalar(
    source: &ScalarField,
    velocity: &MacVelocity,
    dt: f32,
    destination: &mut ScalarField,
) {
    assert_eq!(
        source.grid(),
        velocity.u.grid(),
        "scalar and velocity grids must match"
    );
    assert_eq!(
        source.grid(),
        destination.grid(),
        "source and destination scalar grids must match"
    );

    let grid = source.grid();
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
                    let position = grid.cell_center(i, j);
                    let flow = sample_velocity(velocity, position);
                    let previous_position = grid.clamp_scalar_position(position - dt * flow);
                    row[i + 1] = sample_scalar(source, previous_position);
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let position = grid.cell_center(i, j);
            let flow = sample_velocity(velocity, position);
            let previous_position = grid.clamp_scalar_position(position - dt * flow);
            let value = sample_scalar(source, previous_position);
            destination.set_cell(i, j, value);
        }
    }
}

pub fn advect_scalar_maccormack(
    source: &ScalarField,
    velocity: &MacVelocity,
    dt: f32,
    forward: &mut ScalarField,
    reverse: &mut ScalarField,
    destination: &mut ScalarField,
) {
    assert_eq!(
        source.grid(),
        velocity.u.grid(),
        "scalar and velocity grids must match"
    );
    assert_eq!(
        source.grid(),
        forward.grid(),
        "forward scalar scratch grid must match source"
    );
    assert_eq!(
        source.grid(),
        reverse.grid(),
        "reverse scalar scratch grid must match source"
    );
    assert_eq!(
        source.grid(),
        destination.grid(),
        "destination scalar grid must match source"
    );

    advect_scalar(source, velocity, dt, forward);
    advect_scalar(forward, velocity, -dt, reverse);

    let grid = source.grid();
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
                    let position = grid.cell_center(i, j);
                    let flow = sample_velocity(velocity, position);
                    let previous_position = grid.clamp_scalar_position(position - dt * flow);
                    let corrected = forward.get_cell(i, j)
                        + 0.5 * (source.get_cell(i, j) - reverse.get_cell(i, j));
                    let (min_value, max_value) = sample_scalar_bounds(source, previous_position);
                    row[i + 1] = corrected.clamp(min_value, max_value);
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let position = grid.cell_center(i, j);
            let flow = sample_velocity(velocity, position);
            let previous_position = grid.clamp_scalar_position(position - dt * flow);
            let corrected =
                forward.get_cell(i, j) + 0.5 * (source.get_cell(i, j) - reverse.get_cell(i, j));
            let (min_value, max_value) = sample_scalar_bounds(source, previous_position);
            destination.set_cell(i, j, corrected.clamp(min_value, max_value));
        }
    }
}

pub fn advect_velocity(source: &MacVelocity, dt: f32, destination: &mut MacVelocity) {
    assert_eq!(
        source.u.grid(),
        destination.u.grid(),
        "velocity grids must match"
    );

    let grid = source.u.grid();
    #[cfg(feature = "parallel")]
    let u_row_stride = grid.u_row_stride();
    #[cfg(feature = "parallel")]
    let v_row_stride = grid.v_row_stride();

    #[cfg(feature = "parallel")]
    {
        destination
            .u
            .as_mut_slice()
            .par_chunks_mut(u_row_stride)
            .enumerate()
            .for_each(|(raw_j, row)| {
                if raw_j == 0 || raw_j == grid.ny + 1 {
                    return;
                }

                let j = raw_j - 1;
                for i in 0..=grid.nx {
                    let position = grid.u_face_position(i, j);
                    let flow = sample_velocity(source, position);
                    let previous_position = grid.clamp_u_position(position - dt * flow);
                    row[i + 1] = sample_u(source, previous_position);
                }
            });

        destination
            .v
            .as_mut_slice()
            .par_chunks_mut(v_row_stride)
            .enumerate()
            .for_each(|(raw_j, row)| {
                if raw_j == 0 || raw_j == grid.ny + 2 {
                    return;
                }

                let j = raw_j - 1;
                for i in 0..grid.nx {
                    let position = grid.v_face_position(i, j);
                    let flow = sample_velocity(source, position);
                    let previous_position = grid.clamp_v_position(position - dt * flow);
                    row[i + 1] = sample_v(source, previous_position);
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    for j in 0..grid.ny {
        for i in 0..=grid.nx {
            let position = grid.u_face_position(i, j);
            let flow = sample_velocity(source, position);
            let previous_position = grid.clamp_u_position(position - dt * flow);
            let value = sample_u(source, previous_position);
            destination.u.set_face(i, j, value);
        }
    }

    #[cfg(not(feature = "parallel"))]
    for j in 0..=grid.ny {
        for i in 0..grid.nx {
            let position = grid.v_face_position(i, j);
            let flow = sample_velocity(source, position);
            let previous_position = grid.clamp_v_position(position - dt * flow);
            let value = sample_v(source, previous_position);
            destination.v.set_face(i, j, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use glam::Vec2;

    use super::{
        advect_scalar, advect_scalar_maccormack, advect_velocity, sample_scalar, sample_velocity,
    };
    use crate::math::operators::all_finite;
    use crate::sim::field::{MacVelocity, ScalarField};
    use crate::sim::grid::GridSize;

    fn constant_velocity(grid: GridSize, u_value: f32, v_value: f32) -> MacVelocity {
        let mut velocity = MacVelocity::zeros(grid);

        for j in 0..grid.ny {
            for i in 0..=grid.nx {
                velocity.u.set_face(i, j, u_value);
            }
        }

        for j in 0..=grid.ny {
            for i in 0..grid.nx {
                velocity.v.set_face(i, j, v_value);
            }
        }

        velocity
    }

    #[test]
    fn constant_scalar_field_remains_constant_under_advection() {
        let grid = GridSize::new(8, 6, 1.0).expect("grid should be valid");
        let velocity = constant_velocity(grid, 1.0, -0.25);
        let source = ScalarField::new_filled(grid, 3.0);
        let mut destination = ScalarField::zeros(grid);

        advect_scalar(&source, &velocity, 0.5, &mut destination);

        for j in 0..grid.ny {
            for i in 0..grid.nx {
                assert_relative_eq!(destination.get_cell(i, j), 3.0);
            }
        }

        assert!(all_finite(destination.as_slice()));
    }

    #[test]
    fn scalar_blob_moves_in_flow_direction() {
        let grid = GridSize::new(8, 6, 1.0).expect("grid should be valid");
        let velocity = constant_velocity(grid, 1.0, 0.0);
        let mut source = ScalarField::zeros(grid);
        let mut destination = ScalarField::zeros(grid);

        source.set_cell(2, 3, 1.0);
        advect_scalar(&source, &velocity, 1.0, &mut destination);

        assert_eq!(destination.get_cell(3, 3), 1.0);
        assert_eq!(destination.get_cell(2, 3), 0.0);
    }

    #[test]
    fn maccormack_preserves_constant_scalar_field() {
        let grid = GridSize::new(8, 6, 1.0).expect("grid should be valid");
        let velocity = constant_velocity(grid, 0.5, -0.25);
        let source = ScalarField::new_filled(grid, 2.5);
        let mut forward = ScalarField::zeros(grid);
        let mut reverse = ScalarField::zeros(grid);
        let mut destination = ScalarField::zeros(grid);

        advect_scalar_maccormack(
            &source,
            &velocity,
            0.5,
            &mut forward,
            &mut reverse,
            &mut destination,
        );

        for j in 0..grid.ny {
            for i in 0..grid.nx {
                assert_relative_eq!(destination.get_cell(i, j), 2.5);
            }
        }
    }

    #[test]
    fn maccormack_retains_sharper_peak_than_semi_lagrangian() {
        let grid = GridSize::new(16, 8, 1.0).expect("grid should be valid");
        let velocity = constant_velocity(grid, 0.5, 0.0);
        let mut source = ScalarField::zeros(grid);
        let mut semi_lagrangian = ScalarField::zeros(grid);
        let mut forward = ScalarField::zeros(grid);
        let mut reverse = ScalarField::zeros(grid);
        let mut maccormack = ScalarField::zeros(grid);

        source.set_cell(4, 4, 1.0);

        advect_scalar(&source, &velocity, 1.0, &mut semi_lagrangian);
        advect_scalar_maccormack(
            &source,
            &velocity,
            1.0,
            &mut forward,
            &mut reverse,
            &mut maccormack,
        );

        let semi_peak = semi_lagrangian
            .as_slice()
            .iter()
            .fold(0.0_f32, |acc, value| acc.max(*value));
        let maccormack_peak = maccormack
            .as_slice()
            .iter()
            .fold(0.0_f32, |acc, value| acc.max(*value));

        assert!(maccormack_peak >= semi_peak);
        assert!(maccormack_peak <= 1.0);
    }

    #[test]
    fn constant_velocity_field_remains_constant_under_self_advection() {
        let grid = GridSize::new(6, 5, 1.0).expect("grid should be valid");
        let source = constant_velocity(grid, 0.75, -0.5);
        let mut destination = MacVelocity::zeros(grid);

        advect_velocity(&source, 0.25, &mut destination);

        for j in 0..grid.ny {
            for i in 0..=grid.nx {
                assert_relative_eq!(destination.u.get_face(i, j), 0.75);
            }
        }

        for j in 0..=grid.ny {
            for i in 0..grid.nx {
                assert_relative_eq!(destination.v.get_face(i, j), -0.5);
            }
        }
    }

    #[test]
    fn scalar_sampling_matches_affine_field() {
        let grid = GridSize::new(4, 4, 1.0).expect("grid should be valid");
        let mut field = ScalarField::zeros(grid);

        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let position = grid.cell_center(i, j);
                field.set_cell(i, j, 2.0 * position.x - position.y + 1.0);
            }
        }

        let value = sample_scalar(&field, Vec2::new(1.75, 2.25));
        assert_relative_eq!(value, 2.0 * 1.75 - 2.25 + 1.0);
    }

    #[test]
    fn velocity_sampling_reconstructs_constant_mac_field() {
        let grid = GridSize::new(4, 3, 1.0).expect("grid should be valid");
        let velocity = constant_velocity(grid, 2.5, -1.25);

        let sample = sample_velocity(&velocity, Vec2::new(1.1, 1.7));

        assert_relative_eq!(sample.x, 2.5);
        assert_relative_eq!(sample.y, -1.25);
    }
}
