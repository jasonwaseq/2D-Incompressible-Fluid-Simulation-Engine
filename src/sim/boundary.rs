//! Boundary-condition routines live here.
//!
//! Phase 1 only establishes the module boundary so later phases can keep wall
//! handling centralized instead of scattering it across advection, diffusion,
//! and projection passes.

use super::field::{MacVelocity, ScalarField, SolidMask};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VelocityBoundary {
    NoSlipBox,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarBoundary {
    ZeroGradientBox,
}

pub fn apply_scalar_boundary(field: &mut ScalarField, boundary: ScalarBoundary) {
    match boundary {
        ScalarBoundary::ZeroGradientBox => apply_zero_gradient_scalar(field),
    }
}

pub fn apply_scalar_boundary_with_solids(
    field: &mut ScalarField,
    boundary: ScalarBoundary,
    solids: &SolidMask,
) {
    assert_eq!(
        field.grid(),
        solids.grid(),
        "scalar field and solid mask grids must match"
    );
    apply_scalar_boundary(field, boundary);
    apply_solid_scalar_cells(field, solids);
}

pub fn apply_velocity_boundary(velocity: &mut MacVelocity, boundary: VelocityBoundary) {
    match boundary {
        VelocityBoundary::NoSlipBox => apply_no_slip_box(velocity),
    }
}

pub fn apply_velocity_boundary_with_solids(
    velocity: &mut MacVelocity,
    boundary: VelocityBoundary,
    solids: &SolidMask,
) {
    assert_eq!(
        velocity.u.grid(),
        solids.grid(),
        "velocity field and solid mask grids must match"
    );
    apply_velocity_boundary(velocity, boundary);
    apply_solid_velocity_faces(velocity, solids);
}

fn apply_zero_gradient_scalar(field: &mut ScalarField) {
    let grid = field.grid();

    for j in 1..=grid.ny {
        field.set_raw(0, j, field.get_raw(1, j));
        field.set_raw(grid.nx + 1, j, field.get_raw(grid.nx, j));
    }

    for i in 0..=grid.nx + 1 {
        field.set_raw(i, 0, field.get_raw(i, 1));
        field.set_raw(i, grid.ny + 1, field.get_raw(i, grid.ny));
    }

    field.set_raw(0, 0, 0.5 * (field.get_raw(1, 0) + field.get_raw(0, 1)));
    field.set_raw(
        grid.nx + 1,
        0,
        0.5 * (field.get_raw(grid.nx, 0) + field.get_raw(grid.nx + 1, 1)),
    );
    field.set_raw(
        0,
        grid.ny + 1,
        0.5 * (field.get_raw(1, grid.ny + 1) + field.get_raw(0, grid.ny)),
    );
    field.set_raw(
        grid.nx + 1,
        grid.ny + 1,
        0.5 * (field.get_raw(grid.nx, grid.ny + 1) + field.get_raw(grid.nx + 1, grid.ny)),
    );
}

fn apply_no_slip_box(velocity: &mut MacVelocity) {
    let grid = velocity.u.grid();

    for j in 1..=grid.ny {
        velocity.u.set_raw(1, j, 0.0);
        velocity.u.set_raw(grid.nx + 1, j, 0.0);
        velocity.u.set_raw(0, j, 0.0);
        velocity.u.set_raw(grid.nx + 2, j, 0.0);
    }

    for i in 0..=grid.nx + 2 {
        let bottom = -velocity.u.get_raw(i, 1);
        let top = -velocity.u.get_raw(i, grid.ny);
        velocity.u.set_raw(i, 0, bottom);
        velocity.u.set_raw(i, grid.ny + 1, top);
    }

    for i in 1..=grid.nx {
        velocity.v.set_raw(i, 1, 0.0);
        velocity.v.set_raw(i, grid.ny + 1, 0.0);
        velocity.v.set_raw(i, 0, 0.0);
        velocity.v.set_raw(i, grid.ny + 2, 0.0);
    }

    for j in 0..=grid.ny + 2 {
        let left = -velocity.v.get_raw(1, j);
        let right = -velocity.v.get_raw(grid.nx, j);
        velocity.v.set_raw(0, j, left);
        velocity.v.set_raw(grid.nx + 1, j, right);
    }
}

fn apply_solid_scalar_cells(field: &mut ScalarField, solids: &SolidMask) {
    let grid = field.grid();

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            if solids.is_solid_cell(i, j) {
                field.set_cell(i, j, 0.0);
            }
        }
    }
}

fn apply_solid_velocity_faces(velocity: &mut MacVelocity, solids: &SolidMask) {
    let grid = velocity.u.grid();

    for j in 0..grid.ny {
        for i in 0..=grid.nx {
            if let Some(obstacle_velocity) = solids.u_face_obstacle_velocity(i, j) {
                velocity.u.set_face(i, j, obstacle_velocity);
            }
        }
    }

    for j in 0..=grid.ny {
        for i in 0..grid.nx {
            if let Some(obstacle_velocity) = solids.v_face_obstacle_velocity(i, j) {
                velocity.v.set_face(i, j, obstacle_velocity);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec2;

    use super::{
        apply_scalar_boundary, apply_scalar_boundary_with_solids, apply_velocity_boundary,
        apply_velocity_boundary_with_solids, ScalarBoundary, VelocityBoundary,
    };
    use crate::sim::field::{MacVelocity, ScalarField, SolidMask};
    use crate::sim::grid::GridSize;

    #[test]
    fn scalar_boundary_copies_edge_values_into_ghost_cells() {
        let grid = GridSize::new(3, 2, 1.0).expect("grid should be valid");
        let mut field = ScalarField::zeros(grid);
        field.set_cell(0, 0, 2.0);
        field.set_cell(2, 1, 5.0);

        apply_scalar_boundary(&mut field, ScalarBoundary::ZeroGradientBox);

        assert_eq!(field.get_raw(0, 1), 2.0);
        assert_eq!(field.get_raw(4, 2), 5.0);
    }

    #[test]
    fn velocity_boundary_zeroes_wall_normal_components() {
        let grid = GridSize::new(4, 3, 1.0).expect("grid should be valid");
        let mut velocity = MacVelocity::zeros(grid);

        for j in 0..grid.ny {
            for i in 0..=grid.nx {
                velocity.u.set_face(i, j, 1.0);
            }
        }

        for j in 0..=grid.ny {
            for i in 0..grid.nx {
                velocity.v.set_face(i, j, -1.0);
            }
        }

        apply_velocity_boundary(&mut velocity, VelocityBoundary::NoSlipBox);

        for j in 0..grid.ny {
            assert_eq!(velocity.u.get_face(0, j), 0.0);
            assert_eq!(velocity.u.get_face(grid.nx, j), 0.0);
        }

        for i in 0..grid.nx {
            assert_eq!(velocity.v.get_face(i, 0), 0.0);
            assert_eq!(velocity.v.get_face(i, grid.ny), 0.0);
        }
    }

    #[test]
    fn obstacle_boundary_pins_faces_to_obstacle_velocity() {
        let grid = GridSize::new(5, 5, 1.0).expect("grid should be valid");
        let mut velocity = MacVelocity::zeros(grid);
        let mut solids = SolidMask::empty(grid);
        solids.set_solid_cell(2, 2, true);
        solids.set_cell_velocity(2, 2, Vec2::new(1.5, -0.75));

        apply_velocity_boundary_with_solids(&mut velocity, VelocityBoundary::NoSlipBox, &solids);

        assert_eq!(velocity.u.get_face(2, 2), 1.5);
        assert_eq!(velocity.u.get_face(3, 2), 1.5);
        assert_eq!(velocity.v.get_face(2, 2), -0.75);
        assert_eq!(velocity.v.get_face(2, 3), -0.75);
    }

    #[test]
    fn scalar_boundary_zeroes_values_inside_solids() {
        let grid = GridSize::new(4, 4, 1.0).expect("grid should be valid");
        let mut field = ScalarField::new_filled(grid, 2.0);
        let mut solids = SolidMask::empty(grid);
        solids.set_solid_cell(1, 1, true);

        apply_scalar_boundary_with_solids(&mut field, ScalarBoundary::ZeroGradientBox, &solids);

        assert_eq!(field.get_cell(1, 1), 0.0);
        assert_eq!(field.get_cell(0, 0), 2.0);
    }
}
