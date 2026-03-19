use glam::Vec2;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::grid::{FieldShape, GridSize};

#[derive(Debug, Clone)]
pub struct ScalarField {
    grid: GridSize,
    data: Vec<f32>,
}

impl ScalarField {
    pub fn zeros(grid: GridSize) -> Self {
        Self::new_filled(grid, 0.0)
    }

    pub fn new_filled(grid: GridSize, value: f32) -> Self {
        Self {
            grid,
            data: vec![value; grid.scalar_shape().len()],
        }
    }

    pub fn grid(&self) -> GridSize {
        self.grid
    }

    pub fn shape(&self) -> FieldShape {
        self.grid.scalar_shape()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    pub fn fill(&mut self, value: f32) {
        self.data.fill(value);
    }

    pub fn copy_from(&mut self, other: &Self) {
        assert_eq!(self.grid, other.grid, "scalar field grids must match");
        self.data.copy_from_slice(&other.data);
    }

    #[inline]
    pub fn get_raw(&self, i: usize, j: usize) -> f32 {
        self.data[self.grid.scalar_index_raw(i, j)]
    }

    #[inline]
    pub fn set_raw(&mut self, i: usize, j: usize, value: f32) {
        let index = self.grid.scalar_index_raw(i, j);
        self.data[index] = value;
    }

    #[inline]
    pub fn get_cell(&self, i: usize, j: usize) -> f32 {
        let (raw_i, raw_j) = self.grid.cell_to_scalar_raw(i, j);
        self.get_raw(raw_i, raw_j)
    }

    #[inline]
    pub fn set_cell(&mut self, i: usize, j: usize, value: f32) {
        let (raw_i, raw_j) = self.grid.cell_to_scalar_raw(i, j);
        self.set_raw(raw_i, raw_j, value);
    }
}

#[derive(Debug, Clone)]
pub struct FaceFieldX {
    grid: GridSize,
    data: Vec<f32>,
}

impl FaceFieldX {
    pub fn zeros(grid: GridSize) -> Self {
        Self::new_filled(grid, 0.0)
    }

    pub fn new_filled(grid: GridSize, value: f32) -> Self {
        Self {
            grid,
            data: vec![value; grid.u_shape().len()],
        }
    }

    pub fn grid(&self) -> GridSize {
        self.grid
    }

    pub fn shape(&self) -> FieldShape {
        self.grid.u_shape()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    pub fn fill(&mut self, value: f32) {
        self.data.fill(value);
    }

    pub fn copy_from(&mut self, other: &Self) {
        assert_eq!(self.grid, other.grid, "u-face field grids must match");
        self.data.copy_from_slice(&other.data);
    }

    #[inline]
    pub fn get_raw(&self, i: usize, j: usize) -> f32 {
        self.data[self.grid.u_index_raw(i, j)]
    }

    #[inline]
    pub fn set_raw(&mut self, i: usize, j: usize, value: f32) {
        let index = self.grid.u_index_raw(i, j);
        self.data[index] = value;
    }

    #[inline]
    pub fn get_face(&self, i: usize, j: usize) -> f32 {
        let (raw_i, raw_j) = self.grid.u_face_to_raw(i, j);
        self.get_raw(raw_i, raw_j)
    }

    #[inline]
    pub fn set_face(&mut self, i: usize, j: usize, value: f32) {
        let (raw_i, raw_j) = self.grid.u_face_to_raw(i, j);
        self.set_raw(raw_i, raw_j, value);
    }
}

#[derive(Debug, Clone)]
pub struct FaceFieldY {
    grid: GridSize,
    data: Vec<f32>,
}

impl FaceFieldY {
    pub fn zeros(grid: GridSize) -> Self {
        Self::new_filled(grid, 0.0)
    }

    pub fn new_filled(grid: GridSize, value: f32) -> Self {
        Self {
            grid,
            data: vec![value; grid.v_shape().len()],
        }
    }

    pub fn grid(&self) -> GridSize {
        self.grid
    }

    pub fn shape(&self) -> FieldShape {
        self.grid.v_shape()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    pub fn fill(&mut self, value: f32) {
        self.data.fill(value);
    }

    pub fn copy_from(&mut self, other: &Self) {
        assert_eq!(self.grid, other.grid, "v-face field grids must match");
        self.data.copy_from_slice(&other.data);
    }

    #[inline]
    pub fn get_raw(&self, i: usize, j: usize) -> f32 {
        self.data[self.grid.v_index_raw(i, j)]
    }

    #[inline]
    pub fn set_raw(&mut self, i: usize, j: usize, value: f32) {
        let index = self.grid.v_index_raw(i, j);
        self.data[index] = value;
    }

    #[inline]
    pub fn get_face(&self, i: usize, j: usize) -> f32 {
        let (raw_i, raw_j) = self.grid.v_face_to_raw(i, j);
        self.get_raw(raw_i, raw_j)
    }

    #[inline]
    pub fn set_face(&mut self, i: usize, j: usize, value: f32) {
        let (raw_i, raw_j) = self.grid.v_face_to_raw(i, j);
        self.set_raw(raw_i, raw_j, value);
    }
}

#[derive(Debug, Clone)]
pub struct SolidMask {
    grid: GridSize,
    solid: Vec<u8>,
    velocity: Vec<Vec2>,
}

impl SolidMask {
    pub fn empty(grid: GridSize) -> Self {
        Self {
            grid,
            solid: vec![0; grid.scalar_shape().len()],
            velocity: vec![Vec2::ZERO; grid.scalar_shape().len()],
        }
    }

    pub fn grid(&self) -> GridSize {
        self.grid
    }

    pub fn fill(&mut self, is_solid: bool) {
        self.solid.fill(u8::from(is_solid));
        if !is_solid {
            self.velocity.fill(Vec2::ZERO);
        }
    }

    pub fn clear_motion(&mut self) {
        self.velocity.fill(Vec2::ZERO);
    }

    pub fn is_solid_raw(&self, i: usize, j: usize) -> bool {
        self.solid[self.grid.scalar_index_raw(i, j)] != 0
    }

    pub fn set_solid_raw(&mut self, i: usize, j: usize, is_solid: bool) {
        let index = self.grid.scalar_index_raw(i, j);
        self.solid[index] = u8::from(is_solid);
        if !is_solid {
            self.velocity[index] = Vec2::ZERO;
        }
    }

    pub fn is_solid_cell(&self, i: usize, j: usize) -> bool {
        let (raw_i, raw_j) = self.grid.cell_to_scalar_raw(i, j);
        self.is_solid_raw(raw_i, raw_j)
    }

    pub fn set_solid_cell(&mut self, i: usize, j: usize, is_solid: bool) {
        let (raw_i, raw_j) = self.grid.cell_to_scalar_raw(i, j);
        self.set_solid_raw(raw_i, raw_j, is_solid);
    }

    pub fn cell_velocity(&self, i: usize, j: usize) -> Vec2 {
        let (raw_i, raw_j) = self.grid.cell_to_scalar_raw(i, j);
        self.raw_velocity(raw_i, raw_j)
    }

    pub fn set_cell_velocity(&mut self, i: usize, j: usize, velocity: Vec2) {
        let (raw_i, raw_j) = self.grid.cell_to_scalar_raw(i, j);
        self.set_raw_velocity(raw_i, raw_j, velocity);
    }

    pub fn raw_velocity(&self, i: usize, j: usize) -> Vec2 {
        self.velocity[self.grid.scalar_index_raw(i, j)]
    }

    pub fn set_raw_velocity(&mut self, i: usize, j: usize, velocity: Vec2) {
        let index = self.grid.scalar_index_raw(i, j);
        self.velocity[index] = velocity;
    }

    pub fn is_fluid_cell(&self, i: usize, j: usize) -> bool {
        !self.is_solid_cell(i, j)
    }

    pub fn u_face_obstacle_velocity(&self, i: usize, j: usize) -> Option<f32> {
        assert!(
            i <= self.grid.nx && j < self.grid.ny,
            "u-face index out of bounds: ({i}, {j})"
        );

        let mut sum = 0.0_f32;
        let mut count = 0;

        if i > 0 && self.is_solid_cell(i - 1, j) {
            sum += self.cell_velocity(i - 1, j).x;
            count += 1;
        }

        if i < self.grid.nx && self.is_solid_cell(i, j) {
            sum += self.cell_velocity(i, j).x;
            count += 1;
        }

        if count > 0 {
            Some(sum / count as f32)
        } else {
            None
        }
    }

    pub fn v_face_obstacle_velocity(&self, i: usize, j: usize) -> Option<f32> {
        assert!(
            i < self.grid.nx && j <= self.grid.ny,
            "v-face index out of bounds: ({i}, {j})"
        );

        let mut sum = 0.0_f32;
        let mut count = 0;

        if j > 0 && self.is_solid_cell(i, j - 1) {
            sum += self.cell_velocity(i, j - 1).y;
            count += 1;
        }

        if j < self.grid.ny && self.is_solid_cell(i, j) {
            sum += self.cell_velocity(i, j).y;
            count += 1;
        }

        if count > 0 {
            Some(sum / count as f32)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct MacVelocity {
    pub u: FaceFieldX,
    pub v: FaceFieldY,
}

impl MacVelocity {
    pub fn zeros(grid: GridSize) -> Self {
        Self {
            u: FaceFieldX::zeros(grid),
            v: FaceFieldY::zeros(grid),
        }
    }

    pub fn fill(&mut self, value: f32) {
        self.u.fill(value);
        self.v.fill(value);
    }

    pub fn copy_from(&mut self, other: &Self) {
        self.u.copy_from(&other.u);
        self.v.copy_from(&other.v);
    }

    pub fn cell_center_velocity(&self, i: usize, j: usize) -> Vec2 {
        let u_center = 0.5 * (self.u.get_face(i, j) + self.u.get_face(i + 1, j));
        let v_center = 0.5 * (self.v.get_face(i, j) + self.v.get_face(i, j + 1));
        Vec2::new(u_center, v_center)
    }

    pub fn max_speed(&self) -> f32 {
        let grid = self.u.grid();
        let u_data = self.u.as_slice();
        let v_data = self.v.as_slice();
        let u_stride = grid.u_row_stride();
        let v_stride = grid.v_row_stride();

        #[cfg(feature = "parallel")]
        {
            return (0..grid.ny)
                .into_par_iter()
                .map(|j| {
                    let u_row = (j + 1) * u_stride;
                    let v_row = (j + 1) * v_stride;
                    let v_next_row = v_row + v_stride;
                    let mut row_max = 0.0_f32;

                    for i in 0..grid.nx {
                        let u_center = 0.5 * (u_data[u_row + i + 1] + u_data[u_row + i + 2]);
                        let v_center = 0.5 * (v_data[v_row + i + 1] + v_data[v_next_row + i + 1]);
                        let speed = (u_center * u_center + v_center * v_center).sqrt();
                        row_max = row_max.max(speed);
                    }

                    row_max
                })
                .reduce(|| 0.0_f32, f32::max);
        }

        #[cfg(not(feature = "parallel"))]
        let mut max_speed = 0.0_f32;

        #[cfg(not(feature = "parallel"))]
        for j in 0..grid.ny {
            let u_row = (j + 1) * u_stride;
            let v_row = (j + 1) * v_stride;
            let v_next_row = v_row + v_stride;
            for i in 0..grid.nx {
                let u_center = 0.5 * (u_data[u_row + i + 1] + u_data[u_row + i + 2]);
                let v_center = 0.5 * (v_data[v_row + i + 1] + v_data[v_next_row + i + 1]);
                let speed = (u_center * u_center + v_center * v_center).sqrt();
                max_speed = max_speed.max(speed);
            }
        }

        #[cfg(not(feature = "parallel"))]
        max_speed
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec2;

    use super::{FaceFieldX, FaceFieldY, ScalarField, SolidMask};
    use crate::sim::grid::GridSize;

    #[test]
    fn scalar_field_uses_ghost_padded_indexing() {
        let grid = GridSize::new(4, 3, 1.0).expect("grid should be valid");
        let mut field = ScalarField::zeros(grid);

        field.set_cell(0, 0, 10.0);
        field.set_cell(3, 2, 42.0);
        field.set_raw(0, 0, -1.0);

        assert_eq!(field.get_cell(0, 0), 10.0);
        assert_eq!(field.get_cell(3, 2), 42.0);
        assert_eq!(field.get_raw(0, 0), -1.0);
        assert_eq!(field.shape().width, 6);
        assert_eq!(field.shape().height, 5);
    }

    #[test]
    fn face_fields_store_staggered_dimensions() {
        let grid = GridSize::new(4, 3, 1.0).expect("grid should be valid");
        let mut u = FaceFieldX::zeros(grid);
        let mut v = FaceFieldY::zeros(grid);

        u.set_face(4, 2, 3.5);
        v.set_face(3, 3, -2.25);

        assert_eq!(u.get_face(4, 2), 3.5);
        assert_eq!(v.get_face(3, 3), -2.25);
        assert_eq!(u.shape().width, 7);
        assert_eq!(v.shape().height, 6);
    }

    #[test]
    fn solid_mask_tracks_cell_occupancy() {
        let grid = GridSize::new(2, 2, 1.0).expect("grid should be valid");
        let mut mask = SolidMask::empty(grid);

        assert!(!mask.is_solid_cell(1, 1));
        mask.set_solid_cell(1, 1, true);
        assert!(mask.is_solid_cell(1, 1));
    }

    #[test]
    fn solid_mask_tracks_obstacle_velocity() {
        let grid = GridSize::new(3, 3, 1.0).expect("grid should be valid");
        let mut mask = SolidMask::empty(grid);
        mask.set_solid_cell(1, 1, true);
        mask.set_cell_velocity(1, 1, Vec2::new(2.0, -1.5));

        assert_eq!(mask.cell_velocity(1, 1), Vec2::new(2.0, -1.5));
        assert_eq!(mask.u_face_obstacle_velocity(1, 1), Some(2.0));
        assert_eq!(mask.v_face_obstacle_velocity(1, 1), Some(-1.5));
    }

    #[test]
    fn mac_velocity_reconstructs_cell_center_velocity() {
        let grid = GridSize::new(3, 2, 1.0).expect("grid should be valid");
        let mut velocity = super::MacVelocity::zeros(grid);

        velocity.u.set_face(1, 1, 2.0);
        velocity.u.set_face(2, 1, 4.0);
        velocity.v.set_face(1, 1, -1.0);
        velocity.v.set_face(1, 2, 1.0);

        assert_eq!(velocity.cell_center_velocity(1, 1), Vec2::new(3.0, 0.0));
    }
}
