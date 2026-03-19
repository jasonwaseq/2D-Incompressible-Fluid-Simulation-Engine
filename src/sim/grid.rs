use std::error::Error;
use std::fmt::{Display, Formatter};

use glam::Vec2;

pub const GHOST_LAYERS: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldShape {
    pub width: usize,
    pub height: usize,
}

impl FieldShape {
    pub fn len(self) -> usize {
        self.width * self.height
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridSize {
    pub nx: usize,
    pub ny: usize,
    pub cell_size: f32,
}

impl GridSize {
    pub fn new(nx: usize, ny: usize, cell_size: f32) -> Result<Self, GridError> {
        if nx == 0 {
            return Err(GridError::ZeroWidth);
        }

        if ny == 0 {
            return Err(GridError::ZeroHeight);
        }

        if !(cell_size.is_finite() && cell_size > 0.0) {
            return Err(GridError::InvalidCellSize(cell_size));
        }

        Ok(Self { nx, ny, cell_size })
    }

    pub fn cell_count(&self) -> usize {
        self.nx * self.ny
    }

    pub fn u_face_count(&self) -> usize {
        (self.nx + 1) * self.ny
    }

    pub fn v_face_count(&self) -> usize {
        self.nx * (self.ny + 1)
    }

    pub fn scalar_shape(&self) -> FieldShape {
        FieldShape {
            width: self.nx + 2 * GHOST_LAYERS,
            height: self.ny + 2 * GHOST_LAYERS,
        }
    }

    #[inline]
    pub fn scalar_row_stride(&self) -> usize {
        self.nx + 2 * GHOST_LAYERS
    }

    pub fn u_shape(&self) -> FieldShape {
        FieldShape {
            width: self.nx + 1 + 2 * GHOST_LAYERS,
            height: self.ny + 2 * GHOST_LAYERS,
        }
    }

    #[inline]
    pub fn u_row_stride(&self) -> usize {
        self.nx + 1 + 2 * GHOST_LAYERS
    }

    pub fn v_shape(&self) -> FieldShape {
        FieldShape {
            width: self.nx + 2 * GHOST_LAYERS,
            height: self.ny + 1 + 2 * GHOST_LAYERS,
        }
    }

    #[inline]
    pub fn v_row_stride(&self) -> usize {
        self.nx + 2 * GHOST_LAYERS
    }

    #[inline]
    pub fn scalar_index_raw(&self, i: usize, j: usize) -> usize {
        Self::flatten_index(self.scalar_shape(), i, j, "scalar field")
    }

    #[inline]
    pub fn u_index_raw(&self, i: usize, j: usize) -> usize {
        Self::flatten_index(self.u_shape(), i, j, "u-face field")
    }

    #[inline]
    pub fn v_index_raw(&self, i: usize, j: usize) -> usize {
        Self::flatten_index(self.v_shape(), i, j, "v-face field")
    }

    #[inline]
    pub fn cell_to_scalar_raw(&self, i: usize, j: usize) -> (usize, usize) {
        self.assert_interior_cell(i, j);
        (i + GHOST_LAYERS, j + GHOST_LAYERS)
    }

    #[inline]
    pub fn u_face_to_raw(&self, i: usize, j: usize) -> (usize, usize) {
        assert!(
            i <= self.nx && j < self.ny,
            "u-face index out of bounds: ({i}, {j}) for interior dimensions {}x{}",
            self.nx + 1,
            self.ny
        );

        (i + GHOST_LAYERS, j + GHOST_LAYERS)
    }

    #[inline]
    pub fn v_face_to_raw(&self, i: usize, j: usize) -> (usize, usize) {
        assert!(
            i < self.nx && j <= self.ny,
            "v-face index out of bounds: ({i}, {j}) for interior dimensions {}x{}",
            self.nx,
            self.ny + 1
        );

        (i + GHOST_LAYERS, j + GHOST_LAYERS)
    }

    pub fn domain_size(&self) -> Vec2 {
        Vec2::new(
            self.nx as f32 * self.cell_size,
            self.ny as f32 * self.cell_size,
        )
    }

    pub fn cell_center(&self, i: usize, j: usize) -> Vec2 {
        self.assert_interior_cell(i, j);

        Vec2::new(
            (i as f32 + 0.5) * self.cell_size,
            (j as f32 + 0.5) * self.cell_size,
        )
    }

    pub fn u_face_position(&self, i: usize, j: usize) -> Vec2 {
        assert!(
            i <= self.nx && j < self.ny,
            "u-face index out of bounds: ({i}, {j}) for interior dimensions {}x{}",
            self.nx + 1,
            self.ny
        );

        Vec2::new(i as f32 * self.cell_size, (j as f32 + 0.5) * self.cell_size)
    }

    pub fn v_face_position(&self, i: usize, j: usize) -> Vec2 {
        assert!(
            i < self.nx && j <= self.ny,
            "v-face index out of bounds: ({i}, {j}) for interior dimensions {}x{}",
            self.nx,
            self.ny + 1
        );

        Vec2::new((i as f32 + 0.5) * self.cell_size, j as f32 * self.cell_size)
    }

    pub fn clamp_scalar_position(&self, position: Vec2) -> Vec2 {
        let half_h = 0.5 * self.cell_size;
        let domain = self.domain_size();

        Vec2::new(
            position.x.clamp(half_h, domain.x - half_h),
            position.y.clamp(half_h, domain.y - half_h),
        )
    }

    pub fn clamp_u_position(&self, position: Vec2) -> Vec2 {
        let half_h = 0.5 * self.cell_size;
        let domain = self.domain_size();

        Vec2::new(
            position.x.clamp(0.0, domain.x),
            position.y.clamp(half_h, domain.y - half_h),
        )
    }

    pub fn clamp_v_position(&self, position: Vec2) -> Vec2 {
        let half_h = 0.5 * self.cell_size;
        let domain = self.domain_size();

        Vec2::new(
            position.x.clamp(half_h, domain.x - half_h),
            position.y.clamp(0.0, domain.y),
        )
    }

    fn assert_interior_cell(&self, i: usize, j: usize) {
        assert!(
            i < self.nx && j < self.ny,
            "cell index out of bounds: ({i}, {j}) for interior dimensions {}x{}",
            self.nx,
            self.ny
        );
    }

    fn flatten_index(shape: FieldShape, i: usize, j: usize, label: &str) -> usize {
        assert!(
            i < shape.width && j < shape.height,
            "{label} raw index out of bounds: ({i}, {j}) for raw dimensions {}x{}",
            shape.width,
            shape.height
        );

        i + j * shape.width
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GridError {
    ZeroWidth,
    ZeroHeight,
    InvalidCellSize(f32),
}

impl Display for GridError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroWidth => write!(f, "grid width must be greater than zero"),
            Self::ZeroHeight => write!(f, "grid height must be greater than zero"),
            Self::InvalidCellSize(value) => write!(f, "cell_size must be > 0.0, got {value}"),
        }
    }
}

impl Error for GridError {}

#[cfg(test)]
mod tests {
    use glam::Vec2;

    use super::{FieldShape, GridSize};

    #[test]
    fn grid_reports_expected_shapes() {
        let grid = GridSize::new(4, 3, 1.0).expect("grid should be valid");

        assert_eq!(grid.scalar_shape(), FieldShape { width: 6, height: 5 });
        assert_eq!(grid.u_shape(), FieldShape { width: 7, height: 5 });
        assert_eq!(grid.v_shape(), FieldShape { width: 6, height: 6 });
    }

    #[test]
    fn raw_indices_are_row_major() {
        let grid = GridSize::new(4, 3, 1.0).expect("grid should be valid");

        assert_eq!(grid.scalar_index_raw(2, 3), 20);
        assert_eq!(grid.u_index_raw(3, 2), 17);
        assert_eq!(grid.v_index_raw(4, 1), 10);
    }

    #[test]
    fn interior_indices_map_to_ghost_padded_raw_indices() {
        let grid = GridSize::new(8, 5, 1.0).expect("grid should be valid");

        assert_eq!(grid.cell_to_scalar_raw(0, 0), (1, 1));
        assert_eq!(grid.cell_to_scalar_raw(7, 4), (8, 5));
        assert_eq!(grid.u_face_to_raw(8, 4), (9, 5));
        assert_eq!(grid.v_face_to_raw(7, 5), (8, 6));
    }

    #[test]
    fn grid_reports_physical_sample_positions() {
        let grid = GridSize::new(4, 3, 2.0).expect("grid should be valid");

        assert_eq!(grid.cell_center(1, 2), Vec2::new(3.0, 5.0));
        assert_eq!(grid.u_face_position(4, 1), Vec2::new(8.0, 3.0));
        assert_eq!(grid.v_face_position(2, 3), Vec2::new(5.0, 6.0));
    }
}
