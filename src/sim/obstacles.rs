use glam::Vec2;

use super::field::SolidMask;

pub trait ObstaclePrimitive: std::fmt::Debug + Send + Sync {
    fn stamp(&self, solids: &mut SolidMask);
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CircleObstacle {
    pub center: Vec2,
    pub radius: f32,
    pub velocity: Vec2,
}

impl CircleObstacle {
    pub fn new(center: Vec2, radius: f32, velocity: Vec2) -> Self {
        assert!(
            radius.is_finite() && radius > 0.0,
            "radius must be positive"
        );
        Self {
            center,
            radius,
            velocity,
        }
    }
}

impl ObstaclePrimitive for CircleObstacle {
    fn stamp(&self, solids: &mut SolidMask) {
        let grid = solids.grid();
        let radius_sq = self.radius * self.radius;

        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let position = grid.cell_center(i, j);
                let inside = (position - self.center).length_squared() <= radius_sq;
                solids.set_solid_cell(i, j, inside);

                if inside {
                    solids.set_cell_velocity(i, j, self.velocity);
                }
            }
        }
    }
}

pub fn clear_obstacles(solids: &mut SolidMask) {
    solids.fill(false);
}

#[cfg(test)]
mod tests {
    use glam::Vec2;

    use super::{clear_obstacles, CircleObstacle, ObstaclePrimitive};
    use crate::sim::field::SolidMask;
    use crate::sim::grid::GridSize;

    #[test]
    fn circle_obstacle_stamps_solid_cells_and_motion() {
        let grid = GridSize::new(9, 9, 1.0).expect("grid should be valid");
        let mut solids = SolidMask::empty(grid);
        let obstacle = CircleObstacle::new(Vec2::new(4.5, 4.5), 2.0, Vec2::new(1.0, 0.0));

        obstacle.stamp(&mut solids);

        assert!(solids.is_solid_cell(4, 4));
        assert_eq!(solids.cell_velocity(4, 4), Vec2::new(1.0, 0.0));
        assert!(!solids.is_solid_cell(0, 0));
    }

    #[test]
    fn clearing_obstacles_resets_occupancy() {
        let grid = GridSize::new(5, 5, 1.0).expect("grid should be valid");
        let mut solids = SolidMask::empty(grid);
        CircleObstacle::new(Vec2::new(2.5, 2.5), 1.5, Vec2::new(0.0, 1.0)).stamp(&mut solids);

        clear_obstacles(&mut solids);

        for j in 0..grid.ny {
            for i in 0..grid.nx {
                assert!(!solids.is_solid_cell(i, j));
            }
        }
    }
}
