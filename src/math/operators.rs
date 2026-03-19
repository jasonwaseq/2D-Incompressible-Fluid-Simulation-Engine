use glam::Vec2;

pub fn clamp01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

pub fn clamp_vec2(value: Vec2, min: Vec2, max: Vec2) -> Vec2 {
    Vec2::new(value.x.clamp(min.x, max.x), value.y.clamp(min.y, max.y))
}

pub fn all_finite(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use glam::Vec2;

    use super::{all_finite, clamp01, clamp_vec2};

    #[test]
    fn clamp01_limits_values() {
        assert_eq!(clamp01(-0.5), 0.0);
        assert_eq!(clamp01(1.5), 1.0);
    }

    #[test]
    fn clamp_vec2_limits_each_axis() {
        let value = clamp_vec2(Vec2::new(-1.0, 2.0), Vec2::ZERO, Vec2::ONE);
        assert_eq!(value, Vec2::new(0.0, 1.0));
    }

    #[test]
    fn all_finite_detects_nans() {
        assert!(all_finite(&[0.0, 1.0, 2.0]));
        assert!(!all_finite(&[0.0, f32::NAN]));
    }
}
