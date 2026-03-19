pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

pub fn bilerp(c00: f32, c10: f32, c01: f32, c11: f32, tx: f32, ty: f32) -> f32 {
    let bottom = lerp(c00, c10, tx);
    let top = lerp(c01, c11, tx);
    lerp(bottom, top, ty)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::{bilerp, lerp};

    #[test]
    fn lerp_interpolates_linearly() {
        assert_relative_eq!(lerp(0.0, 10.0, 0.25), 2.5);
    }

    #[test]
    fn bilerp_is_exact_for_affine_data() {
        let affine = |x: f32, y: f32| 2.0 * x - 3.0 * y + 5.0;

        let sample = bilerp(
            affine(0.0, 0.0),
            affine(1.0, 0.0),
            affine(0.0, 1.0),
            affine(1.0, 1.0),
            0.25,
            0.75,
        );

        assert_relative_eq!(sample, affine(0.25, 0.75));
    }
}
