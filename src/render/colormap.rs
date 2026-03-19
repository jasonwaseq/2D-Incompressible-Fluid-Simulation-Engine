fn lerp_channel(a: u8, b: u8, t: f32) -> u8 {
    let blended = a as f32 + (b as f32 - a as f32) * t.clamp(0.0, 1.0);
    blended.round() as u8
}

pub fn rgba(r: u8, g: u8, b: u8) -> [u8; 4] {
    [r, g, b, 255]
}

pub fn grayscale(value: f32) -> [u8; 4] {
    let clamped = value.clamp(0.0, 1.0);
    let byte = (clamped * 255.0).round() as u8;
    rgba(byte, byte, byte)
}

pub fn fire(value: f32) -> [u8; 4] {
    let clamped = value.clamp(0.0, 1.0);

    if clamped < 0.33 {
        let t = clamped / 0.33;
        rgba(lerp_channel(12, 160, t), lerp_channel(20, 40, t), lerp_channel(32, 20, t))
    } else if clamped < 0.66 {
        let t = (clamped - 0.33) / 0.33;
        rgba(lerp_channel(160, 255, t), lerp_channel(40, 170, t), lerp_channel(20, 20, t))
    } else {
        let t = (clamped - 0.66) / 0.34;
        rgba(255, lerp_channel(170, 245, t), lerp_channel(20, 220, t))
    }
}

pub fn signed_blue_red(value: f32) -> [u8; 4] {
    let clamped = value.clamp(-1.0, 1.0);

    if clamped >= 0.0 {
        let t = clamped;
        rgba(lerp_channel(240, 210, t), lerp_channel(240, 45, t), lerp_channel(240, 38, t))
    } else {
        let t = -clamped;
        rgba(lerp_channel(240, 38, t), lerp_channel(240, 90, t), lerp_channel(240, 210, t))
    }
}

#[cfg(test)]
mod tests {
    use super::{fire, grayscale, signed_blue_red};

    #[test]
    fn grayscale_clamps_values() {
        assert_eq!(grayscale(-1.0), [0, 0, 0, 255]);
        assert_eq!(grayscale(2.0), [255, 255, 255, 255]);
    }

    #[test]
    fn fire_map_brightens_with_intensity() {
        let dark = fire(0.1);
        let bright = fire(0.9);

        assert!(bright[0] >= dark[0]);
    }

    #[test]
    fn signed_map_changes_color_with_sign() {
        assert_ne!(signed_blue_red(-0.5), signed_blue_red(0.5));
    }
}
