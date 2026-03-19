struct RenderParams {
    surface_and_grid: vec4<u32>,
    mode_and_flags: vec4<u32>,
    primary_scales: vec4<f32>,
    secondary_scales: vec4<f32>,
};

struct VsOut {
    @builtin(position) position: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> params: RenderParams;

@group(1) @binding(0)
var<storage, read> density: array<f32>;
@group(1) @binding(1)
var<storage, read> pressure: array<f32>;
@group(1) @binding(2)
var<storage, read> divergence: array<f32>;
@group(1) @binding(3)
var<storage, read> u_field: array<f32>;
@group(1) @binding(4)
var<storage, read> v_field: array<f32>;
@group(1) @binding(5)
var<storage, read> vorticity: array<f32>;
@group(1) @binding(6)
var<storage, read> obstacles: array<vec4<f32>>;

fn surface_width() -> u32 {
    return params.surface_and_grid.x;
}

fn surface_height() -> u32 {
    return params.surface_and_grid.y;
}

fn nx() -> u32 {
    return params.surface_and_grid.z;
}

fn ny() -> u32 {
    return params.surface_and_grid.w;
}

fn mode() -> u32 {
    return params.mode_and_flags.x;
}

fn density_scale() -> f32 {
    return params.primary_scales.x;
}

fn pressure_scale() -> f32 {
    return params.primary_scales.y;
}

fn divergence_scale() -> f32 {
    return params.primary_scales.z;
}

fn vorticity_scale() -> f32 {
    return params.primary_scales.w;
}

fn velocity_scale() -> f32 {
    return params.secondary_scales.x;
}

fn scalar_width() -> u32 {
    return nx() + 2u;
}

fn v_width() -> u32 {
    return nx() + 2u;
}

fn u_width() -> u32 {
    return nx() + 3u;
}

fn scalar_cell_index(i: u32, j: u32) -> u32 {
    return (i + 1u) + (j + 1u) * scalar_width();
}

fn u_face_index(i: u32, j: u32) -> u32 {
    return (i + 1u) + (j + 1u) * u_width();
}

fn v_face_index(i: u32, j: u32) -> u32 {
    return (i + 1u) + (j + 1u) * v_width();
}

fn obstacle_cell(i: u32, j: u32) -> vec4<f32> {
    return obstacles[scalar_cell_index(i, j)];
}

fn is_solid_cell(i: u32, j: u32) -> bool {
    return obstacle_cell(i, j).x > 0.5;
}

fn lerp_color(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + clamp(t, 0.0, 1.0) * (b - a);
}

fn fire(value: f32) -> vec3<f32> {
    let clamped = clamp(value, 0.0, 1.0);
    if clamped < 0.33 {
        let t = clamped / 0.33;
        return lerp_color(vec3<f32>(12.0, 20.0, 32.0), vec3<f32>(160.0, 40.0, 20.0), t) / 255.0;
    }
    if clamped < 0.66 {
        let t = (clamped - 0.33) / 0.33;
        return lerp_color(vec3<f32>(160.0, 40.0, 20.0), vec3<f32>(255.0, 170.0, 20.0), t) / 255.0;
    }

    let t = (clamped - 0.66) / 0.34;
    return lerp_color(vec3<f32>(255.0, 170.0, 20.0), vec3<f32>(255.0, 245.0, 220.0), t) / 255.0;
}

fn grayscale(value: f32) -> vec3<f32> {
    let clamped = clamp(value, 0.0, 1.0);
    return vec3<f32>(clamped, clamped, clamped);
}

fn signed_blue_red(value: f32) -> vec3<f32> {
    let clamped = clamp(value, -1.0, 1.0);
    if clamped >= 0.0 {
        return lerp_color(vec3<f32>(240.0, 240.0, 240.0), vec3<f32>(210.0, 45.0, 38.0), clamped) / 255.0;
    }
    return lerp_color(vec3<f32>(240.0, 240.0, 240.0), vec3<f32>(38.0, 90.0, 210.0), -clamped) / 255.0;
}

fn viewport_uv(frag_coord: vec2<f32>) -> vec3<f32> {
    let sw = f32(surface_width());
    let sh = f32(surface_height());
    let sim_aspect = f32(nx()) / f32(ny());
    let surface_aspect = sw / sh;

    if surface_aspect > sim_aspect {
        let viewport_width = sh * sim_aspect;
        let origin_x = 0.5 * (sw - viewport_width);
        if frag_coord.x < origin_x || frag_coord.x >= origin_x + viewport_width {
            return vec3<f32>(0.0, 0.0, 0.0);
        }
        return vec3<f32>((frag_coord.x - origin_x) / viewport_width, frag_coord.y / sh, 1.0);
    }

    let viewport_height = sw / sim_aspect;
    let origin_y = 0.5 * (sh - viewport_height);
    if frag_coord.y < origin_y || frag_coord.y >= origin_y + viewport_height {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    return vec3<f32>(frag_coord.x / sw, (frag_coord.y - origin_y) / viewport_height, 1.0);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );

    var output: VsOut;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return output;
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = viewport_uv(frag_coord.xy);
    if uv.z < 0.5 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let cell_x = min(u32(clamp(uv.x, 0.0, 0.999999) * f32(nx())), nx() - 1u);
    let cell_y = min(u32(clamp(uv.y, 0.0, 0.999999) * f32(ny())), ny() - 1u);

    if is_solid_cell(cell_x, cell_y) {
        return vec4<f32>(vec3<f32>(32.0 / 255.0), 1.0);
    }

    let scalar_index = scalar_cell_index(cell_x, cell_y);
    var color = vec3<f32>(0.0, 0.0, 0.0);

    switch mode() {
        case 0u: {
            color = fire(density[scalar_index] / max(density_scale(), 1.0e-6));
        }
        case 1u: {
            let u_center = 0.5 * (u_field[u_face_index(cell_x, cell_y)] + u_field[u_face_index(cell_x + 1u, cell_y)]);
            let v_center = 0.5 * (v_field[v_face_index(cell_x, cell_y)] + v_field[v_face_index(cell_x, cell_y + 1u)]);
            let speed = length(vec2<f32>(u_center, v_center));
            color = grayscale(speed / max(velocity_scale(), 1.0e-6));
        }
        case 2u: {
            color = signed_blue_red(pressure[scalar_index] / max(pressure_scale(), 1.0e-6));
        }
        case 3u: {
            color = signed_blue_red(divergence[scalar_index] / max(divergence_scale(), 1.0e-6));
        }
        case 4u: {
            color = signed_blue_red(vorticity[scalar_index] / max(vorticity_scale(), 1.0e-6));
        }
        default: {
            color = fire(density[scalar_index] / max(density_scale(), 1.0e-6));
        }
    }

    return vec4<f32>(color, 1.0);
}
