struct SimParams {
    nx: u32,
    ny: u32,
    scalar_width: u32,
    scalar_height: u32,
    u_width: u32,
    u_height: u32,
    v_width: u32,
    v_height: u32,
    cell_size: f32,
    dt: f32,
    viscosity: f32,
    diffusion: f32,
    command_count: u32,
    pressure_phase: u32,
    pressure_anchor_x: u32,
    pressure_anchor_y: u32,
};

struct CommandRecord {
    kind: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    position: vec2<f32>,
    delta: vec2<f32>,
    amount: f32,
    radius: f32,
    _pad3: vec2<f32>,
};

const CMD_DYE: u32 = 1u;
const CMD_FORCE: u32 = 2u;

@group(0) @binding(0)
var<uniform> params: SimParams;

@group(1) @binding(0)
var<storage, read_write> buffer0: array<f32>;
@group(1) @binding(1)
var<storage, read_write> buffer1: array<f32>;
@group(1) @binding(2)
var<storage, read_write> buffer2: array<f32>;
@group(1) @binding(3)
var<storage, read_write> buffer3: array<f32>;
@group(1) @binding(4)
var<storage, read_write> buffer4: array<f32>;
@group(1) @binding(5)
var<storage, read_write> buffer5: array<f32>;
@group(1) @binding(6)
var<storage, read> obstacles: array<vec4<f32>>;
@group(1) @binding(7)
var<storage, read> commands: array<CommandRecord>;

fn scalar_index_raw(i: u32, j: u32) -> u32 {
    return i + j * params.scalar_width;
}

fn scalar_cell_index(i: u32, j: u32) -> u32 {
    return (i + 1u) + (j + 1u) * params.scalar_width;
}

fn u_index_raw(i: u32, j: u32) -> u32 {
    return i + j * params.u_width;
}

fn u_face_index(i: u32, j: u32) -> u32 {
    return (i + 1u) + (j + 1u) * params.u_width;
}

fn v_index_raw(i: u32, j: u32) -> u32 {
    return i + j * params.v_width;
}

fn v_face_index(i: u32, j: u32) -> u32 {
    return (i + 1u) + (j + 1u) * params.v_width;
}

fn domain_size() -> vec2<f32> {
    return vec2<f32>(f32(params.nx) * params.cell_size, f32(params.ny) * params.cell_size);
}

fn clamp_scalar_position(position: vec2<f32>) -> vec2<f32> {
    let half_h = 0.5 * params.cell_size;
    let domain = domain_size();
    return clamp(position, vec2<f32>(half_h, half_h), domain - vec2<f32>(half_h, half_h));
}

fn clamp_u_position(position: vec2<f32>) -> vec2<f32> {
    let half_h = 0.5 * params.cell_size;
    let domain = domain_size();
    return clamp(position, vec2<f32>(0.0, half_h), vec2<f32>(domain.x, domain.y - half_h));
}

fn clamp_v_position(position: vec2<f32>) -> vec2<f32> {
    let half_h = 0.5 * params.cell_size;
    let domain = domain_size();
    return clamp(position, vec2<f32>(half_h, 0.0), vec2<f32>(domain.x - half_h, domain.y));
}

fn bilerp(a00: f32, a10: f32, a01: f32, a11: f32, tx: f32, ty: f32) -> f32 {
    let ax0 = a00 + tx * (a10 - a00);
    let ax1 = a01 + tx * (a11 - a01);
    return ax0 + ty * (ax1 - ax0);
}

fn radial_falloff(sample_position: vec2<f32>, center: vec2<f32>, radius: f32) -> f32 {
    if radius <= 0.0 {
        return 0.0;
    }

    let normalized_distance = length(sample_position - center) / radius;
    if normalized_distance >= 1.0 {
        return 0.0;
    }

    let weight = 1.0 - normalized_distance;
    return weight * weight;
}

fn obstacle_raw(i: u32, j: u32) -> vec4<f32> {
    return obstacles[scalar_index_raw(i, j)];
}

fn obstacle_cell(i: u32, j: u32) -> vec4<f32> {
    return obstacles[scalar_cell_index(i, j)];
}

fn is_solid_cell(i: u32, j: u32) -> bool {
    return obstacle_cell(i, j).x > 0.5;
}

fn obstacle_velocity_cell(i: u32, j: u32) -> vec2<f32> {
    let value = obstacle_cell(i, j);
    return vec2<f32>(value.y, value.z);
}

fn u_face_obstacle(i: u32, j: u32) -> vec2<f32> {
    var sum = 0.0;
    var count = 0.0;

    if i > 0u && is_solid_cell(i - 1u, j) {
        sum = sum + obstacle_velocity_cell(i - 1u, j).x;
        count = count + 1.0;
    }

    if i < params.nx && is_solid_cell(i, j) {
        sum = sum + obstacle_velocity_cell(i, j).x;
        count = count + 1.0;
    }

    if count > 0.0 {
        return vec2<f32>(count, sum / count);
    }

    return vec2<f32>(0.0, 0.0);
}

fn v_face_obstacle(i: u32, j: u32) -> vec2<f32> {
    var sum = 0.0;
    var count = 0.0;

    if j > 0u && is_solid_cell(i, j - 1u) {
        sum = sum + obstacle_velocity_cell(i, j - 1u).y;
        count = count + 1.0;
    }

    if j < params.ny && is_solid_cell(i, j) {
        sum = sum + obstacle_velocity_cell(i, j).y;
        count = count + 1.0;
    }

    if count > 0.0 {
        return vec2<f32>(count, sum / count);
    }

    return vec2<f32>(0.0, 0.0);
}

fn fluid_neighbor_pressure_sum(i: u32, j: u32) -> vec2<f32> {
    var diag = 0.0;
    var sum = 0.0;

    if i > 0u && !is_solid_cell(i - 1u, j) {
        diag = diag + 1.0;
        sum = sum + buffer1[scalar_cell_index(i - 1u, j)];
    }
    if i + 1u < params.nx && !is_solid_cell(i + 1u, j) {
        diag = diag + 1.0;
        sum = sum + buffer1[scalar_cell_index(i + 1u, j)];
    }
    if j > 0u && !is_solid_cell(i, j - 1u) {
        diag = diag + 1.0;
        sum = sum + buffer1[scalar_cell_index(i, j - 1u)];
    }
    if j + 1u < params.ny && !is_solid_cell(i, j + 1u) {
        diag = diag + 1.0;
        sum = sum + buffer1[scalar_cell_index(i, j + 1u)];
    }

    return vec2<f32>(diag, sum);
}

fn scalar_value(source: u32, index: u32) -> f32 {
    switch source {
        case 0u: { return buffer0[index]; }
        case 1u: { return buffer1[index]; }
        case 2u: { return buffer2[index]; }
        case 3u: { return buffer3[index]; }
        case 4u: { return buffer4[index]; }
        case 5u: { return buffer5[index]; }
        default: { return 0.0; }
    }
}

fn u_value(source: u32, index: u32) -> f32 {
    return scalar_value(source, index);
}

fn v_value(source: u32, index: u32) -> f32 {
    return scalar_value(source, index);
}

fn sample_scalar(source: u32, position: vec2<f32>) -> f32 {
    let clamped = clamp_scalar_position(position);
    let inv_h = 1.0 / params.cell_size;
    let gx = clamped.x * inv_h - 0.5;
    let gy = clamped.y * inv_h - 0.5;
    let i0 = u32(floor(gx));
    let j0 = u32(floor(gy));
    let i1 = min(i0 + 1u, params.nx - 1u);
    let j1 = min(j0 + 1u, params.ny - 1u);
    let tx = gx - f32(i0);
    let ty = gy - f32(j0);
    return bilerp(
        scalar_value(source, scalar_cell_index(i0, j0)),
        scalar_value(source, scalar_cell_index(i1, j0)),
        scalar_value(source, scalar_cell_index(i0, j1)),
        scalar_value(source, scalar_cell_index(i1, j1)),
        tx,
        ty,
    );
}

fn sample_u(source: u32, position: vec2<f32>) -> f32 {
    let clamped = clamp_u_position(position);
    let inv_h = 1.0 / params.cell_size;
    let gx = clamped.x * inv_h;
    let gy = clamped.y * inv_h - 0.5;
    let i0 = u32(floor(gx));
    let j0 = u32(floor(gy));
    let i1 = min(i0 + 1u, params.nx);
    let j1 = min(j0 + 1u, params.ny - 1u);
    let tx = gx - f32(i0);
    let ty = gy - f32(j0);
    return bilerp(
        u_value(source, u_face_index(i0, j0)),
        u_value(source, u_face_index(i1, j0)),
        u_value(source, u_face_index(i0, j1)),
        u_value(source, u_face_index(i1, j1)),
        tx,
        ty,
    );
}

fn sample_v(source: u32, position: vec2<f32>) -> f32 {
    let clamped = clamp_v_position(position);
    let inv_h = 1.0 / params.cell_size;
    let gx = clamped.x * inv_h - 0.5;
    let gy = clamped.y * inv_h;
    let i0 = u32(floor(gx));
    let j0 = u32(floor(gy));
    let i1 = min(i0 + 1u, params.nx - 1u);
    let j1 = min(j0 + 1u, params.ny);
    let tx = gx - f32(i0);
    let ty = gy - f32(j0);
    return bilerp(
        v_value(source, v_face_index(i0, j0)),
        v_value(source, v_face_index(i1, j0)),
        v_value(source, v_face_index(i0, j1)),
        v_value(source, v_face_index(i1, j1)),
        tx,
        ty,
    );
}

fn sample_velocity(u_source: u32, v_source: u32, position: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(sample_u(u_source, position), sample_v(v_source, position));
}

@compute @workgroup_size(8, 8, 1)
fn apply_scalar_commands(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.nx || gid.y >= params.ny {
        return;
    }

    if is_solid_cell(gid.x, gid.y) {
        buffer0[scalar_cell_index(gid.x, gid.y)] = 0.0;
        return;
    }

    let index = scalar_cell_index(gid.x, gid.y);
    let position = vec2<f32>((f32(gid.x) + 0.5) * params.cell_size, (f32(gid.y) + 0.5) * params.cell_size);
    var value = buffer0[index];

    for (var command_index: u32 = 0u; command_index < params.command_count; command_index = command_index + 1u) {
        let command = commands[command_index];
        if command.kind == CMD_DYE {
            let weight = radial_falloff(position, command.position, command.radius);
            value = value + command.amount * weight;
        }
    }

    buffer0[index] = value;
}

@compute @workgroup_size(8, 8, 1)
fn apply_u_commands(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x > params.nx || gid.y >= params.ny {
        return;
    }

    let obstacle = u_face_obstacle(gid.x, gid.y);
    if obstacle.x > 0.0 {
        buffer0[u_face_index(gid.x, gid.y)] = obstacle.y;
        return;
    }

    let index = u_face_index(gid.x, gid.y);
    let position = vec2<f32>(f32(gid.x) * params.cell_size, (f32(gid.y) + 0.5) * params.cell_size);
    var value = buffer0[index];

    for (var command_index: u32 = 0u; command_index < params.command_count; command_index = command_index + 1u) {
        let command = commands[command_index];
        if command.kind == CMD_FORCE {
            let weight = radial_falloff(position, command.position, command.radius);
            value = value + command.delta.x * weight;
        }
    }

    buffer0[index] = value;
}

@compute @workgroup_size(8, 8, 1)
fn apply_v_commands(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.nx || gid.y > params.ny {
        return;
    }

    let obstacle = v_face_obstacle(gid.x, gid.y);
    if obstacle.x > 0.0 {
        buffer0[v_face_index(gid.x, gid.y)] = obstacle.y;
        return;
    }

    let index = v_face_index(gid.x, gid.y);
    let position = vec2<f32>((f32(gid.x) + 0.5) * params.cell_size, f32(gid.y) * params.cell_size);
    var value = buffer0[index];

    for (var command_index: u32 = 0u; command_index < params.command_count; command_index = command_index + 1u) {
        let command = commands[command_index];
        if command.kind == CMD_FORCE {
            let weight = radial_falloff(position, command.position, command.radius);
            value = value + command.delta.y * weight;
        }
    }

    buffer0[index] = value;
}

@compute @workgroup_size(8, 8, 1)
fn apply_scalar_boundary(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if i >= params.scalar_width || j >= params.scalar_height {
        return;
    }

    let last_x = params.scalar_width - 1u;
    let last_y = params.scalar_height - 1u;
    let index = scalar_index_raw(i, j);

    if i == 0u && j == 0u {
        buffer0[index] = buffer0[scalar_index_raw(1u, 1u)];
    } else if i == last_x && j == 0u {
        buffer0[index] = buffer0[scalar_index_raw(params.nx, 1u)];
    } else if i == 0u && j == last_y {
        buffer0[index] = buffer0[scalar_index_raw(1u, params.ny)];
    } else if i == last_x && j == last_y {
        buffer0[index] = buffer0[scalar_index_raw(params.nx, params.ny)];
    } else if i == 0u && j > 0u && j < last_y {
        buffer0[index] = buffer0[scalar_index_raw(1u, j)];
    } else if i == last_x && j > 0u && j < last_y {
        buffer0[index] = buffer0[scalar_index_raw(params.nx, j)];
    } else if j == 0u && i > 0u && i < last_x {
        buffer0[index] = buffer0[scalar_index_raw(i, 1u)];
    } else if j == last_y && i > 0u && i < last_x {
        buffer0[index] = buffer0[scalar_index_raw(i, params.ny)];
    } else if i > 0u && i < last_x && j > 0u && j < last_y && obstacle_raw(i, j).x > 0.5 {
        buffer0[index] = 0.0;
    }
}

@compute @workgroup_size(8, 8, 1)
fn apply_u_boundary(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if i >= params.u_width || j >= params.u_height {
        return;
    }

    let last_y = params.u_height - 1u;
    let index = u_index_raw(i, j);

    if i > 0u && i < params.u_width - 1u && j > 0u && j < params.u_height - 1u {
        let obstacle = u_face_obstacle(i - 1u, j - 1u);
        if obstacle.x > 0.0 {
            buffer0[index] = obstacle.y;
            return;
        }
    }

    if j == 0u {
        if i <= 1u || i >= params.nx + 1u {
            buffer0[index] = 0.0;
        } else {
            buffer0[index] = -buffer0[u_index_raw(i, 1u)];
        }
        return;
    }

    if j == last_y {
        if i <= 1u || i >= params.nx + 1u {
            buffer0[index] = 0.0;
        } else {
            buffer0[index] = -buffer0[u_index_raw(i, params.ny)];
        }
        return;
    }

    if i == 0u || i == 1u || i == params.nx + 1u || i == params.nx + 2u {
        buffer0[index] = 0.0;
    }
}

@compute @workgroup_size(8, 8, 1)
fn apply_v_boundary(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if i >= params.v_width || j >= params.v_height {
        return;
    }

    let last_x = params.v_width - 1u;
    let index = v_index_raw(i, j);

    if i > 0u && i < params.v_width - 1u && j > 0u && j < params.v_height - 1u {
        let obstacle = v_face_obstacle(i - 1u, j - 1u);
        if obstacle.x > 0.0 {
            buffer0[index] = obstacle.y;
            return;
        }
    }

    if i == 0u {
        if j <= 1u || j >= params.ny + 1u {
            buffer0[index] = 0.0;
        } else {
            buffer0[index] = -buffer0[v_index_raw(1u, j)];
        }
        return;
    }

    if i == last_x {
        if j <= 1u || j >= params.ny + 1u {
            buffer0[index] = 0.0;
        } else {
            buffer0[index] = -buffer0[v_index_raw(params.nx, j)];
        }
        return;
    }

    if j == 0u || j == 1u || j == params.ny + 1u || j == params.ny + 2u {
        buffer0[index] = 0.0;
    }
}

@compute @workgroup_size(8, 8, 1)
fn advect_scalar(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.nx || gid.y >= params.ny {
        return;
    }

    if is_solid_cell(gid.x, gid.y) {
        buffer3[scalar_cell_index(gid.x, gid.y)] = 0.0;
        return;
    }

    let position = vec2<f32>((f32(gid.x) + 0.5) * params.cell_size, (f32(gid.y) + 0.5) * params.cell_size);
    let flow = sample_velocity(1u, 2u, position);
    let previous_position = clamp_scalar_position(position - params.dt * flow);
    buffer3[scalar_cell_index(gid.x, gid.y)] = sample_scalar(0u, previous_position);
}

@compute @workgroup_size(8, 8, 1)
fn maccormack_correct(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.nx || gid.y >= params.ny {
        return;
    }

    if is_solid_cell(gid.x, gid.y) {
        buffer5[scalar_cell_index(gid.x, gid.y)] = 0.0;
        return;
    }

    let position = vec2<f32>((f32(gid.x) + 0.5) * params.cell_size, (f32(gid.y) + 0.5) * params.cell_size);
    let flow = sample_velocity(3u, 4u, position);
    let previous_position = clamp_scalar_position(position - params.dt * flow);

    let source_center = buffer0[scalar_cell_index(gid.x, gid.y)];
    let forward_center = buffer1[scalar_cell_index(gid.x, gid.y)];
    let reverse_center = buffer2[scalar_cell_index(gid.x, gid.y)];
    let corrected = forward_center + 0.5 * (source_center - reverse_center);

    let inv_h = 1.0 / params.cell_size;
    let gx = previous_position.x * inv_h - 0.5;
    let gy = previous_position.y * inv_h - 0.5;
    let i0 = u32(floor(gx));
    let j0 = u32(floor(gy));
    let i1 = min(i0 + 1u, params.nx - 1u);
    let j1 = min(j0 + 1u, params.ny - 1u);

    let s00 = buffer0[scalar_cell_index(i0, j0)];
    let s10 = buffer0[scalar_cell_index(i1, j0)];
    let s01 = buffer0[scalar_cell_index(i0, j1)];
    let s11 = buffer0[scalar_cell_index(i1, j1)];
    let min_value = min(min(s00, s10), min(s01, s11));
    let max_value = max(max(s00, s10), max(s01, s11));

    buffer5[scalar_cell_index(gid.x, gid.y)] = clamp(corrected, min_value, max_value);
}

@compute @workgroup_size(8, 8, 1)
fn advect_u(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x > params.nx || gid.y >= params.ny {
        return;
    }

    let obstacle = u_face_obstacle(gid.x, gid.y);
    if obstacle.x > 0.0 {
        buffer2[u_face_index(gid.x, gid.y)] = obstacle.y;
        return;
    }

    let position = vec2<f32>(f32(gid.x) * params.cell_size, (f32(gid.y) + 0.5) * params.cell_size);
    let flow = sample_velocity(0u, 1u, position);
    let previous_position = clamp_u_position(position - params.dt * flow);
    buffer2[u_face_index(gid.x, gid.y)] = sample_u(0u, previous_position);
}

@compute @workgroup_size(8, 8, 1)
fn advect_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.nx || gid.y > params.ny {
        return;
    }

    let obstacle = v_face_obstacle(gid.x, gid.y);
    if obstacle.x > 0.0 {
        buffer2[v_face_index(gid.x, gid.y)] = obstacle.y;
        return;
    }

    let position = vec2<f32>((f32(gid.x) + 0.5) * params.cell_size, f32(gid.y) * params.cell_size);
    let flow = sample_velocity(0u, 1u, position);
    let previous_position = clamp_v_position(position - params.dt * flow);
    buffer2[v_face_index(gid.x, gid.y)] = sample_v(1u, previous_position);
}

@compute @workgroup_size(8, 8, 1)
fn diffuse_scalar_jacobi(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.nx || gid.y >= params.ny {
        return;
    }

    if is_solid_cell(gid.x, gid.y) {
        buffer2[scalar_cell_index(gid.x, gid.y)] = 0.0;
        return;
    }

    let alpha = params.diffusion * params.dt / (params.cell_size * params.cell_size);
    let denom = 1.0 + 4.0 * alpha;
    let index = scalar_cell_index(gid.x, gid.y);
    let left_i = select(gid.x, gid.x - 1u, gid.x > 0u);
    let right_i = min(gid.x + 1u, params.nx - 1u);
    let down_j = select(gid.y, gid.y - 1u, gid.y > 0u);
    let up_j = min(gid.y + 1u, params.ny - 1u);

    let left = buffer1[scalar_cell_index(left_i, gid.y)];
    let right = buffer1[scalar_cell_index(right_i, gid.y)];
    let down = buffer1[scalar_cell_index(gid.x, down_j)];
    let up = buffer1[scalar_cell_index(gid.x, up_j)];
    buffer2[index] = (buffer0[index] + alpha * (left + right + down + up)) / denom;
}

@compute @workgroup_size(8, 8, 1)
fn diffuse_u_jacobi(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x > params.nx || gid.y >= params.ny {
        return;
    }

    let obstacle = u_face_obstacle(gid.x, gid.y);
    if obstacle.x > 0.0 {
        buffer2[u_face_index(gid.x, gid.y)] = obstacle.y;
        return;
    }

    let alpha = params.viscosity * params.dt / (params.cell_size * params.cell_size);
    let denom = 1.0 + 4.0 * alpha;
    let i = gid.x + 1u;
    let j = gid.y + 1u;
    let index = u_index_raw(i, j);
    let left = buffer1[u_index_raw(i - 1u, j)];
    let right = buffer1[u_index_raw(i + 1u, j)];
    let down = buffer1[u_index_raw(i, j - 1u)];
    let up = buffer1[u_index_raw(i, j + 1u)];
    buffer2[index] = (buffer0[index] + alpha * (left + right + down + up)) / denom;
}

@compute @workgroup_size(8, 8, 1)
fn diffuse_v_jacobi(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.nx || gid.y > params.ny {
        return;
    }

    let obstacle = v_face_obstacle(gid.x, gid.y);
    if obstacle.x > 0.0 {
        buffer2[v_face_index(gid.x, gid.y)] = obstacle.y;
        return;
    }

    let alpha = params.viscosity * params.dt / (params.cell_size * params.cell_size);
    let denom = 1.0 + 4.0 * alpha;
    let i = gid.x + 1u;
    let j = gid.y + 1u;
    let index = v_index_raw(i, j);
    let left = buffer1[v_index_raw(i - 1u, j)];
    let right = buffer1[v_index_raw(i + 1u, j)];
    let down = buffer1[v_index_raw(i, j - 1u)];
    let up = buffer1[v_index_raw(i, j + 1u)];
    buffer2[index] = (buffer0[index] + alpha * (left + right + down + up)) / denom;
}

@compute @workgroup_size(8, 8, 1)
fn compute_divergence(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.nx || gid.y >= params.ny {
        return;
    }

    if is_solid_cell(gid.x, gid.y) {
        buffer2[scalar_cell_index(gid.x, gid.y)] = 0.0;
        return;
    }

    let inv_h = 1.0 / params.cell_size;
    let flux_x = buffer0[u_face_index(gid.x + 1u, gid.y)] - buffer0[u_face_index(gid.x, gid.y)];
    let flux_y = buffer1[v_face_index(gid.x, gid.y + 1u)] - buffer1[v_face_index(gid.x, gid.y)];
    buffer2[scalar_cell_index(gid.x, gid.y)] = (flux_x + flux_y) * inv_h;
}

@compute @workgroup_size(8, 8, 1)
fn pressure_red_black(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.nx || gid.y >= params.ny {
        return;
    }

    let index = scalar_cell_index(gid.x, gid.y);
    if is_solid_cell(gid.x, gid.y) {
        buffer1[index] = 0.0;
        return;
    }

    if gid.x == params.pressure_anchor_x && gid.y == params.pressure_anchor_y {
        buffer1[index] = 0.0;
        return;
    }

    if ((gid.x + gid.y) & 1u) != params.pressure_phase {
        return;
    }

    let neighbor = fluid_neighbor_pressure_sum(gid.x, gid.y);
    let diag = neighbor.x;
    if diag <= 0.0 {
        buffer1[index] = 0.0;
        return;
    }

    let h2 = params.cell_size * params.cell_size;
    buffer1[index] = (neighbor.y - h2 * buffer0[index]) / diag;
}

@compute @workgroup_size(8, 8, 1)
fn project_u(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x > params.nx || gid.y >= params.ny {
        return;
    }

    let obstacle = u_face_obstacle(gid.x, gid.y);
    let index = u_face_index(gid.x, gid.y);
    if obstacle.x > 0.0 {
        buffer2[index] = obstacle.y;
        return;
    }

    if gid.x == 0u || gid.x == params.nx {
        buffer2[index] = 0.0;
        return;
    }

    if is_solid_cell(gid.x - 1u, gid.y) || is_solid_cell(gid.x, gid.y) {
        buffer2[index] = buffer1[index];
        return;
    }

    let inv_h = 1.0 / params.cell_size;
    let gradient = (buffer0[scalar_cell_index(gid.x, gid.y)] - buffer0[scalar_cell_index(gid.x - 1u, gid.y)]) * inv_h;
    buffer2[index] = buffer1[index] - gradient;
}

@compute @workgroup_size(8, 8, 1)
fn project_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.nx || gid.y > params.ny {
        return;
    }

    let obstacle = v_face_obstacle(gid.x, gid.y);
    let index = v_face_index(gid.x, gid.y);
    if obstacle.x > 0.0 {
        buffer2[index] = obstacle.y;
        return;
    }

    if gid.y == 0u || gid.y == params.ny {
        buffer2[index] = 0.0;
        return;
    }

    if is_solid_cell(gid.x, gid.y - 1u) || is_solid_cell(gid.x, gid.y) {
        buffer2[index] = buffer1[index];
        return;
    }

    let inv_h = 1.0 / params.cell_size;
    let gradient = (buffer0[scalar_cell_index(gid.x, gid.y)] - buffer0[scalar_cell_index(gid.x, gid.y - 1u)]) * inv_h;
    buffer2[index] = buffer1[index] - gradient;
}

@compute @workgroup_size(8, 8, 1)
fn compute_vorticity(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.nx || gid.y >= params.ny {
        return;
    }

    let index = scalar_cell_index(gid.x, gid.y);
    if is_solid_cell(gid.x, gid.y) {
        buffer2[index] = 0.0;
        return;
    }

    let inv_h = 1.0 / params.cell_size;
    let dvdx = (buffer1[v_face_index(gid.x + 1u, gid.y)] - buffer1[v_face_index(gid.x, gid.y)]) * inv_h;
    let dudy = (buffer0[u_face_index(gid.x, gid.y + 1u)] - buffer0[u_face_index(gid.x, gid.y)]) * inv_h;
    buffer2[index] = dvdx - dudy;
}
