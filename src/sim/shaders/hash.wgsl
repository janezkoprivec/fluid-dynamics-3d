struct Particle {
  position: vec3<f32>,
  _pad0: f32,
  velocity: vec3<f32>,
  _pad1: f32,
  acceleration: vec3<f32>,
  _pad2: f32,
  density: f32,
  pressure: f32,
  _pad3: vec2<f32>,
};

struct SimParams {
  gravity: vec3<f32>,
  dt: f32,
  boxMin: vec3<f32>,
  boundaryDamping: f32,
  boxMax: vec3<f32>,
  smoothingRadius: f32,
  restDensity: f32,
  gasConstant: f32,
  viscosity: f32,
  gamma: f32,
  maxPressure: f32,
  boundarySlop: f32,
  particleMass: f32,
  particleCount: u32,
  gridResolution: vec3<u32>,
  wallRepulsion: f32,
  wallDamping: f32,
  wallRange: f32,
  _pad0: f32,
  _pad1: f32,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;
@group(0) @binding(2) var<storage, read_write> hashEntriesA: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> hashEntriesB: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> cellStart: array<u32>;
@group(0) @binding(5) var<storage, read_write> cellEnd: array<u32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.particleCount) {
    return;
  }

  let h = max(params.smoothingRadius, 1e-6);
  let p = particles[i].position;

  // integer cell coordinates
  let cell = vec3<i32>(floor((p - params.boxMin) / h));
  let maxCell = vec3<i32>(params.gridResolution) - vec3<i32>(1, 1, 1);
  let c = clamp(cell, vec3<i32>(0, 0, 0), maxCell);

  // flatten 3D -> 1D
  let resX = params.gridResolution.x;
  let resY = params.gridResolution.y;

  let cx = u32(c.x);
  let cy = u32(c.y);
  let cz = u32(c.z);

  let cellId = cx + cy * resX + cz * (resX * resY);
  hashEntriesA[i] = vec2<u32>(cellId, i);
}