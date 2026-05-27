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

  let keyPair = hashEntriesB[i];
  let key = keyPair.x;

  let cellCount = params.gridResolution.x * params.gridResolution.y * params.gridResolution.z;
  if (key >= cellCount) {
    return;
  }

  // start of key run
  if (i == 0u || hashEntriesB[i - 1u].x != key) {
    cellStart[key] = i;
  }

  // end of key run (exclusive)
  if (i + 1u >= params.particleCount || hashEntriesB[i + 1u].x != key) {
    cellEnd[key] = i + 1u;
  }
}