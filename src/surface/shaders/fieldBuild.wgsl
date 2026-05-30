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

struct FieldParams {
  gridMin: vec3<f32>,
  cellSize: f32,
  gridResolution: vec3<u32>,
  particleCount: u32,
  smoothingRadius: f32,
  particleMass: f32,
  restDensity: f32,
  _pad0: f32,
  boxMin: vec3<f32>,
  _pad1: f32,
  boxMax: vec3<f32>,
  _pad2: f32,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: FieldParams;
@group(0) @binding(2) var<storage, read_write> fieldValues: array<f32>;

const PI: f32 = 3.141592653589793;
const EPS: f32 = 1e-6;

fn flatten3(i: vec3<u32>, res: vec3<u32>) -> u32 {
  return i.x + i.y * res.x + i.z * (res.x * res.y);
}

fn unflatten3(id: u32, res: vec3<u32>) -> vec3<u32> {
  let xy = res.x * res.y;
  let z = id / xy;
  let rem = id - z * xy;
  let y = rem / res.x;
  let x = rem - y * res.x;
  return vec3<u32>(x, y, z);
}

fn poly6_from_r2(r2: f32, h: f32) -> f32 {
  if (h <= 0.0) { return 0.0; }
  let h2 = h * h;
  if (r2 > h2) { return 0.0; }
  let term = h2 - r2;
  let coeff = 315.0 / (64.0 * PI * pow(h, 9.0));
  return coeff * term * term * term;
}

@compute @workgroup_size(128)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  let res = params.gridResolution;
  let fieldCount = res.x * res.y * res.z;
  if (id >= fieldCount) {
    return;
  }

  let idx3 = unflatten3(id, res);
  let samplePos = params.gridMin + vec3<f32>(idx3) * params.cellSize;

  // Sample sits outside the simulation domain — by Müller's convention there
  // is no fluid there (the box wall is the air-fluid interface). Forcing the
  // field to 0 lets MC close the iso-surface flush against the glass.
  let outside =
    samplePos.x < params.boxMin.x || samplePos.x > params.boxMax.x ||
    samplePos.y < params.boxMin.y || samplePos.y > params.boxMax.y ||
    samplePos.z < params.boxMin.z || samplePos.z > params.boxMax.z;
  if (outside) {
    fieldValues[id] = 0.0;
    return;
  }

  let h = max(params.smoothingRadius, EPS);
  var rho = 0.0;
  for (var p: u32 = 0u; p < params.particleCount; p = p + 1u) {
    let d = samplePos - particles[p].position;
    let r2 = dot(d, d);
    rho = rho + params.particleMass * poly6_from_r2(r2, h);
  }

  // WGSL has no isFinite; exponent 0xFF in IEEE-754 binary32 means inf or NaN.
  if (((bitcast<u32>(rho) >> 23u) & 0xFFu) == 0xFFu) {
    rho = 0.0;
  }
  fieldValues[id] = rho;
}