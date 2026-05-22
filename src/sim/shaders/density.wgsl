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
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;

const PI: f32 = 3.141592653589793;
const EPS: f32 = 1e-6;

fn poly6(r: f32, h: f32) -> f32 {
  if (h <= 0.0 || r > h) {
    return 0.0;
  }
  let h2 = h * h;
  let r2 = r * r;
  let term = h2 - r2;
  let coeff = 315.0 / (64.0 * PI * pow(h, 9.0));
  return coeff * term * term * term;
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.particleCount) {
    return;
  }

  let pi = particles[i];
  let h = params.smoothingRadius;

  var rho = 0.0;
  for (var j: u32 = 0u; j < params.particleCount; j = j + 1u) {
    let pj = particles[j];
    let r = length(pi.position - pj.position);
    rho = rho + params.particleMass * poly6(r, h);
  }

  rho = max(rho, EPS);

  let rho0 = max(params.restDensity, EPS);
  let pRaw = params.gasConstant * (pow(rho / rho0, params.gamma) - 1.0);
  let pClamped = clamp(pRaw, 0.0, params.maxPressure);

  var outP = pi;
  outP.density = rho;
  outP.pressure = pClamped;
  particles[i] = outP;
}