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
  particleCount: u32,
  _pad0: u32,

  gridResolution: vec3<u32>,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;

fn reflectAxis(p: f32, v: f32, lo: f32, hi: f32, e: f32) -> vec2<f32> {
  var pp = p;
  var vv = v;
  if (pp < lo) {
    pp = lo + (lo - pp);
    if (vv < 0.0) {
      vv = -vv * e;
    }
  } else if (pp > hi) {
    pp = hi - (pp - hi);
    if (vv > 0.0) {
      vv = -vv * e;
    }
  }
  return vec2<f32>(pp, vv);
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.particleCount) { return; }

  var p = particles[idx];

  let accel = params.gravity;
  p.acceleration = accel;
  p.velocity = p.velocity + accel * params.dt;
  p.position = p.position + p.velocity * params.dt;

  let rx = reflectAxis(p.position.x, p.velocity.x, params.boxMin.x, params.boxMax.x, params.boundaryDamping);
  let ry = reflectAxis(p.position.y, p.velocity.y, params.boxMin.y, params.boxMax.y, params.boundaryDamping);
  let rz = reflectAxis(p.position.z, p.velocity.z, params.boxMin.z, params.boxMax.z, params.boundaryDamping);

  p.position = vec3<f32>(rx.x, ry.x, rz.x);
  p.velocity = vec3<f32>(rx.y, ry.y, rz.y);

  particles[idx] = p;
}
