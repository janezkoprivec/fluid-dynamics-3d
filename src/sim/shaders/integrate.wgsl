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

// Boundary handling. Within `range` of either wall on this axis:
//   - quadratic outward repulsion (cures the SPH wall kernel-deficit that
//     would otherwise trap particles in corners)
//   - linear normal-velocity damping opposing motion INTO the wall (kills
//     the spring-bounce that pure repulsion produces). The damper only
//     fires when the velocity component points toward the wall, so it can
//     never push a particle back into the boundary.
fn wallAxis(p: f32, v: f32, lo: f32, hi: f32, range: f32, k: f32, c: f32) -> f32 {
  if (range <= 0.0) { return 0.0; }
  var a: f32 = 0.0;
  let dLo = p - lo;
  if (dLo > 0.0 && dLo < range) {
    let t = (range - dLo) / range;
    a = a + k * t * t;
    if (v < 0.0) {
      a = a - c * v * t;
    }
  }
  let dHi = hi - p;
  if (dHi > 0.0 && dHi < range) {
    let t = (range - dHi) / range;
    a = a - k * t * t;
    if (v > 0.0) {
      a = a - c * v * t;
    }
  }
  return a;
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.particleCount) { return; }

  var p = particles[idx];

  let range = params.smoothingRadius * params.wallRange;
  let k = params.wallRepulsion;
  let c = params.wallDamping;
  let wallAccel = vec3<f32>(
    wallAxis(p.position.x, p.velocity.x, params.boxMin.x, params.boxMax.x, range, k, c),
    wallAxis(p.position.y, p.velocity.y, params.boxMin.y, params.boxMax.y, range, k, c),
    wallAxis(p.position.z, p.velocity.z, params.boxMin.z, params.boxMax.z, range, k, c),
  );

  let accel = p.acceleration + wallAccel;
  p.velocity = p.velocity + accel * params.dt;
  p.position = p.position + p.velocity * params.dt;

  let rx = reflectAxis(p.position.x, p.velocity.x, params.boxMin.x, params.boxMax.x, params.boundaryDamping);
  let ry = reflectAxis(p.position.y, p.velocity.y, params.boxMin.y, params.boxMax.y, params.boundaryDamping);
  let rz = reflectAxis(p.position.z, p.velocity.z, params.boxMin.z, params.boxMax.z, params.boundaryDamping);

  p.position = vec3<f32>(rx.x, ry.x, rz.x);
  p.velocity = vec3<f32>(rx.y, ry.y, rz.y);

  particles[idx] = p;
}
