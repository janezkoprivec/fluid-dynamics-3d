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
@group(1) @binding(2) var<storage, read> sortedEntries: array<vec2<u32>>;
@group(1) @binding(3) var<storage, read> cellStart: array<u32>;
@group(1) @binding(4) var<storage, read> cellEnd: array<u32>;

const PI: f32 = 3.141592653589793;
const EPS: f32 = 1e-6;

const EMPTY: u32 = 0xffffffffu;

fn flattenCell(c: vec3<u32>, res: vec3<u32>) -> u32 {
  return c.x + c.y * res.x + c.z * (res.x * res.y);
}

fn particleCell(pos: vec3<f32>, boxMin: vec3<f32>, h: f32, res: vec3<u32>) -> vec3<i32> {
  let raw = vec3<i32>(floor((pos - boxMin) / max(h, EPS)));
  let hi = vec3<i32>(res) - vec3<i32>(1, 1, 1);
  return clamp(raw, vec3<i32>(0, 0, 0), hi);
}

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

  let base = particleCell(pi.position, params.boxMin, h, params.gridResolution);

  var rho = 0.0;
  for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
      for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
        let nc = base + vec3<i32>(dx, dy, dz);
        if (nc.x < 0 || nc.y < 0 || nc.z < 0) { continue; }
        if (nc.x >= i32(params.gridResolution.x) ||
            nc.y >= i32(params.gridResolution.y) ||
            nc.z >= i32(params.gridResolution.z)) { continue; }
  
        let cid = flattenCell(
          vec3<u32>(u32(nc.x), u32(nc.y), u32(nc.z)),
          params.gridResolution,
        );
        let start = cellStart[cid];
        let end = cellEnd[cid];
        if (start == EMPTY || end == EMPTY || end <= start) { continue; }
  
        for (var s: u32 = start; s < end; s = s + 1u) {
          let pjIdx = sortedEntries[s].y;
          let pj = particles[pjIdx];
          let r = length(pi.position - pj.position);
          rho = rho + params.particleMass * poly6(r, h);
        }
      }
    }
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