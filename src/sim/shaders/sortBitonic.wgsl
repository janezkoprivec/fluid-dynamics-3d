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

struct SortParams {
  j: u32,
  k: u32,
  mode: u32,       // 0 = copy A->B, 1 = compare/swap on B
  paddedCount: u32, // next power of two >= particleCount
};

const INVALID: vec2<u32> = vec2<u32>(0xffffffffu, 0xffffffffu);

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;
@group(0) @binding(2) var<storage, read_write> hashEntriesA: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> hashEntriesB: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> cellStart: array<u32>;
@group(0) @binding(5) var<storage, read_write> cellEnd: array<u32>;
@group(0) @binding(6) var<uniform> sort: SortParams;

fn lessThan(a: vec2<u32>, b: vec2<u32>) -> bool {
  return (a.x < b.x) || (a.x == b.x && a.y < b.y);
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= sort.paddedCount) {
    return;
  }

  // mode 0: copy A -> B, filling padded slots with the INVALID sentinel.
  // Sentinels live in the buffer so subsequent compare/swap stages can
  // read/write them just like real entries — which preserves the bitonic
  // invariant when particleCount is not a power of two.
  if (sort.mode == 0u) {
    if (i < params.particleCount) {
      hashEntriesB[i] = hashEntriesA[i];
    } else {
      hashEntriesB[i] = INVALID;
    }
    return;
  }

  let ixj = i ^ sort.j;
  if (ixj <= i) {
    return;
  }

  let a = hashEntriesB[i];
  let b = hashEntriesB[ixj];

  let ascending = (i & sort.k) == 0u;
  var lo = a;
  var hi = b;

  if (ascending) {
    if (lessThan(b, a)) {
      lo = b;
      hi = a;
    }
  } else {
    if (lessThan(a, b)) {
      lo = b;
      hi = a;
    }
  }

  hashEntriesB[i] = lo;
  hashEntriesB[ixj] = hi;
}