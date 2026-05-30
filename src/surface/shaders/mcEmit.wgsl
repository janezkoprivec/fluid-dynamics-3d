// Marching-cubes triangle emission. For each cube whose case is non-trivial,
// look up the triangle table, linearly interpolate edge intersections from
// the scalar field, and write positions + gradient-based normals at the
// compact slot derived from cubePrefix.

struct EmitParams {
  gridMin: vec3<f32>,
  cellSize: f32,
  gridResolution: vec3<u32>,
  vertexCapacity: u32,
  cubeResolution: vec3<u32>,
  _pad0: u32,
  isoValue: f32,
  _pad1: f32,
  _pad2: f32,
  _pad3: f32,
};

@group(0) @binding(0) var<uniform> params: EmitParams;
@group(0) @binding(1) var<storage, read> fieldValues: array<f32>;
@group(0) @binding(2) var<storage, read> cubeCase: array<u32>;
@group(0) @binding(3) var<storage, read> cubePrefix: array<u32>;
@group(0) @binding(4) var<storage, read> triTable: array<i32>;
@group(0) @binding(5) var<storage, read_write> vertexOut: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> normalOut: array<vec4<f32>>;

const TRI_STRIDE: u32 = 16u;
const EPS: f32 = 1e-6;

// Standard Paul Bourke marching cubes corner layout:
//   v0=(0,0,0) v1=(1,0,0) v2=(1,1,0) v3=(0,1,0)
//   v4=(0,0,1) v5=(1,0,1) v6=(1,1,1) v7=(0,1,1)
// Edge index -> (cornerA, cornerB)
const EDGE_CORNER_A: array<u32, 12> = array<u32, 12>(
  0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 0u, 1u, 2u, 3u,
);
const EDGE_CORNER_B: array<u32, 12> = array<u32, 12>(
  1u, 2u, 3u, 0u, 5u, 6u, 7u, 4u, 4u, 5u, 6u, 7u,
);
const CORNER_OFF_X: array<u32, 8> = array<u32, 8>(0u, 1u, 1u, 0u, 0u, 1u, 1u, 0u);
const CORNER_OFF_Y: array<u32, 8> = array<u32, 8>(0u, 0u, 1u, 1u, 0u, 0u, 1u, 1u);
const CORNER_OFF_Z: array<u32, 8> = array<u32, 8>(0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u);

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

fn cornerOffset(corner: u32) -> vec3<u32> {
  return vec3<u32>(CORNER_OFF_X[corner], CORNER_OFF_Y[corner], CORNER_OFF_Z[corner]);
}

fn fieldAt(c: vec3<u32>) -> f32 {
  return fieldValues[flatten3(c, params.gridResolution)];
}

fn gradientAt(c: vec3<u32>) -> vec3<f32> {
  let res = params.gridResolution;
  var x0 = c.x;
  var x1 = c.x;
  if (c.x > 0u) { x0 = c.x - 1u; }
  if (c.x + 1u < res.x) { x1 = c.x + 1u; }
  var y0 = c.y;
  var y1 = c.y;
  if (c.y > 0u) { y0 = c.y - 1u; }
  if (c.y + 1u < res.y) { y1 = c.y + 1u; }
  var z0 = c.z;
  var z1 = c.z;
  if (c.z > 0u) { z0 = c.z - 1u; }
  if (c.z + 1u < res.z) { z1 = c.z + 1u; }
  let dx = fieldAt(vec3<u32>(x1, c.y, c.z)) - fieldAt(vec3<u32>(x0, c.y, c.z));
  let dy = fieldAt(vec3<u32>(c.x, y1, c.z)) - fieldAt(vec3<u32>(c.x, y0, c.z));
  let dz = fieldAt(vec3<u32>(c.x, c.y, z1)) - fieldAt(vec3<u32>(c.x, c.y, z0));
  return vec3<f32>(dx, dy, dz);
}

@compute @workgroup_size(128)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  let cres = params.cubeResolution;
  let cubeCount = cres.x * cres.y * cres.z;
  if (id >= cubeCount) { return; }

  let caseIdx = cubeCase[id];
  if (caseIdx == 0u || caseIdx == 255u) { return; }

  let c = unflatten3(id, cres);
  let baseWrite = cubePrefix[id] * 3u;
  let iso = params.isoValue;
  let cap = params.vertexCapacity;

  for (var i: u32 = 0u; i < TRI_STRIDE; i = i + 1u) {
    let e = triTable[caseIdx * TRI_STRIDE + i];
    if (e < 0) { break; }
    let eIdx = u32(e);

    let outIdx = baseWrite + i;
    if (outIdx >= cap) { break; }

    let aCorner = EDGE_CORNER_A[eIdx];
    let bCorner = EDGE_CORNER_B[eIdx];
    let cornerA = c + cornerOffset(aCorner);
    let cornerB = c + cornerOffset(bCorner);

    let pA = params.gridMin + vec3<f32>(cornerA) * params.cellSize;
    let pB = params.gridMin + vec3<f32>(cornerB) * params.cellSize;
    let fA = fieldAt(cornerA);
    let fB = fieldAt(cornerB);

    let denom = fB - fA;
    var t: f32 = 0.5;
    if (abs(denom) > EPS) { t = (iso - fA) / denom; }
    let tc = clamp(t, 0.0, 1.0);
    let pos = mix(pA, pB, tc);

    let gA = gradientAt(cornerA);
    let gB = gradientAt(cornerB);
    let grad = mix(gA, gB, tc);
    let glen = length(grad);
    var normal = vec3<f32>(0.0, 1.0, 0.0);
    if (glen > EPS) {
      // Density field: gradient points toward higher density (interior).
      // Outward surface normal is -gradient.
      normal = -grad / glen;
    }

    vertexOut[outIdx] = vec4<f32>(pos, 1.0);
    normalOut[outIdx] = vec4<f32>(normal, 0.0);
  }
}
