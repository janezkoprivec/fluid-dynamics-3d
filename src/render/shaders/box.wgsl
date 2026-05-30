// Wireframe + faint glass face renderer for the sim's bounding box.
// Two entry-point pairs: `vs_edges`/`fs_edges` for the 12 edges (line-list),
// `vs_faces`/`fs_faces` for the 6 face panels (triangle-list).

struct Camera {
  viewProj: mat4x4<f32>,
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  eye: vec3<f32>,
  _pad0: f32,
};

struct BoxParams {
  boxMin: vec3<f32>,
  edgeAlpha: f32,
  boxMax: vec3<f32>,
  faceAlpha: f32,
  edgeColor: vec3<f32>,
  _pad0: f32,
  faceColor: vec3<f32>,
  _pad1: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> params: BoxParams;

// 8-corner AABB layout: corner index bits = (x, y, z).
fn corner(idx: u32) -> vec3<f32> {
  let x = select(params.boxMin.x, params.boxMax.x, (idx & 1u) != 0u);
  let y = select(params.boxMin.y, params.boxMax.y, (idx & 2u) != 0u);
  let z = select(params.boxMin.z, params.boxMax.z, (idx & 4u) != 0u);
  return vec3<f32>(x, y, z);
}

// 12 edges as 24 vertex indices (line-list).
const EDGE_INDICES: array<u32, 24> = array<u32, 24>(
  0u, 1u,  0u, 2u,  0u, 4u,
  1u, 3u,  1u, 5u,
  2u, 3u,  2u, 6u,
  3u, 7u,
  4u, 5u,  4u, 6u,
  5u, 7u,
  6u, 7u,
);

// 6 faces as 36 vertex indices (triangle-list, 2 tris/face).
// Face winding gives outward-pointing geometric normals.
const FACE_INDICES: array<u32, 36> = array<u32, 36>(
  // -X face (0,2,4,6)
  0u, 4u, 6u,  0u, 6u, 2u,
  // +X face (1,3,5,7)
  1u, 3u, 7u,  1u, 7u, 5u,
  // -Y face (0,1,4,5)
  0u, 1u, 5u,  0u, 5u, 4u,
  // +Y face (2,3,6,7)
  2u, 6u, 7u,  2u, 7u, 3u,
  // -Z face (0,1,2,3)
  0u, 2u, 3u,  0u, 3u, 1u,
  // +Z face (4,5,6,7)
  4u, 5u, 7u,  4u, 7u, 6u,
);

struct EdgeOut {
  @builtin(position) clip: vec4<f32>,
};

@vertex
fn vs_edges(@builtin(vertex_index) vi: u32) -> EdgeOut {
  let p = corner(EDGE_INDICES[vi]);
  var out: EdgeOut;
  out.clip = camera.viewProj * vec4<f32>(p, 1.0);
  return out;
}

@fragment
fn fs_edges() -> @location(0) vec4<f32> {
  return vec4<f32>(params.edgeColor * params.edgeAlpha, params.edgeAlpha);
}

struct FaceOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
};

fn faceNormal(faceIdx: u32) -> vec3<f32> {
  // 6 faces in order matching FACE_INDICES blocks.
  switch (faceIdx) {
    case 0u: { return vec3<f32>(-1.0, 0.0, 0.0); }
    case 1u: { return vec3<f32>( 1.0, 0.0, 0.0); }
    case 2u: { return vec3<f32>(0.0, -1.0, 0.0); }
    case 3u: { return vec3<f32>(0.0,  1.0, 0.0); }
    case 4u: { return vec3<f32>(0.0, 0.0, -1.0); }
    default: { return vec3<f32>(0.0, 0.0,  1.0); }
  }
}

@vertex
fn vs_faces(@builtin(vertex_index) vi: u32) -> FaceOut {
  let p = corner(FACE_INDICES[vi]);
  let n = faceNormal(vi / 6u);
  var out: FaceOut;
  out.clip = camera.viewProj * vec4<f32>(p, 1.0);
  out.worldPos = p;
  out.normal = n;
  return out;
}

@fragment
fn fs_faces(in: FaceOut) -> @location(0) vec4<f32> {
  // Fresnel-driven tint so the glass panels look like glass, not a flat scrim.
  let V = normalize(camera.eye - in.worldPos);
  var nrm = in.normal;
  if (dot(nrm, V) < 0.0) { nrm = -nrm; }
  let NdotV = clamp(dot(nrm, V), 0.0, 1.0);
  let fres = pow(1.0 - NdotV, 4.0);
  let alpha = clamp(params.faceAlpha * (0.25 + 0.75 * fres), 0.0, 1.0);
  let rgb = params.faceColor * alpha;
  return vec4<f32>(rgb, alpha);
}
