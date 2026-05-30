struct McClassifyParams {
  gridResolution: vec3<u32>,
  _pad0: u32,
  cubeResolution: vec3<u32>,
  _pad1: u32,
  isoValue: f32,
  _pad2: f32,
  _pad3: f32,
  _pad4: f32,
};

@group(0) @binding(0) var<uniform> params: McClassifyParams;
@group(0) @binding(1) var<storage, read> fieldValues: array<f32>;
@group(0) @binding(2) var<storage, read> numTrisTable: array<u32>;
@group(0) @binding(3) var<storage, read_write> cubeCase: array<u32>;
@group(0) @binding(4) var<storage, read_write> cubeTriCount: array<u32>;

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

fn fieldAt(v: vec3<u32>) -> f32 {
  return fieldValues[flatten3(v, params.gridResolution)];
}

@compute @workgroup_size(128)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  let cres = params.cubeResolution;
  let cubeCount = cres.x * cres.y * cres.z;
  if (id >= cubeCount) {
    return;
  }

  let c = unflatten3(id, cres);
  let v0 = c + vec3<u32>(0u, 0u, 0u);
  let v1 = c + vec3<u32>(1u, 0u, 0u);
  let v2 = c + vec3<u32>(1u, 1u, 0u);
  let v3 = c + vec3<u32>(0u, 1u, 0u);
  let v4 = c + vec3<u32>(0u, 0u, 1u);
  let v5 = c + vec3<u32>(1u, 0u, 1u);
  let v6 = c + vec3<u32>(1u, 1u, 1u);
  let v7 = c + vec3<u32>(0u, 1u, 1u);

  var mask: u32 = 0u;
  if (fieldAt(v0) >= params.isoValue) { mask = mask | 1u; }
  if (fieldAt(v1) >= params.isoValue) { mask = mask | 2u; }
  if (fieldAt(v2) >= params.isoValue) { mask = mask | 4u; }
  if (fieldAt(v3) >= params.isoValue) { mask = mask | 8u; }
  if (fieldAt(v4) >= params.isoValue) { mask = mask | 16u; }
  if (fieldAt(v5) >= params.isoValue) { mask = mask | 32u; }
  if (fieldAt(v6) >= params.isoValue) { mask = mask | 64u; }
  if (fieldAt(v7) >= params.isoValue) { mask = mask | 128u; }

  cubeCase[id] = mask;
  cubeTriCount[id] = numTrisTable[mask];
}