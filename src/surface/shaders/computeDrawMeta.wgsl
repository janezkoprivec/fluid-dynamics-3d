// Reads the last entries of cubePrefix and cubeTriCount to compute the total
// triangle count, clamps to vertex capacity, and writes WebGPU `drawIndirect`
// args (vertexCount, instanceCount, firstVertex, firstInstance).

struct MetaParams {
  cubeCount: u32,
  vertexCapacity: u32,
};

@group(0) @binding(0) var<uniform> params: MetaParams;
@group(0) @binding(1) var<storage, read> cubeTriCount: array<u32>;
@group(0) @binding(2) var<storage, read> cubePrefix: array<u32>;
@group(0) @binding(3) var<storage, read_write> drawArgs: array<u32>;

@compute @workgroup_size(1)
fn cs_main() {
  if (params.cubeCount == 0u) {
    drawArgs[0] = 0u;
    drawArgs[1] = 1u;
    drawArgs[2] = 0u;
    drawArgs[3] = 0u;
    return;
  }
  let last = params.cubeCount - 1u;
  let totalTris = cubePrefix[last] + cubeTriCount[last];
  let totalVerts = totalTris * 3u;
  let capped = min(totalVerts, params.vertexCapacity);
  drawArgs[0] = capped;
  drawArgs[1] = 1u;
  drawArgs[2] = 0u;
  drawArgs[3] = 0u;
}
