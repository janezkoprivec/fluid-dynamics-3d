// Final pass of the multi-level scan: each workgroup adds the matching
// block-offset value into every element of its block, converting per-block
// local prefixes into the full exclusive prefix.

struct AddParams {
  n: u32,
};

@group(0) @binding(0) var<uniform> params: AddParams;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> data: array<u32>;

const BLOCK_SIZE: u32 = 256u;

@compute @workgroup_size(BLOCK_SIZE)
fn cs_main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(workgroup_id) wgid: vec3<u32>,
) {
  let i = gid.x;
  if (i >= params.n) { return; }
  data[i] = data[i] + offsets[wgid.x];
}
