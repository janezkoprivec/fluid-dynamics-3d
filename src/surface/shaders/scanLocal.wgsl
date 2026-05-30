// Per-block exclusive prefix scan. Each workgroup scans BLOCK_SIZE elements
// from inBuf, writes block-local exclusive prefixes to outBuf, and writes
// the total of its block into blockSums[workgroup_id].
//
// `params.n` is the logical element count. Workgroups past n/BLOCK still
// execute; threads with global index >= n contribute 0 and skip writes.

struct ScanParams {
  n: u32,
};

@group(0) @binding(0) var<uniform> params: ScanParams;
@group(0) @binding(1) var<storage, read> inBuf: array<u32>;
@group(0) @binding(2) var<storage, read_write> outBuf: array<u32>;
@group(0) @binding(3) var<storage, read_write> blockSums: array<u32>;

const BLOCK_SIZE: u32 = 256u;
var<workgroup> sdata: array<u32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn cs_main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wgid: vec3<u32>,
) {
  let i = gid.x;
  let tid = lid.x;

  var val: u32 = 0u;
  if (i < params.n) { val = inBuf[i]; }
  sdata[tid] = val;
  workgroupBarrier();

  // Hillis-Steele inclusive scan
  var offset: u32 = 1u;
  loop {
    if (offset >= BLOCK_SIZE) { break; }
    var v: u32 = 0u;
    if (tid >= offset) { v = sdata[tid - offset]; }
    workgroupBarrier();
    sdata[tid] = sdata[tid] + v;
    workgroupBarrier();
    offset = offset * 2u;
  }

  let excl = sdata[tid] - val;
  if (i < params.n) {
    outBuf[i] = excl;
  }

  if (tid == BLOCK_SIZE - 1u) {
    blockSums[wgid.x] = sdata[tid];
  }
}
