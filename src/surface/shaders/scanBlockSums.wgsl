// Single-workgroup exclusive prefix scan. Used at the top of the multi-level
// scan, where the remaining element count fits inside one block of BLOCK_SIZE.

struct ScanParams {
  n: u32,
};

@group(0) @binding(0) var<uniform> params: ScanParams;
@group(0) @binding(1) var<storage, read> inBuf: array<u32>;
@group(0) @binding(2) var<storage, read_write> outBuf: array<u32>;

const BLOCK_SIZE: u32 = 256u;
var<workgroup> sdata: array<u32, BLOCK_SIZE>;

@compute @workgroup_size(BLOCK_SIZE)
fn cs_main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;

  var val: u32 = 0u;
  if (tid < params.n) { val = inBuf[tid]; }
  sdata[tid] = val;
  workgroupBarrier();

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
  if (tid < params.n) {
    outBuf[tid] = excl;
  }
}
