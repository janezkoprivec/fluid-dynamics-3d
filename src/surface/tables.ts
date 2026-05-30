import { MC_CASE_TABLE } from './mcCaseTable';

export interface MarchingCubesTables {
  readonly edgeTableHost: Uint32Array;
  readonly numTrisTableHost: Uint32Array;
  readonly triTableHost: Int32Array;
  readonly edgeTableBuffer: GPUBuffer;
  readonly numTrisTableBuffer: GPUBuffer;
  readonly triTableBuffer: GPUBuffer;
  dispose(): void;
}

const CASE_COUNT = 256;
const TRI_STRIDE = 16;

export function createMarchingCubesTables(device: GPUDevice): MarchingCubesTables {
  if (MC_CASE_TABLE.length !== CASE_COUNT * TRI_STRIDE) {
    throw new Error(
      `MC_CASE_TABLE length mismatch: got ${MC_CASE_TABLE.length}, expected ${CASE_COUNT * TRI_STRIDE}`,
    );
  }

  const triTableHost = new Int32Array(MC_CASE_TABLE);
  const edgeTableHost = new Uint32Array(CASE_COUNT);
  const numTrisTableHost = new Uint32Array(CASE_COUNT);

  for (let c = 0; c < CASE_COUNT; c++) {
    let edgeMask = 0;
    let edgeCount = 0;
    const base = c * TRI_STRIDE;
    for (let i = 0; i < TRI_STRIDE; i++) {
      const e = triTableHost[base + i]!;
      if (e < 0) break;
      edgeMask |= 1 << e;
      edgeCount++;
    }
    if (edgeCount % 3 !== 0) {
      throw new Error(`Invalid tri row at case ${c}: ${edgeCount} edge refs`);
    }
    edgeTableHost[c] = edgeMask >>> 0;
    numTrisTableHost[c] = Math.floor(edgeCount / 3);
  }

  const edgeTableBuffer = createStorageBufferU32(device, 'surface/mc-edge-table', edgeTableHost);
  const numTrisTableBuffer = createStorageBufferU32(
    device,
    'surface/mc-num-tris-table',
    numTrisTableHost,
  );
  const triTableBuffer = createStorageBufferI32(device, 'surface/mc-tri-table', triTableHost);

  return {
    edgeTableHost,
    numTrisTableHost,
    triTableHost,
    edgeTableBuffer,
    numTrisTableBuffer,
    triTableBuffer,
    dispose(): void {
      edgeTableBuffer.destroy();
      numTrisTableBuffer.destroy();
      triTableBuffer.destroy();
    },
  };
}

function createStorageBufferU32(
  device: GPUDevice,
  label: string,
  data: Uint32Array,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: Math.max(4, data.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(
    buffer,
    0,
    data.buffer as ArrayBuffer,
    data.byteOffset,
    data.byteLength,
  );
  return buffer;
}

function createStorageBufferI32(
  device: GPUDevice,
  label: string,
  data: Int32Array,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: Math.max(4, data.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(
    buffer,
    0,
    data.buffer as ArrayBuffer,
    data.byteOffset,
    data.byteLength,
  );
  return buffer;
}