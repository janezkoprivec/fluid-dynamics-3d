import type { ParticleAllocation } from '../sim/particles';
import { SIM_BOX_MAX, SIM_BOX_MIN } from '../sim';
import {
  DEFAULT_SURFACE_PARAMS,
  type SurfaceCounters,
  type SurfaceFieldStats,
  type SurfaceGridResolution,
  type SurfaceMeshBindingResources,
  type SurfaceParams,
  type SurfacePipeline,
  type SurfaceResources,
  type SurfaceSimParamsView,
} from './types';
import fieldBuildWgsl from './shaders/fieldBuild.wgsl?raw';
import mcClassifyWgsl from './shaders/mcClassify.wgsl?raw';
import scanLocalWgsl from './shaders/scanLocal.wgsl?raw';
import scanBlockSumsWgsl from './shaders/scanBlockSums.wgsl?raw';
import scanAddOffsetsWgsl from './shaders/scanAddOffsets.wgsl?raw';
import computeDrawMetaWgsl from './shaders/computeDrawMeta.wgsl?raw';
import mcEmitWgsl from './shaders/mcEmit.wgsl?raw';
import { createMarchingCubesTables } from './tables';

const FIELD_PARAMS_BYTE_SIZE = 80;
const MC_CLASSIFY_PARAMS_BYTE_SIZE = 48;
const EMIT_PARAMS_BYTE_SIZE = 64;
const SCAN_PARAMS_BYTE_SIZE = 16;
const ADD_OFFSETS_PARAMS_BYTE_SIZE = 16;
const DRAW_META_PARAMS_BYTE_SIZE = 16;
const DRAW_INDIRECT_BYTE_SIZE = 16;

const FIELD_BUILD_WG_SIZE = 128;
const MC_CLASSIFY_WG_SIZE = 128;
const MC_EMIT_WG_SIZE = 128;
const SCAN_BLOCK_SIZE = 256;

type ParticleBinding = Pick<ParticleAllocation, 'count' | 'gpuBuffer'>;

interface ScanLocalStage {
  kind: 'local';
  n: number;
  numBlocks: number;
  paramsBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
}

interface ScanTopStage {
  kind: 'top';
  n: number;
  paramsBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
}

type ScanStage = ScanLocalStage | ScanTopStage;

interface AddOffsetsStage {
  n: number;
  paramsBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
}

interface ScanResources {
  cubePrefix: GPUBuffer;
  intermediates: GPUBuffer[];
  stages: ScanStage[];
  addStages: AddOffsetsStage[];
}

export function createSurfacePipeline(
  device: GPUDevice,
  initialAlloc: ParticleAllocation,
  simParamsView: () => SurfaceSimParamsView,
): SurfacePipeline {
  let alloc: ParticleBinding = initialAlloc;
  let disposed = false;
  const params: SurfaceParams = { ...DEFAULT_SURFACE_PARAMS };
  let simParams: SurfaceSimParamsView = { ...simParamsView() };

  // ---------- Field-params uniform ----------
  const fieldParamsHost = new ArrayBuffer(FIELD_PARAMS_BYTE_SIZE);
  const fieldParamsF32 = new Float32Array(fieldParamsHost);
  const fieldParamsU32 = new Uint32Array(fieldParamsHost);
  const fieldParamsBuffer = device.createBuffer({
    label: 'surface/field-params',
    size: FIELD_PARAMS_BYTE_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // ---------- MC classify params uniform ----------
  const mcClassifyParamsHost = new ArrayBuffer(MC_CLASSIFY_PARAMS_BYTE_SIZE);
  const mcClassifyParamsF32 = new Float32Array(mcClassifyParamsHost);
  const mcClassifyParamsU32 = new Uint32Array(mcClassifyParamsHost);
  const mcClassifyParamsBuffer = device.createBuffer({
    label: 'surface/mc-classify-params',
    size: MC_CLASSIFY_PARAMS_BYTE_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // ---------- Emit params uniform ----------
  const emitParamsHost = new ArrayBuffer(EMIT_PARAMS_BYTE_SIZE);
  const emitParamsF32 = new Float32Array(emitParamsHost);
  const emitParamsU32 = new Uint32Array(emitParamsHost);
  const emitParamsBuffer = device.createBuffer({
    label: 'surface/emit-params',
    size: EMIT_PARAMS_BYTE_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // ---------- Draw-meta params uniform ----------
  const drawMetaParamsHost = new ArrayBuffer(DRAW_META_PARAMS_BYTE_SIZE);
  const drawMetaParamsU32 = new Uint32Array(drawMetaParamsHost);
  const drawMetaParamsBuffer = device.createBuffer({
    label: 'surface/draw-meta-params',
    size: DRAW_META_PARAMS_BYTE_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // ---------- Marching-cubes lookup tables (static) ----------
  const mcTables = createMarchingCubesTables(device);

  // ---------- Shader modules ----------
  const fieldBuildModule = device.createShaderModule({
    label: 'surface/field-build.wgsl',
    code: fieldBuildWgsl,
  });
  const mcClassifyModule = device.createShaderModule({
    label: 'surface/mc-classify.wgsl',
    code: mcClassifyWgsl,
  });
  const scanLocalModule = device.createShaderModule({
    label: 'surface/scan-local.wgsl',
    code: scanLocalWgsl,
  });
  const scanBlockSumsModule = device.createShaderModule({
    label: 'surface/scan-block-sums.wgsl',
    code: scanBlockSumsWgsl,
  });
  const scanAddOffsetsModule = device.createShaderModule({
    label: 'surface/scan-add-offsets.wgsl',
    code: scanAddOffsetsWgsl,
  });
  const computeDrawMetaModule = device.createShaderModule({
    label: 'surface/compute-draw-meta.wgsl',
    code: computeDrawMetaWgsl,
  });
  const mcEmitModule = device.createShaderModule({
    label: 'surface/mc-emit.wgsl',
    code: mcEmitWgsl,
  });

  // ---------- Bind group layouts ----------
  const fieldBuildBgl = device.createBindGroupLayout({
    label: 'surface/field-build-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const mcClassifyBgl = device.createBindGroupLayout({
    label: 'surface/mc-classify-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const scanLocalBgl = device.createBindGroupLayout({
    label: 'surface/scan-local-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const scanTopBgl = device.createBindGroupLayout({
    label: 'surface/scan-top-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const addOffsetsBgl = device.createBindGroupLayout({
    label: 'surface/add-offsets-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const drawMetaBgl = device.createBindGroupLayout({
    label: 'surface/draw-meta-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const emitBgl = device.createBindGroupLayout({
    label: 'surface/emit-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });

  // ---------- Compute pipelines ----------
  const fieldBuildPipeline = device.createComputePipeline({
    label: 'surface/field-build-pipeline',
    layout: device.createPipelineLayout({
      label: 'surface/field-build-layout',
      bindGroupLayouts: [fieldBuildBgl],
    }),
    compute: { module: fieldBuildModule, entryPoint: 'cs_main' },
  });
  const mcClassifyPipeline = device.createComputePipeline({
    label: 'surface/mc-classify-pipeline',
    layout: device.createPipelineLayout({
      label: 'surface/mc-classify-layout',
      bindGroupLayouts: [mcClassifyBgl],
    }),
    compute: { module: mcClassifyModule, entryPoint: 'cs_main' },
  });
  const scanLocalPipeline = device.createComputePipeline({
    label: 'surface/scan-local-pipeline',
    layout: device.createPipelineLayout({
      label: 'surface/scan-local-layout',
      bindGroupLayouts: [scanLocalBgl],
    }),
    compute: { module: scanLocalModule, entryPoint: 'cs_main' },
  });
  const scanTopPipeline = device.createComputePipeline({
    label: 'surface/scan-top-pipeline',
    layout: device.createPipelineLayout({
      label: 'surface/scan-top-layout',
      bindGroupLayouts: [scanTopBgl],
    }),
    compute: { module: scanBlockSumsModule, entryPoint: 'cs_main' },
  });
  const addOffsetsPipeline = device.createComputePipeline({
    label: 'surface/add-offsets-pipeline',
    layout: device.createPipelineLayout({
      label: 'surface/add-offsets-layout',
      bindGroupLayouts: [addOffsetsBgl],
    }),
    compute: { module: scanAddOffsetsModule, entryPoint: 'cs_main' },
  });
  const drawMetaPipeline = device.createComputePipeline({
    label: 'surface/draw-meta-pipeline',
    layout: device.createPipelineLayout({
      label: 'surface/draw-meta-layout',
      bindGroupLayouts: [drawMetaBgl],
    }),
    compute: { module: computeDrawMetaModule, entryPoint: 'cs_main' },
  });
  const emitPipeline = device.createComputePipeline({
    label: 'surface/emit-pipeline',
    layout: device.createPipelineLayout({
      label: 'surface/emit-layout',
      bindGroupLayouts: [emitBgl],
    }),
    compute: { module: mcEmitModule, entryPoint: 'cs_main' },
  });

  // ---------- Grid-sized resources ----------
  let fieldValueCount = gridVertexCount(params.gridResolution);
  let fieldValues = createStorageBuffer('surface/field-values', fieldValueCount * 4);
  let cubeResolution = cubeResolutionFromGrid(params.gridResolution);
  let cubeCount = gridCubeCount(cubeResolution);
  let cubeCase = createStorageBuffer('surface/cube-case', cubeCount * 4);
  let cubeTriCount = createStorageBuffer('surface/cube-tri-count', cubeCount * 4);

  // ---------- Scan resources (rebuilt with grid) ----------
  let scan: ScanResources = buildScanResources(cubeCount);

  // ---------- Emit / mesh resources (rebuilt with grid OR maxTriangles) ----------
  let vertexCapacity = Math.max(3, params.maxTriangles * 3);
  let vertexBuffer = createStorageBuffer(
    'surface/vertex-out',
    vertexCapacity * 16,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  let normalBuffer = createStorageBuffer(
    'surface/normal-out',
    vertexCapacity * 16,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const drawIndirectBuffer = device.createBuffer({
    label: 'surface/draw-indirect',
    size: DRAW_INDIRECT_BYTE_SIZE,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.INDIRECT |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(
    drawIndirectBuffer,
    0,
    new Uint32Array([0, 1, 0, 0]),
  );

  // ---------- Bind groups (rebuilt on resource changes) ----------
  let fieldBuildBindGroup = createFieldBuildBindGroup();
  let mcClassifyBindGroup = createMcClassifyBindGroup();
  let drawMetaBindGroup = createDrawMetaBindGroup();
  let emitBindGroup = createEmitBindGroup();

  function createFieldBuildBindGroup(): GPUBindGroup {
    return device.createBindGroup({
      label: 'surface/field-build-bg',
      layout: fieldBuildBgl,
      entries: [
        { binding: 0, resource: { buffer: alloc.gpuBuffer } },
        { binding: 1, resource: { buffer: fieldParamsBuffer } },
        { binding: 2, resource: { buffer: fieldValues } },
      ],
    });
  }
  function createMcClassifyBindGroup(): GPUBindGroup {
    return device.createBindGroup({
      label: 'surface/mc-classify-bg',
      layout: mcClassifyBgl,
      entries: [
        { binding: 0, resource: { buffer: mcClassifyParamsBuffer } },
        { binding: 1, resource: { buffer: fieldValues } },
        { binding: 2, resource: { buffer: mcTables.numTrisTableBuffer } },
        { binding: 3, resource: { buffer: cubeCase } },
        { binding: 4, resource: { buffer: cubeTriCount } },
      ],
    });
  }
  function createDrawMetaBindGroup(): GPUBindGroup {
    return device.createBindGroup({
      label: 'surface/draw-meta-bg',
      layout: drawMetaBgl,
      entries: [
        { binding: 0, resource: { buffer: drawMetaParamsBuffer } },
        { binding: 1, resource: { buffer: cubeTriCount } },
        { binding: 2, resource: { buffer: scan.cubePrefix } },
        { binding: 3, resource: { buffer: drawIndirectBuffer } },
      ],
    });
  }
  function createEmitBindGroup(): GPUBindGroup {
    return device.createBindGroup({
      label: 'surface/emit-bg',
      layout: emitBgl,
      entries: [
        { binding: 0, resource: { buffer: emitParamsBuffer } },
        { binding: 1, resource: { buffer: fieldValues } },
        { binding: 2, resource: { buffer: cubeCase } },
        { binding: 3, resource: { buffer: scan.cubePrefix } },
        { binding: 4, resource: { buffer: mcTables.triTableBuffer } },
        { binding: 5, resource: { buffer: vertexBuffer } },
        { binding: 6, resource: { buffer: normalBuffer } },
      ],
    });
  }

  function buildScanResources(count: number): ScanResources {
    const cubePrefix = createStorageBuffer(
      'surface/cube-prefix',
      Math.max(1, count) * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    const intermediates: GPUBuffer[] = [];
    const stages: ScanStage[] = [];
    const addStages: AddOffsetsStage[] = [];

    if (count <= 1) {
      // Trivial: exclusive prefix of a 0/1-element array is just [0].
      device.queue.writeBuffer(cubePrefix, 0, new Uint32Array([0]));
      return { cubePrefix, intermediates, stages, addStages };
    }

    // Plan element counts at each level (n_k).
    const ns: number[] = [];
    {
      let n = count;
      while (true) {
        ns.push(n);
        const nb = Math.ceil(n / SCAN_BLOCK_SIZE);
        if (nb === 1) break;
        n = nb;
      }
    }

    // Allocate the per-level prefix outputs. outputs[0] is the externally
    // visible cubePrefix; later outputs are scratch intermediates.
    const outputs: GPUBuffer[] = [cubePrefix];
    for (let k = 1; k < ns.length; k++) {
      const buf = createStorageBuffer(
        `surface/scan-out-${k}`,
        ns[k]! * 4,
      );
      outputs.push(buf);
      intermediates.push(buf);
    }

    // Allocate blockSums buffers. bsBuf[k] (size ns[k+1]) is emitted by
    // level k and consumed as input to level k+1.
    const bsBuf: GPUBuffer[] = [];
    for (let k = 0; k < ns.length - 1; k++) {
      const buf = createStorageBuffer(
        `surface/scan-bs-${k}`,
        ns[k + 1]! * 4,
      );
      bsBuf.push(buf);
      intermediates.push(buf);
    }

    // Build the forward scan stages.
    for (let k = 0; k < ns.length; k++) {
      const n = ns[k]!;
      const inputBuf = k === 0 ? cubeTriCount : bsBuf[k - 1]!;
      const outputBuf = outputs[k]!;
      const numBlocks = Math.ceil(n / SCAN_BLOCK_SIZE);
      const paramsBuffer = device.createBuffer({
        label: `surface/scan-params-${k}`,
        size: SCAN_PARAMS_BYTE_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([n, 0, 0, 0]));

      if (numBlocks === 1) {
        const bindGroup = device.createBindGroup({
          label: `surface/scan-top-bg-${k}`,
          layout: scanTopBgl,
          entries: [
            { binding: 0, resource: { buffer: paramsBuffer } },
            { binding: 1, resource: { buffer: inputBuf } },
            { binding: 2, resource: { buffer: outputBuf } },
          ],
        });
        stages.push({ kind: 'top', n, paramsBuffer, bindGroup });
      } else {
        const blockSumsBuf = bsBuf[k]!;
        const bindGroup = device.createBindGroup({
          label: `surface/scan-local-bg-${k}`,
          layout: scanLocalBgl,
          entries: [
            { binding: 0, resource: { buffer: paramsBuffer } },
            { binding: 1, resource: { buffer: inputBuf } },
            { binding: 2, resource: { buffer: outputBuf } },
            { binding: 3, resource: { buffer: blockSumsBuf } },
          ],
        });
        stages.push({ kind: 'local', n, numBlocks, paramsBuffer, bindGroup });
      }
    }

    // addOffsets passes, top-down: outputs[k-1] += outputs[k][blockIdx].
    for (let k = ns.length - 1; k >= 1; k--) {
      const data = outputs[k - 1]!;
      const offsets = outputs[k]!;
      const n = ns[k - 1]!;
      const paramsBuffer = device.createBuffer({
        label: `surface/add-offsets-params-${k}`,
        size: ADD_OFFSETS_PARAMS_BYTE_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([n, 0, 0, 0]));
      const bindGroup = device.createBindGroup({
        label: `surface/add-offsets-bg-${k}`,
        layout: addOffsetsBgl,
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: offsets } },
          { binding: 2, resource: { buffer: data } },
        ],
      });
      addStages.push({ n, paramsBuffer, bindGroup });
    }

    return { cubePrefix, intermediates, stages, addStages };
  }

  function disposeScan(s: ScanResources): void {
    s.cubePrefix.destroy();
    for (const b of s.intermediates) b.destroy();
    for (const st of s.stages) st.paramsBuffer.destroy();
    for (const st of s.addStages) st.paramsBuffer.destroy();
  }

  function rebuildGridSizedBuffersIfNeeded(prev: SurfaceGridResolution): void {
    if (sameResolution(prev, params.gridResolution)) return;
    fieldValueCount = gridVertexCount(params.gridResolution);
    cubeResolution = cubeResolutionFromGrid(params.gridResolution);
    cubeCount = gridCubeCount(cubeResolution);
    fieldValues.destroy();
    cubeCase.destroy();
    cubeTriCount.destroy();
    fieldValues = createStorageBuffer('surface/field-values', fieldValueCount * 4);
    cubeCase = createStorageBuffer('surface/cube-case', cubeCount * 4);
    cubeTriCount = createStorageBuffer('surface/cube-tri-count', cubeCount * 4);
    disposeScan(scan);
    scan = buildScanResources(cubeCount);
    rebuildAllBindGroups();
  }

  function rebuildVertexBuffersIfNeeded(prevMax: number): void {
    const target = Math.max(3, params.maxTriangles * 3);
    if (target === vertexCapacity && prevMax === params.maxTriangles) return;
    vertexCapacity = target;
    vertexBuffer.destroy();
    normalBuffer.destroy();
    vertexBuffer = createStorageBuffer(
      'surface/vertex-out',
      vertexCapacity * 16,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    );
    normalBuffer = createStorageBuffer(
      'surface/normal-out',
      vertexCapacity * 16,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    );
    emitBindGroup = createEmitBindGroup();
  }

  function rebuildAllBindGroups(): void {
    fieldBuildBindGroup = createFieldBuildBindGroup();
    mcClassifyBindGroup = createMcClassifyBindGroup();
    drawMetaBindGroup = createDrawMetaBindGroup();
    emitBindGroup = createEmitBindGroup();
  }

  function refreshSimParamsFromView(): void {
    simParams = { ...simParams, ...simParamsView() };
  }

  function writeFieldParams(): void {
    const [nx, ny, nz] = params.gridResolution;
    fieldParamsF32[0] = params.gridMin[0];
    fieldParamsF32[1] = params.gridMin[1];
    fieldParamsF32[2] = params.gridMin[2];
    fieldParamsF32[3] = params.cellSize;
    fieldParamsU32[4] = nx >>> 0;
    fieldParamsU32[5] = ny >>> 0;
    fieldParamsU32[6] = nz >>> 0;
    // Use the live `simParams.particleCount` (set by the view-getter to
     // sim.activeCount), not the buffer capacity — otherwise the field-build
     // loop runs over stashed/inactive slots every frame for no reason.
    fieldParamsU32[7] = (simParams.particleCount | 0) >>> 0;
    fieldParamsF32[8] = simParams.smoothingRadius;
    fieldParamsF32[9] = simParams.particleMass;
    fieldParamsF32[10] = simParams.restDensity;
    fieldParamsF32[11] = 0;
    fieldParamsF32[12] = SIM_BOX_MIN[0];
    fieldParamsF32[13] = SIM_BOX_MIN[1];
    fieldParamsF32[14] = SIM_BOX_MIN[2];
    fieldParamsF32[15] = 0;
    fieldParamsF32[16] = SIM_BOX_MAX[0];
    fieldParamsF32[17] = SIM_BOX_MAX[1];
    fieldParamsF32[18] = SIM_BOX_MAX[2];
    fieldParamsF32[19] = 0;
    device.queue.writeBuffer(fieldParamsBuffer, 0, fieldParamsHost);
  }

  function writeMcClassifyParams(): void {
    const [nx, ny, nz] = params.gridResolution;
    const [cx, cy, cz] = cubeResolution;
    mcClassifyParamsU32[0] = nx >>> 0;
    mcClassifyParamsU32[1] = ny >>> 0;
    mcClassifyParamsU32[2] = nz >>> 0;
    mcClassifyParamsU32[3] = 0;
    mcClassifyParamsU32[4] = cx >>> 0;
    mcClassifyParamsU32[5] = cy >>> 0;
    mcClassifyParamsU32[6] = cz >>> 0;
    mcClassifyParamsU32[7] = 0;
    mcClassifyParamsF32[8] = params.isoValue;
    mcClassifyParamsF32[9] = 0;
    mcClassifyParamsF32[10] = 0;
    mcClassifyParamsF32[11] = 0;
    device.queue.writeBuffer(mcClassifyParamsBuffer, 0, mcClassifyParamsHost);
  }

  function writeEmitParams(): void {
    const [nx, ny, nz] = params.gridResolution;
    const [cx, cy, cz] = cubeResolution;
    emitParamsF32[0] = params.gridMin[0];
    emitParamsF32[1] = params.gridMin[1];
    emitParamsF32[2] = params.gridMin[2];
    emitParamsF32[3] = params.cellSize;
    emitParamsU32[4] = nx >>> 0;
    emitParamsU32[5] = ny >>> 0;
    emitParamsU32[6] = nz >>> 0;
    emitParamsU32[7] = vertexCapacity >>> 0;
    emitParamsU32[8] = cx >>> 0;
    emitParamsU32[9] = cy >>> 0;
    emitParamsU32[10] = cz >>> 0;
    emitParamsU32[11] = 0;
    emitParamsF32[12] = params.isoValue;
    emitParamsF32[13] = 0;
    emitParamsF32[14] = 0;
    emitParamsF32[15] = 0;
    device.queue.writeBuffer(emitParamsBuffer, 0, emitParamsHost);
  }

  function writeDrawMetaParams(): void {
    drawMetaParamsU32[0] = cubeCount >>> 0;
    drawMetaParamsU32[1] = vertexCapacity >>> 0;
    drawMetaParamsU32[2] = 0;
    drawMetaParamsU32[3] = 0;
    device.queue.writeBuffer(drawMetaParamsBuffer, 0, drawMetaParamsHost);
  }

  function meshBinding(): SurfaceMeshBindingResources {
    return {
      vertexBuffer,
      normalBuffer,
      drawIndirectBuffer,
      vertexCapacity,
    };
  }

  return {
    get params(): SurfaceParams {
      return params;
    },

    update(encoder: GPUCommandEncoder): void {
      if (disposed || !params.enabled) return;
      if (alloc.count === 0 || cubeCount === 0) {
        // Make sure indirect args don't fire stale geometry.
        device.queue.writeBuffer(
          drawIndirectBuffer,
          0,
          new Uint32Array([0, 1, 0, 0]),
        );
        return;
      }

      refreshSimParamsFromView();
      // (Don't override simParams.particleCount with alloc.count here —
      // the view-getter is the source of truth, and it returns activeCount.)
      writeFieldParams();
      writeMcClassifyParams();
      writeEmitParams();
      writeDrawMetaParams();

      const pass = encoder.beginComputePass({ label: 'surface/update' });

      // 1. field build
      pass.setPipeline(fieldBuildPipeline);
      pass.setBindGroup(0, fieldBuildBindGroup);
      pass.dispatchWorkgroups(
        Math.ceil(fieldValueCount / FIELD_BUILD_WG_SIZE),
      );

      // 2. mc classify
      pass.setPipeline(mcClassifyPipeline);
      pass.setBindGroup(0, mcClassifyBindGroup);
      pass.dispatchWorkgroups(
        Math.ceil(cubeCount / MC_CLASSIFY_WG_SIZE),
      );

      // 3. scan stages (forward)
      for (const stage of scan.stages) {
        if (stage.kind === 'local') {
          pass.setPipeline(scanLocalPipeline);
          pass.setBindGroup(0, stage.bindGroup);
          pass.dispatchWorkgroups(stage.numBlocks);
        } else {
          pass.setPipeline(scanTopPipeline);
          pass.setBindGroup(0, stage.bindGroup);
          pass.dispatchWorkgroups(1);
        }
      }

      // 4. addOffsets passes (backward, top-down)
      for (const addStage of scan.addStages) {
        pass.setPipeline(addOffsetsPipeline);
        pass.setBindGroup(0, addStage.bindGroup);
        pass.dispatchWorkgroups(Math.ceil(addStage.n / SCAN_BLOCK_SIZE));
      }

      // 5. write indirect draw args
      pass.setPipeline(drawMetaPipeline);
      pass.setBindGroup(0, drawMetaBindGroup);
      pass.dispatchWorkgroups(1);

      // 6. emit triangles
      pass.setPipeline(emitPipeline);
      pass.setBindGroup(0, emitBindGroup);
      pass.dispatchWorkgroups(Math.ceil(cubeCount / MC_EMIT_WG_SIZE));

      pass.end();
    },

    rebindParticles(next): void {
      alloc = next;
      // particleCount stays as whatever the view-getter says — don't
      // overwrite it with the buffer capacity here.
      fieldBuildBindGroup = createFieldBuildBindGroup();
    },

    setParams(patch): void {
      const prevGrid = params.gridResolution;
      const prevMax = params.maxTriangles;
      Object.assign(params, patch);
      params.gridResolution = sanitizeResolution(params.gridResolution);
      params.cellSize = Math.max(1e-4, params.cellSize);
      params.isoValue = Number.isFinite(params.isoValue)
        ? params.isoValue
        : DEFAULT_SURFACE_PARAMS.isoValue;
      params.maxTriangles = Math.max(
        1024,
        Math.floor(params.maxTriangles || DEFAULT_SURFACE_PARAMS.maxTriangles),
      );

      rebuildGridSizedBuffersIfNeeded(prevGrid);
      rebuildVertexBuffersIfNeeded(prevMax);
    },

    setSimParams(patch): void {
      Object.assign(simParams, patch);
    },

    resources(): SurfaceResources {
      return {
        params: { ...params },
        particleCount: alloc.count,
        particleBuffer: alloc.gpuBuffer,
        fieldValues,
        fieldValueCount,
        cubeResolution,
        cubeCount,
        cubeCase,
        cubeTriCount,
        cubePrefix: scan.cubePrefix,
        mesh: meshBinding(),
        simParams: { ...simParams },
      };
    },

    meshBinding,

    async readbackFieldStats(): Promise<SurfaceFieldStats> {
      const bytes = Math.max(4, fieldValueCount * 4);
      const staging = device.createBuffer({
        label: 'surface/field-values-readback',
        size: bytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = device.createCommandEncoder({
        label: 'surface/field-values-readback',
      });
      encoder.copyBufferToBuffer(fieldValues, 0, staging, 0, bytes);
      device.queue.submit([encoder.finish()]);

      await staging.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(staging.getMappedRange().slice(0));
      let min = Number.POSITIVE_INFINITY;
      let max = Number.NEGATIVE_INFINITY;
      let nonFiniteCount = 0;
      for (let i = 0; i < values.length; i++) {
        const v = values[i]!;
        if (!Number.isFinite(v)) {
          nonFiniteCount++;
          continue;
        }
        if (v < min) min = v;
        if (v > max) max = v;
      }
      staging.unmap();
      staging.destroy();
      if (!Number.isFinite(min)) min = 0;
      if (!Number.isFinite(max)) max = 0;
      return { min, max, nonFiniteCount, sampleCount: values.length };
    },

    async readbackCounters(): Promise<SurfaceCounters> {
      // Approximate: read drawIndirect (vertexCount = capped triangle count * 3)
      // and last cubePrefix + last cubeTriCount to get the raw total triangles.
      const drawStaging = device.createBuffer({
        label: 'surface/draw-indirect-readback',
        size: DRAW_INDIRECT_BYTE_SIZE,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const prefixStaging = device.createBuffer({
        label: 'surface/prefix-tail-readback',
        size: 8,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = device.createCommandEncoder({
        label: 'surface/counters-readback',
      });
      encoder.copyBufferToBuffer(
        drawIndirectBuffer,
        0,
        drawStaging,
        0,
        DRAW_INDIRECT_BYTE_SIZE,
      );
      if (cubeCount > 0) {
        const tailOffset = (cubeCount - 1) * 4;
        encoder.copyBufferToBuffer(
          scan.cubePrefix,
          tailOffset,
          prefixStaging,
          0,
          4,
        );
        encoder.copyBufferToBuffer(
          cubeTriCount,
          tailOffset,
          prefixStaging,
          4,
          4,
        );
      }
      device.queue.submit([encoder.finish()]);

      await Promise.all([
        drawStaging.mapAsync(GPUMapMode.READ),
        prefixStaging.mapAsync(GPUMapMode.READ),
      ]);
      const drawView = new Uint32Array(drawStaging.getMappedRange().slice(0));
      const prefixView = new Uint32Array(prefixStaging.getMappedRange().slice(0));
      const cappedVertexCount = drawView[0] ?? 0;
      const lastPrefix = prefixView[0] ?? 0;
      const lastCount = prefixView[1] ?? 0;
      const totalTriangles = cubeCount > 0 ? lastPrefix + lastCount : 0;
      drawStaging.unmap();
      prefixStaging.unmap();
      drawStaging.destroy();
      prefixStaging.destroy();
      return {
        totalTriangles,
        cappedVertexCount,
        capacity: vertexCapacity,
      };
    },

    dispose(): void {
      if (disposed) return;
      disposed = true;
      fieldValues.destroy();
      cubeCase.destroy();
      cubeTriCount.destroy();
      vertexBuffer.destroy();
      normalBuffer.destroy();
      drawIndirectBuffer.destroy();
      fieldParamsBuffer.destroy();
      mcClassifyParamsBuffer.destroy();
      emitParamsBuffer.destroy();
      drawMetaParamsBuffer.destroy();
      disposeScan(scan);
      mcTables.dispose();
    },
  };

  function createStorageBuffer(
    label: string,
    bytes: number,
    usage: GPUBufferUsageFlags = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  ): GPUBuffer {
    return device.createBuffer({
      label,
      size: Math.max(4, bytes),
      usage,
    });
  }
}

function gridVertexCount(res: SurfaceGridResolution): number {
  return res[0] * res[1] * res[2];
}

function sanitizeResolution(res: SurfaceGridResolution): SurfaceGridResolution {
  return [
    Math.max(2, Math.floor(res[0])),
    Math.max(2, Math.floor(res[1])),
    Math.max(2, Math.floor(res[2])),
  ];
}

function cubeResolutionFromGrid(
  grid: SurfaceGridResolution,
): SurfaceGridResolution {
  return [
    Math.max(1, grid[0] - 1),
    Math.max(1, grid[1] - 1),
    Math.max(1, grid[2] - 1),
  ];
}

function gridCubeCount(cubeRes: SurfaceGridResolution): number {
  return cubeRes[0] * cubeRes[1] * cubeRes[2];
}

function sameResolution(
  a: SurfaceGridResolution,
  b: SurfaceGridResolution,
): boolean {
  return a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
}

export * from './types';
