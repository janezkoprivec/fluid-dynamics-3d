import hashWgsl from './shaders/hash.wgsl?raw';
import sortBitonicWgsl from './shaders/sortBitonic.wgsl?raw';
import clearCellsWgsl from './shaders/clearCells.wgsl?raw';
import cellTableWgsl from './shaders/cellTable.wgsl?raw';
import type { ParticleAllocation } from './particles';

const WORKGROUP_SIZE = 64;
const HASH_ENTRY_STRIDE_BYTES = 8; // vec2<u32>
const U32_BYTES = 4;
const SORT_PARAMS_BYTE_SIZE = 16; // vec4<u32>

function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return Math.max(1, p);
}

export interface SpatialHash {
  encode(encoder: GPUCommandEncoder, gridResolution: [number, number, number]): void;
  rebindParticles(alloc: ParticleAllocation): void;
  dispose(): void;
}

export function createSpatialHash(
  device: GPUDevice,
  initialAlloc: ParticleAllocation,
  paramsBuffer: GPUBuffer,
): SpatialHash {
  // Each bitonic stage needs its own slot in the uniform buffer; we point at
  // them via dynamic uniform offsets. The stride must satisfy the device's
  // minimum uniform-buffer offset alignment (256 on most desktop GPUs).
  const sortParamsStride = Math.max(
    device.limits.minUniformBufferOffsetAlignment,
    SORT_PARAMS_BYTE_SIZE,
  );

  let alloc = initialAlloc;
  let paddedCount = nextPow2(alloc.count);
  let cellCount = 1;
  let numStages = 0;

  let hashEntriesA = createHashEntriesBuffer('spatial-hash/entries-a', paddedCount);
  let hashEntriesB = createHashEntriesBuffer('spatial-hash/entries-b', paddedCount);
  let cellStart = createCellBuffer('spatial-hash/cell-start', cellCount);
  let cellEnd = createCellBuffer('spatial-hash/cell-end', cellCount);

  let sortParamsCapacity = 1;
  let sortParamsBuffer = createSortParamsBuffer(sortParamsCapacity);

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'spatial-hash/bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      {
        binding: 6,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'uniform',
          hasDynamicOffset: true,
          minBindingSize: SORT_PARAMS_BYTE_SIZE,
        },
      },
    ],
  });

  const layout = device.createPipelineLayout({
    label: 'spatial-hash/layout',
    bindGroupLayouts: [bindGroupLayout],
  });

  const hashPipeline = device.createComputePipeline({
    label: 'spatial-hash/hash-pipeline',
    layout,
    compute: {
      module: device.createShaderModule({ label: 'hash.wgsl', code: hashWgsl }),
      entryPoint: 'cs_main',
    },
  });

  const sortPipeline = device.createComputePipeline({
    label: 'spatial-hash/sort-pipeline',
    layout,
    compute: {
      module: device.createShaderModule({ label: 'sortBitonic.wgsl', code: sortBitonicWgsl }),
      entryPoint: 'cs_main',
    },
  });

  const clearCellsPipeline = device.createComputePipeline({
    label: 'spatial-hash/clear-cells-pipeline',
    layout,
    compute: {
      module: device.createShaderModule({ label: 'clearCells.wgsl', code: clearCellsWgsl }),
      entryPoint: 'cs_main',
    },
  });

  const cellTablePipeline = device.createComputePipeline({
    label: 'spatial-hash/cell-table-pipeline',
    layout,
    compute: {
      module: device.createShaderModule({ label: 'cellTable.wgsl', code: cellTableWgsl }),
      entryPoint: 'cs_main',
    },
  });

  let bindGroup = createBindGroup();
  writeStagesForPaddedCount();

  function createHashEntriesBuffer(label: string, count: number): GPUBuffer {
    return device.createBuffer({
      label,
      size: Math.max(1, count) * HASH_ENTRY_STRIDE_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
  }

  function createCellBuffer(label: string, count: number): GPUBuffer {
    return device.createBuffer({
      label,
      size: Math.max(1, count) * U32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
  }

  function createSortParamsBuffer(numSlots: number): GPUBuffer {
    return device.createBuffer({
      label: 'spatial-hash/sort-params',
      size: Math.max(1, numSlots) * sortParamsStride,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  function createBindGroup(): GPUBindGroup {
    return device.createBindGroup({
      label: 'spatial-hash/bg',
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: alloc.gpuBuffer } },
        { binding: 1, resource: { buffer: paramsBuffer } },
        { binding: 2, resource: { buffer: hashEntriesA } },
        { binding: 3, resource: { buffer: hashEntriesB } },
        { binding: 4, resource: { buffer: cellStart } },
        { binding: 5, resource: { buffer: cellEnd } },
        {
          binding: 6,
          resource: { buffer: sortParamsBuffer, size: SORT_PARAMS_BYTE_SIZE },
        },
      ],
    });
  }

  function ensureCellBuffers(gridResolution: [number, number, number]): void {
    const next = Math.max(1, gridResolution[0] * gridResolution[1] * gridResolution[2]);
    if (next === cellCount) return;

    cellStart.destroy();
    cellEnd.destroy();

    cellCount = next;
    cellStart = createCellBuffer('spatial-hash/cell-start', cellCount);
    cellEnd = createCellBuffer('spatial-hash/cell-end', cellCount);
    bindGroup = createBindGroup();
  }

  // Pack every (j, k, mode, paddedCount) stage of the bitonic network into
  // the uniform buffer once. paddedCount only changes when the particle
  // count changes, so this is called from the constructor and from
  // rebindParticles — never per-frame.
  function writeStagesForPaddedCount(): void {
    const stages: Array<{ j: number; k: number; mode: number }> = [];
    stages.push({ j: 0, k: 0, mode: 0 }); // copy A -> B (and fill padding)
    for (let k = 2; k <= paddedCount; k <<= 1) {
      for (let j = k >> 1; j > 0; j >>= 1) {
        stages.push({ j, k, mode: 1 });
      }
    }
    numStages = stages.length;

    if (numStages > sortParamsCapacity) {
      sortParamsBuffer.destroy();
      sortParamsCapacity = numStages;
      sortParamsBuffer = createSortParamsBuffer(numStages);
      bindGroup = createBindGroup();
    }

    const u32PerStride = sortParamsStride / U32_BYTES;
    const host = new Uint32Array(numStages * u32PerStride);
    for (let s = 0; s < numStages; s++) {
      const o = s * u32PerStride;
      host[o + 0] = stages[s].j >>> 0;
      host[o + 1] = stages[s].k >>> 0;
      host[o + 2] = stages[s].mode >>> 0;
      host[o + 3] = paddedCount >>> 0;
    }
    device.queue.writeBuffer(sortParamsBuffer, 0, host);
  }

  return {
    encode(encoder, gridResolution): void {
      ensureCellBuffers(gridResolution);

      const particleGroups = Math.ceil(alloc.count / WORKGROUP_SIZE);
      const cellGroups = Math.ceil(cellCount / WORKGROUP_SIZE);
      const sortGroups = Math.ceil(paddedCount / WORKGROUP_SIZE);

      // hash: write (cellId, particleIndex) into hashEntriesA
      {
        const pass = encoder.beginComputePass({ label: 'hash' });
        pass.setPipeline(hashPipeline);
        pass.setBindGroup(0, bindGroup, [0]);
        pass.dispatchWorkgroups(particleGroups);
        pass.end();
      }

      // bitonic sort on hashEntriesB. Each stage uses a dynamic uniform
      // offset to look up its own (j, k, mode) row.
      for (let s = 0; s < numStages; s++) {
        const offset = s * sortParamsStride;
        const pass = encoder.beginComputePass({
          label: s === 0 ? 'sort-copy' : `sort-stage-${s}`,
        });
        pass.setPipeline(sortPipeline);
        pass.setBindGroup(0, bindGroup, [offset]);
        pass.dispatchWorkgroups(sortGroups);
        pass.end();
      }

      // clear cellStart / cellEnd
      {
        const pass = encoder.beginComputePass({ label: 'clearCells' });
        pass.setPipeline(clearCellsPipeline);
        pass.setBindGroup(0, bindGroup, [0]);
        pass.dispatchWorkgroups(cellGroups);
        pass.end();
      }

      // build cellStart / cellEnd from sorted hashEntriesB
      {
        const pass = encoder.beginComputePass({ label: 'cellTable' });
        pass.setPipeline(cellTablePipeline);
        pass.setBindGroup(0, bindGroup, [0]);
        pass.dispatchWorkgroups(particleGroups);
        pass.end();
      }
    },

    rebindParticles(next): void {
      const nextPadded = nextPow2(next.count);
      const paddedChanged = nextPadded !== paddedCount;
      if (paddedChanged) {
        hashEntriesA.destroy();
        hashEntriesB.destroy();
        hashEntriesA = createHashEntriesBuffer('spatial-hash/entries-a', nextPadded);
        hashEntriesB = createHashEntriesBuffer('spatial-hash/entries-b', nextPadded);
        paddedCount = nextPadded;
      }
      alloc = next;
      bindGroup = createBindGroup();
      if (paddedChanged) {
        writeStagesForPaddedCount();
      }
    },

    dispose(): void {
      hashEntriesA.destroy();
      hashEntriesB.destroy();
      cellStart.destroy();
      cellEnd.destroy();
      sortParamsBuffer.destroy();
    },
  };
}
