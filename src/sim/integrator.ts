import integrateWgsl from './shaders/integrate.wgsl?raw';
import densityWgsl from './shaders/density.wgsl?raw';
import forcesWgsl from './shaders/forces.wgsl?raw';
import type { ParticleAllocation } from './particles';
import { createSpatialHash, type SpatialHash } from './spatialHash';

const PARAMS_BYTE_SIZE = 112;
const WORKGROUP_SIZE = 64;

export interface IntegratorState {
  gravity: [number, number, number];
  dt: number;

  boxMin: [number, number, number];
  boxMax: [number, number, number];
  boundaryDamping: number;
  boundarySlop: number;

  smoothingRadius: number; // h
  restDensity: number;     // rho0
  gasConstant: number;     // k
  viscosity: number;       // mu
  gamma: number;           // Tait gamma
  maxPressure: number;
  particleMass: number;     // m

  gridResolution: [number, number, number]; // placeholder for hash stage

  wallRepulsion: number;
  wallDamping: number;
  wallRange: number;
}

export interface Integrator {
  encode(
    encoder: GPUCommandEncoder,
    state: IntegratorState,
    timestampWrites?: GPUComputePassTimestampWrites,
  ): void;
  rebindParticles(alloc: ParticleAllocation): void;
  dispose(): void;
}

export function createIntegrator(
  device: GPUDevice,
  initialAlloc: ParticleAllocation,
): Integrator {
  const module = device.createShaderModule({
    label: 'integrate.wgsl',
    code: integrateWgsl,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'integrator/bgl',
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' },
      },
    ],
  });

  const neighborBindGroupLayout = device.createBindGroupLayout({
    label: 'integrator/neighbors-bgl',
    entries: [
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // sortedEntries
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // cellStart
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // cellEnd
    ],
  });

  const pipeline = device.createComputePipeline({
    label: 'integrator/pipeline',
    layout: device.createPipelineLayout({
      label: 'integrator/layout',
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: { module, entryPoint: 'cs_main' },
  });

  const densityModule = device.createShaderModule({
    label: 'density.wgsl',
    code: densityWgsl,
  });

  const densityPipeline = device.createComputePipeline({
    label: 'density/pipeline',
    layout: device.createPipelineLayout({
      label: 'density/layout',
      bindGroupLayouts: [bindGroupLayout, neighborBindGroupLayout],
    }),
    compute: { module: densityModule, entryPoint: 'cs_main' },
  });

  const forcesModule = device.createShaderModule({
    label: 'forces.wgsl',
    code: forcesWgsl,
  });

  const forcesPipeline = device.createComputePipeline({
    label: 'forces/pipeline',
    layout: device.createPipelineLayout({
      label: 'forces/layout',
      bindGroupLayouts: [bindGroupLayout, neighborBindGroupLayout],
    }),
    compute: { module: forcesModule, entryPoint: 'cs_main' },
  });

  const paramsBuffer = device.createBuffer({
    label: 'integrator/params',
    size: PARAMS_BYTE_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const paramsHost = new ArrayBuffer(PARAMS_BYTE_SIZE);
  const paramsF32 = new Float32Array(paramsHost);
  const paramsU32 = new Uint32Array(paramsHost);

  


  let alloc = initialAlloc;
  let bindGroup = createBindGroup();

  let spatialHash: SpatialHash = createSpatialHash(device, alloc, paramsBuffer);
  let neighborBindGroup = createNeighborBindGroup();

  function createBindGroup(): GPUBindGroup {
    return device.createBindGroup({
      label: 'integrator/bg',
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: alloc.gpuBuffer } },
        { binding: 1, resource: { buffer: paramsBuffer } },
      ],
    });
  }

  function createNeighborBindGroup(): GPUBindGroup {
    const r = spatialHash.resources();
    return device.createBindGroup({
      label: 'integrator/neighbors-bg',
      layout: neighborBindGroupLayout,
      entries: [
        { binding: 2, resource: { buffer: r.sortedEntries } },
        { binding: 3, resource: { buffer: r.cellStart } },
        { binding: 4, resource: { buffer: r.cellEnd } },
      ],
    });
  }

  function writeParams(state: IntegratorState): void {
    // SimParams uniform packing (28 * 4 = 112 bytes). Must match the WGSL
    // `struct SimParams` in density.wgsl / forces.wgsl / integrate.wgsl.
    // f32[ 0.. 2] gravity.xyz
    // f32[ 3]     dt
    // f32[ 4.. 6] boxMin.xyz
    // f32[ 7]     boundaryDamping
    // f32[ 8..10] boxMax.xyz
    // f32[11]     smoothingRadius
    // f32[12]     restDensity
    // f32[13]     gasConstant
    // f32[14]     viscosity
    // f32[15]     gamma
    // f32[16]     maxPressure
    // f32[17]     boundarySlop
    // f32[18]     particleMass
    // u32[19]     particleCount
    // u32[20]     gridResolution.x
    // u32[21]     gridResolution.y
    // u32[22]     gridResolution.z
    // f32[23]     wallRepulsion
    // f32[24]     wallDamping
    // f32[25]     wallRange
    // f32[26]     _pad0
    // f32[27]     _pad1

    paramsF32[0] = state.gravity[0];
    paramsF32[1] = state.gravity[1];
    paramsF32[2] = state.gravity[2];
    paramsF32[3] = state.dt;

    paramsF32[4] = state.boxMin[0];
    paramsF32[5] = state.boxMin[1];
    paramsF32[6] = state.boxMin[2];
    paramsF32[7] = state.boundaryDamping;

    paramsF32[8] = state.boxMax[0];
    paramsF32[9] = state.boxMax[1];
    paramsF32[10] = state.boxMax[2];
    paramsF32[11] = state.smoothingRadius;

    paramsF32[12] = state.restDensity;
    paramsF32[13] = state.gasConstant;
    paramsF32[14] = state.viscosity;
    paramsF32[15] = state.gamma;

    paramsF32[16] = state.maxPressure;
    paramsF32[17] = state.boundarySlop;
    paramsF32[18] = state.particleMass;

    paramsU32[19] = alloc.count;
    paramsU32[20] = state.gridResolution[0] >>> 0;
    paramsU32[21] = state.gridResolution[1] >>> 0;
    paramsU32[22] = state.gridResolution[2] >>> 0;

    paramsF32[23] = state.wallRepulsion;
    paramsF32[24] = state.wallDamping;
    paramsF32[25] = state.wallRange;
    paramsF32[26] = 0;
    paramsF32[27] = 0;

    device.queue.writeBuffer(paramsBuffer, 0, paramsHost);
  }

  return {
    encode(encoder, state, timestampWrites): void {
      writeParams(state);
      const groups = Math.ceil(alloc.count / WORKGROUP_SIZE);

      spatialHash.encode(encoder, state.gridResolution);
      neighborBindGroup = createNeighborBindGroup();

      // pass 1: density + pressure
      {
        const densityPass = encoder.beginComputePass({ label: 'density' });
        densityPass.setPipeline(densityPipeline);
        densityPass.setBindGroup(0, bindGroup);
        densityPass.setBindGroup(1, neighborBindGroup);
        densityPass.dispatchWorkgroups(groups);
        densityPass.end();
      }
      // pass 2: forces
      {
        const forcesPass = encoder.beginComputePass({ label: 'forces' });
        forcesPass.setPipeline(forcesPipeline);
        forcesPass.setBindGroup(0, bindGroup);
        forcesPass.setBindGroup(1, neighborBindGroup);
        forcesPass.dispatchWorkgroups(groups);
        forcesPass.end();
      }
      // pass 3: existing integrate
      {
        const desc: GPUComputePassDescriptor = { label: 'integrate' };
        if (timestampWrites) desc.timestampWrites = timestampWrites;
        const pass = encoder.beginComputePass(desc);
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(groups);
        pass.end();
      }
    },
    rebindParticles(next): void {
      alloc = next;
      bindGroup = createBindGroup();
      spatialHash.rebindParticles(next);
      neighborBindGroup = createNeighborBindGroup();
    },
    dispose(): void {
      spatialHash.dispose();
      paramsBuffer.destroy();
    },
  };
}
