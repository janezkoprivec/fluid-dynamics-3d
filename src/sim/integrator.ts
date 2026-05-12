import integrateWgsl from './shaders/integrate.wgsl?raw';
import type { ParticleAllocation } from './particles';

const PARAMS_BYTE_SIZE = 64;
const WORKGROUP_SIZE = 64;

export interface IntegratorState {
  gravity: [number, number, number];
  restitution: number;
  dt: number;
  boxMin: [number, number, number];
  boxMax: [number, number, number];
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

  const pipeline = device.createComputePipeline({
    label: 'integrator/pipeline',
    layout: device.createPipelineLayout({
      label: 'integrator/layout',
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: { module, entryPoint: 'cs_main' },
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

  function writeParams(state: IntegratorState): void {
    paramsF32[0] = state.gravity[0];
    paramsF32[1] = state.gravity[1];
    paramsF32[2] = state.gravity[2];
    paramsF32[3] = state.dt;
    paramsF32[4] = state.boxMin[0];
    paramsF32[5] = state.boxMin[1];
    paramsF32[6] = state.boxMin[2];
    paramsF32[7] = state.restitution;
    paramsF32[8] = state.boxMax[0];
    paramsF32[9] = state.boxMax[1];
    paramsF32[10] = state.boxMax[2];
    paramsU32[11] = alloc.count;
    device.queue.writeBuffer(paramsBuffer, 0, paramsHost);
  }

  return {
    encode(encoder, state, timestampWrites): void {
      writeParams(state);
      const desc: GPUComputePassDescriptor = { label: 'integrate' };
      if (timestampWrites) desc.timestampWrites = timestampWrites;
      const pass = encoder.beginComputePass(desc);
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      const groups = Math.ceil(alloc.count / WORKGROUP_SIZE);
      pass.dispatchWorkgroups(groups);
      pass.end();
    },
    rebindParticles(next): void {
      alloc = next;
      bindGroup = createBindGroup();
    },
    dispose(): void {
      paramsBuffer.destroy();
    },
  };
}
