import pointSpritesWgsl from './shaders/pointSprites.wgsl?raw';
import type { OrbitCamera } from './camera';
import type { ParticleAllocation } from '../sim/particles';

const CAMERA_BUFFER_SIZE = 256;
const RPARAMS_BUFFER_SIZE = 32;
const VERTICES_PER_SPRITE = 6;

export interface PointSpritesParams {
  pointSize: number;
  background: [number, number, number, number];
}

export interface PointSpritesRenderer {
  draw(
    encoder: GPUCommandEncoder,
    targetView: GPUTextureView,
    depthView: GPUTextureView,
    viewport: { width: number; height: number },
    state: PointSpritesParams,
    timestampWrites?: GPURenderPassTimestampWrites,
  ): void;
  rebindParticles(alloc: ParticleAllocation): void;
  dispose(): void;
}

export function createPointSpritesRenderer(
  device: GPUDevice,
  surfaceFormat: GPUTextureFormat,
  camera: OrbitCamera,
  initialAlloc: ParticleAllocation,
): PointSpritesRenderer {
  const module = device.createShaderModule({
    label: 'pointSprites.wgsl',
    code: pointSpritesWgsl,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'pointSprites/bgl',
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      },
    ],
  });

  const pipeline = device.createRenderPipeline({
    label: 'pointSprites/pipeline',
    layout: device.createPipelineLayout({
      label: 'pointSprites/layout',
      bindGroupLayouts: [bindGroupLayout],
    }),
    vertex: { module, entryPoint: 'vs_main' },
    fragment: {
      module,
      entryPoint: 'fs_main',
      targets: [{ format: surfaceFormat }],
    },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: {
      format: 'depth24plus',
      depthWriteEnabled: true,
      depthCompare: 'less',
    },
  });

  const cameraBuffer = device.createBuffer({
    label: 'pointSprites/camera',
    size: CAMERA_BUFFER_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const cameraHost = new ArrayBuffer(CAMERA_BUFFER_SIZE);
  const cameraF32 = new Float32Array(cameraHost);

  const rparamsBuffer = device.createBuffer({
    label: 'pointSprites/rparams',
    size: RPARAMS_BUFFER_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const rparamsHost = new Float32Array(RPARAMS_BUFFER_SIZE / 4);

  let alloc = initialAlloc;
  let bindGroup = createBindGroup();

  function createBindGroup(): GPUBindGroup {
    return device.createBindGroup({
      label: 'pointSprites/bg',
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: alloc.gpuBuffer } },
        { binding: 1, resource: { buffer: cameraBuffer } },
        { binding: 2, resource: { buffer: rparamsBuffer } },
      ],
    });
  }

  function uploadCamera(): void {
    cameraF32.set(camera.viewProj, 0);
    cameraF32.set(camera.view, 16);
    cameraF32.set(camera.proj, 32);
    cameraF32[48] = camera.eye[0]!;
    cameraF32[49] = camera.eye[1]!;
    cameraF32[50] = camera.eye[2]!;
    cameraF32[51] = 0;
    device.queue.writeBuffer(cameraBuffer, 0, cameraHost);
  }

  function uploadRenderParams(
    viewport: { width: number; height: number },
    state: PointSpritesParams,
  ): void {
    rparamsHost[0] = viewport.width;
    rparamsHost[1] = viewport.height;
    rparamsHost[2] = state.pointSize;
    rparamsHost[3] = 0;
    device.queue.writeBuffer(rparamsBuffer, 0, rparamsHost);
  }

  return {
    draw(encoder, targetView, depthView, viewport, state, timestampWrites): void {
      uploadCamera();
      uploadRenderParams(viewport, state);

      const desc: GPURenderPassDescriptor = {
        label: 'pointSprites',
        colorAttachments: [
          {
            view: targetView,
            clearValue: {
              r: state.background[0],
              g: state.background[1],
              b: state.background[2],
              a: state.background[3],
            },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
        depthStencilAttachment: {
          view: depthView,
          depthClearValue: 1,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      };
      if (timestampWrites) desc.timestampWrites = timestampWrites;
      const pass = encoder.beginRenderPass(desc);

      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(VERTICES_PER_SPRITE, alloc.count, 0, 0);
      pass.end();
    },
    rebindParticles(next): void {
      alloc = next;
      bindGroup = createBindGroup();
    },
    dispose(): void {
      cameraBuffer.destroy();
      rparamsBuffer.destroy();
    },
  };
}
