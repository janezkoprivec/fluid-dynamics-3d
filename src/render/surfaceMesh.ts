import surfaceMeshWgsl from './shaders/surfaceMesh.wgsl?raw';
import type { OrbitCamera } from './camera';

const CAMERA_BUFFER_SIZE = 256;
const PARAMS_BUFFER_SIZE = 64;

export interface SurfaceMeshBinding {
  vertexBuffer: GPUBuffer;
  normalBuffer: GPUBuffer;
  drawIndirectBuffer: GPUBuffer;
  vertexCapacity: number;
}

export interface SurfaceMeshState {
  baseColor: [number, number, number];
  ambient: number;
  shininess: number;
  specularStrength: number;
  fresnelStrength: number;
  transmission: number;
  lightDir: [number, number, number];
  background: [number, number, number, number];
}

export type SurfaceMeshLoadAction = 'clear' | 'load';

export interface SurfaceMeshRenderer {
  draw(
    encoder: GPUCommandEncoder,
    targetView: GPUTextureView,
    depthView: GPUTextureView,
    state: SurfaceMeshState,
    loadAction: SurfaceMeshLoadAction,
    timestampWrites?: GPURenderPassTimestampWrites,
  ): void;
  rebindSurface(binding: SurfaceMeshBinding): void;
  clearBinding(): void;
  hasBinding(): boolean;
  dispose(): void;
}

export function createSurfaceMeshRenderer(
  device: GPUDevice,
  surfaceFormat: GPUTextureFormat,
  camera: OrbitCamera,
): SurfaceMeshRenderer {
  const module = device.createShaderModule({
    label: 'surfaceMesh.wgsl',
    code: surfaceMeshWgsl,
  });

  const bgl = device.createBindGroupLayout({
    label: 'surfaceMesh/bgl',
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      },
    ],
  });

  const blend: GPUBlendState = {
    color: {
      srcFactor: 'src-alpha',
      dstFactor: 'one-minus-src-alpha',
      operation: 'add',
    },
    alpha: {
      srcFactor: 'one',
      dstFactor: 'one-minus-src-alpha',
      operation: 'add',
    },
  };

  const pipeline = device.createRenderPipeline({
    label: 'surfaceMesh/pipeline',
    layout: device.createPipelineLayout({
      label: 'surfaceMesh/layout',
      bindGroupLayouts: [bgl],
    }),
    vertex: { module, entryPoint: 'vs_main' },
    fragment: {
      module,
      entryPoint: 'fs_main',
      targets: [{ format: surfaceFormat, blend }],
    },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: {
      format: 'depth24plus',
      depthWriteEnabled: true,
      depthCompare: 'less',
    },
  });

  const cameraBuffer = device.createBuffer({
    label: 'surfaceMesh/camera',
    size: CAMERA_BUFFER_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const cameraHost = new ArrayBuffer(CAMERA_BUFFER_SIZE);
  const cameraF32 = new Float32Array(cameraHost);

  const paramsBuffer = device.createBuffer({
    label: 'surfaceMesh/params',
    size: PARAMS_BUFFER_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const paramsHost = new ArrayBuffer(PARAMS_BUFFER_SIZE);
  const paramsF32 = new Float32Array(paramsHost);

  let binding: SurfaceMeshBinding | null = null;
  let bindGroup: GPUBindGroup | null = null;

  function createBindGroup(b: SurfaceMeshBinding): GPUBindGroup {
    return device.createBindGroup({
      label: 'surfaceMesh/bg',
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: b.vertexBuffer } },
        { binding: 1, resource: { buffer: b.normalBuffer } },
        { binding: 2, resource: { buffer: cameraBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } },
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

  function uploadParams(state: SurfaceMeshState): void {
    paramsF32[0] = state.baseColor[0];
    paramsF32[1] = state.baseColor[1];
    paramsF32[2] = state.baseColor[2];
    paramsF32[3] = state.ambient;
    paramsF32[4] = state.lightDir[0];
    paramsF32[5] = state.lightDir[1];
    paramsF32[6] = state.lightDir[2];
    paramsF32[7] = state.shininess;
    paramsF32[8] = state.specularStrength;
    paramsF32[9] = state.fresnelStrength;
    paramsF32[10] = state.transmission;
    paramsF32[11] = 0;
    paramsF32[12] = 0;
    paramsF32[13] = 0;
    paramsF32[14] = 0;
    paramsF32[15] = 0;
    device.queue.writeBuffer(paramsBuffer, 0, paramsHost);
  }

  return {
    draw(encoder, targetView, depthView, state, loadAction, timestampWrites): void {
      if (!binding || !bindGroup) return;
      uploadCamera();
      uploadParams(state);

      const colorAttachment: GPURenderPassColorAttachment =
        loadAction === 'clear'
          ? {
              view: targetView,
              clearValue: {
                r: state.background[0],
                g: state.background[1],
                b: state.background[2],
                a: state.background[3],
              },
              loadOp: 'clear',
              storeOp: 'store',
            }
          : {
              view: targetView,
              loadOp: 'load',
              storeOp: 'store',
            };

      const depthAttachment: GPURenderPassDepthStencilAttachment =
        loadAction === 'clear'
          ? {
              view: depthView,
              depthClearValue: 1,
              depthLoadOp: 'clear',
              depthStoreOp: 'store',
            }
          : {
              view: depthView,
              depthLoadOp: 'load',
              depthStoreOp: 'store',
            };

      const desc: GPURenderPassDescriptor = {
        label: 'surfaceMesh',
        colorAttachments: [colorAttachment],
        depthStencilAttachment: depthAttachment,
      };
      if (timestampWrites) desc.timestampWrites = timestampWrites;

      const pass = encoder.beginRenderPass(desc);
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.drawIndirect(binding.drawIndirectBuffer, 0);
      pass.end();
    },
    rebindSurface(next): void {
      binding = next;
      bindGroup = createBindGroup(next);
    },
    clearBinding(): void {
      binding = null;
      bindGroup = null;
    },
    hasBinding(): boolean {
      return binding !== null;
    },
    dispose(): void {
      cameraBuffer.destroy();
      paramsBuffer.destroy();
    },
  };
}
