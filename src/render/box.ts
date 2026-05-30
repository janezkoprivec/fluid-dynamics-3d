import boxWgsl from './shaders/box.wgsl?raw';
import type { OrbitCamera } from './camera';

const CAMERA_BUFFER_SIZE = 256;
const PARAMS_BUFFER_SIZE = 80;

const EDGE_VERTEX_COUNT = 24;
const FACE_VERTEX_COUNT = 36;

export interface BoxState {
  boxMin: [number, number, number];
  boxMax: [number, number, number];
  edgeColor: [number, number, number];
  edgeAlpha: number;
  faceColor: [number, number, number];
  faceAlpha: number;
  drawEdges: boolean;
  drawFaces: boolean;
}

export interface BoxRenderer {
  draw(
    encoder: GPUCommandEncoder,
    targetView: GPUTextureView,
    depthView: GPUTextureView,
    state: BoxState,
  ): void;
  dispose(): void;
}

export function createBoxRenderer(
  device: GPUDevice,
  surfaceFormat: GPUTextureFormat,
  camera: OrbitCamera,
): BoxRenderer {
  const module = device.createShaderModule({
    label: 'box.wgsl',
    code: boxWgsl,
  });

  const bgl = device.createBindGroupLayout({
    label: 'box/bgl',
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      },
    ],
  });

  const layout = device.createPipelineLayout({
    label: 'box/layout',
    bindGroupLayouts: [bgl],
  });

  const blend: GPUBlendState = {
    color: {
      srcFactor: 'one',
      dstFactor: 'one-minus-src-alpha',
      operation: 'add',
    },
    alpha: {
      srcFactor: 'one',
      dstFactor: 'one-minus-src-alpha',
      operation: 'add',
    },
  };

  const edgePipeline = device.createRenderPipeline({
    label: 'box/edges',
    layout,
    vertex: { module, entryPoint: 'vs_edges' },
    fragment: {
      module,
      entryPoint: 'fs_edges',
      targets: [{ format: surfaceFormat, blend }],
    },
    primitive: { topology: 'line-list', cullMode: 'none' },
    depthStencil: {
      format: 'depth24plus',
      depthWriteEnabled: false,
      depthCompare: 'less-equal',
    },
  });

  const facePipeline = device.createRenderPipeline({
    label: 'box/faces',
    layout,
    vertex: { module, entryPoint: 'vs_faces' },
    fragment: {
      module,
      entryPoint: 'fs_faces',
      targets: [{ format: surfaceFormat, blend }],
    },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: {
      format: 'depth24plus',
      depthWriteEnabled: false,
      depthCompare: 'less',
    },
  });

  const cameraBuffer = device.createBuffer({
    label: 'box/camera',
    size: CAMERA_BUFFER_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const cameraHost = new ArrayBuffer(CAMERA_BUFFER_SIZE);
  const cameraF32 = new Float32Array(cameraHost);

  const paramsBuffer = device.createBuffer({
    label: 'box/params',
    size: PARAMS_BUFFER_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const paramsHost = new ArrayBuffer(PARAMS_BUFFER_SIZE);
  const paramsF32 = new Float32Array(paramsHost);

  const bindGroup = device.createBindGroup({
    label: 'box/bg',
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: cameraBuffer } },
      { binding: 1, resource: { buffer: paramsBuffer } },
    ],
  });

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

  function uploadParams(state: BoxState): void {
    paramsF32[0] = state.boxMin[0];
    paramsF32[1] = state.boxMin[1];
    paramsF32[2] = state.boxMin[2];
    paramsF32[3] = state.edgeAlpha;
    paramsF32[4] = state.boxMax[0];
    paramsF32[5] = state.boxMax[1];
    paramsF32[6] = state.boxMax[2];
    paramsF32[7] = state.faceAlpha;
    paramsF32[8] = state.edgeColor[0];
    paramsF32[9] = state.edgeColor[1];
    paramsF32[10] = state.edgeColor[2];
    paramsF32[11] = 0;
    paramsF32[12] = state.faceColor[0];
    paramsF32[13] = state.faceColor[1];
    paramsF32[14] = state.faceColor[2];
    paramsF32[15] = 0;
    paramsF32[16] = 0;
    paramsF32[17] = 0;
    paramsF32[18] = 0;
    paramsF32[19] = 0;
    device.queue.writeBuffer(paramsBuffer, 0, paramsHost);
  }

  return {
    draw(encoder, targetView, depthView, state): void {
      if (!state.drawEdges && !state.drawFaces) return;
      uploadCamera();
      uploadParams(state);

      const desc: GPURenderPassDescriptor = {
        label: 'box',
        colorAttachments: [
          { view: targetView, loadOp: 'load', storeOp: 'store' },
        ],
        depthStencilAttachment: {
          view: depthView,
          depthLoadOp: 'load',
          depthStoreOp: 'store',
        },
      };

      const pass = encoder.beginRenderPass(desc);
      pass.setBindGroup(0, bindGroup);
      if (state.drawFaces) {
        pass.setPipeline(facePipeline);
        pass.draw(FACE_VERTEX_COUNT, 1, 0, 0);
      }
      if (state.drawEdges) {
        pass.setPipeline(edgePipeline);
        pass.draw(EDGE_VERTEX_COUNT, 1, 0, 0);
      }
      pass.end();
    },
    dispose(): void {
      cameraBuffer.destroy();
      paramsBuffer.destroy();
    },
  };
}
