import { createOrbitCamera, type OrbitCamera } from './camera';
import {
  createPointSpritesRenderer,
  type PointSpritesParams,
  type PointSpritesRenderer,
} from './pointSprites';
import type { ParticleAllocation } from '../sim/particles';
import { hexToRgba } from '../util/math';

export interface RenderParams {
  pointSize: number;
  backgroundColor: string;
}

export const DEFAULT_RENDER_PARAMS: RenderParams = {
  pointSize: 6,
  backgroundColor: '#0a0c10',
};

export interface Renderer {
  readonly camera: OrbitCamera;
  readonly params: RenderParams;
  draw(
    encoder: GPUCommandEncoder,
    targetView: GPUTextureView,
    depthView: GPUTextureView,
    viewport: { width: number; height: number },
    timestampWrites?: GPURenderPassTimestampWrites,
  ): void;
  rebindParticles(alloc: ParticleAllocation): void;
  setParams(patch: Partial<RenderParams>): void;
  dispose(): void;
}

export function createRenderer(
  device: GPUDevice,
  surfaceFormat: GPUTextureFormat,
  initialAlloc: ParticleAllocation,
): Renderer {
  const camera = createOrbitCamera();
  const params: RenderParams = { ...DEFAULT_RENDER_PARAMS };
  const sprites: PointSpritesRenderer = createPointSpritesRenderer(
    device,
    surfaceFormat,
    camera,
    initialAlloc,
  );

  function spritesState(): PointSpritesParams {
    return {
      pointSize: params.pointSize,
      background: hexToRgba(params.backgroundColor),
    };
  }

  return {
    camera,
    get params(): RenderParams {
      return params;
    },
    draw(encoder, targetView, depthView, viewport, timestampWrites): void {
      camera.update();
      sprites.draw(
        encoder,
        targetView,
        depthView,
        viewport,
        spritesState(),
        timestampWrites,
      );
    },
    rebindParticles(alloc): void {
      sprites.rebindParticles(alloc);
    },
    setParams(patch): void {
      Object.assign(params, patch);
    },
    dispose(): void {
      sprites.dispose();
    },
  };
}

export type { OrbitCamera } from './camera';
