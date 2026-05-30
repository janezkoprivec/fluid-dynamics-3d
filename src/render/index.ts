import { createOrbitCamera, type OrbitCamera } from './camera';
import {
  createPointSpritesRenderer,
  type PointSpritesParams,
  type PointSpritesRenderer,
} from './pointSprites';
import {
  createSurfaceMeshRenderer,
  type SurfaceMeshBinding,
  type SurfaceMeshRenderer,
  type SurfaceMeshState,
} from './surfaceMesh';
import { createBoxRenderer, type BoxRenderer, type BoxState } from './box';
import type { ParticleAllocation } from '../sim/particles';
import { SIM_BOX_MAX, SIM_BOX_MIN } from '../sim';
import { hexToRgba } from '../util/math';

export type RenderMode = 'points' | 'surface' | 'both';

export interface RenderParams {
  pointSize: number;
  backgroundColor: string;
  mode: RenderMode;
  surfaceBaseColor: string;
  surfaceAmbient: number;
  surfaceShininess: number;
  surfaceSpecular: number;
  surfaceFresnel: number;
  surfaceTransmission: number;
  showBoxEdges: boolean;
  showBoxFaces: boolean;
  boxEdgeColor: string;
  boxFaceColor: string;
  boxEdgeAlpha: number;
  boxFaceAlpha: number;
}

export const DEFAULT_RENDER_PARAMS: RenderParams = {
  pointSize: 6,
  backgroundColor: '#0a0c10',
  mode: 'points',
  surfaceBaseColor: '#1f4a7a',
  surfaceAmbient: 0.22,
  surfaceShininess: 64,
  surfaceSpecular: 0.55,
  surfaceFresnel: 0.7,
  surfaceTransmission: 0.55,
  showBoxEdges: true,
  showBoxFaces: true,
  boxEdgeColor: '#7fb8ff',
  boxFaceColor: '#a8d2ff',
  boxEdgeAlpha: 0.55,
  boxFaceAlpha: 0.05,
};

const SURFACE_LIGHT_DIR: [number, number, number] = [0.3, 0.85, 0.4];

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
  setDrawCount(n: number): void;
  rebindSurface(binding: SurfaceMeshBinding): void;
  clearSurfaceBinding(): void;
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
  const surfaceMesh: SurfaceMeshRenderer = createSurfaceMeshRenderer(
    device,
    surfaceFormat,
    camera,
  );
  const box: BoxRenderer = createBoxRenderer(device, surfaceFormat, camera);

  function spritesState(): PointSpritesParams {
    return {
      pointSize: params.pointSize,
      background: hexToRgba(params.backgroundColor),
    };
  }

  function surfaceState(): SurfaceMeshState {
    return {
      baseColor: hexToRgbTriple(params.surfaceBaseColor),
      ambient: params.surfaceAmbient,
      shininess: params.surfaceShininess,
      specularStrength: params.surfaceSpecular,
      fresnelStrength: params.surfaceFresnel,
      transmission: params.surfaceTransmission,
      lightDir: SURFACE_LIGHT_DIR,
      background: hexToRgba(params.backgroundColor),
    };
  }

  function boxState(): BoxState {
    return {
      boxMin: [SIM_BOX_MIN[0], SIM_BOX_MIN[1], SIM_BOX_MIN[2]],
      boxMax: [SIM_BOX_MAX[0], SIM_BOX_MAX[1], SIM_BOX_MAX[2]],
      edgeColor: hexToRgbTriple(params.boxEdgeColor),
      edgeAlpha: params.boxEdgeAlpha,
      faceColor: hexToRgbTriple(params.boxFaceColor),
      faceAlpha: params.boxFaceAlpha,
      drawEdges: params.showBoxEdges,
      drawFaces: params.showBoxFaces,
    };
  }

  return {
    camera,
    get params(): RenderParams {
      return params;
    },
    draw(encoder, targetView, depthView, viewport, timestampWrites): void {
      camera.update();
      const mode = params.mode;
      const surfaceReady = surfaceMesh.hasBinding();
      const wantSurface = (mode === 'surface' || mode === 'both') && surfaceReady;
      const wantPoints = mode === 'points' || mode === 'both' || !surfaceReady;

      if (wantPoints) {
        sprites.draw(
          encoder,
          targetView,
          depthView,
          viewport,
          spritesState(),
          timestampWrites,
        );
        if (wantSurface) {
          surfaceMesh.draw(encoder, targetView, depthView, surfaceState(), 'load');
        }
      } else if (wantSurface) {
        surfaceMesh.draw(
          encoder,
          targetView,
          depthView,
          surfaceState(),
          'clear',
          timestampWrites,
        );
      }
      box.draw(encoder, targetView, depthView, boxState());
    },
    rebindParticles(alloc): void {
      sprites.rebindParticles(alloc);
    },
    setDrawCount(n): void {
      sprites.setDrawCount(n);
    },
    rebindSurface(binding): void {
      surfaceMesh.rebindSurface(binding);
    },
    clearSurfaceBinding(): void {
      surfaceMesh.clearBinding();
    },
    setParams(patch): void {
      Object.assign(params, patch);
    },
    dispose(): void {
      sprites.dispose();
      surfaceMesh.dispose();
      box.dispose();
    },
  };
}

function hexToRgbTriple(hex: string): [number, number, number] {
  const rgba = hexToRgba(hex);
  return [rgba[0], rgba[1], rgba[2]];
}

export type { OrbitCamera } from './camera';
export type { SurfaceMeshBinding } from './surfaceMesh';
