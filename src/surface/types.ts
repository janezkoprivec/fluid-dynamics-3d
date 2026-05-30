export type SurfaceGridResolution = [number, number, number];

export interface SurfaceSimParamsView {
  smoothingRadius: number;
  particleMass: number;
  restDensity: number;
  particleCount: number;
}

export interface SurfaceParams {
  enabled: boolean;
  gridResolution: SurfaceGridResolution; // vertex-grid resolution
  gridMin: [number, number, number];
  cellSize: number;
  isoValue: number;
  maxTriangles: number;
}

export const DEFAULT_SURFACE_PARAMS: SurfaceParams = {
  enabled: false,
  gridResolution: [40, 40, 40],
  // Grid extends 2 cells past the sim box on every side so the field-build
  // pass (which clamps density to 0 outside the box) gives MC a clean
  // air-side to close the iso-surface against the glass walls.
  // usable span = 40 - 1 - 4 = 35 cells across the box ⇒ cellSize = 0.84/35.
  cellSize: 0.84 / 35,
  gridMin: [-0.42 - (2 * 0.84) / 35, -0.42 - (2 * 0.84) / 35, -0.42 - (2 * 0.84) / 35],
  // ~half rest density — Müller 2003's convention for SPH iso-extraction.
  isoValue: 500.0,
  maxTriangles: 1_000_000,
};

export interface SurfaceMeshBindingResources {
  vertexBuffer: GPUBuffer;
  normalBuffer: GPUBuffer;
  drawIndirectBuffer: GPUBuffer;
  vertexCapacity: number;
}

export interface SurfaceResources {
  params: SurfaceParams;
  particleCount: number;
  particleBuffer: GPUBuffer;

  fieldValues: GPUBuffer;
  fieldValueCount: number;

  cubeResolution: SurfaceGridResolution;
  cubeCount: number;
  cubeCase: GPUBuffer;
  cubeTriCount: GPUBuffer;
  cubePrefix: GPUBuffer;

  mesh: SurfaceMeshBindingResources;
  simParams: SurfaceSimParamsView;
}

export interface SurfaceFieldStats {
  min: number;
  max: number;
  nonFiniteCount: number;
  sampleCount: number;
}

export interface SurfaceCounters {
  totalTriangles: number;
  cappedVertexCount: number;
  capacity: number;
}

export interface SurfacePipeline {
  readonly params: SurfaceParams;
  update(encoder: GPUCommandEncoder): void;
  rebindParticles(alloc: { count: number; gpuBuffer: GPUBuffer }): void;
  setParams(patch: Partial<SurfaceParams>): void;
  setSimParams(patch: Partial<SurfaceSimParamsView>): void;
  resources(): SurfaceResources;
  meshBinding(): SurfaceMeshBindingResources;
  readbackFieldStats(): Promise<SurfaceFieldStats>;
  readbackCounters(): Promise<SurfaceCounters>;
  dispose(): void;
}
