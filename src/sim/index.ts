import type { TgpuRoot } from 'typegpu';
import {
  allocateParticles,
  seedRandomCube,
  type ParticleAllocation,
} from './particles';
import {
  createIntegrator,
  type Integrator,
  type IntegratorState,
} from './integrator';

export interface SimParams {
  particleCount: number;
  gravity: [number, number, number];
  restitution: number;
  timestep: number;
  paused: boolean;
}

export const DEFAULT_SIM_PARAMS: SimParams = {
  particleCount: 16_384,
  gravity: [0, -9.81, 0],
  restitution: 0.4,
  timestep: 1 / 120,
  paused: false,
};

export const SIM_BOX_MIN: [number, number, number] = [-0.42, -0.42, -0.42];
export const SIM_BOX_MAX: [number, number, number] = [0.42, 0.42, 0.42];
const SEED_HALF_EXTENT = 0.21;

export interface Sim {
  readonly params: SimParams;
  readonly allocation: ParticleAllocation;
  step(
    encoder: GPUCommandEncoder,
    dt: number,
    timestampWrites?: GPUComputePassTimestampWrites,
  ): void;
  reset(newCount?: number): void;
  setParams(patch: Partial<SimParams>): void;
  dispose(): void;
}

export function createSim(
  root: TgpuRoot,
  device: GPUDevice,
  initial: Partial<SimParams> = {},
): Sim {
  const params: SimParams = { ...DEFAULT_SIM_PARAMS, ...initial };
  let alloc = allocateParticles(root, params.particleCount);
  seedRandomCube(alloc, { halfExtent: SEED_HALF_EXTENT });

  const integrator: Integrator = createIntegrator(device, alloc);

  function currentIntegratorState(dt: number): IntegratorState {
    return {
      gravity: params.gravity,
      restitution: params.restitution,
      dt,
      boxMin: SIM_BOX_MIN,
      boxMax: SIM_BOX_MAX,
    };
  }

  return {
    get params(): SimParams {
      return params;
    },
    get allocation(): ParticleAllocation {
      return alloc;
    },
    step(encoder, dt, timestampWrites): void {
      if (params.paused) return;
      integrator.encode(encoder, currentIntegratorState(dt), timestampWrites);
    },
    reset(newCount): void {
      if (newCount !== undefined && newCount !== params.particleCount) {
        params.particleCount = Math.max(64, Math.floor(newCount));
        alloc.gpuBuffer.destroy();
        alloc = allocateParticles(root, params.particleCount);
        integrator.rebindParticles(alloc);
      }
      seedRandomCube(alloc, { halfExtent: SEED_HALF_EXTENT });
    },
    setParams(patch): void {
      Object.assign(params, patch);
    },
    dispose(): void {
      integrator.dispose();
      alloc.gpuBuffer.destroy();
    },
  };
}

export type { ParticleAllocation } from './particles';
