import {
  allocateParticles,
  seedPackedCube,
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

  particleMass: number;
  smoothingRadius: number;
  restDensity: number;
  gasConstant: number;
  viscosity: number;
  gamma: number;
  maxPressure: number;

  // Boundary handling. Within `wallRange * smoothingRadius` of any wall,
  // particles feel:
  //   - outward acceleration `wallRepulsion * t^2`,
  //   - normal-velocity damping `wallDamping * v_n * t` opposing motion
  //     into the wall. The damper kills the bounce that pure repulsion
  //     would otherwise produce.
  // `t` ramps from 0 at the band edge to 1 at the wall.
  wallRepulsion: number;
  wallDamping: number;
  wallRange: number;
}

// Defaults tuned for water-like behaviour at this sim scale (0.84 m box,
// 1536 particles ≈ 30 mm equilibrium spacing). Derivations:
//   - particleMass × particleCount / box_volume ≈ ρ₀ (1000 kg/m³)
//   - smoothing h ≈ 2 × natural spacing → ~30–40 neighbors per particle,
//     the Müller sweet spot
//   - gasConstant chosen so SPH sound speed c = √(k γ / ρ₀) ≈ 3.2 m/s,
//     roughly 10× max particle velocity, the WCSPH stiffness convention
//   - viscosity μ ≈ 2 is the lowest value that stays stable with the
//     1/480 s substep at this h; lower (≈ water's real ~0.001 Pa·s)
//     would be unstable without an implicit solver
//   - wall band kept on but reduced: at full strength the boundary
//     particles get over-damped and look like syrup against the glass;
//     fully off gives the kernel-deficiency tongues. These mid values
//     suppress the tongues while staying out of the way of free flow.
export const DEFAULT_SIM_PARAMS: SimParams = {
  particleCount: 1536,
  gravity: [0, -9.81, 0],
  restitution: 0.1,
  timestep: 1 / 240,
  paused: false,

  particleMass: 0.05,
  smoothingRadius: 0.075,
  restDensity: 1000,
  gasConstant: 1500,
  viscosity: 2.0,
  gamma: 7.0,
  maxPressure: 60_000,

  wallRepulsion: 15.0,
  wallDamping: 3.0,
  wallRange: 0.7,
};

const MAX_SUBSTEP_DT = 1 / 480;

export const SIM_BOX_MIN: [number, number, number] = [-0.42, -0.42, -0.42];
export const SIM_BOX_MAX: [number, number, number] = [0.42, 0.42, 0.42];
const SEED_HALF_EXTENT = 0.21;
// Place the packed-cube seed in the bottom-back-left corner of the sim box,
// padded by `SEED_WALL_PAD` so particles don't spawn inside the wall
// repulsion zone (must be ≥ smoothingRadius to avoid an initial kick).
const SEED_WALL_PAD = 0.1;
const SEED_ORIGIN_MIN: [number, number, number] = [
  SIM_BOX_MIN[0] + SEED_WALL_PAD,
  SIM_BOX_MIN[1] + SEED_WALL_PAD,
  SIM_BOX_MIN[2] + SEED_WALL_PAD,
];

const BOUNDARY_SLOP = 0.002;
const GRID_RESOLUTION: [number, number, number] = [64, 64, 64];

export interface Sim {
  readonly params: SimParams;
  readonly allocation: ParticleAllocation;
  readonly activeCount: number;
  step(
    encoder: GPUCommandEncoder,
    dt: number,
    timestampWrites?: GPUComputePassTimestampWrites,
  ): void;
  reset(newCount?: number): void;
  setParams(patch: Partial<SimParams>): void;
  setActiveCount(n: number): void;
  dispose(): void;
}

export function createSim(
  device: GPUDevice,
  initial: Partial<SimParams> = {},
): Sim {
  const params: SimParams = { ...DEFAULT_SIM_PARAMS, ...initial };
  let alloc = allocateParticles(device, params.particleCount);
  seedPackedCube(alloc, {
    halfExtent: SEED_HALF_EXTENT,
    originMin: SEED_ORIGIN_MIN,
  });

  const integrator: Integrator = createIntegrator(device, alloc);
  let activeCount = params.particleCount;

  function currentIntegratorState(dt: number): IntegratorState {
    return {
      gravity: params.gravity,
      dt,
      boxMin: SIM_BOX_MIN,
      boxMax: SIM_BOX_MAX,
      boundaryDamping: params.restitution,
      boundarySlop: BOUNDARY_SLOP,

      smoothingRadius: params.smoothingRadius,
      restDensity: params.restDensity,
      gasConstant: params.gasConstant,
      viscosity: params.viscosity,
      gamma: params.gamma,
      maxPressure: params.maxPressure,
      particleMass: params.particleMass,
      gridResolution: GRID_RESOLUTION,

      wallRepulsion: params.wallRepulsion,
      wallDamping: params.wallDamping,
      wallRange: params.wallRange,
      activeCount,
    };
  }

  return {
    get params(): SimParams {
      return params;
    },
    get allocation(): ParticleAllocation {
      return alloc;
    },
    get activeCount(): number {
      return activeCount;
    },
    step(encoder, dt, timestampWrites): void {
      if (params.paused || dt <= 0) return;
      if (activeCount <= 0) return;
      const substeps = Math.max(1, Math.ceil(dt / MAX_SUBSTEP_DT));
      const subDt = dt / substeps;
      for (let s = 0; s < substeps; s++) {
        const isLast = s === substeps - 1;
        integrator.encode(
          encoder,
          currentIntegratorState(subDt),
          isLast ? timestampWrites : undefined,
        );
      }
    },
    reset(newCount): void {
      if (newCount !== undefined && newCount !== params.particleCount) {
        params.particleCount = Math.max(64, Math.floor(newCount));
        alloc.gpuBuffer.destroy();
        alloc = allocateParticles(device, params.particleCount);
        integrator.rebindParticles(alloc);
      }
      seedPackedCube(alloc, {
        halfExtent: SEED_HALF_EXTENT,
        originMin: SEED_ORIGIN_MIN,
      });
      activeCount = params.particleCount;
    },
    setParams(patch): void {
      Object.assign(params, patch);
    },
    setActiveCount(n): void {
      activeCount = Math.max(0, Math.min(Math.floor(n), alloc.count));
    },
    dispose(): void {
      integrator.dispose();
      alloc.gpuBuffer.destroy();
    },
  };
}

export type { ParticleAllocation } from './particles';
