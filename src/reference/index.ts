import { CpuSphSolver } from './cpuSph';
import { SphKernels } from './kernels';
import { ReferenceScenarios } from './scenarios';
import type {
  ReferenceParticle,
  ReferenceSimParams,
  ReferenceStepStats,
} from './types';

export type ReferenceScenarioName = 'damBreak' | 'drop' | 'containerFill';

export class CpuReferenceSim {
  private readonly params: ReferenceSimParams;
  private readonly kernels: SphKernels;
  private readonly scenarios: ReferenceScenarios;
  private _solver: CpuSphSolver;
  private scenario: ReferenceScenarioName = 'damBreak';

  constructor(
    params: ReferenceSimParams,
    initialParticles: ReferenceParticle[] = [],
    kernels: SphKernels = new SphKernels(),
    scenarios: ReferenceScenarios = new ReferenceScenarios(),
  ) {
    this.params = { ...params };
    this.kernels = kernels;
    this.scenarios = scenarios;
    const particles =
      initialParticles.length > 0
        ? initialParticles
        : this.createScenarioParticles(this.scenario);
    this._solver = new CpuSphSolver(particles, this.params, this.kernels);
  }

  get solver(): CpuSphSolver {
    return this._solver;
  }

  step(dt = this.params.timestep): void {
    this._solver.step(dt);
  }

  getParticles(): ReadonlyArray<ReferenceParticle> {
    return this._solver.getParticles();
  }

  getStats(): ReferenceStepStats {
    return this._solver.getStats();
  }

  getValidationSnapshot(): {
    stats: ReferenceStepStats;
    particleCount: number;
    timestep: number;
  } {
    return {
      stats: this.getStats(),
      particleCount: this.getParticles().length,
      timestep: this.params.timestep,
    };
  }

  getParams(): Readonly<ReferenceSimParams> {
    return this.params;
  }

  setParams(patch: Partial<ReferenceSimParams>): void {
    const shouldRebuildSolver =
      patch.mass !== undefined ||
      patch.restDensity !== undefined ||
      patch.smoothingRadius !== undefined;
    const clonedParticles = shouldRebuildSolver
      ? this._solver.getParticles().map((p) => ({
          position: { ...p.position },
          velocity: { ...p.velocity },
          density: p.density,
          pressure: p.pressure,
        }))
      : [];
    Object.assign(this.params, patch);
    if (shouldRebuildSolver) {
      this._solver = new CpuSphSolver(clonedParticles, this.params, this.kernels);
    }
  }

  resetScenario(name: ReferenceScenarioName = this.scenario): void {
    this.scenario = name;
    this._solver = new CpuSphSolver(
      this.createScenarioParticles(name),
      this.params,
      this.kernels,
    );
  }

  resetDamBreak(): void {
    this.resetScenario('damBreak');
  }

  resetDrop(): void {
    this.resetScenario('drop');
  }

  resetContainerFill(): void {
    this.resetScenario('containerFill');
  }

  resetWithParticles(particles: ReferenceParticle[]): void {
    this._solver = new CpuSphSolver(particles, this.params, this.kernels);
  }

  private createScenarioParticles(name: ReferenceScenarioName): ReferenceParticle[] {
    switch (name) {
      case 'damBreak':
        return this.scenarios.createDamBreak(this.params);
      case 'drop':
        return this.scenarios.createDrop(this.params);
      case 'containerFill':
        return this.scenarios.createContainerFill(this.params);
    }
  }
}

export { CpuSphSolver } from './cpuSph';
export { SphKernels } from './kernels';
export { ReferenceScenarios } from './scenarios';
export type {
  ReferenceParticle,
  ReferenceSimParams,
  ReferenceStepStats,
  Vec3,
} from './types';

// ────────────────────────────────────────────────────────────────────────────
// Parity harness (CPU reference ↔ GPU sim)
//
// Aggregate metrics chosen so CPU/GPU drift is measurable without exact
// per-particle equality (GPU runs in non-deterministic neighbor order, uses a
// slightly different wall model, and accumulates floats in parallel — so
// statistical comparison is the only realistic goal).

export interface ParitySample {
  step: number;
  density: { min: number; avg: number; max: number };
  maxSpeed: number;
  kineticEnergy: number;
  momentumMag: number;
}

export interface ParityToleranceBand {
  abs: number;
  rel: number;
}

export interface ParityTolerance {
  density: ParityToleranceBand;
  maxSpeed: ParityToleranceBand;
  kineticEnergy: ParityToleranceBand;
  momentumMag: ParityToleranceBand;
}

// A sample passes if either the absolute OR relative band is met. Bands are
// calibrated for the parity-test mode in `runParityTest` (boundary models
// aligned, identical substep cadence) — they are tight enough to catch sign /
// unit / kernel-coefficient bugs but loose enough to absorb float
// accumulation-order drift between CPU sequential and GPU parallel reductions.
export const DEFAULT_PARITY_TOLERANCE: ParityTolerance = {
  density: { abs: 10, rel: 0.05 },
  maxSpeed: { abs: 0.05, rel: 0.10 },
  kineticEnergy: { abs: 2e-3, rel: 0.15 },
  momentumMag: { abs: 2e-3, rel: 0.20 },
};

export function sampleParityFromReference(
  step: number,
  particles: ReadonlyArray<import('./types').ReferenceParticle>,
  mass: number,
): ParitySample {
  let minRho = Number.POSITIVE_INFINITY;
  let maxRho = Number.NEGATIVE_INFINITY;
  let sumRho = 0;
  let maxSpeed = 0;
  let ke = 0;
  let px = 0;
  let py = 0;
  let pz = 0;
  for (const p of particles) {
    const rho = p.density;
    if (rho < minRho) minRho = rho;
    if (rho > maxRho) maxRho = rho;
    sumRho += rho;
    const vx = p.velocity.x;
    const vy = p.velocity.y;
    const vz = p.velocity.z;
    const v2 = vx * vx + vy * vy + vz * vz;
    const s = Math.sqrt(v2);
    if (s > maxSpeed) maxSpeed = s;
    ke += 0.5 * mass * v2;
    px += vx * mass;
    py += vy * mass;
    pz += vz * mass;
  }
  const n = Math.max(1, particles.length);
  return {
    step,
    density: { min: minRho, avg: sumRho / n, max: maxRho },
    maxSpeed,
    kineticEnergy: ke,
    momentumMag: Math.sqrt(px * px + py * py + pz * pz),
  };
}

// Mirrors `sampleParityFromReference` but reads the packed GPU particle layout
// (see `src/sim/particles.ts`). Caller passes `stride` (in f32 units) so this
// stays decoupled from the particle layout module.
export function sampleParityFromGpuBuffer(
  step: number,
  data: Float32Array,
  count: number,
  stride: number,
  mass: number,
): ParitySample {
  let minRho = Number.POSITIVE_INFINITY;
  let maxRho = Number.NEGATIVE_INFINITY;
  let sumRho = 0;
  let maxSpeed = 0;
  let ke = 0;
  let px = 0;
  let py = 0;
  let pz = 0;
  for (let i = 0; i < count; i++) {
    const o = i * stride;
    const vx = data[o + 4] ?? 0;
    const vy = data[o + 5] ?? 0;
    const vz = data[o + 6] ?? 0;
    const rho = data[o + 12] ?? 0;
    if (rho < minRho) minRho = rho;
    if (rho > maxRho) maxRho = rho;
    sumRho += rho;
    const v2 = vx * vx + vy * vy + vz * vz;
    const s = Math.sqrt(v2);
    if (s > maxSpeed) maxSpeed = s;
    ke += 0.5 * mass * v2;
    px += vx * mass;
    py += vy * mass;
    pz += vz * mass;
  }
  const n = Math.max(1, count);
  return {
    step,
    density: { min: minRho, avg: sumRho / n, max: maxRho },
    maxSpeed,
    kineticEnergy: ke,
    momentumMag: Math.sqrt(px * px + py * py + pz * pz),
  };
}

export interface ParityMetricResult {
  cpu: number;
  gpu: number;
  absDiff: number;
  relDiff: number;
  pass: boolean;
}

export interface ParityComparison {
  step: number;
  pass: boolean;
  metrics: {
    minDensity: ParityMetricResult;
    avgDensity: ParityMetricResult;
    maxDensity: ParityMetricResult;
    maxSpeed: ParityMetricResult;
    kineticEnergy: ParityMetricResult;
    momentumMag: ParityMetricResult;
  };
}

function checkMetric(
  cpu: number,
  gpu: number,
  band: ParityToleranceBand,
): ParityMetricResult {
  const absDiff = Math.abs(gpu - cpu);
  const denom = Math.max(Math.abs(cpu), 1e-9);
  const relDiff = absDiff / denom;
  const pass = absDiff <= band.abs || relDiff <= band.rel;
  return { cpu, gpu, absDiff, relDiff, pass };
}

export function compareParitySamples(
  cpu: ParitySample,
  gpu: ParitySample,
  tolerance: ParityTolerance = DEFAULT_PARITY_TOLERANCE,
): ParityComparison {
  const metrics = {
    minDensity: checkMetric(cpu.density.min, gpu.density.min, tolerance.density),
    avgDensity: checkMetric(cpu.density.avg, gpu.density.avg, tolerance.density),
    maxDensity: checkMetric(cpu.density.max, gpu.density.max, tolerance.density),
    maxSpeed: checkMetric(cpu.maxSpeed, gpu.maxSpeed, tolerance.maxSpeed),
    kineticEnergy: checkMetric(
      cpu.kineticEnergy,
      gpu.kineticEnergy,
      tolerance.kineticEnergy,
    ),
    momentumMag: checkMetric(
      cpu.momentumMag,
      gpu.momentumMag,
      tolerance.momentumMag,
    ),
  };
  const pass = Object.values(metrics).every((m) => m.pass);
  return { step: cpu.step, pass, metrics };
}
