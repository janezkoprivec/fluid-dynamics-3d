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
