import { SphKernels } from './kernels';
import type {
  ReferenceParticle,
  ReferenceSimParams,
  ReferenceStepStats,
  Vec3,
} from './types';
import { VectorMath } from './math';

export class CpuSphSolver {
  private readonly particles: ReferenceParticle[];
  private readonly params: ReferenceSimParams;
  private readonly particleMass: number;
  private stats: ReferenceStepStats;
  private stepCount = 0;

  private accelerations: Vec3[] = [];

  constructor(
    particles: ReferenceParticle[],
    params: ReferenceSimParams,
    _kernels: SphKernels = new SphKernels(),
  ) {
    this.particles = particles;
    this.params = params;
    this.particleMass =
      params.mass !== undefined && params.mass > 0
        ? params.mass
        : this.estimateParticleMass();
    this.stats = {
      minDensity: params.restDensity,
      maxDensity: params.restDensity,
      avgDensity: params.restDensity,
      maxSpeed: 0,
      totalKineticEnergy: 0,
      centerOfMass: VectorMath.zero(),
      linearMomentum: VectorMath.zero(),
    };
    this.updateMotionStats();
  }

  step(dt = this.params.timestep): void {
    const maxSubstepDt = this.params.maxSubstepDt ?? 1 / 480;
    const rawSubsteps = Math.ceil(Math.max(dt, 0) / Math.max(maxSubstepDt, 1e-6));
    const substeps = Math.min(8, Math.max(1, rawSubsteps));
    const subDt = dt / substeps;
    for (let s = 0; s < substeps; s++) {
      this.computeDensityAndPressure();
      this.computeForces();
      this.integrate(subDt);
      this.applyBoundaries();
    }
    this.updateMotionStats();
    this.stepCount += 1;
    const logEvery = this.params.logEveryNSteps ?? 0;
    if (logEvery > 0 && this.stepCount % logEvery === 0) {
      const s = this.stats;
      console.info(
        `[ReferenceSPH] step=${this.stepCount} rho(min/avg/max)=` +
          `${s.minDensity.toFixed(2)}/${s.avgDensity.toFixed(2)}/${s.maxDensity.toFixed(2)} ` +
          `maxSpeed=${s.maxSpeed.toFixed(3)} ` +
          `KE=${s.totalKineticEnergy.toExponential(3)} ` +
          `|P|=${VectorMath.length(s.linearMomentum).toExponential(3)}`,
      );
    }
  }

  computeDensityAndPressure(): void {
    const mass = this.particleMass;

    let minRho = +Infinity;
    let maxRho = -Infinity;

    let sumRho = 0; 

    for (let i = 0; i < this.particles.length; i++) {
      const pi = this.particles[i];
      let rho = 0; 
      for (let j = 0; j < this.particles.length; j++) {
        const pj = this.particles[j];

        const r = VectorMath.distance(pi.position, pj.position);

        if (r <= this.params.smoothingRadius) {
          rho += mass * SphKernels.poly6(r, this.params.smoothingRadius);
        }
      }

      rho = Math.max(rho, 1e-6);

      const pressure = this.params.gasConstant *
        (Math.pow(rho / this.params.restDensity, this.params.gamma ?? 7) - 1);
      const maxPressure = this.params.maxPressure ?? this.params.gasConstant * 50;
      const p = Math.min(Math.max(pressure, 0), maxPressure);

      pi.density = rho;
      pi.pressure = p;

      minRho = Math.min(minRho, rho);
      maxRho = Math.max(maxRho, rho);
      sumRho += rho;
    
    }

    this.stats = {
      minDensity: minRho,
      maxDensity: maxRho,
      avgDensity: sumRho / this.particles.length,
      maxSpeed: this.stats.maxSpeed,
      totalKineticEnergy: this.stats.totalKineticEnergy,
      centerOfMass: this.stats.centerOfMass,
      linearMomentum: this.stats.linearMomentum,
    };
  }

  computeForces(): void {
    const n = this.particles.length;
    const h = this.params.smoothingRadius;
    const mu = this.params.viscosity;
    const g = this.params.gravity;
    const m = this.particleMass;
    const eps = 1e-6;
    // store this on class, or local and pass to integrate somehow
    this.accelerations = new Array(n);
    for (let i = 0; i < n; i++) {
      const pi = this.particles[i];
      const rhoI = Math.max(pi.density, eps);
      let force = VectorMath.zero();

      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const pj = this.particles[j];
        const rhoJ = Math.max(pj.density, eps);
        const rVec = VectorMath.sub(pi.position, pj.position);
        const r = VectorMath.length(rVec);
        if (r > h || r <= eps) continue;

        // Pressure
        const dir = VectorMath.normalizeSafe(rVec, eps);
        const gradMag = SphKernels.spikyGradient(r, h);
        // With rVec=(xi-xj) and positive grad magnitude, this is (-∇W_ij).
        const negGradW = VectorMath.scale(dir, gradMag);
        const pressureCoeff = (m * (pi.pressure + pj.pressure)) / (2 * rhoJ);
        force = VectorMath.add(force, VectorMath.scale(negGradW, pressureCoeff));

        // Viscosity
        const lapW = SphKernels.viscosityLaplacian(r, h);
        const velDiff = VectorMath.sub(pj.velocity, pi.velocity);
        const viscCoeff = (mu * m * lapW) / rhoJ;
        force = VectorMath.add(force, VectorMath.scale(velDiff, viscCoeff));
      }
      // External force uses density scaling in Müller's formulation.
      force = VectorMath.add(force, VectorMath.scale(g, rhoI));
      this.accelerations[i] = VectorMath.scale(force, 1 / rhoI);
    }
  }

  integrate(dt: number): void {
    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];
      const a = this.accelerations[i] ?? VectorMath.zero();
  
      // Symplectic Euler: velocity first, then position
      p.velocity.x += a.x * dt;
      p.velocity.y += a.y * dt;
      p.velocity.z += a.z * dt;
  
      p.position.x += p.velocity.x * dt;
      p.position.y += p.velocity.y * dt;
      p.position.z += p.velocity.z * dt;
    }
  }

  applyBoundaries(): void {
    const boxMin = this.params.boxMin ?? { x: -1, y: -1, z: -1 };
    const boxMax = this.params.boxMax ?? { x: 1, y: 1, z: 1 };
    const damp = this.params.boundaryDamping ?? 0.4;
    const slop = this.params.boundarySlop ?? 0;
    const tangential = this.params.boundaryTangentialDamping ?? 1;
    for (const p of this.particles) {
      if (p.position.x < boxMin.x) {
        p.position.x = boxMin.x + slop;
        p.velocity.x = Math.abs(p.velocity.x) * damp;
        p.velocity.y *= tangential;
        p.velocity.z *= tangential;
      } else if (p.position.x > boxMax.x) {
        p.position.x = boxMax.x - slop;
        p.velocity.x = -Math.abs(p.velocity.x) * damp;
        p.velocity.y *= tangential;
        p.velocity.z *= tangential;
      }
      if (p.position.y < boxMin.y) {
        p.position.y = boxMin.y + slop;
        p.velocity.y = Math.abs(p.velocity.y) * damp;
        p.velocity.x *= tangential;
        p.velocity.z *= tangential;
      } else if (p.position.y > boxMax.y) {
        p.position.y = boxMax.y - slop;
        p.velocity.y = -Math.abs(p.velocity.y) * damp;
        p.velocity.x *= tangential;
        p.velocity.z *= tangential;
      }
      if (p.position.z < boxMin.z) {
        p.position.z = boxMin.z + slop;
        p.velocity.z = Math.abs(p.velocity.z) * damp;
        p.velocity.x *= tangential;
        p.velocity.y *= tangential;
      } else if (p.position.z > boxMax.z) {
        p.position.z = boxMax.z - slop;
        p.velocity.z = -Math.abs(p.velocity.z) * damp;
        p.velocity.x *= tangential;
        p.velocity.y *= tangential;
      }
    }
  }

  getParticles(): ReadonlyArray<ReferenceParticle> {
    return this.particles;
  }

  getStats(): ReferenceStepStats {
    return this.stats;
  }

  private updateMotionStats(): void {
    const mass = this.particleMass;
    let maxSpeed = 0;
    let totalKineticEnergy = 0;
    let totalMass = 0;
    let com = VectorMath.zero();
    let momentum = VectorMath.zero();
    for (const p of this.particles) {
      const speed = VectorMath.length(p.velocity);
      maxSpeed = Math.max(maxSpeed, speed);
      totalKineticEnergy += 0.5 * mass * speed * speed;
      totalMass += mass;
      com = VectorMath.add(com, VectorMath.scale(p.position, mass));
      momentum = VectorMath.add(momentum, VectorMath.scale(p.velocity, mass));
    }
    if (totalMass > 0) com = VectorMath.scale(com, 1 / totalMass);
    this.stats = {
      ...this.stats,
      maxSpeed,
      totalKineticEnergy,
      centerOfMass: com,
      linearMomentum: momentum,
    };
  }

  private estimateParticleMass(): number {
    const n = this.particles.length;
    if (n === 0) return 1;
    const h = this.params.smoothingRadius;
    if (h <= 0) return 1;
    let sumKernelSums = 0;
    for (let i = 0; i < n; i++) {
      let kernelSum = 0;
      const pi = this.particles[i];
      for (let j = 0; j < n; j++) {
        const pj = this.particles[j];
        const r = VectorMath.distance(pi.position, pj.position);
        if (r <= h) kernelSum += SphKernels.poly6(r, h);
      }
      sumKernelSums += kernelSum;
    }
    const avgKernelSum = sumKernelSums / n;
    return this.params.restDensity / Math.max(avgKernelSum, 1e-6);
  }
}
