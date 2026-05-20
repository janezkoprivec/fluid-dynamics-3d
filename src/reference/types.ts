export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface ReferenceParticle {
  position: Vec3;
  velocity: Vec3;
  density: number;
  pressure: number;
}

export interface ReferenceSimParams {
  particleCount: number;
  smoothingRadius: number;
  mass?: number;
  restDensity: number;
  gasConstant: number;
  maxPressure?: number;
  viscosity: number;
  gravity: Vec3;
  boxMin?: Vec3;
  boxMax?: Vec3;
  boundaryDamping?: number;
  boundarySlop?: number;
  boundaryTangentialDamping?: number;
  maxSubstepDt?: number;
  logEveryNSteps?: number;
  timestep: number;
  gamma?: number;
}

export interface ReferenceStepStats {
  minDensity: number;
  maxDensity: number;
  avgDensity: number;
  maxSpeed: number;
  totalKineticEnergy: number;
  centerOfMass: Vec3;
  linearMomentum: Vec3;
}
