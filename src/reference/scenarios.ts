import type { ReferenceParticle, ReferenceSimParams } from './types';

export class ReferenceScenarios {
  private getBounds(params: ReferenceSimParams): {
    min: { x: number; y: number; z: number };
    max: { x: number; y: number; z: number };
  } {
    return {
      min: params.boxMin ?? { x: -1, y: -1, z: -1 },
      max: params.boxMax ?? { x: 1, y: 1, z: 1 },
    };
  }

  private particleSpacing(params: ReferenceSimParams): number {
    return Math.max(0.01, params.smoothingRadius * 0.55);
  }

  private clamp(v: number, lo: number, hi: number): number {
    return Math.min(hi, Math.max(lo, v));
  }

  private makeParticle(
    x: number,
    y: number,
    z: number,
    params: ReferenceSimParams,
  ): ReferenceParticle {
    return {
      position: { x, y, z },
      velocity: { x: 0, y: 0, z: 0 },
      density: params.restDensity,
      pressure: 0,
    };
  }

  createDamBreak(params: ReferenceSimParams): ReferenceParticle[] {
    const count = Math.max(1, Math.floor(params.particleCount));
    const side = Math.max(1, Math.ceil(Math.cbrt(count)));
    const spacing = this.particleSpacing(params);
    const particles: ReferenceParticle[] = [];
    const { min, max } = this.getBounds(params);
    const margin = spacing;
    const x0 = min.x + margin;
    const y0 = min.y + margin;
    const z0 = min.z + margin;
    const xMax = max.x - margin;
    const yMax = max.y - margin;
    const zMax = max.z - margin;

    for (let x = 0; x < side && particles.length < count; x++) {
      for (let y = 0; y < side && particles.length < count; y++) {
        for (let z = 0; z < side && particles.length < count; z++) {
          particles.push(
            this.makeParticle(
              this.clamp(x0 + x * spacing, x0, xMax),
              this.clamp(y0 + y * spacing, y0, yMax),
              this.clamp(z0 + z * spacing, z0, zMax),
              params,
            ),
          );
        }
      }
    }

    return particles;
  }

  createDrop(params: ReferenceSimParams): ReferenceParticle[] {
    const count = Math.max(1, Math.floor(params.particleCount));
    const spacing = this.particleSpacing(params);
    const side = Math.max(1, Math.ceil(Math.cbrt(count)));
    const { min, max } = this.getBounds(params);
    const center = {
      x: (min.x + max.x) * 0.5,
      y: min.y + (max.y - min.y) * 0.75,
      z: (min.z + max.z) * 0.5,
    };
    const candidates: Array<{ x: number; y: number; z: number; d2: number }> = [];
    const margin = spacing;
    const xLo = min.x + margin;
    const yLo = min.y + margin;
    const zLo = min.z + margin;
    const xHi = max.x - margin;
    const yHi = max.y - margin;
    const zHi = max.z - margin;

    for (let x = 0; x < side; x++) {
      for (let y = 0; y < side; y++) {
        for (let z = 0; z < side; z++) {
          const px = this.clamp(center.x + (x - (side - 1) / 2) * spacing, xLo, xHi);
          const py = this.clamp(center.y + (y - (side - 1) / 2) * spacing, yLo, yHi);
          const pz = this.clamp(center.z + (z - (side - 1) / 2) * spacing, zLo, zHi);
          const dx = px - center.x;
          const dy = py - center.y;
          const dz = pz - center.z;
          candidates.push({ x: px, y: py, z: pz, d2: dx * dx + dy * dy + dz * dz });
        }
      }
    }

    // Pick nearest lattice points to form a compact droplet.
    candidates.sort((a, b) => a.d2 - b.d2);
    return candidates
      .slice(0, count)
      .map((p) => this.makeParticle(p.x, p.y, p.z, params));
  }

  createContainerFill(params: ReferenceSimParams): ReferenceParticle[] {
    const count = Math.max(1, Math.floor(params.particleCount));
    const spacing = this.particleSpacing(params);
    const particles: ReferenceParticle[] = [];
    const { min, max } = this.getBounds(params);
    const margin = spacing;
    const x0 = min.x + margin;
    const y0 = min.y + margin;
    const z0 = min.z + margin;
    const xMax = max.x - margin;
    const yMax = max.y - margin;
    const zMax = max.z - margin;

    const nx = Math.max(1, Math.ceil(Math.sqrt(count)));
    const nz = nx;
    const perLayer = nx * nz;
    const ny = Math.max(1, Math.ceil(count / perLayer));

    for (let y = 0; y < ny && particles.length < count; y++) {
      for (let x = 0; x < nx && particles.length < count; x++) {
        for (let z = 0; z < nz && particles.length < count; z++) {
          particles.push(
            this.makeParticle(
              this.clamp(x0 + x * spacing, x0, xMax),
              this.clamp(y0 + y * spacing, y0, yMax),
              this.clamp(z0 + z * spacing, z0, zMax),
              params,
            ),
          );
        }
      }
    }

    return particles;
  }
}
