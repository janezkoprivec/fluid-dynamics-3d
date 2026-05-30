// GPU particle memory contract (must match WGSL `struct Particle` exactly in
// `src/sim/shaders/integrate.wgsl` and `src/render/shaders/pointSprites.wgsl`).
//
// Layout / offsets (bytes), total stride = 64 bytes:
//   0..15  : position.xyz + _pad0
//  16..31  : velocity.xyz + _pad1
//  32..47  : acceleration.xyz + _pad2
//  48..55  : density + pressure
//  56..63  : _pad3.xy

export const PARTICLE_STRIDE = 64;
export const PARTICLE_F32_STRIDE = 16;

export interface ParticleAllocation {
  count: number;
  gpuBuffer: GPUBuffer;
  byteSize: number;
  device: GPUDevice;
}

export function allocateParticles(
  device: GPUDevice,
  count: number,
): ParticleAllocation {
  console.log('allocateParticles', count);
  const byteSize = count * PARTICLE_STRIDE;
  const gpuBuffer = device.createBuffer({
    label: 'particles',
    size: byteSize,
    usage:
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  return { count, gpuBuffer, byteSize, device };
}

// Read the particle buffer back to a host-side Float32Array. The layout is
// `PARTICLE_F32_STRIDE` floats per particle (see byte-offset table at the top
// of this file). Allocates and destroys a transient staging buffer each call —
// intended for diagnostics / parity harnesses, not per-frame use.
export async function readbackParticles(
  alloc: ParticleAllocation,
): Promise<Float32Array> {
  const { device, gpuBuffer, byteSize } = alloc;
  const staging = device.createBuffer({
    label: 'particles/readback',
    size: byteSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder({ label: 'particles/readback' });
  encoder.copyBufferToBuffer(gpuBuffer, 0, staging, 0, byteSize);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const copy = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return copy;
}

export interface SeedOptions {
  halfExtent: number;
  seed?: number;
}

export function seedRandomCube(
  alloc: ParticleAllocation,
  opts: SeedOptions,
): void {
  const { count, device, gpuBuffer } = alloc;
  const half = opts.halfExtent;
  const rng = mulberry32(opts.seed ?? 0xC0FFEE);
  const data = new Float32Array(count * PARTICLE_F32_STRIDE);
  for (let i = 0; i < count; i++) {
    const o = i * PARTICLE_F32_STRIDE;
    data[o + 0] = (rng() * 2 - 1) * half;
    data[o + 1] = (rng() * 2 - 1) * half;
    data[o + 2] = (rng() * 2 - 1) * half;
  }
  device.queue.writeBuffer(gpuBuffer, 0, data);
}

export interface PackedSeedOptions {
  halfExtent: number;
  jitter?: number; // small random noise, e.g. 0.05 * spacing
  seed?: number;
  // Minimum corner of the seed region. Defaults to a cube centered on the
  // origin: [-halfExtent, -halfExtent, -halfExtent].
  originMin?: [number, number, number];
}

export function seedPackedCube(
  alloc: ParticleAllocation,
  opts: PackedSeedOptions,
): void {
  const { count, device, gpuBuffer } = alloc;
  const half = opts.halfExtent;
  const n = Math.ceil(Math.cbrt(count));
  const span = half * 2;
  const spacing = n > 1 ? span / (n - 1) : 0;
  const jitterAmp = (opts.jitter ?? 0) * spacing;
  const rng = mulberry32(opts.seed ?? 0xC0FFEE);
  const [ox, oy, oz] = opts.originMin ?? [-half, -half, -half];

  const data = new Float32Array(count * PARTICLE_F32_STRIDE);
  let i = 0;
  for (let z = 0; z < n && i < count; z++) {
    for (let y = 0; y < n && i < count; y++) {
      for (let x = 0; x < n && i < count; x++) {
        const o = i * PARTICLE_F32_STRIDE;
        const jx = (rng() * 2 - 1) * jitterAmp;
        const jy = (rng() * 2 - 1) * jitterAmp;
        const jz = (rng() * 2 - 1) * jitterAmp;
        data[o + 0] = ox + x * spacing + jx;
        data[o + 1] = oy + y * spacing + jy;
        data[o + 2] = oz + z * spacing + jz;
        i++;
      }
    }
  }

  device.queue.writeBuffer(gpuBuffer, 0, data);
}

// Park every particle at a sentinel position far below the sim box. Inactive
// slots stay here while a scenario gradually activates them; the integrator
// skips them (early-return on idx >= particleCount) so they never feel
// gravity or wall collisions.
export interface StashOptions {
  stashPosition?: [number, number, number];
}

const DEFAULT_STASH_POS: [number, number, number] = [0, -1000, 0];

export function seedStash(
  alloc: ParticleAllocation,
  opts: StashOptions = {},
): void {
  const { count, device, gpuBuffer } = alloc;
  const [sx, sy, sz] = opts.stashPosition ?? DEFAULT_STASH_POS;
  const data = new Float32Array(count * PARTICLE_F32_STRIDE);
  for (let i = 0; i < count; i++) {
    const o = i * PARTICLE_F32_STRIDE;
    data[o + 0] = sx;
    data[o + 1] = sy;
    data[o + 2] = sz;
    // velocity, acceleration, density, pressure all start at 0.
  }
  device.queue.writeBuffer(gpuBuffer, 0, data);
}

// Write a contiguous batch of "newly active" particles starting at the given
// slot. Used by pour/spawn scenarios to ramp the population frame by frame.
//
// Particles are distributed along the imaginary trajectory the pour would
// have produced at a steady rate — particle k of N is treated as having been
// emitted k/rate seconds ago, so its position is at `origin + v0*t + 0.5*g*t²`
// and its velocity is `v0 + g*t`. This keeps SPH density near rest as the
// new particles enter (instead of bunching them at one point, which spikes
// density past rest, maxes out pressure, and explodes the stream).
export interface ParticleBatchOptions {
  position: [number, number, number];
  velocity: [number, number, number];
  ratePerSecond: number; // used to space particles along the stream
  gravity?: [number, number, number];
  jitter?: number; // perpendicular jitter radius (world units)
  rng?: () => number;
}

export function writeParticleBatch(
  alloc: ParticleAllocation,
  startIndex: number,
  batchCount: number,
  opts: ParticleBatchOptions,
): void {
  const { count, device, gpuBuffer } = alloc;
  const start = Math.max(0, Math.floor(startIndex));
  const n = Math.max(0, Math.min(Math.floor(batchCount), count - start));
  if (n === 0) return;

  const rng = opts.rng ?? Math.random;
  const jitter = opts.jitter ?? 0;
  const [px, py, pz] = opts.position;
  const [vx, vy, vz] = opts.velocity;
  const [gx, gy, gz] = opts.gravity ?? [0, 0, 0];
  const dtPerParticle = 1 / Math.max(1, opts.ratePerSecond);

  const data = new Float32Array(n * PARTICLE_F32_STRIDE);
  for (let i = 0; i < n; i++) {
    const o = i * PARTICLE_F32_STRIDE;
    // particle k (0..N-1) was emitted (N-1-k)/rate seconds ago — newest
    // at the source, oldest furthest along the trajectory.
    const t = (n - 1 - i) * dtPerParticle;
    const halfTsq = 0.5 * t * t;
    const cx = px + vx * t + gx * halfTsq;
    const cy = py + vy * t + gy * halfTsq;
    const cz = pz + vz * t + gz * halfTsq;
    const jx = (rng() * 2 - 1) * jitter;
    const jy = (rng() * 2 - 1) * jitter;
    const jz = (rng() * 2 - 1) * jitter;
    data[o + 0] = cx + jx;
    data[o + 1] = cy + jy;
    data[o + 2] = cz + jz;
    data[o + 4] = vx + gx * t;
    data[o + 5] = vy + gy * t;
    data[o + 6] = vz + gz * t;
    // density/pressure left at 0; the first density pass will recompute.
  }
  device.queue.writeBuffer(
    gpuBuffer,
    start * PARTICLE_F32_STRIDE * 4,
    data,
  );
}

// Pipe-style spawn: emit one or more "pucks" of GRID × GRID particles, each
// puck living in a plane perpendicular to the pour velocity. Pucks are
// trajectory-spread so successive puck planes don't overlap with the
// previous puck — i.e. puck k of M was emitted (M − 1 − k) / rate seconds
// ago and has flown velocity · t + ½ g t² further down the pipe by now.
//
// Choosing `pipeSpacing` close to the SPH equilibrium spacing (m, h, rho0
// dependent) keeps the in-stream density near rest density, which avoids
// the pressure spike that bunch-spawn produces.
export interface PuckSpawnOptions {
  origin: [number, number, number];
  velocity: [number, number, number];
  gridSize: number;          // emits gridSize × gridSize particles per puck
  pipeSpacing: number;        // spacing between adjacent grid particles
  pucksPerSecond: number;
  gravity?: [number, number, number];
  jitter?: number;
  rng?: () => number;
}

export interface PuckSpawnResult {
  particlesWritten: number;
}

export function writeParticlePucks(
  alloc: ParticleAllocation,
  startIndex: number,
  pucksRequested: number,
  opts: PuckSpawnOptions,
): PuckSpawnResult {
  const { count, device, gpuBuffer } = alloc;
  const start = Math.max(0, Math.floor(startIndex));
  const perPuck = Math.max(1, Math.floor(opts.gridSize)) ** 2;
  const capacityPucks = Math.floor((count - start) / perPuck);
  const numPucks = Math.max(0, Math.min(Math.floor(pucksRequested), capacityPucks));
  if (numPucks === 0) return { particlesWritten: 0 };

  const rng = opts.rng ?? Math.random;
  const jitter = opts.jitter ?? 0;
  const [px, py, pz] = opts.origin;
  const [vx, vy, vz] = opts.velocity;
  const [gx, gy, gz] = opts.gravity ?? [0, 0, 0];
  const dtPerPuck = 1 / Math.max(1, opts.pucksPerSecond);

  // Build an orthonormal basis (vHat, u, w) with vHat = velocity direction.
  const vLen = Math.hypot(vx, vy, vz);
  const vHat: [number, number, number] =
    vLen > 1e-6 ? [vx / vLen, vy / vLen, vz / vLen] : [0, -1, 0];
  // Reference axis chosen so it's not parallel to vHat.
  const ref: [number, number, number] =
    Math.abs(vHat[1]) < 0.99 ? [0, 1, 0] : [1, 0, 0];
  // u = normalize(cross(vHat, ref))
  const u0 = vHat[1] * ref[2] - vHat[2] * ref[1];
  const u1 = vHat[2] * ref[0] - vHat[0] * ref[2];
  const u2 = vHat[0] * ref[1] - vHat[1] * ref[0];
  const uLen = Math.hypot(u0, u1, u2);
  const u: [number, number, number] =
    uLen > 1e-6 ? [u0 / uLen, u1 / uLen, u2 / uLen] : [1, 0, 0];
  // w = cross(vHat, u) — already unit length.
  const w: [number, number, number] = [
    vHat[1] * u[2] - vHat[2] * u[1],
    vHat[2] * u[0] - vHat[0] * u[2],
    vHat[0] * u[1] - vHat[1] * u[0],
  ];

  const N = Math.max(1, Math.floor(opts.gridSize));
  const offset = (N - 1) / 2;
  const spacing = opts.pipeSpacing;

  const totalParticles = numPucks * perPuck;
  const data = new Float32Array(totalParticles * PARTICLE_F32_STRIDE);
  let writeIdx = 0;
  for (let p = 0; p < numPucks; p++) {
    // Puck index 0 is oldest in this batch (furthest along trajectory),
    // numPucks − 1 is freshest (at origin).
    const t = (numPucks - 1 - p) * dtPerPuck;
    const ht2 = 0.5 * t * t;
    const cx = px + vx * t + gx * ht2;
    const cy = py + vy * t + gy * ht2;
    const cz = pz + vz * t + gz * ht2;
    const vtX = vx + gx * t;
    const vtY = vy + gy * t;
    const vtZ = vz + gz * t;

    for (let r = 0; r < N; r++) {
      const dr = (r - offset) * spacing;
      for (let c = 0; c < N; c++) {
        const dc = (c - offset) * spacing;
        const jx = (rng() * 2 - 1) * jitter;
        const jy = (rng() * 2 - 1) * jitter;
        const jz = (rng() * 2 - 1) * jitter;
        const o = writeIdx * PARTICLE_F32_STRIDE;
        data[o + 0] = cx + u[0] * dc + w[0] * dr + jx;
        data[o + 1] = cy + u[1] * dc + w[1] * dr + jy;
        data[o + 2] = cz + u[2] * dc + w[2] * dr + jz;
        data[o + 4] = vtX;
        data[o + 5] = vtY;
        data[o + 6] = vtZ;
        writeIdx++;
      }
    }
  }

  device.queue.writeBuffer(
    gpuBuffer,
    start * PARTICLE_F32_STRIDE * 4,
    data,
  );

  return { particlesWritten: totalParticles };
}

export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
