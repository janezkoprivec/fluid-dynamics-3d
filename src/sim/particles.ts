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
  const byteSize = count * PARTICLE_STRIDE;
  const gpuBuffer = device.createBuffer({
    label: 'particles',
    size: byteSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  return { count, gpuBuffer, byteSize, device };
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

function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
