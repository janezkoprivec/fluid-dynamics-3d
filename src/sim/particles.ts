import * as d from 'typegpu/data';
import type { TgpuRoot, TgpuBuffer, StorageFlag } from 'typegpu';

// GPU particle memory contract (must match WGSL `struct Particle` exactly in
// `src/sim/shaders/integrate.wgsl` and `src/render/shaders/pointSprites.wgsl`).
//
// Layout / offsets (bytes), total stride = 64 bytes:
//   0..15  : position.xyz + _pad0
//  16..31  : velocity.xyz + _pad1
//  32..47  : acceleration.xyz + _pad2
//  48..55  : density + pressure
//  56..63  : _pad3.xy
//
// Padding fields are intentional for alignment; do not remove/reorder fields
// without updating all matching WGSL structs and upload paths.

export const Particle = d.struct({
  position: d.vec3f,
  _pad0: d.f32,

  velocity: d.vec3f,
  _pad1: d.f32,

  acceleration: d.vec3f,
  _pad2: d.f32,

  density: d.f32,
  pressure: d.f32,
  _pad3: d.vec2f,
});
export type ParticleSchema = typeof Particle;

export type ParticleArraySchema = d.WgslArray<ParticleSchema>;
export type ParticleBuffer = TgpuBuffer<ParticleArraySchema> & StorageFlag;

export interface ParticleAllocation {
  count: number;
  schema: ParticleArraySchema;
  buffer: ParticleBuffer;
  gpuBuffer: GPUBuffer;
  byteSize: number;
}

export function allocateParticles(
  root: TgpuRoot,
  count: number,
): ParticleAllocation {
  const schema = d.arrayOf(Particle, count);
  const buffer = root.createBuffer(schema).$usage('storage');
  const gpuBuffer = root.unwrap(buffer);
  return {
    count,
    schema,
    buffer,
    gpuBuffer,
    byteSize: d.sizeOf(schema),
  };
}

export interface SeedOptions {
  halfExtent: number;
  seed?: number;
}

export function seedRandomCube(
  alloc: ParticleAllocation,
  opts: SeedOptions,
): void {
  const { count } = alloc;
  const half = opts.halfExtent;
  const rng = mulberry32(opts.seed ?? 0xC0FFEE);
  const data = new Array(count);
  for (let i = 0; i < count; i++) {
    data[i] = {
      position: d.vec3f(
        (rng() * 2 - 1) * half,
        (rng() * 2 - 1) * half,
        (rng() * 2 - 1) * half,),
      _pad0: 0,
      velocity: d.vec3f(0, 0, 0),
      _pad1: 0,
      acceleration: d.vec3f(0, 0, 0),
      _pad2: 0,
      density: 0,
      pressure: 0,
      _pad3: d.vec2f(0, 0),
    };
  }
  alloc.buffer.write(data);
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
