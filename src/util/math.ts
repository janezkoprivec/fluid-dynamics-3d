import { vec3 } from 'wgpu-matrix';
import type { Vec3 } from 'wgpu-matrix';

export function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function hexToRgba(hex: string): [number, number, number, number] {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return [0, 0, 0, 1];
  const n = parseInt(m[1]!, 16);
  return [
    ((n >> 16) & 0xff) / 255,
    ((n >> 8) & 0xff) / 255,
    (n & 0xff) / 255,
    1,
  ];
}

export function vec3FromArray(a: readonly [number, number, number]): Vec3 {
  return vec3.fromValues(a[0], a[1], a[2]);
}

export function arrayFromVec3(v: Vec3): [number, number, number] {
  return [v[0]!, v[1]!, v[2]!];
}
