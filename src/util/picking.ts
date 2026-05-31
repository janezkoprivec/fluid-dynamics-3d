// Camera-ray picking utilities for mouse interaction.
//
// All matrices use the existing WebGPU convention from `render/camera.ts`:
// clip-space Y is up, z in [0, 1]. NDC near plane is at z=0, far at z=1.

import { mat4, vec3 } from 'wgpu-matrix';
import type { Mat4, Vec3 } from 'wgpu-matrix';

export interface Ray {
  origin: Vec3;
  dir: Vec3; // unit length
}

// Map a canvas-pixel coordinate (origin top-left) into normalized device
// coords for unprojection: NDC X right, Y up.
export function pixelToNdc(
  pixelX: number,
  pixelY: number,
  canvasWidth: number,
  canvasHeight: number,
): [number, number] {
  const ndcX = (pixelX / Math.max(1, canvasWidth)) * 2 - 1;
  const ndcY = 1 - (pixelY / Math.max(1, canvasHeight)) * 2;
  return [ndcX, ndcY];
}

// Build a world-space ray from an NDC point on the near plane through the
// matching point on the far plane. Caller supplies the live `viewProj`; this
// inverts it on every call (cheap; we only call from input events, not the
// frame loop).
export function screenToRay(viewProj: Mat4, ndcX: number, ndcY: number): Ray {
  const inv = mat4.inverse(viewProj);

  const near = unproject(inv, ndcX, ndcY, 0);
  const far = unproject(inv, ndcX, ndcY, 1);

  const dir = vec3.create();
  vec3.sub(far, near, dir);
  vec3.normalize(dir, dir);

  return { origin: near, dir };
}

function unproject(invViewProj: Mat4, x: number, y: number, z: number): Vec3 {
  // (x, y, z, 1) → invViewProj · v → divide by w → world point.
  const m = invViewProj as unknown as Float32Array;
  const wx = m[0]! * x + m[4]! * y + m[8]! * z + m[12]!;
  const wy = m[1]! * x + m[5]! * y + m[9]! * z + m[13]!;
  const wz = m[2]! * x + m[6]! * y + m[10]! * z + m[14]!;
  const ww = m[3]! * x + m[7]! * y + m[11]! * z + m[15]!;
  const w = ww !== 0 ? 1 / ww : 1;
  return vec3.fromValues(wx * w, wy * w, wz * w);
}

// Slab test against an axis-aligned box. Returns the entry distance along
// the ray (≥ 0) if the ray hits the box, or `null` if it misses. If the
// ray origin is already inside the box, returns 0.
export function intersectRayWithBox(
  ray: Ray,
  boxMin: readonly [number, number, number],
  boxMax: readonly [number, number, number],
): number | null {
  const ox = ray.origin[0]!;
  const oy = ray.origin[1]!;
  const oz = ray.origin[2]!;
  const dx = ray.dir[0]!;
  const dy = ray.dir[1]!;
  const dz = ray.dir[2]!;

  let tMin = -Infinity;
  let tMax = Infinity;

  for (let axis = 0; axis < 3; axis++) {
    const o = axis === 0 ? ox : axis === 1 ? oy : oz;
    const d = axis === 0 ? dx : axis === 1 ? dy : dz;
    const lo = boxMin[axis]!;
    const hi = boxMax[axis]!;
    if (Math.abs(d) < 1e-8) {
      if (o < lo || o > hi) return null;
      continue;
    }
    const inv = 1 / d;
    let t1 = (lo - o) * inv;
    let t2 = (hi - o) * inv;
    if (t1 > t2) {
      const tmp = t1;
      t1 = t2;
      t2 = tmp;
    }
    if (t1 > tMin) tMin = t1;
    if (t2 < tMax) tMax = t2;
    if (tMin > tMax) return null;
  }

  if (tMax < 0) return null;
  return Math.max(0, tMin);
}

export function pointOnRay(ray: Ray, t: number): [number, number, number] {
  return [
    ray.origin[0]! + ray.dir[0]! * t,
    ray.origin[1]! + ray.dir[1]! * t,
    ray.origin[2]! + ray.dir[2]! * t,
  ];
}

export function clampPointToBox(
  p: readonly [number, number, number],
  boxMin: readonly [number, number, number],
  boxMax: readonly [number, number, number],
): [number, number, number] {
  return [
    Math.min(boxMax[0]!, Math.max(boxMin[0]!, p[0]!)),
    Math.min(boxMax[1]!, Math.max(boxMin[1]!, p[1]!)),
    Math.min(boxMax[2]!, Math.max(boxMin[2]!, p[2]!)),
  ];
}
