import { mat4, vec3 } from 'wgpu-matrix';
import type { Mat4, Vec3 } from 'wgpu-matrix';
import { clamp } from '../util/math';

export interface CameraParams {
  fovY: number;
  near: number;
  far: number;
  minRadius: number;
  maxRadius: number;
}

const DEFAULTS: CameraParams = {
  fovY: (45 * Math.PI) / 180,
  near: 0.05,
  far: 100,
  minRadius: 0.5,
  maxRadius: 30,
};

export interface OrbitCamera {
  view: Mat4;
  proj: Mat4;
  viewProj: Mat4;
  eye: Vec3;
  setAspect(aspect: number): void;
  update(): void;
  attach(target: HTMLElement): () => void;
  reset(): void;
}

export function createOrbitCamera(
  initial: { radius?: number; theta?: number; phi?: number; target?: Vec3 } = {},
): OrbitCamera {
  const params = DEFAULTS;
  let radius = initial.radius ?? 3.5;
  let theta = initial.theta ?? Math.PI * 0.25;
  let phi = initial.phi ?? Math.PI * 0.35;
  const target: Vec3 = initial.target ?? vec3.create(0, 0, 0);

  const view = mat4.create();
  const proj = mat4.create();
  const viewProj = mat4.create();
  const eye = vec3.create();

  let aspect = 1;
  let dirty = true;

  function setAspect(a: number): void {
    aspect = Math.max(0.0001, a);
    dirty = true;
  }

  function compute(): void {
    const sp = Math.sin(phi);
    eye[0] = target[0]! + radius * sp * Math.cos(theta);
    eye[1] = target[1]! + radius * Math.cos(phi);
    eye[2] = target[2]! + radius * sp * Math.sin(theta);
    mat4.lookAt(eye, target, [0, 1, 0], view);
    mat4.perspective(params.fovY, aspect, params.near, params.far, proj);
    mat4.multiply(proj, view, viewProj);
    dirty = false;
  }

  function attach(el: HTMLElement): () => void {
    let dragMode: 'orbit' | 'pan' | null = null;
    let lastX = 0;
    let lastY = 0;

    const onPointerDown = (e: PointerEvent): void => {
      if (e.button === 0) dragMode = 'orbit';
      else if (e.button === 2) dragMode = 'pan';
      else return;
      lastX = e.clientX;
      lastY = e.clientY;
      el.setPointerCapture(e.pointerId);
      e.preventDefault();
    };
    const onPointerMove = (e: PointerEvent): void => {
      if (!dragMode) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      if (dragMode === 'orbit') {
        theta -= dx * 0.005;
        phi = clamp(phi - dy * 0.005, 0.05, Math.PI - 0.05);
      } else {
        const panScale = radius * 0.0015;
        const right = vec3.create(
          Math.cos(theta),
          0,
          Math.sin(theta),
        );
        const up = vec3.create(0, 1, 0);
        const move = vec3.create();
        vec3.scale(right, -dx * panScale, move);
        vec3.addScaled(target, up, dy * panScale, target);
        vec3.add(target, move, target);
      }
      dirty = true;
    };
    const onPointerUp = (e: PointerEvent): void => {
      dragMode = null;
      if (el.hasPointerCapture(e.pointerId)) {
        el.releasePointerCapture(e.pointerId);
      }
    };
    const onWheel = (e: WheelEvent): void => {
      const factor = Math.exp(e.deltaY * 0.001);
      radius = clamp(radius * factor, params.minRadius, params.maxRadius);
      dirty = true;
      e.preventDefault();
    };
    const onContextMenu = (e: MouseEvent): void => e.preventDefault();

    el.addEventListener('pointerdown', onPointerDown);
    el.addEventListener('pointermove', onPointerMove);
    el.addEventListener('pointerup', onPointerUp);
    el.addEventListener('pointercancel', onPointerUp);
    el.addEventListener('wheel', onWheel, { passive: false });
    el.addEventListener('contextmenu', onContextMenu);

    return () => {
      el.removeEventListener('pointerdown', onPointerDown);
      el.removeEventListener('pointermove', onPointerMove);
      el.removeEventListener('pointerup', onPointerUp);
      el.removeEventListener('pointercancel', onPointerUp);
      el.removeEventListener('wheel', onWheel);
      el.removeEventListener('contextmenu', onContextMenu);
    };
  }

  function reset(): void {
    radius = 3.5;
    theta = Math.PI * 0.25;
    phi = Math.PI * 0.35;
    target[0] = 0;
    target[1] = 0;
    target[2] = 0;
    dirty = true;
  }

  compute();

  return {
    view,
    proj,
    viewProj,
    eye,
    setAspect,
    update(): void {
      if (dirty) compute();
    },
    attach,
    reset,
  };
}
