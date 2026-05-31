// Wand input layer: shift + left-drag picks an anchor point inside the sim
// box and pushes water along the drag direction (or pulls it toward the
// anchor in "pull" mode).
//
// Lives next to the orbit camera. The camera handler (see render/camera.ts)
// early-returns on shift+left so the two never fight for the pointer.

import {
  clampPointToBox,
  intersectRayWithBox,
  pixelToNdc,
  pointOnRay,
  screenToRay,
} from '../util/picking';
import type { InteractionState, WandMode } from '../sim/interaction';
import type { OrbitCamera } from '../render/camera';

export interface WandConfig {
  radius: number;
  strength: number;
  mode: 'push' | 'pull';
}

export interface WandHandle {
  setConfig(patch: Partial<WandConfig>): void;
  getConfig(): WandConfig;
  // Pulled by the rAF loop. Returns the wand contribution to the
  // interaction uniform (fills only the mouse* fields; rigid* untouched).
  sample(): WandSample;
  dispose(): void;
}

export interface WandSample {
  active: WandMode;
  pos: [number, number, number];
  dir: [number, number, number];
  radius: number;
  strength: number;
}

const ZERO_SAMPLE: WandSample = {
  active: 'off',
  pos: [0, 0, 0],
  dir: [0, 0, 0],
  radius: 0,
  strength: 0,
};

// Number of recent samples to average mouseDir over. Smooths out per-frame
// jitter while still tracking the drag direction with ~50 ms lag at 60 fps.
const DIR_SMOOTHING_FRAMES = 3;

export interface AttachOptions {
  canvas: HTMLCanvasElement;
  camera: OrbitCamera;
  boxMin: readonly [number, number, number];
  boxMax: readonly [number, number, number];
  initial: WandConfig;
}

export function attachWand(opts: AttachOptions): WandHandle {
  const config: WandConfig = { ...opts.initial };

  let dragging = false;
  let lastWorldPos: [number, number, number] | null = null;
  let currentWorldPos: [number, number, number] | null = null;
  // Smoothed direction. We rebuild it from the per-frame deltas observed
  // during pointermove, normalized at sample time.
  const dirHistory: Array<[number, number, number]> = [];

  function pickWorldPos(e: PointerEvent): [number, number, number] | null {
    const rect = opts.canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const [ndcX, ndcY] = pixelToNdc(px, py, rect.width, rect.height);
    const ray = screenToRay(opts.camera.viewProj, ndcX, ndcY);
    const t = intersectRayWithBox(ray, opts.boxMin, opts.boxMax);
    if (t === null) {
      // Miss — keep the previous anchor if we have one so the wand doesn't
      // teleport when the cursor swings briefly off the box.
      return currentWorldPos ?? null;
    }
    const p = pointOnRay(ray, t);
    return clampPointToBox(p, opts.boxMin, opts.boxMax);
  }

  function pushDir(delta: [number, number, number]): void {
    const len = Math.hypot(delta[0], delta[1], delta[2]);
    if (len < 1e-6) return;
    dirHistory.push([delta[0] / len, delta[1] / len, delta[2] / len]);
    if (dirHistory.length > DIR_SMOOTHING_FRAMES) dirHistory.shift();
  }

  function averagedDir(): [number, number, number] {
    if (dirHistory.length === 0) return [0, 0, 0];
    let sx = 0,
      sy = 0,
      sz = 0;
    for (const d of dirHistory) {
      sx += d[0];
      sy += d[1];
      sz += d[2];
    }
    const len = Math.hypot(sx, sy, sz);
    if (len < 1e-6) return [0, 0, 0];
    return [sx / len, sy / len, sz / len];
  }

  function clearDrag(): void {
    dragging = false;
    lastWorldPos = null;
    currentWorldPos = null;
    dirHistory.length = 0;
  }

  const onPointerDown = (e: PointerEvent): void => {
    if (e.button !== 0 || !e.shiftKey) return;
    if (e.pointerType !== 'mouse' && e.pointerType !== 'pen') return;
    const p = pickWorldPos(e);
    if (!p) return;
    dragging = true;
    currentWorldPos = p;
    lastWorldPos = p;
    dirHistory.length = 0;
    opts.canvas.setPointerCapture(e.pointerId);
    e.preventDefault();
  };

  const onPointerMove = (e: PointerEvent): void => {
    if (!dragging) return;
    const p = pickWorldPos(e);
    if (!p) return;
    if (lastWorldPos) {
      pushDir([
        p[0] - lastWorldPos[0],
        p[1] - lastWorldPos[1],
        p[2] - lastWorldPos[2],
      ]);
    }
    lastWorldPos = p;
    currentWorldPos = p;
  };

  const onPointerUp = (e: PointerEvent): void => {
    if (!dragging) return;
    clearDrag();
    if (opts.canvas.hasPointerCapture(e.pointerId)) {
      opts.canvas.releasePointerCapture(e.pointerId);
    }
  };

  opts.canvas.addEventListener('pointerdown', onPointerDown);
  opts.canvas.addEventListener('pointermove', onPointerMove);
  opts.canvas.addEventListener('pointerup', onPointerUp);
  opts.canvas.addEventListener('pointercancel', onPointerUp);

  return {
    setConfig(patch): void {
      Object.assign(config, patch);
    },
    getConfig(): WandConfig {
      return { ...config };
    },
    sample(): WandSample {
      if (!dragging || !currentWorldPos) return ZERO_SAMPLE;
      const dir = averagedDir();
      // Pull mode doesn't need a drag direction — the shader derives the
      // pull vector from particle-to-anchor each frame. We still want the
      // anchor and falloff data though.
      if (config.mode === 'pull') {
        return {
          active: 'pull',
          pos: currentWorldPos,
          dir: [0, 0, 0],
          radius: config.radius,
          strength: config.strength,
        };
      }
      // Push mode with no movement yet → emit "off" so the user sees nothing
      // until the drag actually starts.
      if (dir[0] === 0 && dir[1] === 0 && dir[2] === 0) return ZERO_SAMPLE;
      return {
        active: 'push',
        pos: currentWorldPos,
        dir,
        radius: config.radius,
        strength: config.strength,
      };
    },
    dispose(): void {
      opts.canvas.removeEventListener('pointerdown', onPointerDown);
      opts.canvas.removeEventListener('pointermove', onPointerMove);
      opts.canvas.removeEventListener('pointerup', onPointerUp);
      opts.canvas.removeEventListener('pointercancel', onPointerUp);
    },
  };
}

export function applyWandSample(
  state: InteractionState,
  sample: WandSample,
): void {
  state.mouseActive = sample.active;
  state.mousePos = sample.pos;
  state.mouseDir = sample.dir;
  state.mouseRadius = sample.radius;
  state.mouseStrength = sample.strength;
}
