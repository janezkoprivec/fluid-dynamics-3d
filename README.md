# fluid-dynamics-3d

Phase 0 skeleton for a 3D SPH fluid simulation that will later render via
marching cubes. This phase contains no SPH code; it exists to validate the
Vite + TypeScript + TypeGPU + WebGPU toolchain end-to-end and to lock in the
module layout the SPH solver will plug into. ~16 k particles fall under
gravity into a `[-1, 1]^3` box and bounce with a configurable restitution
coefficient, rendered as round, lit point sprites.

## Stack

- Vite + TypeScript (`strict: true`)
- [`typegpu`](https://docs.swmansion.com/TypeGPU/) for typed buffers + root
- [`wgpu-matrix`](https://github.com/greggman/wgpu-matrix) for vec/mat math
- [`tweakpane`](https://tweakpane.github.io/docs/) for the parameter GUI
- [`stats.js`](https://github.com/mrdoob/stats.js) for the FPS overlay

## Getting started

```
npm install
npm run dev
```

Open the printed URL in a WebGPU-capable browser (Chrome 113+ on a supported
GPU). If WebGPU is unavailable a fatal error overlay explains why.

## Controls

- **Drag** to orbit, **wheel** to zoom, **right-drag** to pan.
- The Tweakpane in the top-right exposes particle count, gravity,
  restitution, timestep, pause/reset, point size, and background color.

## Layout

```
src/
  main.ts            entrypoint
  app.ts             WebGPU init, frame loop, timing pool
  sim/               particle buffer + gravity-only integrator (Phase 0)
  render/            orbit camera + point-sprite renderer
  ui/                tweakpane + stats wrapper
  util/              WebGPU acquisition, resize, math helpers
  reference/         empty placeholder for the Phase 1 CPU SPH reference
```

## What "done" looks like (Phase 0 acceptance)

- Particles fall, hit the floor, and settle into a damped pile.
- `Paused`, `Reset`, and live changes to `Gravity` / `Restitution` work
  without reloading.
- Changing `Particle count` and hitting `Reset` reallocates buffers cleanly.
- Window resize keeps the image sharp on HiDPI displays.
- The console logs adapter info and device limits at startup and is silent
  thereafter.

Next phase will add a CPU SPH reference solver under `src/reference/` and
the first GPU SPH pipelines under `src/sim/`.
