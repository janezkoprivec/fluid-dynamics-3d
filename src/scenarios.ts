// Scenario layer. A scenario configures the sim/surface/renderer for a
// specific demo (pour, splash, …) and ramps particle population over time.
//
// Architecture: scenarios are passive controllers driven by app.ts each
// frame via `tick(dt)`. They don't own resources — they mutate the sim's
// active-particle window and write batches into the existing buffer.

import { mulberry32, writeParticlePucks, seedStash } from './sim/particles';
import type { Sim } from './sim';
import type { Renderer } from './render';
import type { SurfacePipeline } from './surface';

export interface ScenarioCtx {
  sim: Sim;
  surface: SurfacePipeline;
  renderer: Renderer;
}

export interface Scenario {
  readonly id: ScenarioId;
  readonly label: string;
  start(ctx: ScenarioCtx): void;
  tick(ctx: ScenarioCtx, dt: number): void;
  stop?(ctx: ScenarioCtx): void;
}

export type ScenarioId = 'pour10k' | 'pour4k';

interface PourConfig {
  targetCount: number;
  origin: [number, number, number];
  velocity: [number, number, number];
  gridSize: number;          // pipe cross-section: gridSize × gridSize particles
  pipeSpacing: number;        // spacing between grid particles (m)
  pucksPerSecond: number;
  jitter: number;
  rngSeed: number;
}

function createPour(id: ScenarioId, label: string, cfg: PourConfig): Scenario {
  let activeCount = 0;
  let pucksAccumulator = 0;
  let rng = mulberry32(cfg.rngSeed);
  const particlesPerPuck = cfg.gridSize * cfg.gridSize;

  return {
    id,
    label,
    start(ctx: ScenarioCtx): void {
      activeCount = 0;
      pucksAccumulator = 0;
      rng = mulberry32(cfg.rngSeed);

      // Stretch the buffer to the scenario's target count and stash every
      // slot below the sim box. Sim.reset rebuilds the spatial-hash bindings
      // for us, so the next step picks up the new allocation cleanly.
      ctx.sim.reset(cfg.targetCount);
      seedStash(ctx.sim.allocation);
      ctx.sim.setActiveCount(0);
      ctx.renderer.setDrawCount(0);

      // Water-ish SPH. Stiffer pressure response than the default so the
      // pile spreads laterally on impact instead of stacking vertically.
      ctx.sim.setParams({
        paused: false,
        particleMass: 0.02,
        smoothingRadius: 0.07,
        viscosity: 1.0,
        gasConstant: 700,
        restDensity: 1000,
        gamma: 7,
        maxPressure: 60_000,
        timestep: 1 / 360,
      });

      // Iso ≈ 0.55 × rest density. Lower values include too much of the
      // smoothing-kernel "halo" (surface looks puffy, cubic); higher values
      // miss the thin connections between particle clusters (surface breaks
      // up into disconnected blobs).
      ctx.surface.setParams({
        enabled: true,
        isoValue: 550,
        gridResolution: [56, 56, 56],
        maxTriangles: 1_500_000,
      });
      ctx.renderer.setParams({ mode: 'both' });
      // Renderer needs to know about the (possibly reallocated) surface
      // buffers; app.ts owns the binding sync so we don't do it here.
    },

    tick(ctx: ScenarioCtx, dt: number): void {
      if (activeCount >= cfg.targetCount) return;
      pucksAccumulator += dt * cfg.pucksPerSecond;
      if (pucksAccumulator < 1) return;

      const wantPucks = Math.floor(pucksAccumulator);
      pucksAccumulator -= wantPucks;
      const remainingParticles = cfg.targetCount - activeCount;
      const releasePucks = Math.min(
        wantPucks,
        Math.floor(remainingParticles / particlesPerPuck),
      );
      if (releasePucks <= 0) return;

      const result = writeParticlePucks(
        ctx.sim.allocation,
        activeCount,
        releasePucks,
        {
          origin: cfg.origin,
          velocity: cfg.velocity,
          gridSize: cfg.gridSize,
          pipeSpacing: cfg.pipeSpacing,
          pucksPerSecond: cfg.pucksPerSecond,
          gravity: ctx.sim.params.gravity,
          jitter: cfg.jitter,
          rng,
        },
      );
      activeCount += result.particlesWritten;
      ctx.sim.setActiveCount(activeCount);
      ctx.renderer.setDrawCount(activeCount);
    },
  };
}

// Pipe geometry: 4×4 puck @ 28 mm spacing → 84×84 mm cross-section.
// For h=0.07, m=0.02, ρ₀=1000 the SPH equilibrium spacing is ≈28 mm, so
// puck rate × stream velocity = pipe spacing keeps stream density at rest
// without compression. Velocity 2.24 m/s × 80 pucks/s ≈ 28 mm between pucks.
const PRESETS: Record<ScenarioId, Scenario> = {
  pour10k: createPour('pour10k', 'Pour 10k (water)', {
    targetCount: 10_000,
    origin: [0.0, 0.32, 0.0],
    velocity: [0.0, -2.24, 0.0],
    gridSize: 4,
    pipeSpacing: 0.028,
    pucksPerSecond: 80, // 80 × 16 = 1280 particles/sec ⇒ ~7.8 s for 10k
    jitter: 0.002,
    rngSeed: 0xC0FFEE,
  }),
  pour4k: createPour('pour4k', 'Pour 4k (quick)', {
    targetCount: 4_000,
    origin: [0.0, 0.32, 0.0],
    velocity: [0.0, -2.24, 0.0],
    gridSize: 4,
    pipeSpacing: 0.028,
    pucksPerSecond: 80, // ~3 s for 4k
    jitter: 0.002,
    rngSeed: 0xC0FFEE,
  }),
};

export interface ScenarioManager {
  current(): Scenario | null;
  start(id: ScenarioId, ctx: ScenarioCtx): void;
  stop(ctx: ScenarioCtx): void;
  tick(ctx: ScenarioCtx, dt: number): void;
  list(): ReadonlyArray<{ id: ScenarioId; label: string }>;
}

export function createScenarioManager(): ScenarioManager {
  let current: Scenario | null = null;
  return {
    current(): Scenario | null {
      return current;
    },
    start(id, ctx): void {
      if (current?.stop) current.stop(ctx);
      const next = PRESETS[id];
      if (!next) return;
      current = next;
      next.start(ctx);
    },
    stop(ctx): void {
      if (current?.stop) current.stop(ctx);
      current = null;
    },
    tick(ctx, dt): void {
      if (!current) return;
      current.tick(ctx, dt);
    },
    list(): ReadonlyArray<{ id: ScenarioId; label: string }> {
      return Object.values(PRESETS).map((s) => ({ id: s.id, label: s.label }));
    },
  };
}
