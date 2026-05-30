import { acquireGpu, showFatalError } from './util/webgpu';
import { createManagedSurface } from './util/resize';
import {
  createSim,
  DEFAULT_SIM_PARAMS,
  SIM_BOX_MAX,
  SIM_BOX_MIN,
  type SimParams,
} from './sim';
import {
  allocateParticles,
  PARTICLE_F32_STRIDE,
  readbackParticles,
} from './sim/particles';
import {
  createRenderer,
  type RenderParams,
} from './render';
import {
  createGui,
  type SimulationSource,
} from './ui/gui';
import type { SurfaceParams } from './surface';
import { createStats } from './ui/stats';
import {
  CpuReferenceSim,
  compareParitySamples,
  DEFAULT_PARITY_TOLERANCE,
  sampleParityFromGpuBuffer,
  sampleParityFromReference,
  type ParityComparison,
  type ReferenceParticle,
  type ReferenceSimParams,
} from './reference';
import { createSurfacePipeline } from './surface';
import { createScenarioManager, type ScenarioId } from './scenarios';

const PARITY_STEPS = 600;
const PARITY_SAMPLE_EVERY = 30;
const PARITY_DT = 1 / 240;

export interface App {
  start(): Promise<void>;
}

export function createApp(): App {
  return {
    async start(): Promise<void> {
      try {
        await run();
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        showFatalError(msg);
      }
    },
  };
}

async function run(): Promise<void> {
  const canvas = document.getElementById('gfx') as HTMLCanvasElement | null;
  const statsHost = document.getElementById('stats');
  const timingsHost = document.getElementById('timings');
  if (!canvas || !statsHost || !timingsHost) {
    throw new Error('missing DOM elements: #gfx / #stats / #timings');
  }

  const gpu = await acquireGpu();
  const { device, hasTimestampQuery } = gpu;
  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('failed to acquire WebGPU canvas context');

  const surfaceFormat = navigator.gpu.getPreferredCanvasFormat();
  const surface = createManagedSurface(device, context, canvas, surfaceFormat);

  const sim = createSim(device, DEFAULT_SIM_PARAMS);
  const renderer = createRenderer(device, surfaceFormat, sim.allocation);
  const surfacePipeline = createSurfacePipeline(device, sim.allocation, () => ({
    smoothingRadius: sim.params.smoothingRadius,
    particleMass: sim.params.particleMass,
    restDensity: sim.params.restDensity,
    // Use activeCount, not buffer capacity — otherwise the surface
    // field-build sums density over stashed slots too, which (if anything
    // ever races those slots' positions) gives ghost geometry mid-air.
    particleCount: sim.activeCount,
  }));
  syncSurfaceBinding();
  let source: SimulationSource = 'gpu';

  const scenarioManager = createScenarioManager();
  function scenarioCtx(): { sim: typeof sim; surface: typeof surfacePipeline; renderer: typeof renderer } {
    return { sim, surface: surfacePipeline, renderer };
  }

  function syncSurfaceBinding(): void {
    if (surfacePipeline.params.enabled) {
      renderer.rebindSurface(surfacePipeline.meshBinding());
    } else {
      renderer.clearSurfaceBinding();
    }
  }

  let referenceAlloc = allocateParticles(device, sim.params.particleCount);
  let referenceSim = new CpuReferenceSim(toReferenceParams(sim.params));
  uploadReferenceParticles();

  const detachCamera = renderer.camera.attach(canvas);
  applyAspect();
  surface.onResize(applyAspect);

  function applyAspect(): void {
    const { size } = surface.resources();
    renderer.camera.setAspect(size.width / size.height);
  }

  const stats = createStats(statsHost, timingsHost);
  let parityRunning = false;
  const gui = createGui(sim.params, renderer.params, surfacePipeline.params, source, scenarioManager.list(), {
    onReset(count: number): void {
      if (source === 'gpu') {
        sim.reset(count);
        renderer.rebindParticles(sim.allocation);
        surfacePipeline.rebindParticles(sim.allocation);
        syncSurfaceBinding();
        return;
      }
      rebuildReference(count);
    },
    onSimChange(next: SimParams): void {
      sim.setParams(next);
      surfacePipeline.setSimParams({
        smoothingRadius: next.smoothingRadius,
        particleMass: next.particleMass,
        restDensity: next.restDensity,
        particleCount: next.particleCount,
      });

      if (next.particleCount !== referenceAlloc.count) {
        rebuildReference(next.particleCount);
        return;
      }
      referenceSim.setParams(toReferenceParams(next));
      uploadReferenceParticles();
    },
    onRenderChange(next: RenderParams): void {
      renderer.setParams(next);
    },
    onSurfaceChange(next: SurfaceParams): void {
      surfacePipeline.setParams(next);
      // Grid / capacity edits may have allocated new buffers; resync binding.
      syncSurfaceBinding();
    },
    onSourceChange(next: SimulationSource): void {
      source = next;
      renderer.rebindParticles(source === 'gpu' ? sim.allocation : referenceAlloc);
      // The surface pipeline reads the GPU sim buffer; with the CPU reference
      // we have no live particle source for it.
      if (source === 'gpu') {
        syncSurfaceBinding();
      } else {
        renderer.clearSurfaceBinding();
      }
    },
    onParityRun(): void {
      if (parityRunning) return;
      runParityTest().catch((err) => {
        console.error('[Parity] aborted:', err);
      });
    },
    onScenarioStart(id: ScenarioId): void {
      scenarioManager.start(id, scenarioCtx());
      // The scenario may have resized the buffer and toggled the surface;
      // refresh the renderer's view of both.
      renderer.rebindParticles(sim.allocation);
      surfacePipeline.rebindParticles(sim.allocation);
      syncSurfaceBinding();
      renderer.setDrawCount(sim.activeCount);
      // Push the scenario-applied params back to the GUI so the sliders
      // reflect what's actually running.
      gui.refresh();
    },
    onScenarioStop(): void {
      scenarioManager.stop(scenarioCtx());
    },
  });

  const timing = createTimingPool(device, hasTimestampQuery);

  let lastT = performance.now();
  let running = true;

  gpu.onLost(() => {
    // Stop the rAF loop the instant the device dies. Continuing to call into a
    // dead device is what frequently turns a recoverable device-loss into a
    // full content-process crash on Firefox Nightly.
    running = false;
  });

  function frame(now: number): void {
    if (!running) return;
    stats.begin();

    const dtReal = Math.min(0.05, (now - lastT) / 1000);
    lastT = now;
    const dt = sim.params.paused ? 0 : sim.params.timestep || dtReal;

    const { depthView, size } = surface.resources();
    const targetView = context!.getCurrentTexture().createView();

    const encoder = device.createCommandEncoder({ label: 'frame' });

    const simMarks = timing.markSim();
    if (parityRunning) {
      // Parity harness drives both sims itself; the rAF loop only renders.
    } else if (source === 'gpu') {
      if (dt > 0) scenarioManager.tick(scenarioCtx(), dt);
      sim.step(encoder, dt, simMarks.gpu);
    } else if (dt > 0) {
      referenceSim.step(dt);
      uploadReferenceParticles();
    }

    if (source === 'gpu') {
      surfacePipeline.update(encoder);
    }

    const renderMarks = timing.markRender();
    renderer.draw(
      encoder,
      targetView,
      depthView,
      {
        width: size.width * size.devicePixelRatio,
        height: size.height * size.devicePixelRatio,
      },
      renderMarks.gpu,
    );

    timing.resolveAndRead(encoder);
    device.queue.submit([encoder.finish()]);

    simMarks.endCpu();
    renderMarks.endCpu();
    const [simMs, renderMs] = timing.latest();
    stats.setTimings(simMs, renderMs);

    stats.end();
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);

  let disposed = false;
  function teardown(): void {
    if (disposed) return;
    disposed = true;
    running = false;
    detachCamera();
    gui.dispose();
    stats.dispose();
    renderer.dispose();
    surfacePipeline.dispose();
    sim.dispose();
    referenceAlloc.gpuBuffer.destroy();
    timing.dispose();
    surface.dispose();
    try { context!.unconfigure(); } catch { /* may already be torn down */ }
    try { device.destroy(); } catch { /* already lost */ }
  }
  // pagehide is far more reliable than beforeunload in modern browsers,
  // and fires for both navigation away and full reloads.
  window.addEventListener('pagehide', teardown);
  window.addEventListener('beforeunload', teardown);

  function rebuildReference(count: number): void {
    referenceAlloc.gpuBuffer.destroy();
    referenceAlloc = allocateParticles(device, Math.max(64, Math.floor(count)));
    referenceSim = new CpuReferenceSim(
      toReferenceParams({
        ...sim.params,
        particleCount: referenceAlloc.count,
      }),
    );
    uploadReferenceParticles();
    if (source === 'reference') renderer.rebindParticles(referenceAlloc);
  }

  function uploadReferenceParticles(): void {
    writeParticlesToBuffer(
      referenceAlloc.gpuBuffer,
      referenceSim.getParticles(),
      referenceAlloc.count,
    );
  }

  function writeParticlesToBuffer(
    buffer: GPUBuffer,
    particles: ReadonlyArray<ReferenceParticle>,
    count: number,
  ): void {
    const data = new Float32Array(count * PARTICLE_F32_STRIDE);
    for (let i = 0; i < count; i++) {
      const p = particles[i];
      if (!p) continue;
      const o = i * PARTICLE_F32_STRIDE;
      data[o + 0] = p.position.x;
      data[o + 1] = p.position.y;
      data[o + 2] = p.position.z;
      data[o + 4] = p.velocity.x;
      data[o + 5] = p.velocity.y;
      data[o + 6] = p.velocity.z;
      // acceleration (o+8..10) left zero — GPU integrator overwrites it
      data[o + 12] = p.density ?? 0;
      data[o + 13] = p.pressure ?? 0;
    }
    device.queue.writeBuffer(buffer, 0, data);
  }

  // Lockstep CPU reference and GPU sim from an identical particle seed and
  // emit aggregate (min/avg/max density, max speed, KE, |P|) comparisons every
  // PARITY_SAMPLE_EVERY steps. Validates step-3 GPU math against the CPU
  // oracle; rAF render keeps running but the sim step is suspended.
  //
  // Boundary alignment: GPU normally adds a wall-repulsion band on top of the
  // plane-collision reflect, and CPU damps tangential velocity at impact —
  // neither has a counterpart on the other side. Both are zeroed for the test
  // and restored in `finally`, so we're comparing SPH math (density, pressure,
  // viscosity, integration) and not boundary-model divergence.
  async function runParityTest(): Promise<void> {
    parityRunning = true;
    const wasPaused = sim.params.paused;
    const previousSource = source;

    // Snapshot any param we will mutate. `finally` restores from these.
    const savedSimParams = {
      wallRepulsion: sim.params.wallRepulsion,
      wallDamping: sim.params.wallDamping,
      wallRange: sim.params.wallRange,
    };
    const savedRefParams = {
      maxSubstepDt: referenceSim.getParams().maxSubstepDt,
      boundaryTangentialDamping:
        referenceSim.getParams().boundaryTangentialDamping,
    };

    try {
      // The GPU buffer count is fixed at construction. If the CPU reference
      // was resized away from it, realign before lockstep stepping.
      if (referenceAlloc.count !== sim.allocation.count) {
        rebuildReference(sim.allocation.count);
      }

      // Align stepping and boundary models so the harness measures SPH math,
      // not known-different secondary effects.
      sim.setParams({ wallRepulsion: 0, wallDamping: 0, wallRange: 0 });
      referenceSim.setParams({
        maxSubstepDt: 1 / 480, // matches GPU's MAX_SUBSTEP_DT
        boundaryTangentialDamping: 1.0, // GPU integrator has no tangent decay
      });

      // Use the CPU dam-break scenario as the canonical seed for both sims.
      referenceSim.resetDamBreak();
      const seedParticles = referenceSim.getParticles().map(cloneParticle);
      const count = Math.min(seedParticles.length, sim.allocation.count);
      writeParticlesToBuffer(sim.allocation.gpuBuffer, seedParticles, count);
      // Reset CPU sim to a freshly cloned copy so step 0 is identical to GPU.
      referenceSim.resetWithParticles(seedParticles.map(cloneParticle));

      // Keep the renderer pointing at whichever side the user already had up;
      // the parity loop drives both regardless of `source`.
      const mass = sim.params.particleMass;
      const comparisons: ParityComparison[] = [];

      console.group(
        `[Parity] ${PARITY_STEPS} steps · dt=${PARITY_DT.toFixed(5)} · ` +
          `N=${count} · sample every ${PARITY_SAMPLE_EVERY} ` +
          `(walls zeroed, substep=1/480)`,
      );

      // sim.step bails when paused; force-run for the parity loop. The
      // `finally` block restores the user's previous paused state.
      sim.setParams({ paused: false });

      for (let step = 1; step <= PARITY_STEPS; step++) {
        referenceSim.step(PARITY_DT);

        const encoder = device.createCommandEncoder({
          label: `parity/step-${step}`,
        });
        sim.step(encoder, PARITY_DT);
        device.queue.submit([encoder.finish()]);

        if (step % PARITY_SAMPLE_EVERY === 0 || step === PARITY_STEPS) {
          // readbackParticles submits its own copy command after the sim work;
          // queue ordering guarantees it observes the post-step buffer.
          const gpuData = await readbackParticles(sim.allocation);
          const cpuSample = sampleParityFromReference(
            step,
            referenceSim.getParticles(),
            mass,
          );
          const gpuSample = sampleParityFromGpuBuffer(
            step,
            gpuData,
            count,
            PARTICLE_F32_STRIDE,
            mass,
          );
          const cmp = compareParitySamples(
            cpuSample,
            gpuSample,
            DEFAULT_PARITY_TOLERANCE,
          );
          comparisons.push(cmp);
          logParitySample(cmp);
        }
      }

      const passed = comparisons.filter((c) => c.pass).length;
      console.log(
        `[Parity] ${passed}/${comparisons.length} samples within tolerance`,
      );
      console.groupEnd();
    } finally {
      parityRunning = false;
      sim.setParams({ paused: wasPaused, ...savedSimParams });
      referenceSim.setParams(savedRefParams);
      // Both sims are now in the post-parity state. Honor the previous source
      // so the renderer shows whichever buffer the user was looking at.
      source = previousSource;
      renderer.rebindParticles(
        source === 'gpu' ? sim.allocation : referenceAlloc,
      );
    }
  }
}

function cloneParticle(p: ReferenceParticle): ReferenceParticle {
  return {
    position: { x: p.position.x, y: p.position.y, z: p.position.z },
    velocity: { x: p.velocity.x, y: p.velocity.y, z: p.velocity.z },
    density: p.density,
    pressure: p.pressure,
  };
}

function logParitySample(cmp: ParityComparison): void {
  const m = cmp.metrics;
  const tag = cmp.pass ? 'OK ' : 'OUT';
  const fmt = (r: { cpu: number; gpu: number; relDiff: number }): string =>
    `${r.cpu.toExponential(3)}→${r.gpu.toExponential(3)} (Δ ${(r.relDiff * 100).toFixed(2)}%)`;
  console.log(
    `[Parity ${tag}] step=${String(cmp.step).padStart(4)} ` +
      `ρ_min=${fmt(m.minDensity)} ρ_avg=${fmt(m.avgDensity)} ρ_max=${fmt(m.maxDensity)} ` +
      `|v|max=${fmt(m.maxSpeed)} KE=${fmt(m.kineticEnergy)} |P|=${fmt(m.momentumMag)}`,
  );
}

function toReferenceParams(sim: SimParams): ReferenceSimParams {
  return {
    particleCount: sim.particleCount,
    smoothingRadius: sim.smoothingRadius,
    mass: sim.particleMass,
    restDensity: sim.restDensity,
    gasConstant: sim.gasConstant,
    maxPressure: sim.maxPressure,
    viscosity: sim.viscosity,
    gamma: sim.gamma,
    gravity: {
      x: sim.gravity[0],
      y: sim.gravity[1],
      z: sim.gravity[2],
    },
    boxMin: { x: SIM_BOX_MIN[0], y: SIM_BOX_MIN[1], z: SIM_BOX_MIN[2] },
    boxMax: { x: SIM_BOX_MAX[0], y: SIM_BOX_MAX[1], z: SIM_BOX_MAX[2] },
    boundaryDamping: Math.min(sim.restitution, 0.15),
    boundarySlop: 0.002,
    boundaryTangentialDamping: 0.92,
    maxSubstepDt: 1 / 720,
    logEveryNSteps: 30,
    timestep: Math.min(sim.timestep, 1 / 240),
  };
}

interface PassMarks {
  gpu: GPUComputePassTimestampWrites | GPURenderPassTimestampWrites | undefined;
  endCpu(): void;
}

interface TimingPool {
  markSim(): PassMarks;
  markRender(): PassMarks;
  resolveAndRead(encoder: GPUCommandEncoder): void;
  latest(): [number, number];
  dispose(): void;
}

const SIM_PAIR = 0;
const RENDER_PAIR = 1;
const QUERY_COUNT = 4;

function createTimingPool(
  device: GPUDevice,
  hasTimestampQuery: boolean,
): TimingPool {
  if (!hasTimestampQuery) return createFallbackTimings();

  const querySet = device.createQuerySet({
    label: 'timing/queries',
    type: 'timestamp',
    count: QUERY_COUNT,
  });
  const resolveBuffer = device.createBuffer({
    label: 'timing/resolve',
    size: QUERY_COUNT * 8,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const readPool: GPUBuffer[] = [];
  let inFlight = false;
  let latest: [number, number] = [0, 0];
  let cpuSim = 0;
  let cpuRender = 0;

  function obtainRead(): GPUBuffer {
    return (
      readPool.pop() ??
      device.createBuffer({
        label: 'timing/read',
        size: QUERY_COUNT * 8,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      })
    );
  }

  function pairWrites(pair: number): GPUComputePassTimestampWrites {
    return {
      querySet,
      beginningOfPassWriteIndex: pair * 2,
      endOfPassWriteIndex: pair * 2 + 1,
    };
  }

  function makeMarks(pair: number, store: (ms: number) => void): PassMarks {
    const t0 = performance.now();
    return {
      gpu: pairWrites(pair),
      endCpu(): void {
        store(performance.now() - t0);
      },
    };
  }

  return {
    markSim(): PassMarks {
      return makeMarks(SIM_PAIR, (ms) => (cpuSim = ms));
    },
    markRender(): PassMarks {
      return makeMarks(RENDER_PAIR, (ms) => (cpuRender = ms));
    },
    resolveAndRead(encoder): void {
      encoder.resolveQuerySet(querySet, 0, QUERY_COUNT, resolveBuffer, 0);
      if (inFlight) return;
      const read = obtainRead();
      encoder.copyBufferToBuffer(resolveBuffer, 0, read, 0, QUERY_COUNT * 8);
      inFlight = true;
      read
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          const view = new BigUint64Array(read.getMappedRange().slice(0));
          read.unmap();
          readPool.push(read);
          latest = [nsDiff(view, SIM_PAIR), nsDiff(view, RENDER_PAIR)];
        })
        .catch(() => {})
        .finally(() => {
          inFlight = false;
        });
    },
    latest(): [number, number] {
      if (latest[0] > 0 || latest[1] > 0) return latest;
      return [cpuSim, cpuRender];
    },
    dispose(): void {
      resolveBuffer.destroy();
      for (const b of readPool) b.destroy();
      querySet.destroy();
    },
  };
}

function nsDiff(data: BigUint64Array, pair: number): number {
  const a = data[pair * 2];
  const b = data[pair * 2 + 1];
  if (a === undefined || b === undefined) return 0;
  const diff = b > a ? b - a : 0n;
  return Number(diff) / 1_000_000;
}

function createFallbackTimings(): TimingPool {
  let cpuSim = 0;
  let cpuRender = 0;

  function make(setter: (ms: number) => void): PassMarks {
    const t0 = performance.now();
    return {
      gpu: undefined,
      endCpu(): void {
        setter(performance.now() - t0);
      },
    };
  }

  return {
    markSim(): PassMarks {
      return make((ms) => (cpuSim = ms));
    },
    markRender(): PassMarks {
      return make((ms) => (cpuRender = ms));
    },
    resolveAndRead(): void {},
    latest(): [number, number] {
      return [cpuSim, cpuRender];
    },
    dispose(): void {},
  };
}
