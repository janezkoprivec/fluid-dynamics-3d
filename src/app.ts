import { acquireGpu, showFatalError } from './util/webgpu';
import { createManagedSurface } from './util/resize';
import {
  createSim,
  DEFAULT_SIM_PARAMS,
  SIM_BOX_MAX,
  SIM_BOX_MIN,
  type SimParams,
} from './sim';
import { allocateParticles, PARTICLE_F32_STRIDE } from './sim/particles';
import {
  createRenderer,
  type RenderParams,
} from './render';
import {
  createGui,
  type SimulationSource,
} from './ui/gui';
import { createStats } from './ui/stats';
import { CpuReferenceSim, type ReferenceSimParams } from './reference';

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
  let source: SimulationSource = 'gpu';

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
  const gui = createGui(sim.params, renderer.params, source, {
    onReset(count: number): void {
      if (source === 'gpu') {
        sim.reset(count);
        renderer.rebindParticles(sim.allocation);
        return;
      }
      rebuildReference(count);
    },
    onSimChange(next: SimParams): void {
      sim.setParams(next);
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
    onSourceChange(next: SimulationSource): void {
      source = next;
      renderer.rebindParticles(source === 'gpu' ? sim.allocation : referenceAlloc);
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
    if (source === 'gpu') {
      sim.step(encoder, dt, simMarks.gpu);
    } else if (dt > 0) {
      referenceSim.step(dt);
      uploadReferenceParticles();
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
    const particles = referenceSim.getParticles();
    const count = referenceAlloc.count;
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
    device.queue.writeBuffer(referenceAlloc.gpuBuffer, 0, data);
  }
}

function toReferenceParams(sim: SimParams): ReferenceSimParams {
  return {
    particleCount: sim.particleCount,
    smoothingRadius: 0.1,
    restDensity: 1000,
    gasConstant: 800,
    maxPressure: 40_000,
    viscosity: 2.0,
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
