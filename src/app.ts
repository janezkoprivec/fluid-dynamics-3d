import tgpu from 'typegpu';
import { acquireGpu, showFatalError } from './util/webgpu';
import { createManagedSurface } from './util/resize';
import { createSim, DEFAULT_SIM_PARAMS, type SimParams } from './sim';
import {
  createRenderer,
  type RenderParams,
} from './render';
import { createGui } from './ui/gui';
import { createStats } from './ui/stats';

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

  const { device, hasTimestampQuery } = await acquireGpu();
  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('failed to acquire WebGPU canvas context');

  const root = tgpu.initFromDevice({ device });
  const surfaceFormat = navigator.gpu.getPreferredCanvasFormat();
  const surface = createManagedSurface(device, context, canvas, surfaceFormat);

  const sim = createSim(root, device, DEFAULT_SIM_PARAMS);
  const renderer = createRenderer(device, surfaceFormat, sim.allocation);

  const detachCamera = renderer.camera.attach(canvas);
  applyAspect();
  surface.onResize(applyAspect);

  function applyAspect(): void {
    const { size } = surface.resources();
    renderer.camera.setAspect(size.width / size.height);
  }

  const stats = createStats(statsHost, timingsHost);
  const gui = createGui(sim.params, renderer.params, {
    onReset(count: number): void {
      sim.reset(count);
      renderer.rebindParticles(sim.allocation);
    },
    onSimChange(next: SimParams): void {
      sim.setParams(next);
    },
    onRenderChange(next: RenderParams): void {
      renderer.setParams(next);
    },
  });

  const timing = createTimingPool(device, hasTimestampQuery);

  let lastT = performance.now();
  let running = true;

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
    sim.step(encoder, dt, simMarks.gpu);

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

  window.addEventListener('beforeunload', () => {
    running = false;
    detachCamera();
    gui.dispose();
    stats.dispose();
    renderer.dispose();
    sim.dispose();
    timing.dispose();
    surface.dispose();
    root.destroy();
  });
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
