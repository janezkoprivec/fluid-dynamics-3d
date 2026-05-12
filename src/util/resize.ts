export interface SurfaceSize {
  width: number;
  height: number;
  devicePixelRatio: number;
}

export interface SurfaceResources {
  depthTexture: GPUTexture;
  depthView: GPUTextureView;
  size: SurfaceSize;
}

export interface ManagedSurface {
  configure(): void;
  resources(): SurfaceResources;
  dispose(): void;
  onResize(cb: (s: SurfaceSize) => void): () => void;
}

const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';

export function createManagedSurface(
  device: GPUDevice,
  context: GPUCanvasContext,
  canvas: HTMLCanvasElement,
  format: GPUTextureFormat,
): ManagedSurface {
  let depthTexture: GPUTexture | null = null;
  let depthView: GPUTextureView | null = null;
  let size: SurfaceSize = currentSize(canvas);
  const listeners = new Set<(s: SurfaceSize) => void>();

  function rebuild(): void {
    size = currentSize(canvas);
    canvas.width = Math.max(1, Math.floor(size.width * size.devicePixelRatio));
    canvas.height = Math.max(1, Math.floor(size.height * size.devicePixelRatio));

    context.configure({
      device,
      format,
      alphaMode: 'opaque',
    });

    depthTexture?.destroy();
    depthTexture = device.createTexture({
      label: 'depth',
      size: { width: canvas.width, height: canvas.height, depthOrArrayLayers: 1 },
      format: DEPTH_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    depthView = depthTexture.createView();
  }

  function onResize(): void {
    rebuild();
    for (const cb of listeners) cb(size);
  }

  const observer = new ResizeObserver(onResize);
  observer.observe(canvas);
  window.addEventListener('resize', onResize);
  rebuild();

  return {
    configure: rebuild,
    resources(): SurfaceResources {
      if (!depthTexture || !depthView) {
        throw new Error('surface not configured');
      }
      return { depthTexture, depthView, size };
    },
    dispose(): void {
      observer.disconnect();
      window.removeEventListener('resize', onResize);
      depthTexture?.destroy();
    },
    onResize(cb): () => void {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
  };
}

function currentSize(canvas: HTMLCanvasElement): SurfaceSize {
  const rect = canvas.getBoundingClientRect();
  return {
    width: Math.max(1, rect.width),
    height: Math.max(1, rect.height),
    devicePixelRatio: Math.min(window.devicePixelRatio || 1, 2),
  };
}

export { DEPTH_FORMAT };
