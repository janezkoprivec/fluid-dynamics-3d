export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  hasTimestampQuery: boolean;
  onLost(cb: (info: GPUDeviceLostInfo) => void): void;
}

export async function acquireGpu(): Promise<GpuContext> {
  if (!('gpu' in navigator)) {
    throw new Error(
      'WebGPU is not available in this browser. Try Chrome 113+ on a supported GPU.',
    );
  }

  // Default to the integrated GPU. Reasoning:
  // - Firefox Nightly + NVIDIA on Linux/Wayland: device-lost crash within ~20s.
  // - Chrome + NVIDIA on Linux: vkAllocateMemory OOM after exactly 3 swapchain
  //   frames (Dawn/NVIDIA-Vulkan/Linux interaction; not fixable from JS).
  // - The integrated AMD (RADV) and Intel paths are stable across browsers.
  // Override with ?gpu=high to opt back into the discrete adapter.
  void isFirefox;
  const powerPreference = gpuPreferenceOverride() ?? 'low-power';
  const adapter = await navigator.gpu.requestAdapter({ powerPreference });
  if (!adapter) {
    throw new Error('No suitable GPU adapter was found.');
  }

  const wantTimestamp = timestampOptIn();
  const adapterHasTimestamp = adapter.features.has('timestamp-query');
  const requestTimestamp = wantTimestamp && adapterHasTimestamp;
  const requiredFeatures: GPUFeatureName[] = [];
  if (requestTimestamp) requiredFeatures.push('timestamp-query');

  const device = await adapter.requestDevice({ requiredFeatures });

  const hasTimestampQuery = requestTimestamp && device.features.has('timestamp-query');

  const lostListeners = new Set<(info: GPUDeviceLostInfo) => void>();
  device.lost.then((info) => {
    const reason = (info as GPUDeviceLostInfo).reason ?? 'unknown';
    const msg = `WebGPU device lost (${reason}): ${info.message || 'no message'}`;
    console.error('[gpu]', msg);
    showFatalError(msg);
    for (const cb of lostListeners) cb(info);
  });

  device.addEventListener?.('uncapturederror', (ev) => {
    const e = ev as GPUUncapturedErrorEvent;
    console.error('[gpu] uncaptured error:', e.error.message);
  });

  logAdapterInfo(adapter, device, {
    timestampQueryRequested: wantTimestamp,
    timestampQueryAdapter: adapterHasTimestamp,
    timestampQueryEnabled: hasTimestampQuery,
  });
  return {
    adapter,
    device,
    hasTimestampQuery,
    onLost(cb): void {
      lostListeners.add(cb);
    },
  };
}

function timestampOptIn(): boolean {
  try {
    const params = new URLSearchParams(window.location.search);
    const v = params.get('ts');
    return v === '1' || v === 'true';
  } catch {
    return false;
  }
}

function isFirefox(): boolean {
  try {
    return /firefox/i.test(navigator.userAgent);
  } catch {
    return false;
  }
}

function gpuPreferenceOverride(): GPUPowerPreference | undefined {
  try {
    const v = new URLSearchParams(window.location.search).get('gpu');
    if (v === 'high' || v === 'high-performance') return 'high-performance';
    if (v === 'low' || v === 'low-power') return 'low-power';
    return undefined;
  } catch {
    return undefined;
  }
}

interface TimestampStatus {
  timestampQueryRequested: boolean;
  timestampQueryAdapter: boolean;
  timestampQueryEnabled: boolean;
}

function logAdapterInfo(
  adapter: GPUAdapter,
  device: GPUDevice,
  ts: TimestampStatus,
): void {
  const info = adapter.info;
  console.info('[gpu] adapter', {
    vendor: info?.vendor ?? 'unknown',
    architecture: info?.architecture ?? 'unknown',
    device: info?.device ?? 'unknown',
    description: info?.description ?? '',
  });

  const limits = device.limits;
  console.info('[gpu] limits', {
    maxBufferSize: limits.maxBufferSize,
    maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
    maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension,
    maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
  });

  console.info('[gpu] features', {
    timestampQuery: ts,
    available: [...adapter.features].sort(),
  });
}

export function showFatalError(message: string): void {
  const el = document.getElementById('error');
  if (el) {
    el.textContent = message;
    el.classList.add('visible');
  }
  console.error(message);
}
