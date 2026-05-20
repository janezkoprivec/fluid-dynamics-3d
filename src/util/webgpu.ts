export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  hasTimestampQuery: boolean;
}

export async function acquireGpu(): Promise<GpuContext> {
  if (!('gpu' in navigator)) {
    throw new Error(
      'WebGPU is not available in this browser. Try Chrome 113+ on a supported GPU.',
    );
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  });
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

  device.lost.then((info) => {
    const reason = (info as GPUDeviceLostInfo).reason ?? 'unknown';
    const msg = `WebGPU device lost (${reason}): ${info.message || 'no message'}`;
    console.error('[gpu]', msg);
    showFatalError(msg);
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
  return { adapter, device, hasTimestampQuery };
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
