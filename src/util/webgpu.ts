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

  const hasTimestampQuery = adapter.features.has('timestamp-query');
  const requiredFeatures: GPUFeatureName[] = [];
  if (hasTimestampQuery) requiredFeatures.push('timestamp-query');

  const device = await adapter.requestDevice({ requiredFeatures });
  device.lost.then((info) => {
    console.error('[gpu] device lost:', info.message);
  });

  logAdapterInfo(adapter, device, hasTimestampQuery);
  return { adapter, device, hasTimestampQuery };
}

function logAdapterInfo(
  adapter: GPUAdapter,
  device: GPUDevice,
  hasTimestampQuery: boolean,
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
    timestampQuery: hasTimestampQuery,
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
