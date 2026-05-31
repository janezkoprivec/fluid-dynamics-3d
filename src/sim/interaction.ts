// External-force injection point for the mouse "wand". Lives as a small
// uniform read by the forces pass.
//
// Layout (must match `struct InteractionParams` in forces.wgsl exactly):
//
//   f32[0..2]  mousePos.xyz
//   f32[3]     mouseRadius
//   f32[4..6]  mouseDir.xyz
//   f32[7]     mouseStrength
//   u32[8]     mouseActive   (0 = off, 1 = push, 2 = pull)
//   u32[9..11] _pad
//
// Total: 48 bytes (multiple of 16).

const INTERACTION_BYTE_SIZE = 48;

export type WandMode = 'off' | 'push' | 'pull';

export interface InteractionState {
  mouseActive: WandMode;
  mousePos: [number, number, number];
  mouseRadius: number;
  mouseDir: [number, number, number];
  mouseStrength: number;
}

export const ZERO_INTERACTION: InteractionState = {
  mouseActive: 'off',
  mousePos: [0, 0, 0],
  mouseRadius: 0,
  mouseDir: [0, 0, 0],
  mouseStrength: 0,
};

export interface Interaction {
  readonly buffer: GPUBuffer;
  readonly bindGroupLayout: GPUBindGroupLayout;
  readonly bindGroup: GPUBindGroup;
  write(state: InteractionState): void;
  dispose(): void;
}

export function createInteraction(device: GPUDevice): Interaction {
  const buffer = device.createBuffer({
    label: 'interaction/params',
    size: INTERACTION_BYTE_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const host = new ArrayBuffer(INTERACTION_BYTE_SIZE);
  const f32 = new Float32Array(host);
  const u32 = new Uint32Array(host);

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'interaction/bgl',
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' },
      },
    ],
  });
  const bindGroup = device.createBindGroup({
    label: 'interaction/bg',
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer } }],
  });

  // Start zeroed so any pipeline reading the uniform before app.ts writes
  // it sees "no interaction" instead of garbage.
  device.queue.writeBuffer(buffer, 0, host);

  return {
    buffer,
    bindGroupLayout,
    bindGroup,
    write(state: InteractionState): void {
      f32[0] = state.mousePos[0];
      f32[1] = state.mousePos[1];
      f32[2] = state.mousePos[2];
      f32[3] = state.mouseRadius;
      f32[4] = state.mouseDir[0];
      f32[5] = state.mouseDir[1];
      f32[6] = state.mouseDir[2];
      f32[7] = state.mouseStrength;
      u32[8] = wandModeToU32(state.mouseActive);
      u32[9] = 0;
      u32[10] = 0;
      u32[11] = 0;
      device.queue.writeBuffer(buffer, 0, host);
    },
    dispose(): void {
      buffer.destroy();
    },
  };
}

function wandModeToU32(mode: WandMode): number {
  if (mode === 'push') return 1;
  if (mode === 'pull') return 2;
  return 0;
}
