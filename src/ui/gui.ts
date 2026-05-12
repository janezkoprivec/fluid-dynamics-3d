import { Pane } from 'tweakpane';
import type { SimParams } from '../sim';
import type { RenderParams } from '../render';

export interface GuiCallbacks {
  onReset: (count: number) => void;
  onSimChange: (params: SimParams) => void;
  onRenderChange: (params: RenderParams) => void;
}

export interface Gui {
  dispose(): void;
}

export function createGui(
  sim: SimParams,
  render: RenderParams,
  cb: GuiCallbacks,
): Gui {
  const pane = new Pane({ title: 'fluid-dynamics-3d' });

  const simState = {
    particleCount: sim.particleCount,
    gravity: {
      x: sim.gravity[0],
      y: sim.gravity[1],
      z: sim.gravity[2],
    },
    restitution: sim.restitution,
    timestep: sim.timestep,
    paused: sim.paused,
  };

  const renderState = {
    pointSize: render.pointSize,
    backgroundColor: render.backgroundColor,
  };

  const simFolder = pane.addFolder({ title: 'Simulation' });
  simFolder.addBinding(simState, 'particleCount', {
    label: 'Particle count',
    min: 64,
    max: 262_144,
    step: 64,
  });
  simFolder.addBinding(simState, 'gravity', {
    label: 'Gravity',
    x: { min: -20, max: 20 },
    y: { min: -20, max: 20 },
    z: { min: -20, max: 20 },
  });
  simFolder.addBinding(simState, 'restitution', {
    label: 'Restitution',
    min: 0,
    max: 1,
    step: 0.01,
  });
  simFolder.addBinding(simState, 'timestep', {
    label: 'Timestep',
    min: 1 / 480,
    max: 1 / 30,
    step: 0.0001,
  });
  simFolder.addBinding(simState, 'paused', { label: 'Paused' });
  simFolder.addButton({ title: 'Reset' }).on('click', () => {
    cb.onReset(simState.particleCount);
  });

  simFolder.on('change', () => {
    cb.onSimChange({
      particleCount: simState.particleCount,
      gravity: [simState.gravity.x, simState.gravity.y, simState.gravity.z],
      restitution: simState.restitution,
      timestep: simState.timestep,
      paused: simState.paused,
    });
  });

  const renderFolder = pane.addFolder({ title: 'Render' });
  renderFolder.addBinding(renderState, 'pointSize', {
    label: 'Point size',
    min: 1,
    max: 24,
    step: 0.5,
  });
  renderFolder.addBinding(renderState, 'backgroundColor', {
    label: 'Background',
  });
  renderFolder.on('change', () => {
    cb.onRenderChange({
      pointSize: renderState.pointSize,
      backgroundColor: renderState.backgroundColor,
    });
  });

  return {
    dispose(): void {
      pane.dispose();
    },
  };
}
