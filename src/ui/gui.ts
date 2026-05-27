import { Pane } from 'tweakpane';
import type { SimParams } from '../sim';
import type { RenderParams } from '../render';

export type SimulationSource = 'gpu' | 'reference';
type SimPresetId =
  | 'custom'
  | 'cpu512'
  | 'cpu1024'
  | 'cpu1536'
  | 'cpu2048'
  | 'cpu2500';

interface SimPreset {
  particleCount: number;
  timestep: number;
  restitution: number;
}

const SIM_PRESETS: Record<Exclude<SimPresetId, 'custom'>, SimPreset> = {
  cpu512: { particleCount: 512, timestep: 1 / 240, restitution: 0.15 },
  cpu1024: { particleCount: 1024, timestep: 1 / 240, restitution: 0.15 },
  cpu1536: { particleCount: 1536, timestep: 1 / 300, restitution: 0.15 },
  cpu2048: { particleCount: 2048, timestep: 1 / 360, restitution: 0.15 },
  cpu2500: { particleCount: 2500, timestep: 1 / 420, restitution: 0.15 },
};

export interface GuiCallbacks {
  onReset: (count: number) => void;
  onSimChange: (params: SimParams) => void;
  onRenderChange: (params: RenderParams) => void;
  onSourceChange: (source: SimulationSource) => void;
}

export interface Gui {
  dispose(): void;
}

export function createGui(
  sim: SimParams,
  render: RenderParams,
  source: SimulationSource,
  cb: GuiCallbacks,
): Gui {
  const pane = new Pane({ title: 'fluid-dynamics-3d' });
  const sourceState = { source };
  let applyingPreset = false;

  const simState = {
    preset: 'custom' as SimPresetId,
    particleCount: sim.particleCount,
    gravity: {
      x: sim.gravity[0],
      y: sim.gravity[1],
      z: sim.gravity[2],
    },
    restitution: sim.restitution,
    timestep: sim.timestep,
    paused: sim.paused,
    particleMass: sim.particleMass,
    smoothingRadius: sim.smoothingRadius,
    restDensity: sim.restDensity,
    gasConstant: sim.gasConstant,
    viscosity: sim.viscosity,
    gamma: sim.gamma,
    maxPressure: sim.maxPressure,
    wallRepulsion: sim.wallRepulsion,
    wallDamping: sim.wallDamping,
    wallRange: sim.wallRange,
  };

  function currentSimParams(): SimParams {
    return {
      particleCount: simState.particleCount,
      gravity: [simState.gravity.x, simState.gravity.y, simState.gravity.z],
      restitution: simState.restitution,
      timestep: simState.timestep,
      paused: simState.paused,
      particleMass: simState.particleMass,
      smoothingRadius: simState.smoothingRadius,
      restDensity: simState.restDensity,
      gasConstant: simState.gasConstant,
      viscosity: simState.viscosity,
      gamma: simState.gamma,
      maxPressure: simState.maxPressure,
      wallRepulsion: simState.wallRepulsion,
      wallDamping: simState.wallDamping,
      wallRange: simState.wallRange,
    };
  }

  const renderState = {
    pointSize: render.pointSize,
    backgroundColor: render.backgroundColor,
  };

  const simFolder = pane.addFolder({ title: 'Simulation' });
  simFolder.addBinding(sourceState, 'source', {
    label: 'Source',
    options: {
      GPU: 'gpu',
      Reference: 'reference',
    },
  });
  simFolder.addBinding(simState, 'preset', {
    label: 'Preset',
    options: {
      Custom: 'custom',
      'CPU 512': 'cpu512',
      'CPU 1024': 'cpu1024',
      'CPU 1536': 'cpu1536',
      'CPU 2048': 'cpu2048',
      'CPU 2500': 'cpu2500',
    },
  }).on('change', (ev) => {
    const presetId = ev.value as SimPresetId;
    if (presetId === 'custom') return;
    const preset = SIM_PRESETS[presetId];
    if (!preset) return;
    applyingPreset = true;
    simState.particleCount = preset.particleCount;
    simState.timestep = preset.timestep;
    simState.restitution = preset.restitution;
    pane.refresh();
    applyingPreset = false;
    cb.onSimChange(currentSimParams());
    cb.onReset(simState.particleCount);
  });
  simFolder.addBinding(simState, 'particleCount', {
    label: 'Particle count',
    min: 64,
    max: 262_144,
    step: 64,
  }).on('change', (ev) => {
    if (applyingPreset) return;
    if (!ev.last) return;
    cb.onReset(simState.particleCount);
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

  const sphFolder = simFolder.addFolder({ title: 'SPH', expanded: false });
  sphFolder.addBinding(simState, 'particleMass', {
    label: 'Mass',
    min: 0.001,
    max: 1.0,
    step: 0.001,
  });
  sphFolder.addBinding(simState, 'smoothingRadius', {
    label: 'Smoothing h',
    min: 0.01,
    max: 0.5,
    step: 0.005,
  });
  sphFolder.addBinding(simState, 'restDensity', {
    label: 'Rest density',
    min: 100,
    max: 4000,
    step: 10,
  });
  sphFolder.addBinding(simState, 'gasConstant', {
    label: 'Gas constant k',
    min: 1,
    max: 5000,
    step: 1,
  });
  sphFolder.addBinding(simState, 'viscosity', {
    label: 'Viscosity',
    min: 0,
    max: 50,
    step: 0.1,
  });
  sphFolder.addBinding(simState, 'gamma', {
    label: 'Tait gamma',
    min: 1,
    max: 10,
    step: 0.1,
  });
  sphFolder.addBinding(simState, 'maxPressure', {
    label: 'Max pressure',
    min: 1000,
    max: 200_000,
    step: 1000,
  });

  const wallFolder = simFolder.addFolder({ title: 'Walls', expanded: false });
  wallFolder.addBinding(simState, 'wallRepulsion', {
    label: 'Repulsion K',
    min: 0,
    max: 200,
    step: 1,
  });
  wallFolder.addBinding(simState, 'wallDamping', {
    label: 'Damping c',
    min: 0,
    max: 40,
    step: 0.1,
  });
  wallFolder.addBinding(simState, 'wallRange', {
    label: 'Range (× h)',
    min: 0,
    max: 3,
    step: 0.05,
  });

  simFolder.addButton({ title: 'Reset' }).on('click', () => {
    cb.onReset(simState.particleCount);
  });

  simFolder.on('change', () => {
    if (applyingPreset) return;
    simState.preset = 'custom';
    cb.onSourceChange(sourceState.source);
    cb.onSimChange(currentSimParams());
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
