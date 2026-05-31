import { Pane } from 'tweakpane';
import type { SimParams } from '../sim';
import type { RenderMode, RenderParams } from '../render';
import type { SurfaceGridResolution, SurfaceParams } from '../surface';
import type { ScenarioId } from '../scenarios';

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

type SurfacePresetId = 'custom' | 'g24' | 'g32' | 'g40' | 'g48' | 'g56' | 'g64';

const SURFACE_PRESETS: Record<Exclude<SurfacePresetId, 'custom'>, number> = {
  g24: 24,
  g32: 32,
  g40: 40,
  g48: 48,
  g56: 56,
  g64: 64,
};

const SIM_BOX_HALF_EXTENT = 0.42;
// Pad the MC grid outside the sim box. The field-build pass clamps density
// to zero in this padded region, which lets MC close the iso-surface flush
// against the glass walls (Müller's wetted-contact convention).
const SURFACE_PAD_CELLS = 2;

export interface WandUiConfig {
  radius: number;
  strength: number;
  mode: 'push' | 'pull';
}

export interface GuiCallbacks {
  onReset: (count: number) => void;
  onSimChange: (params: SimParams) => void;
  onRenderChange: (params: RenderParams) => void;
  onSurfaceChange: (params: SurfaceParams) => void;
  onSourceChange: (source: SimulationSource) => void;
  onParityRun?: () => void;
  onScenarioStart: (id: ScenarioId) => void;
  onScenarioStop: () => void;
  onWandChange: (config: WandUiConfig) => void;
}

export const DEFAULT_WAND_UI: WandUiConfig = {
  radius: 0.12,
  strength: 15,
  mode: 'push',
};

export interface Gui {
  refresh(): void;
  dispose(): void;
}

export function createGui(
  sim: SimParams,
  render: RenderParams,
  surface: SurfaceParams,
  source: SimulationSource,
  scenarios: ReadonlyArray<{ id: ScenarioId; label: string }>,
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
    mode: render.mode,
    pointSize: render.pointSize,
    backgroundColor: render.backgroundColor,
    surfaceBaseColor: render.surfaceBaseColor,
    surfaceAmbient: render.surfaceAmbient,
    surfaceShininess: render.surfaceShininess,
    surfaceSpecular: render.surfaceSpecular,
    surfaceFresnel: render.surfaceFresnel,
    surfaceTransmission: render.surfaceTransmission,
    showBoxEdges: render.showBoxEdges,
    showBoxFaces: render.showBoxFaces,
    boxEdgeColor: render.boxEdgeColor,
    boxFaceColor: render.boxFaceColor,
    boxEdgeAlpha: render.boxEdgeAlpha,
    boxFaceAlpha: render.boxFaceAlpha,
  };

  function currentRenderParams(): RenderParams {
    return {
      mode: renderState.mode,
      pointSize: renderState.pointSize,
      backgroundColor: renderState.backgroundColor,
      surfaceBaseColor: renderState.surfaceBaseColor,
      surfaceAmbient: renderState.surfaceAmbient,
      surfaceShininess: renderState.surfaceShininess,
      surfaceSpecular: renderState.surfaceSpecular,
      surfaceFresnel: renderState.surfaceFresnel,
      surfaceTransmission: renderState.surfaceTransmission,
      showBoxEdges: renderState.showBoxEdges,
      showBoxFaces: renderState.showBoxFaces,
      boxEdgeColor: renderState.boxEdgeColor,
      boxFaceColor: renderState.boxFaceColor,
      boxEdgeAlpha: renderState.boxEdgeAlpha,
      boxFaceAlpha: renderState.boxFaceAlpha,
    };
  }

  const surfaceState = {
    enabled: surface.enabled,
    preset: presetIdForResolution(surface.gridResolution),
    gridResolution: surface.gridResolution[0],
    isoValue: surface.isoValue,
    maxTriangles: surface.maxTriangles,
  };

  function gridFromSlider(n: number): SurfaceGridResolution {
    const v = Math.max(8, Math.min(96, Math.floor(n)));
    return [v, v, v];
  }
  function cellSizeFromResolution(res: SurfaceGridResolution): number {
    // Reserve PAD cells on each side of the box for the wall-closure region.
    const span = SIM_BOX_HALF_EXTENT * 2;
    const usable = Math.max(1, res[0] - 1 - 2 * SURFACE_PAD_CELLS);
    return span / usable;
  }
  function gridMinForResolution(res: SurfaceGridResolution): [number, number, number] {
    const cell = cellSizeFromResolution(res);
    const start = -SIM_BOX_HALF_EXTENT - SURFACE_PAD_CELLS * cell;
    return [start, start, start];
  }
  function currentSurfaceParams(): SurfaceParams {
    const res = gridFromSlider(surfaceState.gridResolution);
    return {
      enabled: surfaceState.enabled,
      gridResolution: res,
      gridMin: gridMinForResolution(res),
      cellSize: cellSizeFromResolution(res),
      isoValue: surfaceState.isoValue,
      maxTriangles: surfaceState.maxTriangles,
    };
  }

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

  if (cb.onParityRun) {
    simFolder.addButton({ title: 'Run parity test (CPU↔GPU)' }).on('click', () => {
      cb.onParityRun?.();
    });
  }

  simFolder.on('change', () => {
    if (applyingPreset) return;
    simState.preset = 'custom';
    cb.onSourceChange(sourceState.source);
    cb.onSimChange(currentSimParams());
  });

  const renderFolder = pane.addFolder({ title: 'Render' });
  renderFolder.addBinding(renderState, 'mode', {
    label: 'Mode',
    options: {
      Points: 'points' as RenderMode,
      Surface: 'surface' as RenderMode,
      Both: 'both' as RenderMode,
    },
  });
  renderFolder.addBinding(renderState, 'pointSize', {
    label: 'Point size',
    min: 1,
    max: 24,
    step: 0.5,
  });
  renderFolder.addBinding(renderState, 'backgroundColor', {
    label: 'Background',
  });
  const surfaceShadingFolder = renderFolder.addFolder({
    title: 'Surface shading',
    expanded: false,
  });
  surfaceShadingFolder.addBinding(renderState, 'surfaceBaseColor', {
    label: 'Base color',
  });
  surfaceShadingFolder.addBinding(renderState, 'surfaceAmbient', {
    label: 'Ambient',
    min: 0,
    max: 1,
    step: 0.01,
  });
  surfaceShadingFolder.addBinding(renderState, 'surfaceShininess', {
    label: 'Shininess',
    min: 1,
    max: 128,
    step: 1,
  });
  surfaceShadingFolder.addBinding(renderState, 'surfaceSpecular', {
    label: 'Specular',
    min: 0,
    max: 1,
    step: 0.01,
  });
  surfaceShadingFolder.addBinding(renderState, 'surfaceFresnel', {
    label: 'Fresnel',
    min: 0,
    max: 2,
    step: 0.01,
  });
  surfaceShadingFolder.addBinding(renderState, 'surfaceTransmission', {
    label: 'Transmission',
    min: 0,
    max: 1,
    step: 0.01,
  });
  const boxFolder = renderFolder.addFolder({
    title: 'Glass box',
    expanded: false,
  });
  boxFolder.addBinding(renderState, 'showBoxEdges', { label: 'Edges' });
  boxFolder.addBinding(renderState, 'showBoxFaces', { label: 'Faces' });
  boxFolder.addBinding(renderState, 'boxEdgeColor', { label: 'Edge color' });
  boxFolder.addBinding(renderState, 'boxEdgeAlpha', {
    label: 'Edge alpha',
    min: 0,
    max: 1,
    step: 0.01,
  });
  boxFolder.addBinding(renderState, 'boxFaceColor', { label: 'Face color' });
  boxFolder.addBinding(renderState, 'boxFaceAlpha', {
    label: 'Face alpha',
    min: 0,
    max: 0.5,
    step: 0.005,
  });
  renderFolder.on('change', () => {
    cb.onRenderChange(currentRenderParams());
  });

  const surfaceFolder = pane.addFolder({ title: 'Surface (MC)' });
  surfaceFolder.addBinding(surfaceState, 'enabled', { label: 'Enabled' });
  surfaceFolder.addBinding(surfaceState, 'preset', {
    label: 'Grid preset',
    options: {
      Custom: 'custom' as SurfacePresetId,
      '24³': 'g24' as SurfacePresetId,
      '32³': 'g32' as SurfacePresetId,
      '40³': 'g40' as SurfacePresetId,
      '48³': 'g48' as SurfacePresetId,
      '56³': 'g56' as SurfacePresetId,
      '64³': 'g64' as SurfacePresetId,
    },
  }).on('change', (ev) => {
    const presetId = ev.value as SurfacePresetId;
    if (presetId === 'custom') return;
    const res = SURFACE_PRESETS[presetId];
    if (res === undefined) return;
    applyingPreset = true;
    surfaceState.gridResolution = res;
    pane.refresh();
    applyingPreset = false;
    cb.onSurfaceChange(currentSurfaceParams());
  });
  surfaceFolder.addBinding(surfaceState, 'gridResolution', {
    label: 'Grid n³',
    min: 8,
    max: 96,
    step: 1,
  });
  surfaceFolder.addBinding(surfaceState, 'isoValue', {
    label: 'Iso value',
    min: 0,
    max: 5000,
    step: 5,
  });
  surfaceFolder.addBinding(surfaceState, 'maxTriangles', {
    label: 'Max triangles',
    min: 10_000,
    max: 5_000_000,
    step: 10_000,
  });
  surfaceFolder.on('change', () => {
    if (applyingPreset) return;
    surfaceState.preset = presetIdForN(surfaceState.gridResolution);
    cb.onSurfaceChange(currentSurfaceParams());
  });

  const scenarioFolder = pane.addFolder({ title: 'Scenarios', expanded: true });
  for (const sc of scenarios) {
    scenarioFolder.addButton({ title: `▶ ${sc.label}` }).on('click', () => {
      cb.onScenarioStart(sc.id);
    });
  }
  scenarioFolder.addButton({ title: '■ Stop scenario' }).on('click', () => {
    cb.onScenarioStop();
  });

  // Wand (shift + left-drag pushes water around the picked anchor).
  const wandState = { ...DEFAULT_WAND_UI };
  const wandFolder = pane.addFolder({ title: 'Wand (shift+drag)', expanded: true });
  wandFolder.addBinding(wandState, 'mode', {
    label: 'Mode',
    options: { Push: 'push', Pull: 'pull' },
  });
  wandFolder.addBinding(wandState, 'radius', {
    label: 'Radius',
    min: 0.03,
    max: 0.4,
    step: 0.005,
  });
  wandFolder.addBinding(wandState, 'strength', {
    label: 'Strength',
    min: 0,
    max: 80,
    step: 0.5,
  });
  wandFolder.on('change', () => {
    cb.onWandChange({ ...wandState });
  });

  return {
    refresh(): void {
      // Re-pull state from the live param objects (they get mutated by
      // scenarios). Tweakpane reads the source-of-truth, so this just
      // re-flows the bindings.
      simState.particleCount = sim.particleCount;
      simState.timestep = sim.timestep;
      simState.paused = sim.paused;
      simState.particleMass = sim.particleMass;
      simState.smoothingRadius = sim.smoothingRadius;
      simState.restDensity = sim.restDensity;
      simState.gasConstant = sim.gasConstant;
      simState.viscosity = sim.viscosity;
      simState.gamma = sim.gamma;
      simState.maxPressure = sim.maxPressure;
      renderState.mode = render.mode;
      surfaceState.enabled = surface.enabled;
      surfaceState.gridResolution = surface.gridResolution[0];
      surfaceState.isoValue = surface.isoValue;
      surfaceState.maxTriangles = surface.maxTriangles;
      surfaceState.preset = presetIdForResolution(surface.gridResolution);
      pane.refresh();
    },
    dispose(): void {
      pane.dispose();
    },
  };
}

function presetIdForResolution(res: SurfaceGridResolution): SurfacePresetId {
  return presetIdForN(res[0]);
}

function presetIdForN(n: number): SurfacePresetId {
  for (const [id, value] of Object.entries(SURFACE_PRESETS)) {
    if (value === n) return id as SurfacePresetId;
  }
  return 'custom';
}
