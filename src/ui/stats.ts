import Stats from 'stats.js';

export interface StatsOverlay {
  begin(): void;
  end(): void;
  setTimings(sim: number, render: number): void;
  dispose(): void;
}

export function createStats(host: HTMLElement, timings: HTMLElement): StatsOverlay {
  const stats = new Stats();
  stats.showPanel(0);
  host.appendChild(stats.dom);

  return {
    begin(): void {
      stats.begin();
    },
    end(): void {
      stats.end();
    },
    setTimings(sim, render): void {
      timings.textContent =
        `sim    ${sim.toFixed(2).padStart(6)} ms\n` +
        `render ${render.toFixed(2).padStart(6)} ms`;
    },
    dispose(): void {
      host.removeChild(stats.dom);
    },
  };
}
