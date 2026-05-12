# Repo Guide — what lives where, and which phase touches what

This repo holds the **planning artifacts** for the seminar:
`SEMINAR_PLAN.md` (the master plan) and `SKELETON_PROMPT.md` (the prompt
that produced the Phase 0 project). The actual code lives next door at
`../fluid-dynamics-3d/` — a Vite + TypeScript + TypeGPU project.

This guide answers three questions:

1. Which file is which?
2. For each phase, which files are **new**, which are **modified**, and
   which are **read-only context**?
3. What's the smallest set of files I should open when starting a phase?

> Convention used below: paths starting with `src/` are inside the
> `fluid-dynamics-3d/` project. Paths starting with `./` or no prefix are
> inside this planning repo.

---

## 1. Repo map (current state, end of Phase 0)

### This planning repo (`3d-spf-fluid-simulation/`)

| File                  | Purpose                                                          |
|-----------------------|------------------------------------------------------------------|
| `SEMINAR_PLAN.md`     | Master plan: deliverables, reading list, phased implementation.  |
| `SKELETON_PROMPT.md`  | Verbatim prompt that produced the Phase 0 skeleton.              |
| `REPO_GUIDE.md`       | This file.                                                       |

### Code repo (`../fluid-dynamics-3d/`)

```
fluid-dynamics-3d/
├── README.md                       project-facing intro
├── package.json / tsconfig.json / vite.config.ts
├── index.html                      canvas + overlay shell
└── src/
    ├── main.ts                     entrypoint, calls app.start()
    ├── style.css                   minimal reset + overlay styles
    ├── app.ts                      WebGPU init, frame loop, timing pool
    ├── sim/
    │   ├── index.ts                public sim API (init, step, reset, params)
    │   ├── particles.ts            particle buffer layout + RNG seeding
    │   ├── integrator.ts           Phase 0: gravity-only integrator pass
    │   └── shaders/
    │       └── integrate.wgsl      gravity + box-wall reflection
    ├── render/
    │   ├── index.ts                public render API
    │   ├── camera.ts               orbit camera + view/proj matrices
    │   ├── pointSprites.ts         particle render pipeline
    │   └── shaders/
    │       └── pointSprites.wgsl   vertex + fragment for round point sprites
    ├── ui/
    │   ├── gui.ts                  Tweakpane setup
    │   └── stats.ts                stats.js wrapper + ms readout
    ├── util/
    │   ├── webgpu.ts               adapter/device acquisition + error overlay
    │   ├── resize.ts               canvas + depth texture resize
    │   └── math.ts                 small helpers on top of wgpu-matrix
    └── reference/                  empty (.gitkeep) — Phase 1 CPU SPH lives here
```

The file layout was chosen so that every later phase **adds files into
the same folders**; the public APIs (`sim/index.ts`, `render/index.ts`,
`ui/gui.ts`) are the stable seams.

---

## 2. Stable seams (don't break these without thinking)

These are the contracts the rest of the code depends on. Touch them
deliberately; prefer **adding fields** over reshaping them.

| Seam                              | What it guarantees                                                                  |
|-----------------------------------|--------------------------------------------------------------------------------------|
| `sim/index.ts` → `Sim` interface  | `step(encoder, dt, timestampWrites?)`, `reset(count?)`, `setParams(patch)`.          |
| `sim/particles.ts` → `Particle`   | `d.struct({ position: vec3f, velocity: vec3f })`. New fields are added here.         |
| `render/index.ts` → `Renderer`    | `draw(encoder, target, depth, viewport, timestampWrites?)`, `rebindParticles()`.     |
| `app.ts` frame loop               | Builds *one* command encoder, runs `sim.step` then `renderer.draw`, then submits.    |
| `ui/gui.ts` callback shape        | `{ onReset, onSimChange, onRenderChange }` — extend, don't reshape.                  |
| `util/webgpu.ts` `acquireGpu()`   | Returns `{ adapter, device, hasTimestampQuery }`. Request new features here.         |

Rule of thumb: if a change would break one of these signatures, plan the
ripple first. Phase 3 will widen `Particle`, Phase 5 will add a second
renderer — both fit through these seams.

---

## 3. Phase-by-phase touch map

For each phase: ✚ = new file, ✎ = modified, → = read context only.
Filenames in `code` font are inside `fluid-dynamics-3d/src/` unless noted.

### Phase 0 — Toolchain bootstrap ✅ done

All files in the layout above were created in this phase. No further
work here; everything listed exists and is wired up.

### Phase 1 — CPU SPH reference

**Goal:** `O(n²)` JS WCSPH oracle. Lives entirely under
`src/reference/`; **must not** touch the GPU pipeline yet.

| File                                          | Why                                                                  |
|-----------------------------------------------|----------------------------------------------------------------------|
| ✚ `reference/cpuSph.ts`                       | Core stepper: density → pressure → forces → integrate.               |
| ✚ `reference/kernels.ts`                      | `poly6`, `spikyGrad`, `viscosityLaplacian`.                          |
| ✚ `reference/types.ts`                        | `Particle = { position, velocity, density, pressure }`.              |
| ✚ `reference/scenarios.ts`                    | Initial conditions: dam-break, drop, container fill.                 |
| ✚ `reference/index.ts`                        | Public API: `createCpuSim`, `step`, `snapshot`.                      |
| ✎ `app.ts`                                    | Add a "renderer source" toggle: GPU particles vs. CPU reference.     |
| ✎ `ui/gui.ts`                                 | Expose `Source: GPU/CPU`, `Scenario`, viscosity, gas constant, `h`.  |
| → `sim/particles.ts`                          | Keep schema in sync conceptually; mirror field names.                |

**Open these first:** `reference/cpuSph.ts` (new), then `app.ts` and
`ui/gui.ts` for wiring. The GPU `sim/` folder stays frozen this phase.

### Phase 2 — GPU port without neighbor search

**Goal:** the same WCSPH solver but on the GPU, with `O(n²)` density
loop in WGSL. Restricted to ~2–4 k particles.

| File                                          | Why                                                                  |
|-----------------------------------------------|----------------------------------------------------------------------|
| ✎ `sim/particles.ts`                          | Extend schema with `density: f32`, `pressure: f32`.                  |
| ✚ `sim/density.ts`                            | Compute pass: `O(n²)` density.                                       |
| ✚ `sim/pressure.ts`                           | EOS (Tait) — small 1D dispatch.                                      |
| ✚ `sim/forces.ts`                             | Pressure-grad + viscosity + gravity.                                 |
| ✎ `sim/integrator.ts`                         | Switch from "gravity only" to "consume `force` buffer".              |
| ✚ `sim/shaders/density.wgsl`                  | Naïve double loop, kernel from `kernels.wgsl`.                       |
| ✚ `sim/shaders/pressure.wgsl`                 | Tait EOS.                                                            |
| ✚ `sim/shaders/forces.wgsl`                   | Sum pressure + viscosity + gravity per particle.                     |
| ✚ `sim/shaders/kernels.wgsl`                  | Shared `poly6`/`spiky`/`viscosity` functions (imported as `?raw`).   |
| ✎ `sim/index.ts`                              | Sequence the new passes; expose new params (`h`, `ρ₀`, `k`, `μ`).    |
| ✎ `ui/gui.ts`                                 | Add SPH-parameter folder.                                            |
| → `reference/*`                               | A/B oracle — compare density histograms after N steps.               |

**Open these first:** `sim/particles.ts`, then create the four new
shader pairs in lockstep. Wire them up from `sim/index.ts` last.

### Phase 3 — Spatial-hash neighbor search

**Goal:** lift particle count to 64 k+ at 60 fps. This phase introduces
the largest new subsystem — a hash + sort + cell-table pipeline.

| File                                          | Why                                                                  |
|-----------------------------------------------|----------------------------------------------------------------------|
| ✚ `sim/neighbors/index.ts`                    | Public API for the neighborhood subsystem.                           |
| ✚ `sim/neighbors/hash.ts`                     | Pass: per particle, write `(cellId, particleIndex)`.                 |
| ✚ `sim/neighbors/radixSort.ts`                | Multi-pass WGSL radix sort (treat as a library; unit-test alone).    |
| ✚ `sim/neighbors/cellTable.ts`                | Build `cellStart[]` / `cellEnd[]` from sorted pairs.                 |
| ✚ `sim/neighbors/reorder.ts`                  | Optional but recommended: SoA reorder for memory coherence.          |
| ✚ `sim/neighbors/shaders/*.wgsl`              | One WGSL file per pass above.                                        |
| ✎ `sim/shaders/density.wgsl`                  | Replace `O(n²)` loop with a 27-cell loop.                            |
| ✎ `sim/shaders/forces.wgsl`                   | Same: 27-cell loop instead of full sweep.                            |
| ✎ `sim/index.ts`                              | Insert `hash → sort → cellTable → reorder` before density.           |
| ✎ `ui/gui.ts`                                 | Add `Cell size (h)`, `Hash visualization` toggle.                    |
| ✎ `render/shaders/pointSprites.wgsl`          | Optional: color-by-`cellId` debug mode.                              |
| → Bridson §3 / Ihmsen §3 / Harada 2007        | Reading list for this phase.                                         |

**Open these first:** unit-test the radix sort end-to-end *before*
plugging it into SPH (it's the highest-risk piece). The seminar plan's
risk table calls this out explicitly.

### Phase 4 — Validation and tuning

**Goal:** confirm correctness with the spatial-hash version and dial in
parameters that look like water.

| File                                          | Why                                                                  |
|-----------------------------------------------|----------------------------------------------------------------------|
| ✚ `tools/abCompare.ts`                        | Run CPU + GPU on identical seed; log density histogram, KE curve.    |
| ✚ `reference/scenarios.ts` (extend)           | Add drop / double dam / container fill / column collapse.            |
| ✎ `sim/index.ts`                              | Plumb scenario selection through `reset()`.                          |
| ✎ `sim/integrator.ts`                         | Adaptive CFL timestep: `dt ≤ 0.4 · h / max_speed`.                   |
| ✎ `ui/gui.ts`                                 | Scenario presets dropdown, CFL safety factor, tuning sliders.        |
| → All shaders                                 | Read-only — should not change in this phase.                         |

**Open these first:** `tools/abCompare.ts` — getting a side-by-side
diff working makes every subsequent tweak measurable.

### Phase 5 — Surface reconstruction (marching cubes)

**Goal:** render an actual fluid surface. Big enough to be its own
subsystem; it lives next to `render/`, not under it.

| File                                          | Why                                                                  |
|-----------------------------------------------|----------------------------------------------------------------------|
| ✚ `surface/index.ts`                          | Public API: `buildSurface(encoder, particles) -> SurfaceMesh`.       |
| ✚ `surface/densityField.ts`                   | Splat kernel into a 3D scalar grid (reuse neighbor table).           |
| ✚ `surface/marchingCubes.ts`                  | CPU MC first (Option A). Read-back density, emit a vertex buffer.    |
| ✚ `surface/mcTables.ts`                       | 256-entry case + edge tables (Paul Bourke's canonical numbers).      |
| ✚ `surface/shaders/densitySplat.wgsl`         | Compute pass: density-field generation.                              |
| ✚ `surface/shaders/marchingCubes.wgsl`        | Option B: two-pass GPU MC. Add when CPU MC works first.              |
| ✚ `render/surfaceMesh.ts`                     | Pipeline: Phong + Fresnel + fake refraction over the MC vertex buf.  |
| ✚ `render/shaders/surface.wgsl`               | Vertex + fragment for the shaded surface.                            |
| ✎ `render/index.ts`                           | Add `mode: 'particles' \| 'surface'`; route `draw()` accordingly.    |
| ✎ `ui/gui.ts`                                 | `Renderer: Particles / Surface`, `Grid resolution`, `Iso value`.     |
| → Lorensen & Cline 1987, Ihmsen §7            | Reading for this phase.                                              |

**Open these first:** `surface/marchingCubes.ts` with the CPU
implementation against a hardcoded analytic field (a sphere) before
plugging in the splatted density.

### Phase 6 — Polish and instrumentation

**Goal:** make the demo presentable and evaluation-ready.

| File                                          | Why                                                                  |
|-----------------------------------------------|----------------------------------------------------------------------|
| ✚ `ui/buttons.ts`                             | Reset, Pause, Step, Screenshot helpers.                              |
| ✚ `ui/csvExport.ts`                           | Collect frame-time samples, emit CSV download.                       |
| ✚ `render/screenshot.ts`                      | Off-screen render + PNG download.                                    |
| ✎ `app.ts`                                    | Wire the timing pool into a per-stage breakdown (hash, sort, …).     |
| ✎ `ui/gui.ts`                                 | Add `Renderer` choice, camera presets, capture controls.             |
| ✚ `render/cameraPresets.ts`                   | Named viewpoints per scenario.                                       |
| → `util/webgpu.ts`                            | Already gates timestamp queries; expand if you add more features.    |

**Open these first:** extend the existing `TimingPool` in `app.ts` from
two pairs (sim/render) to one pair per pass. Everything else hangs off
that breakdown.

### Phase 7 — Evaluation

**Goal:** produce the numbers and screenshots the report needs.

No new source files in the engine; runs use the CSV/screenshot tooling
built in Phase 6.

| File                                          | Why                                                                  |
|-----------------------------------------------|----------------------------------------------------------------------|
| ✚ `evaluation/runMatrix.md`                   | The exact scenarios × particle counts × grid resolutions you ran.    |
| ✚ `evaluation/*.csv`                          | Raw frame-time samples (gitignored is fine; archive them too).       |
| ✚ `evaluation/screenshots/`                   | One per (scenario, particle count, renderer).                        |
| ✚ `evaluation/charts.ipynb` *or* `.ts`        | Plotting script: stacked bar chart of per-stage timing.              |

**Open these first:** `evaluation/runMatrix.md` — write down the
measurement protocol *before* collecting data, so each run is
reproducible.

### Phase 8 — Writeup

**Goal:** the report. Lives in `../report/` or `../seminar-report/`,
**not** in either of these repos. Only the references to source files
and figures cross over.

| File                                          | Why                                                                  |
|-----------------------------------------------|----------------------------------------------------------------------|
| (external)                                    | LaTeX or Markdown report; figures from `evaluation/`.                |
| → `SEMINAR_PLAN.md` §4.8                      | Suggested chapter outline.                                           |

---

## 4. Where do new dependencies go?

| If you're adding…                              | Reach for…                                                          |
|------------------------------------------------|---------------------------------------------------------------------|
| GPU feature gate (subgroups, f16, …)           | `util/webgpu.ts` — request and surface as a boolean.                |
| New shader source                              | A `*.wgsl` next to the module that owns it; import via `?raw`.      |
| Shared WGSL helpers                            | `sim/shaders/kernels.wgsl` (or its surface analogue) — one file.    |
| New CPU helper math                            | `util/math.ts` (small) or its own file under `util/`.               |
| New GUI control                                | `ui/gui.ts` only; never sprinkle Tweakpane calls across modules.    |
| New SPH parameter                              | `sim/index.ts` `SimParams` *and* `ui/gui.ts` *and* the relevant     |
|                                                | shader's uniform struct — keep the three in lockstep.               |
| Debug visualization                            | A flag on `RenderParams`, branch inside `pointSprites.wgsl`, GUI.   |

---

## 5. Recommended editing flow per phase

1. **Re-read** the relevant section of `SEMINAR_PLAN.md` (§4.X) and the
   listed papers from §2 *before* opening the editor.
2. **Sketch** the data flow on paper: which buffers exist, which passes
   read/write what, where the new bindings sit. Pin this in a comment
   at the top of the new module.
3. **Add** new files first (they compile in isolation), then **modify**
   the seam files (`sim/index.ts`, `render/index.ts`, `ui/gui.ts`, or
   `app.ts`) last. Keeps merge surface tiny if you branch.
4. **Validate** before tuning: every phase from 2 onward has an A/B
   check against either the CPU reference (`reference/`) or an
   analytic test (`tools/`). Don't skip them — they save days.
5. **Commit** at every "Definition of done" milestone from
   `SEMINAR_PLAN.md`. Tag the Phase 4 milestone — that's the last
   chance to roll back to a known-good solver before surface work.

---

## 6. Quick reference — "I want to change X, where do I go?"

| Symptom / desire                                | File                                                            |
|-------------------------------------------------|-----------------------------------------------------------------|
| Particles tunnel through the floor              | `sim/shaders/integrate.wgsl` (P0/P2) or `forces.wgsl` (P2+).    |
| FPS is bad                                      | `app.ts` `TimingPool` first; usually `density` or `sort`.       |
| Particle pile looks weird                       | `sim/shaders/kernels.wgsl` (P2+) or `h` / `ρ₀` in `ui/gui.ts`.  |
| Surface has holes                               | `surface/densityField.ts` (smoothing) and `marchingCubes.ts`.   |
| Add a slider                                    | `ui/gui.ts` only.                                               |
| Change camera default                           | `render/camera.ts` (constants at top).                          |
| Change background color default                 | `render/index.ts` `DEFAULT_RENDER_PARAMS`.                      |
| WebGPU not available message                    | `util/webgpu.ts` + `style.css` `#error` block.                  |
| Crashes on resize                               | `util/resize.ts` (depth texture lifetime).                      |

Keep this table extended as you go.
