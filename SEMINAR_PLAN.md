# Seminar Plan — 3D SPH Fluid Simulation with Marching Cubes

> Topic 2.11 — *Smoothed-particle hydrodynamics for fluid simulation.*
> Implement a basic SPH fluid solver (density, pressure, viscosity, gravity),
> render with surface reconstruction (marching cubes), evaluate performance
> and visual quality under varying inputs.

This document is the complete plan for the seminar deliverable. Thesis-scope
extensions (PCISPH/IISPH/DFSPH, anisotropic kernels, screen-space rendering,
two-phase flow, etc.) are intentionally out of scope here.

---

## 1. Deliverables

The seminar requires three deliverables. Plan each from the start; do not
leave them for the end.

1. **Working interactive demo.** Real-time 3D SPH simulation with a
   marching-cubes rendered fluid surface, runnable in a modern browser,
   with a GUI exposing the parameters used in evaluation.
2. **Evaluation.** Quantitative measurements of performance (frame time,
   per-stage breakdown) and visual quality across:
   - particle count (e.g. 4k, 16k, 64k, 256k),
   - initial conditions (dam break, drop, double dam break, container fill),
   - timestep and viscosity sweeps,
   - kernel support radius `h` / rest density `ρ₀` choices.
3. **Written report.** Methods, implementation details, results, discussion.
   Slovenian or English per the course standard.

A short demo video (30–60 s) of the best-looking scene is worth recording
once everything is tuned — it's reusable for the thesis later.

---

## 2. Required reading (in this order)

### Must read before writing code

1. **Müller, Charypar, Gross (2003)** — *"Particle-based fluid simulation
   for interactive applications."* The shortest path to a working WCSPH
   solver. Gives the kernel formulas (poly6 for density, spiky for pressure
   gradient, viscosity kernel for the Laplacian) and integration scheme.
2. **Ihmsen et al. (2014)** — *"SPH fluids in computer graphics."*
   (Seminar reference [1].) Read:
   - §2 Fundamentals,
   - §3 Neighborhood search,
   - §4.1 Weakly Compressible SPH (WCSPH),
   - §6 Boundary handling,
   - §7 Surface reconstruction.
   Skim the rest — it's a survey.
3. **Harada, Koshizuka, Kawaguchi (2007)** — *"Smoothed particle
   hydrodynamics on GPUs."* (Seminar reference [2].) Spatial-hash neighbor
   search architecture on GPU. APIs are old; the architecture is current.

### Read alongside implementation

4. **Monaghan (1992)** — *"Smoothed particle hydrodynamics."* (Seminar
   reference [3].) The physics-side foundational reference. Cite, sample
   for background — don't try to read cover to cover.
5. **Lorensen & Cline (1987)** — *"Marching cubes: a high resolution 3D
   surface construction algorithm."* Plus the canonical 256-entry lookup
   table (Paul Bourke's website has clean versions).
6. **Bridson, *Fluid Simulation for Computer Graphics* (2nd ed., 2015).**
   Reference book. Skim chapter 1 (math/notation) and the chapter on
   particle methods. Most useful when something is confusing.

### WebGPU / TypeGPU

7. **WebGPU spec primer** — https://www.w3.org/TR/webgpu/ — read the
   introduction and section on compute pipelines once.
8. **WGSL spec primer** — https://www.w3.org/TR/WGSL/ — keep open as
   reference.
9. **TypeGPU docs** — https://docs.swmansion.com/TypeGPU/ — read
   *Fundamentals → Functions, Bind groups, Pipelines, Resolve/Linker.*

### Optional but useful

10. **Sebastian Lague — "Coding Adventure: Simulating Fluids"** (YouTube).
    Excellent intuition video for SPH; not a substitute for the papers
    but good after them.
11. **NVIDIA GPU Gems 3, Ch. 32 — "Broad-phase collision detection with
    CUDA"** — the original GPU spatial hash exposition; useful for
    implementing the sort + cell-table construction.

---

## 3. Technology stack (decided)

| Layer            | Choice                              | Notes                                                          |
|------------------|-------------------------------------|----------------------------------------------------------------|
| Runtime          | WebGPU                              | Compute shaders, storage buffers, atomics                      |
| Language         | TypeScript (strict)                 | Hot kernels in WGSL                                            |
| GPU wrapper      | TypeGPU                             | Buffer/bind-group typing + linker; raw WGSL for kernels        |
| Linear algebra   | `wgpu-matrix`                       | WebGPU-friendly mat4/vec3 types                                |
| GUI              | `tweakpane`                         | Parameter sliders, presets                                     |
| Perf overlay     | `stats.js`                          | FPS + ms/frame                                                 |
| Bundler / dev    | Vite                                | TS + HMR out of the box                                        |
| Sort             | Hand-written GPU radix sort in WGSL | ~200 lines; cite implementation in report                      |
| Solver variant   | WCSPH (weakly compressible SPH)     | Tait equation of state                                         |
| Neighbor search  | Uniform spatial hash, cell size `h` | Sort + `cellStart[]/cellEnd[]` table                           |
| Time integration | Symplectic Euler (leapfrog optional)| Stable for SPH at small `dt`                                   |
| Surface          | Density splatting → marching cubes  | CPU MC acceptable for seminar; GPU MC if time permits          |
| Surface shading  | Phong + Fresnel + fake refraction   | Plain rasterized triangle mesh                                 |
| Reference solver | Small CPU JS SPH                    | Unit tests + debugging oracle for the GPU version              |

---

## 4. Implementation plan (phased)

Each phase ends in a runnable demo so progress is always visible.
Do not start a phase before the previous one demonstrably works.

### Phase 0 — Toolchain bootstrap

**Goal:** Vite + TypeScript + TypeGPU project that renders ~10k particles
falling under gravity into a box, with a working GUI and stats overlay.

- Scaffold with Vite + TS strict mode.
- Add `typegpu`, `wgpu-matrix`, `tweakpane`, `stats.js`.
- Verify WebGPU adapter request and device limits at startup.
- Module skeleton (`src/sim/`, `src/render/`, `src/util/`, `src/ui/`).
- Camera (orbit controls).
- Particle storage buffer; gravity integrator compute pass.
- Point-sprite render pass (no surface yet).
- Tweakpane wired to a few parameters (gravity, particle count, reset).

> See `SKELETON_PROMPT.md` for the exact prompt to build this phase.

**Definition of done:** open the page, see ~10k particles fall, hit reset,
adjust gravity slider, frame time stable on your machine.

### Phase 1 — CPU SPH reference

**Goal:** small (~1–2k particles) **CPU JavaScript** WCSPH implementation
that produces visually plausible 2D or 3D fluid in real time. This is the
*oracle* you will validate the GPU version against later.

- Particle struct: `position, velocity, density, pressure`.
- Naïve `O(n²)` neighbor search (no spatial hash yet).
- Kernels: poly6 (density), spiky-gradient (pressure), viscosity Laplacian.
- Tait EOS for pressure.
- Forces: pressure + viscosity + gravity.
- Symplectic Euler integration.
- Box boundary: reflective walls with mild damping.
- Render: point sprites, color by density or speed.

**Definition of done:** dam-break scenario splashes correctly, particles
settle, density looks stable around `ρ₀`. Sanity-print min/max density
each frame.

> Keep this around forever — it is your truth source for unit tests in
> Phase 4. Stash it under `src/reference/` and do not delete it.

### Phase 2 — GPU port without neighbor search

**Goal:** same WCSPH solver, on GPU, but still `O(n²)` neighbor search
(direct loop in a compute shader). Restrict to ~2–4k particles.

- Storage buffers for positions, velocities, densities, pressures.
- Compute pipelines:
  1. **Density** (read positions, write density via `O(n²)` loop).
  2. **Pressure** from density via EOS (1D dispatch, no neighbor loop).
  3. **Forces** (read positions/velocities/density/pressure, write force).
  4. **Integrate** (write new positions/velocities, apply box BC).
- Render pass unchanged from Phase 0 (point sprites).
- A/B test against the CPU reference: same seed, same dt, compare density
  histograms after N steps. Should match to within float tolerance.

**Definition of done:** GPU run reproduces CPU run's behavior; frame time
dominated by the `O(n²)` density loop as expected.

### Phase 3 — Spatial-hash neighbor search

**Goal:** lift particle count to 64k+ at 60fps.

Pipeline (each step is a separate compute pass):

1. **Hash:** per particle, compute `cellId = hash(floor(pos/h))`. Write
   `(cellId, particleIndex)` pairs.
2. **Sort:** GPU radix sort by `cellId`.
3. **Build cell table:** per cell, find `cellStart[cellId]` and
   `cellEnd[cellId]` by scanning neighbor pairs in the sorted array.
4. **Reorder** (optional but recommended): copy particle data into
   sorted order so subsequent passes have coherent memory access.
5. **Use the table** in the density + force passes (loop only over the
   27 surrounding cells in 3D).

Sub-tasks:

- Implement and unit-test the radix sort in isolation (sorting random
  `u32`s) before integrating with SPH.
- Visualization mode: color particles by `cellId` to debug hashing.

**Definition of done:** 64k particles at 60fps on your dev machine;
profile shows density + force passes scale roughly linearly with `n`.

### Phase 4 — Validation and tuning

**Goal:** confirm correctness, then make it look good.

- Validation: re-run the CPU/GPU comparison from Phase 2 with the
  spatial-hash version. Numerical drift is OK; *statistical* equivalence
  is required (density histogram, kinetic energy curve).
- Tune `h`, `ρ₀`, gas constant, viscosity until a dam break looks like
  water and is stable for >30 s wall-clock without exploding.
- Add the rest of the initial-condition presets (drop, double dam,
  container fill, column collapse).
- Add adaptive timestep (CFL: `dt ≤ 0.4 · h / max_speed`).

**Definition of done:** the demo runs all preset scenarios robustly;
particles never tunnel through the floor; energy curve is bounded.

### Phase 5 — Surface reconstruction

**Goal:** replace point sprites with a real fluid surface via marching
cubes.

1. **Density field generation (GPU compute):** allocate a 3D scalar grid
   (e.g. 128³). For each particle, splat its kernel contribution into the
   surrounding cells using atomic add (or, more cache-friendly, do the
   reverse: each cell loops over the particles in its neighborhood via the
   same spatial hash and accumulates contributions).
2. **Marching cubes:**
   - **Option A (start here):** CPU marching cubes. Read the density
     field back to CPU each frame, run a JS MC implementation, upload the
     resulting vertex buffer. Slow but simple, fine for the seminar at
     moderate grids (≤64³).
   - **Option B (preferred final):** GPU marching cubes. Two compute
     passes:
     1. Per voxel: compute the case index (0–255) and the triangle count;
        write to an atomic counter.
     2. Per voxel: write actual triangle vertices into a vertex storage
        buffer at the offset reserved in pass 1.
     Then a regular render pass rasterizes the buffer.
3. **Shading:** Phong + Fresnel + a cheap fake refraction (offset UV of
   a skybox/back-pass by the normal). Skybox can be a simple gradient.

**Definition of done:** dam break looks like a moving fluid sheet; you
can toggle between particles view and surface view in the GUI.

### Phase 6 — Polish and instrumentation

**Goal:** make the demo presentable and instrument it for evaluation.

- Camera presets per scenario.
- "Reset", "Pause", "Step", "Capture screenshot" buttons.
- Per-stage timing via WebGPU **timestamp queries** (gate behind a
  feature check; not all browsers expose them). Otherwise, CPU wall-clock
  per dispatch.
- CSV export of frame-time samples.
- Choice of "renderer: particles | marching cubes" in GUI (for the
  evaluation chapter).

**Definition of done:** you can record a clean evaluation run by clicking
"start measurement", running for N seconds, and getting a CSV.

### Phase 7 — Evaluation

**Goal:** produce the numbers and screenshots the report needs.

Measurements (each on the same machine, with stats.js disabled to remove
its overhead):

- **Performance vs. particle count:** 4k, 16k, 64k, 256k. Total ms/frame
  + per-stage breakdown (hash, sort, cell table, density, force,
  integrate, splat, MC, render).
- **Performance vs. grid resolution:** 32³, 64³, 128³ for the MC grid.
- **Visual quality:** screenshots of the same instant in each scenario at
  each particle count.
- **Parameter sensitivity:** brief sweep of viscosity and gas constant;
  qualitative description of failure modes (overdamped, explosive).
- **Stability:** longest stable run at each particle count.

Save raw CSVs + screenshots to `evaluation/`.

### Phase 8 — Writeup

**Goal:** the report.

Suggested chapter outline (adjust to your course's expected format):

1. Introduction and motivation.
2. Background: SPH formulation, kernels, EOS, boundaries.
3. Implementation:
   1. Architecture overview.
   2. Neighbor search (spatial hash, radix sort).
   3. SPH stages on GPU.
   4. Surface reconstruction (density splatting + marching cubes).
   5. Boundary handling.
4. Results: tables and charts from Phase 7.
5. Discussion: bottlenecks, parameter sensitivity, visual quality,
   limitations.
6. Conclusion + brief note on planned thesis extensions.
7. References (1–3 from the seminar brief, plus Müller 2003, Bridson,
   Lorensen & Cline, and TypeGPU/WebGPU specs).

---

## 5. Risks and contingencies

| Risk                                                  | Mitigation                                                                                                |
|-------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| WebGPU instability / browser bug                      | Stick to Chrome stable; verify on a second machine early.                                                  |
| GPU radix sort proves harder than expected            | Fall back to CPU sort over a GPU readback for moderate `n` while the rest of the pipeline matures.        |
| Marching cubes produces holes / non-watertight surface| Smooth the density field with a 3D box blur before MC; verify case-table sign conventions match samples.   |
| Particles explode (Tait too stiff, `dt` too large)    | Adaptive CFL timestep; clamp max velocity; reduce gas constant.                                            |
| Performance disappointing                             | Profile per-stage; usual culprits are sort and unsorted memory access in density/force passes.            |
| Time pressure                                         | Surface reconstruction is the most cuttable item — CPU MC at 32³ is enough to pass the surface requirement.|

---

## 6. Definition of "seminar done"

- Browser demo runs all preset scenarios at the target particle count
  without crashing for at least 1 minute each.
- Evaluation CSVs and screenshots collected, charts drawn.
- Report PDF written and reviewed at least once by a peer.
- Project repository tagged `seminar-final`.

Anything beyond this is thesis territory and stays out of this plan.
