# Capabilities matrix

These are the capabilities of the **mature LBM core** within Kraken's broader
method-agnostic platform. For how LBM fits the platform — and what is shipped
versus on the roadmap across methods — see the
[architecture overview](architecture.md).

This page is the single-source-of-truth for **what the LBM core can do today**,
what is partially implemented, and what is out of scope. Rows were last checked
against the code on `dev/platform` in November 2026 (after v0.5.1). Every
entry links to the theory page that derives it, the example that exercises
it, and the API or .krk reference that documents how to call it.

A ✓ means the feature runs and a test in `test/` exercises it. Rows
marked **code present, later release** exist in the codebase (and in some cases
are fully tested) but are not part of the validated surface.

Legend: ✓ = works and tested · ~ = implemented, partial validation ·
— = not applicable · ✗ = not yet

---

## 1. Core LBM

| Capability | 2D | 3D | GPU | .krk | Theory | Example | API |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| D2Q9 / D3Q19 lattice | ✓ | ✓ | ✓ | ✓ | [01](theory/01_lbm_fundamentals.md), [02](theory/02_d2q9_lattice.md), [06](theory/06_from_2d_to_3d.md) | [Poiseuille 2D](examples/01_poiseuille_2d.md), [Cavity (2D & 3D)](examples/04_cavity_2d.md) | [lattice](api/lattice.md) |
| BGK collision | ✓ | ✓ | ✓ | ✓ | [03](theory/03_bgk_collision.md) | [Cavity 2D](examples/04_cavity_2d.md) | [collision](api/collision.md) |
| MRT collision | ✓ | ✗ | ✓ | ~ EHD only (`ns_scheme = mrt`) | [12](theory/12_mrt.md) | — | [collision](api/collision.md) |
| TRT collision | ✓ | ✓ | ✓ | ~ 2D: `Physics collision = trt`, Newtonian, no body force, wall boundaries only | — | — | `fused_trt_libb_v2_step_3d!` (3D cut-link driver, Julia API) |
| Streaming (periodic/wall) | ✓ | ✓ | ✓ | ✓ | [04](theory/04_streaming.md) | [Couette](examples/02_couette_2d.md) | [streaming](api/streaming.md) |
| Macroscopic recovery (ρ, u) | ✓ | ✓ | ✓ | — | [01](theory/01_lbm_fundamentals.md) | — | [macroscopic](api/macroscopic.md) |
| Body force (Guo forcing) | ✓ | ✓ | ✓ | ✓ | [07](theory/07_body_forces.md) | [Poiseuille 2D](examples/01_poiseuille_2d.md) | [collision](api/collision.md) |
| Axisymmetric (z, r) | ✓ | — | ✓ | ✓ | [09](theory/09_axisymmetric.md) | [Hagen–Poiseuille](examples/09_hagen_poiseuille.md) | [drivers](api/drivers.md) |

**Limitations:** MRT is D2Q9 only, and a plain `.krk` file cannot select it (only the
EHD solver reads `ns_scheme = mrt`). TRT in 3D exists for the cut-link obstacle and
sphere drivers and is called from Julia, not selected from `.krk`. D3Q27 is not implemented.

## 2. Boundary conditions

| BC type | 2D | 3D | Spatial f(x,y,z) | Time f(t) | .krk syntax |
|---|:-:|:-:|:-:|:-:|---|
| Wall (bounce-back) | ✓ | ✓ | — | — | `Boundary north wall` |
| Velocity (Zou-He) | ✓ | ✓ | ✓ | ✓ | `Boundary west velocity(ux=4*U*y*(H-y)/H^2)` |
| Pressure/density outlet | ✓ (west, east) | ~ (east and top faces only; one no-NaN smoke test; see issue #20) | ~ | ✓ | `Boundary east pressure(rho=1.0)` |
| Periodic | ✓ | ✓ | — | — | `Boundary x periodic` |
| Fixed temperature (Dirichlet) | ✓ | ✓ | ✗ | ✗ | `Boundary west wall(T=1.0)` |
| Adiabatic (zero-flux) | ✓ | ✓ | — | — | default when no `T=` specified |
| Outflow / Neumann / symmetry | ~ | ~ | — | — | recognized, partial |

- Theory: [05 BCs](theory/05_boundary_conditions.md), [19 spatial BCs](theory/19_spatial_bcs.md)
- .krk: [BC types](krk/bc_types.md), [Expressions](krk/expressions.md)
- Expression whitelist: arithmetic, `sin/cos/tan/exp/log/sqrt/abs/clamp`,
  variables `x, y, z, t, Lx, Ly, Lz, Nx, Ny, Nz, dx, dy, dz` + `Define`d vars.

**Limitations:** thermal BCs are scalar only (no `T = f(x,y,t)`).
Outflow/Neumann/symmetry are recognized by the parser but kernels are
minimal — use with caution.

## 3. Thermal

| Capability | 2D | 3D | Refined | GPU | .krk | Theory / Example |
|---|:-:|:-:|:-:|:-:|:-:|---|
| Passive scalar transport (DDF) | ✓ | ✓ | in development | ✓ | `Module thermal` | [08 thermal DDF](theory/08_thermal_ddf.md) |
| Boussinesq buoyancy | ✓ | ✓ | in development | ✓ | auto when hot/cold BCs set | [Rayleigh-Bénard](examples/08_rayleigh_benard.md) |
| Heat conduction (pure) | ✓ | ✓ | — | ✓ | ✓ | [Heat conduction](examples/07_heat_conduction.md) |
| Natural convection (cavity) | ✓ | ✓ | in development | ✓ | ✓ | driver `run_natural_convection_2d/3d` |
| Natural convection, generalised Newtonian (power law, Carreau-Yasuda…) | ✓ | ✗ | ✗ | ✓ (one A100, Float64) | ✓ | [GNF natural convection](users/benchmarks/gnf-natural-convection.md); driver `run_natural_convection_gnf_2d` |
| ν(T) Arrhenius (kernel) | ✓ | ✗ | ✗ | ✓ | ✗ | code: `collide_boussinesq_vt_2d!` |
| ν(T) modified Arrhenius (Rc) | ✓ | ✗ | ✗ | ✓ | ✗ | driver via `Rc` keyword |
| α(T), κ(T) variable diffusivity | ✗ | ✗ | — | — | ✗ | not implemented |
| Validation vs De Vahl Davis (Ra=1e3) | ✓ 1.4% | — | in development | ✓ | — | Nu = 1.118 ref |
| Validation vs Fusegi (3D Ra=1e3) | — | ~ 5% | in development | ✓ | — | Nu = 1.085 ref |

**Limitations:**
- ν(T) is wired at the **driver** level (`run_natural_convection_2d(; Rc=5)`)
  but **not yet dispatched from .krk**. Adding `Rc = 5` in `Physics {}` does
  not activate the variable-viscosity kernel. Workaround: call the Julia
  API directly.
- No custom `ν(x, y, z, t, γ̇, T)` user-defined function inlined into
  kernels — this requires a parser extension (not implemented).
- Thermal BCs are scalar only; `T = sin(π*y)*t/T_max` is not supported yet.

## 4. Grid refinement (under development)

Grid refinement (patch and conservative AMR) is under development and not
validated; it is not part of the supported feature set. No refinement
capability is listed as working here. The code and its tests remain in the
repository for development purposes. The reasons it is not validated are
given in [Limitations](theory/10_limitations.md).
The `.krk` `Refine` keyword is parsed but is under development and not
validated; see [directives](krk/directives.md).

## 5. Geometry and obstacles

| Capability | Status | .krk syntax | Notes |
|---|:-:|---|---|
| Implicit condition (circle, box, sphere) | ✓ | `Obstacle cyl { (x-cx)^2 + (y-cy)^2 <= R^2 }` | any boolean expression |
| STL import (binary + ASCII) | ✓ | `Obstacle body stl(file="geom.stl", scale=0.5)` | auto-detect format |
| 3D voxelization (Möller–Trumbore ray casting) | ✓ | — | requires watertight mesh |
| 2D voxelization (slice at z_slice) | ✓ | `stl(file=..., z_slice=0.0)` | — |
| Named primitives `circle()`, `box()`, `sphere()` | ✗ | not implemented | ergonomic sugar for common shapes |
| Multiple obstacles | ✓ | repeat `Obstacle` block | — |
| Fluid region (inverse logic) | ✓ | `Fluid inside { x < L/2 }` | — |

- Theory: — (voxelization is implementation detail, not LBM theory)
- API: [io](api/io.md) — `read_stl`, `voxelize_2d`, `voxelize_3d`

**Limitation:** no named primitives yet. Users must spell out
`(x-cx)^2 + (y-cy)^2 <= R^2`. A possible syntax, not implemented:
```
Obstacle cyl circle(center=[0.5, 0.5], radius=0.1)
Obstacle box rectangle(xmin=0.2, xmax=0.8, ymin=0.3, ymax=0.7)
Obstacle sph sphere(center=[0.5, 0.5, 0.5], radius=0.1)
```

## 6. Sanity checks (parameter validation)

Running `sanity_check(setup)` — and the check that `run_simulation` does
automatically — covers the following:

| Check | Level | What it verifies |
|---|---|---|
| Relaxation τ = 3ν + 0.5 | error/warn | τ < 0.5 unstable; τ < 0.51 marginal; τ > 10 over-diffusive |
| Compressibility | error/warn | U_ref (Ma = U/c_s); Ma > 0.1 flags compressibility error |
| Spatial resolution | warn | N < 10, N/Re < 1 (boundary layer under-resolved) |
| Thermal τ_α = 3α + 0.5 | error/warn | same bounds as flow τ |
| Prandtl number | warn | Pr < 0.1 or > 10 extreme for SRT thermal |
| Rheology τ_min | warn | from `nu_min` in GNF models |
| Fine-grid checks (refinement) | warn | under development, not validated |
| Two-phase density/viscosity ratio | warn | code present, not part of the validated surface |

- .krk: [Sanity checks](krk/sanity.md)
- Parameter summary is printed at run-time with 2D/3D tag.

**Limitations / planned:**
- No Ra-vs-N thermal BL resolution check (planned — ~ Ra^{1/4} minimum N).
- No CFL check for prescribed-velocity advection.
- Custom user checks not yet exposed in .krk.

## 7. KRK DSL

| Feature | Status | Reference |
|---|:-:|---|
| `.krk` extension + VSCode highlighting | ✓ | [overview](krk/overview.md) |
| Core directives (Simulation, Domain, Physics, Run, Output) | ✓ | [directives](krk/directives.md) |
| Boundary / Initial / Obstacle / Fluid / Velocity blocks | ✓ | [directives](krk/directives.md) |
| Modules: `thermal`, `axisymmetric` | ✓ | [modules](krk/modules.md) |
| Rheology block (GNF + VE parsing) | ✓ | [directives](krk/directives.md) |
| `Refine` block | in development | [directives](krk/directives.md) |
| Define user variables | ✓ | [expressions](krk/expressions.md) |
| Expressions (math + `x, y, z, t`) | ✓ | [expressions](krk/expressions.md) |
| **Greek letters** (ν/nu, ρ/rho, σ/sigma, τ/tau …) | ✓ | [aliases](krk/aliases.md) |
| ASCII aliases (`nu` → `ν`) | ✓ | [aliases](krk/aliases.md) |
| **CLI wrapper** (`kraken run sim.krk`) | ✓ | `bin/kraken` (also `kraken info sim.krk`) |
| `Backend <cpu\|cuda\|metal>` directive | ✓ | [directives](krk/directives.md) |
| Preset templates (cavity_2d, etc.) | ✓ | [presets](krk/presets.md) |
| Setup helpers (`Setup reynolds = 100`) | ✓ | [helpers](krk/helpers.md) |
| Sweep (parametric studies) | ✓ | [directives](krk/directives.md) |

**Limitations:**
- No `include` directive for reusable .krk fragments.
- No conditional blocks (`if Re > 100 { ... }`).
- Sweeps are Cartesian product; no Latin hypercube.

## 8. GPU support

| Backend | Status | File type | Notes |
|---|:-:|---|---|
| CPU (KernelAbstractions) | ✓ | Float32/Float64 | reference |
| Metal (Apple Silicon) | ~ | Float32 recommended | run on an M3 Max (throughput CSV only); no Metal test in the suite |
| CUDA (NVIDIA) | ~ | Float32/Float64 | run on A100 and H100 on the QUT Aqua cluster (certification CSV for A100 only); CUDA tests in the suite self-skip without a GPU, and CI has no GPU runner |
| ROCm (AMD) | ✗ | — | KA supports it, not tested |
| oneAPI (Intel) | ✗ | — | — |

- All kernels use `KernelAbstractions.@kernel` → single source, multi-backend.
- **Fix a82957c:** `unsafe_trunc` instead of `trunc(Int, ...)` in all
  refinement and dualgrid kernels (required for Metal; allocated on GPU).

**Performance reference:** the one committed CUDA throughput measurement is 3461 MLUPS
(BGK D2Q9, Float64, `N = 2048`, A100-40GB), 0.32 of that GPU's bandwidth ceiling
(`benchmarks/results/certification_a100.csv`; see
[GPU certification](users/benchmarks/gpu-certification.md)). No H100 LBM throughput
number is published. Apple M3 Max throughput is in [Performance](benchmarks/performance.md).

## 9. Rheology

Beyond Oldroyd-B (the validated viscoelastic model), the models below are
**implemented in code**; their validation level differs from row to row. The "GPU"
column means the collision kernel is KernelAbstractions-based, which does not imply a
GPU test. "Thermal coupling" is the shift of the model parameters with temperature
(`thermal_shift_factor`); it is implemented for the generalised-Newtonian models only,
and only one driver calls the thermal rheology kernel (`collide_rheology_thermal_2d!`,
via `run_natural_convection_gnf_2d`, 2D natural convection, reachable from `.krk`).

| Model | 2D | 3D | GPU | Thermal coupling |
|---|:-:|:-:|:-:|:-:|
| Newtonian | ✓ | ✓ | ✓ | kernel only (no driver) |
| Power-law | ✓ | ✓ (isothermal; Julia API) | ✓ | kernel only (no driver) |
| Carreau-Yasuda | ✓ | ✓ (isothermal; Julia API) | ✓ | kernel only (no driver) |
| Carreau-Yasuda steady adjoint (`gnf_steady_gradient`; tests `test/ad/gnf_adjoint_test.jl`, `test/ad/gnf_adjoint_enzyme_test.jl`; one A100, Float64, run archived under `benchmarks/validation/gnf_adjoint/results/`) | ✓ | ✗ | ✓ (one A100) | — |
| Cross | ✓ | ✓ (isothermal; Julia API) | ✓ | kernel only (no driver) |
| Bingham (Papanastasiou) | ✓ | ✗ | ✓ | kernel only (no driver) |
| Herschel–Bulkley | ✓ | ✗ | ✓ | kernel only (no driver) |
| Oldroyd-B (log-conf) | ✓ | ✓ (FVFD) | ✓ | ✗ |
| FENE-P | ✓ | ✓ (FVFD) | ✓ | ✗ |
| Saramito (EVP) | model type and `.krk` parsing only | ✗ | ✗ (no solver kernel) | ✗ |

**3D generalised-Newtonian (Julia API only).** `run_gnf_channel_3d` (D3Q19 TRT, LI-BB
walls, Guo forcing; Power-law, Carreau-Yasuda, Cross) is isothermal, has no `.krk`
keyword yet, and is verified against an exact planar-channel solution on CPU Float64
(`test/analytical/gnf3d_channel_order.jl`, observed order 1.93 to 2.00) and on an A100
(Float64, N = 32 to 256, order 1.95 to 2.001, archived sweep, not a test). With a
constant viscosity it equals the Newtonian step (`test/analytical/gnf3d_parity.jl`); the
strain-rate estimate is in `test/analytical/gnf3d_strain_rate.jl`. Bingham and
Herschel-Bulkley have no 3D path. See
[3D GNF channel](users/benchmarks/gnf3d-channel-convergence.md).

**3D viscoelastic (since v0.5.0).** Oldroyd-B, FENE-P, Giesekus and PTT (linear and
exponential) run in 3D in log-conformation form on the FVFD path only
(`run_viscoelastic_fvfd_poiseuille_3d`, `run_viscoelastic_fvfd_extensional_3d`). Oldroyd-B
is validated against analytic solutions and RheoTool on periodic Poiseuille and planar
extension; FENE-P, Giesekus and PTT against their closed-form simple-shear fixed points
(FENE-P also against its transcendental planar-extension fixed point), and in the coupled
flows through their Oldroyd-B limit and qualitative trends — see
[Viscoelastic constitutive models](users/benchmarks/ve-constitutive-models.md). The 3D
LBM-CDE drivers (sphere, Couette, Poiseuille) accept Oldroyd-B only, in direct conformation
form, and the sphere driver is not validated against a viscoelastic reference. The 3D
viscoelastic tests run on CPU; the GPU and thermal-coupling columns above refer to 2D.

Effective viscosity dispatches at compile-time (zero-cost abstraction
via Julia's JIT). See `src/rheology/` (models, viscosity, strain_rate).

## 10. Output and diagnostics

| Output | Status | .krk syntax |
|---|:-:|---|
| VTK (+ PVD time series) | ✓ | `Output vtk every 1000 [rho, ux, uy]` |
| PNG snapshots (CairoMakie) | ✓ | `Output png every 500 [uy]` |
| GIF animations | ✓ | `Output gif every 100 [ux] fps=15` |
| Diagnostics CSV | ✓ | `Diagnostics every 100 [step, drag, lift]` |
| ParaView-ready layout | ✓ | — (VTK pvd files) |

- API: [io](api/io.md), [postprocess](api/postprocess.md)
- Helper: `open_paraview(result)` opens the latest output in ParaView.

---

## Not part of the validated surface

These features exist in the codebase (tests often pass) but are not documented
as validated:

- **Multiphase (VOF + PLIC)**: `src/kernels/vof_2d.jl`, `src/kernels/pressure_vof_2d.jl`
- **Phase-field (Allen-Cahn)**: `src/kernels/phasefield_2d.jl`
- **Shan–Chen spinodal**: `src/drivers/multiphase.jl`
- **Species transport**: `src/kernels/species_2d.jl`
- **Viscoelastic cylinder driver**: `src/drivers/viscoelastic.jl`
- **3D generalised-Newtonian rheology**: kernels are 2D only (3D
  viscoelastic: see §9)
- **3D MRT**: not implemented (D3Q19 BGK and TRT only)
- **Left-out test modules**: conservative-tree AMR, multiblock, SLBM and Gmsh drag
  have test files that the default suite does not run; set `KRAKEN_TEST_EXPERIMENTAL=1`
  to run them.

---

## Quick decision helper

| I want to… | I need… |
|---|---|
| lid-driven cavity 2D | BGK + wall BCs + velocity BC. [Example](examples/04_cavity_2d.md) |
| flow past a cylinder | BGK + inlet parabolic BC + obstacle (`Obstacle { ... }`). [Example](examples/06_cylinder_2d.md) |
| natural convection in a cavity | `Module thermal` + two `wall(T=...)` BCs. [Example](examples/08_rayleigh_benard.md) |
| refine near a wall | not available: grid refinement is under development and not validated |
| run on H100/A100 | `Backend cuda` in the `.krk` file, or `run_simulation(...; backend=CUDABackend())` |
| vary the viscosity with temperature | call `run_natural_convection_2d(; Rc=10)` directly (not via .krk yet) |
| use an STL geometry | `Obstacle body stl(file="geom.stl")` |
| sweep parameters | `Sweep Re = [100, 200, 400]` |
| run from the terminal | `kraken run sim.krk` (CLI wrapper, `bin/kraken`) |
