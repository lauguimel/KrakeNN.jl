---
module: lbm
path: src/lattice/
owner_concern: lbm-operator
status: implemented
last_verified: 2026-05-31
depends_on:
  - backend
---

# lbm — module implication map

The `lbm` trunk owns the **lattice topology** (`src/lattice/`: D2Q9 / D3Q19
velocity sets, weights, opposite tables, `equilibrium`) plus the **core
stream / collide / forcing operators** that every isothermal driver runs each
timestep (`src/kernels/`: the hand-written fused BGK/TRT kernels, the shared
`feq_2d` / `moments_2d` inline helpers, the macroscopic getters, and the
kernel-DSL brick library `src/kernels/dsl/` that re-emits these operators
verbatim as composable bricks). It is the backend-agnostic
`KernelAbstractions` path; physics modules (thermal, viscoelastic,
multiphase) extend it but do not own it. This map covers the lattice
constants and the Newtonian BGK/TRT/SLBM/LI-BB operator core only.

## Public surface

Exported into `Kraken` (see the `export` blocks in `src/KrakeNN.jl`):

- Lattice types: `AbstractLattice{D,Q}`, `D2Q9 <: AbstractLattice{2,9}`,
  `D3Q19 <: AbstractLattice{3,19}`.
- Lattice accessors: `lattice_dim`, `lattice_q`, `weights`, `velocities_x`,
  `velocities_y`, `velocities_z`, `opposite`, `cs2` (returns `1/3`), and
  `equilibrium(lattice, ρ, u…, q)` — the SVector-backed reference equilibrium.
- Non-fused operators: `stream_2d!`, `collide_2d!`, `stream_3d!`, `collide_3d!`
  (pull-stream with domain-edge halfway-BB + standalone BGK collide).
- Fused single-launch operators: `fused_bgk_step!` (stream+BB+BGK collide+moments
  in one kernel), `fused_trt_step!` / `trt_rates(ν; Λ=3/16)` (two-relaxation-time),
  `fused_trt_libb_step!`, `fused_trt_libb_v2_step!`, `fused_trt_libb_v2_step_3d!`
  (cut-link Bouzidi LI-BB), `precompute_q_wall_cylinder`,
  `aa_even_step!` / `aa_odd_step!` (A-A pattern, single buffer),
  `persistent_fused_bgk!` / `persistent_aa_bgk!` (persistent-kernel variants).
- Macroscopic getters: `compute_macroscopic_2d!`, `compute_macroscopic_3d!`,
  `compute_macroscopic_forced_2d!`, `compute_macroscopic_forced_3d!`,
  `compute_macroscopic_pressure_2d!`.
- Kernel DSL: `LBMBrick` (abstract), `LBMSpec(bricks…; stencil=:D2Q9)`,
  `build_lbm_kernel(backend, spec)` (compiles+caches a spec into a fused
  `@kernel`), `spec_args(spec)`, and the brick multimethods `required_args`,
  `emit_code`, `phase`. Exported bricks: `PullSLBM`, `CollideTRTLocalDirect`,
  `PullSLBM_3D`, `CollideTRTLocalDirect_3D` (the rest of the brick zoo —
  `PullHalfwayBB`, `PullSLBMBiquad`, `PullSLBMQuartic`, `RescaleNonEq`,
  `SolidInert`, `SolidSwapBB`, `Moments`, `MomentsGuo`, `RecomputeMoments`,
  `CollideBGKDirect`, `CollideTRTDirect`, `CollideTRT`,
  `CollideTRTLocalGuoDirect`, `CollideRegularizedTRTLocal`, `ApplyLiBB`,
  `ApplyLiBBPrePhase`, `ApplyHalfwayBBPrePhase`, `WriteF`, `WriteFLiBB`,
  `WriteMoments`, plus the `_3D` siblings) is `KrakeNN.`-accessible but unexported.
- Integrated-frame moment/equilibrium helpers (refinement-side, exported):
  `d2q9_cx`, `d2q9_cy`, `d2q9_opposite`, `d3q19_cx`, `d3q19_cy`, `d3q19_cz`,
  `d3q19_opposite`, `mass_F`, `momentum_F`, `moments_F`,
  `collide_BGK_integrated_D2Q9!`, `collide_Guo_integrated_D2Q9!`,
  `fill_equilibrium_integrated_D2Q9!`.

## Reads from

- `backend` (`KernelAbstractions`) — every operator dispatches through
  `KernelAbstractions.get_backend(f)` and launches a `@kernel`; `build_lbm_kernel`
  takes `backend` as its first argument and keys its cache on `typeof(backend)`.
  This is the only structural runtime dependency of the trunk path.
- Otherwise the lattice/operator core is a **leaf**: the velocity sets, weights,
  opposite tables and equilibrium are self-contained `StaticArrays` constants in
  `d2q9.jl` / `d3q19.jl`; the DSL bricks copy their code fragments verbatim from
  the hand-written kernels and depend on no sibling module's types. Drivers,
  geometry, units and BC modules read FROM this trunk (e.g. `units` consumes
  `cs2`/the `√3` Mach factor), not the other way around.

## Writes to

- **Mutates the output distribution array `f_out` (or in-place `f`) in place**,
  plus the moment fields `ρ_out`/`ux_out`/`uy_out`(`/uz_out`). This is the entire
  blast radius: each kernel writes all `Q` populations and the 3–4 moments per
  cell. Double-buffering (`f_out`/`f_in` swap) is owned by the driver, not here —
  except the A-A pattern (`aa_even_step!`/`aa_odd_step!`) which mutates a single
  buffer in place.
- Solid cells: `fused_bgk_step!`/`fused_trt_step!` write a bounce-back swap of the
  pulled pops and force `ρ=1, u=0`; the DSL `SolidInert` brick instead writes
  rest-equilibrium (`f_q = w_q`, `ρ=1, u=0`).
- **Compile-time mutation only**: `build_lbm_kernel` mutates the module-global
  `LBM_KERNEL_CACHE` dict and `Core.eval`s a `gensym`'d kernel into the `Kraken`
  module namespace (one entry per `(stencil, brick-tuple, backend)` key). No
  per-step allocation, no files, no other global registry.

## Backend constraints

- **Backend-agnostic / KernelAbstractions-clean.** All operators are `@kernel`
  functions launched on `get_backend(f)`; the same source runs on CPU, CUDA and
  Metal. No host-side allocation inside the hot loop.
- **Branchless edge handling**: the pull-stream uses `ifelse` with `max/min`
  *clamped* indices so the dead read stays in-bounds — `ifelse` evaluates both
  branches on GPU, so an unclamped OOB index would segfault. Do not "simplify"
  the clamps away.
- **Float32 caveat**: TRT relaxation rates are passed as `ET(s_plus)`/`ET(s_minus)`
  in the lattice eltype; at low τ the collision rate loses Float32 precision (see
  Metal F32 R-drift). `feq_2d` casts every literal through `T(...)` to stay in the
  working precision.
- **DSL register discipline**: `build_lbm_kernel` emits only the *union of*
  `required_args` over the spec's bricks (canonically sorted via
  `CANONICAL_ARG_ORDER`), so unused arrays never enter the signature → no GPU
  register pressure from dummy params.
- **World-age / `invokelatest`**: the DSL `Core.eval`s a fresh kernel; the
  returned constructor and its launches are wrapped in `Base.invokelatest`
  (Julia 1.12 strict world-age for `eval`'d module bindings — a direct
  cross-world `getfield` warns and can segfault).
- The `equilibrium`/`weights`/`opposite` accessors return `SVector`/`SMatrix`
  constants — cheap, allocation-free, but the `SVector`-indexed `equilibrium` is
  the *reference* path; hot kernels use the unrolled `feq_2d(Val(q), …)` form.

## Failure modes

- **TRT rate label swap (silent 13× over-drag)** — before 2026-04-15
  `trt_rates` returned `s_plus`/`s_minus` swapped, applying the Λ-derived rate to
  the even mode → effective ν ≈ `Λ/(9ν)`, inflating real Re by ~13× and
  over-dragging every benchmark relying on `trt_rates`. Now `s_plus = 1/(3ν+0.5)`
  (even mode, sets ν), `s_minus = 1/(Λ/(3ν)+0.5)` (odd mode). See the historical
  note in `fused_trt_2d.jl`.
- **`Λ=Inf` does NOT give BGK** — passing a large Λ drives `s_plus→0` so
  `a = s_minus/2`, which is not BGK. To recover BGK exactly call
  `fused_bgk_step!` or pass `Λ = (1/s − 0.5)²`. Documented in `fused_trt_step!`.
- **D3Q19 opposite table is 1-based, self-opposite rest = index 1** — `_D3Q19_OPP`
  maps `0→0` as entry `1` and pairs are `(2,3)(4,5)(6,7)(8,11)(9,10)(12,15)
  (13,14)(16,19)(17,18)`; a naive `q→q+9` pairing will scramble bounce-back. The
  pairing is also hard-coded into the `_3D` LI-BB bricks — keep them in sync with
  `d3q19.jl`.
- **DSL brick vocabulary is implicit dataflow** — bricks communicate through
  shared local names (`fp1..fp9`, `ρ/ux/uy/usq`, `feq*`, `fp*c`, `fp*_new`); there
  is no dependency graph. Ordering a `Collide*` before `Moments`, or `WriteFLiBB`
  without a preceding `ApplyLiBB`, produces an `UndefVar`/wrong-result kernel that
  compiles silently. The brick code is COPIED VERBATIM from the hand kernels so
  tests assert bit-exact `.==`; do not refactor a brick for clarity.
- **Double-Bouzidi two-pass trap (M34 v1)** — reusing a BC-containing spec
  (one with `ApplyLiBBPrePhase`) as the pass-1 spec fires the cut-link BC twice
  per step → over-bounce, NaN on the Aqua cut-link matrix. Pass-1 must be a fresh
  RAW spec with no BC brick.
- **Closed-box smoke hides cut-link bugs** — with `q_wall=0.5` the LI-BB
  `_libb_branch` collapses algebraically to halfway-BB, so a closed box smoke
  passes while a real cut-link is broken. Smoke MUST exercise a curved boundary
  (cylinder R=4–8) to flag Bouzidi-FL regressions.
- **`SolidSwapBB` is legacy / a known LI-BB bug source** — flagged in `bricks.jl`;
  cut-link specs must use `SolidInert` (rest-equilibrium) so that fluid neighbours
  pulling from a solid read a well-scaled `w_q`, not stale swapped junk.

## Touch order

For a suspected trunk-operator bug (wrong drag/ν, NaN, conservation drift),
inspect in this order:

1. `src/kernels/fused_trt_2d.jl` — first stop for any ν/Re/drag mismatch:
   `trt_rates` (the historical swap), the `a`/`b` even/odd split, the Λ default.
   `fused_bgk_2d.jl` for the BGK analogue.
2. `src/kernels/equilibrium_helpers.jl` — `feq_2d(Val(q),…)` and `moments_2d`;
   a wrong sign here corrupts every kernel since they all inline it. Cross-check
   `equilibrium_helpers_3d.jl` for the 3D `feq_3d`/`moments_3d`.
3. `src/lattice/d2q9.jl` / `src/lattice/d3q19.jl` — velocity-set ordering, weights,
   and the `_OPP` opposite tables (the 1-based, self-opposite-rest quirk).
4. `src/kernels/dsl/bricks.jl` (+ `bricks_3d.jl`) — for any DSL-built kernel:
   confirm the brick *order*, the `required_args` union, and the `_libb_branch`
   call arguments. `_libb_branch` itself lives in `src/kernels/li_bb_2d.jl`.
5. `src/kernels/dsl/lbm_builder.jl` + `lbm_spec.jl` — if the kernel signature,
   phase partitioning (`:pre_solid`/`:solid`/`:fluid`), cache key, or world-age
   `invokelatest` wrapping is suspect.
6. `src/kernels/macroscopic.jl` — when reported moments diverge from the
   in-kernel collide moments (Integrated vs Guo half-force convention mismatch).
7. `src/kernels/collide_stream_2d.jl` — the non-fused `stream_2d!`/`collide_2d!`
   reference path; useful as a bisection oracle against the fused kernels.
