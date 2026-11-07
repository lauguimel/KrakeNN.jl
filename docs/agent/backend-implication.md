---
module: backend
path: src/ (cross-cutting — no src/backend/ dir yet)
owner_concern: backend-dispatch
status: implemented
last_verified: 2026-05-31
depends_on:
  - lbm
---

# backend — module implication map

There is **no `src/backend/` directory**. "Backend" is a *cross-cutting concern*
implemented entirely through [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
(KA) and scattered across `src/kernels/`, `src/drivers/`, `src/simulation_runner.jl`,
`src/curvilinear/`, `src/multiblock/`, and `src/refinement/`. The contract this
map documents is the **device-agnostic launch idiom + the GPU-safety rules every
`@kernel` must obey** so that the same source runs on `CPU()`, `CUDABackend()`,
and `MetalBackend()` unchanged. The main module docstring (`src/KrakeNN.jl`)
advertises "automatic GPU acceleration via KernelAbstractions.jl" — this file is
the operational backing of that claim. There is no Kraken-owned `select_backend`
abstraction: the backend is **passed in by the caller** and **recovered from the
data** inside kernels.

## Public surface

Kraken exposes **no backend type of its own**; it re-uses KA's and the GPU
package's. The de-facto public surface is the set of conventions and the one
launcher idiom callers must use:

- **Backend objects (not owned by Kraken)**: `KernelAbstractions.CPU()` (the
  default everywhere), `CUDA.CUDABackend()`, `Metal.MetalBackend()`. CUDA and
  Metal are hard `[deps]` in `Project.toml` (CUDA "5", Metal "1", KernelAbstractions
  "0.9"); a user picks one and threads it through the `backend=` kwarg.
- **`backend=` kwarg** — every public entry point takes it, defaulting to
  `KernelAbstractions.CPU()`: `run_simulation(...; backend=CPU())`
  (`simulation_runner.jl`), `initialize_2d` / `initialize_3d` / `run_cavity_2d` /
  `run_cavity_3d` / `run_taylor_green_2d` and the other `drivers/basic.jl`
  entry points, plus the dispatch helpers `_run_thermal`, `_run_refined`,
  `_run_axisymmetric`, `_run_twophase_vof`, `_run_d3q19`, `_run_gmsh_slbm_drag`.
- **`KernelAbstractions.get_backend(array) -> backend`** — the universal "what
  device am I on?" call. Used by ~every kernel wrapper (`fused_bgk_step!`,
  `collide_mrt_*!`, `stream_periodic_*!`, the `viscoelastic_2d` / `li_bb` /
  `refinement_exchange_2d/3d` / `drag_gpu` wrappers, …). This is THE way Kraken
  resolves dispatch — from the data, never from a global.
- **Allocators**: `KernelAbstractions.zeros(backend, T, dims...)` (drivers,
  `simulation_runner` Fx/Fy fields) and `KernelAbstractions.allocate(backend, T,
  dims...)` (host→device copies). Helpers `_copy_to_backend(backend, T, host)`,
  `_copy_bool_to_backend(backend, host)`, `_allocate_block_state_as(block, T,
  backend, ng)` in `simulation_runner.jl`; `transfer_slbm_geometry(geom,
  backend)` / `transfer_reflected_wall_geometry(rwg, backend)` in
  `curvilinear/slbm.jl`.
- **Launch idiom** (the contract a new kernel wrapper must follow), e.g.
  `fused_bgk_step!`: `backend = get_backend(f_in)` → `kernel! =
  fused_bgk_step_kernel!(backend)` → `kernel!(args...; ndrange=(Nx,Ny))`. An
  optional `workgroupsize` tuple is plumbed by `persistent_bgk_2d.jl`
  (`fused_bgk_step_kernel!(backend, workgroupsize)`); default `nothing` lets KA
  pick.
- **`KernelAbstractions.synchronize(backend)`** — the host barrier; called once
  after a launch batch (e.g. `collide_stream_2d.jl` `sync && synchronize`,
  `persistent_bgk_2d` single trailing sync, the per-step syncs in
  `simulation_runner.jl`, `ghost_fluid_2d.jl`, `vof_2d.jl`, `smooth_vof_2d.jl`,
  `li_bb_3d_v2.jl`, `cylinder_libb.jl`).

## Reads from

- `lbm` (`src/lattice/`, `src/kernels/`) — the only sibling this concern is
  coupled to. Kernels read lattice topology (D2Q9/D3Q19 weights, opposite tables,
  `cs²`) and the inlined equilibrium/moment helpers (`feq_2d(::Val{q}, …)`,
  `moments_2d(...)` in `kernels/equilibrium_helpers.jl`, and the 3D variants).
  These are `@inline` + `Val`-dispatched precisely so they compile *into* the
  kernel with no runtime dispatch — see Backend constraints.
- **External (not Kraken modules)**: `KernelAbstractions` (the launch/alloc/sync
  API), and `CUDA` / `Metal` (only ever named by their backend constructors;
  there is **no `using CUDA` / `using Metal` in `src/`** — confirmed by grep, so
  the core stays backend-agnostic and the GPU packages are touched only by the
  caller and in `test/`).
- Reads nothing from `units`, `io-krk`, `geometry`, or the `physics-*` modules:
  backend dispatch is downstream of all of them (it just receives the arrays they
  produced).

## Writes to

- **Mutates the device arrays in place** — this is the entire blast radius.
  Kernels write `f_out`, `ρ`, `ux`, `uy` (and thermal `g`, VE conformation,
  refinement ghost/reflux buffers, drag accumulators). The backend concern owns
  *where* those arrays physically live (host RAM vs CUDA/Metal device memory),
  set at allocation time by the `backend` argument.
- **Allocates device memory** via `KA.zeros` / `KA.allocate` — the only place new
  GPU buffers are minted. `_copy_to_backend` does `allocate` + `copyto!(dev,
  T.(host))`, performing the host→device transfer **and** the `Float64→T`
  element-type cast in one shot.
- **Writes no files, mutates no global registry.** Unlike `units`, there is no
  backend registry; dispatch is purely value-based via `get_backend`.
- **Mutates per timestep**: yes — kernels fire inside the time loop. But the
  backend *resolution* (`get_backend`) and *allocation* happen once at setup; the
  hot loop only re-launches already-compiled kernels and calls `synchronize`.

## Backend constraints

This is the load-bearing section: the GPU-safety contract every `@kernel` body
must satisfy so one source runs on CPU/CUDA/Metal.

- **No dynamic allocation inside a kernel.** Kernel bodies allocate nothing on
  the heap — all working values are scalars/`NTuple`s held in registers (see
  `fused_bgk_step_kernel!`: pull-stream into `fp1..fp9` locals, collide, store).
  Buffers are pre-allocated on the host with `KA.zeros`/`KA.allocate`.
- **No host callbacks / no `@warn`/`@error`/`println`/IO inside a kernel.** A
  kernel may not call back into Julia runtime services; logging and error
  reporting live in the host wrapper, never the `@kernel`.
- **No runtime dynamic dispatch inside a kernel.** Equilibrium/moment helpers are
  `@inline` and **`Val`-dispatched** (`feq_2d(::Val{q}, …)`) so the direction is a
  compile-time constant — abstract-typed dispatch would break GPU codegen. The
  `eltype(f)` / `ET(ω)` pattern casts host scalars to the array element type
  *before* the launch (`kernel!(..., ET(ω); ndrange=…)`), keeping the kernel
  monomorphic.
- **No `round/floor/trunc(Int, x)` in a kernel — use `unsafe_trunc(Int, x)`.**
  Standard `Int(...)` / `round(Int,...)` emit a bounds-check/throw path that
  Metal and CUDA codegen reject. Refinement interpolation
  (`refinement_exchange_2d.jl` / `_3d.jl`, `dualgrid_2d.jl`, `dsl/bricks.jl`)
  computes floor indices with `unsafe_trunc(Int, xc)` then `clamp`s into range
  explicitly. The clamp, not the conversion, guards the bounds.
- **`@Const` and `@index(Global, NTuple)`**: read-only inputs are marked
  `@Const(f_in)`; the thread index is `i,j = @index(Global, NTuple)` (3D adds
  `k`). `ndrange` is the array extent `(Nx,Ny[,Nz])`; `workgroupsize` is left to
  KA unless tuned.
- **`synchronize(backend)` is a host barrier, not free.** It must wrap a *batch*
  of launches, not sit between every kernel — `persistent_bgk_2d.jl` exists to
  amortize launch+sync overhead by running `Nt` fused steps with a single
  trailing `synchronize`.
- **Float32 caveat**: on `MetalBackend()` (M-series), `T=Float32` is the practical
  default and is numerically *not* bit-equivalent to CUDA F32 (see Failure
  modes). The element type `T` is chosen by the caller and propagated through
  every allocation and the `ET(...)` casts; nothing in the kernels assumes
  Float64.

## Failure modes

- **Metal F32 R-drift (2026-05-27 postmortem)** — on the M3 Max `MetalBackend()`
  with `T=Float32`, cylinder `Cd` diverges from CUDA F32/F64 as `R` grows
  (≈ +0.2 / −1.9 / −3.2 at R=10/30/50). CUDA F32 ≡ CUDA F64, so this is an **MPS
  quirk, not generic F32 imprecision**. Footgun: trusting a local Metal F32 result
  at R≥30 to within ±2–3 `Cd`. Run validation runs on CUDA F64 (Aqua), use Metal
  F32 only for short canaries.
- **`Int(...)` instead of `unsafe_trunc` ⇒ Metal/CUDA codegen failure.** Any new
  refinement/interpolation kernel that converts a `Float` index with
  `round/floor(Int, …)` will compile on CPU and fail to build on GPU (the throw
  path is unsupported). The fix is the established `unsafe_trunc(Int, x)` +
  explicit `clamp` pattern (3D refinement GPU port, see
  `refinement_exchange_3d.jl`).
- **Forgetting `ET(ω)` / passing a `Float64` scalar into an `F32` kernel** ⇒ a
  type-unstable kernel argument that either errors at codegen or silently
  promotes. Always cast scalars to `eltype(array)` before the launch (the
  `fused_bgk_step!` wrapper is the reference).
- **`synchronize` per-kernel in a hot loop** ⇒ launch-bound throughput collapse.
  Symptom: GPU near-idle, wall time dominated by host barriers. The `persistent_*`
  kernels and the `sync` boolean kwarg on `collide_stream_2d!` / `macroscopic!`
  exist specifically to batch syncs.
- **Benchmark loyalty trap (do NOT compare CPU-scalar vs GPU-custom).** A
  CPU-pure code path benchmarked against a GPU custom kernel produces a fake
  "20× too slow" gap. The AMR-D `route_native` path is CPU-pure while the
  Filippova–Hänel path is GPU — the apparent refinement slowdown that drove the
  Kraken-E pivot was likely this artifact. Always benchmark like-for-like (same
  backend, same `T`).
- **`get_backend` on the wrong array** ⇒ kernel launched on the wrong device. The
  wrapper must call `get_backend` on an array that genuinely lives on the target
  device (e.g. `get_backend(f_in)`, `get_backend(st.f)`), not on a host scratch
  array — otherwise the launch silently runs on `CPU()`.

## Touch order

For a suspected backend/GPU bug (build failure on CUDA/Metal, wrong-device launch,
F32 drift, sync/perf anomaly), inspect in this order:

1. **The failing kernel's wrapper** in `src/kernels/<kernel>.jl` — confirm the
   idiom: `get_backend(<device array>)` → `kernel!(backend[, workgroupsize])` →
   `kernel!(...; ndrange=...)` → scalar args cast with `ET(...)`. 80% of
   "runs on CPU but not GPU" bugs are a missing cast or a bad `get_backend` arg.
2. **The `@kernel` body itself** — scan for the forbidden patterns: heap
   allocation, `Int/round/floor/trunc(Int,…)` (should be `unsafe_trunc`+`clamp`),
   any IO/`@warn`, any abstract-typed/dynamic dispatch (helpers must be `@inline`
   + `Val`).
3. `src/kernels/equilibrium_helpers.jl` (+ `_3d`) — if the bug is numeric on GPU,
   verify the inlined `feq_*`/`moments_*` helpers stay monomorphic in `T`.
4. `src/simulation_runner.jl` — the dispatch hub: `run_simulation` `backend=`
   plumbing, `_run_*` branch, `_copy_to_backend`/`_copy_bool_to_backend` (the
   host→device transfer + `T.()` cast), per-step `synchronize` placement.
5. `src/drivers/basic.jl` — `initialize_2d/3d` allocation site
   (`KA.zeros(backend, T, …)`); a wrong-device array usually originates here.
6. `src/kernels/refinement_exchange_2d.jl` / `_3d.jl`, `dualgrid_2d.jl`,
   `dsl/bricks.jl` — for index-conversion GPU build failures (the `unsafe_trunc`
   call sites).
7. `src/curvilinear/slbm.jl` (`transfer_slbm_geometry` /
   `transfer_reflected_wall_geometry`) and `src/multiblock/state.jl` — for
   SLBM/multiblock device-transfer bugs (geometry arrays left on host).
8. `Project.toml` `[deps]`/`[compat]` — version skew of `CUDA`/`Metal`/
   `KernelAbstractions` is the last suspect for a codegen-only regression.
