---
module: incns-cavity-mg
path: src/methods/inc_ns/cavity_mg.jl; src/methods/inc_ns/cavity_mg_cuda.jl
owner_concern: pressure-velocity-coupling
status: implemented
last_verified: 2026-06-11
depends_on:
  - solve-poisson-mg
---

# incns-cavity-mg — module implication map

The backend-parametric steady SIMPLE solver for the 2D lid-driven cavity
(issues #7/#8): every field allocated via `atype`, every elementwise op a KA
`@kernel` on `backend_ka`, BOTH the momentum predictor (Helmholtz-shifted,
`σ = (1/αu - 1)·4/h²` = the SIMPLE under-relaxation) and the pressure correction
(all-Neumann, σ=0) solved by the matrix-free MG. The per-iteration physics is
FUSED into four kernels; with the fixed-cycles fast path an off-stride iteration
performs ZERO host syncs (a static launch sequence the CUDA-graph companion
captures). **Registered in `src/KrakeNN.jl`** and exported
(`solve_incns_cavity_mg`); also reachable as `solve(params, IncNS(:cavity_mg))`
(`src/methods/inc_ns/method.jl`). The CUDA-graph companion
`cavity_mg_cuda.jl` stays manual-load (CUDSS is not a package dependency).

## Public surface

- `solve_incns_cavity_mg(; nx=128, ny=128, U_lid=1.0, Re=100.0,
  relax=(u=0.7,p=0.3), tol=1e-7, vel_tol=1e-6, maxiter=8000, L=1.0,
  backend_ka=CPU(), atype=Array{Float64}, mg_tol, mg_maxcycles, mom_mg_tol,
  mom_mg_maxcycles, norm_stride=25, mg_cycles=3, mom_mg_cycles=1,
  mg_mixed_precision=false, mom_mg_mixed_precision=false, static_gauge=false,
  offstride_executor=nothing, verbose=false) -> NamedTuple` — GPU run = same
  call with `backend_ka=CUDABackend(), atype=CuArray{Float64}`.
  GPU-efficiency seams: `norm_stride` (outer norms only every n iterations —
  gates ONLY the stop decision), `mg_cycles`/`mom_mg_cycles` (fixed inner
  V-cycles, zero inner reductions), the mixed-precision flags (separate for
  pressure vs momentum because the operators differ), `static_gauge`
  (allocation-free two-kernel gauge reduction, CUDA-graph requirement),
  `offstride_executor` (injectable `f(S)` running one full off-stride
  iteration — the graph seam).
- `_cavity_mg_offstride_step!(S)` + phase functions `_cav_phase1!..4!` — the
  documented iteration decomposition over the state NamedTuple `S` (de-facto
  public: the executor contract).
- CUDA companion (`cavity_mg_cuda.jl`, loaded ONLY under `using CUDA`):
  `solve_incns_cavity_mg_cuda_graph(; warmup=2, ...)` — identical numerics,
  off-stride iterations replayed from a captured CUDA graph (forces
  `static_gauge=true`, requires `mg_cycles>0`); `CavityCudaGraphExecutor`
  (warmup -> capture -> replay lifecycle, degrades to plain execution on
  capture failure; reports `graph_captured`/`graph_launches`/`graph_fallback`).

## Reads from

`solve-poisson-mg` (`src/solve/poisson_mg.jl`, guarded include — transitively
the `solve-linear` tags): `solve_poisson_mg` (3 calls per iteration: u, v
momentum + pressure), `build_mg_hierarchy`, `_mg_eltype_variant`. The CUDA
companion additionally reads CUDA.jl's graph API (`capture`/`instantiate`/
`launch` — asserted at include time). KernelAbstractions + `LinearAlgebra.norm`
otherwise. No `.krk` hook.

## Writes to

No globals, no I/O. All device buffers are allocated ONCE before the loop (in
`S` and the shared `mg_hier`/`mg_hier_f32`) and mutated in place — the
allocation-stability contract the CUDA-graph executor ASSERTS (`S.u` witness).
Returns HOST copies (`Array(u)` etc.) plus scalars; `S.counters` is host-side
bookkeeping (NOT updated by replayed graph iterations — documented).

## Backend constraints

- Same source CPU/CUDA; no host scalar indexing of device arrays anywhere; host
  syncs only at on-stride norm checks (and one final barrier).
- Defaults (`norm_stride=25, mg_cycles=3, mom_mg_cycles=1`) ARE the fast path;
  `norm_stride=1, mg_cycles=0` recovers the legacy every-iteration,
  tolerance-driven behaviour.
- Fusion rule (header): only ops whose per-cell results depend on already
  globally-consistent data are fused — recomputing a neighbour's compact
  gradient inline replaces a stored-array barrier, bit-identically. Respect
  this rule when adding kernels.
- Measured (Aqua A100, `benchmarks/krk/inc_ns/cavity_gpu_bench.{jl,pbs}`,
  `benchmarks/results/cavity_gpu_aqua_a100.md`): GPU↔CPU BIT-EXACT
  (‖Δ‖∞ ~1e-16); 4.26x at 512² pre-fast-path — the solve is LATENCY-bound
  (GPU ~4.5% util), which is what norm_stride / fixed cycles / fusion / the
  CUDA graph attack (cumulative-ablation mode of
  `benchmarks/krk/inc_ns/cavity_gpu_bench.jl`, commit cbbce7f70).

## Failure modes

- **Wall convention**: the MG Dirichlet operator is GHOST-0; on a cell-centred
  grid the wall sits at the FACE, so the OPERATOR is identical and the "+2/h²
  vs +1/h²" distinction lives in the SOURCE only — the lid injects
  `+2·U_lid/h²` on the north row (u-momentum). Folding the lid as `+U_lid/h²`
  "to match the operator" halves the effective lid speed (plausible-looking,
  wrong profiles vs Ghia).
- **Gauge**: fixed-cycles pressure solves skip zero-mean projections; `pcorr`
  carries an arbitrary constant which the solver gauges (device-resident mean
  subtraction). Removing the gauge lets `p` drift unboundedly across thousands
  of iterations.
- **σ cancels at the fixed point** (`+σ·u_old` in the RHS): the converged field
  is σ-independent and satisfies the UN-relaxed steady momentum equation.
  "Simplifying" away the `σ u_old` RHS term turns the predictor into a damped
  (wrong) equation.
- **norm_stride semantics**: iterates are UNCHANGED off-stride; the solver only
  stops at the next check (up to stride-1 iterations later, MORE converged).
  Receipt for the full fast path (`norm_stride=25, mg_cycles=3,
  mom_mg_cycles=1`): fields within 4.3e-5 RELATIVE of the stride-1 reference,
  stopping 11 iterations later — `test/scratch/incns_cavity_mg_fastpath_driver.jl`.
- **Mixed precision**: converged Ghia Re=100 deviation ≤8.2e-10 vs all-F64 —
  `test/scratch/incns_cavity_mg_mixed_precision_driver.jl`. Both flags default
  OFF; defaults stay bit-identical to the previous revision.
- **CUDA-graph traps** (cavity_mg_cuda.jl header, all handled): capture records
  WITHOUT executing (the executor launches once post-capture); first-touch JIT
  invalidates capture (warmup runs + one retry, then permanent fallback —
  correctness unchanged); library `sum!` may allocate → capture rejects async
  allocs, hence `static_gauge` forced ON; buffer reallocation = stale device
  pointers (witness assert); call via `Base.invokelatest` from the same
  top-level expression that includes the files (world age).
- Validation: Ghia Re=100 max centreline deviation 0.689% of `U_lid` (128²,
  gate ≤5%), Re=1000 2.31% (512²) — `test/analytical/incns_cavity_mg_ghia.jl`
  (env `INCNS_MG_GHIA_SKIP_RE1000=1` skips the long case).

## Touch order

1. `src/methods/inc_ns/cavity_mg.jl` — kernels, phase functions, the solver
   loop (the header + section comments encode every design rule above).
2. `test/analytical/incns_cavity_mg_ghia.jl` — the Ghia gates; run with
   `INCNS_MG_GHIA_SKIP_RE1000=1` for a fast check.
3. `src/solve/poisson_mg.jl` — if an inner solve (not the coupling) is wrong:
   σ handling, fixed-cycles, mixed precision live there.
4. `src/methods/inc_ns/cavity_mg_cuda.jl` — graph-only failures (capture,
   fallback, counters).
5. `src/methods/inc_ns/cavity.jl` — the assembled-CHOLMOD sibling these
   numerics were matched against (d-coefficient, σ folding).
6. `test/scratch/incns_cavity_mg_{driver,fastpath_driver,mixed_precision_driver}.jl`
   — targeted drivers; `benchmarks/krk/inc_ns/cavity_gpu_bench.jl` for GPU
   parity/perf.
