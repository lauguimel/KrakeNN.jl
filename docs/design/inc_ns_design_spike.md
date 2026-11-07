# Design spike — incompressible steady-state Navier–Stokes method

**Issues:** #7 (the `IncNS` method) · #8 (the shared GPU linear-solve service).
**Branch target:** `dev/platform`. **Status:** decision record — pre-implementation (Phase 0 closed).

## Purpose

Settle two open choices before any production code (issue #7, *Design space*):
1. the **steady incompressible coupling scheme**;
2. the **pressure-Poisson / linear-solve path** (#8).

Hard constraints from the platform: GPU-native, `KernelAbstractions`-portable
(same source CPU + CUDA, F64 on CUDA), embedded solids via FVFD **cut-cell
fractional faces**, reuse the existing FVFD advective operators, and stay
**adjoint-friendly** (later `SteadyAdjoint` capability).

## Scope refinement (user direction, 2026-06-10)

Target: **incompressible laminar** flow, Newtonian first (no turbulence model). Two
extensions beyond issue #7's original "steady-only, transient stays LBM's job":

- **Steady AND unsteady.** One shared incompressible core (the operators + the
  pressure-Poisson service) driven by either a **steady driver (SIMPLE)** or an
  **unsteady driver (projection / fractional-step / PISO, implicit-in-diffusion)**.
  Both reuse the same Poisson service and operators; only the outer loop differs.
- **Branchable to viscoelastic.** The momentum stress term is a **pluggable
  constitutive closure** (`AbstractClosure`): Newtonian (`μ∇²u`, ≡ `∇·(2μD)` at
  `∇·u=0`) now; viscoelastic later via the **existing FVFD log-conformation
  operators + `∇·τ`** — no separate solver. Unifies steady/unsteady × Newtonian/VE
  in one FVFD discretization (same BCs, cut-cell geometry, constitutive modules as
  the current VE work).

Reinforcement: the pressure-Poisson matrix is **geometry-only ⇒ constant across
time steps** ⇒ cuDSS *factorize-once* amortizes over an entire transient run —
strengthening Decision 2 (assembled-sparse) for unsteady, not just steady.

## Framing decision — adopt a general performant linear-solve foundation

We deliberately build #8 on a **general assembled-sparse linear solver** from the
start, using the best available performant backend per platform. The
`[weakdeps]` extension mechanism (precedent: `ext/KrakeNNADExt.jl` ↔ Enzyme)
makes high-performance dependencies **free for base users**: they load only on
CUDA, `using KrakeNN` stays light. Consequences:
- there is **no solver-imposed restriction** forcing a segregated scheme — a
  general sparse solver can handle a saddle-point or Newton-Jacobian system too
  (assemble → hand to the solver). The coupling scheme is therefore chosen on
  **algorithmic merit**, not solver limitation;
- the only real constraint these libraries impose is that cuDSS/AMGX are
  **CUDA-only** → a CPU sibling is needed for development/CI. `LinearSolve.jl`
  supplies exactly this: one abstraction dispatching cuDSS/AMGX on CUDA and
  CHOLMOD/UMFPACK (SuiteSparse) on CPU.

## Decision 1 — coupling scheme: **SIMPLE for v1, coupled/JFNK for v2**

| Scheme | Linear solve needed | Robustness | Conv. rate | Adjoint | Phase |
|---|---|---|---|---|---|
| **SIMPLE/SIMPLER** | scalar pressure-Poisson (cuDSS) | good (under-relax) | linear, many outer its | clean (steady residual) | **v1 ✓** |
| Coupled monolithic | saddle-point (assemble → cuDSS) | best | quadratic | excellent | v2 |
| JFNK + Schur/PCD | Poisson + Schur precond | needs globalization | quadratic | excellent | v2 |
| Pseudo-transient (PTC) | scalar Poisson | best far from soln | between | clean | fallback |

**Rationale for SIMPLE v1 (chosen on merit, not constraint):**
- robustness + fewest moving parts → lowest-risk route to the first green rung;
- **direct validation parity** with the in-repo `simpleFoam`/`icoFoam` cavity
  reference kit (`benchmarks/results/rheotool_compare/newtonian/`) — same
  algorithm family;
- its pressure-Poisson is now solved by **cuDSS** (fast, robust), not a
  hand-rolled solver;
- converges to the steady NS residual `R(u,p)=0`, a clean target for the later
  discrete `SteadyAdjoint`.

**Momentum predictor:** the `u*` advection–diffusion system has coefficients that
change every outer iteration (advection depends on current `u`) → **not**
factorize-once; solve it with a few matrix-free / iterative sweeps
(Jacobi/GS/Krylov, diagonally dominant under under-relaxation). Only the
**pressure-Poisson is constant-coefficient and fixed across outer iterations** →
this is where cuDSS's *factorize-once, back-substitute-each-iteration* wins, and
it is the dominant cost. The factorize-once amortization is the key reason
sparse-direct fits **steady** so well.

**v2 escalation:** coupled/JFNK reuses the **same** assembled-sparse + cuDSS
infrastructure (assemble the larger system, hand to the solver) → cheap to reach
once #8's foundation is general. PTC kept as a robustness fallback for stalled
high-Re cases (same Poisson kernel).

**Unsteady driver (per Scope refinement):** the transient regime uses a
**projection / fractional-step (or PISO)** time-integration — implicit in
diffusion, with one pressure-Poisson solve per step — over the **same** operators
+ Poisson service. SIMPLE is the steady driver; both sit on the shared core, with
the viscous/stress term supplied by the branchable constitutive closure.

## Decision 2 — linear-solve path (#8): **assembled-sparse, cuDSS + CHOLMOD via LinearSolve.jl**

| Path | Backend | Steady fit | New dep | Role |
|---|---|---|---|---|
| **cuDSS sparse-direct** | CUDA F64 | **excellent** (factorize once) | `CUDSS.jl` `[weakdep]` | **primary, CUDA** |
| **CHOLMOD/UMFPACK** | CPU | excellent | SuiteSparse (stdlib) | **primary, CPU/CI** |
| AMGX algebraic MG | CUDA | good, O(N) | `AMGX.jl` `[weakdep]` | large-3D / out-of-core-of-direct |
| Matrix-free geometric MG | KA CPU+CUDA | good | none | portability fallback (non-CUDA GPUs) |
| FFT | — | — | none | **rejected**: cut cells break separability |

**Plan:** `LinearSolve.jl` is the abstraction (core or thin weakdep — confirm in
Phase A); **cuDSS** (CUDA, F64) and **CHOLMOD/UMFPACK** (CPU) are the primary
backends, cuDSS behind `[weakdeps]` mirroring `ext/KrakeNNADExt.jl`. **AMGX** is a
later option for very large 3D where direct factorization runs out of memory.
A **matrix-free geometric multigrid** is demoted to a later fully-portable
fallback (non-CUDA GPUs / no-cuDSS environments), **not** a v1 deliverable.
**FFT rejected** for the general embedded-boundary case.

Why this over a hand-rolled matrix-free multigrid (the earlier draft):
- the `[weakdeps]` pattern makes the performant libs free for base users;
- **algebraic** multigrid / sparse-direct absorb the cut-cell variable stencils
  from the matrix coefficients — no hand-coded geometric coarse operator near
  cuts (this was the earlier draft's main flagged risk; it dissolves);
- steady ⇒ factorize-once amortization makes sparse-direct ideal;
- less bespoke numerics to build, debug, and keep competitive vs NVIDIA libs.

## Decision 2 — RECONSIDERED for GPU performance (user, 2026-06-10)

Evidence from the Aqua A100 bench (`benchmarks/results/poisson_gpu_aqua_a100.md`):
cuDSS gave ~30× solve vs CPU but the job averaged only **~9% GPU utilization** —
modest next to LBM's ~1000× CPU→GPU. Root cause: sparse-direct factorization
(elimination tree) and triangular solves are **sequentially dependent** → poor
GPU occupancy. LBM is embarrassingly-parallel local stencils → saturates the GPU.

**Revised priority:** the primary Poisson path for GPU performance and 3D
scalability is **matrix-free geometric multigrid / MG-preconditioned CG**
(Jacobi/Chebyshev smoothers + restriction/prolongation = KA stencils, like LBM):
O(N) and GPU-saturating, vs cuDSS O(N^1.5)-2D / O(N²)-3D and GPU-starved. This
**promotes the matrix-free MG that the first spike demoted to a fallback** — the
"performant deps" framing under-weighted GPU occupancy. cuDSS stays as a robust
2D correctness baseline + steady factorize-once option (and a parity reference).

**Honest cap:** the pressure-Poisson is elliptic (global coupling — information
crosses the whole domain each solve), fundamentally less GPU-trivial than LBM's
local explicit updates. Matrix-free MG ≫ cuDSS on GPU but will not reach LBM's
1000×. The FVFD steady solver's real win over LBM is **iteration count**
(~1e3 vs ~1e6–1e7 to reach steady), not per-solve GPU throughput — that is why
elliptic is the right tool for the steady regime regardless of kernel speed.

## Adjoint forward-compatibility (corrected)

The adjoint of a linear solve `A x = b` is the **transpose solve**
`Aᵀ λ = ∂J/∂x` — one does **not** differentiate through the solver. cuDSS solves
transpose systems directly. What must be differentiable is the **assembly** (how
`A`, `b` depend on geometry/parameters), reachable by Enzyme over the assembly
kernels (consistent with `KrakeNNADExt`). **Matrix-free is therefore NOT required
for `SteadyAdjoint`.** Define the canonical converged object as
`R(u,p) = [momentum; continuity] = 0`; `SteadyAdjoint` later solves
`Rᵤᵀ λ = ∂J/∂u` via the same assembled + cuDSS service on the transpose.

## Numerical risks (resolve in Phase 1/2, flagged here)

1. **Cut-cell Poisson assembly.** Build the Laplacian as **sparse coefficients**
   with face-fraction weighting and a consistent cut-cell volume. Keep it
   **symmetric** so CPU can use CHOLMOD (SPD Cholesky) and CUDA cuDSS its SPD
   path; if symmetry cannot be preserved cleanly near cuts, fall back to a
   general (LU) factorization or AMG, which tolerate asymmetry. Assembled
   coefficients are also **easier to unit-test** than a matrix-free transpose.
2. **Singular all-Neumann Poisson.** Wall/embedded BCs are Neumann → the matrix
   is singular; **pin one reference pressure DOF** (or mass-gauge augmentation,
   precedent in the in-repo AD GMRES `src/ad/ad_adjoint.jl`) to make it
   non-singular for the direct factorization.
3. **Inlet/outlet BCs.** velocity-Dirichlet inlet, pressure-Dirichlet or
   convective outlet; Poiseuille uses periodic / pressure-gradient drive.

## Operators to build (Phase 1, before the solver)

`src/fvfd/` has advection / ∇u / ∇·τ / cell→face / cut-cell fractions but **no**
elliptic half. Phase 1 adds, as **assembled sparse coefficients** (cut-cell
aware) with operator canaries (#3):
- cell-centred **Laplacian / diffusion** (symmetric, face-fraction weighted);
- **velocity divergence** `∇·u` (continuity residual + Poisson RHS);
- **pressure gradient** `∇p` (momentum correction).

## Decision summary

- **Coupling (v1):** SIMPLE; matrix-free/iterative momentum predictor; pressure-Poisson via cuDSS. Coupled/JFNK = v2 on the same infra; PTC = robustness fallback.
- **Linear solve (#8):** assembled-sparse via `LinearSolve.jl`; **cuDSS** (CUDA F64) + **CHOLMOD/UMFPACK** (CPU) primary, cuDSS behind `[weakdeps]`; AMGX for large 3D; matrix-free MG = later portability fallback; FFT rejected.
- **Adjoint:** transpose solve via the same service; differentiate the **assembly**, not the solver — matrix-free not required.
- **Numerical crux:** symmetric cut-cell Poisson assembly + singular-Neumann reference pin.

## Next phases (post-spike)

1. **#8 Phase A** — assembled cut-cell Poisson (sparse coeffs) + `LinearSolve.jl` wiring + CHOLMOD (CPU) + `CUDSS` extension (CUDA F64) + manufactured-solution test + CPU↔CUDA parity + reference-DOF pin.
2. **#7 Phase 1** — elliptic FVFD operators (Laplacian, ∇·u, ∇p) as assembled coefficients + canaries.
3. **#7 Phase 3** — `IncNS <: AbstractMethod` SIMPLE loop (iterative momentum predictor + cuDSS pressure-correction), mirroring the LBM contract skeleton.
4. **Validation** — Poiseuille ≤1 %, cavity vs Ghia 1982 / icoFoam ≤1 %/≤5 %, `.krk` repro.
5. **v2** — coupled/JFNK on the same assembled + cuDSS infra; AMGX for large 3D; matrix-free MG portability fallback.
