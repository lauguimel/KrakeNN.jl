---
module: physics-viscoelastic
path: src/rheology/
owner_concern: constitutive
status: implemented
last_verified: 2026-05-31
depends_on:
  - lbm
  - physics-newtonian
  - geometry
  - bc
  - backend
---

# physics-viscoelastic — module implication map

The `physics-viscoelastic` module is the **LEGACY population-coupled** non-Newtonian
constitutive layer present on the `slbm-paper` branch: the GNF/viscoelastic model
hierarchy (`src/rheology/`), the population-side collision kernels that read a
shear-rate-dependent viscosity (`src/kernels/collide_rheology_2d.jl`), the
conformation-tensor evolution kernels (`src/kernels/viscoelastic_2d.jl`), and the
single end-to-end driver `run_viscoelastic_cylinder_2d` (`src/drivers/viscoelastic.jl`).
It evolves Θ = log(C) (or τ_p directly) on the macroscopic velocity field and
injects the polymer-stress divergence back into a Guo-forced BGK step.

> **CRITICAL branch fact.** The M8-validated log-conformation **finite-volume**
> solver `run_viscoelastic_logfv_cylinder_coupled_2d` does **NOT** live on this
> branch. It is on the separate `feat/ve-logfv-on-v03` branch, merge **PENDING**.
> Everything documented here is the LEGACY population-VE path. Do not reach for the
> log-FV driver from `slbm-paper`; it is not callable here (grep confirms absent in
> `src/` on `slbm-paper`).

## Public surface

All names are `export`ed from `Kraken.jl` (so usable bare after `using KrakeNN`).
Grouped by source file:

- **Model hierarchy** (`src/rheology/models.jl`): abstract roots `AbstractRheology`,
  `GeneralizedNewtonian`, `Viscoelastic`; thermal-coupling tags
  `AbstractThermalCoupling`, `IsothermalCoupling`, `ArrheniusCoupling{T}`,
  `WLFCoupling{T}`; GNF model structs `Newtonian{T,TC}`, `PowerLaw{T,TC}`,
  `CarreauYasuda{T,TC}`, `Cross{T,TC}`, `Bingham{T,TC}`, `HerschelBulkley{T,TC}`
  (all Papanastasiou-regularised where a yield stress applies); VE formulation tags
  `StressFormulation`, `LogConfFormulation` (subtypes of `VEFormulation`); VE model
  structs `OldroydB{T,F,TC}`, `FENEP{T,F,TC}`, `Saramito{T,F,TC}`. Each model has a
  keyword convenience constructor that `promote_type`s its scalar args.
- **Viscosity dispatch** (`src/rheology/viscosity.jl`):
  `effective_viscosity(model, gamma_dot) -> ν` and
  `effective_viscosity_thermal(model, gamma_dot, T_local) -> ν` — compile-time
  dispatched per concrete GNF type; `thermal_shift_factor(tc, T_local) -> a_T`
  (defined in `models.jl`).
- **Strain-rate diagnostic** (`src/rheology/strain_rate.jl`):
  `strain_rate_magnitude_2d(f1..f9, feq1..feq9, rho, tau) -> γ̇` and
  `strain_rate_magnitude_3d(f1..f19, feq1..feq19, rho, tau) -> γ̇` — purely local,
  from the non-equilibrium stress tensor.
- **2×2 symmetric linear algebra** (`src/rheology/linalg.jl`):
  `eigen_sym2x2(a11,a12,a22)`, `mat_exp_sym2x2`, `mat_log_sym2x2`,
  `decompose_velocity_gradient(...)` — branchless host/GPU helpers for the
  log-conformation kernel.
- **GNF collision kernels** (`src/kernels/collide_rheology_2d.jl`):
  `collide_rheology_2d!(f, is_solid, rheology, tau_field)`,
  `collide_rheology_guo_2d!(f, is_solid, rheology, tau_field, Fx, Fy)` (Guo forcing),
  `collide_rheology_thermal_2d!(f, is_solid, rheology, tau_field, Fx, Fy, Temp)`
  (thermo-rheological). All dispatch on `rheology::GeneralizedNewtonian`.
- **VE evolution kernels** (`src/kernels/viscoelastic_2d.jl`):
  `compute_polymeric_force_2d!(Fx_p, Fy_p, tau_xx, tau_xy, tau_yy)` (∂τ/∂x_j → body
  force), `evolve_stress_2d!(...)` (UCM stress form),
  `evolve_logconf_2d!(...; lambda, L_max)` (Fattal–Kupferman log-conf),
  `compute_stress_from_conf_2d!(...; G, L_max)`,
  `compute_stress_from_logconf_2d!(...; G, L_max)` (τ_p = G·f(trC)·(C−I)).
- **Driver** (`src/drivers/viscoelastic.jl`):
  `run_viscoelastic_cylinder_2d(; Nx, Ny, radius, cx, cy, u_in, ν_s/nu_s, ν_p/nu_p,
  lambda, L_max, formulation=:stress, max_steps, avg_window, backend, FT)` — confined
  Oldroyd-B (`L_max=0`) / FENE-P (`L_max>0`) cylinder; returns a NamedTuple with
  `Cd, Cd_s, Cd_p, Fx_drag, Fy_drag`, the `tau_p_*`/`Theta_*` fields and `Re, Wi, beta`.
  Also `compute_polymeric_drag_2d(tau_p_xx, tau_p_xy, tau_p_yy, is_solid, Nx, Ny;
  extrapolate=true)` — wall stress-integral polymer drag (host loop, not exported).

## Reads from

- `physics-newtonian` — the driver bootstraps state via `initialize_cylinder_2d`
  (`src/drivers/basic.jl`), advances populations with `stream_2d!`,
  `collide_guo_field_2d!` (`src/kernels/collide_guo_2d.jl`) and reads moments with
  `compute_macroscopic_2d!`. Solvent drag uses `compute_drag_mea_2d`
  (`src/drivers/basic.jl`). The VE layer rides on the Newtonian population machinery.
- `lbm` — D2Q9/D3Q19 lattice convention (the hard-coded `cx`/`cy` link tables in
  `compute_polymeric_drag_2d`, the per-q `cs²=1/3` weights in
  `strain_rate_magnitude_*`, and `feq_2d`/`moments_2d`/`bounce_back_2d!` consumed by
  the collision kernels). Read-only.
- `bc` — `apply_zou_he_west_2d!` (velocity inlet) and
  `apply_zou_he_pressure_east_2d!` (pressure outlet) in the driver time loop; walls
  via streaming bounce-back.
- `geometry` — the `is_solid` mask (from `initialize_cylinder_2d`) is read to decide
  bounce-back cells and to find fluid-solid links for the polymer drag integral.
- `backend` — `KernelAbstractions.get_backend`, `KernelAbstractions.zeros`, and the
  `backend`/`FT` kwargs select CPU/CUDA/Metal allocation and kernel launch.

## Writes to

- **Returns** an immutable NamedTuple from `run_viscoelastic_cylinder_2d` (host
  `Array` copies of `ux/uy/ρ`, the `tau_p_*` and `Theta_*` tensor fields, plus
  `Cd/Cd_s/Cd_p/Fx_drag/Fy_drag/Re/Wi/beta`). `compute_polymeric_drag_2d` returns
  `(Fx, Fy)`.
- **Mutates kernel output arrays in place**: the collision kernels overwrite the 9
  (or 19) population slots of `f` per cell AND write the realised relaxation time into
  `tau_field[i,j]` (the implicit γ̇↔τ coupling carries `tau_prev` from the previous
  step). The VE kernels write `Theta_*_new`, `tau_p_*`, and `Fx_p/Fy_p` arrays.
- **Driver-internal state churn**: it ping-pongs `f_in`/`f_out` (rebinds locals, not a
  `BlockState` field), and `copyto!`s `Θ_*_new → Θ_*` and (stress form) `Θ_* → tau_p_*`
  every step.
- **Side effects**: `@info` logs at setup and at the end (`Cd Cd_s Cd_p Fx_s Fx_p`).
  No files written, no global registry mutated. Models are immutable value structs.

## Backend constraints

- **Kernels are KernelAbstractions-clean and allocation-free in the hot path.** The
  GNF collision and VE evolution kernels read/write fixed-stride arrays with no
  dynamic allocation; `effective_viscosity`, the strain-rate functions and the 2×2
  linalg are all `@inline` and branchless (`ifelse`, `atan`-based eigen) so they
  inline into the GPU kernel and specialise per concrete `rheology` type at JIT time
  (zero-overhead dispatch). Float32 is supported via the `FT`/`eltype` threading.
- **The DRIVER is NOT fully GPU-resident.** `compute_polymeric_drag_2d` calls
  `Array(...)` on four device arrays and runs a scalar host loop EVERY averaging step
  — a per-step device→host copy + CPU triple-nested loop. On GPU backends this is a
  hard synchronisation point and the dominant cost during the averaging window.
- **Per-step VE cost.** Each step does stream + BCs + a Guo collision + a moment pass +
  a conformation evolve + a stress/force pass (≈3 extra full-grid kernels beyond a
  Newtonian step), so the population-VE path is several× the per-cell cost of a plain
  Newtonian step.
- **Stability is τ/Float32-sensitive.** The log-conf kernel clamps the Peterlin
  denominator (`max(L²−trC, 0.01)`) and `mat_log_sym2x2` clamps eigenvalues to
  `1e-30`; on Float32 the `exp(±λ)` round-trip in the eigenbasis loses precision at
  large stretch.

## Failure modes

This is the high-value section. The legacy population-VE path has a documented
history of self-consistent but wrong Cd; cite the receipts before trusting a number.

- **M48 silent fixture toggle** — an uncommitted `embedded_gradient=true` flip in one
  VE fixture diverged its numerics from siblings and burned ~6 missions (R=50 NaN +
  plateau drift) before a JSONL grep found it. **Always grep fixture toggles BEFORE
  any hypothesis.** (See MEMORY: M48 toggle flip postmortem.)
- **`formulation=:stress` is the driver DEFAULT but is the unstable form.** The kwarg
  default in `run_viscoelastic_cylinder_2d` is `:stress` (direct τ_p UCM), which loses
  positive-definiteness at high Wi (`StressFormulation` docstring). For high-Wi runs
  pass `formulation=:logconf` explicitly — the default does NOT pick the stable path.
- **`Cd_pressure` dominates the cylinder Cd gap, NOT the constitutive model.** The
  Wi=1 Cd gap is ~80% pressure × front-pole (a halfway-BB wall-BC effect), and the
  R=60 NaN is a polymer back-force divergence at the front shoulder — same mechanism.
  Do wall decomposition (Cd_p + Cd_s + Cd_pressure per θ) FIRST; volume L2(τ) and peak
  τ_xx are not monotonic in Cd (MEMORY: M32 Phase 4 verdicts; Cd wall vs volume).
- **NaN triage fingerprint.** ≥90% NaN domain ⇒ BC over-bounce; bilateral
  front-shoulder arcs at θ≈±45°, r−R∈[0,7] ⇒ polymer back-force divergence
  (D2bis fingerprint) (MEMORY: NaN uniform vs arc diagnostic).
- **Per-step host-copy drag integral.** `compute_polymeric_drag_2d` is a CPU loop with
  one-sided 1.5/−0.5 extrapolation of τ_p from cell centre to wall; the wall-ring
  index frame matters — the legacy `:idx` decomposition has bitten Cd_polymer by tens
  of percent in the FV post-processing (MEMORY: wall-ring idx frame). On a closed box
  with q_wall=0.5 the smoke collapses to halfway-BB and hides cut-link bugs — use an
  R=4–8 cut-link cylinder smoke instead (MEMORY: smoke must exercise cut-links).
- **Acoustic-scaling Cd U-shape (M59–M61).** Sweeping R at fixed Re under acoustic
  scaling drifts τ and gives a non-monotone "U-shape" Cd artifact (NOT a bug); fix τ_LU
  and use diffusive scaling for R-sweeps (MEMORY: diffusive scaling rule).
- **Oldroyd-B has NO shear-thinning.** A Cd minimum at intermediate Wi is wake
  elongation (N1∝Wi²), not shear-thinning; do not over-read it (MEMORY: Oldroyd-B no
  shear-thinning). The GNF `effective_viscosity` clamps `gamma_dot` to `1e-30` to avoid
  the `0^(n-1)` power-law singularity — a very small γ̇ silently returns `nu_max`.

## Touch order

For a suspected viscoelastic/rheology bug (wrong Cd, NaN, wrong ν), inspect:

1. `src/drivers/viscoelastic.jl` — the orchestration: the `:stress` vs `:logconf`
   branch, the `extrapolate` polymer-drag call, the averaging window, the f_in/f_out
   swap, and the Cd normalisation. Most "wrong Cd / wrong number" bugs are the
   formulation default or the host-loop drag here. Check fixture toggles FIRST (M48).
2. `src/kernels/viscoelastic_2d.jl` — the conformation evolution: `evolve_logconf_2d!`
   (Fattal–Kupferman source/relaxation, Peterlin clamp), `evolve_stress_2d!` (UCM
   upwind), `compute_polymeric_force_2d!` (stress-divergence body force). NaN at the
   front shoulder usually traces to the force/relaxation terms here.
3. `src/rheology/linalg.jl` — for log-conf instability: the `eigen_sym2x2` /
   `decompose_velocity_gradient` degeneracy branch (`|Δ|<1e-12`) and the eigenvalue
   clamps; a wrong eigenframe corrupts the whole Θ update.
4. `src/kernels/collide_rheology_2d.jl` — for GNF (γ̇-dependent ν) Cd/stability:
   the `tau_field` (γ̇↔τ implicit) coupling, `ω = 1/(3ν+0.5)`, the Guo forcing block.
5. `src/rheology/viscosity.jl` — the per-model `effective_viscosity` closed form and
   its `clamp`/`max(γ̇,1e-30)` guards; a wrong exponent or clamp lives here.
6. `src/rheology/strain_rate.jl` — the Π^neq → S → γ̇ algebra (the `cs²=1/3` denominator
   and the per-q contribution lists); a sign/denominator slip mis-scales every viscosity.
7. `src/rheology/models.jl` — only for constructor/promote or new-model issues; the
   value structs themselves rarely host runtime bugs.

> If the bug points at the finite-volume log-conf coupled solver, STOP: that code is on
> `feat/ve-logfv-on-v03`, not here — switch branches (merge pending) rather than editing
> this legacy population path.
