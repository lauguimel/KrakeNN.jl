---
module: ad
path: src/ad/; ext/KrakeNNADExt.jl
owner_concern: lbm-operator
status: implemented
last_verified: 2026-06-04
depends_on:
  - lbm
  - physics-newtonian
  - physics-thermal
  - bc
  - geometry
  - io-krk
---

# ad — module implication map

The `src/ad/` path owns the steady shape-adjoint capabilities:
`dCd/dR` for a residual-converged D2Q9 TRT/Li-BB cylinder and `dNu/dL`
for the coupled differentially heated Boussinesq cavity. Core code is
Enzyme-free; `ext/KrakeNNADExt.jl` supplies the reverse-mode seams when the
weak dependency `Enzyme` is loaded. The central maintenance contract is that
`src/ad/ad_step.jl`, `src/ad/ad_thermal_step.jl`, and `src/ad/ad_qoi.jl`
stay bit mirrors of their production operator/QoI paths, guarded by the AD
anti-drift tests.

## Public surface

- `steady_shape_sensitivity(; Nx, Ny, radius, u_in, ν, qoi=:drag, wrt=:radius, tol=1e-12, ...)` — exported drag API; returns a NamedTuple with `gradient` (`dCd/dR`), `qoi_value`, `value`, `solver`, `terms`, forward residual fields, and optional `fd_check`.
- `steady_shape_sensitivity(; N, Ra, Pr, qoi=:nusselt, wrt=:wall_position, L, q_hot, q_cold, ...)` — exported thermal API; returns the same result shape with `gradient` (`dNu/dL`) and thermal wall-position terms.
- `.krk` dispatch: parsed `Sensitivity { qoi = drag, wrt = radius }` and `Sensitivity { qoi = nusselt, wrt = wall_position }` requests are routed by `run_simulation(setup)` to `steady_shape_sensitivity`.
- Core implementation helpers are de-facto public as `KrakeNN.<name>` but are not exported: `ad_forward_solve`, `ad_thermal_forward_solve`, `ad_step!`, `ad_thermal_cut_step!`, `cd_pure`, `nu_pure`, `cd_production`, `gmres_adjoint`, `ad_gauge_augmented_adjoint`, `ad_assemble_radius_terms`, `ad_assemble_wall_position_terms`, `ad_fd_dCd_dR`, `ad_fd_dNu_dL`.
- Extension seam methods imported by `KrakeNNADExt`: `_ad_dJdf`, `_ad_vjp_GtT`, `_ad_dqwall_terms`, `_ad_dNudw`, `_ad_thermal_vjp_GtT`, `_ad_thermal_dqwall_terms`. In core they throw "Load Enzyme"; the extension replaces them with Enzyme reverse passes.

## Reads from

- `lbm` / `physics-newtonian` — D2Q9 weights, directions, opposite pairs, TRT rate convention, and the production TRT/Li-BB algebra that `ad_step!` mirrors.
- `physics-thermal` — Boussinesq DDF conventions, Guo buoyancy, thermal BGK advection, Dirichlet temperature wall algebra, and the hot-wall Nusselt stencil mirrored by `ad_thermal_cut_step!` / `nu_pure`.
- `bc` — the west velocity / east pressure Zou-He rebuild algebra mirrored inside `ad_apply_zou_he_rebuild!`.
- `geometry` — `precompute_q_wall_cylinder` / `dq_wall_dR_cylinder` for drag and `ad_cavity_wall_geometry` for the thermal moving-wall `q_wall` / `dq_dL` chain.
- `io-krk` — the parser-owned `Sensitivity` request and runner extraction of drag keywords (`Nx`, `Ny`, `radius`, `u_in`, `ν`, `ρ_out`) and thermal keywords (`N`, `Ra`, `Pr`, `L`, `wall_position`, `q_hot`, `q_cold`, `T_hot`, `T_cold`) plus tolerances.
- `Project.toml` weakdep/extension metadata — `KrakeNNADExt = "Enzyme"` keeps Enzyme out of the unconditional `using KrakeNN` load path.

## Writes to

- Returns a fresh NamedTuple from `steady_shape_sensitivity`; it writes no files and mutates no global registries.
- Mutates local host arrays during the forward fixed-point solves (`f_in`/`f_out` for drag, stacked `w=(f,g)` buffers for thermal) and during one-step AD products (`out`, cotangent arrays). These arrays are allocated inside the call or copied from inputs.
- `ext/KrakeNNADExt.jl` allocates cotangent work arrays for drag (`dJ/df`, `(dG/df)^T v`, `dCd/dq_wall`, `d(lambda^T G)/dq_wall`) and thermal (`dNu/dw`, coupled `(dG/dw)^T v`, explicit `dNu/dq_wall`, and flow/thermal `d(lambda^T G)/dq_wall`).
- Optional `fd_check=true` re-runs two additional perturbed forwards (`R ± h` for drag, `L ± h` for thermal) and returns their finite-difference data; it does not cache results globally.

## Backend constraints

- CPU Float64 only. The public API does not accept a backend keyword, and `.krk` sensitivity dispatch rejects non-`Float64` execution.
- No GPU kernel is differentiated. `ad_step!` and `ad_thermal_cut_step!` are plain Julia, host-side mirrors of the production fused paths; production GPU forward runs still use the production drivers.
- Memory is O(1) in the number of forward steps because Enzyme tapes one step at the converged state (`f*` for drag, stacked `w*=(f*,g*)` for thermal), not the transient history. GMRES still allocates host Krylov work arrays proportional to the state size and restart.
- Enzyme is optional at package load but mandatory at call time. Without `using Enzyme`, core stubs throw before any derivative work starts.
- Drag q-wall Enzyme paths have finite-difference fallbacks and directional finite-difference guards because Enzyme can return nonfinite or zero cut-link cotangents for this mutation-heavy path. Thermal q-wall terms return a directional finite-difference guard for the combined flow-plus-thermal moving-wall contraction.
- The thermal adjoint uses gauge-augmented GMRES to pin the singular flow mass mode; removing the gauge can make the coupled `rho = 1` mode look like an adjoint failure.

## Failure modes

- **Weakdep seam not loaded** — calling `steady_shape_sensitivity` after only `using KrakeNN` fails at the first Enzyme seam (`_ad_dJdf`, `_ad_dNudw`, or a VJP/q-wall seam) with the explicit "Load Enzyme" error. This is intentional packaging behavior from M-AD-P5.
- **Bit-mirror drift** — any production change to `fused_trt_libb_v2_step!`, the Zou-He rebuild, `compute_drag_libb_mei_2d`, direction tables, TRT rates, `fused_natconv_step!`, Boussinesq/thermal collision algebra, thermal wall BCs, or the Nusselt stencil must be reflected in `src/ad/ad_step.jl`, `src/ad/ad_thermal_step.jl`, and/or `src/ad/ad_qoi.jl`. The AD anti-drift tests in `test/ad/test_ad_sensitivity.jl` are the receipt: inline `Cd` and `Nu` must remain equivalent to their production mirrors.
- **Dropped direct diameter term** — `dCd/dR` includes `direct_D = -Cd/R` because `Cd = 2Fx/(u_ref^2 D)` and `D=2R`. Losing this term gives a plausible but wrong sign/magnitude; see `ad_assemble_radius_terms`.
- **Dropped moving-wall term** — `dNu/dL` includes explicit Nusselt wall geometry plus implicit flow-wall and thermal-wall terms. Losing either implicit contribution can still leave a plausible central-FD scale but breaks the moving wall contract; see `ad_assemble_wall_position_terms`.
- **Differentiating the Boolean mask / invalid cut interval** — the cylinder `is_solid` mask is held constant, and the cavity wall cut distances must stay in `(0, 1]`. The derivative is valid only within a smooth cut-set interval; finite differences crossing topology or wall-index changes can disagree by construction.
- **Forward not converged tightly enough** — drag and Nusselt can lag the population residual, so sensitivity runs use residual-converged forwards (`tol` near `1e-12` for drag validation, `1e-11` for thermal validation). A loose forward tolerance contaminates both the adjoint and central FD check.
- **Missing mass gauge in thermal adjoint** — the coupled Boussinesq state has a singular `rho = 1` flow mass mode. The thermal path must use `ad_gauge_augmented_adjoint`; a plain solve can stall even when the VJP is correct.
- **Unsupported API symbols** — anything except `qoi=:drag, wrt=:radius` or `qoi=:nusselt, wrt=:wall_position` throws `ArgumentError`. Do not document unvalidated pairs as extension points that work today.
- **Incomplete `.krk` thermal setup** — `qoi=nusselt` dispatch requires `Module thermal`, D2Q9, a square cavity grid, `Ra`, and `Pr`. Missing these fails before derivative work starts.

## Touch order

For an AD sensitivity bug, inspect in this order:

1. `src/ad/ad_api.jl` — public keyword contract, Enzyme-stub calls, result fields, supported `qoi`/`wrt`, and tolerance wiring.
2. `src/ad/ad_forward.jl` — residual-converged host solves, cylinder/cavity geometry setup, inlet/thermal parameters, and `Cd`/`Nu` evaluation at the converged state.
3. `src/ad/ad_qoi.jl` — inline MEI drag mirror, inline Nusselt mirror, q-wall finite-difference guards, and `lambda^T G` scalars used for implicit geometry terms.
4. `src/ad/ad_geometry.jl` — `direct_D`, `ad_cavity_wall_geometry`, q-wall contractions, finite-difference `dCd/dR` / `dNu/dL`, and returned `terms`.
5. `src/ad/ad_adjoint.jl` — Richardson-to-GMRES solve, gauge-augmented GMRES, linear residual norms, restart limits, and `rhohat` stall detection.
6. `ext/KrakeNNADExt.jl` — Enzyme reverse passes for `dJ/df`, `dNu/dw`, `(dG/df)^T v`, coupled `(dG/dw)^T v`, q-wall cotangents, runtime-activity fallback, and directional-FD guards.
7. `src/ad/ad_thermal_step.jl` — the unfused coupled Boussinesq bit mirror; check buoyancy `dG_f/dg`, advection `dG_g/df`, flow cut-links, thermal Dirichlet links, and mass-gradient support before changing constants.
8. `src/ad/ad_step.jl` — the unfused TRT/Li-BB plus Zou-He bit mirror; check this against `src/kernels/li_bb_2d_v2.jl` and `src/bc/rebuild_2d.jl` before changing constants.
9. `.krk` path: `src/ad/ad_krk.jl`, `src/io/krk/directives.jl`, `src/io/krk/parser.jl`, and `src/simulation_runner.jl` — only after the direct Julia API works but `Sensitivity { ... }` dispatch fails.
