# M5-A kinetic-moment BSD design

## Problem statement

M4 measured a structural split between the force field reconstructed by
centered finite differences and the force field actually supplied to the
LBM Guo collision in the cavity driver.  In the saved N=64 cavity
snapshots at `u_max` in `{0.005, 0.002, 0.001}`, the relative
`L2(F_FD - F_Guo) / L2(F_FD)` is 53.5--53.8 %, with the maximum
cellwise difference at `(i, j) = (16, 63)`; see the mandate M4 entry and
`bench/viscoelastic_logfv/analyse_cavity_guo_vs_fd_2d.jl`.

The audit script computes three fields in `compute_force_fields`
(`analyse_cavity_guo_vs_fd_2d.jl:203-220`): `F_FD = div(tau_p)` by
centered finite differences, `lap_u` by centered finite differences, and
`F_Guo = F_FD - zeta * nu_p * lap_u`.  The discrepancy is therefore not
the polymer stress divergence alone.  It is dominated by the BSD
correction `-zeta * nu_p * lap_u` evaluated with the explicit
FD-central velocity laplacian.

The production cavity path applies the same split in
`run_viscoelastic_logfv_cavity_coupled_2d`: Step 6 computes
`fx_poly, fy_poly = div(tau_p)` (`src/drivers/viscoelastic_logfv_2d.jl:1068-1071`),
Step 7 applies `logfv_bsd_correct_force_bc_aware_2d!`
(`viscoelastic_logfv_2d.jl:1073-1077`), and Step 8 passes the corrected
`fx_total, fy_total` to `fused_trt_libb_v2_guo_field_step!`
(`viscoelastic_logfv_2d.jl:1079-1083`).

The current BSD kernel path uses the discrete velocity laplacian

```text
lap_u = (u[i+1,j] - 2u[i,j] + u[i-1,j]) / dx^2
      + (u[i,j+1] - 2u[i,j] + u[i,j-1]) / dy^2
```

as implemented in `src/kernels/logconformation_fv_2d.jl:579-582`, with
a solid-aware variant at `logconformation_fv_2d.jl:609-631` and the
FVFD-backed wrapper at `logconformation_fv_2d.jl:708-717`.  The LBM,
however, relaxes the pulled D2Q9 populations through the TRT collision
operator, so its viscous response is the implicit lattice operator
encoded in the non-equilibrium moments of those pulled populations.
At finite N=64 resolution, those two laplacians are not interchangeable;
M4 quantified the mismatch as roughly 54 % in force L2.

## Continuous-limit theory

For D2Q9 in lattice units, `cs^2 = 1/3`.  The BGK Chapman-Enskog
identity for the raw non-equilibrium second moment is

```text
Pi^{neq}_{alpha beta}
  = sum_q c_{q alpha} c_{q beta} (f_q - f_q^eq)
  = -2 rho cs^2 tau S_{alpha beta} + O(Ma^2)
```

with `S_{alpha beta} = 1/2 (partial_alpha u_beta + partial_beta u_alpha)`.
The hydrodynamic viscous stress uses the half-step correction
`(1 - 1/(2tau)) Pi^{neq}`, which gives
`nu = cs^2 * (tau - 1/2)`.

For TRT, the symmetric even mode is relaxed by `s_plus`.  The codebase
documents and implements this in `src/kernels/fused_trt_2d.jl:100-128`:
`s_plus = 1 / (3nu + 1/2)` and
`nu = (1 / s_plus - 1/2) / 3`.  Therefore the symmetric traceless
non-equilibrium moment satisfies, to the CE order used here,

```text
Pi^{neq,raw}_{alpha beta}
  = -(2 rho cs^2 / s_plus) S_{alpha beta} + O(Ma^2),
```

while the lattice viscous stress moment that appears in the momentum
equation is

```text
Pi^{neq,visc}_{alpha beta}
  = (1 - s_plus / 2) Pi^{neq,raw}_{alpha beta}.
```

The same prefactor appears in the Guo collision brick as
`guo_pref = 1 - s_plus / 2` (`src/kernels/dsl/bricks.jl:168-171`).
Taking the incompressible divergence of the symmetric traceless part,

```text
nu_eff * nabla^2 u_alpha
  = cs^2 * (1 / s_plus - 1/2) * nabla^2 u_alpha
  = -(1 / rho) partial_beta Pi^{neq,visc}_{alpha beta}
    + O(Ma^2).
```

Thus the BSD term should be formed from the same lattice viscosity used
by the LBM step:

```text
-zeta * nu_p * nabla^2 u_alpha
  = (zeta * nu_p / nu_eff)
    * (1 / rho) partial_beta Pi^{neq,visc}_{alpha beta}
    + O(Ma^2),

nu_eff = cs^2 * (1 / s_plus - 1/2).
```

In the cavity driver, `nu_lbm_t = nu_s_t + bsd_t * nu_p_t`
(`src/drivers/viscoelastic_logfv_2d.jl:896-901`) is the viscosity passed
to `fused_trt_libb_v2_guo_field_step!` (`viscoelastic_logfv_2d.jl:1080-1083`).
Therefore `s_plus` must be derived from that LBM viscosity, and
`nu_eff` must be reconstructed from `s_plus = trt_rates(nu_lbm_t)[1]`,
not from the bare solvent viscosity alone.  For the default M4/M4b cavity setting
`zeta = bsd_fraction = 0.75`, the scale is
`zeta * nu_p / (nu_s + zeta * nu_p)`.

## Discrete consistency requirement

For this mission, "bit-exact" means the BSD correction and the Guo body
force are assembled from the same per-cell lattice non-equilibrium
moment and the same finite-volume divergence stencil, so their
difference is limited to normal floating-point rounding.  The strict
IEEE sense of bit equality is not achievable if the moment extraction is
performed outside the generated LI-BB kernel, because
`fused_trt_libb_v2_guo_field_step!` performs pull streaming, solid
handling, and `ApplyLiBBPrePhase()` before `Moments()`
(`src/kernels/li_bb_2d_v2.jl:49-54`).  If Phase B extracts moments from
`f_in` before that pre-phase, wall-adjacent cells with `q_wall > 0` can
legitimately differ.

The discrete operators are:

1. FD-centered, current: the interior second-difference laplacian at
   `src/kernels/logconformation_fv_2d.jl:579-582`.
2. Solid-aware FD-centered, current: the one-sided/zero fallback path at
   `logconformation_fv_2d.jl:622-626`, with helpers at
   `logconformation_fv_2d.jl:846-873`, and the FVFD-backed version at
   `src/fvfd/operators_2d.jl:863-908`.
3. Kinetic-moment, proposed:

```text
-zeta * nu_p * nabla^2 u_alpha
  = (zeta * nu_p / nu_eff)
    * (1 / rho) * (partial_beta Pi^{neq,visc}_{alpha beta})_FV.
```

Pick (3).  Compute `Pi_xx`, `Pi_xy`, and `Pi_yy` from the same D2Q9
population convention as `feq_2d(Val(q), rho, ux, uy, usq)`
(`src/kernels/equilibrium_helpers.jl:8-60`).  Then compute
`partial_x Pi_xx + partial_y Pi_xy` and
`partial_x Pi_xy + partial_y Pi_yy` at the same cell centers as the
polymer force.  This replaces the explicit velocity laplacian in the
BSD correction with the lattice-implied viscous operator.

The sign convention must be fixed in Phase B.  If the divergence kernel
returns the raw per-mass divergence
`D_alpha = (1 / rho) partial_beta Pi^{neq,visc}_{alpha beta}`, then
the total force is

```text
F_total_alpha = F_poly_alpha + (zeta * nu_p / nu_eff) * D_alpha.
```

If the helper instead returns the implied lattice laplacian
`L_alpha = -(1 / (rho * nu_eff)) partial_beta Pi^{neq,visc}_{alpha beta}`,
then the total force is

```text
F_total_alpha = F_poly_alpha - zeta * nu_p * L_alpha.
```

The implementation must not mix these two conventions.  The validation
script should include a sign-sensitive manufactured snapshot so the
wrong convention fails immediately.

With this choice, the field passed as `Fx_field, Fy_field` into
`fused_trt_libb_v2_guo_field_step!`
(`src/kernels/li_bb_2d_v2.jl:115-127`) is the polymer force plus the
negative of the LBM's own lattice-viscous contribution implied by
`Pi^{neq}`.  The Guo source `S_q` in
`CollideTRTDirectGuoField` (`src/kernels/dsl/bricks.jl:149-199`) then
receives a force whose BSD part is built from the same `s_plus`,
`guo_pref`, D2Q9 equilibrium, and cell-centered moment stencil.  The
expected precision ceiling is about `1e-12 / norm(F_FD)` in Float64 and
about `1e-6 / norm(F_FD)` in Float32 for interior cells.  Wall-adjacent
LI-BB pre-phase cells may be worse unless the extraction is fused inside
or immediately before the same generated kernel.

## Proposed kernel API

Phase B should add a dedicated kinetic BSD kernel file and keep the
interface backend-neutral.  The greppable API contract is:

```julia
# NEW FILE:
# /Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic/src/kernels/bsd_kinetic.jl

@kernel function compute_pi_neq_2d_kernel!(
    Pi_xx, Pi_xy, Pi_yy,
    @Const(f), @Const(rho), @Const(ux), @Const(uy), @Const(is_solid),
    Nx, Ny,
)
    # Π^{neq}_{αβ} = Σ_q c_{qα} c_{qβ} (f_q − f_q^eq)
    # ...
end

function compute_bsd_force_kinetic_2d!(
    fx_out, fy_out, fx_poly, fy_poly,
    f, rho, ux, uy, is_solid,
    zeta, nu_p, nu_s, s_plus, dx, dy;
    sync::Bool=true,
)
    # 1. extract Π^{neq} on the lattice
    # 2. compute (∂_β Π^{neq}_{αβ}) via FV divergence on cell-centres
    # 3. fx_out = fx_poly − (zeta * nu_p / nu_s) * div(Π^{neq})_x
    # 4. fy_out = fy_poly − (zeta * nu_p / nu_s) * div(Π^{neq})_y
end
```

The implementation note attached to that sketch is mandatory: the
argument named `nu_s` must not be used blindly as the denominator.  The
denominator for the kinetic correction is
`nu_eff = cs2 * (inv(s_plus) - 1/2)`, where `cs2 = 1/3`, matching
`trt_rates` (`src/kernels/fused_trt_2d.jl:125-128`).  If the local
variable `div(Pi)` denotes raw `+(1/rho) partial_beta Pi^{neq,visc}`,
use the plus-sign formula from the previous section.  If it denotes the
already-negated implicit laplacian, the pseudocode's minus sign is
correct.

Backend compatibility: both kernels must use `KernelAbstractions.@kernel`
and only fixed D2Q9 arithmetic, so they run on `CPU()`,
`CUDABackend()`, and `MetalBackend()`.  Inputs are
`f::AbstractArray{T,3}` with layout `(Nx, Ny, 9)`, and all scalar fields
are `AbstractMatrix{T}`.  Do not add dependencies.

Memory: the hot path needs three temporary `(Nx, Ny)` matrices:
`Pi_xx`, `Pi_xy`, and `Pi_yy`.  Prefer pre-allocation in the driver
beside the other hot-loop matrices (`src/drivers/viscoelastic_logfv_2d.jl:924-970`)
over allocating inside `compute_bsd_force_kinetic_2d!`.  The current
file does not contain a named `PreAllocBuffers` struct; if Phase B adds
one, these three arrays belong there.  For the cavity-only Phase B,
direct pre-loop allocation beside `fx_total` and `fy_total` is the
least invasive option.

Solid-cell policy: set `Pi_* = 0` and `fx_out = fy_out = 0` on solid
cells.  For a fluid cell, the FV divergence samples the same neighbours
as the existing solid-aware first-derivative helpers
(`src/kernels/logconformation_fv_2d.jl:804-843`): use centered
`left/right` or `down/up` if both neighbours exist and are fluid; use
second-order one-sided `center, right, right2` or
`center, up, up2` if only the forward side has two fluid cells; use
second-order one-sided `center, left, left2` or
`center, down, down2` if only the backward side has two fluid cells; use
first-order one-sided if exactly one fluid neighbour exists; otherwise
return zero.  Domain boundary handling should follow the FVFD BC index
pattern in `src/fvfd/operators_2d.jl:1-110` when this is promoted
beyond the closed cavity.

Moment extraction should eventually be fused into the LI-BB V2 generated
kernel immediately after `ApplyLiBBPrePhase()` and `Moments()` in the
spec at `src/kernels/li_bb_2d_v2.jl:49-54`.  For the cavity-only
prototype, an unfused pre-step kernel is acceptable for interior-cell
validation, but wall-adjacent results must be labelled as pre-phase
sensitive.

## Wire-in point

The surgical insertion point is
`run_viscoelastic_logfv_cavity_coupled_2d`
(`src/drivers/viscoelastic_logfv_2d.jl:860`).  The existing
`bsd_fraction::Real=1.0` kwarg is at
`viscoelastic_logfv_2d.jl:865`, and it is normalized to `bsd_t` at
`viscoelastic_logfv_2d.jl:899`.  Add one kwarg beside it:

```julia
# In the kwarg block (around line 865):
bsd_kind::Symbol = :fd,    # :fd (default, unchanged) | :kinetic
```

At Step 7 (`viscoelastic_logfv_2d.jl:1073-1077`), replace the direct
call with:

```julia
if bsd_kind === :kinetic
    compute_bsd_force_kinetic_2d!(
        fx_total, fy_total, fx_poly, fy_poly,
        f_in, rho, ux, uy, is_solid,
        bsd_t, nu_p_t, nu_s_t, s_plus_t, dx, dy,
    )
else
    logfv_bsd_correct_force_bc_aware_2d!(
        fx_total, fy_total, fx_poly, fy_poly, ux, uy, is_solid,
        bsd_t, nu_p_t, dx, dy, logfv_bc; sync=false,
    )
end
```

Phase B must compute `s_plus_t` with the same convention as
`fused_trt_libb_v2_guo_field_step!`, i.e. `trt_rates(nu_lbm_t)[1]`;
see `src/kernels/li_bb_2d_v2.jl:119-125` and
`src/kernels/fused_trt_2d.jl:125-128`.  The kwarg must validate
`bsd_kind in (:fd, :kinetic)`.

Edit budget: about 10--30 lines in
`src/drivers/viscoelastic_logfv_2d.jl`, about 80--150 lines in the new
`src/kernels/bsd_kinetic.jl`, and about 2 lines in `src/Kraken.jl` to
include/export the new function near the existing kernel includes and
exports (`src/Kraken.jl:27-80`, `src/Kraken.jl:141-188`).  The default
must remain `:fd`, so no caller changes behaviour unless it explicitly
sets `bsd_kind=:kinetic`.

Future expanded Phase B+1 threading should inventory every related
entrypoint in `src/drivers/viscoelastic_logfv_2d.jl`.  Current grep
sites include the lower channel/step loop around `427-439`, the channel
diagnostic at `1463-1465`, the frozen-channel diagnostic at
`1750-1754`, the Poiseuille coupled loop at `2798-2802`, the square
periodic loop at `3055-3059`, and the BFS passive Guo call at
`3201-3204`.  M5-A/M5-B should only wire the primary cavity driver at
`860-1090`.

## Validation strategy

Phase B should add
`bench/viscoelastic_logfv/analyse_cavity_bsd_kinetic_2d.jl`, mirroring
the M4 script structure in
`bench/viscoelastic_logfv/analyse_cavity_guo_vs_fd_2d.jl`.  The M4
script names the pure polymer force `F_FD` and the FD-BSD force
`F_Guo`; the Phase B script should disambiguate those as
`F_poly_FD = div(tau_p)` and
`F_FD_BSD = F_poly_FD - zeta * nu_p * lap_u_FD`, then add
`F_kinetic` from `compute_bsd_force_kinetic_2d!`.

The validation fields are:

```text
F_poly_FD        = div(tau_p)
F_FD_BSD         = F_poly_FD - zeta * nu_p * lap_u_FD
F_LBM_implicit   = F_poly_FD + (zeta * nu_p / nu_eff)
                   * (1 / rho) div(Pi^{neq,visc})_FV
F_kinetic        = same construction as F_LBM_implicit, through the new API
```

`F_LBM_implicit` is the body force implied by the LBM's own
non-equilibrium moments for the same `(rho, u, f)` snapshot and the same
FV divergence stencil.  The kinetic BSD aims to make `F_kinetic` and
`F_LBM_implicit` agree to rounding precision because both fields are
computed from the same `Pi^{neq}` data and the same sign convention.

Assertions on the saved cavity snapshots, with `F_FD` in the brief's
assertion (A) interpreted as the current FD-BSD total force
`F_FD_BSD`:

```text
(A) ||F_kinetic - F_FD_BSD||_2 / ||F_FD_BSD||_2 <= 5%
    loose; the two are allowed to differ by discretisation order at N=64.

(B) ||F_kinetic - F_LBM_implicit||_2 / ||F_LBM_implicit||_2 <= 1e-6
    in Float64, or <= 1e-3 in Float32.

(C) ||F_poly_FD - F_LBM_implicit||_2 / ||F_LBM_implicit||_2 ~= 54%
    sanity check that the M4 discrepancy is reproduced.
```

The initial assertion for (B) should be interior-only: cells with
`2 <= i <= Nx-1`, `2 <= j <= Ny-1`, and no adjacent solid/cut link in
the stencil.  This matches the M4 script's interior reduction
(`analyse_cavity_guo_vs_fd_2d.jl:253-289`) and avoids conflating the
kinetic operator with LI-BB pre-phase substitutions.  A second,
non-gating diagnostic should report all-cell and wall-adjacent norms.
If pre-phase LI-BB substitutions perturb `f` at cells with `q_wall > 0`,
the all-cell (B) norm may degrade to about `1e-4` near walls until the
moment extraction is fused into the generated LI-BB kernel.

The self-test mode should use a small synthetic `(rho, u, f)` fixture
inside `mktempdir()` following the existing memory rule and the M4
script's self-test style (`analyse_cavity_guo_vs_fd_2d.jl:337-388`).
No production-cost CPU run is required for Phase B.

## Risks and unknowns

- Wall-adjacent cells where halfway BB / LI-BB distorts `f` before
  moments are read.  Mitigation: assert (B) on interior cells only at
  first, then add a "kinetic-moment plus FD blend near walls" or fused
  extraction option if all-cell norms matter.
- TRT vs BGK rate convention.  The traceless symmetric moment uses
  `s_plus`, and `trt_rates` defines `nu = (1 / s_plus - 1/2) / 3`
  (`src/kernels/fused_trt_2d.jl:100-128`).  Mitigation: compute
  `s_plus_t = trt_rates(nu_lbm_t)[1]` and test a one-mode shear fixture
  to catch a factor-of-two or swapped-rate error.
- Raw moment versus viscous moment.  The raw `f - f^eq` moment needs the
  `guo_pref = 1 - s_plus / 2` half-step correction to represent the
  hydrodynamic stress (`src/kernels/dsl/bricks.jl:168-171`).
  Mitigation: document `Pi^{neq,visc} = guo_pref * Pi^{neq,raw}` and
  include the prefactor in both `F_kinetic` and `F_LBM_implicit`.
- Float32 precision on Metal.  `Pi^{neq}` is a difference of nearly
  equal populations.  Mitigation: set the Float32 ceiling at `1e-3` for
  validation, with an expected interior best case near `1e-6`.
- Multi-block interface coupling.  Kinetic extraction needs a 3x3
  stencil of valid `f` after streaming/pre-phase; at block interfaces,
  ghost cells must carry populations, not only `(rho, u)`.  Mitigation:
  defer to Phase B+1; the cavity is single-block.
- Polymer-solvent prefactor ambiguity.  The denominator is neither the
  polymer viscosity nor always the bare solvent viscosity; it is
  `nu_eff = cs2 * (1 / s_plus - 1/2)`, which equals
  `nu_s + zeta * nu_p` in the current cavity driver.  Mitigation:
  derive it from `s_plus`, not from a separately passed scalar.
- Coupling with `_logfv_cavity_apply_wall_gradient_correction!`.
  The driver applies wall-gradient correction before the polymer source
  (`src/drivers/viscoelastic_logfv_2d.jl:1036-1043`; kernel
  `viscoelastic_logfv_2d.jl:782-828`), while the current BSD path
  samples `u` directly.  Mitigation: kinetic BSD bypasses that gradient
  correction because the wall response is already encoded in the
  post-stream/pre-collision non-equilibrium moment; validate this with
  separate wall-adjacent diagnostics.
- Sign convention drift.  The raw divergence form and the implied
  laplacian form differ by a minus sign.  Mitigation: expose one helper
  name that states the convention, and add a sign-sensitive test before
  running cavity snapshots.

## Phase B scope estimate

Files to create: 2.

- `src/kernels/bsd_kinetic.jl`
- `bench/viscoelastic_logfv/analyse_cavity_bsd_kinetic_2d.jl`

Files to modify: 2.

- `src/Kraken.jl`: include/export the new kinetic BSD function.
- `src/drivers/viscoelastic_logfv_2d.jl`: thread `bsd_kind` through
  `run_viscoelastic_logfv_cavity_coupled_2d` only, not the other
  drivers in Phase B.

Estimated time: 3--5 hours of focused implementation for the cavity-only
path, plus 2--3 hours for validation against the M4 snapshots.

Blast radius: cavity driver only.  Other viscoelastic drivers, including
channel, cylinder, 3D, contraction, Poiseuille canaries, and BFS
canaries, keep the `:fd` default or remain untouched, so their behaviour
does not change.

Recommendation: implement cavity-only first.  Multi-driver rollout
should be a Phase B+1 follow-up after the cavity result confirms or
refutes the M4b BSD-sweep hypothesis.
