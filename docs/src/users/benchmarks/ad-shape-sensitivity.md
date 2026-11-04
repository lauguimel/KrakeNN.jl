# Steady Shape Sensitivity (AD)

Kraken can compute steady shape sensitivities for two validated CPU Float64 AD
paths: cylinder drag, `dCd/dR`, on the D2Q9 TRT/Li-BB confined-cylinder path,
and thermal Nusselt sensitivity, `dNu/dL`, on the coupled Boussinesq cavity.
The public entry point is `steady_shape_sensitivity`; the `.krk` surface is a
`Sensitivity { ... }` block. The current validated pairs are exactly
`qoi=:drag, wrt=:radius` and `qoi=:nusselt, wrt=:wall_position`.

The result is a derivative of Kraken's own discrete operator at a residual-
converged steady state, not a transient tape. On the cylinder path, the forward
solve first converges the fixed point

```text
f* = G(f*, R)
```

and the adjoint then solves

```text
(I - dG/df)^T lambda = dJ/df
```

with one-step reverse-mode vector-Jacobian products. Each VJP re-linearises the
same single LBM step at `f*`, so memory is O(1) in the number of forward steps:
there is no unrolled trajectory and no tape of the transient history. Richardson
iteration is attempted first; GMRES is used on the same matrix-free operator when
the contraction is too slow.

## Shape Chain

The radius enters through the cut-link geometry,

```text
R -> q_wall(R) -> {G, J}
```

with the Boolean `is_solid` mask held fixed inside one smooth cut-set interval.
Kraken does not differentiate through the ray-circle precompute. Instead it
contracts Enzyme cotangents with the analytic `dq_wall_dR_cylinder` derivative:

```text
dCd/dR =
  <dCd/dq_wall, dq_wall/dR>
  - Cd/R
  + <d(lambda^T G)/dq_wall, dq_wall/dR>
```

The middle `-Cd/R` term is the direct derivative of the drag normalization
`D = 2R`. The gradient should be read as the smooth-piece derivative; if changing
`R` changes the set of cut links, the discrete `Cd(R)` can kink.

## CPU AD, GPU Production

The differentiated path is deliberately CPU Float64. Enzyme is a weak dependency:
`using Kraken` loads the core AD stubs without Enzyme, and `using Enzyme` activates
`ext/KrakenADExt.jl`, which supplies the reverse passes. Calling the sensitivity
API without loading Enzyme raises a clear error.

These paths do not differentiate fused GPU kernels. The AD steps in `src/ad/`
are unfused, plain-Julia mirrors of the production operators: TRT/Li-BB plus MEI
drag for the cylinder, and the coupled Boussinesq natural-convection step plus
hot-wall Nusselt sum for the cavity. The permanent anti-drift checks compare the
inline AD drag with `compute_drag_libb_mei_2d`, the inline steady state with the
production `run_cylinder_libb_2d` path, and the inline Nusselt value with the
thermal driver formula.

## Julia Example

```julia
using Kraken
using Enzyme

result = steady_shape_sensitivity(;
    Nx=48,
    Ny=16,
    cx=12,
    cy=8,
    radius=3.75,
    u_in=0.05,
    ν=0.05,
    qoi=:drag,
    wrt=:radius,
    tol=1e-9,
    max_steps=60_000,
    gmres_tol=1e-9,
    adjoint_tol=1e-8,
    fd_check=true,
    fd_h=0.05,
)

result.gradient        # dCd/dR
result.qoi_value       # Cd at the converged state
result.value           # current alias of the Cd value
result.terms           # explicit q_wall, direct_D, implicit q_wall pieces
result.fd_check.relerr # populated only when fd_check=true
```

The production default tolerance is tighter (`tol=1e-12`). The example above uses
the small `.krk` smoke geometry so the full AD path is cheap enough for local
tests.

## .krk Example

`examples/sensitivity_cylinder.krk` uses the same request declaratively:

```krk
Simulation sensitivity_cylinder D2Q9
Domain L = 48 x 16  N = 48 x 16

Define U = 0.05
Define H = 16
Define cx = 12
Define cy = 8
Define R = 3.75
Define tol = 1e-9
Define gmres_tol = 1e-9
Define adjoint_tol = 1e-8

Physics nu = 0.05

Obstacle cylinder wall(radius = R) { (x - cx)^2 + (y - cy)^2 <= R^2 }

Boundary west  velocity(ux = 4*U*y*(H - y)/H^2, uy = 0)
Boundary east  pressure(rho = 1.0)
Boundary south wall
Boundary north wall

Run 60000 steps

Sensitivity { qoi = drag, wrt = radius }
```

Running this file through `run_simulation` dispatches to the same AD API:

```julia
using Kraken
using Enzyme

result = run_simulation("examples/sensitivity_cylinder.krk")
result.gradient
```

## Thermal: Nusselt Sensitivity (Natural Convection)

The thermal pair computes `dNu/dL`: the derivative of the hot-wall average
Nusselt number with respect to the cavity wall position `L`. The forward solve
converges the coupled Boussinesq fixed point on the stacked state
`w = (f, g)`, where `f` is the flow DDF and `g` is the thermal DDF:

```text
w* = G(w*, L)
(I - dG/dw)^T lambda = dNu/dw
```

The new linearisation terms are the off-diagonal coupling blocks: buoyancy
`dG_f/dg`, where temperature drives the Guo body force on `f`, and advection
`dG_g/df`, where the velocity recovered from `f` drives the thermal equilibrium
for `g`. The adjoint solve is therefore one coupled GMRES problem on the stacked
state, with a mass-gauge augmentation for the singular `rho = 1` mode.

The wall parameter moves both wall operators. With `q_hot` fixed, the east-wall
cut distance is implied by `L`; the same moving `q_wall` reaches the flow
bounce-back terms and the thermal Dirichlet terms. The returned `terms` split
the gradient into explicit Nusselt wall geometry, implicit flow-wall geometry,
and implicit thermal-wall geometry.

```julia
using Kraken
using Enzyme

result = steady_shape_sensitivity(;
    qoi=:nusselt,
    wrt=:wall_position,
    N=16,
    Ra=1e3,
    Pr=0.71,
    L=16.2,
    q_hot=0.5,
    q_cold=0.7,
    tol=1e-11,
    max_steps=450_000,
    gmres_tol=1e-10,
    adjoint_tol=1e-10,
    fd_check=true,
    fd_h=0.01,
)

result.gradient        # dNu/dL
result.qoi_value       # Nu at the converged state
result.terms           # explicit_qwall, flow_qwall, thermal_qwall pieces
result.fd_check.relerr # populated only when fd_check=true
```

`examples/sensitivity_cavity_nusselt.krk` uses the same request declaratively:

```krk
Simulation sensitivity_cavity_nusselt D2Q9
Domain L = 1.0 x 1.0  N = 16 x 16

Define q_hot = 0.5
Define q_cold = 0.7
Define L = 16.2
Define tol = 1e-11
Define gmres_tol = 1e-10
Define adjoint_tol = 1e-10

Physics nu = 0.05 alpha = 0.07042253521126761 Pr = 0.71 Ra = 1e3
Module thermal

Boundary west  wall T = 1.0
Boundary east  wall T = 0.0
Boundary south wall
Boundary north wall

Run 450000 steps

Sensitivity { qoi = nusselt, wrt = wall_position }
```

The thermal validation ladder is the `AD thermal (Nusselt)` testset. Its
load-bearing gates are the coupled one-step VJP, the `dNu/dβ_g` adjoint check,
the conduction geometry chain, and the full convection-cavity `dNu/dL` check:

| Rung | Check | Gate |
|---|---|---|
| TC0 | coupled VJP on the stacked `(f, g)` state | relative agreement `6.1e-11` |
| TC1 | `dNu/dβ_g` adjoint check | finite-difference agreement `6.9e-6` |
| TC2 | conduction geometry chain | analytic `-alpha DeltaT/L^2` agreement `2.7e-15` |
| TC3 | convection cavity `dNu/dL` | central finite-difference agreement `1.0e-3` (test gate `< 2e-2`) |

The TC3 value is the one the `TC3 cavity dNu/dL` testset of
`test/ad/test_ad_sensitivity.jl` gives (CPU, Float64, measured 2026-11-04).

## Validation

The validation ladder checks the derivative machinery before the cylinder case is
trusted:

| Rung | Check | Gate |
|---|---|---|
| C0 | one-step VJP against dense finite-difference Jacobian transpose | Enzyme VJP matches the discrete step |
| C1 | steady fixed-point adjoint on a Poiseuille/body-force case | adjoint solve and finite-difference derivative agree |
| C2 | geometry chain on planar Poiseuille flow rate | analytic anchor `Q proportional to H^3` |
| C3 | confined-cylinder `d(Cd)/dR` | central finite difference agreement to `3.0e-3` (test gate `< 1e-2`) |

The C3 value is the one the `C3 cylinder radius sensitivity` testset of
`test/ad/test_ad_sensitivity.jl` gives (CPU, Float64, measured 2026-11-04).

The C2 anchor is the substitute reference for the geometry derivative: RheoTool
can compute a drag value for its own finite-volume operator, but it cannot
differentiate Kraken's LBM cut-link operator with respect to `R`. For derivatives
of this discrete operator, the meaningful references are the analytic Poiseuille
law `Q proportional to H^3` and central finite differences of Kraken's own
residual-converged forward solve.

## Caveats

- Only CPU Float64 is supported on the AD path.
- Validated pairs are `qoi=:drag, wrt=:radius` and
  `qoi=:nusselt, wrt=:wall_position`.
- `using Enzyme` is required before calling the API.
- The derivative is a smooth-piece cut-link derivative; avoid radius values where
  a cut link enters or leaves the mask when comparing to finite differences.
- The Nusselt path is the D2Q9 square, differentially heated cavity. Its `.krk`
  dispatch requires `Module thermal`; `L` must keep the implied wall cut
  distances inside `(0, 1]`.
- The GPU production path remains the production path for forward simulations;
  this page documents the separate differentiable path and its anti-drift checks.
