# Steady Shape Sensitivity — Viscoelastic (AD)

Kraken can compute a cheap steady shape sensitivity of the polymeric drag on a
confined Oldroyd-B cylinder, `d(Cd_polymer)/dR`, with respect to the cylinder
radius. This extends the validated CPU Float64 AD track (cylinder drag `dCd/dR`
and thermal Nusselt `dNu/dL`, documented on the
[Steady Shape Sensitivity (AD)](ad-shape-sensitivity.md) page) to the coupled
viscoelastic solver. The public entry point is the same
`steady_shape_sensitivity`; the new validated pair is
`qoi=:polymer_drag, wrt=:radius`. The `.krk` surface is the same
`Sensitivity { ... }` block with `qoi = polymer_drag`.

The result is a derivative of Kraken's own discrete log-conformation operator at
a residual-converged steady state, not a transient tape. The forward solve first
converges the coupled fixed point on the stacked state `w = (f, psi)`, where `f`
is the D2Q9 flow distribution and `psi` is the log-conformation tensor,

```text
w* = G(w*, R)
```

and the adjoint then solves the open-BC system

```text
(I - dG/dw)^T lambda = dJ/dw
```

with one-step reverse-mode vector-Jacobian products. Each VJP re-linearises the
same single coupled `(f, psi)` step at `w*`, so memory is O(1) in the number of
forward steps: there is no unrolled trajectory and no tape of the transient
history. The polymeric drag QoI `J = Fx_polymer` is differentiated wrt the `psi`
block, since the polymer stress `tau_p = (nu_p/Wi)(C - I)` is recovered from the
log-conformation field at the cut points.

## Shape Chain

The radius enters through both the FVFD cut-cell geometry and the LBM cut links,

```text
R -> { geom(R), q_wall(R) } -> { G, J }
```

with the Boolean `is_solid` mask held fixed inside one smooth cut-set interval.
The gradient is assembled as the explicit geometry partial plus the implicit
state response,

```text
dJ/dR = (partial J / partial R)|geom  +  lambda^T . (dG/dR)
```

The explicit term is the only finite difference in the production gradient: a
central difference of the QoI over the geometry (cut points and embedded `g`) at
a frozen state `w*`. The state-response term is finite-difference-free — it
contracts the adjoint `lambda` with the analytic `dG/dR`, built from the analytic
field derivatives `d(geom)/dR` and `dq_wall/dR` through a single forward-JVP of
the coupled step. The analytic `dG/dR` must be exact because the net derivative
is a catastrophic cancellation (see Validation): a finite-differenced `dG/dR`
would inject truncation noise into the cancelling terms.

## CPU AD, GPU Production

The differentiated path is deliberately CPU Float64 and separate from the GPU
production forward. The AD steps in `src/ad/ad_ve_*.jl` are unfused, plain-Julia
mirrors of the production M8 viscoelastic coupled step: pull-stream plus LI-BB
cut links plus TRT-Guo collide for `f`, the embedded FVFD log-conformation
advection / constitutive substeps / polymer-force divergence for `psi`, and the
fused west-velocity / east-pressure Zou-He rebuild. These mirrors are kept bit-
exact to the production operator per step, guarded by a permanent anti-drift
check that compares the inline polymeric-drag QoI against the production
`compute_polymeric_drag_2d` on the same `(tau, q_wall)` to machine zero.

Enzyme is a weak dependency: `using Kraken` loads the core AD stubs without
Enzyme, and `using Enzyme` activates `ext/KrakenADExt.jl`, which supplies the
reverse passes (`dJ/dw`, the coupled `dG^T . v`, and the `dG/dR` forward-JVP).
Calling the sensitivity API without loading Enzyme raises a clear error.

## Julia Example

```julia
using Kraken
using Enzyme

result = steady_shape_sensitivity(;
    qoi=:polymer_drag,
    wrt=:radius,
    Nx=32,
    Ny=32,
    radius=8.130,
    cx=16.350,
    cy=15.650,
    Wi=0.5,
    beta=0.5,
    nu_p=0.02,
    nu_s=0.08,
    Fx_body=2e-4,
    fwd_tol=1e-13,
    bc=:open,
    fd_check=false,
)

result.gradient        # d(Cd_polymer)/dR
result.qoi_value       # Cd_polymer at the converged state
result.value           # alias of the Cd_polymer value
result.terms           # explicit geometry + implicit state-response pieces
result.terms.bc        # :open
result.solver.gauge    # :ungauged for the open cylinder
result.fd_check.relerr # populated only when fd_check=true
```

`Wi` sets the relaxation time (`lambda = Wi`); the polymer stress prefactor is
`nu_p/Wi`. The solvent viscosity `nu_s` sets the TRT relaxation rates and
`Fx_body` is the frozen driving body force (used as the inlet mean by default).
The forward reconverges to `fwd_tol=1e-13`; do not loosen this floor (see
Validation).

## .krk Example

`examples/sensitivity_cylinder_polymer.krk` issues the same request
declaratively. The `qoi=polymer_drag` dispatch requires `Module viscoelastic`,
D2Q9, and an `oldroyd_b` rheology block:

```krk
Simulation sensitivity_cylinder_polymer D2Q9
Domain L = 32 x 32  N = 32 x 32

Define R = 8.130
Define cx = 16.350
Define cy = 15.650
Define Wi = 0.5
Define beta = 0.5
Define Fx_body = 2e-4
Define samples = 16

Physics R = R Wi = Wi beta = beta cx = cx cy = cy Fx_body = Fx_body samples = samples
Rheology oldroyd_b { nu_s = 0.08 nu_p = 0.02 lambda = Wi }
Module viscoelastic

Obstacle cylinder wall(radius = R) { (x - cx)^2 + (y - cy)^2 <= R^2 }
Boundary west  velocity(ux = Fx_body, uy = 0)
Boundary east  pressure(rho = 1.0)
Boundary south wall
Boundary north wall

Run 60000 steps

Sensitivity { qoi = polymer_drag, wrt = radius }
```

Running this file through `run_simulation` dispatches to the same AD API:

```julia
using Kraken
using Enzyme

result = run_simulation("examples/sensitivity_cylinder_polymer.krk")
result.gradient
```

## Validation

The standalone adjoint `dJ/dR` matches a central finite difference of the
converged forward to **0.63%** (under the 1% target; re-measured on 2026-10-12 after
the TRT Guo forcing fix, #67, it was 0.42% before). The finite-difference
reference rebuilds the matched geometry at `R ± h`, reconverges each forward
tightly, and central-differences the inline polymeric drag at the perturbed
converged states; it also asserts the cut-link topology (cut count and solid
count) is unchanged across `±h`.

| Rung | Check | Gate |
|---|---|---|
| C0 | one-step coupled VJP on the stacked `(f, psi)` state | Enzyme VJP matches the discrete step |
| C1 | steady fixed-point adjoint on the coupled VE map | adjoint solve and finite-difference derivative agree |
| C2 | per-channel `lambda^T (dG/dR)` against frozen-solid finite difference | analytic geometry seed validated through the operator |
| C3 | confined-cylinder `d(Cd_polymer)/dR` | central finite difference agreement to `0.63%` |

The single most important practical caveat is the forward tolerance. The net
`d(Cd_polymer)/dR` is a **catastrophic cancellation** between the
explicit geometry partial and the implicit state response. The forward must
reconverge to `fwd_tol=1e-13`; a looser `1e-11` patience floor poisons the
finite-difference reference, blowing the agreement out to **53.3%**. The tight
floor is therefore the default in `ad_ve_forward_solve` and in the API.

The validated envelope is `Wi <= 1` and `beta >= 0.5`. As with the Newtonian
cut-link derivative, the reference for differentiating this discrete operator is
a central finite difference of Kraken's own residual-converged forward solve, not
an external code: a finite-volume tool can produce a polymeric drag value but
cannot differentiate Kraken's LBM/FVFD cut-link operator with respect to `R`.

## Caveats

- Only CPU Float64 is supported on the AD path. The GPU production path remains
  the path for forward simulations; this is the separate differentiable path,
  kept bit-exact per step and guarded by the anti-drift check.
- The validated pair is `qoi=:polymer_drag, wrt=:radius`. Any other pair throws
  `ArgumentError`.
- `using Enzyme` is required before calling the API.
- The forward must reach `fwd_tol=1e-13`. A looser floor silently poisons the
  finite-difference cross-check (53.3% at `1e-11`) even when the adjoint itself
  is correct.
- The derivative is a smooth-piece cut-link / cut-cell derivative; avoid radius
  values where a cut link or cut cell enters or leaves the mask when comparing to
  finite differences.
- The cylinder uses an open inlet/outlet boundary condition (west velocity
  inlet, east pressure outlet). The open BC pins the `rho = 1` mass mode, so the
  adjoint is solved ungauged (`solver.gauge == :ungauged`); the mass-gauged path
  is reserved for the closed / periodic boundary conditions.
- The validated envelope is `Wi <= 1`, `beta >= 0.5`. Higher `Wi` or lower
  `beta` is outside the validated range.
