# C5 local immersed-wall MMS: scientific STOP

Date: 2026-11-05. CPU, macOS arm64, Julia 1.12.5, Float64. This is step 3
only. No closure is selected. Steps 4/5, integrated force/torque objectives,
closed-domain mass tests, shape derivatives and Aqua execution have not begun.
No production residual formula was changed by this diagnostic.

The two choices that reproduce the complete affine manufactured solution are
L/M1/linear-pressure and Q/M1/linear-pressure. Neither reaches the required
1.8 order for physical wall traction from the **solved** local problem.
The permanent suite preserves eight failed scientific selection gates with
`@test_broken`: a passing test process does not make this scientific gate green.

## Selection evidence

L/Q is the residual's linear/quadratic ghost velocity; M1/M2 is its nominal-face/
physical-wall mass velocity. Z/L is its zero-gradient/linear nominal-face
pressure. Surface reconstruction always uses the same quadratic R10 slope and
pressure extrapolation to the actual cut for every row of the table.

The selection gate uses quadratic velocity, linear nonconstant pressure,
variable viscosity, all six orientations, and the last refinement 16 to 32.
Errors are RMS of the Euclidean traction error over every measured cut link
(`sqrt(sum(norm(error)^2)/number_of_links)`), **not** arc-weighted L2 or an
integrated resultant. The last-order column is the minimum over the six angles;
the listed angle attains it. A three-grid fit uses all 8/16/32 data equally in
log(h), and is an additional diagnostic, not a replacement for the gate.

| Ghost | Mass | Residual pressure | Affine exact | Worst last RMS order | Angle | Minimum three-grid fit |
|---|---|---|---|---:|---:|---:|
| L | M1 | Z | no | 0.713549 | 45° | 0.109552 |
| L | M1 | L | yes | 0.529989 | 45° | 0.761534 |
| L | M2 | Z | no | -1.040200 | 0° | -0.231995 |
| L | M2 | L | no | -0.999550 | 0° | -0.227422 |
| Q | M1 | Z | no | 0.932279 | 0° | 0.837806 |
| Q | M1 | L | yes | 0.715822 | 0° | 0.484403 |
| Q | M2 | Z | no | -1.012003 | 0° | -0.203942 |
| Q | M2 | L | no | -0.928863 | 0° | -0.199249 |

Across grids and orientations, affine injected `max(abs(R))/h^2` is
3.576e-12 for L/M1/L and 9.920e-12 for Q/M1/L. Z pressure gives an affine
momentum defect of 0.4 and a nominal-face pressure error of 0.05. M2/L gives
an affine mass defect up to 0.292625. The generic divergence-free affine
field has nonzero diagonal and off-diagonal gradients; at 45° its diagonal
strains happen to cancel the M2 error, which the other orientations expose.

For Q/M1/L at 30°, quadratic injected momentum truncation in cut cells is
0.271250 at N=16 and 0.271875 at N=32, while strict-interior error is below
9e-13. Injected surface traction RMS is approximately 7e-14: the quadratic
wall derivative and linear pressure reconstruction are polynomial identities.
After solving, traction RMS is 0.011134651 and 0.005808821 (order 0.93874).
All local solved equations satisfy `max(abs(R))/h^2 <= 2.10e-11`.
The persistent truncation and low traction order therefore are not an
unconverged linear solve.

An additional cubic-velocity/quadratic-pressure probe prevents roundoff-only
reconstruction checks from being mistaken for a measured order. Its raw
per-angle pressure, viscous and total traction errors and both refinement
orders are retained separately. Pressure and viscous parts are reported
separately so cancellation in the total traction remains visible. This probe does not
replace the required quadratic-velocity selection gate. Grid-to-grid cut
positions vary, so its raw injected pressure orders need not equal two on
each coarse pair; the independent exact pressure error is
`-a_e*q*(q+1)*h^2`, with `a_e=0.2` on x links and `0.25` on y links.

## Manufactured fields and measurement contract

The physical square is fixed at [0,1]^2. Grids N=8,16,32 and normal angles
0°,15°,30°,45°,60°,75° were fixed before measuring. The solid halfplane is
`n dot ((x,y)-(0.5,0.5)) >= 0.073`. Every cut whose fluid cell centre is in
the physical square is measured: no corner, small-q or near-tangent link is
removed. `local_mms_links.csv` enumerates all 484 measured links across grids
and angles, once each because geometry is independent of closure and field.

Five exact-state halo layers surround the physical square. Beyond the inner
three halo layers the solid is capped to leave an outer two-cell fluid collar.
The cap is a convex intersection with the halfplane; its wall data are also
manufactured, and all opposite fluid cells are checked. It prevents invalid
outer-grid ghost reads when evaluating the full production kernel. Neither
cap nor domain boundary can enter a radius-two stencil in the physical square.
The full `fv_residual` kernel is evaluated; only physical-square fluid rows
and unknowns enter the reduced linear solve. Halo velocity and pressure are
fixed to the analytic values. No halo equation or solid identity equation is
claimed solved. This is a localized wall test, not closed-domain validation.

The affine velocity is
`u=0.3+0.4x-0.7y`, `v=-0.2+0.6x-0.4y`.
The quadratic field adds
`(0.2x^2+0.3xy-0.15y^2, -0.15x^2-0.4xy-0.15y^2)`.
The cubic probe further adds `(0.07x^3+0.09xy^2, -0.21x^2y-0.03y^3)`.
All are divergence free. Pressure is `0.2+0.8x+0.5y`, with additional
`0.2x^2+0.15xy+0.25y^2` for the cubic probe. Thus the pressure normal
derivative is nonzero at every measured wall point. Viscosity is
`nu=1+0.2x+0.3y`, evaluated at cell centres for the residual and at the actual
cut for physical traction.

The source is the continuous Stokes expression
`grad(p)-div(nu*(grad(u)+grad(u)'))`, evaluated analytically and independently
checked by automatic differentiation. **No source comes from a discrete
residual.** The existing coloured production Jacobian is restricted to the
unknown subset and factored once per grid/configuration. All three fields
share that linear operator, so its factor is reused for their three right
hand sides. A single correction solves each linear Stokes problem.

`trunc_*` CSV columns always refer to exact injection, including on rows
labelled `solved`. They divide integrated residuals by h^2. Cut cells have an
immersed link; strict interior cells have no solid cell in their radius-two
box; `near` contains every remaining fluid cell. These groups partition the
physical fluid cells without discarding any. `equation_residual` instead
belongs to the row's actual state. `ghost` and `face` compare to the analytic
solid-centre and nominal-face velocity; the face average is not expected to
reproduce quadratic values at its midpoint. `cell_gradient` is at P and
`surface_gradient` is at the true cut. Those gradient errors are maximum
absolute component errors. Velocity and pressure errors are also maximum
absolute component errors over all solved cells; their gauge is fixed by
the exact pressure halo.

The internal surface helper reconstructs the full gradient from its quadratic
axial slope and the prescribed tangential derivative, explicitly dividing
by `e dot n`. Its positive transversality threshold is 1e-8. Errors name the
affected cells; no replacement or merging is implemented. Wall pressure is
`pP+q*(pP-pO)`, independent of residual closure, and traction is
`p*n-nu*(G+G')*n` (fluid on solid). Exact quadratic injected errors are labelled
`exact_to_roundoff` in the order CSV, never fitted as convergence orders.

## Localized residual evidence and next scope

The unchanged production path calls `_fv_wall_stress` from
`src/methods/inc_ns/fv_steady_residual.jl:368`. At lines 254 and 257 the cross
stress uses the tangential cell gradient `gP[2]` or `gP[3]` at P on the nominal
face. For quadratic fields with mixed derivatives this omits the half-cell
displacement: for an x link the missing tangential-gradient increment is
`(h/2)*partial_x(partial_y(u))`. Its stress error is O(h), hence its
contribution to `R/h^2` is O(1), consistent with the measured cut-cell defect.
Q ghost values, Q cell gradients and the physical-surface reconstruction
remain polynomial exact on injected quadratic fields. None of those identities
removes the residual defect or proves solved traction order.

The smallest next mission is a separately scoped examination of that nominal
cross-stress location and its Jacobian dependencies, using these tests as the
unchanged gate. This delivery does not implement that correction or authorize
the image-point fallback. No Aqua job or submission command was prepared.

## Reproduction and files

From the repository root, in the already prepared offline test environment:

```bash
env JULIA_DEPOT_PATH="$PWD/tmp/c5_depot:$HOME/.julia" JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 KRAKEN_C5_MMS_OUTPUT=benchmarks/fv_c5 "$HOME/.julia/juliaup/julia-1.12.5+0.aarch64.apple.darwin14/Julia-1.12.app/Contents/Resources/julia/bin/julia" --startup-file=no --project=tmp/c5_env -e 'using Kraken, Test; include("test/analytical/incns_fv_immersed_mms.jl"); include("test/analytical/incns_fv_immersed.jl")'
git diff --check
```

For a normal project environment, all five evidence CSV files regenerate with
`KRAKEN_C5_MMS_OUTPUT=benchmarks/fv_c5 julia --project test/analytical/incns_fv_immersed_mms.jl`.
The test is registered in `test/runtests.jl`. Evidence output is opt-in;
ordinary test runs do not write artifacts. No ignored temporary script is
needed to derive the order or selection tables.

Acceptance: 5,540 new assertions pass, 8 scientific gates are known broken,
and all 733 existing immersed geometry/C1 compatibility/Jacobian/viscosity/
adjoint assertions pass. The final new local configuration testsets take at most
2.54 s including first-call compilation; the largest immersed acceptance testset takes
6.9 s. The warm N=32 triplet pilot (three solves sharing one factor) took
0.0245 s. Each new MMS and immersed acceptance testset is below 30 s;
no long CPU simulation was run for this diagnostic.

The separate final existing residual/Newton/GNF/adjoint/compiler suite also
passes all 1,858 assertions, including the compiler guards. Its historical
GNF aggregate testset takes 41.4 s and is an explicit exception to the timing
claim above; it was not shortened or replaced. Its execution receipt is
`tmp/c5_s345/final_existing_cpu.log`.

- `local_mms_metrics.csv`: 864 rows, eight choices × six angles × three grids
  × three manufactured fields × injected/solved states.
- `local_mms_orders.csv`: every angle's first pair, last pair and three-grid
  fit, with total/pressure/viscous traction RMS and maximum errors separate.
- `local_mms_selection.csv`: eight-row selection table above.
- `local_mms_links.csv`: all measured geometric cuts, no filtering.
- `local_mms_timings.csv`: configuration/angle testset times.
- Raw command output: `tmp/c5_s345/acceptance.log`; final MMS evidence-writer
  check: `tmp/c5_s345/final_mms.log`; pilot output:
  `tmp/c5_s345/pilot.log`. Both are local execution receipts.
