# C5 final local selection and closed-domain stop

Date: 2026-11-05. Branch `feat/fv-c5`, base `844e2992f`.
This note records local CPU Float64 checks and prepared GPU files only.

## Selection contract

Q/M1/linear pressure is selected by the final aligned-reference gate in
`benchmarks/fv_c5/README.md`: affine exactness, velocity order at least 1.8,
and pressure plus pressure/viscous/total traction orders at least the matching
aligned C1 order minus 0.2 on every refinement pair. Each immersed order is
the minimum across all six angles. The previous viscous-order requirement
is not met; its evidence is retained. Improving pressure and wall-shear order
for all C1-C5 walls remains a separate task.

## Distinct force and torque contracts

`FVWallForce(dir; kind=:residual_flux)` sums the complete numerical wall
momentum flux, including convection. `FVWallTorque(center; kind=:residual_flux)`
uses nominal face centres. The residual and this observer share the same
wall-flux components; a post-processing pressure override is not permitted.
Calls without `kind` retain the previous pressure-plus-viscosity convention.

`kind=:surface` uses the actual cut, quadratic axial velocity differentiation,
the prescribed tangential wall derivative, physical wall viscosity, extrapolated
pressure, and midpoint angular arc weights. Analytic geometry metadata carries
these quantities through parameter differentiation. Near-tangent cuts raise an
explicit error; points are neither replaced nor merged. Raw five-argument
immersed geometry remains available for residual-only data.

For a closed domain, write the momentum residual as
`R_P = sum_f F_Pf - h^2 s_P`. Internal fluxes cancel in the force sum, giving
`F_wall = sum_P R_P + h^2 sum_P s_P`. The corresponding moment identity is

```math
T_{\rm wall} = \sum_P r_P\times(R_P+h^2s_P)
 -\sum_{P|N}(r_P-r_N)\times F_{PN}
 +\sum_{f\in\mathrm{wall}}(r_f-r_P)\times F_{Pf}.
```

Each internal face appears once in the middle sum. Dropping its transport
couple or the wall half-cell shift would give a different moment balance.
A constant symmetric stress has zero residual-flux force and nominal-face
torque on a closed Cartesian contour. Point normals with midpoint arc weights
need not integrate a constant stress exactly on irregular cuts; the physical
surface objective is not subject to that discrete contour identity.

## M1 mass identity and failed gate

For the Q ghost closure, the nominal face velocity is exactly

```math
u_f=\frac{1-q}{2(1+q)}u_O+\frac{3q-2}{2q}u_P
    +\frac{u_w}{q(1+q)},\qquad
B(U)=h\sum_{\mathrm{cut}} e\cdot u_f.
```

Interior mass fluxes, including pressure stabilization, cancel pairwise.
With zero mass source, the sum of all continuity rows is `B(U)`. A pressure
pin replaces one of those rows: a converged pinned solve therefore satisfies
`B(U)=R_mass(pin)` up to its remaining algebraic residual, not necessarily zero.
The test restores the omitted row and measures `R_mass(pin)/h^2` separately.

The off-centre annulus has `(cx,cy)=(0.5113,0.4937)`, radii `(0.14,0.34)`.
Grids are 24, 40, 48; grid 32 is explicitly rejected by the existing
opposite-fluid-cell contract. Taylor-Couette uses unit inner angular velocity
and a fixed outer wall. The independent manufactured flow derives from
`psi=(r^2-Ri^2)(r^2-Ro^2)(1+0.4X+0.7Y+0.3XY)/Ro^4`, so its non-rigid wall
velocity is tangent to both circles and continuously mass compatible.
Its source is the analytic continuous Stokes source, checked independently
by automatic differentiation; no discrete correction or mass source is added.

The frozen order gate requires the global balance, normalized pin defect and
pin-induced velocity change to converge at least as fast as velocity error,
on both consecutive pairs and the 24-to-48 pair. The pin change must also be
no larger than the velocity discretization error. The algebraic identities
allow only `512eps()*max(sum(abs(wall flux)),1)`; solve residual divided by
`h^2` must be at most `1e-8`.

| Quantity, 24 to 48 | Taylor-Couette order | Closed MMS order |
|---|---:|---:|
| Velocity maximum error | 2.042049 | 2.561271 |
| Absolute global balance | 0.864577 | 1.238239 |
| Absolute pin continuity divided by h^2 | -1.135423 | -0.761761 |
| Pin-induced velocity change | 0.301706 | 0.675368 |

Both cases fail. The normalized pin defect grows from `0.00393153` to
`0.00863689` (Couette) and from `0.0441014` to `0.0747766` (MMS). Changing the
pin from the first to the last fluid cell changes velocity by less than its
discretization error, but not at the required order. The largest normalized
solve residual across both pin positions is `9.03e-10`, so solver convergence
does not explain the omitted-row defect. All values, both pins, injected and
solved states are in `benchmarks/fv_c5/closed_mass.csv`.

Fifteen scientific order checks remain `@test_broken`; the successful
MMS 40-to-48 pair cannot override the failures on the other pairs.
The step-5 stop rule applies: no `FVShapeMap`, no adjoint/finite-difference/
analytic comparison of `dT/dR1`, no M3, and no convergence campaign is attempted.
The final local selection does not establish closed-domain conservation.

## Prepared GPU check

The opt-in test uses a small immersed cylinder in an inlet/outlet channel,
Float64, both force objectives, viscosity sensitivity, relative CPU/GPU agreement
at `1e-12`, and zero device allocation in warmed Newton and adjoint calls.
The benchmark then times the same immersed Stokes problem at 256 and 512.
It has no CPU timing mode. Existing kernel inference guards continue to build
with `KernelAbstractions.CPU()`; these checks cannot establish GPU correctness.
The PBS file requests one A100 on `gpu0n009`, eight CPU cores, 32 GB and two
hours. Neither setup nor GPU execution has been run in this session.
