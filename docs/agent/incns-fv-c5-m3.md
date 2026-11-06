# C5 M3 mass compatibility

Date: 2026-11-05. Local CPU Float64 validation; GPU files are preparation only.
This note supersedes the M1 decision in `incns-fv-c5-s45b.md` when describing
the current default. The previous measurements remain historical evidence.

## Mathematical contract

Bodies are four-connected components of the complete solid mask. Regions
are connected fluid components, with pressure pins treated as fluid. Each
body/region pair with an immersed L/Q face owns one extra unknown, appended
after the cell velocity and pressure entries. The outer solid of an annulus
retains its pair even though that solid reaches the domain boundary.

The compatibility variable is a numerical flux correction, not a physical
wall-normal velocity. With outward flux positive,

```math
\widetilde F_w=F_w-\lambda_{b,r}h,\qquad
R_{b,r}=\sum_{w\in(b,r)}F_w-\lambda_{b,r}A_{b,r}-Q_{b,r},\qquad
A_{b,r}=\sum_{w\in(b,r)}h.
```

Continuity and convective wall momentum use the same corrected flux.
Physical wall data and ghost interpolation are unchanged. The target is the
physical wall flux facing that region, supplied analytically when known.
Rigid motion and the closed manufactured streamfunction have exactly zero
flux. A numerical target must carry its quadrature error estimate.

For each region, summing cell continuity cancels all internal faces. With
one pressure pin and zero volume mass source, the arbitrary-state identity
is

```math
C_{\rm pin}=\sum_b R_{b,r}+\sum_bQ_{b,r}+Q_{\rm ext}
              -\sum_{i\ne\rm pin}C_i.
```

A volume source adds `-h^2 sum(source_mass)` to the right-hand side. Units
here are flux, before division by cell area. Pair rows at zero enforce their
individual target fluxes; the identity then restores the omitted pin row at
the algebraic solve accuracy. Pressure-level fixing retains the existing
logic: an outlet fixes pressure, while a region with only prescribed velocity
still needs a pin.

## Assembly and differentiation

The pair row is affine in the state. For a Q ghost, each face contributes

```math
F_w=hs\left[\frac{1-q}{2(1+q)}u_O
       +\frac{3q-2}{2q}u_P+\frac{u_w}{q(1+q)}\right].
```

Here `u` denotes the axial velocity and `s` its outward sign. The L closure
instead has coefficients `0`, `(2q-1)/(2q)`, and `1/(2q)`. These coefficients
and the `-A` diagonal populate the appended rows directly. They do not enter
column coloring. The lambda columns do enter the local continuity and
momentum pattern, so disjoint pairs can share colors. Parameter derivatives
still differentiate the pair row, including cut fractions and flux targets.

The adjoint state includes every compatibility unknown. Residual-flux force
and torque include their direct dependence on lambda through convective
momentum. Surface traction has no direct dependence on lambda, but its
derivative includes the changed solution. C1, domain walls, and geometries
without immersed faces retain their original state dimension and arithmetic.

## Validation receipt

The first four gates pass in CPU Float64. Independent arbitrary-state flux
identities differ by at most `2.14e-14`; colored plus direct Jacobians differ
from dense ForwardDiff by at most `8.89e-16`. Tests cover L/Q, convection
off/on, a nonzero radial target, one solid body bordering two fluid regions,
and three appended compatibility variables. A target-only dual perturbation
has exactly the expected negative-unit derivative in the pair rows.

| Small problem | Unknowns M1 / M3 | Stored entries M1 / M3 | Colors M1 / M3 |
|---|---:|---:|---:|
| Circle, Stokes | 432 / 433 | 6764 / 6842 | 33 / 32 |
| Circle, convection | 432 / 433 | 6764 / 6842 | 33 / 32 |
| Annulus, Stokes | 1200 / 1202 | 5748 / 6032 | 31 / 33 |
| Annulus, convection | 1200 / 1202 | 5748 / 6032 | 31 / 33 |

The greedy coloring can use fewer colors after an extra column changes its
ordering. No pair row is included in the colored support. Removing the
independent solid identity equations leaves full-rank blocks of 388 and 299
unknowns respectively. The smallest singular values are at least `0.002429`
and `0.003415`; the numerical-rank threshold is `n*eps()*sigma_max`, at most
`2.07e-11`. Thus neither small augmented matrix has an unresolved null mode.

For both force components and torque, both objective kinds, radius, horizontal
center and angular velocity, with convection off/on, the largest adjoint/FD
absolute error is `1.46e-9` and relative error `6.63e-8`. Centered differences
use steps `1e-4` and `2e-5`, retaining both measurements in the CSV. The direct
lambda derivatives of residual-flux objectives are nonzero with convection
and match dense differentiation. All 360 adjoint assertions pass.

The closed Taylor-Couette and asymmetric non-rigid MMS use the previous
grids 24, 40 and 48 and exact zero flux targets. Orders below use endpoints
24 to 48; every consecutive L-infinity and RMS velocity order also exceeds
1.8 (minimum 1.847523 for MMS L-infinity on 40 to 48).

| Case and mode | Velocity maximum-error order | Global mass behavior | Pin continuity divided by h^2 |
|---|---:|---|---|
| Taylor-Couette M1 | 2.042049 | order 0.864577 | grows, order -1.135423 |
| Taylor-Couette M3 | 2.037090 | magnitude at most 6.18e-18 | magnitude at most 1.62e-15 |
| Closed MMS M1 | 2.561271 | order 1.238239 | grows, order -0.761761 |
| Closed MMS M3 | 2.564349 | magnitude at most 2.94e-15 | magnitude at most 2.34e-13 |

M3 pressure comparisons remove each field's arbitrary additive constant.
Moving the pressure pin changes velocity by at most `8.89e-16` and pressure
by at most `1.95e-13`. The global and omitted continuity residuals obey the
actual pair/retained-row solve bound plus reduction roundoff; no fitted order
is assigned to roundoff-level values. Velocity discretization errors remain
above `7.5e-4`, well above the chosen solve accuracy.

Evidence: `benchmarks/fv_c5/closed_mass_m3_identity.csv`,
`closed_mass_m3_jacobian.csv`, `closed_mass_m3_adjoint.csv`,
`closed_mass_m3_convergence.csv`, and `closed_mass_m3_orders.csv`.
The earlier M1 measurements remain in `closed_mass.csv`.

## Open channel at imposed flow

The channel is 2 by 1, with a cylinder centered at `(0.513,0.487)` of radius
0.16, viscosity 1 and Stokes flow. The sampled parabolic inlet is normalized
on every grid to the identical flow 0.1. The outlet fixes pressure; there is
no pin. Pressure drop is the difference between cross-section mean pressures
at the fixed physical positions x=0.25 and x=1.75, interpolated between
cell-center columns. No grid was removed or substituted.

| N | Inlet minus outlet M1 | Inlet minus outlet M3 | Pressure drop M1 | Pressure drop M3 |
|---|---:|---:|---:|---:|
| 24 | -1.185030e-4 | 1.53e-16 | 8.180825 | 8.175126 |
| 32 | 1.277964e-4 | -1.39e-17 | 8.222481 | 8.228461 |
| 40 | 3.455640e-5 | 2.91e-16 | 8.216910 | 8.218450 |
| 48 | 1.858463e-5 | -1.39e-16 | 8.230160 | 8.231022 |

The absolute M1 flux-defect orders are -0.262442, 5.861005, and 3.402001;
the endpoint magnitude order is 2.672742, with a sign change. These are not
evidence of a uniform asymptotic order. M3 is roundoff limited, so fitting a
flux-defect order would be meaningless. Pressure drop is nonmonotonic for
both modes on both consecutive triplets: the unequal-grid Richardson
estimator reports no order. The measurements establish flux compatibility,
not pressure-drop convergence. Evidence is in `channel_mass.csv`,
`channel_orders.csv`, and `channel_flux_orders.csv`; `channel_mass.jl`
reproduces the bounded cases. The channel suite has 98 passing assertions.

## Minimal shape map and Taylor-Couette derivative

`FVShapeMap(builder, base_problem, nparams)` creates the existing
`FVProblemMap` with a frozen reference topology. The checked analytic builder
re-evaluates radius/gap/intersection/transversality constraints. The wrapper
rejects changed cell tags, cut links, wall identities and body/region pairs.
Errors identify changed cells; there is no topology smoothing.

For unit inner angular speed and viscosity, with outer radius 0.34 and inner
radius 0.14, the fluid-on-inner-wall torque and derivative are

```math
T=-\frac{4\pi\nu\omega R_1^2R_2^2}{R_2^2-R_1^2},\qquad
\frac{dT}{dR_1}=-\frac{8\pi\nu\omega R_1R_2^4}{(R_2^2-R_1^2)^2}
=-5.102007555954.
```

The off-center geometry is the same as the closed test. All four fixed grids
24,40,48,64 and every radius/center perturbation at the fixed FD steps
`1e-5,2e-6,4e-7` satisfy the unchanged geometry contract.

| N | Residual-flux dT/dR1 | Absolute analytic error | Surface dT/dR1 | Absolute analytic error |
|---|---:|---:|---:|---:|
| 24 | -5.233058158 | 0.131050602 | 8.987657590 | 14.089665146 |
| 40 | -5.085099842 | 0.016907714 | -4.903846863 | 0.198160693 |
| 48 | -5.088144411 | 0.013863145 | -5.006409303 | 0.095598253 |
| 64 | -5.098695668 | 0.003311888 | -4.563834550 | 0.538173006 |

Both discrete objectives pass adjoint versus centered FD for radius and
horizontal center. Maximum best-fixed-step absolute errors are `1.83e-8`
for residual flux and `5.67e-6` for surface. The latter occurs at N=24 and
passes the pre-existing mixed bound `2e-7+1e-6*abs(derivative)`; it must not
be described as a pure relative-error bound.

Residual-flux consecutive analytic-error orders are 4.008831, 1.088934,
4.976728 (endpoint 3.749955), meeting the planned 0.8 criterion on these
four grids. This does not establish that the asymptotic order is 3.75.
Surface error rises on the final pair (order -6.006720): its discrete
derivative is verified, but continuous shape sensitivity is not validated.
The cause of this nonmonotonicity is not established by this work.
The shape suite has 190 passing assertions. `shape_derivative.jl` regenerates
`shape_geometry.csv`, `shape_fd_sweep.csv`, `shape_derivatives.csv`, and
`shape_orders.csv`; the classification is explicit in the last file.

## Prepared GPU validation

The small CUDA/cuDSS fixture selects M3 and an exact zero cylinder flux.
It checks CPU/GPU state, lambda, residual, both force objectives and viscosity
adjoints at the existing 1e-12 parity bounds, then zero device allocation in
warmed Newton and adjoint calls. The new numerical kernels and modified
norm/checksum kernels pass CPU inference guards using
`KernelAbstractions.CPU()`. Those guards do not establish CUDA correctness.

`c5_gpu.jl` compares M1/M3 at 256 and 512: unknowns, stored matrix entries,
colors, first symbolic+numeric factorization, numeric refactorization, and
retained device memory for the factorization and complete system. Memory is
measured after synchronization and pool reclamation; it is not a peak-memory
measurement. The existing PBS sampler separately records device usage.

Host-only preflight passes 23 assertions: the 16-square fixture solves and
both adjoints are finite; large checks construct geometry only; GPU files
parse. Largest preflight testset: 7.0 seconds. No large CPU solve is run.
GPU execution is forbidden for this session. CUDA parity, zero device
allocation, device memory, and GPU factorization timing remain unverified.

## Final verification and remaining limits

The complete FV CPU sequence passes on the final source: 8,157 assertions,
15 deliberately preserved M1 scientific failures (`@test_broken`), and one
GPU opt-in skip. Total wall time is 189.90 seconds; the largest individual
testset is 26.63 seconds. The timing watchdog never triggers and the test
files remain byte-identical throughout the run.

| Entry point or group | Passing assertions | Known failures or skips |
|---|---:|---:|
| steady_residual | 53 | 0 |
| newton | 121 | 0 |
| gnf | 731 | 0 |
| adjoint, including existing inference guards | 953 | 0 |
| immersed, including geometry | 1005 | 0 |
| immersed_mms | 3915 | 0 |
| aligned_c1 | 226 | 0 |
| immersed_objectives | 149 | 0 |
| immersed_mass, explicit historical M1 | 110 | 15 |
| M3 compatibility, GNF, quadrature and new compiler guards | 90 | 0 |
| M3 algebra and null-space tests | 72 | 0 |
| M3 force/torque adjoints | 360 | 0 |
| M3 closed convergence | 84 | 0 |
| Open channel | 98 | 0 |
| Shape derivative | 190 | 0 |
| GPU execution | 0 | 1 opt-in skip |

Receipts: `tmp/c5_m3_validation/verified_cpu.log`,
`tmp/c5_m3_validation/verified_cpu_summary.json`, and the separate
`tmp/c5_m3_gpu_preflight.log`. The final archive `tmp/src_c5.tar.gz` includes
current source, tests, benchmark evidence and both C5 notes. Ownership
metadata is neutralized; every regular member is verified byte-for-byte
against the worktree. Its SHA256 is in `tmp/src_c5.sha256`.

The last review found an implementation error in inherited quadrature
metadata: rebuilding from a lowered boundary kept Q but reset its uncertainty
to zero. The corrected constructor preserves both; a ten-assertion test
uses an angular harmonic with a nonzero 128/256 quadrature discrepancy.
Explicitly replacing Q still sets its numerical estimate to zero, as the
caller then supplies the physical target. No error in the reviewed M3
formulation was found by the implemented checks.

Remaining unverified claims are pressure-drop convergence in the open
channel, continuous sensitivity for the surface-torque objective, and CUDA
compilation/parity/allocation/memory/timing. A positive residual-flux shape
criterion on four grids does not validate arbitrary geometry changes or
topology crossings. No thresholds were relaxed. No commit, push, package
dependency operation, network access or GPU job was performed.

The required stop rules are a singular augmented Jacobian after fixing
pressure, or a closed-flow velocity order below 1.8. Later flow and shape
studies must not bypass either stop.
