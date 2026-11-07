# C5 immersed walls: M3 mass compatibility

## Current C5-M3 decision (2026-11-05)

M3 is the default for immersed linear/quadratic closures. M1 remains an
explicit comparison mode. One compatibility unknown is added per connected
solid body / fluid region pair. Both continuity and convective wall momentum
use its corrected flux, and the discrete adjoint includes it. Exact physical
flux targets are supplied with `wall_flux`; fallback circle quadrature reports
an error estimate. C1 and domain-wall arithmetic remain unchanged.
The [M3 note](../../docs/agent/incns-fv-c5-m3.md) gives the equations,
assembly contract, rank evidence and validation scope.

The first four validation gates pass. Dense differentiation agrees with the
assembled Jacobian to `8.89e-16`; small pinned systems have full rank.
The circle/annulus tests use 32/33 colors, compared with 33/31 for M1.
Both force components and torque pass adjoint/finite-difference checks for
R, cx and angular velocity, with convection off/on: maximum absolute error
`1.46e-9`, maximum relative error `6.63e-8`.

| Closed case / mode, N=24 to 48 | Velocity maximum-error order | Global mass balance | Pin continuity / h^2 |
|---|---:|---|---|
| Taylor-Couette M1 | 2.042049 | order 0.864577 | grows, order -1.135423 |
| Taylor-Couette M3 | 2.037090 | at most 6.18e-18 | at most 1.62e-15 |
| Non-rigid MMS M1 | 2.561271 | order 1.238239 | grows, order -0.761761 |
| Non-rigid MMS M3 | 2.564349 | at most 2.94e-15 | at most 2.34e-13 |

The M3 roundoff bounds cover all three grids (24,40,48) and both pin
positions. Every consecutive velocity order exceeds 1.8. Pin movement changes
velocity by at most `8.89e-16` and pressure, after removing its constant, by
at most `1.95e-13`. The five `closed_mass_m3_*.csv` files preserve the new
algebra, adjoint and closed-flow measurements. `closed_mass.csv` remains the
M1 reference; its known failures still describe that explicit mode.

The open channel uses a fixed sampled inlet flow 0.1, a pressure outlet, and
pressure sections x=0.25/1.75. The cylinder has center `(0.513,0.487)`, radius
0.16 in a 2-by-1 channel. Both modes use exactly the same four grids.

| N | Inlet minus outlet M1 | Inlet minus outlet M3 | Delta p M1 | Delta p M3 |
|---|---:|---:|---:|---:|
| 24 | -1.185030e-4 | 1.53e-16 | 8.180825 | 8.175126 |
| 32 | 1.277964e-4 | -1.39e-17 | 8.222481 | 8.228461 |
| 40 | 3.455640e-5 | 2.91e-16 | 8.216910 | 8.218450 |
| 48 | 1.858463e-5 | -1.39e-16 | 8.230160 | 8.231022 |

M1 flux-defect magnitude orders are -0.262/5.861/3.402, with a sign change;
the endpoint estimate 2.673 is not an established asymptotic order. M3 is
roundoff limited. Delta p is nonmonotonic in both modes: neither has a
supported Richardson order on these grids. `channel_mass.jl` regenerates
`channel_mass.csv`, `channel_orders.csv`, and `channel_flux_orders.csv`.

The minimal `FVShapeMap` freezes the analytic geometry topology and rejects
cell changes or resolution-contract violations. For Taylor-Couette,
the analytic inner torque derivative is `dT/dR1=-5.102007555954`.

| N | Residual-flux dT/dR1 | Absolute analytic error | Surface dT/dR1 | Absolute analytic error |
|---|---:|---:|---:|---:|
| 24 | -5.233058158 | 0.131050602 | 8.987657590 | 14.089665146 |
| 40 | -5.085099842 | 0.016907714 | -4.903846863 | 0.198160693 |
| 48 | -5.088144411 | 0.013863145 | -5.006409303 | 0.095598253 |
| 64 | -5.098695668 | 0.003311888 | -4.563834550 | 0.538173006 |

Both objectives pass the discrete adjoint/FD gate for R1 and cx at the fixed
steps `1e-5,2e-6,4e-7`. Maximum best-step absolute errors are `1.83e-8`
(residual flux) and `5.67e-6` (surface), within the frozen mixed absolute/
relative bound. Residual-flux analytic-error orders are 4.009/1.089/4.977,
meeting the planned 0.8 criterion. Surface error rises from N=48 to N=64:
**surface is a verified discrete derivative without continuous validation**.
`shape_derivative.jl` reproduces the four `shape_*.csv` evidence files.
No grid, finite-difference step, or frozen threshold was adjusted to pass.

The GPU fixture now uses M3. `c5_gpu.jl` compares M1/M3 at 256 and 512,
including matrix size, colors, first factorization, numeric refactorization,
and retained device memory. CPU inference guards and host-only fixture,
geometry, and syntax checks pass. CUDA parity, zero device allocation,
memory and timings remain unverified: no GPU job is launched here.
Final CPU receipt: **8,157 passing assertions**, 15 preserved M1 scientific
failures marked `@test_broken`, and one GPU opt-in skip. All FV CPU entry
points pass on the final source, including geometry, compiler guards and
every new M3/shape/channel test. The sequence takes 189.90 seconds; the
largest individual testset takes 26.63 seconds. The separate prepared-GPU
host preflight passes 23 assertions (largest testset 7.0 seconds).
Receipts are `tmp/c5_m3_validation/verified_cpu.log` and
`verified_cpu_summary.json`, plus `tmp/c5_m3_gpu_preflight.log`.

The final audit also caught and fixed loss of the quadrature error estimate
when rebuilding from an existing immersed boundary. The new 10-assertion
test preserves Q together with its estimate; an explicit replacement target
supersedes that estimate. No inconsistency in the reviewed M3 equations was
found. The unresolved scientific points are delta-p convergence and the
continuous surface-torque derivative, plus all actual CUDA measurements.

`git diff --check`, edit-scope and forbidden-identifier checks pass. The
current source snapshot is `tmp/src_c5.tar.gz`, including the M3 note and all
new tests/scripts, with neutral archive ownership and byte-verified contents;
`tmp/src_c5.sha256` records its checksum. No commit, push or GPU job was made.

## Historical C5-S45b: Q/M1/Lin local selection and closed-mass stop

Date: 2026-11-05. CPU, macOS arm64, Julia 1.12.5, Float64.
Final local selection; integrated objectives; closed-domain M1 stop.
The nominal-face tangential stress now uses `gP+(gP-gO)/2`, where O is the
opposite fluid cell. The residual and its force derivative share this formula.
C1 preserves its arithmetic, and the residual Jacobian retains its existing
radius-two (5 by 5) dependencies and 33–35 colours in the immersed tests.
The physical `:surface` reconstruction is unchanged.

**Q/M1/Lin is selected by the final local criterion.** Affine exactness and
velocity order at least 1.8 pass. Pressure values, pressure traction, viscous
traction and total traction each meet the matching aligned C1 order minus
0.2 on every refinement pair, using the minimum over all six angles.
The earlier total-traction and two-pair viscous-order criteria still fail;
their evidence is preserved below. This selection does not establish second
order pressure or wall shear. Improving their orders across C1–C5 is a
separate later track. Q also reproduces the quadratic manufactured problem
to roundoff, including solved traction; that identity alone does not establish
a convergence order. Closed-domain, shape-derivative and GPU results are not
provided by this local experiment. The separate closed-domain checks below
fail the mass gate, so the selected local closure is not cleared for shape
derivative validation or larger campaigns.

## C5-S45b: objectives, closed mass, and prepared GPU files

Both integrated force/torque contracts are available as `kind=:residual_flux`
and `kind=:surface`. The former shares the residual's complete numerical
wall momentum flux and uses nominal face lever arms. The latter uses physical
cuts and the specified arc quadrature. The detailed balance identities,
geometry metadata and differentiability contract are in the
[C5 note](../../docs/agent/incns-fv-c5-s45b.md).

The step-4 comparison uses a 12 by 12 Stokes case, h=1, circle
`(cx,cy,R)=(6.13,5.87,2.21)`, angular velocity 0.2, viscosity `0.3+0.01x`,
and a north wall moving at `(0.1,0)`. Radius, horizontal centre and angular
velocity are differentiated without changing cell classification. Centred
finite differences use steps 1e-4 and 2e-5; the table reports the largest
best-step error over those three parameters. The torque centre is fixed.

| Objective | Maximum absolute adjoint/FD error | Maximum relative error |
|---|---:|---:|
| Residual-flux force | 5.081e-11 | 3.686e-9 |
| Residual-flux torque | 3.329e-10 | 6.553e-11 |
| Surface force | 9.682e-12 | 1.603e-9 |
| Surface torque | 2.139e-10 | 8.651e-11 |

The frozen bound is `2e-7 + 1e-6*abs(adjoint derivative)`; each FD state also
has a residual maximum below 2e-11. Individual derivatives and errors are
in [objective_derivatives.csv](objective_derivatives.csv), transcribed from
the `incns_fv_immersed_objectives.jl` test log. These are discrete derivatives
of an open-domain unit case, not the stopped closed-domain dT/dR1 study.
The momentum and moment balance identities pass, including convection.
Constant stress gives zero residual-flux force and nominal-face torque to
roundoff. With the mandated midpoint arc quadrature, the corresponding
surface sums are not generally zero; no weights are corrected to hide that.

**Step 5 stops at M1 mass compatibility.** On the off-centre annulus at
N=24,40,48, both Taylor-Couette and the asymmetric, non-rigid closed MMS
converge algebraically, but their global mass balance and omitted continuity
row do not converge as fast as velocity error. There is no mass-source
correction, M2 rerun or M3 implementation.

| Case, N=24 to 48 | Velocity error order | Global balance order | Pin continuity / h^2 order | Pin-induced velocity order |
|---|---:|---:|---:|---:|
| Taylor-Couette | 2.042049 | 0.864577 | -1.135423 | 0.301706 |
| Closed MMS | 2.561271 | 1.238239 | -0.761761 | 0.675368 |
| Required relative to velocity | reference | at least reference | at least reference | at least reference |

The normalized omitted continuity defect increases from 0.00393153 to
0.00863689 for Couette, and from 0.0441014 to 0.0747766 for the MMS. The
largest solve residual divided by h^2 over both pin positions is 9.03e-10.
Moving the pin changes velocity by less than its discretization error on each
grid, but its decay fails the required order. The independent analytic wall
sum matches the sum of restored continuity rows to roundoff. Thus this is a
conservation limitation, not a failed linear solve. All injected/solved
values and both pin positions are in [closed_mass.csv](closed_mass.csv).
The known order failures are retained as 15 `@test_broken` checks.

The existing geometry contract rejects N=32 because an opposite cell is not
fluid; N=24,40,48 are valid without dropping any link. This shows that radius
and wall-separation bounds alone do not guarantee an admissible cut stencil.
The stop rule prevents `FVShapeMap` and the requested adjoint/FD/analytic
`dT/dR1` comparison: no value or continuous derivative order is claimed.

The GPU test and three launch files are prepared, not executed. The small
case is a cylinder in an inlet/outlet channel, with both force objectives and
viscosity derivatives, relative CPU/GPU agreement 1e-12 (absolute for the
near-zero residual), and zero device allocation in hot Newton/adjoint calls.
The same GPU-only script then times 256^2 and 512^2. All GPU claims remain
unverified. A failed small check prevents the larger timings.

### Aqua layout and commands (prepared only)

The archive is a snapshot of the current files, including uncommitted changes;
`git archive HEAD` would omit the present work. From this worktree root:

```bash
tar -czf tmp/src_c5.tar.gz Project.toml src ext test/analytical test/runtests.jl benchmarks/fv_c5 docs/agent/incns-fv-c5-s45b.md
```

After transferring `src_c5.tar.gz` into `~/kraken_c5/`, the exact Aqua layout
and commands are:

```text
~/kraken_c5/
  src_c5.tar.gz
  src_c5/                 # archive extracted here; Project.toml, src/, ext/, ...
  env/                    # setup creates the Julia environment
  depot/                  # setup creates the private package cache
  logs/                   # job receipts and sampled device memory
```

```bash
cd ~/kraken_c5
mkdir -p src_c5
tar -xzf src_c5.tar.gz -C src_c5
bash src_c5/benchmarks/fv_c5/setup_c5.sh
qsub src_c5/benchmarks/fv_c5/c5_gpu.pbs
```

The PBS request is `select=1:ncpus=8:ngpus=1:mem=32GB:gpu_id=A100:host=gpu0n009`,
with `walltime=02:00:00`. Setup installs on the login node; computation uses
the private environment offline with `JULIA_CUDA_USE_COMPAT=false`.
Nothing is submitted by this session, and these prepared commands do not
lift the closed-domain mass stop.

### Final local verification receipt

All FV CPU entry points passed together in one Julia process, including
the geometry checks included by `immersed` and compiler guards included by
`adjoint`. Total: 7,263 passing assertions, 15 documented mass-order failures,
and one GPU opt-in skip. The largest individual testset took 26.8 seconds;
no local long simulation was launched. The GPU fixture separately passed
five CPU unit assertions in 5.7 seconds; this does not validate CUDA.
The 256/512 geometry constructors also pass four assertions in at most
0.6 seconds, without creating a Newton solver or running either flow.

| Entry point | Passing assertions | Known scientific failures |
|---|---:|---:|
| steady_residual | 53 | 0 |
| newton | 121 | 0 |
| gnf | 731 | 0 |
| adjoint, including compiler guards | 953 | 0 |
| immersed, including geometry | 1005 | 0 |
| immersed_mms | 3915 | 0 |
| aligned_c1 | 226 | 0 |
| immersed_objectives | 149 | 0 |
| immersed_mass | 110 | 15 |

Session receipts: `tmp/c5_s45b_cpu.log`, `tmp/c5_s45b_gpu_preflight.log`,
and `tmp/c5_s45b_gpu_geometry.log`.
The final selection CSVs were regenerated from that final run; all original
MMS metrics and the previous viscous-gate CSV reproduce byte-for-byte.
Shell syntax, Julia parsing of prepared GPU scripts, `git diff --check`,
allowed edit scope and the forbidden-identifier scan pass. GPU compilation,
parity, allocation and timings remain unmeasured. No commit or push was made.

Files changed or created in this session:

- Source: `src/KrakeNN.jl`; `src/methods/inc_ns/fv_adjoint.jl`,
  `fv_steady_residual.jl`, `fv_immersed.jl`, `fv_immersed_surface.jl`, and new
  `fv_immersed_objectives.jl`.
- Tests: `test/runtests.jl`; `test/analytical/incns_fv_immersed.jl`,
  `incns_fv_immersed_mms.jl`, `incns_fv_immersed_mms_gate.jl`,
  `incns_fv_immersed_mms_evidence.jl`; new `incns_fv_immersed_objectives.jl`,
  `incns_fv_immersed_mass.jl`, `incns_fv_immersed_mass_helpers.jl`,
  `incns_fv_immersed_gpu.jl`, `incns_fv_immersed_gpu_helpers.jl`.
- Evidence here: updated `local_mms_selection.csv`; new
  `local_mms_previous_selection.csv`, `local_mms_final_gate.csv`,
  `objective_derivatives.csv`, `closed_mass.csv`.
- GPU preparation here: new `c5_gpu.jl`, `c5_gpu.pbs`, `setup_c5.sh`.
- Documentation: this README and new `docs/agent/incns-fv-c5-s45b.md`.

## Selection evidence

L/Q denotes the residual's linear/quadratic ghost velocity; M1 denotes its
nominal-face mass velocity; Z/Lin denotes zero-gradient/linear nominal-face
pressure. M2 is discarded and is not rerun. All previous eight-choice data,
including its negative evidence, are preserved byte-for-byte in
[`historical_s3/`](historical_s3/README.md).

Each order below is the minimum over the same six angles for that individual
refinement pair. Surface error is the RMS Euclidean traction error over every
measured cut link, `sqrt(sum(norm(error)^2)/number_of_links)`. It is neither an
arc-weighted norm nor an integrated resultant. No small-q, corner or angle
filter is applied. Every per-angle error and order is in the CSVs.

| Ghost / pressure (M1) | Affine exact | Quadratic traction 16→32 / 32→64 / 64→128 | Cubic traction 16→32 / 32→64 / 64→128 | Original criterion selected |
|---|---|---|---|---|
| L / Z | no | 0.557 / 0.983 / 0.045 | 0.541 / 0.974 / 0.212 | no |
| L / Lin | yes | 0.128 / −1.892 / 0.210 | 0.095 / −1.462 / 0.194 | no |
| Q / Z | no | 0.973 / 0.897 / 0.738 | 0.951 / 0.894 / 0.706 | no |
| Q / Lin | yes | exact polynomial, roundoff limited | 0.953 / 0.909 / 1.003 | no |

The final criterion applies to all four choices on the solved cubic-velocity,
quadratic-pressure probe. It requires affine exactness, velocity order at least
1.8 on every pair, and each pressure/traction order at least its matching A0
order minus 0.2 on every pair. Each order is the minimum over all six angles;
no angle or pair is excluded. These are comparison thresholds authorized
after measuring A0, not evidence that the original targets were reached.

| Q/M1/Lin metric | Minimum orders 16→32 / 32→64 / 64→128 | Final thresholds, same pairs | All pairs pass |
|---|---|---|---|
| Velocity maximum | 1.873960 / 1.946512 / 1.963361 | 1.8 / 1.8 / 1.8 | yes |
| Pressure maximum | 1.003561 / 1.013568 / 1.013321 | 0.714875 / 0.752254 / 0.774085 | yes |
| Pressure traction RMS | 1.058799 / 0.959200 / 0.947907 | 0.761273 / 0.807771 / 0.821212 | yes |
| Viscous traction RMS | 1.395302 / 1.373290 / 0.972205 | 0.672904 / 0.724383 / 0.756599 | yes |
| Total traction RMS | 0.953085 / 0.908707 / 1.003468 | 0.887991 / 0.880542 / 0.862635 | yes |

All three other choices fail the final criterion. L/Z and Q/Z are not affine
exact. Their velocity minima are 1.073110 / 1.581303 / 1.967771 and
1.478527 / 1.300440 / 1.214770 respectively. L/Lin is affine exact, but its
velocity minima are 1.603653 / 1.839077 / 1.893402; its pressure minima are
0.040166 / 0.367266 / 0.122632, also below every final pressure threshold.
[`local_mms_final_gate.csv`](local_mms_final_gate.csv) records all 60
metric/pair decisions for the four choices, including their other failures.
[`local_mms_selection.csv`](local_mms_selection.csv) records the final decision
and separate original/previous decisions. The tests assert selection for
Q/M1/Lin and explicit nonselection for each other choice.

### Historical criteria and preserved negative evidence

The original total-traction criterion required order at least 1.8 on every
pair; all four choices failed, as shown above. The subsequent revised
criterion for Q/M1/Lin required affine exactness; velocity order at least 1.8
on every pair; viscous traction RMS order at least 1.8 on at least two pairs;
and pressure-value maximum plus total traction RMS orders no lower than the
matching A0 orders minus 0.2. Its outcomes are unchanged in
[`local_mms_previous_selection.csv`](local_mms_previous_selection.csv).

Affine exactness passes. Velocity minima are 1.873960 / 1.946512 / 1.963361;
pressure minima are 1.003561 / 1.013568 / 1.013321, and total-traction minima
are 0.953085 / 0.908707 / 1.003468. Those conditions pass. The viscous
condition fails with minima 1.395302 / 1.373290 / 0.972205, hence 0/3 pairs.
The complete previous boolean decisions and reference orders are preserved in
[`local_mms_revised_gate.csv`](local_mms_revised_gate.csv).

| Angle | Viscous order 16→32 | 32→64 | 64→128 | Smallest error / roundoff budget over four grids |
|---|---:|---:|---:|---:|
| 0° | 1.395302 | 2.794812 | 0.972205 | 283389.57 |
| 15° | 1.708460 | 2.057570 | 1.342253 | 669.79 |
| 30° | 1.807227 | 1.820195 | 1.806871 | 414.73 |
| 45° | 2.697306 | 1.373290 | 2.497257 | 55468.07 |
| 60° | 1.856702 | 1.929302 | 1.889548 | 447.37 |
| 75° | 1.728183 | 2.049175 | 1.447912 | 697.48 |

All 24 viscous errors and their unchanged `_c5m_metric_budget` roundoff
budgets are retained in the 18 pair rows of
[`local_mms_viscous_gate.csv`](local_mms_viscous_gate.csv), including repeated
endpoints so each pair is independently checkable. Errors span 1.83e-5 to
1.73e-3; even the smallest error/budget ratio is 414.73. The orders vary
with grid and angle and are not roundoff-limited. Changing cut positions
may contribute, but this experiment does not isolate that cause.

The A0 baseline is the independently reproducible
`aligned_c1_orders.csv` cubic-velocity/quadratic-pressure solved probe. The
gate loader requires both pressure probes on all three pairs exactly once,
plus all matching quadratic-pressure traction components, finite positive
errors, and stored orders equal to `log2(error_coarse/error_fine)`. Every A0
pressure order from both probes must be at most 1.3; any disagreement stops
the reference-based selection before it can run. The actual comparison uses only
the matching cubic-velocity/quadratic-pressure reference.
The normal MMS suite reads that recorded baseline without repeating A0
solves; the registered A0 test remains separate. Tests and the selection
CSV writer use the same final gate helper. The six historical bookkeeping
checks remain, and 21 final bookkeeping checks require every metric and pair,
reject invalid or roundoff-limited orders, and retain the affine condition.

For Q/M1/Lin, the largest quadratic injected cut momentum defect is 8.03e-9
in `R/h^2`; it is within the pre-existing roundoff budget, which scales with
`1/q` and `h^-2`. The largest quadratic traction RMS over injected and solved
states is 5.52e-12. At 30°, N=16/32, the old cut defect 0.271250/0.271875
becomes 1.02e-11/1.82e-11. Affine residual maxima are 2.34e-9 (L/Lin) and
7.50e-9 (Q/Lin) across the four grids; Z retains the affine defect of 0.4.
All solved local equations satisfy `max(abs(R))/h^2 <= 9.84e-9`.

The remaining Q/M1/Lin cubic probe has the following minimum orders over the
six angles. The gradient norm is the maximum absolute component over cut
links, so its minima can be sensitive to grid-to-grid cut positions.

| Metric | 16→32 | 32→64 | 64→128 |
|---|---:|---:|---:|
| Solved velocity values | 1.874 | 1.947 | 1.963 |
| Solved pressure values | 1.004 | 1.014 | 1.013 |
| Solved cell gradients | −0.0001 | −0.122 | −0.105 |
| Solved surface gradients | 0.152 | 0.527 | −0.432 |
| Solved surface traction RMS | 0.953 | 0.909 | 1.003 |
| Injected cut momentum truncation | 0.382 | 0.976 | 0.737 |
| Injected cut mass truncation | 1.623 | 1.727 | −1.529 |

The order CSV also separates pressure and viscous traction, maximum and RMS
errors, ghost and nominal-face values, and surface pressure. A pair touching
the existing roundoff budget is labelled `roundoff_limited`, never fitted as
an order. Exact-state velocity/pressure values are zero by construction and
are likewise not used to infer convergence. The cubic probe retains nonzero
errors; the independent injected pressure error is
`-a_e*q*(q+1)*h^2`, with `a_e=0.2` on x links and `0.25` on y links.

## Manufactured fields and measurement contract

The physical square is fixed at [0,1]^2. Grids N=16,32,64,128 and normal angles
0°,15°,30°,45°,60°,75° were fixed before measuring. The solid halfplane is
`n dot ((x,y)-(0.5,0.5)) >= 0.073`. Every cut whose fluid cell centre is in
the physical square is measured: no corner, small-q or near-tangent link is
removed. `local_mms_links.csv` enumerates all 2,042 measured links across grids
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
`roundoff_limited` in the order CSV, never fitted as convergence orders.

## Executed path and remaining cause

A five-evaluation trace before the fix confirms the production route
`fv_residual → residual kernel → fv_cell_residual → _fv_wall_stress`
(600 wall-stress calls). At N=8, 30°, the nominal quadratic stress error was
0.0340625 and cut momentum truncation was 0.2725. Extrapolating the tangential
gradient to the nominal face removes this missing half-cell displacement;
the corrected cut defect is 1.62e-12. Regression tests cover all four link
directions, Float32/Float64, Newtonian and generalized-Newtonian laws.

The force's local derivative must read O's tangential neighbours too. Its
local support grows from five to nine cells (15 to 27 scalar derivatives)
for L/Q; C1 keeps five cells. Dense derivative comparisons pass. The new
reads were already present in the residual Jacobian pattern, so that pattern
and its colour count do not grow. Host support construction continues using
host tags when given device-backed problem arrays. CUDA execution remains
unverified by this local mission.

A bounded attribution experiment keeps the same exact halo and Q/M1/Lin
operator and independently changes velocity and pressure polynomial degrees.
At 30°, quadratic velocity with quadratic pressure gives solved traction
orders 2.074 / 1.995 / 1.996. Cubic velocity with **linear** pressure still
gives 1.090 / 1.047 / 1.027. The first-order error therefore does not require
quadratic pressure extrapolation at the wall.

The linear solve is also split into corrections driven by momentum rows
and by continuity rows. These partial corrections are diagnostics, not full
solutions. For cubic velocity/linear pressure at 30°, the momentum-only
traction error falls 7.57e-4 → 1.92e-4 → 4.72e-5 → 1.17e-5; the mass-only
error falls 3.16e-3 → 1.61e-3 → 8.12e-4 → 4.06e-4 and dominates at N=128.
The continuity defect includes the exact strict-interior term
`h^2*(u_xxx+v_yyy)/6 = 0.04h^2`, as well as cut and near-wall contributions.
Their spatial effects were not separated. This local arrangement with exact
velocity and pressure halo therefore retains a first-order pressure response
to the continuity defect; it does not isolate the role of the halo, the wall
mass flux and pressure coupling from one another.

At 30° on the standard cubic probe, solved pressure traction RMS decreases
4.35e-3 → 4.24e-4 while viscous traction RMS decreases 1.23e-3 → 2.85e-5.
Under the original criterion, total-traction error was controlled by pressure.
The previous revised criterion instead stopped at the separate viscous-order
condition. The final criterion compares all components with aligned C1;
meeting it does not remove either earlier failure. No further correction or
image-point fallback is introduced by this selection change.

## Open pressure and wall-shear order track

Improving pressure and wall-shear convergence across C1–C5 remains separate
later work. The present local pressure response is approximately first order,
and the minimum viscous traction orders are 1.395302 / 1.373290 / 0.972205.
The final C1 comparison therefore permits local closure selection without
claiming second order for either quantity. The relative roles of continuity
truncation, the exact halo and wall coupling remain unresolved here.

## Reproduction and files

From the repository root, in the prepared offline environment:

```bash
env JULIA_DEPOT_PATH="$PWD/tmp/c5_depot:$HOME/.julia" JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 KRAKEN_C5_MMS_OUTPUT=benchmarks/fv_c5 "$HOME/.julia/juliaup/julia-1.12.5+0.aarch64.apple.darwin14/Julia-1.12.app/Contents/Resources/julia/bin/julia" --startup-file=no --project=tmp/c5_env test/analytical/incns_fv_immersed_mms.jl
env JULIA_DEPOT_PATH="$PWD/tmp/c5_depot:$HOME/.julia" JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 "$HOME/.julia/juliaup/julia-1.12.5+0.aarch64.apple.darwin14/Julia-1.12.app/Contents/Resources/julia/bin/julia" --startup-file=no --project=tmp/c5_env benchmarks/fv_c5/local_mms_attribution.jl
git diff --check
```

The registered MMS test writes evidence only when `KRAKEN_C5_MMS_OUTPUT` is
set. The attribution script writes its two CSVs to its own directory by
default and accepts the same output override. Both use continuous analytic
sources; source formulas for independent pressure degrees are checked by
automatic differentiation in the registered suite.

- `local_mms_metrics.csv`: 576 rows, four M1 choices × six angles × four grids
  × three fields × injected/solved states.
- `local_mms_orders.csv`: 1,536 rows, 16 metrics, all three refinement pairs.
- `local_mms_selection.csv`: four final choices plus separate original/previous decisions.
- `local_mms_final_gate.csv`: 60 final metric/pair decisions and matching A0 thresholds.
- `local_mms_previous_selection.csv`: unchanged previous four-choice selection evidence.
- `local_mms_revised_gate.csv`: preserved previous Q gate decisions and A0 reference orders.
- `local_mms_viscous_gate.csv`: preserved six-angle errors, orders and roundoff budgets.
- `local_mms_links.csv`: every measured geometric cut, without filtering.
- `local_mms_timings.csv`: one time per configuration, angle and grid.
- `local_mms_attribution.csv`: two angles, four grids, four velocity/pressure
  polynomial combinations and four correction states.
- `local_mms_attribution_timings.csv`: eight bounded experiment times.
- `historical_s3/`: unchanged pre-fix README and five evidence files, M2 included.

Historical pre-A0 validation receipts are in `tmp/c5_s3b/`: production trace in
`trace_wall_stress.log`, targeted source acceptance in `immersed_acceptance.log`,
full FV CPU verification in `all_fv_cpu.log`, final GNF grouping in
`gnf_final.log`, and attribution in `attribution.log`.

That pre-A0 complete FV CPU set passed 6,747 assertions: 1,858 existing residual,
Newton, GNF, adjoint and compiler checks; 1,005 immersed checks; and 3,884 MMS
checks. Four scientific selection gates remain known broken. The separate
attribution script passes 32 solve-residual checks in 5.56 s. `git diff --check`
passes, and the six archived files are byte-identical to the pre-fix revision.

The pre-A0 MMS file took 15.21 s; its configuration testsets took at most
1.1 s, and the slowest N=128 triplet takes 0.687 s. The historical GNF aggregate
is now reported as independent testsets inside a lexical `let` scope, without
removing any check. Its final file still takes 57.46 s; its longest individual
testset takes 27.8 s. This changes reporting granularity, not computational
cost. Every individual testset is below 30 s in the final receipts. No long
CPU flow or Aqua job was run for this mission.

## A0 aligned C1 reference

The aligned reference uses the original C1 domain-wall path with
`prob.immersed === nothing`. Its east boundary is the fixed physical plane
`x=0.5`, exactly a cell face on every grid. With `H=5`, the full grid has
`Nx=H+N/2`, `Ny=N+2H`, and both coordinates are shifted by `H*h`. Only cells
in `[0,0.5] x [0,1]` are unknown; five layers of exact velocity and pressure
remain fixed on the west, north and south. The same analytic cubic velocity,
continuous Stokes source, variable viscosity, sparse restricted solve and
norms above are reused. Both quadratic and linear pressure are measured.

The surface observer needs wall geometry, so a separate same-shape raw
`q=0.5` view supplies the existing `_fv_surface_reconstruction` helper with
the wall position and velocity. This view is never passed to residual or
Jacobian assembly. The helper reads the wall cell and its opposite fluid
cell, without reading outside the east domain. All N wall links are measured.
Here `cut` metric columns mean aligned wall cells, `near` means the next
column, and `interior` means cells at least two columns from the wall.

Runtime checks confirm the boundary midpoint and its nonzero analytic
velocity. Five full production residual calls make 880 reads from the east
boundary velocity array on the tiny N=8 fixture, while returning the same
values as the uninstrumented calls. The full kernel equals direct production
cell calls. An extended grid with one solid column, explicit `:c1`, `q=0.5`
and identical wall data gives exactly equal restricted residuals at exact
and perturbed states, and exactly equal restricted Jacobians (96 unknowns).
The test also confirms that a tagged internal face without immersed data
would return zero velocity; it is therefore not used as the actual solve.

Measured solved orders, with every consecutive refinement pair retained:

| Pressure degree / metric | 16→32 | 32→64 | 64→128 |
|---|---:|---:|---:|
| Quadratic / pressure maximum | 0.914875 | 0.952254 | 0.974085 |
| Quadratic / total traction RMS | 1.087991 | 1.080542 | 1.062635 |
| Quadratic / pressure traction RMS | 0.961273 | 1.007771 | 1.021212 |
| Quadratic / viscous traction RMS | 0.872904 | 0.924383 | 0.956599 |
| Linear / pressure maximum | 0.911772 | 0.951603 | 0.974317 |
| Linear / total traction RMS | 0.880917 | 0.912939 | 0.939267 |
| Linear / pressure traction RMS | 1.026240 | 1.047937 | 1.047343 |
| Linear / viscous traction RMS | 0.903193 | 0.951790 | 0.975354 |

Both probes have pressure orders below 1.3. For the original total-traction
order gate of 1.8, the user-requested interpretation is: "not reachable by a
wall closure alone: the base scheme’s pressure is first order". This decision
is limited to the present local probes; it is not a proved global accuracy
limit or a sole-cause attribution. The measurement does not
separate C1 wall arithmetic, pressure coupling and the exact halo. It
motivated the previous pressure/total-traction comparison above, whose
separate viscous condition failed. The final comparison now includes each
traction component and selects Q/M1/Lin; the prior failure is retained. No
closed-domain, force, shape-derivative or GPU validation is claimed;
later-stage validation follows from this reference alone.

Reproduce only the new evidence in the prepared offline environment:

```bash
env JULIA_DEPOT_PATH="$PWD/tmp/c5_depot:$HOME/.julia" JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 KRAKEN_C5_A0_OUTPUT=benchmarks/fv_c5 "$HOME/.julia/juliaup/julia-1.12.5+0.aarch64.apple.darwin14/Julia-1.12.app/Contents/Resources/julia/bin/julia" --startup-file=no --project=tmp/c5_env test/analytical/incns_fv_aligned_c1.jl
```

`aligned_c1_metrics.csv` contains 16 rows (two probes, four grids, injected
and solved states); `aligned_c1_orders.csv` contains 192 metric/pair rows.
Exact injected values and roundoff-level pairs are labelled
`roundoff_limited`, without a numerical fit. All solved pressure and traction
orders are finite. `aligned_c1_route.csv` records the runtime equivalence
checks; `aligned_c1_timings.csv` records the four grid testsets. Output is
opt-in through `KRAKEN_C5_A0_OUTPUT`; normal suite runs write no artifacts.
The executable is also registered in the FV immersed testset. All previous
evidence files are preserved.

The previous validation passed 3,890 MMS assertions (including six bookkeeping
checks) and 226 A0 assertions, with four known broken selection checks.
Receipts are `tmp/a0_gate_validation.log` and `tmp/a0_gate_final_validation.log`.
At that stage, MMS metrics, orders and links, plus A0 metrics, orders and route
evidence, were byte-identical to the prior files. Existing timings were
preserved; only selection and the two gate-evidence CSVs were updated.

The final-selection validation passes 3,915 MMS assertions and 226 aligned
C1 assertions, with no known-broken selection checks. The longest individual
testsets take 12.5 s and 8.9 s respectively. Receipts are
`tmp/c5_final_selection/mms.log` and `tmp/c5_final_selection/aligned.log`.
All original measurements, previous revised decisions and per-angle viscous
evidence remain unchanged; only the final selection and final gate CSVs
describe the new criterion. The previous selection CSV is preserved verbatim.
