# Steady FV energy equation, thermal coupling and its adjoint

This page documents the temperature part of the steady colocated finite-volume
(FV) solver: the Newton solver `fv_newton!` of the steady residual (see
[Incompressible steady Navier–Stokes](incompressible-navier-stokes.md) for the
flow part), extended with an energy equation, a viscosity that depends on the
temperature, and a discrete adjoint that differentiates through the
temperature. Everything here is 2D, steady, Float64 on CPU and on CUDA (through
cuDSS). There is no transient energy equation.

## What is solved

The unknowns of each fluid cell are `(u, v, p, T)`. The temperature obeys

```math
\nabla\cdot(\mathbf{u}\,T) - \alpha\,\nabla^2 T \;=\; 2\,\sigma\,\nu\,(S\!:\!S) + q ,
```

where `alpha` is the thermal diffusivity, `sigma` the heating coefficient
(`sigma = 0` switches viscous heating off), `nu` the kinematic viscosity,
`S:S` the squared strain-rate tensor (Green–Gauss cell gradient of the momentum
unknowns, the same one the viscosity law uses) and `q` an optional independent
source. The equation is built as the fourth row of the same cell-integrated
residual as momentum and continuity, so the same coloured-dual Jacobian and the
same Newton iteration solve all of it at once.

- Convection uses the continuity (Rhie–Chow) mass flux of each face, with
  second-order linear upwind reconstruction (`scheme = :linear_upwind`, falling
  back to first order where the upstream fluid cell is absent) or plain
  `:upwind`. Momentum convection has its own option on
  `fv_steady_problem`: `convection = :upwind` (first order, default) or
  `:central` (second order, the face value itself, no numerical diffusion;
  oscillations possible at cell Péclet well above 2, see
  `test/analytical/incns_fv_convection_central.jl`).
- Thermal boundary conditions follow the flow sides (see the next section);
  every connected fluid component must touch a prescribed temperature (a
  Dirichlet wall, an inlet or a Dirichlet solid), else the problem is rejected.
  Immersed walls (`fv_immersed_walls`, `fv_immersed_boundary`)
  take their wall temperature from `solid` (one value or a function
  `(wall, x, y) -> T`); `fv_immersed_wall_heat` sums the wall fluxes.
- Viscosity depends on temperature only when the model asks for it with
  `FVStateTemperature()` in the `fields` tuple of `FVLawViscosity`. Laws are
  scalar laws with inputs `(shear rate, temperature)`: any
  `AbstractScalarLaw`, for instance `E5ThermalLaw`, the eight-parameter
  temperature- and shear-dependent law of the E5 case. Without
  `FVStateTemperature` the temperature is passive (one-way).
- The flow is driven by a fixed body force (`source` of `fv_steady_problem`) or
  by an imposed mean velocity (`u_mean_target = U`). In the second case the
  uniform force `Fx` becomes one more unknown, constrained so that the mean
  x-velocity over fluid cells equals `U`; it requires periodic x sides.
  `fv_flow_force(prob, U)` reads the solved force.

### Thermal boundary conditions

`FVEnergy(left, right, bottom, top, solid)` sets the temperature of the W, E, S
and N sides and of the voxel-solid faces. `alpha` is the diffusivity, `h` the
cell size, `T_P` the cell temperature, and `F` the outward volume flux of the
face (`h` times the normal velocity). A Dirichlet face uses the half-cell
gradient `2 alpha (T_P - T_b)`: the value is imposed on the face, half a cell
from the cell centre.

| Side | Allowed values | Face terms of the thermal row | Backing test |
|---|---|---|---|
| `:wall` | finite `T_w` or `:adiabatic` | upwind convection of the prescribed wall velocity (zero for no-slip), plus `2 alpha (T_P - T_w)` if `T_w` is set | `incns_fv_energy_test.jl` (conduction, Couette/Poiseuille heating, Nahme in `incns_fv_thermal_coupling_test.jl`) |
| `:inlet` | finite `T_in` (required) | `F T_in` for inflow (`T_P` if the face reverses) `+ 2 alpha (T_P - T_in)`; `F` from the prescribed inlet velocity | `incns_fv_energy_open_bc_test.jl`: row formulas, Graetz, energy balance |
| `:outlet` | `:adiabatic` placeholder (required) | `F T_P` in either direction, no diffusive flux (zero normal gradient); `F = h u_P` (or `h v_P`), the continuity outlet flux | same file: row formulas, constant-temperature identity with reversed faces, Graetz, energy balance |
| `:periodic` | `:adiabatic` placeholder | periodic transport | `incns_fv_energy_test.jl` |
| `solid` | finite `T_s` or `:adiabatic` (default) | `2 alpha (T_P - T_s)` on every solid face, no convection (solid faces are no-slip) | `incns_fv_energy_open_bc_test.jl`: row formulas, energy balance, Jacobian, adjoint |

These are the conditions of the LBM open-die driver (`run_gnf_open_steady`,
branch `feat/gnf-open-steady`): its inlet is a Dirichlet face carried by the
inlet flux, its outlet is zero-gradient outflow, its walls and solid faces are
Dirichlet half a cell from the fluid centre. One temperature serves every
solid face; a per-cell solid temperature field is not supported.

The new terms depend only on the cell's own `(u, v, T)`, so the Jacobian
pattern is unchanged and the coloured-dual Jacobian and the thermal adjoint
carry them without extra code. `incns_fv_energy_open_bc_test.jl` checks, on
CPU in Float64:

- every thermal row against the formulas above written out by hand, with
  prescribed face velocities (difference at most 1e-13);
- the Graetz problem (parabolic inlet at `T = 1`, walls at `T = 0`, 384 × 64
  cells, `Pe = U G / alpha = 200`): local Nusselt number within 2 % of the
  30-mode parallel-plate series for `x` from `G` to `5G`, once with exact face
  velocities and once with the coupled Stokes Newton solve; the same gate and
  series as test O2 of the LBM driver;
- a global energy balance at convergence (inflow enthalpy and conduction,
  outflow enthalpy, wall and solid heat, viscous heating) summing to at most
  1e-9 of its largest term, and inflow equal to outflow to 1e-10;
- the constant-temperature identity (thermal row = `T` times the continuity
  row) with reversed outlet faces;
- coloured against dense dual Jacobian and central finite differences for
  Newtonian, GNF, prescribed-field law and temperature-dependent GNF models;
- the thermal adjoint (E5 and learned laws) against central finite differences
  of full re-solves, with the T4 thresholds.

### Why the unknown is `theta = T - T_ref`

At a few hundred kelvin the spacing of floating-point numbers is large compared
with the residuals Newton must reach (6e-14 K at 468 K), and the convective term
would carry `T_ref` times the continuity residual. The raw state therefore
stores `theta = T - T_ref`, with `T_ref` set by `FVEnergy(T_ref = ...)` (default:
mean of the prescribed wall temperatures). Everything the user touches stays in
absolute temperature: wall values, `fv_state`, `fv_fields`, the viscosity laws,
the objectives. Only the raw state vector (`U[4:4:end]`, `fv_residual(U, prob)`)
holds `theta`. The solution does not depend on `T_ref` up to round-off.

### Stopping rule and round-off floor gate

For a problem with energy, `atol`/`rtol` of `fv_newton!` do not select the
stopping rule. Newton stops, with `gate = :tolerance`, when every equation's
scaled maximum residual is at most 1e-10 and the last accepted temperature
increment is at most `1e-10 * DeltaT`. The scales are computed once from the
case data and stay fixed through the line searches.

On realistic coefficients (E5 lattice units, Péclet number about 800, T about
468 K) the energy residual cannot go below the rounding error of the sum that
defines it, so the first rule is unreachable. The second way to stop, the
round-off floor gate (`gate = :roundoff_floor`), accepts the iterate when the
increment criterion holds, every non-energy equation is within 1e-10, and in
every thermal row `|R| <= floor`, with

```math
\mathrm{floor} = C\,\varepsilon\,\sum |\text{terms of the row}|, \qquad C = 64 ,
```

where `eps` is the machine epsilon and the sum adds the absolute values of the
convective fluxes, the diffusive fluxes, the wall terms, the heating and the
source: the magnitude of what cancels in the row. `C = 64` is 4 faces times
(convective plus diffusive term) times a margin of 8 for the accumulated
rounding of a short sum of products; it was fixed before any measurement and is
not tuned (`KrakeNN.FV_ENERGY_FLOOR_C`). The tolerance gate is tried first. The
Newton and continuation results carry `gate` and, for Newton, `energy_floor`
(`passed`, `excess`, `max_floor`, `max_ratio`); in the E5 stage 2 measurements below the
largest `|R|/floor` is 0.025 (300x100) and 0.023 (600x200).

### Continuation in `sigma` and the fold

Strong heating can have no steady solution (thermal runaway: the Nahme
problem has a fold). `fv_thermal_continuation(prob; U0, initial_step = 0.25,
max_halvings = 5, maxit = 30, to, linsolve)` starts from an isothermal solve at
`sigma = 0` and raises `sigma` to `prob.energy.sigma_nom` in steps. A step that
fails (line search, iteration limit, singular factorization) is undone and the
increment halved, at most `max_halvings` times in the whole run. The result has
`converged`, `reason` (`:converged`, `:step_halving_limit`, ...), `last_sigma`,
`history` (per step: `sigma`, Newton `stats`, `gate`, `singular_min`), `attempts`
(including failed ones), the final state `U` and `solver`. `singular_min` is an
inverse-iteration estimate of the smallest singular value of the Jacobian; it
decreasing along the path is a fold warning, not a proof. `FVEnergy(sigma,
sigma_nom)` fixes the stopping scales to the final heating from the first step.
No arclength continuation exists: past the fold the run returns the last
converged state.

### Adjoint through the temperature

`fv_steady_gradient(solver, U, theta, objective; params)` differentiates a
scalar objective of the converged state with respect to law parameters
(`params = FVLawParameters(n)`), through momentum, continuity and the energy
equation (one refactorization of the Jacobian and one transposed solve). New
objectives:

- `fv_mean_temperature(prob; T_wall, target = 0)`:
  `0.5 * (mean_fluid(T - T_wall) - target)^2`;
- `fv_temperature_l2(prob, target; weights = 1)`:
  `0.5 * sum_fluid(weights * (T - target)^2)`;
- `fv_flow_force_objective(prob)`: the solved `Fx` of an `u_mean_target` problem.

After a continuation the adjoint differentiates only the final steady problem at
`last_sigma`, with `sigma` held fixed; the continuation path is not
differentiated.

### Usage

A heated plane Poiseuille channel with `nu(T) = nu0 exp(-beta (T - Tw))`,
then the gradient of the mean temperature rise with respect to `(nu0, beta, Tw)`
(CPU; on GPU pass `to = CUDA.CuArray, linsolve = CUDABackendTag()` to the
continuation, with `using CUDA, CUDSS`):

```julia
using KrakeNN

struct ExpLaw <: AbstractScalarLaw end        # inputs: (shear rate, temperature)
KrakeNN.law_ninputs(::ExpLaw) = 2
KrakeNN.law_nparams(::ExpLaw) = 3
@inline KrakeNN.law_value(::ExpLaw, x::Tuple, theta::AbstractVector) =
    theta[1] * exp(-theta[2] * (x[2] - theta[3]))

theta = [0.3, 1.0, 1.0]                       # nu0, beta, Tw
visc  = FVLawViscosity(ExpLaw(), theta, (FVStateTemperature(),); eps = 0.02)
prob  = fv_steady_problem(2, 32; bc = (W = :periodic, E = :periodic, S = :wall, N = :wall),
                          viscosity = visc, source = (x, y) -> (0.6, 0.0, 0.0),
                          energy = FVEnergy(alpha = 0.2, sigma = 1.9, bottom = 1.0))

cont = fv_thermal_continuation(prob)          # sigma: 0 -> 1.9
cont.converged, cont.reason, cont.gates
u, v, p, T = fv_fields(prob, Array(cont.U))   # T in absolute temperature

obj = fv_mean_temperature(prob; T_wall = 1.0)
r   = fv_steady_gradient(cont.solver, cont.U, theta, obj; params = FVLawParameters(3))
r.grad                                        # dJ/d(nu0, beta, Tw)
```

For a fixed mean velocity instead of a fixed force, add `u_mean_target = 0.05`
(any finite value) to `fv_steady_problem`, keep a uniform x `source` (its value
is the initial guess of `Fx`), and read the force with `fv_flow_force(prob, cont.U)`. The energy-only solve with frozen
flow (`fv_energy_solver`, `fv_energy_solve!`) accepts prescribed face
velocities through `FVEnergy(face_velocity = (uf, vf))`.

### 3D problems and the PIV objective

The same `fv_steady_gradient` call works on a 3D problem
(`fv_steady_problem(Nx, Ny, Nz; ...)`, isothermal for now): parameter maps
`FVUniformViscosity`, `FVViscosityField` (`Nx Ny Nz` values),
`FVSideVelocity(prob, side; shape = (x, y, z) -> (u, v, w))` on the six sides,
`FVUniformSource(comp)` and `FVLawParameters(n)`; objectives `fv_flow_rate(prob, i)`,
`fv_flow_force_objective(prob)` and `FVWallForce(dir; region)` (`dir = 1..3`,
regions include `:T`, `:B`). For calibration against plane PIV,

```julia
ops = map(pl -> piv_operator(pl, zeros(Nx, Ny, Nz); h = (h, h, h)), planes)  # CuArray template on GPU
obj = FVPIVObjective(prob, ops, data; mask, sigma_u = 0.05,
                     terms = ((fv_flow_force_objective(prob), 1.0, dp_measured),   # (J_t - target)^2 / 2
                              (FVWallForce(1; region = :solid), 0.1)))             # weight * J_t
r   = fv_steady_gradient(solver, U, theta, obj; params = FVLawParameters(n))
```

sums `piv_misfit / sigma_u^2` over the planes and the weighted terms. The PIV
planes and linear terms are evaluated on the solver backend without dual
numbers or a host copy of the state; nonlinear terms (the wall force) use host
duals. Upwind fluxes use `abs`, differentiated by ForwardDiff with the sign
bit of the value (`abs'(+0.0) = 1`, `abs'(-0.0) = -1`): at a face with exactly
zero mass flux the gradient is one of the two one-sided derivatives, not
their mean, so finite-difference checks should avoid such symmetric states.
On GPU the all-linear objective and its gradient allocate at most 1 KiB of
device memory per call (a fixed amount, the same at every grid size; measured
by `benchmarks/fv3d/i2_adjoint_measure.jl`).

### 3D problems without a direct factorisation: `KrylovBackendTag`

The direct factors of a 3D Jacobian grow faster than the grid (about 25 GB at
48³). `KrylovBackendTag()` replaces them by restarted GMRES on the assembled
Jacobian, behind the same calls. GMRES needs a preconditioner, an approximate
inverse of the Jacobian that is cheap to apply; the tag offers five:

- Vanka multigrid (`precond = :vanka`): a coupled (monolithic) multigrid with
  a Vanka smoother, i.e. one multigrid cycle on the whole velocity–pressure
  system, whose smoother solves, cell by cell, the small local system made of
  the cell's velocities and its pressure (constant prolongation with
  `vanka_prolong = :constant`, damping `vanka_omega`, default 0.5).
- Block LSC (`precond = :mg, block = :upper, schur_approx = :lsc,
  mg_cycles = 4`): a block-triangular preconditioner with an LSC
  Schur-complement approximation, i.e. the velocity block is solved by
  aggregation multigrid (4 cycles) and the pressure through the
  least-squares commutator (LSC) approximation of the Schur complement.
- ILU(0) (`precond = :ilu0`): an incomplete LU factorisation of the whole
  Jacobian that keeps only its sparsity pattern.
- Block SIMPLE (`precond = :simple`): ILU(0) of the velocity block and an LU
  of the SIMPLE Schur approximation C − D diag(A)⁻¹ G for the pressure
  (`schur = :ilu0` replaces that LU by an ILU(0)).
- Block SIMPLE with multigrid (`precond = :mg`): the same splitting with one
  aggregation-multigrid cycle for each block.

The default (`precond = :auto`) picks one from the Jacobian:

- 3D problems without energy equation: Vanka multigrid with
  `vanka_prolong = :constant` and `vanka_omega = 0.5`;
- 3D problems with an energy equation: block LSC with `mg_cycles = 4`;
- 2D problems: ILU(0), as before.

With the default, these fields are set by the choice; to tune them, name
`precond` explicitly (an explicit `precond = :vanka` uses trilinear
prolongation unless `vanka_prolong = :constant` is given).

What was measured on the 3D benchmarks (`benchmarks/fv3d/README.md`):

- Vanka multigrid with `vanka_omega = 0.5` and constant prolongation passes
  the D2 gate (GMRES iterations, memory, agreement with the direct solve) at
  32³, 48³ and 64³ on the flow-rate duct and on the body-force duct; with
  `vanka_omega = 0.7` it fails on the body-force duct.
- Block LSC on the 3D energy cases: heated duct to N = 48 and the
  manufactured energy solution to 64³ pass their gates, GPU peak 4.7 GB.
- Law calibration on the V5 twin (H100): 143 s and 5.07 GiB GPU peak with
  Vanka multigrid, 402 s and 6.94 GiB with block LSC, same fitted law.

Range of the default: the Vanka multigrid default is reliable up to
Re ≈ 250 at N = 32 with first-order upwind convection (measured: 208 GMRES
iterations per solve). Beyond that, Newton or GMRES fail (measured failure at
Re ≈ 415); block LSC at Re ≈ 250 was not measured to the end. Second-order
convection and pseudo-transient continuation for this path are planned for
v0.8.

```julia
solver = fv_newton_solver(prob; to = CuArray,
                          linsolve = KrylovBackendTag(precond = :simple))   # `using CUDA, CUDSS`
st = fv_newton!(solver, U; atol = 1e-10, dtau0 = 1.0, dtau_max = 1e4)
r  = fv_steady_gradient(solver, U, theta, obj; params = FVLawParameters(n))
lin_krylov_log(solver.cache)    # iterations, residual, time of every solve
```

Each Newton step is solved to `rtol = 1e-6` (an inexact Newton step; the
Newton iteration still converges to the requested residual) and the adjoint
(transposed) solve to `rtol_transpose = 1e-10`, both measured on the true
residual. With an imposed flow rate (`u_mean_target`), the border unknown is
left out of the preconditioner by default (`border = :krylov`; always with
`:simple`, `:mg` and `:vanka`). A zero or non-finite ILU(0) pivot raises an
error: switch preconditioner or use the direct backend for that case. ILU(0)
alone stalls on 32³ ducts with an imposed flow rate; the iteration counts and
memory of every preconditioner are measured by
`benchmarks/fv3d/i4_krylov_measure.jl`.

## Validation

Test files are in `test/analytical/`; each frozen threshold carries its date and
source in the file header.

**Nahme channel (continuum reference).** `test/reference/fv_thermal_nahme/`
holds an independent solution (SciPy collocation, `solve_bvp`, no KrakeNN code)
of the fully developed channel with `nu(T) = nu0 exp(-beta (T - Tw))` and
viscous heating, for Poiseuille and Couette (`incns_fv_thermal_coupling_test.jl`).
The FV error must be second-order (order at least 1.8 on N = 32, 64, 128) and,
at N = 128, below `8 C / 128^2`, with `C` the maximum curvature of the reference.
The same reference gives the fold at `sigma_crit = 2.611`; the test asks for
`sigma = 3` and checks that the continuation stops at the last converged step
with a decreasing singular-value estimate. Derivative consistency of the
Jacobian and the adjoint (against forward duals and against central finite
differences of full re-solves) is in `incns_fv_thermal_adjoint_test.jl`.

**E5 comparison against the lattice-Boltzmann (LBM) code.** The E5 case is a
periodic cell with six cylinders (voxel mask), a temperature- and
shear-dependent viscosity with eight parameters `(eta0, lambda, beta, n0,
slope, amp, Tc, width)`, a bottom isothermal wall and adiabatic top. The FV
solver is run on the same voxel mask and the same lattice coefficients as the
LBM, with nothing tuned on the FV side. The comparison is staged:

1. flow only (stage 1): FV momentum with the LBM temperature imposed;
2. heat only (stage 2): FV energy equation with the LBM face velocities and heat
   source imposed; this tests the discrete energy operator alone;
3. full coupling (stage 3): flow, temperature and viscosity solved together by
   continuation, then the adjoint of `m`, the mean temperature rise of the fluid,
   compared with the LBM gradient `dm/dtheta` over the eight parameters, and
   checked against central finite differences of FV re-solves on `eta0` and
   `beta`.

Measured on one NVIDIA H100 (10 GB MIG slice), Float64, 2026-11-08:

| Quantity (FV vs LBM) | 300x100 | 600x200 |
|---|---|---|
| Stage 1, flow-rate ratio − 1 | +3.5 % | +1.4 % |
| Stage 2, `m` ratio − 1 | 9e-12 | −2e-8 |
| Stage 2, power balance (bottom flux vs heating) | 3e-15 | −3e-15 |
| Stage 3, `m` ratio − 1 | +3.1 % | +1.4 % |
| Stage 3, flow-rate ratio − 1 | +3.3 % | +1.4 % |
| Stage 3, `dm/dtheta` relative norm difference | 6.0 % | 2.4 % |
| Stage 3, worst single component (`beta`) | 10.7 % | 4.5 % |
| Stage 3, signs of the 8 gradient components | 8 of 8 agree | 8 of 8 agree |
| Stage 3, adjoint vs FV central FD (`eta0`, `beta`) | 2.5e-6, 2.6e-6 | 2.7e-6, 2.7e-6 |
| Stage 3, power balance | 2e-15 | 2e-16 |
| Stage 3, continuation | 9 steps, 54 Newton iterations, 1 halving | same |
| Stage 3, continuation wall time / adjoint | 24 s / 5 s | 84 s / 14 s |

The LBM value of `m` itself moves from 26.33 K to 27.14 K between the two grids
(+3.1 %), so part of the FV–LBM difference at 300x100 is the LBM grid error. The
stage 3 gate was `roundoff_floor` on every continuation step except the first
(isothermal) one. Stage 2 agreeing to 1e-8 or better shows the energy operator is
the same as the LBM one once velocities and source are given; the remaining
gap in stage 3 comes from the flow and the viscosity.

**Acceptance thresholds** (decided 2026-11-08 from the 600x200 job measurement of
that day, with margin; they apply to the 600x200 grid, stage 3 only):

- `|m_FV / m_LBM − 1| <= 2 %`;
- `|flow-rate ratio − 1| <= 2 %`;
- `||dm/dtheta_FV − dm/dtheta_LBM|| / ||dm/dtheta_LBM|| <= 3 %`, each component
  within 6 %;
- all 8 signs agree.

The measurement above meets all of them; 300x100 does not, and is reported
without being judged. The driver writes them as a `[thresholds]` table at the
end of `summary.toml`.

### Running the comparison

The driver is `benchmarks/fv_thermal/e5_compare.jl` (see
`benchmarks/fv_thermal/README.md`). It reads an export directory of the LBM case
(`case.toml`, binary fields, optional `gradient.toml`) and is configured by
environment variables: `KRAKEN_E5_BACKEND` (`cpu` or `cuda`), `KRAKEN_E5_GRID`
(`300x100` or `600x200`), `KRAKEN_E5_EXPORT`, `KRAKEN_E5_OUT`,
`KRAKEN_E5_STAGES` (`1,2,3`), `KRAKEN_E5_FD` (finite differences on or off) and
`KRAKEN_E5_INIT` (`zero` or `lbm` initial flow; the converged solution does not
depend on it).

```sh
KRAKEN_E5_BACKEND=cuda KRAKEN_E5_GRID=600x200 julia --project=. benchmarks/fv_thermal/e5_compare.jl
```

A smoke test on a small synthetic cell (CPU, a few seconds) is
`test/analytical/incns_fv_e5_compare_smoke_test.jl`. The real grids need a GPU:
stage 3 at 600x200 has 480 000 unknowns and 13 million Jacobian entries.

### Die prediction

`benchmarks/fv_thermal/die_predict.jl` solves flow and temperature in the
tapered die of the LBM open-die driver (same voxel mask, inlet profile,
dimensionless groups and boundary temperatures, `h = 1`) with a given law: by
default the Carreau–WLF law of that driver through `FVGNFViscosity`, or any FV
viscosity model returned by `die_viscosity(c)` in a file named by
`KRAKEN_DIE_LAW_FILE` (for instance a learned law in `FVLawViscosity`). It
writes `summary.toml` (Δp, `Δp / dp_scale`, mass and energy balances,
mixing-cup outlet temperature, Newton statistics) and the fields on the LBM
cells; with `KRAKEN_DIE_TRUTH` it also compares them with the LBM fields. See
`benchmarks/fv_thermal/README.md`. Not yet run at the paper resolution.

## Known limits

- Two grids only (300x100 and 600x200). Two points do not establish a
  convergence order of the FV–LBM difference; the figures above are
  measurements, not an extrapolation.
- Float64 only. Float32 is not validated.
- Steady only; no transient energy equation, and no differentiation of the
  continuation path.
- Inlet/outlet thermal sides, Dirichlet solids and immersed walls with energy
  are 2D only; the 3D energy equation (`FVEnergy3D`) takes wall and periodic
  sides only. A per-cell solid temperature is not supported. With
  `face_velocity`, an outlet reads the wrap slot (`uf[Nx, j]`, `vf[i, Ny]`),
  so two opposite outlets are rejected.
- The round-off floor gate accepts a residual up to a multiple of the rounding
  error; it certifies that the residual is at the precision of the sum, not that
  the iterate is closer to the exact discrete solution than that.
- `fv_thermal_continuation` has no arclength step: it cannot follow a branch past
  a fold.

## Conformation tensor (viscoelastic, 3D)

The 3D steady problem accepts an opt-in conformation equation, solved in the
same Newton system as `(u, v, w, p)`:

```julia
conf = FVConformation3D(; lambda = 1.0, nu_p = 0.5, alpha = 0.4)   # alpha = 0: Oldroyd-B
p = fv_steady_problem(2, 32, 2; nu = 0.5, conformation = conf,
                      bc = (W = :periodic, E = :periodic, S = :wall, N = :wall,
                            B = :periodic, T = :periodic),
                      source = (x, y, z) -> (2.0, 0.0, 0.0, 0.0))
U = fv_state(p, zeros(2, 32, 2), zeros(2, 32, 2), zeros(2, 32, 2), zeros(2, 32, 2))
fv_newton!(fv_newton_solver(p), U)
c = fv_conformation_fields(p, U)          # (; xx, xy, xz, yy, yz, zz)
```

The unknown is `c` itself (six components per cell, the last six cell slots),
not `log c`. Each fluid cell solves
`u·∇c − L c − c Lᵀ + Rlx(c)/λ = 0`, `L_ij = ∂u_i/∂x_j`, and the momentum
traction gains `τ_p = (ν_p/λ) S(c)`, `S(c) = c − I` except for FENE-P; `nu` is
the solvent viscosity. The relaxation term has the general isotropic form
`Rlx(c) = b0 I + b1 (c − I) + b2 (c − I)²` (by the Cayley–Hamilton theorem every
isotropic function of the symmetric `c` can be written so), with scalar
coefficients of the invariants of `c` and of the parameters `theta`:

| law | `theta` | `(b0, b1, b2)` | `S(c)` |
|-----|---------|----------------|--------|
| `PolynomialRelaxation()` Oldroyd-B / Giesekus | `(1, 0)` / `(1, α)` | `(0, 1, α)` | `c − I` |
| `LinearPTTRelaxation()` | `(ε,)` | `(0, 1 + ε(tr c − 3), 0)` | `c − I` |
| `ExponentialPTTRelaxation()` | `(ε,)` | `(0, exp(ε(tr c − 3)), 0)` | `c − I` |
| `FENEPRelaxation()` | `(L²,)` | `(f − 1, f, 0)`, `f = (L² − 3)/(L² − tr c)` | `f c − I` |

PTT is the upper-convected form (`ξ = 0`): the Gordon–Schowalter term
`ξ (D τ + τ D)` depends on the rate of deformation, not on `c` alone, so it is not
of this form and is not implemented. FENE-P is defined only while `tr c < L²`;
nothing clips an iterate that crosses the bound. In simple shear (`Wi = λ γ̇`)
all three give `η_p(γ̇) = η_p/f`, `Ψ1 = 2 η_p λ/f²`, `Ψ2 = 0`, with
`f³ − f² = 2εWi²` (linear PTT), `f³ − f² = 2Wi²/L²` (FENE-P) and
`f² ln f² = 4εWi²` (exponential PTT); references and DOIs are in the
docstrings. A new law subtypes `AbstractRelaxationLaw` and returns
`(b0, b1, b2)` (or `(b1, b2)`, meaning `b0 = 0`) from the invariants of `c`,
optionally with its own `relaxation_stress`, so the law can later be learned
without touching the residual. Discretisation: first-order upwind advection with the Rhie–Chow mass
flux, Green–Gauss velocity gradient, face mean of `c` in the polymer stress and
linear extrapolation `(3 c_P − c_O)/2` on wall faces. There is no boundary
condition on `c` (walls carry no flux). The Jacobian comes from the same
coloured ForwardDiff assembly; `conformation = nothing` leaves the problem
unchanged.

Validation (`test/analytical/incns_fv3d_conformation_test.jl`): Oldroyd-B plane
Poiseuille (velocity equal to the Newtonian `η_0 = η_s + η_p` solution, `c_xy`,
`N1 = 2 η_p λ γ̇²` and `N2 = 0` at the cell centres); Giesekus, linear and
exponential PTT and FENE-P simple shear (pointwise, BigFloat) and plane Couette
against the closed-form viscometric functions above; dual derivatives of each
law against central differences.

Pseudo-transient continuation on the conformation rows is opt-in:
`fv_newton!(s, U; dtau0 = 1.0, shift_conformation = true)` adds the same
diagonal shift `h³/Δτ` (the term `∂c/∂τ`) to the six `c` rows as to the
momentum rows. Only the Newton path changes; the residual is the steady one, so
the converged state is the same (checked against plain Newton in the test). It
is not a transient solver.

### Square-duct slice and continuation in De

`fv_set_conformation!(solver, conf)` replaces `lambda`, `nu_p` and `theta` of a
conformation-active solver in place (same law type), so a continuation in the
Deborah number keeps the Jacobian pattern, the colouring and the one symbolic
analysis:

```julia
for De in (0.1, 0.3, 0.6, 1.0)
    fv_set_conformation!(s, FVConformation3D(; lambda = De, nu_p = 1.0, alpha = 0.4))
    fv_newton!(s, U)                      # starts from the previous solution
end
```

`benchmarks/ve_duct/slice_conformation.jl` solves the secondary flow of a
Giesekus fluid in a square duct on one periodic slice (`Nx = 1`, walls on y and
z, imposed mean axial velocity) on the CPU or on CUDA (`VE_BACKEND`), with this
continuation (plain Newton, then a damped retry, then halved De steps), and
writes a TOML summary (iterations per De, `max|u_xy|/U`, RMS secondary speed,
smallest eigenvalue of `c`, symmetry defect, device memory) plus one CSV of cell
fields per De. The case is the RheoTool reference of
kraken-research/hidden-rheology/ve-duct: side `H = 1`, `U_mean = 1`,
`η_s = η_p = 1` (`β = 0.5`, `Re = ρ U H / η_0 = 0.5`), `α = 0.4`,
`De = λ U_mean / H` (the RheoTool runs set `λ = De`). Test (h) of the
conformation test file checks a 12 × 12 slice at De 0.1: the mirror and diagonal
symmetries, the eight-vortex sense of the RheoTool solution (along the diagonals
from the corners toward the centre, along the wall bisectors toward the walls)
and, for Oldroyd-B, `N2 = 0` and no secondary flow.

Adjoint and sensitivities: `FVRelaxationParameters(:lambda, :nu_p, :theta,
:nu_s)` (any non-empty subset, `θ` in that order) maps the relaxation time, the
polymer viscosity (`G = ν_p/λ` follows both), the free law parameters (default:
`α` for Giesekus with `b1 = 1` fixed, `ε` for PTT, `L²` for FENE-P; other entries
through `theta_slots`) and the uniform solvent viscosity. It plugs into
`fv_steady_adjoint`, `fv_steady_tangent`, `fv_steady_gradient` and
`fv_set_params!` like the other parameter maps; the coefficient derivatives are
taken by dual numbers in the conformation residual kernel, on CPU and GPU.
Values outside the admissible range (`λ, ν_p, ν_s ≤ 0`, `α ∉ [0, 1/2]`, `ε < 0`,
`L² ≤ 3`) raise `ArgumentError`; they are never clipped. The adjoint and the
tangent first call `fv_check_conformation_domain(prob, U)`, which raises
`DomainError` when `c` is not positive definite in some fluid cell or, for
FENE-P, when `max tr c ≥ L²`; `fv_conformation_domain` returns the smallest
eigenvalue, the largest trace and the FENE-P margin without throwing.
Combinations covered by the derivative gate (test file `incns_fv3d_conformation_gate_test.jl`, square
duct slice, objective = section PIV + `Fx` + mean `N1/G`): Oldroyd-B, Giesekus,
linear and exponential PTT, FENE-P with `λ, ν_p, ν_s` and the law parameter, the
learned law (network weights), on CPU and GPU at 8 × 8, and Giesekus `λ, α` and
one learned-law weight on GPU at 24 × 24; adjoint against
tangent, centred finite differences and the dual chunk size. For Oldroyd-B in a
straight duct the velocity and `Fx` do not depend on `λ` (zero derivative,
checked): only a conformation term in the objective makes `λ` identifiable.

Transposed solve at 10 unknowns per cell (`benchmarks/fv3d/ve_transpose_measure.jl`,
Giesekus square duct N³, De 0.1 → 0.3, `rtol_transpose = 1e-10`, GPU, 2026-11-16).
The direct cuDSS solve needs 1.2, 6.8 and 23 GB at its peak at N = 16, 24, 32 and
runs out of memory at 48³ on an 80 GB GPU. With a rib of solid cells at 32³ its
transposed residual is 1.7e-7, against 1e-11 to 1e-15 everywhere else. GMRES with
the `:auto` preconditioner, which is ILU(0) at 10 fields, does not converge at
N ≥ 24 (500 iterations). Passing the preconditioner explicitly works:

- block LSC (`precond = :mg, block = :upper, schur_approx = :lsc, mg_cycles = 4`)
  converges in 44, 65, 115 and 168 transposed iterations (0.5 to 9.5 s) at
  N = 16, 24, 32, 48;
- Vanka multigrid (`precond = :vanka`) converges in 70, 119, 137 and 289
  iterations (0.2 to 8.2 s).

`:auto` is not changed. With conformation unknowns, pass `precond`
explicitly. The same script, with `FV3D_VE_GEOM=slice`, measures the
1 × N × N slice. With `FV3D_VE_REFINE=k`, it adds `k` steps of iterative
refinement to the direct transposed solve.

Planar and 3D cases (`benchmarks/fv3d/ve_adjoint_cases.jl`, Giesekus `λ, α`,
same gates G1/G2 as the test file). The first case is a periodic array of
cylinders between two walls, made of solid cells, N × N × Nz with z periodic.
Nz = 1 and Nz = 2 must give the same state to 1e-12, with `w, c_xz, c_yz` below
1e-13. The second is a periodic square duct N³ with a solid constriction (half
the side open), at 24³ on the GPU, with block LSC and with cuDSS. The script
reports the adjoint time divided by the Newton time. Not run yet.

Limits: wall (zero normal velocity) and periodic sides only; Newtonian
solvent; no energy equation together with conformation; PTT with `ξ ≠ 0` not
available; the duct benchmark's damped retry still shifts the momentum rows only
(`shift_conformation` is not passed there).

Learned relaxation law: `NeuralRelaxation{H}(; alpha0, scale)` gives
`(b0, b1, b2) = (0, 1, alpha0) + M(x) − M(0)`, `M` a one-hidden-layer tanh
network of `x = (I1 − 3, I2 − 3, I3 − 1) / scale` with `H` units and no output
bias (it would cancel), `7H` parameters stored as an array (host or GPU) rather
than a tuple. `relaxation_init(law; rng, output_std = 0)` draws the hidden layer
and sets the output weights to zero, so the law starts as Giesekus with mobility
`alpha0`; `FVRelaxationParameters(:theta)` maps the weights (or a subset through
`theta_slots`). The form does not guarantee a positive definite `c`; each state
is checked by `fv_check_conformation_domain`, so the law is only known to work
on the states that were tested.
