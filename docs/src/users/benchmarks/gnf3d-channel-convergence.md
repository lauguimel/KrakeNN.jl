# Generalised-Newtonian channel (3D) — analytical convergence

A planar channel driven by a uniform body force is the simplest 3D flow whose exact
solution is known for a shear-rate-dependent viscosity. This page documents the check of
Kraken's 3D generalised-Newtonian (GNF) lattice-Boltzmann path against that exact solution:
the observed order of convergence is 2 for the two Carreau cases and for the shear-thinning
power law, and between 1.93 and 1.95 for the shear-thickening power law (CPU, Float64,
N = 16 → 64). The GPU run of the same cases has not been done.

## The case

- **Lattice and collision**: D3Q19, two-relaxation-time (TRT) collision with magic parameter
  `3/16`; the relaxation rates are recomputed in every cell from the local viscosity
  `ν(γ̇)`, where `γ̇` is the shear-rate magnitude estimated from the non-equilibrium
  stress of the populations.
- **Forcing**: uniform body force `F` along the flow axis, Guo scheme (half-force velocity
  shift, with the force correction included in the strain-rate estimate).
- **Walls**: linearly interpolated bounce-back (LI-BB) at the two parallel walls; `q_w` is
  the fraction of the lattice link lying in the fluid, so `q_w = 0.5` is the half-way wall
  and `q_w = 0.3` an off-lattice wall. `N` fluid nodes span a gap `H = N − 1 + 2 q_w`.
- **Periodic** in the two other directions (4 nodes including ghosts, the default).
- **Viscosity laws**: `PowerLaw`, `CarreauYasuda` (with `a = 2`, i.e. Carreau), `Cross`
  are accepted; the convergence study uses the first two. The theory is in
  `docs/src/theory/14_rheology.md`.

## Analytical reference

With a uniform force `F`, the momentum balance across the gap gives a linear shear stress
`|τ(z)| = F·|z − H/2|`. The velocity is the integral of the inverse stress law
`γ̇ = τ⁻¹(|τ|)`, computed independently of Kraken's viscosity and strain-rate code
(`test/analytical/gnf3d_reference.jl`; the Carreau inverse is solved by bisection to
floating-point resolution, the power law is closed-form). The reference viscosity profile
is compared as well as the velocity.

| Law | Parameters |
|---|---|
| Carreau | `η₀ = 0.3`, `η∞ = 0.01`, `n = 0.5`, `a = 2`; `λ·γ̇_wall = 10`, wall rate scaled with `H` so the dimensionless problem is the same at every `N` |
| Power law, shear-thinning | `n = 0.7`, `ν ∈ [10⁻³, 1]` |
| Power law, shear-thickening | `n = 1.5`, `ν ∈ [10⁻³, 1]` |

## Run it

```julia
using Kraken, KernelAbstractions
include(joinpath(pkgdir(Kraken), "test", "analytical", "gnf3d_reference.jl"))

ref = _gnf3d_reference(:carreau, 16, 0.5)       # law, N, q_w
out = run_gnf_channel_3d(; rheology = ref.rheology, N = 16, q_w = 0.5, F = ref.F,
        nu_init = ref.nu, u_init = ref.u,
        max_steps = 60_000, tol_ss = 1e-9, check_every = 200,
        backend = KernelAbstractions.CPU(), FT = Float64)

out.converged                       # true
_gnf3d_rel_l2(out.u, ref.u)         # relative L2 velocity error
```

On an Apple-silicon laptop this converges in 5000 steps and about 3 s, with a relative L2
velocity error of 4.6 × 10⁻³ at N = 16 (measured while writing this page). `out.u` and
`out.nu` are the transverse-averaged velocity and viscosity profiles at `out.positions`.
Other laws: `:power_thin`, `:power_thick`.

## Observed orders

Source: `test/analytical/gnf3d_channel_order.jl`, CPU, Float64, `N = 16, 32, 64`, relative
L2 error on the velocity profile, measured 2026-11-02 (macOS arm64, Julia 1.12.5). The
order between two resolutions is `log(e₁/e₂) / log(H₂/H₁)`, using the gaps `H`. The test
requires at least 1.8 on each adjacent pair.

| Case | Order 16 → 32 | Order 32 → 64 | Error at N = 64 |
|---|---|---|---|
| Carreau, `q_w = 0.5` | 2.000 | 2.000 | 2.896 × 10⁻⁴ |
| Carreau, `q_w = 0.3` | 1.990 | 1.995 | 4.282 × 10⁻⁴ |
| Power law, `n = 0.7` | 2.000 | 2.000 | 1.714 × 10⁻⁴ |
| Power law, `n = 1.5` | 1.932 | 1.950 | 2.025 × 10⁻⁴ |

The test also freezes an accuracy limit at twice each N = 64 error. The shear-thickening
profile is the least regular at the centreline and has the lowest observed order; it still
passes the same 1.8 gate.
Other checks in the same file: the run converges, velocity components transverse to the
flow stay below 10⁻¹⁰ of the peak, and the viscosity stays strictly inside the model bounds.

Companion tests in `test/analytical/`: `gnf3d_strain_rate.jl` (the strain-rate estimate,
including the pure-shear sign fix) and `gnf3d_parity.jl` (with a constant viscosity, the
GNF step is compared with the existing Newtonian step). Neither is a convergence study.

## Not yet verified

- **GPU.** No GPU run exists. The script `benchmarks/validation/gnf3d/run_gnf3d_gpu.jl`
  repeats these cases on `CUDABackend()` up to N = 256 and adds a Float32 row; it has not been
  run. Until it is, nothing on this page is a GPU result, and no throughput is claimed.
- **`.krk` route.** 3D GNF is available from Julia only (`run_gnf_channel_3d`). The `.krk`
  dispatch is not yet wired.
- **Float32.** Only the Float64 CPU orders are established.
- **Limits of the implementation.** Isothermal only (a thermal coupling raises an error).
  Supported viscosity laws: Newtonian, `PowerLaw`, `CarreauYasuda`, `Cross`; yield-stress
  models are not supported. Zou–He open faces are limited to flow along `x`, without body
  force, and take the viscosity at the faces by zero normal gradient; that route is not part
  of the convergence study above. The planar channel is the only geometry covered.
- **Power-law range.** For `n = 0.7`, the sweep cannot go beyond N = 128 with the present
  viscosity bounds (`ν_max = 1`), because the analytical viscosity at N = 256 exceeds them.
