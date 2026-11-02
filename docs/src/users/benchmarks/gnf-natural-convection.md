# Generalised Newtonian natural convection

Natural convection of a generalised Newtonian fluid (viscosity depending on the
local shear rate) in a differentially heated square cavity: hot west wall, cold east
wall, adiabatic top and bottom, no-slip walls, Pr = 0.71. Driver
`run_natural_convection_gnf_2d`, which couples the thermal double-distribution
D2Q9 solver of the [Newtonian cavity](thermal-natural-convection.md) to the
shear-rate-dependent collision of the rheology chapter of the theory section
(`docs/src/theory/14_rheology.jl`, not in the published navigation).

> **Status: CPU-validated only, qualitative.** The Newtonian limit and one
> De Vahl Davis point are checked quantitatively. The dependence on the power-law
> index is checked as a trend only. There is no comparison with the power-law
> literature yet and no GPU run (see [Not yet verified](#Not-yet-verified)).

## Definitions

Ra and Pr are defined with the **reference kinematic viscosity** ``\nu_0``
(`nu0`), not with a reference shear rate:

```math
\mathrm{Pr} = \frac{\nu_0}{\alpha}, \qquad
\mathrm{Ra} = \frac{\beta g\,\Delta T\,H^3}{\nu_0\,\alpha}
```

In the code (`gnf_reference_scales`) this gives `alpha = nu0/Pr` and
`beta_g = Ra*nu0*alpha/(ΔT*H^3)`. The local viscosity of the fluid changes the
flow but not these definitions.

The helper `cavity_powerlaw(n; nu0, alpha, H)` builds a power-law fluid
``\nu = K\dot\gamma^{\,n-1}`` with

```math
K = \nu_0\left(\frac{\alpha}{H^2}\right)^{1-n},
```

so that the viscosity equals ``\nu_0`` at the shear rate ``\alpha/H^2``. This
**diffusive shear scale is provisional**: it is a placeholder until the published
power-law Ra/Pr definition is implemented, and it is not a literature mapping.
Values of Nu for ``n \neq 1`` therefore depend on this choice.

## Run it on CPU

This runs in about ten seconds per index on a laptop CPU (N = 32, Ra = 10⁴):

```julia
using Kraken

Pr, nu0, N = 0.71, 0.05, 32
scales = gnf_reference_scales(; Ra=1e4, Pr, nu0, H=Float64(N), T_hot=1.0, T_cold=0.0)

for n in (0.8, 1.0, 1.2)
    model = cavity_powerlaw(n; nu0, alpha=scales.alpha, H=N)
    r = run_natural_convection_gnf_2d(; N, Ra=1e4, Pr, nu0, rheology=model,
        max_steps=8000, check_every=1000, convergence_rtol=1e-5, min_steps=5000)
    println("n=$n  Nu=$(round(r.Nu, digits=3))  converged=$(r.converged)  ",
            "clamped=$(r.clamped_fraction)")
end
```

Output measured when this page was written (CPU, Float64):

```
n=0.8 Nu=3.496 steps=8000 converged=true clamped=0.0
n=1.0 Nu=2.492 steps=8000 converged=true clamped=0.0
n=1.2 Nu=1.822 steps=8000 converged=true clamped=0.0
```

The result is a `NamedTuple` with `Nu`, `Temp`, `ux`, `uy`, `rho`, the viscosity
field `nu_field`, `clamped_fraction` (share of nodes sitting at a viscosity cutoff
at the last step), `converged` and `Nu_drift`.

The same physics from a `.krk` file: a `Rheology` block other than Newtonian in a
`natural_convection` case routes to this driver.

```
Simulation natural_convection_gnf D2Q9
Domain L = 1 x 1 N = 16 x 16
Physics nu = 0.05 Ra = 1000 Pr = 0.71
Module thermal
Rheology power_law { K = 0.05 n = 0.9 nu_min = 0.001 nu_max = 1.0 }
Boundary west wall
Boundary east wall
Boundary south wall
Boundary north wall
Run 3000 steps
```

Parsed with `parse_kraken` and run with `run_simulation(setup)`, this snippet ran on
CPU and returned `Nu = 1.129` after 3000 steps. In this route `K` is the value you
write in the block; the `cavity_powerlaw` scaling is not applied.

## What was validated

All checks are in `test/test_natconv_gnf.jl` (CPU, Float64), run on the date of
this page; all 35 assertions passed.

| Check (testset) | Measured | Gate |
|---|---|---|
| ``n = 1`` against the Newtonian driver `run_natural_convection_2d`, N = 64, Ra = 10³ ("Newtonian limit and De Vahl Davis") | Nu = 1.1423 for both, relative gap 3.9e-16 | gap < 1 % |
| Nu against De Vahl Davis (1983), Nu = 1.118, same case | Nu = 1.1423, i.e. +2.2 % | < 10 % (same frozen gate as `test_thermal.jl`) |
| Nu against the power-law index, N = 32, Ra = 10⁴ ("QUALITATIVE literature trend") | Nu = 3.496 / 2.492 / 1.822 for ``n`` = 0.8 / 1.0 / 1.2 | Nu(0.8) > Nu(1.0) > Nu(1.2); Nu drift < 1e-5 over three successive 1000-step checks; clamped fraction < 1 % (measured 0.0) |
| Carreau-Yasuda, N = 8, 5 steps ("Carreau-Yasuda cavity API") | finite fields, viscosity between ``\eta_\infty`` and ``\eta_0`` | smoke test only |
| `.krk` case reaches the driver ("Parsed thermal rheology reaches run_simulation") | `power_law` block routed, 5 steps, finite fields | smoke test only |

Reading of the table:

- The ``n = 1`` row shows that the new driver and the Newtonian driver solve the
  same discrete problem; it does not validate the non-Newtonian physics.
- The De Vahl Davis row is a 2.2 % agreement on a 64² mesh at Ra = 10³. The
  Newtonian benchmark page reaches 0.79 % on a 192² mesh for the Newtonian driver.
- The trend row says that shear-thinning raises the heat transfer and
  shear-thickening lowers it, as expected physically. It carries no quantitative
  claim, since the shear scale of ``K`` is provisional.

## Not yet verified

- **No comparison with Turan, Sachdeva, Chakraborty and Poole** (J. Non-Newtonian
  Fluid Mech. 166 (2011) 1049-1063, DOI 10.1016/j.jnnfm.2011.06.003). Their Ra/Pr
  definitions and tabulated Nu were not available when the driver was written, so
  no number here should be read as agreeing or disagreeing with that paper.
- **No GPU run.** The sweep is prepared in `benchmarks/validation/natconv_gnf/`
  (N = 256, Ra = 10⁴ and 10⁵, ``n`` from 0.6 to 1.8, CUDA Float64) and has not been
  executed.
- **Coarse meshes only**: N = 64 for the Newtonian limit, N = 32 for the index trend.
  No mesh-convergence study was made.
- **Two Rayleigh numbers only**: 10³ (Newtonian limit) and 10⁴ (trend).
- **Float32** was not tested.
- The shear scale ``\alpha/H^2`` in `cavity_powerlaw` is provisional (see above).

## Reference

de Vahl Davis, G. (1983), *Natural convection of air in a square cavity: a bench
mark numerical solution*, Int. J. Numer. Methods Fluids 3, 249-264,
DOI 10.1002/fld.1650030305.
