# Generalised Newtonian natural convection

Natural convection of a generalised Newtonian fluid (viscosity depending on the
local shear rate) in a differentially heated square cavity: hot west wall, cold east
wall, adiabatic top and bottom, no-slip walls. Driver
`run_natural_convection_gnf_2d`, which couples the thermal double-distribution
D2Q9 solver of the [Newtonian cavity](thermal-natural-convection.md) to the
shear-rate-dependent collision of the rheology chapter of the theory section
(`docs/src/theory/14_rheology.jl`, not in the published navigation).

> **Status: CPU-validated at coarse grids only.** The Newtonian limit, one
> De Vahl Davis point and one power-law point of Turan et al. (2011) are checked
> quantitatively. Five power-law cases of that paper were run on CPU at N ≤ 64
> as a check of the parameter mapping; the GPU comparison on finer grids is
> prepared and pending (see [Not yet verified](#Not-yet-verified)).

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
``\nu = K\dot\gamma^{\,n-1}``, with ``\dot\gamma = \sqrt{2\,S_{ij}S_{ij}}``, and

```math
K = \nu_0\left(\frac{\alpha}{H^2}\right)^{1-n},
```

so that the viscosity equals ``\nu_0`` at the shear rate ``\alpha/H^2``.

### Correspondence with Turan et al. (2011)

Turan, Sachdeva, Chakraborty and Poole (2011) define a nominal viscosity from
the characteristic shear rate ``\alpha/L^2`` (their Eq. 5),
``\nu_\text{nom} = (K/\rho)\,(\alpha/L^2)^{n-1}``, and use it in Ra and Pr
(their Eq. 6):

```math
\mathrm{Pr} = \frac{K}{\rho}\,\alpha^{n-2}L^{2-2n}, \qquad
\mathrm{Ra} = \frac{g\beta\,\Delta T\,L^{2n+1}}{\alpha^{n}\,(K/\rho)} .
```

With `cavity_powerlaw`, ``\nu_0`` is exactly their ``\nu_\text{nom}``: inserting
``\nu_0 = K(\alpha/H^2)^{n-1}`` in the two definitions above gives these
expressions with ``L = H``. Their shear-rate invariant
``\sqrt{e_{kl}e_{kl}/2}`` (with ``e_{ij} = 2S_{ij}``) is the one used by Kraken.

| Turan et al. (2011) | Kraken keyword or quantity |
|---|---|
| ``L`` | `H = N` (lattice units) |
| ``K/\rho`` | `K` of the `PowerLaw` built by `cavity_powerlaw` |
| ``n`` | `n` |
| ``\mu_\text{nom}/\rho`` (Eq. 5) | `nu0` |
| Ra, Pr (Eq. 6) | `Ra`, `Pr` |
| ``T_H`` (east wall), ``T_C`` (west wall) | `T_hot` (west wall), `T_cold` (east wall) |
| mean ``\overline{Nu}`` on a side wall | `Nu` (hot wall) |

`nu0` has no counterpart in the paper: once Ra, Pr and ``n`` are fixed, it only
sets the lattice time step. Two runs with different `nu0` must give the same Nu;
measured on CPU: Ra = 10⁴, ``n`` = 1, N = 32, Nu = 2.529 (`nu0` = 0.05) and 2.537
(`nu0` = 0.1); Ra = 10³, ``n`` = 0.6, N = 48, Nu = 2.122 (`nu0` = 0.15) and 2.124
(`nu0` = 0.3).

What differs from the paper, apart from the method (they solve the steady
equations with a finite-volume code on a 100 × 100 non-uniform mesh):

- the mirror image (hot wall on the other side), which leaves the mean Nu unchanged;
- the buoyancy reference temperature: Kraken uses ``(T_\text{hot}+T_\text{cold})/2``,
  the paper ``T_C``. The difference is a uniform body force, balanced by pressure;
- `PowerLaw` clips the viscosity to `[nu_min, nu_max]`, the paper does not. The
  result reports `clamped_fraction`; it was 0 in every case below;
- the Dirichlet thermal walls sit on the first and last nodes, while the no-slip
  walls are half-way and Ra and Nu use ``H = N``. A conduction-only cavity therefore
  reports Nu ≈ N/(N − 1) instead of 1 (measured 1.036 at N = 32, Ra = 10). This
  offset of order 1/N is shared with the Newtonian driver;
- the paper's power-law data are for Pr ≥ 10; Pr = 0.71 is used here only for the
  Newtonian benchmark.

## Run it on CPU

This runs in about ten seconds per index on a laptop CPU (N = 32, Ra = 10⁴,
Pr = 0.71):

```julia
using Kraken

Pr, nu0, N = 0.71, 0.05, 32
scales = gnf_reference_scales(; Ra=1e4, Pr, nu0, H=Float64(N), T_hot=1.0, T_cold=0.0)

for n in (0.8, 1.0, 1.2)
    model = cavity_powerlaw(n; nu0, alpha=scales.alpha, H=N)
    r = run_natural_convection_gnf_2d(; N, Ra=1e4, Pr, nu0, rheology=model,
        max_steps=8000, check_every=1000, convergence_rtol=1e-5, min_steps=5000)
    println("n=$n Nu=$(round(r.Nu, digits=3)) steps=$(r.steps) ",
            "converged=$(r.converged) clamped=$(r.clamped_fraction)")
end
```

Output measured when this page was written (CPU, Float64):

```
n=0.8 Nu=3.496 steps=8000 converged=true clamped=0.0
n=1.0 Nu=2.492 steps=8000 converged=true clamped=0.0
n=1.2 Nu=1.822 steps=8000 converged=true clamped=0.0
```

Pr = 0.71 lies outside the power-law data of Turan et al. (2011), so these values
have no published counterpart. For a case of the paper, use Pr = 10 or more:

```julia
Pr, nu0, N = 10.0, 0.3, 48
model = cavity_powerlaw(0.6; nu0, alpha=nu0/Pr, H=N, nu_min=1e-3, nu_max=2.0)
r = run_natural_convection_gnf_2d(; N, Ra=1e3, Pr, nu0, rheology=model,
    max_steps=60000, check_every=1000, convergence_rtol=1e-5, min_steps=5000)
r.Nu   # 2.124 measured; the paper gives 2.07 (Fig. 9a)
```

The result is a `NamedTuple` with `Nu`, `Temp`, `ux`, `uy`, `rho`, the viscosity
field `nu_field`, `clamped_fraction` (share of nodes sitting at a viscosity cutoff
at the last step), `steps`, `converged` and `Nu_drift`.

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
write in the block; the `cavity_powerlaw` scaling is not applied, so Ra and Pr
follow the Turan definitions only if you write ``K = \nu_0(\alpha/H^2)^{1-n}``
yourself.

## Comparison with Turan et al. (2011), CPU, coarse grids

Pr = 10. Paper values are read from their figures (no table gives them), with the
reading uncertainty in brackets; page numbers are journal pages. Kraken: CPU,
Float64, `nu_min = 1e-3`, `nu_max = 2.0`, stopping after three successive
1000-step Nu changes below 1e-5; every run converged with `clamped_fraction = 0`.

| Ra | ``n`` | Paper Nu (source) | N | `nu0` | Kraken Nu | Kraken − paper |
|---|---|---|---|---|---|---|
| 10³ | 0.6 | 2.07 (± 0.07, Fig. 9a, p. 1059) | 32 | 0.1 | 2.191 | +5.8 % |
| | | | 48 | 0.3 | 2.124 | +2.6 % |
| | | | 64 | 0.2 | 2.088 | +0.9 % |
| 10⁴ | 0.6 | 5.40 (± 0.02, Fig. 10, p. 1060) | 32 | 0.1 | 6.715 | +24.4 % |
| | | | 64 | 0.2 | 6.037 | +11.8 % |
| 10⁴ | 0.8 | 3.38 (± 0.02, Fig. 10, p. 1060) | 32 | 0.1 | 3.933 | +16.4 % |
| | | | 64 | 0.2 | 3.609 | +6.8 % |
| 10⁴ | 1.0 | 2.29 (± 0.015, Figs. 10–11) | 32 | 0.1 | 2.537 | +10.8 % |
| | | | 64 | 0.2 | 2.399 | +4.8 % |
| 10⁴ | 1.8 | 1.10 (± 0.01, Fig. 11, p. 1061) | 32 | 0.02 | 1.146 | +4.2 % |

Reading of the table:

- Every Kraken value is above the paper and the gap shrinks when N doubles, by a
  factor 2.1 to 2.4 at Ra = 10⁴. This is what a grid error looks like; a wrong
  Ra/Pr mapping would leave an ``n``-dependent gap that does not shrink. These
  grids are too coarse to separate the two completely, which is the purpose of
  the GPU comparison.
- The thinner boundary layers of the shear-thinning fluid need finer grids: at
  N = 64 the gap is 4.8 % for ``n`` = 1 and 11.8 % for ``n`` = 0.6.
- For ``n`` = 1.8, `nu0` must be small. With `nu0` = 0.1 the viscosity near the walls
  reaches 1.44 (relaxation time 4.8) and Nu = 1.196 (+8.8 %); with `nu0` = 0.02 it
  stays below 0.32 and Nu = 1.146. The BGK wall error grows with the relaxation
  time, and a shear-thickening fluid has its largest viscosity at the walls.

## What was validated

All checks are in `test/test_natconv_gnf.jl` (CPU, Float64), run on the date of
this page; all 41 assertions passed.

| Check (testset) | Measured | Gate |
|---|---|---|
| ``n = 1`` against the Newtonian driver `run_natural_convection_2d`, N = 64, Ra = 10³ ("Newtonian limit and De Vahl Davis") | Nu = 1.1423 for both, relative gap 3.9e-16 | gap < 1 % |
| Nu against De Vahl Davis (1983), Nu = 1.118, same case | Nu = 1.1423, i.e. +2.2 % | < 10 % (same frozen gate as `test_thermal.jl`) |
| Nu against the power-law index, N = 32, Ra = 10⁴, Pr = 0.71 ("QUALITATIVE literature trend") | Nu = 3.496 / 2.492 / 1.822 for ``n`` = 0.8 / 1.0 / 1.2 | Nu(0.8) > Nu(1.0) > Nu(1.2); Nu drift < 1e-5 over three successive 1000-step checks; clamped fraction < 1 % (measured 0.0) |
| Turan et al. (2011), Ra = 10³, Pr = 10, ``n`` = 0.6, N = 48 ("Turan et al. (2011) power-law point, CPU") | Nu = 2.124, +2.6 % against 2.07 | < 4 %: the reference's own uncertainty (figure reading ± 0.07, plus their mesh uncertainty ≤ 0.35 %); converged; clamped fraction 0 |
| Carreau-Yasuda, N = 8, 5 steps ("Carreau-Yasuda cavity API") | finite fields, viscosity between ``\eta_\infty`` and ``\eta_0`` | smoke test only |
| `.krk` case reaches the driver ("Parsed thermal rheology reaches run_simulation") | `power_law` block routed, 5 steps, finite fields | smoke test only |

Reading of the table:

- The ``n = 1`` row shows that the new driver and the Newtonian driver solve the
  same discrete problem; it does not validate the non-Newtonian physics.
- The De Vahl Davis row is a 2.2 % agreement on a 64² mesh at Ra = 10³, of which
  about 1.6 % is the N/(N − 1) thermal-wall offset described above (estimated,
  not separated by a run). The Newtonian benchmark page reaches 0.79 % on a 192²
  mesh for the Newtonian driver.
- The trend row says that shear-thinning raises the heat transfer and
  shear-thickening lowers it, as expected physically; Pr = 0.71 has no published
  power-law counterpart.
- The Turan row is the one quantitative power-law check. Its 4 % gate is set by
  the precision of the published value, not by Kraken's error.

## Not yet verified

- **No GPU run.** The comparison with Turan et al. (2011) on N = 64, 128 and 256
  (the five cases of the table above, CUDA Float64) is prepared in
  `benchmarks/validation/natconv_gnf/run_turan_comparison.jl` and has not been
  executed. The older qualitative sweep at Pr = 0.71 (`run_natconv_gnf.jl`) is
  not run either.
- **Grid convergence** is not shown: the CPU comparison stops at N = 64.
- **Ra = 10⁵ and 10⁶** are not compared, nor Table 2 of the paper (Ra = 10⁶,
  Pr = 100, the only tabulated Nu values), which needs a thermal relaxation time
  within 0.01 of 1/2 and about 10⁷ steps at N = 512.
- **Pr dependence** is not checked: all power-law comparisons use Pr = 10.
- **Float32** was not tested.
- The thermal-wall offset of order 1/N is documented, not corrected.

## References

de Vahl Davis, G. (1983), *Natural convection of air in a square cavity: a bench
mark numerical solution*, Int. J. Numer. Methods Fluids 3, 249-264,
DOI 10.1002/fld.1650030305.

Turan, O., Sachdeva, A., Chakraborty, N., Poole, R.J. (2011), *Laminar natural
convection of power-law fluids in a square enclosure with differentially heated
side walls subjected to constant temperatures*, J. Non-Newtonian Fluid Mech. 166,
1049-1063, DOI 10.1016/j.jnnfm.2011.06.003.
