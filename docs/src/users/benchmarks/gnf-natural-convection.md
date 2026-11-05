# Generalised Newtonian natural convection

Natural convection of a generalised Newtonian fluid (viscosity depending on the
local shear rate) in a differentially heated square cavity: hot west wall, cold east
wall, adiabatic top and bottom, no-slip walls. Driver
`run_natural_convection_gnf_2d`, which couples the thermal double-distribution
D2Q9 solver of the [Newtonian cavity](thermal-natural-convection.md) to the
shear-rate-dependent collision of the rheology chapter of the theory section
([rheology theory page](../../theory/14_rheology.md)).

> **Status: CPU-validated, N ≤ 96; GPU run at N = 64, 128 and 256.** The
> Newtonian limit, one De Vahl Davis point and two power-law points of Turan et
> al. (2011) are checked by the unit test. Eight power-law cases of that paper
> were run on CPU at N = 32 to 96, and the six Pr = 10 cases on one A100 at
> N = 64, 128 and 256 (see [the GPU comparison](#Comparison-with-Turan-et-al.-(2011),-GPU)).
> One of the six, Ra = 10⁴ with ``n`` = 1.8, is not converged.

!!! warning "Turan et al. comparisons measured before v0.7"
    The two Turan et al. (2011) tables, the `nu0` check of the correspondence
    section and the readings that follow each table were measured before
    v0.7: Nu was averaged over rows `2 … N − 1`, and the adiabatic top and
    bottom walls bounced back the diagonal thermal populations. Since v0.7,
    Nu is averaged over all rows and the adiabatic walls reflect these
    populations specularly (see the
    [thermal theory page](../../theory/08_thermal_ddf.md)). Until they are
    re-measured (the GPU ones on GPU), treat them as pre-v0.7 values. On CPU at N = 48,
    the change moves Nu from 1.9933 to 1.9843 at Ra = 10³, ``n`` = 0.6, and
    the Ra = 10⁴ extrapolated value from 5.393 to 5.3765. The code outputs and
    the [What was validated](#What-was-validated) table are re-measured for
    v0.7.

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
measured on CPU at N = 64, Ra = 10³, ``n`` = 0.6: Nu = 1.9931 (`nu0` = 0.3) and
1.9914 (`nu0` = 0.1), a 0.08 % difference. This holds only while the relaxation
time stays moderate: at Ra = 10⁴, ``n`` = 1.4, `nu0` = 0.3 lets the viscosity near
the walls reach 1.69 (relaxation time 5.6) and gives Nu = 1.431, against 1.368
with `nu0` = 0.1 (see the comparison below).

What differs from the paper, apart from the method (they solve the steady
equations with a finite-volume code on a 100 × 100 non-uniform mesh):

- the mirror image (hot wall on the other side), which leaves the mean Nu unchanged;
- the buoyancy reference temperature: Kraken uses ``(T_\text{hot}+T_\text{cold})/2``,
  the paper ``T_C``. The difference is a uniform body force, balanced by pressure;
- `PowerLaw` clips the viscosity to `[nu_min, nu_max]`, the paper does not. The
  result reports `clamped_fraction`; it was 0 in the CPU runs below, but not in every GPU run:
  in `benchmarks/validation/natconv_gnf/results/turan_comparison.csv` it is 6e-5 at
  Ra = 10³, n = 0.6, N = 256, 1.2e-4 to 9.8e-4 at Ra = 10³, n = 1.8, and 0.17 % to 8 % at
  Ra = 10⁴, n = 1.8 (N = 64 to 256, runs that did not converge);
- the fixed-temperature walls are half-way anti-bounce-back walls, half a lattice
  spacing outside the first and last nodes, at the same place as the no-slip
  walls; the adiabatic walls reflect the thermal populations specularly; Ra, Pr
  and Nu use the cavity width ``H = N``, and Nu is the mean over all rows
  `1 … N` of a second-order one-sided wall gradient. These are the walls and the
  Nu formula of the Newtonian driver;
- the paper's power-law data are for Pr ≥ 10; Pr = 0.71 is used here only for the
  Newtonian benchmark.

## Run it on CPU

This runs in about two seconds per index on a laptop CPU (N = 32, Ra = 10⁴,
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

Output measured on 2026-11-04 (v0.7, CPU, Float64):

```
n=0.8 Nu=3.013 steps=8000 converged=true clamped=0.0
n=1.0 Nu=2.239 steps=8000 converged=true clamped=0.0
n=1.2 Nu=1.689 steps=8000 converged=true clamped=0.0
```

Pr = 0.71 lies outside the power-law data of Turan et al. (2011), so these values
have no published counterpart. For a case of the paper, use Pr = 10 or more:

```julia
Pr, nu0, N = 10.0, 0.3, 48
model = cavity_powerlaw(0.6; nu0, alpha=nu0/Pr, H=N, nu_min=1e-3, nu_max=2.0)
r = run_natural_convection_gnf_2d(; N, Ra=1e3, Pr, nu0, rheology=model,
    max_steps=60000, check_every=1000, convergence_rtol=1e-5, min_steps=5000)
r.Nu   # 1.984 measured (v0.7); the paper gives 2.07 ± 0.07 (Fig. 9a), see below
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
CPU and returned `Nu = 1.043` after 3000 steps (v0.7). In this route `K` is the value you
write in the block; the `cavity_powerlaw` scaling is not applied, so Ra and Pr
follow the Turan definitions only if you write ``K = \nu_0(\alpha/H^2)^{1-n}``
yourself.

## Comparison with Turan et al. (2011), CPU

Pr = 10. Paper values are read from their figures (no table gives them), with the
reading uncertainty in brackets; page numbers are journal pages. Kraken: CPU,
Float64, `nu_min = 1e-3`, `nu_max = 2.0`, stopping after three successive
1000-step relative Nu changes below 1e-6 (1e-5 for N = 32 and 48 at Ra = 10⁴,
the settings of the unit test). Every run listed converged and had no node at a
viscosity cutoff, unless stated (the CSV above lists the GPU runs where some nodes sit at
a cutoff).

| Ra | ``n`` | Paper Nu (source) | N | `nu0` | Kraken Nu | Kraken − paper |
|---|---|---|---|---|---|---|
| 10³ | 0.6 | 2.07 (± 0.07, Fig. 9a, p. 1059) | 48 | 0.3 | 1.9933 | −3.7 % |
| | | | 64 | 0.3 | 1.9931 | −3.7 % |
| | | | 64 | 0.1 | 1.9914 | −3.8 % |
| | | | 96 | 0.3 | 1.9923 | −3.8 % |
| 10³ | 1.0 | 1.14 (± 0.02, Fig. 9b, p. 1059) | 64 | 0.3 | 1.1201 | −1.7 % |
| 10³ | 1.8 | 1.01 (± 0.005, Fig. 9c, p. 1059) | 64 | 0.05 | 1.0088 | −0.1 % |
| 10⁴ | 0.6 | 5.40 (± 0.02, Fig. 10, p. 1060) | 32 | 0.3 | 5.2934 | −2.0 % |
| | | | 48 | 0.3 | 5.3487 | −1.0 % |
| | | | 64 | 0.3 | 5.3678 | −0.6 % |
| | | | 96 | 0.3 | 5.3796 | −0.4 % |
| | | | extrapolated (64, 96) | | 5.389 | −0.2 % |
| 10⁴ | 0.8 | 3.38 (± 0.02, Fig. 10, p. 1060) | 64 | 0.3 | 3.3564 | −0.7 % |
| 10⁴ | 1.0 | 2.29 (± 0.015, Figs. 10–11) | 64 | 0.3 | 2.2980 | +0.3 % |
| 10⁴ | 1.4 | 1.36 (± 0.01, Fig. 11, p. 1061) | 64 | 0.3 | 1.4305 | +5.2 % |
| | | | 64 | 0.1 | 1.3676 | +0.6 % |
| 10⁴ | 1.8 | 1.10 (± 0.01, Fig. 11, p. 1061) | 64 | 0.1 | 1.113, not converged, 22 % of nodes at `nu_max` | not usable |
| | | | 64 | 0.05 | 1.088, not converged (see below) | −1.1 % |

The extrapolated row assumes an error in 1/N²: at fixed `nu0` the grid error and
the compressibility error of the lattice both scale that way. The measured ratio
of successive differences (N = 32, 48, 64) is 2.90, against 2.86 for an exact
1/N² error. The pairs (32, 48) and (48, 64) extrapolate to 5.393 and 5.392.

Reading of the table:

- At Ra = 10⁴, for ``n`` = 0.6 to 1.4 (``n`` = 1.4 with `nu0` = 0.1), Kraken
  agrees with the paper within 0.7 % on a 64² grid, and within 0.2 % after
  extrapolation for ``n`` = 0.6. The same
  mapping of Ra and Pr is used at every point, so this rules out a mismatch of
  definitions between Kraken and the paper.
- At Ra = 10³, ``n`` = 0.6, Kraken is 3.7 % below the paper value, just outside
  its reading uncertainty. This gap is not a grid error: Nu changes by 0.05 %
  from N = 48 to N = 96 and by 0.08 % when `nu0` is divided by 3. Its origin is
  not measured. What is known: Fig. 9a has a 0 to 40 axis, so its reading is
  coarse; the same figure gives 5.44 at Ra = 10⁴ where Fig. 10 gives 5.40; and
  for ``n`` = 1 it gives 1.14, 2 % above the Newtonian value of Kraken at Pr = 10
  (1.120), whereas changing Pr from 0.71 to 10 moves Kraken's Nu by only 0.14 %
  and De Vahl Davis gives 1.117 at Pr = 0.71. The gap is therefore likely a
  reading or plotting offset of Fig. 9 at Ra = 10³ (inferred, not proved). A real
  difference between the two solutions at Ra = 10³ cannot be excluded without
  tabulated values.
- For a shear-thickening fluid the viscosity is largest at the walls, where the
  shear is largest, and the BGK wall error grows with the relaxation time. With
  `nu0` = 0.3, ``n`` = 1.4 reaches a wall viscosity of 1.69 and Nu is 5.2 % high;
  with `nu0` = 0.1 the viscosity stays below 0.58 and the gap falls to 0.6 %.
  For ``n`` = 1.8, `nu0` = 0.3 and 0.1 put 84 % and 22 % of the nodes at the
  `nu_max` cutoff at Ra = 10⁴ and give no usable result. With `nu0` = 0.05,
  Ra = 10³ converges 0.1 % below the paper (0.1 % of the nodes at `nu_min`),
  while Ra = 10⁴ is still drifting after 1.5 million steps (1.8 diffusion times,
  last relative change 1.8e-5 per 1000 steps), with 0.1 % of the nodes at
  `nu_max`, and stands 1.1 % below the paper.

## Comparison with Turan et al. (2011), GPU

Same cases and paper values as above, run on one A100 (CUDA, Float64, Julia
1.12.5) at N = 64, 128 and 256 from commit 1c70037 (tag `archive/gnf-thermal-turan-gpu`); the 20-minute job is archived
in `benchmarks/validation/natconv_gnf/results/`. `nu0` is 0.3, except 0.05 for
``n`` = 1.8. Each cell is Kraken Nu and its signed gap to the paper value; the
paper values are figure readings with the uncertainty given in the CPU table.

| Ra | ``n`` | Paper Nu | N = 64 | N = 128 | N = 256 | Converged |
|---|---|---|---|---|---|---|
| 10³ | 0.6 | 2.07 | 1.9931 (−3.7 %) | 1.9917 (−3.8 %) | 1.9905 (−3.8 %) | yes |
| 10³ | 1.0 | 1.14 | 1.1201 (−1.7 %) | 1.1186 (−1.9 %) | 1.1181 (−1.9 %) | yes |
| 10³ | 1.8 | 1.01 | 1.0089 (−0.1 %) | 1.0087 (−0.1 %) | 1.0087 (−0.1 %) | yes |
| 10⁴ | 0.6 | 5.40 | 5.3678 (−0.6 %) | 5.3828 (−0.3 %) | 5.3840 (−0.3 %) | yes |
| 10⁴ | 1.0 | 2.29 | 2.2980 (+0.4 %) | 2.2824 (−0.3 %) | 2.2774 (−0.5 %) | yes |
| 10⁴ | 1.8 | 1.10 | 1.0880 (−1.1 %) | 1.0714 (−2.6 %) | 1.0645 (−3.2 %) | no |

- Ra = 10³, ``n`` = 0.6: 1.993 / 1.992 / 1.991 against 2.07 (−3.7 % to −3.8 %).
  Nu changes by 0.13 % from N = 64 to 256, as little as on CPU, so the gap is not a grid
  error. The paper value is read from a figure (± 0.07).
- Ra = 10⁴, ``n`` = 0.6 and ``n`` = 1.0: within 0.6 % of the paper at every N.
- Ra = 10³, ``n`` = 1.8: within 0.13 %.
- Ra = 10⁴, ``n`` = 1.8 is not a validated point. It stopped at the step limit
  without converging (last Nu change 4.0e-5 at N = 64, 4.2e-6 at N = 128,
  1.7e-6 at N = 256), and the viscosity sits at its upper bound `nu_max` = 2.0 on
  0.2 %, 5.9 % and 8.0 % of the nodes. The table values are listed only for
  completeness.
- At N = 64 the GPU and CPU values agree to 2e-5 or better in Nu (for example
  1.99308 on both for Ra = 10³, ``n`` = 0.6; the largest difference is 1.00883
  on CPU against 1.00885 on GPU for Ra = 10³, ``n`` = 1.8, which stopped at
  different step counts). The CPU values are quoted to five decimals.
- The job script returns status 1 on purpose: the Ra = 10⁴, ``n`` = 1.8 rows are
  not converged and the 3D refined check fails. The job ended with exit
  status 1 for these two reasons, after writing all rows.

The 3D refined thermal check (`benchmarks/validate_thermal_ref_3d_gpu.jl`)
reproduces on GPU the defect seen on CPU (conduction error 3.5e-2 against a 0.02
gate; refined coarse Nu 7.12 against 1.085); grid refinement is under development
and not validated (issue #98).

## What was validated

All checks are in `test/test_natconv_gnf.jl` (CPU, Float64), run on 2026-11-04
with v0.7; all 52 assertions passed (83 s).

| Check (testset) | Measured | Gate |
|---|---|---|
| ``n = 1`` against the Newtonian driver `run_natural_convection_2d`, N = 64, Ra = 10³ ("Newtonian limit and De Vahl Davis") | Nu = 1.1175 for both, relative gap 0 | gap < 1e-6 |
| Nu against De Vahl Davis (1983), hot-wall Nu₀ = 1.117 (cavity average 1.118; the test gate uses 1.118), same case | +0.05 % against 1.117 (−0.04 % against 1.118) | < 0.5 % (same frozen gate as `test_thermal.jl`) |
| Nu against the power-law index, N = 32, Ra = 10⁴, Pr = 0.71 ("QUALITATIVE literature trend") | Nu = 3.013 / 2.239 / 1.689 for ``n`` = 0.8 / 1.0 / 1.2 | Nu(0.8) > Nu(1.0) > Nu(1.2); Nu drift < 1e-5 over three successive 1000-step checks; clamped fraction < 1 % (measured 0.0) |
| Turan et al. (2011), Ra = 10⁴, Pr = 10, ``n`` = 0.6, N = 32 and 48 extrapolated in 1/N² ("Turan et al. (2011) power-law points, CPU") | Nu = 5.3765, −0.44 % against 5.40 | < 1 %: the reference's own uncertainty (figure reading ± 0.02, i.e. 0.37 %, plus their mesh uncertainty ≤ 0.35 %), rounded up; converged; clamped fraction 0 |
| Same testset, Ra = 10³, ``n`` = 0.6, N = 48 | Nu = 1.9843 (−4.1 % against 2.07) | within 0.1 % of Kraken's own converged value 1.98434 (regression gate); the gap to the paper is documented above, not gated |
| Carreau-Yasuda, N = 8, 5 steps ("Carreau-Yasuda cavity API") | finite fields, viscosity between ``\eta_\infty`` and ``\eta_0`` | smoke test only |
| `.krk` case reaches the driver ("Parsed thermal rheology reaches run_simulation") | `power_law` block routed, 5 steps, finite fields | smoke test only |

Reading of the table:

- The ``n = 1`` row shows that the new driver and the Newtonian driver solve the
  same discrete problem; it does not validate the non-Newtonian physics.
- The De Vahl Davis row is a 0.04 % agreement on a 64² mesh at Ra = 10³; at
  Ra = 10⁴ the same comparison (not in the test) gives 2.2432 against 2.238,
  +0.23 %.
- The trend row says that shear-thinning raises the heat transfer and
  shear-thickening lowers it, as expected physically; Pr = 0.71 has no published
  power-law counterpart.
- The two Turan rows split the comparison on purpose. At Ra = 10⁴ the figure
  is read to 0.4 % and Kraken agrees with it within 0.5 %, so the test gates
  Kraken against the paper with a 1 % gate set by the paper's precision. At
  Ra = 10³ a gate against 2.07 would have to be at least 5 %, which would mostly
  test the reading of a figure and would let a 4 % change of Kraken's solution
  pass unseen; the test therefore freezes Kraken's own converged value, so that
  any change of the discrete solution fails, and the 4.1 % gap stays
  documented.

## Not yet verified

- **GPU coverage.** The six cases were run on one A100 only. Ra = 10⁴ with
  ``n`` = 1.8 did not converge at any N, so it has no validated value. The older
  qualitative sweep at Pr = 0.71 (`run_natconv_gnf.jl`) is not run.
- **The Ra = 10³ gap** of −3.7 % for ``n`` = 0.6 is explained by inference only.
- **Shear-thickening fluids** need a small `nu0`; ``n`` = 1.8 at Ra = 10⁴ did
  not reach a steady state on CPU (N = 64) or on GPU (N = 64 to 256).
- **Ra = 10⁵ and 10⁶** are not compared, nor Table 2 of the paper (Ra = 10⁶,
  Pr = 100, the only tabulated Nu values), which needs a thermal relaxation time
  within 0.01 of 1/2 and about 10⁷ steps at N = 512.
- **Pr dependence** is not checked: all power-law comparisons use Pr = 10.
- **Float32** was not tested.

## References

de Vahl Davis, G. (1983), *Natural convection of air in a square cavity: a bench
mark numerical solution*, Int. J. Numer. Methods Fluids 3, 249-264,
DOI 10.1002/fld.1650030305.

Turan, O., Sachdeva, A., Chakraborty, N., Poole, R.J. (2011), *Laminar natural
convection of power-law fluids in a square enclosure with differentially heated
side walls subjected to constant temperatures*, J. Non-Newtonian Fluid Mech. 166,
1049-1063, DOI 10.1016/j.jnnfm.2011.06.003.
