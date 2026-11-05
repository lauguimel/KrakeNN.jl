# Thermal Natural Convection

Steady 2D natural convection in a differentially-heated square cavity at
Ra = 10³, 10⁴, 10⁵ (Pr = 0.71). Kraken's double-distribution thermal LBM matches the
canonical **de Vahl Davis (1983)** benchmark to **Nu err < 1 %** at every Ra (velocity
extrema < 1 %), cross-checked against OpenFOAM `buoyantBoussinesqSimpleFoam`.

de Vahl Davis 1983 (black markers), Kraken LBM (blue squares), OpenFOAM (orange
triangles); the OF Ra = 10⁴ point is omitted as under-converged.

![Nu vs Ra](thermal-natural-convection-nu.png)

The Ra = 10⁵ convergence ladder shows the monotone descent of the Nu error with mesh
and the crossing of the 1 % gate at N ≈ 384.

![Nu convergence at Ra=1e5](thermal-natural-convection-convergence.png)

## Result

| Ra   | Backend   | Mesh | Nu (Kraken) | Nu (dVD) | Nu err      |
|------|-----------|------|-------------|----------|-------------|
| 10³  | CPU F64   | 192² | 1.126       | 1.117    | **+0.79 %** |
| 10⁴  | Metal F32 | 320² | 2.259       | 2.238    | **+0.93 %** |
| 10⁵  | Metal F32 | 384² | 4.544       | 4.509    | **+0.79 %** |

!!! warning "Measured before the thermal walls moved (issue #97)"
    The rows above, the convergence plot and the 3D numbers below were measured
    when the thermal Dirichlet walls sat on the first and last nodes (cavity width
    `N − 1`) while the no-slip walls sat half a spacing outside them (width `N`).
    The thermal walls are now half-way too, so both cavities have width `N`. They
    also predate v0.7, in which the adiabatic walls reflect the thermal
    populations specularly instead of bouncing the diagonals back and `Nu` is
    averaged over all rows. Until they are re-measured on GPU, treat them
    as pre-v0.7 values; see [Wall position](#wall-position-re-measured-on-cpu) for the
    numbers that have been re-run.

Both mid-plane velocity extrema (`u_max*`, `v_max*`, normalised by `α/L`) are under
1 % at every Ra. OpenFOAM independently corroborates de Vahl Davis at Ra = 10³
(−0.56 %) and Ra = 10⁵ (−0.60 %) — two CFD codes (FVM SIMPLE and LBM) bracketing the
benchmark from opposite signs. Full data:
`benchmarks/results/repro/thermal/kraken_natconv_results.csv`.

**Per-Ra precision recipe (a real finding).** The thermal boundary layer thins as
Ra^(1/4), so the resolution needed to clear the 1 % gate grows with Ra: at Ra = 10⁵
the Nu error descends monotonically and only crosses below 1 % at **N ≈ 384** (see the
convergence plot). At the opposite end, **Ra = 10³ must run in Float64**: its buoyancy
force `β g ∝ 1/N³ ≈ 10⁻⁹` LU sinks below the Float32 epsilon for `N ≥ 320` and the
convection cell collapses toward conduction. Where both precisions are valid
(Ra = 10⁴/10⁵ at N=128) **Metal-F32 and CPU-F64 agree on Nu to 0.07–0.09 %**.

## Wall position (re-measured on CPU)

The hot and cold walls use the half-way anti-bounce-back rule: each wall sits half
a lattice spacing outside the first (last) node, exactly where the no-slip
bounce-back wall of the flow sits. The cavity width is `H = N` for the flow, the
temperature and the Rayleigh number. `Nu` is the hot-wall average of the
second-order one-sided gradient through the wall value and the first two nodes,
`∂T/∂n = (−8 T_w + 9 T_1 − T_2) / (3 Δx)`, over all rows `1 … N` (each row
covers one spacing of the wall). The adiabatic top and bottom walls reflect the
thermal populations specularly: zero heat flux through the wall, flux along it
kept (see the [thermal theory page](../../theory/08_thermal_ddf.md)).

Measured 2026-11-04 with `run_natural_convection_2d` (v0.7), CPU Float64,
`ν = 0.05`, `Pr = 0.71`, run to a steady `Nu`: at `N = 64`, 30 000 steps at
`Ra = 10³` and 60 000 at `Ra = 10⁴`; doubling the step count moves `Nu` by
`2 × 10⁻⁶` and less than `10⁻⁷`. The pure-conduction row uses `N = 32` and
40 000 steps; the `N = 96` rows ran 70 000 and 135 000 steps without a
step-doubling check. The middle column was measured on 2026-11-03, before v0.7,
with `Nu` averaged over rows `2 … N − 1` and the diagonal thermal populations
bounced back at the adiabatic walls.

| Case            | N  | Nu, on-node walls | Nu, half-way walls, before v0.7 | Nu, v0.7          | Reference         |
|-----------------|----|-------------------|---------------------------------|-------------------|-------------------|
| Ra = 0          | 32 | 1.0351            | 1.0000049                       | 1 (`Nu − 1 = 1.2 × 10⁻¹²`) | 1 (conduction)    |
| Ra = 10³        | 64 | 1.1423 (+2.26 %)  | 1.1185 (+0.13 %)                | 1.1175 (+0.05 %)  | dVD Nu₀ = 1.117   |
| Ra = 10⁴        | 64 | 2.3585 (+5.39 %)  | 2.2535 (+0.69 %)                | 2.2432 (+0.23 %)  | dVD Nu₀ = 2.238   |
| Ra = 10³        | 96 | not measured      | 1.1183 (+0.12 %)                | 1.1177 (+0.06 %)  | dVD Nu₀ = 1.117   |
| Ra = 10⁴        | 96 | not measured      | 2.2509 (+0.58 %)                | 2.2440 (+0.27 %)  | dVD Nu₀ = 2.238   |

With the walls on the nodes, the thermal cavity was one spacing narrower than the
flow cavity, an error of order `1/N` on `Nu`. Moving the walls removes it; at
`N = 64` both Rayleigh numbers are now within 1 % of de Vahl Davis. The v0.7
change (specular adiabatic walls, all rows in the average) lowers `Nu` by 0.09 %
at `Ra = 10³` and 0.46 % at `Ra = 10⁴` at `N = 64`. At `Ra = 10³` the hot-wall
`Nu` on `N = 16, 32, 64` (1.11379, 1.11677, 1.11753) converges with an observed
order of 1.97 (test `test/analytical/thermal_adiabatic_walls.jl`).

In pure conduction the half-way Dirichlet walls and the specular adiabatic walls
give the linear profile on every row, the wall rows included, and `Nu = 1` to
round-off: the largest deviation is `6.9 × 10⁻¹⁴` at `N = 16` and
`2.5 × 10⁻¹³` at `N = 32` in 2D, `3.1 × 10⁻¹⁵` for the 3D box at `N = 12`
(same test). Before v0.7, bouncing the diagonal populations back at the
adiabatic walls bent the profile near the top and bottom walls (row 1 off by
`2 × 10⁻³` at `N = 16`).

## Methodology

**Kraken (thermal LBM, double-distribution D2Q9).** D2Q9 BGK flow + a second D2Q9
distribution advecting temperature, with Boussinesq buoyancy `F = β g (T − T₀)` (the code applies no density factor)
coupled back into the flow. Hot left wall (`T = 1`), cold right wall (`T = 0`),
adiabatic top/bottom (specular reflection of the thermal populations), all walls
no-slip (halfway bounce-back), hot and cold walls
half-way anti-bounce-back (see [Wall position](#wall-position-re-measured-on-cpu)).
Driver `run_natural_convection_2d`, iterated to a steady Nusselt residual.

**OpenFOAM cross-check (`buoyantBoussinesqSimpleFoam`, FVM).** v2512 (local Docker),
steady SIMPLE, laminar; Ra set by fixing `ν = 10⁻³`, `α = ν/Pr`, `β ΔT = 1`, varying
`g`. Nu from `snGrad(T)` on the hot patch. Meshes 128² (Ra = 10³), 192² (Ra = 10⁵).

**Reference.** de Vahl Davis, G. (1983), *Natural convection of air in a square cavity:
a bench mark numerical solution*, Int. J. Numer. Methods Fluids 3, 249–264 — Pr = 0.71,
hot-wall Nu₀ = 1.117 / 2.238 / 4.509 at Ra = 10³ / 10⁴ / 10⁵ (Table V; his cavity average Nu̅ is 1.118 / 2.243 / 4.519). Kraken computes a hot-wall mean, so this page compares with Nu₀.

## 3D — cubic cavity

The same double-distribution thermal LBM extends to a **cubic** cavity (D3Q19 flow +
D3Q19 temperature, driver `run_natural_convection_3d`), referenced to Tric et al. (2000)
and cross-checked against Fusegi et al. (1991). A single-mesh exploratory run at 96³
(Metal F32) gives Nu errors of **+1.45 % / +3.36 % / +6.29 %** at Ra = 10³ / 10⁴ / 10⁵
vs Tric — the boundary layer thins as Ra^(1/4), so the residual grows with Ra as
expected. These three points are tabulated in
`benchmarks/results/repro/thermal/kraken_natconv_3d_results.csv`.

> **Status: preliminary, not yet a quantitative benchmark.** Only the 96³ point above is
> bundled. A full mesh-convergence ladder (N = 48 / 64 / 96 / 128) and a Float32-vs-Float64
> cross-check are not yet reproducible from this repo; reaching the < 2 % gate at Ra = 10⁵
> is expected to require N ≥ 128 (an HPC-class run). Consistent with the
> [validation matrix](validation-matrix.md), the quantitative 3D thermal benchmark is a
> **future item**; the 3D driver is wired and smoke-validated today.

References: Tric, E., Labrosse,
G. & Betrouni, M. (2000), Int. J. Heat Mass Transfer 43, 4043–4056; Fusegi, T., Hyun,
J.M., Kuwahara, K. & Farouk, B. (1991), Int. J. Heat Mass Transfer 34, 1543–1557.

## Caveats

- **OpenFOAM Ra = 10⁴ is not converged** in the iteration budget (Nu ≈ 1.55 at 6000
  iterations, still drifting; true steady state needs ~15–20 k). It is flagged
  `_UNCONVERGED` in the CSV and omitted from the plot — a slow-SIMPLE artifact of the
  reference, not a Kraken issue; Kraken's Ra = 10⁴ stands on its independent dVD
  validation (+0.93 %).
- **Low-Ra precision floor.** Ra = 10³ requires Float64 (the `β g ∝ 1/N³` body force
  underflows F32 for `N ≥ 320`). Use CPU F64 at low Ra; reserve Metal F32 for Ra ≥ 10⁴.
- **Rayleigh–Bénard is qualitative only.** The bottom-heated, periodic-side demo
  (`examples/rayleigh_benard.krk`) develops convection rolls above Ra_c ≈ 1708 but
  asserts no quantitative roll-cell reference.

## Reproduce

```julia
using Kraken
run_simulation("examples/natural_convection.krk")      # differentially-heated cavity
run_simulation("examples/rayleigh_benard.krk")          # qualitative Rayleigh–Bénard
run_simulation("examples/natural_convection_3d.krk")    # cubic 3D cavity
```

The presets set `ν`, `Pr`, `Ra`, the thermal module and the cavity walls; set the
backend and resolution per the precision recipe to reproduce a specific table entry
(CPU F64 at N=192 for Ra=10³; Metal F32 at N=320/384 for Ra=10⁴/10⁵). Plots are
regenerated with `bench/scratch/plot_thermal_natconv.jl` (CairoMakie, `--project=docs`).
