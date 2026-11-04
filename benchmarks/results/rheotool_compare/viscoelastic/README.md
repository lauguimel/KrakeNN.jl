# Viscoelastic reference-solver comparison — confined-cylinder drag (Oldroyd-B)

§3 release-mandate artifact for the **viscoelastic** module: a literal comparison of
Kraken's Oldroyd-B solver against the viscoelastic reference solver **RheoTool**
(rheoFoam, OpenFOAM-9, log-conformation) on the canonical confined-cylinder problem —
the standard High-Weissenberg-Number-Problem (HWNP) testbed of Alves, Oliveira & Pinho
(2001).

## Case

Confined cylinder of radius `R` on the channel centreline, blockage ratio `D/H = 0.5`
(half-height `2R`), upstream/downstream length `15R`. Oldroyd-B fluid, `Re = 1`, solvent
fraction `β = η_s/η₀ = 0.59` (the Boger-fluid value used throughout the cylinder
literature). Diffusive scaling: fixed lattice viscosity `ν_total = 0.15`, `τ = 0.95`;
`u_mean = ν_total·Re/R`, `λ = Wi·R/u_mean`. Channel walls half-way bounce-back; cylinder wall linearly interpolated (Bouzidi) bounce-back with the analytic cut-link distance `q` (solver option `wall_bc = :halfwayBB`, which applies the interpolation).

| Solver | Method | Grid | Notes |
|--------|--------|------|-------|
| **Kraken** | TRT-LBM + log-conformation finite-volume polymer transport | `R = 50` (cylinder diameter `100` LU), `±15R` channel | MUSCL–superbee advection of the conformation field, interpolated bounce-back cylinder wall, v0.3 cut-link momentum-exchange drag. Diffusive scaling `ν_total = 0.15` (`τ = 0.95`), `300 000` steps. Backend **CUDA Float64 (Aqua A100)**, job 22199679; re-measured with the same settings in jobs 26763246/47/48 (2026-10-14). |
| **RheoTool `rheoFoam`** | FVM, log-conformation Oldroyd-BLog | `Cylinder/Oldroyd-BLog` tutorial mesh | OpenFOAM-9 base. The archived `rheoFoam` logs show a single process and end time `t = 6` or `t = 10` (see the provenance note on the benchmark page); the logs of the `β = 0.59` runs behind the stored `Cd` are not archived. `β = 0.59`, `Re = 1`, `D/H = 0.5`. |
| **Alves–Oliveira–Pinho (2001)** | published reference | finite-volume high-resolution | the canonical cylinder HWNP benchmark family. |

## Files

- `kraken_cd_vs_wi.csv` — Kraken drag at `Wi ∈ {0.1, 0.5, 1.0}`, R = 50, β = 0.59.
  Columns: `Wi, Cd, Cd_s, Cd_p`. `Cd` is the total drag coefficient (`Cd_kraken`,
  source column 26); `Cd_s`/`Cd_p` are the raw solvent/polymer wall-ring integrator
  diagnostics (source columns 27/28), reported verbatim. Source: post-fix re-measurement
  (2026-10-14), `aqua:~/kraken_guo_remeasure/out/<src>_<tag>_R50_<jobid>/cyl_bigsweep_v2_*.csv`
  (not in the repository). Before 2026-10-14 the source was
  `bench/viscoelastic_logfv/v03_cutlink_reval_22199679/reval_R50/cyl_bigsweep_v2_*.csv`.
- `rheotool_cd_vs_wi.csv` — RheoTool reference drag at the same `Wi`. Columns: `Wi, Cd`.
- `error_norms.csv` — per `Wi`, the relative error `(Cd_kraken − Cd_rheotool)/Cd_rheotool`
  (and its absolute percentage). This is the §3 integrated-quantity gate.
- `cylinder_oldroyd_b.krk` — the Kraken case file (smoke wiring; the production driver
  rebuilds the cylinder channel from `R`, `L_up`, `L_down`). Running it with
  `run_simulation("cylinder_oldroyd_b.krk")` is only a smoke test (240×32, 20 steps): it
  does not reproduce the table. The table values need the `R = 50`, 300 000-step GPU run.
- `comparison.png` — Cd vs Wi: Kraken filled markers + connecting line, RheoTool dashed
  reference line + open squares, per-point relative error annotated. Dark Documenter
  theme. Also copied to `docs/src/users/benchmarks/viscoelastic-cylinder.png`.
  Regenerated on 2026-10-19 from the post-fix `kraken_cd_vs_wi.csv`.
- `plot.py` — **self-contained reproducer**: reads the two `*_cd_vs_wi.csv` files in this
  directory and regenerates `comparison.png` (csv + matplotlib + seaborn; LaTeX if a
  system `latex` is present, else mathtext — no external paths). Run
  `conda run -n kraken-v0-3-figures python plot.py`.
- `m8_refs/` — the prior dev-viscoelastic "M8" study CSVs reproduced by this revalidation
  (`matrix_halfwayBB_R{10,30,50}.csv` = Cd vs Wi per R; `beta_R50.csv` = β-sweep;
  `staircase_R{30,50}.csv` = high-Wi stability envelope; `N1_comparison.csv` = wake-N1
  Kraken/RheoTool ratio). These are pre-fix values, kept unchanged as the record of that
  study. The v0.3 re-validation (job 22199679) reproduced the M8 `matrix_halfwayBB_R50`
  Cd to 4–5 significant figures, both with the pre-fix forcing (see the note below).

## Headline result — Cd vs Wi (R = 50, β = 0.59)

| Wi  | Kraken Cd | Kraken Cd_p | RheoTool Cd | rel. error |
|-----|-----------|-------------|-------------|------------|
| 0.1 | 129.9514  | 16.096      | 130.43      | **−0.37 %** |
| 0.5 | 118.9187  | 14.843      | 119.71      | **−0.66 %** |
| 1.0 | 119.4912  | 14.182      | 120.40      | **−0.75 %** |

All three clear the strict **< 1 %** integrated-quantity gate. Kraken reproduces the
characteristic elastic signature reported across the cylinder literature: a shallow drag
**minimum near `Wi ≈ 0.5`** followed by an **elastic upturn** toward `Wi = 1`, not just a
single matched value.

## Tolerance met

§3 requires **≤ 1 % integrated**. The integrated drag `Cd` lands at −0.37 / −0.66 / −0.75 %
across `Wi ∈ {0.1, 0.5, 1.0}` — **all three under 1 %**.

**Verdict: PASS at the strict < 1 % integrated-Cd gate.**

## Provenance

**NOT CI-reproducible.** All Kraken numbers are **Float64 on an NVIDIA A100 (CUDA)**,
300 000 steps, run on the Aqua HPC cluster (job **22199679**, re-measured in jobs
26763246/47/48 on 2026-10-14). These are not produced by
the local test suite or by GitHub CI — reproducing them requires an A100-class GPU and an
overnight run. The RheoTool references are rheoFoam (OpenFOAM-9, Oldroyd-BLog,
log-conformation); the number of cores and the end time of the runs behind the stored `Cd` are not documented (archived logs of other runs: one process, `t = 6` or `10`). This artifact re-derives the comparison
CSVs, error norms and comparison plot from those raw outputs. The v0.3 cut-link drag
reproduced the prior dev-viscoelastic "M8" study (`m8_refs/`) to 4–5 significant figures;
both used the pre-fix forcing.

### Re-measurement after the TRT Guo forcing fix (2026-10-14)

The TRT Guo forcing defect (#67) injected `g·F` per step instead of `F`, with
`g = 1.019` at the `ν = 0.15` used here. The Kraken values were re-measured with the
settings of job 22199679 (Aqua A100, CUDA Float64, 300 000 steps):

- post-fix source `edbc20916`: jobs 26763246/47/48 (`β = 0.59`, `R = 10/30/50`),
  26766115 (`β` sweep, `R = 50`), 26766116/26766117 (high-`Wi` staircase, `R = 50`/`R = 30`);
- pre-fix control, source `249146d4d`: job 26763249, `Cd = 119.2410` at `Wi = 1`,
  `R = 50`, the published value.

At `β = 0.59` the fix moves `Cd` by +0.03 % to +0.23 % (`Wi = 0.1` to `1`,
`R = 10` to `50`). `kraken_cd_vs_wi.csv` and `error_norms.csv` hold the post-fix
values, and `comparison.png` is regenerated from them. `m8_refs/*.csv` are kept
pre-fix as the dated M8 record. Post-fix values for the files in `m8_refs/`:

- `β = 0.30`, `R = 50`: `Cd = 108.46` at `Wi = 0.5` (RheoTool 109.92, −1.33 %) and
  108.61 at `Wi = 1` (RheoTool 107.28, +1.24 %). The RheoTool `Wi = 1` run is not steady
  at `t = 20` (`Cd` 109.7 → 106.8 → 107.3 over `t = 15–20`), so that cell is not a
  validation point. `β = 0.1` and `0.01` still return NaN.
- wake-`N1` ratio Kraken/RheoTool: 0.771 / 0.705 (`β = 0.59`, `Wi = 0.5 / 1`),
  0.894 / 1.448 (`β = 0.30`, `Wi = 0.5 / 1`).
- staircase, all NaN-free to `Wi = 10`: `Cd` at `Wi = 3` is 131.63 (`R = 50`) and 118.45
  (`R = 30`), previously 141.84 and 116.26.

## References

- M. A. Alves, P. J. Oliveira, F. T. Pinho (2001), *The flow of viscoelastic fluids past
  a cylinder: finite-volume high-resolution methods*, J. Non-Newtonian Fluid Mech. **97**,
  207–232.
- M. A. Hulsen, R. Fattal, R. Kupferman (2005), *Flow of viscoelastic fluids past a
  cylinder at high Weissenberg number: stabilized simulations using matrix logarithms*,
  J. Non-Newtonian Fluid Mech. **127**, 27–39.
- R. Fattal, R. Kupferman (2004), *Constitutive laws for the matrix-logarithm of the
  conformation tensor*, J. Non-Newtonian Fluid Mech. **123**, 281–285.
- S. Claus, T. N. Phillips (2013), *Viscoelastic flow around a confined cylinder using
  spectral/hp element methods*, J. Non-Newtonian Fluid Mech. **200**, 131–146.
