# RheoTool reference: viscoelastic lid-driven cavity, Oldroyd-B, Wi = 1, beta = 0.5

Reference data for issue #51, milestone 1: RheoTool results only, no Kraken output. It is
the comparison target of the later Kraken pull requests (Newtonian entry gate, Wi = 1
validation). The N = 64 subset read by the tests is in
[`test/reference/rheotool_cavity_wi1/`](../../../../test/reference/rheotool_cavity_wi1/).
This file records provenance and column definitions only; for the analysis, see the
analysis comment on issue #51.

## Provenance

- Produced by Guillaume Maitrejean; jobs ran from 2026-09-24 19:13:34 to 2026-09-25
  02:32:06 (UTC+10; start times in `runs.csv`).
- Machine: QUT Aqua cluster, PBS Pro, one serial job per run (`ncpus=1`, `mem=8GB`).
  Compute nodes (`cpu1n040`, `cpu1n044`, `cpu1n045`, `cpu1n047`) per run in `runs.csv`.
- Code: solver `rheoFoam`, model `Oldroyd-BLog`, OpenFOAM-9 build `9-b456138dc4bc`
  (printed in every log). The rheoTool source tree in the container is at commit
  `81f3e8cf6079e84afd04c5d847389c338404b268` of <https://github.com/fppimenta/rheoTool>
  (ChangeLog "Version 6.0"), with two modified build scripts and one modified of70
  tutorial file; no of90 solver, library or tutorial file is changed. That the binaries
  were built from this tree is inferred from the image build history and the binary
  dates, not checked by rebuilding. The rheoTool README points to Pimenta & Alves (2017)
  for the solver theory.
- Container: Apptainer sandbox built on 2025-06-26 from
  `docker://guiguitcho/openfoam9-rheotool:v1.2` (tag and build date read from the
  sandbox's `.singularity.d/labels.json`). The build did not record the image
  digest. The Apptainer cache on Aqua holds, with that date, the manifest Docker Hub lists
  for `v1.2` (`sha256:5a87c8733c94820e1ceb6606a44e5c8eb7c19de2f2a4f665a56954dd90ca81fd`);
  that the sandbox came from it is an inference.

## Case

Tutorial `of90/tutorials/rheoFoam/Cavity/Oldroyd-BLog` of rheoTool `81f3e8c`; template in
[`bench/rheotool/cavity_ve_wi1/`](../../../../bench/rheotool/cavity_ve_wi1/).

- Fluid: Oldroyd-B, log-conformation form, `etaS = etaP = 0.5` (beta = 0.5), `lambda = 1`
  (Wi = 1), `Re = rho U L/(etaS + etaP) = rho`. `rho = 1`, except `rho = 0.01` (the
  tutorial value) in the two `ve_re001_*` runs. Newtonian control:
  `constant/constitutiveProperties.newtonian`, `eta = 1`, `rho = 1`.
- Lid (`0/U`, `codedFixedValue`, as shipped), at the lid face centres:
  `u_x(x, t) = 8 (1 + tanh(8 (t - 0.5))) x^2 (1 - x)^2`, `u_y = 0`, i.e.
  `16 x^2 (1 - x)^2` (the regularised lid "R1" of Sousa et al. (2016)) times a ramp
  equal to 3.354e-4 at t = 0 and 0.99966 at t = 1. Other
  walls no-slip; `frontAndBack` `empty` (2D). Start from rest, `tau = theta = 0`.
- Mesh: one block, unit square, N x N x 1 uniform cells.
- Time: fixed `deltaT = 2e-4` (1e-4 in the `dt1e-4` runs), `endTime = 10` (30 in
  `ve_re1_n64_t30`); fields written every time unit, 12 significant digits.
- Numerics, as shipped: `CrankNicolson 1`; `GaussDefCmpw cubista` convection of `U`,
  `theta`, `tau`; `Gauss linear` gradients and Laplacians; `stabilization coupling`;
  segregated `SIMPLE` with `nInIter 1` (one pass per step); `p`, `U` by Eigen BiCGSTAB to
  1e-12; `theta` by PBiCG/DILU to 1e-10, `relTol 0`, `minIter 0`.

Changes from the tutorial (`diff -ru` against rheoTool `81f3e8c`):

1. `constant/constitutiveProperties`: `rho` 0.01 -> 1.
2. `system/blockMeshDict`: new entry `N`; block `(127 127 1)` -> `($N $N 1)`.
3. `system/controlDict`: `writeInterval` 2 -> 1; the energy function object writes 0 in
   the elastic column when there is no `tau` or no `etaP` (formulas unchanged).
4. Added: `constitutiveProperties.newtonian`, `run_aqua.pbs`, `extract_cavity.py`,
   `build_tables.py`.

`run_aqua.pbs` sets `N`, `rho`, `endTime`, `deltaT` with `foamDictionary` and, for
`MODEL=newt`, copies the Newtonian properties over the viscoelastic ones.
`ve_re001_n127_asshipped` differs from the tutorial only by items 2 (same 127 x 127 mesh),
3 and 4.

## Runs

Source: `runs.csv` (also host, PBS job, start time). Every finished run has exit status 0
and a `log.rheoFoam` ending with `End`.

| tag | model | N | Re | deltaT | endTime | wall time |
|---|---|---|---|---|---|---|
| `newt_re1_n64` | Newtonian | 64 | 1 | 2e-4 | 10 | 00:04:12 |
| `newt_re1_n128` | Newtonian | 128 | 1 | 2e-4 | 10 | 00:19:08 |
| `newt_re1_n256` | Newtonian | 256 | 1 | 2e-4 | 10 | 01:28:38 |
| `ve_re1_n64` | Oldroyd-BLog | 64 | 1 | 2e-4 | 10 | 00:08:04 |
| `ve_re1_n128` | Oldroyd-BLog | 128 | 1 | 2e-4 | 10 | 00:33:22 |
| `ve_re1_n256` | Oldroyd-BLog | 256 | 1 | 2e-4 | 10 | 02:34:04 |
| `ve_re1_n64_dt1e-4` | Oldroyd-BLog | 64 | 1 | 1e-4 | 10 | 00:16:34 |
| `ve_re1_n128_dt1e-4` | Oldroyd-BLog | 128 | 1 | 1e-4 | 10 | 01:16:20 |
| `ve_re1_n256_dt1e-4` | Oldroyd-BLog | 256 | 1 | 1e-4 | 10 | 06:19:23 |
| `ve_re1_n64_t30` | Oldroyd-BLog | 64 | 1 | 2e-4 | 30 | 00:25:17 |
| `ve_re001_n128` | Oldroyd-BLog | 128 | 0.01 | 2e-4 | 10 | 00:35:08 |
| `ve_re001_n127_asshipped` | Oldroyd-BLog | 127 | 0.01 | 2e-4 | 10 | 00:30:35 |

## Files and columns

`extract_cavity.py` reduces each run to three files; `build_tables.py` joins them, copying
values as text (`%.12g`). One header line per file; times in L/U.

- `runs.csv`: `tag, model, N, Re, deltaT, endTime, host, pbs_job, start, wall_time`.
- `energy.csv`: `tag, t, kinetic_energy_mean, elastic_energy_mean`. Logged every 20 steps
  (0.004 at `deltaT = 2e-4`), kept every 0.008 (`--energy-spacing 0.008`), t = 0 and
  t = 10 included. For `ve_re1_n64`, `ve_re1_n64_t30` and the three `dt1e-4` runs the
  kinetic-energy peak row is dropped: the table's peak is up to 3.27e-7 lower and one log
  interval later than the full log's.
- `vortex.csv`: `tag, t, x_centre, y_centre, psi_min, max_abs_column_integral_ux`, every
  written time.
- `profiles_t10.csv`: `tag, t, line, coord, ux, uy, tau_xx, tau_xy, tau_yy, C_xx, C_xy, C_yy`
  at t = 10 (and t = 30 for `ve_re1_n64_t30`); stress columns empty for Newtonian runs.

Definitions:

- Energies, as coded in `system/controlDict`: `kinetic_energy_mean =
  (0.5/nCells) sum_cells |U|^2` and `elastic_energy_mean = (0.5/nCells) (lambda/etaP)
  sum_cells tr(tau)`, the cell mean of `tr(C - I)/2`. Plain cell means, equal to volume
  means only because the mesh is uniform. Newtonian runs log 0 for the elastic energy.
  Logged times carry accumulated rounding (e.g. 9.00000000001; the `dt1e-4` runs end at
  9.99999999999): always match the nearest row.
- Stream function: `psi = int_0^y u_x dy'` from the bottom wall, midpoint rule on cell
  centres: `psi[j, i] = h (u[0, i] + ... + u[j-1, i] + u[j, i]/2)`, `h = 1/N`; negative
  in the primary vortex.
- Vortex centre: `x_centre, y_centre` are the stationary point, found by Newton
  iteration, of a 4 x 4 tensor-product cubic interpolant of `psi` around its discrete
  minimum; `psi_min` is the interpolant there. The full rule, fallback included, is in the
  docstring of `vortex_centre` in `extract_cavity.py`; no row here uses the fallback.
- `max_abs_column_integral_ux`: largest `|int_0^1 u_x dy|` over cell columns, a
  consistency check.
- Lines `x=0.5` (against y) and `y=0.5`, `y=0.75`, `y=0.9`, `y=0.95` (against x);
  `coord = (k + 1/2)/N`. Values are linear interpolations between the two cell-centre rows
  (or columns) around the line, weight `w = s - floor(s)`, `s = line_position * N - 1/2`,
  on the upper one. These are not OpenFOAM `postProcessing/sampleDict` values, which use
  another stencil at some points on y = 0.75; do not mix the two.
- `C = I + (lambda/etaP) tau`, with `lambda = 1`, `etaP = 0.5` given on the
  `extract_cavity.py` command line (defaults), not read from the case.
- `ve_re1_n64_t30` after t = 29: from t = 29.0052 the `theta` solves start skipping
  iterations (tolerance 1e-10, `minIter 0`) and from about t = 29.4 `theta` is no longer
  updated. These rows are a solver-frozen state, not a converged steady state.

## Use of these data

How to compare a Kraken result with these data (same N, Re, t = 10, same reduction
script and `psi` construction) and what can be gated:
`test/reference/rheotool_cavity_wi1/README.md`. The analysis (mesh, time-step, Re and
steadiness studies, literature) is in the comment on issue #51.

## How to regenerate

On the Aqua login node, from a copy of `bench/rheotool/cavity_ve_wi1/`, one job per row of
`runs.csv`, e.g.:

```bash
qsub -l walltime=08:00:00 -N rT_cav_ve128 -v N=128,RHO=1,MODEL=ve,TAG=ve_re1_n128_repro run_aqua.pbs
```

`RHO` = Re, `MODEL` = `ve` or `newt`; add `DT` when `deltaT` is not 2e-4 and `ENDTIME`
when `endTime` is not 10.

**Warning: `run_aqua.pbs` first runs `rm -rf $HOME/kraken-rheotool/cavity_ve_wi1/$TAG`.**
Resubmitting an existing tag deletes that run directory on Aqua, original runs included.
Use a new tag (e.g. `ve_re1_n64_repro`). `build_tables.py` reads only the tags listed in
its `RUNS`, all twelve from one csv directory: extract a reproduction with the original
tag as output prefix into a separate csv directory (e.g. directory
`$ARCHIVE/rheotool_csv_repro`, prefix `$ARCHIVE/rheotool_csv_repro/ve_re1_n64`), copy the
other runs' CSV files into that directory, pass it to `build_tables.py`, and pass
`--results` and `--reference` pointing outside the repository (by default the build
overwrites the committed files); `runs.csv` then still lists the original jobs.

What ran differs in two points. The first eight jobs used an earlier `run_aqua.pbs`
without `DT`; their `deltaT` is 2e-4 (checked in each `controlDict`), the committed
default. The runs removed `rT_cavity.o*`, which misses the job names `rT_cav_*`, so stale
PBS output files (six per directory) were copied into the four run directories submitted
later (`ve_re1_n64_dt1e-4`, `ve_re1_n128_dt1e-4`, `ve_re1_n256_dt1e-4`, `ve_re1_n64_t30`;
no effect on results); the script now removes `rT_cav*.o*`.

Copy back, reduce and build (from the repository root):

```bash
ARCHIVE=/path/outside/the/repository
mkdir -p "$ARCHIVE/rheotool_runs" "$ARCHIVE/rheotool_csv"
rsync -az --include='/[0-9]*/' --include='/[0-9]*/U.gz' --include='/[0-9]*/tau.gz' \
      --include='/kinEner.txt' --include='/log.*' --include='/run_done' \
      --include='/system/' --include='/system/controlDict' --include='/system/blockMeshDict' \
      --include='/constant/' --include='/constant/constitutiveProperties' --exclude='*' \
      maitreje@aqua.qut.edu.au:kraken-rheotool/cavity_ve_wi1/<TAG>/ "$ARCHIVE/rheotool_runs/<TAG>/"
python bench/rheotool/cavity_ve_wi1/extract_cavity.py "$ARCHIVE/rheotool_runs/<TAG>" <N> "$ARCHIVE/rheotool_csv/<TAG>"
python bench/rheotool/cavity_ve_wi1/build_tables.py "$ARCHIVE/rheotool_csv" --energy-spacing 0.008
```

`build_tables.py` writes the four CSV files here and the three in
`test/reference/rheotool_cavity_wi1/`. The committed files were built this way on
2026-09-25 (UTC+10), after the last job ended (Python 3.11.15, numpy 2.4.6); two
independent builds were byte-identical.

## References

Entries in `docs/refs.bib` (keys in brackets).

- Pimenta, F. & Alves, M. A. (2017). Stabilization of an open-source finite-volume solver
  for viscoelastic fluid flows. J. Non-Newton. Fluid Mech. 239, 85-104.
  doi:10.1016/j.jnnfm.2016.12.002 [`pimenta2017stabilization`]
- Sousa, R. G., Poole, R. J., Afonso, A. M., Pinho, F. T., Oliveira, P. J., Morozov, A. &
  Alves, M. A. (2016). Lid-driven cavity flow of viscoelastic liquids. J. Non-Newton.
  Fluid Mech. 234, 129-138. doi:10.1016/j.jnnfm.2016.03.001 [`sousa2016lid`]
