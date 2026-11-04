# Generalized Newtonian natural convection

## Turan et al. (2011) comparison

`run_turan_comparison.jl` compares hot-wall mean Nu for a square power-law
cavity with fixed side-wall temperatures, adiabatic top/bottom and no-slip
walls. Reference: O. Turan, A. Sachdeva, N. Chakraborty and R.J. Poole,
J. Non-Newtonian Fluid Mech. 166 (2011), 1049–1063,
[doi:10.1016/j.jnnfm.2011.06.003](https://doi.org/10.1016/j.jnnfm.2011.06.003).
All six cases have Pr = 10; Nu values are figure readings, not tabulated data.

| Case | n | Ra | Paper Nu | Source (journal page) | Lattice nu0 |
|---|---:|---:|---:|---|---:|
| RA3_N06 | 0.6 | 1e3 | 2.07 ± 0.07 | Fig. 9a, p. 1059 | 0.3 |
| RA3_N10 | 1.0 | 1e3 | 1.14 ± 0.02 | Fig. 9b, p. 1059 | 0.3 |
| RA3_N18 | 1.8 | 1e3 | 1.01 ± 0.005 | Fig. 9c, p. 1059 | 0.05 (`NU0_N18`) |
| RA4_N06 | 0.6 | 1e4 | 5.40 ± 0.02 | Fig. 10, p. 1060 | 0.3 |
| RA4_N10 | 1.0 | 1e4 | 2.29 ± 0.015 | Figs. 10–11, pp. 1060–1061 | 0.3 |
| RA4_N18 | 1.8 | 1e4 | 1.10 ± 0.01 | Fig. 11, p. 1061 | 0.05 (`NU0_N18`) |

`NU0_N18` is a constant at the top of the script. CUDA Float64 grids default
to N = 64, 128, 256, in ascending N order; `--grids=64,128` and
`--cases=RA3_N06,RA4_N06` select a subset. Set alpha = nu0/Pr and H = N in
`cavity_powerlaw`, making Ra/Pr identical to the paper's Eq. 6 definitions.
The thermal walls are half-way, like the no-slip walls, so H = N is the cavity
width. Viscosity bounds are [1e-3, 2.0]. nu0 only sets the lattice time step,
but a shear-thickening fluid has its largest viscosity at the walls, where the
BGK error grows with the relaxation time, so n = 1.8 uses a smaller nu0. For
t_diff = N²/alpha steps, check_every = ceil(0.01 t_diff),
min_steps = ceil(0.1 t_diff), and max_steps = ceil(t_diff). Convergence
requires three consecutive relative Nu changes below 1e-6 after min_steps.
On CPU at N ≤ 96 the runs converged after 0.26 to 0.43 t_diff, except
RA4_N18: at N = 64 it was still drifting (1.8e-5 per 1000 steps, Nu = 1.088)
after 1.8 t_diff, with 0.1 % of the nodes at nu_max. Expect that row to be
flagged unconverged, which makes the job exit with status 1 after all rows
are written.

Outputs are `turan_comparison.csv` and `turan_comparison.log` in `results/`
beside the script, overridable with `--output-dir=<path>`. Both flush each row.
They record signed relative Nu error, steps, convergence, the last Nu change,
final clamped-node fraction, viscosity extrema, wall seconds (including
compilation), and source. Vmax_mid is max |uy| on the horizontal mid-plane
times N/alpha; for even N, the two central j planes are averaged before taking
max |uy|. The log records the device, Julia version, precision and git
revision. Non-finite Nu or unconverged rows cause a nonzero exit after all rows
are saved; smoke waives convergence only.

CPU checks from the repository root:

```sh
# N = 16, at most 200 steps per case; output forced to tmp/m1c/turan_smoke
julia --project benchmarks/validation/natconv_gnf/run_turan_comparison.jl --smoke
# one converged case, about one minute
julia --project benchmarks/validation/natconv_gnf/run_turan_comparison.jl \
    --backend=cpu --grids=64 --cases=RA3_N06 --output-dir=tmp/turan_cpu
```

## GPU job

`turan_gpu.pbs` runs two steps on one A100 and writes everything to
`results/` beside this file:

1. `run_turan_comparison.jl --backend=cuda`: the six cases on N = 64, 128, 256
   (`turan_comparison.csv`, `turan_comparison.log`);
2. `benchmarks/validate_thermal_ref_3d_gpu.jl --backend=cuda --FT=Float64`: the
   3D Newtonian thermal checks (conduction, Ra = 1e3 with refinement patches,
   N = 16), re-run with the half-way thermal walls (`thermal_ref_3d_gpu.log`).

Estimated run time (from the CPU step counts and the 2D GPU throughput, not
measured): about 1 h, at most about 1 h 40 min if every case runs to
max_steps. The two n = 1.8 cases at N = 256 take most of it (13 million steps
each at most). The walltime is 3 h.

For Aqua deployment, copy the repository into a directory of your choice.
On the login node, from that repository root, build a private depot with the
same settings as the PBS file (`$PWD` is the future PBS workdir):

```sh
export JULIA_BIN=/path/to/julia
export JULIA_DEPOT_PATH=$PWD/depot
export JULIA_CUDA_USE_COMPAT=false
"$JULIA_BIN" --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
qsub -P <project-code> -v JULIA_BIN="$JULIA_BIN" benchmarks/validation/natconv_gnf/turan_gpu.pbs
```

The job pins host gpu0n009; avoid gpu0n001. Submission is left to the user.

### GPU results

Job on one A100 (CUDA, Float64, Julia 1.12.5) from commit 1c70037 (kept reachable by the
tag `archive/gnf-thermal-turan-gpu`), 20 min walltime. Files in `results/`: `turan_comparison.csv` and `.log`,
`thermal_ref_3d_gpu.log`, and `turan_gpu_job.log` (the PBS job log, with user,
path, host and project lines removed).

Nu Kraken and signed gap to the Turan value (figure reading; the uncertainty is
in the CSV column `Nu_paper_source`). `nu0` = 0.3, except 0.05 for n = 1.8.

| Ra | n | Paper Nu | N = 64 | N = 128 | N = 256 | Converged |
|---|---|---|---|---|---|---|
| 10³ | 0.6 | 2.07 | 1.9931 (−3.7 %) | 1.9917 (−3.8 %) | 1.9905 (−3.8 %) | yes |
| 10³ | 1.0 | 1.14 | 1.1201 (−1.7 %) | 1.1186 (−1.9 %) | 1.1181 (−1.9 %) | yes |
| 10³ | 1.8 | 1.01 | 1.0089 (−0.1 %) | 1.0087 (−0.1 %) | 1.0087 (−0.1 %) | yes |
| 10⁴ | 0.6 | 5.40 | 5.3678 (−0.6 %) | 5.3828 (−0.3 %) | 5.3840 (−0.3 %) | yes |
| 10⁴ | 1.0 | 2.29 | 2.2980 (+0.4 %) | 2.2824 (−0.3 %) | 2.2774 (−0.5 %) | yes |
| 10⁴ | 1.8 | 1.10 | 1.0880 (−1.1 %) | 1.0714 (−2.6 %) | 1.0645 (−3.2 %) | no |

- Ra = 10³, n = 0.6: 1.993 / 1.992 / 1.991 against 2.07 (−3.7 % to −3.8 %).
  Nu changes by 0.13 % from N = 64 to 256, as little as on CPU, so the gap is not a grid
  error. The paper value is read from a figure (± 0.07).
- Ra = 10⁴, n = 0.6 and n = 1.0: within 0.6 % of the paper at every N.
- Ra = 10³, n = 1.8: within 0.13 %.
- Ra = 10⁴, n = 1.8 is not a validated point. It stopped at the step limit
  without converging (last Nu change 4.0e-5 at N = 64, 4.2e-6 at N = 128,
  1.7e-6 at N = 256), and the viscosity sits at its upper bound `nu_max` = 2.0 on
  0.2 %, 5.9 % and 8.0 % of the nodes. The table values are listed only for
  completeness.
- At N = 64 the GPU and CPU values agree to 2e-5 or better in Nu (for example
  1.99308 on both for Ra = 10³, n = 0.6; the largest difference is 1.00883
  on CPU against 1.00885 on GPU for Ra = 10³, n = 1.8, which stopped at
  different step counts). The CPU values are quoted to five decimals.
- The job script returns status 1 on purpose: the Ra = 10⁴, n = 1.8 rows are
  not converged and the 3D refined check fails. The job ended with exit
  status 1 for these two reasons, after writing all rows.

The 3D refined thermal check (`benchmarks/validate_thermal_ref_3d_gpu.jl`)
reproduces on GPU the defect seen on CPU (conduction error 3.5e-2 against a 0.02
gate; refined coarse Nu 7.12 against 1.085); grid refinement is under development
and not validated (issue #98).


Table 2 (Ra = 1e6, Pr = 100) is not included: at Pr = 100 the thermal
relaxation time is within 0.01 of 1/2, and N = 512 needs of the order of
1e7 steps. This requires a separate study.

## Existing qualitative sweep (Pr = 0.71)

`run_natconv_gnf.jl` prepares a CUDA Float64 sweep at N = 256,
Ra = 1e4/1e5 and n = 0.6/0.8/1.0/1.2/1.4/1.8, intended for an A100 or H100.
**GPU sweep unrun.** With an already instantiated environment:

```sh
julia --project benchmarks/validation/natconv_gnf/run_natconv_gnf.jl
```

On Aqua, export `JULIA_CUDA_USE_COMPAT=false` before Julia. The script writes
`results.csv` and a persistent `run.log` beside itself, recording actual device,
n, Ra, N, steps, Nu, wall time, final clamped fraction, convergence and Nu drift.
Stopping requires three successive 5000-step changes below 1e-6 after
50000 steps; non-converged rows remain marked. Wall time includes compilation.

Ra/Pr use nu0 = 0.05. The shear scale alpha/H^2 is Turan's characteristic
shear rate; Pr = 0.71 is outside their power-law data (Pr >= 10), so this
sweep remains qualitative. Clamped fraction is the final count at either
viscosity cutoff divided by N² fluid nodes, not accumulated over time.

CPU-only import check (does not import CUDA or launch the sweep):

```sh
julia --project -e 'include("benchmarks/validation/natconv_gnf/run_natconv_gnf.jl")'
```
