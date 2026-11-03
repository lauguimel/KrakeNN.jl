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
No GPU results have been produced for this comparison.

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
