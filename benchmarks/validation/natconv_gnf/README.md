# Generalized Newtonian natural convection

## Turan et al. (2011) comparison

`run_turan_comparison.jl` compares hot-wall mean Nu for a square power-law
cavity with fixed side-wall temperatures, adiabatic top/bottom and no-slip
walls. Reference: O. Turan, A. Sachdeva, N. Chakraborty and R.J. Poole,
J. Non-Newtonian Fluid Mech. 166 (2011), 1049–1063,
[doi:10.1016/j.jnnfm.2011.06.003](https://doi.org/10.1016/j.jnnfm.2011.06.003).
All five cases have Pr = 10; Nu values are figure readings, not tabulated data.

| Case | n | Ra | Paper Nu | Source (journal page) | Lattice nu0 |
|---|---:|---:|---:|---|---:|
| RA3_N06 | 0.6 | 1e3 | 2.07 ± 0.07 | Fig. 9a, p. 1059 | 0.1 |
| RA4_N06 | 0.6 | 1e4 | 5.40 ± 0.02 | Fig. 10, p. 1060 | 0.1 |
| RA4_N08 | 0.8 | 1e4 | 3.38 ± 0.02 | Fig. 10, p. 1060 | 0.1 |
| RA4_N10 | 1.0 | 1e4 | 2.29 ± 0.015 | Figs. 10–11, pp. 1060–1061 | 0.1 |
| RA4_N18 | 1.8 | 1e4 | 1.10 ± 0.01 | Fig. 11, p. 1061 | 0.02 |

CUDA Float64 grids default to N = 64, 128, 256, in ascending N order;
`--grids=64,128` overrides them. Set alpha = nu0/Pr and H = N in
`cavity_powerlaw`, making Ra/Pr identical to the paper's Eq. 6 definitions.
Viscosity bounds are [1e-3, 2.0]. The smaller nu0 for n = 1.8 reduces BGK
wall error from large relaxation times. For t_diff = N²/alpha steps,
check_every = ceil(0.01 t_diff), min_steps = ceil(0.1 t_diff), and
max_steps = ceil(1.5 t_diff). Convergence requires three consecutive relative
Nu changes below 1e-6 after min_steps. Cost: about 1–3 GPU-hours, estimated,
not measured. **GPU run pending.**

Outputs are `turan_comparison.csv` and `turan_comparison.log` in `results/`
beside the script, overridable with `--output-dir=<path>`. Both flush each row.
They record signed relative Nu error, steps, convergence, final clamped-node
fraction, viscosity extrema, wall seconds (including compilation), and source.
Vmax_mid is max |uy| on the horizontal mid-plane times N/alpha; for even N,
the two central j planes are averaged before taking max |uy|. The log records
the device, Julia version and precision. Non-finite Nu or unconverged rows
cause a nonzero exit after all rows are saved; smoke waives convergence only.

CPU smoke from the repository root (N = 16, at most 200 steps per case;
never loads CUDA; output forced to `tmp/m1c/turan_smoke`):

```sh
julia --project benchmarks/validation/natconv_gnf/run_turan_comparison.jl --smoke
```

For Aqua deployment, copy the repository into a directory of your choice.
On the login node, from that repository root, build a private depot using
the same three settings as the PBS file (`$PWD` is the future PBS workdir):

```sh
export JULIA_BIN=$HOME/.julia/juliaup/julia-1.12.5+0.x64.linux.gnu/bin/julia
export JULIA_DEPOT_PATH=${KRAKEN_DEPOT:-$PWD/depot}
export JULIA_CUDA_USE_COMPAT=false
"$JULIA_BIN" --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
qsub -P UXJFSK0637 benchmarks/validation/natconv_gnf/turan_gpu.pbs
```

The prepared six-hour PBS job pins gpu0n009; avoid gpu0n001. Submission is
left to the user. No GPU results have been produced for this comparison.

Table 2 (Ra = 1e6, Pr = 100) is not included: at Pr = 100 the thermal
relaxation time is within 0.01 of 1/2, and N = 512 needs of the order of
1e7 steps. This requires a separate study.

Known O(1/N) offset: thermal walls sit on the first/last node while Ra and
Nu use H = N. A conduction-only cavity gives Nu ≈ N/(N−1), measured as
1.036 at N = 32, Ra = 10. The CPU unit test uses N = 48, nu0 = 0.3 for
RA3_N06 and a frozen 4% paper-uncertainty gate; the GPU grids remain untested.

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
