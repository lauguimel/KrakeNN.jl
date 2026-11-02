# Generalized Newtonian natural convection

Prepared CUDA Float64 cavity sweep: N=256, Ra=1e4/1e5 and
n=0.6/0.8/1.0/1.2/1.4/1.8. **GPU sweep unrun.** Intended hardware is an
NVIDIA A100 or H100; the script records the actual device at execution.

From the repository root, with an already instantiated Julia environment:

```sh
julia --project benchmarks/validation/natconv_gnf/run_natconv_gnf.jl
```

On Aqua, export `JULIA_CUDA_USE_COMPAT=false` before Julia. The script writes
`results.csv` and a persistent `run.log` beside itself, logging n, Ra, N, steps, Nu, wall time, final
clamped-node fraction, convergence status and relative Nu drift. Nu stopping
requires three successive 5000-step changes below 1e-6 after 50000 steps;
non-converged rows remain explicitly marked. Wall time includes compilation.

Ra/Pr use reference viscosity nu0=0.05. The power-law consistency uses the
provisional diffusive shear scale alpha/H². These results are intended for
qualitative exploration; no quantitative power-law literature validation is
claimed. Clamped fraction is the final count at either viscosity cutoff
divided by N² fluid nodes, not accumulated over time.

CPU-only import check (does not import CUDA or launch the sweep):

```sh
julia --project -e 'include("benchmarks/validation/natconv_gnf/run_natconv_gnf.jl")'
```
