# Kraken.jl benchmark suite

Single-command benchmarks for the v0.1.0 in-scope physics
(single-phase LBM, thermal, grid refinement). Every benchmark writes
a timestamped CSV into `benchmarks/results/` so the documentation
figures stay in sync with the reported numbers.

## Quick start

```bash
# Fast subset (~5 min CPU, laptop)
julia --project benchmarks/run_all.jl --quick

# Full CPU sweep
julia --project benchmarks/run_all.jl

# Full CPU + GPU sweep (requires CUDA.jl or Metal.jl functional)
julia --project benchmarks/run_all.jl --gpu --hardware-id=apple_m2
```

## Flags

| Flag                      | Purpose                                         |
|---------------------------|-------------------------------------------------|
| `--quick`                 | Reduced subset (skip slow convergence + GPU)    |
| `--gpu`                   | Run GPU benchmarks on top of CPU                |
| `--hardware-id=<key>`     | Label matching `benchmarks/hardware.toml`       |
| `--skip-existing`         | Skip cases whose CSV already exists             |
| `--output-dir=<path>`     | Override `benchmarks/results`                   |

## Hardware declarations

Each result CSV carries a `hardware_id` column matching a section in
`benchmarks/hardware.toml`. The shipped declarations are:

- `apple_m2` — Guillaume's laptop (CPU baseline)
- `aqua_h100` — QUT Aqua H100 GPU node (submit via `hpc/aqua_perf_h100.pbs`)
- `aqua_a100` — QUT Aqua A100 GPU node

Add your own section to `hardware.toml` when running on a new machine,
then pass `--hardware-id=<your_key>` to `run_all.jl`.

## Running on Aqua (HPC)

```bash
# From your laptop
scp hpc/aqua_perf_h100.pbs <user>@aqua.qut.edu.au:~/Kraken.jl/hpc/
ssh <user>@aqua.qut.edu.au 'cd ~/Kraken.jl && qsub hpc/aqua_perf_h100.pbs'

# After the job finishes, pull the CSV back
scp '<user>@aqua.qut.edu.au:~/Kraken.jl/benchmarks/results/*.csv' benchmarks/results/
```

## Scripts

| Script                            | What it measures                        |
|-----------------------------------|-----------------------------------------|
| `convergence_poiseuille.jl`       | L2 error vs N, spatial order            |
| `convergence_taylor_green.jl`     | Temporal decay vs analytical            |
| `convergence_cavity.jl`           | Ghia 1982 centerline error vs N         |
| `convergence_thermal.jl`          | Nu vs Ra (Rayleigh-Benard)              |
| `perf_mlups.jl`                   | MLUPS vs N, CPU + GPU                   |
| `perf_gpu_physics.jl`             | Physics-specific GPU MLUPS breakdown    |
| `perf_optimizations.jl`           | BGK vs AA vs fused on GPU               |

Out-of-scope (multiphase, VOF, rheology, viscoelastic) lives on the
`dev` branch and is not covered here.

## CSV schema

Convergence files use:

```
case, N, error_L1, error_L2, error_max, observed_order, hardware_id
```

Performance files use:

```
case, N, backend, precision, mlups, walltime_s, steps, hardware_id
```

Every CSV filename follows the pattern
`<name>_<hardware_id>_<timestamp>.csv` so successive runs accumulate
rather than overwrite. The documentation pipeline picks the most recent
file matching a given prefix.
