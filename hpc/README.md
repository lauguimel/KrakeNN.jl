# HPC scripts (QUT AQUA)

PBS scripts and helpers for running Kraken.jl benchmarks and large
simulations on the QUT AQUA cluster.

## Convention

AQUA does NOT use git for the project tree — the local repo is rsync'd
to `aqua:~/Kraken.jl/` before each run, and results are pulled back the
same way. The `.git` directory and other machine-local artefacts are
excluded from the rsync.

## Workflow

```bash
# 1. Sync the local tree to AQUA (dry-run by default)
hpc/sync_to_aqua.sh           # preview
hpc/sync_to_aqua.sh --apply   # actually transfer

# 2. Submit a job
ssh aqua 'cd ~/Kraken.jl && qsub hpc/aqua_perf_h100.pbs'
ssh aqua 'qstat -u <user>'   # check queue status

# 3. Wait for the job to finish (PBS sends an email if -m abe is set)

# 4. Pull results back
hpc/pull_results_from_aqua.sh

# 5. Commit the new CSVs / figures
git add benchmarks/results/
git commit -m "bench: aqua h100 sweep $(date +%Y%m%d)"
```

## Available scripts

| File                          | Purpose                                            |
|-------------------------------|----------------------------------------------------|
| `sync_to_aqua.sh`             | rsync local → AQUA (excludes .git, output, etc.)   |
| `pull_results_from_aqua.sh`   | rsync AQUA → local results dir                     |
| `aqua_perf_h100.pbs`          | Full MLUPS sweep on H100 (CPU + GPU, 2h walltime)  |
| `run_natconv_gpu.pbs`         | Natural-convection GPU benchmark (existing)        |
| `run_natconv_cpu.pbs`         | Same on CPU                                        |
| `run_rc_sweep.pbs`            | Rc viscosity sweep for natural convection          |
| `run_rp_comparison.pbs`       | Rayleigh-Plateau pinch comparison                  |
| `run_viscoelastic.pbs`        | Viscoelastic cylinder benchmark                    |
| `run_quick_wins.pbs`          | GPU optimisation quick-wins benchmark              |

## Hardware mapping

The `--hardware-id` flag passed to `benchmarks/run_all.jl` must match a
section in `benchmarks/hardware.toml`:

| Flag             | Where it runs              | PBS script                |
|------------------|----------------------------|---------------------------|
| `apple_m2`       | Local laptop CPU           | (no PBS, local only)      |
| `aqua_h100`      | AQUA H100 node             | `aqua_perf_h100.pbs`      |
| `aqua_a100`      | AQUA A100 node             | (TODO: aqua_perf_a100.pbs)|

## Tips

- Each result CSV is timestamped, so successive runs accumulate rather
  than overwrite. The doc pipeline picks the most recent file matching
  a given prefix.
- The `aqua_perf_h100.pbs` script does a full `Pkg.instantiate()` on
  every run; this is slow the first time but cached afterwards.
- If `JULIA_CUDA_USE_COMPAT=false` causes issues, set it to `true` and
  rebuild CUDA.jl.
- For long jobs, use `qsub -m abe -M your.email@domain` to get email
  notifications, or add the `#PBS -m abe` directive to the script.
