# GNF D3Q19 channel validation

`run_gnf3d_gpu.jl` runs the same analytical channel families as
`test/analytical/gnf3d_channel_order.jl` on `CUDABackend()`:

| Law | Wall fraction | Float64 fluid nodes across gap |
| --- | --- | --- |
| Carreau, n=0.5 | 0.5 and 0.3 | 32, 64, 128, 256 |
| Power law, n=0.7 | 0.5 | 32, 64, 128 |
| Power law, n=1.5 | 0.5 | 32, 64, 128, 256 |

A separate Carreau q=0.5 row uses Float32 at N=128. No smooth-stress
companion is required: both power-law CPU order cases pass the fixed 1.8
threshold. The independent stress-balance references and diffusive parameter
scaling come from `test/analytical/gnf3d_reference.jl`, included with a guard.
The gap is H=N-1+2q_w; orders use successive H ratios, including q_w=0.3.

## Run on a CUDA compute node

```bash
export JULIA_CUDA_USE_COMPAT=false
julia --project benchmarks/validation/gnf3d/run_gnf3d_gpu.jl \
  --output-dir=benchmarks/validation/gnf3d/results
```

Use an allocated NVIDIA A100 or H100 on QUT Aqua; its scheduler is PBS Pro.
This directory does not submit jobs. The script logs CUDA hardware/software
information and warms each model/precision kernel before measuring runs.
The measured GPU result is in `results/` and described under "GPU evidence" below.

Outputs are `gnf3d_order.csv` and `gnf3d.log`. CSV columns are
`law,q_w,N,H,FT,err_L2,err_Linf,p_observed,steps,wall_s,MLUPS`.
Velocity errors are relative profile norms. MLUPS counts every launched cell,
including ghosts and solids; timing includes driver setup, convergence checks,
and final profile transfer, so it is an end-to-end measure.

The log records convergence, tolerance, viscosity extrema, pointwise bounds,
and finite-field checks. Float64 uses relative velocity-change tolerance
1e-10; Float32 uses 2e-4 to accommodate population roundoff at small lattice
velocities. Checks occur every 200 steps. The single Float32 row has no
observed-order claim. Every actual validation row must converge; failed rows
remain in the output and the process exits unsuccessfully after all cases.

## GPU evidence

Files: `results/gnf3d_order.csv` (16 rows) and `results/gnf3d.log`, copied
unchanged from the Aqua output directory. The last line of `gnf3d.log` holds
the absolute path of the CSV on the cluster home directory (`csv=/home/...`);
it was left as written.

| Item | Value |
| --- | --- |
| Code | branch `feat/gnf-3d`, commit `6517b1d92`; `src/` and `benchmarks/validation/gnf3d/` copied to the cluster by rsync, checksum comparison against the commit listed no differing file |
| Scheduler job | `27544762.aqua`, exit status 0, 1 node (`gpu0n009`), 4 CPUs, 32 GB |
| Walltime used | 00:12:22 (12 min); script log started 2026-11-02 15:01 UTC |
| GPU | NVIDIA A100-SXM4-40GB (sm_80), 39.5 GiB available |
| Driver / CUDA | driver 610.57.4 (CUDA 13.3); CUDA runtime 13.2 (artifact), compiler 13.3 |
| Julia / packages | Julia 1.12.5, CUDA.jl 5.11.3, KernelAbstractions 0.9.45, `JULIA_CUDA_USE_COMPAT=false` |
| Precision | Float64 for the 15 convergence rows; Float32 for the last row |
| Outcome | `rows_written=true validation_failures=0` |

Command run by the job script, from the code directory:

```bash
export JULIA_CUDA_USE_COMPAT=false
julia --project=. benchmarks/validation/gnf3d/run_gnf3d_gpu.jl --output-dir=<output directory>
```

Notes on reading the CSV:

- `power_thin` stops at N = 128, as designed (see "Known reference-validity
  limit"); the other laws run to N = 256.
- The Float32 row (Carreau, q_w = 0.5, N = 128) is a run-through check with the
  looser steady tolerance 2e-4 (error 1.1e-2). It is not a convergence result.
- The `MLUPS` column is not a performance measurement: the channel domain is
  small, and timing includes setup and profile transfer. Do not quote it as
  throughput.

## Known reference-validity limit

The n=0.7 model has nu_max=1.0. At N=256 its nearest-centre unclamped
analytical viscosity is 1.07672015411, above that limit, so the unclamped
reference would not be exact there. The n=0.7 sweep therefore stops at
N=128, where every node stays below the limit. A clamped analytical
reference or a wider viscosity range would be needed to go further.

All fluid nodes, including periodic ghosts, must lie strictly within the
model viscosity bounds, with the same 32-epsilon exclusion margin used by
the CPU tests. Orders below 1.8 also fail.

## CPU output smoke

```bash
julia --project benchmarks/validation/gnf3d/run_gnf3d_gpu.jl --smoke
```

This takes two CPU Float64 steps on N=4, checks finite fields, and writes
`tmp/m1b/gpu_smoke/gnf3d_order.csv` and `tmp/m1b/gpu_smoke/gnf3d.log`.
It never imports CUDA or accesses a GPU. The row is labelled `carreau_smoke`;
error, order and MLUPS fields are
`NaN`. This checks parsing and output plumbing only, and supplies no
convergence, accuracy, GPU, or performance evidence. Existing files in the
selected output directory are overwritten.
