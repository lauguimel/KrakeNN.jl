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
GPU run pending: the first attempt failed on a scalar-indexing error in the
driver setup (fixed, not yet re-run). No GPU result is included with this harness.

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
