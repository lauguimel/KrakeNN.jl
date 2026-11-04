# Performance: MLUPs throughput

The standard metric for LBM codes is **Mega Lattice Updates Per Second**
(MLUPs):

```math
\text{MLUPs} = \frac{N_x \times N_y \times N_{\text{steps}}}{t_{\text{wall}} \times 10^6}
```

This page only publishes numbers that are traceable to a CSV file in
`benchmarks/results/`. There is currently no LBM D2Q9 MLUPs CSV for CUDA or
for an H100/A100 GPU in this repository — a headline GPU throughput table
was briefly reintroduced in the v0.2.0 docs without a matching artifact and
has been removed again here. The only fully-sourced H100 figures in
`benchmarks/results/` are for the unrelated FVFD/IncNS SIMPLE cavity solver
(`benchmarks/results/cavity_gpu_ablation_aqua_h100.md`, discussed in
[Incompressible Navier-Stokes](../users/incompressible-navier-stokes.md)),
not for this LBM BGK benchmark. Below is what is actually measured for the
LBM path today.

## CPU baseline (Apple M2, Float64)

| Grid (N×N) | MLUPs |
|:-----------|------:|
| 64×64      | 31.90 |
| 128×128    | 31.85 |
| 256×256    | 26.54 |

Source: `benchmarks/results/perf_mlups_cpu_apple_m2_20260410_115127.csv`.
Throughput stays in the 26–32 MLUPs range across these grid sizes,
consistent with a memory-bandwidth-limited single-core kernel.

## Metal GPU (Apple M3 Max, Float32)

| Grid (N×N)  | MLUPs |
|:------------|------:|
| 64×64       |   6.8 |
| 128×128     |  20.7 |
| 256×256     |  81.8 |
| 512×512     | 257.3 |
| 1024×1024   | 572.7 |
| 2048×2048   | 838.4 |
| 4096×4096   | 920.5 |

Source: `benchmarks/results/perf_mlups_metal_apple_m3max_final.csv`. This is
a local development artifact (single laptop run), not a certified benchmark;
useful for tracking regressions, not for a headline number. Throughput rises
with grid size as launch overhead is amortised, then plateaus below 1000
MLUPs at 4096×4096.

## What is NOT measured here

- No CUDA / NVIDIA GPU MLUPs numbers for the D2Q9 BGK LBM kernel exist in
  this repository as of this writing. Any H100/A100 throughput claim for
  *this* solver should be treated as unverified until a CSV is committed
  under `benchmarks/results/` with a matching `hardware_id` entry in
  `benchmarks/hardware.toml`.
- The AA-pattern / `Float32`-storage optimisations mentioned in earlier
  drafts of this page are not merged into the default driver and are not
  represented by any committed benchmark artifact; no throughput number for
  them is published here.

## Reproduce these benchmarks

```bash
# CPU-only quick sweep (writes perf_mlups_cpu_<hardware-id>_<timestamp>.csv)
julia --project benchmarks/run_all.jl --quick --hardware-id=my_machine

# Include GPU (requires a functional CUDA.jl or Metal.jl backend)
julia --project benchmarks/run_all.jl --gpu --hardware-id=my_gpu
```

Results are written to `benchmarks/results/` as CSV files tagged with the
hardware ID and timestamp. Add your hardware to
[`benchmarks/hardware.toml`](https://github.com/lauguimel/Kraken.jl/blob/main/benchmarks/hardware.toml)
before running so the results are traceable.
