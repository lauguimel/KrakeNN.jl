# Reproducibility bundle — GPU efficiency certification

Self-contained bundle for the **GPU certification** benchmark page
(`docs/src/users/benchmarks/gpu-certification.md`). Regenerates the throughput
and roofline-ratio figure from the shipped CSV, in the locked house plotting
style (seaborn `crest`, LaTeX, serif whitegrid).

## Case

Single-GPU throughput certification of the Newtonian BGK D2Q9 solver in **CUDA
Float64**, driven by the Taylor–Green vortex driver (`run_taylor_green_2d`) over
1000 steps on an `N × N` periodic domain after a discarded warm-up:

```
MLUPS = N² · steps / wallclock_s / 1e6
```

Run on the Aqua (QUT) cluster, node `gpu0n009`, an **NVIDIA A100-40GB** (HBM2e,
1.555 TB/s peak bandwidth), CUDA Float64.

## Roofline / reference constants

LBM is bandwidth-bound; a D2Q9 BGK F64 step moves `2 × 9 × 8 = 144 bytes/update`
(9 populations read and written; 304 bytes is the D3Q19 count), so
`MLUPS_ceiling = peak_BW / 144 / 1e6`. The reference bars in the figure are
**literature / derived constants** (documented here, not fitted from the CSV):

| Reference | MLUPS | Type |
|-----------|------:|------|
| A100-40GB roofline (1.555 TB/s) | 10799 | bandwidth ceiling |
| A100-80GB roofline (2.039 TB/s) | 14160 | bandwidth ceiling |
| Palabos single-GPU F64 D3Q19 TRT | 4656 | published real code (Latt et al. 2021) |

## Files

- `certification_a100.csv` — measured run: `N, steps, backend, precision,
  wallclock_s, MLUPS` at N = 1024 and 2048.
- `plot.py` — self-contained reproducer (csv + matplotlib + seaborn). Reads the
  CSV and writes `comparison.png` (the reference ceilings above are encoded as
  documented constants). Run `python plot.py`. No `.krk` case — this benchmark
  uses the `run_taylor_green_2d` Julia driver, not the `.krk` runner.
- `comparison.png` — left: sustained MLUPS vs N (colour = N via `crest`); right:
  Kraken best vs the roofline ceilings and Palabos as a horizontal bar chart.
  (The page figure `docs/src/users/benchmarks/gpu-certification.png` is a separate plot.)

## Headline result

Best sustained throughput is **3461 MLUPS** at N = 2048: **0.32 of the
A100-40GB roofline** (10799 MLUPS ceiling), below the `≥ 0.5 of roofline` gate, and
0.24 of the 80 GB ceiling. The Palabos number (4656 MLUPS) is for D3Q19, which moves
about twice the bytes per update, so the ratio 0.74 to it is not a like-for-like
comparison.

## Reference

Latt, J. et al. (2021), *Palabos: Parallel Lattice Boltzmann Solver*, Comput.
Math. Appl. **81**, 334–350 (single-GPU F64 D3Q19 TRT corroboration point).
