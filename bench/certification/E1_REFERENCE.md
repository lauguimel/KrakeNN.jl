# E1 — Single-GPU throughput certification (Kraken CUDA-F64 BGK D2Q9)

This document fixes the reference numbers the E1 certification benchmark
(`cavity_bgk_cuda/run_certification.jl`) is graded against. The benchmark
reports sustained **MLUPS** (Mega Lattice Updates Per Second) for the Newtonian
BGK D2Q9 solver in **Float64**; this page tells you what "good" means.

## 1. Roofline ceiling (primary, self-contained)

LBM is memory-bandwidth bound. For a D2Q9 BGK step in F64 the dominant traffic
per lattice update is reading + writing the q=9 populations (9 in, 9 out), i.e.
**2 × 9 × sizeof(Float64) = 144 bytes/update**. (An earlier version of this file used
304 bytes, which is the D3Q19 traffic, 2 × 19 × 8; that overstated the byte count
and understated the ceiling by a factor 2.1.) Hence:

```
max_updates_per_s = peak_BW / (2 * 9 * sizeof(Float64))
                  = peak_BW / 144
MLUPS_ceiling     = max_updates_per_s / 1e6
```

Plugging in vendor peak HBM bandwidths:

| GPU            | Memory | Peak BW    | F64 roofline MLUPS |
|----------------|--------|------------|--------------------|
| **H100**       | HBM3   | ~3.35 TB/s | **~23 264 MLUPS**  |
| **A100-80GB**  | HBM2e  | ~2.039 TB/s | **~14 160 MLUPS** |
| **A100-40GB**  | HBM2e  | ~1.555 TB/s | **~10 799 MLUPS** |

These are hard upper bounds; no F64 D2Q9 kernel can exceed them.

## 2. Published F64 corroboration (recommended primary citation)

**Palabos** sustains **~73 % of A100 FP64 peak** for D3Q19 TRT
(Latt et al., *Comput. Math. Appl.* 2021; PASC '22), i.e. **≈ 4 656 MLUPS**
single-GPU in F64. This is the recommended real-code corroboration point: it is
F64 and directly comparable to the Kraken gate.

## 3. F32 context (NOT comparable — FP32, lower precision)

For situational awareness only — **do not** compare these to the Kraken F64
gate, they are single precision and move half the bytes per update:

**FluidX3D**, FP32, D3Q19 (Lehmann, arXiv:2112.08926):
**A100 ≈ 10 228 MLUPS**, **H100 ≈ 17 602 MLUPS**.

Labelled clearly as FP32 to avoid an apples-to-oranges pass/fail.

## 4. Certification criterion

The certification compares the measured **Kraken CUDA-F64 MLUPS** against:

1. the **roofline ceiling** of the GPU it ran on (Section 1) — the primary,
   self-contained reference; and
2. the **Palabos F64** single-GPU number (Section 2) — independent real-code
   corroboration.

**Pass = ratio ≥ 0.5** of the roofline ceiling (i.e. Kraken is within 2× of the
memory-bandwidth limit for its GPU). On an H100 that means ≥ ~11 630 MLUPS; on an
A100-40GB ≥ ~5 400 MLUPS; on an A100-80GB ≥ ~7 080 MLUPS. The measured Kraken run
(3461 MLUPS on an A100-40GB) is 0.32 of the ceiling and does not meet this gate. The
Palabos figure is D3Q19, which moves about twice the bytes of D2Q9, so it is context,
not a like-for-like comparison.
