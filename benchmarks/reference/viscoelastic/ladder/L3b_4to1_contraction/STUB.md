# L3b — 4:1 planar contraction Oldroyd-B (STUB)

Status: **STUB**. Not implemented in mission M38.

## Why this level

The 4:1 contraction is the canonical industrial-relevance benchmark for
viscoelastic codes: a corner-vortex that grows with De is the standard
qualitative signature. Quantitative pass: corner-vortex length (X_R)
tables match Alves-Pinho-Oliveira 2003.

## Reference

Primary: **Alves, Pinho, Oliveira (2003)** *JNNFM* 110, 45-75. Tabulated
corner-vortex length X_R as a function of De for the planar 4:1
contraction with Oldroyd-B fluid.

Secondary: **rheoTool Contraction41 Oldroyd-BLog** — pre-computed in this
repo at `bench/rheotool/contraction41_oldroydb_log/`. Use as a
second-opinion cross-check independent of Alves-Pinho-Oliveira.

Existing Kraken artefact:
`bench/viscoelastic_logfv/run_contraction41_oldroydb_vs_rheotool.jl`
(LOW-MED robustness per INVENTORY.md; uses rT as ground truth).

## Design sketch

- Geometry: planar 4:1 abrupt contraction. Upstream width 4·H_c, downstream
  width H_c. Channel length 8·H_c upstream, 15·H_c downstream
  (Alves-Pinho-Oliveira mesh M3 or finer).
- Grid: Nx ≈ 600, Ny ≈ 160 (refined near re-entrant corner). MUCH not-cheap.
- Inlet: developed Poiseuille from L1.
- Outlet: zero-gradient.
- Walls: Bouzidi (rectangular but step is grid-aligned, so HWBB is fine
  away from corner; corner needs care).

## Assertions sketch

- Corner-vortex length X_R / H_c matches Alves-Pinho-Oliveira 2003 Table
  3 at De = 1, 3, 5 within 5%.
- Centreline τ_xx peak matches within 10%.

## Cost target

~ 1 h on a single CPU core for one De point. Reasonably needs GPU for a
sweep. Acceptable because L3b is intended as an offline validation, not
CI.

## Existing assets to leverage

- `bench/rheotool/contraction41_oldroydb_log/` — precomputed rT solution.
- `bench/viscoelastic_logfv/run_contraction41_oldroydb_vs_rheotool.jl` —
  existing Kraken vs rT comparison script; promote to L3b's `compare.jl`
  with strict thresholds against Alves-Pinho-Oliveira instead of rT.

## Out of scope until L3b promotion

- 4:1:4 contraction-expansion.
- 3D contractions.
- FENE-P / PTT constitutive variants.
