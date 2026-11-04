# L3a — backward-facing step Oldroyd-B (STUB)

Status: **STUB**. Not implemented in mission M38.

## Why this level

The first geometry with a singular point (the step corner). Tests how
the constitutive law + log-conformation handle a stress concentration
that grows unbounded in the inviscid limit. Used as a discriminator for
log-conformation stabilisation in the literature.

## Reference (proposed)

Primary candidate: **rheoTool BFS Oldroyd-BLog** (no precomputed case
ships with rheoTool tutorials; would need to construct a setup mirroring
the contraction41 case but with a step downstream).

Literature: Alves et al. (2008) discuss BFS for various constitutive
models; Pinho & Whitelaw (1990) provide experimental data for the
Newtonian baseline.

Status of reference: **NEEDS RESEARCH**. The first L3a follow-up mission
should select the primary reference and document it before any
implementation. Candidates to evaluate:
1. rheoTool BFS converged run + paper cross-check
2. Alves 2008 (specific paper TBD; multiple Alves group BFS papers exist)
3. Cross-validation against Basilisk if a BFS case is in
   `/Users/guillaume/Documents/Recherche/Codes CFD/basilisk/src/test/`.

## Design sketch

- Geometry: 2D channel with step at x = x_step; H_up / H_down = 1/2 or
  1/3 (set by reference choice).
- Inlet: developed Poiseuille profile from L1.
- Outlet: zero-gradient.
- Walls: HWBB or Bouzidi (depending on whether step is grid-aligned).
- Discretisation: Nx ≈ 200, Ny ≈ 60 (downstream of step).

## Assertions sketch

- Recirculation length downstream of the step matches reference within
  5%.
- Maximum |τ_xx| at step corner matches reference within 10%.
- Centreline velocity profile downstream matches L1 Poiseuille within
  fully-developed length.

## Cost target

~ 30 min on a single CPU core for one (β, Wi) point. Likely needs GPU
for any parametric sweep.

## Out of scope until L3a promotion

- Wi > 1.
- 3D variants.
