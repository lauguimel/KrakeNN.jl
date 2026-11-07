# Voie D Phase P Milestone

## Publishable Claim

Kraken.jl now has a publishable Phase P target for voie D:

```text
fixed ratio-2 patch refinement for D2Q9 LBM using integrated populations
F_i = f_i * cell_volume, conservative composite ownership, and an explicit
leaf-grid oracle.
```

This is not a native dynamic AMR solver yet. The hot path still validates via
projection to a uniform leaf grid and restriction back to the conservative
composite state.

## Validation Set

The Phase P validation set is intentionally limited to cases already exercised
by the conservative-tree path:

- Couette central patch, analytic profile.
- Poiseuille vertical refinement band at `x=L/2`, analytic profile.
- Poiseuille horizontal refinement band at `y=L/2`, analytic profile.
- Square obstacle fully enclosed by the refined patch, drag/lift comparison
  against a coarse Cartesian run.
- Backward-facing step, called BFS/VFS in session notes, as an open-channel
  velocity-field comparison against a Cartesian leaf-grid run.

Cylinder is not part of the Phase P publication gate.

## Current Reproducible Artifacts

Run:

```bash
julia --project=. scripts/figures/plot_voie_d_phase_p.jl
```

This command is both an artifact generator and a numerical gate. It raises an
assertion failure if the current Couette, Poiseuille, square-obstacle, or
BFS/VFS metrics leave the Phase P acceptance window.

Generated artifacts:

```text
paper/figures/voie_d_poiseuille_bands.pdf
paper/figures/voie_d_poiseuille_bands.png
paper/figures/voie_d_square_obstacle.pdf
paper/figures/voie_d_square_obstacle.png
paper/figures/voie_d_couette_bfs.pdf
paper/figures/voie_d_couette_bfs.png
paper/data/voie_d_phase_p_summary.md
```

Focused test gate:

```bash
julia --project=. -e 'using Test; using KrakeNN; include("test/test_conservative_tree_2d.jl")'
julia --project=. -e 'using Test; using KrakeNN; include("test/test_conservative_tree_topology_2d.jl")'
```

The figure gate is narrower and closer to the publication claim. The test gate
is broader and also protects lower-level conservative-tree primitives.

## Current Numerical Gate

Latest generated summary:

```text
Couette central patch:
  mass drift = -5.21e-11
  L2 error   = 3.95e-4
  Linf error = 6.35e-4

Poiseuille vertical band:
  mass drift = -9.64e-11
  L2 error   = 1.198e-3
  Linf error = 2.375e-3

Poiseuille horizontal band:
  mass drift = -1.00e-10
  L2 error   = 1.126e-3
  Linf error = 2.375e-3

Square obstacle:
  refined/coarse Fx ratio = 0.986428
  refined Fy/Fx           ~ 0

BFS/VFS:
  ux mean delta vs Cartesian leaf = 1.30e-4
  uy mean delta vs Cartesian leaf = -1.77e-5
```

For BFS/VFS, the current gate is velocity/field agreement under open-channel
boundary conditions, not exact mass conservation.

The scripted acceptance gates are:

```text
Couette:
  |mass drift| < 1e-8
  L2 < 1e-3
  Linf < 2e-3

Poiseuille vertical/horizontal bands:
  |mass drift| < 1e-8
  L2 < 2e-3
  Linf < 3e-3

Square obstacle:
  |mass drift| < 1e-8
  0.85 < Fx_refined/Fx_coarse < 1.15
  |Fy/Fx| < 1e-10

BFS/VFS:
  |ux_mean - ux_cart| < 5e-4
  |uy_mean - uy_cart| < 5e-5
  |mass_final - mass_final_cart| < 1.0
```

## Non-Claims

Phase P does not claim:

- dynamic adaptation;
- native `dx`-local AMR streaming;
- temporal subcycling;
- optimized GPU AMR kernels;
- D3Q19 or 3D octree support;
- cylinder validation.

The next milestone after Phase P is route-driven native composite streaming
without collision, compared directly against the current leaf-grid oracle.
