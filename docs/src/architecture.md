# Architecture

KrakeNN is a **GPU-first, method-agnostic PDE framework** written in Julia. The
discretization method is a *pluggable backend* selected in the `.krk` case file,
not a hard-wired property of the solver. Every method is meant to share the same
**problem / solution / observable** contract, so the case description — geometry,
boundary conditions, units, and requested output — stays identical no matter which
method runs underneath.

Today, **the Lattice Boltzmann Method (LBM) is the mature production method**: it
is the one path that is shipped and validated across the v0.2 benchmark suite.
Structured-grid finite-volume / finite-difference co-solvers, for the steady and
elliptic regimes that explicit LBM cannot reach efficiently, are on the roadmap —
not finished.

This page is the honest map of where each capability stands. Nothing below the
**Stable** tier should be read as delivered.

## Maturity

The framework is organised in three tiers. The `Status` column is the contract:
**Stable** means shipped and validated in v0.2; **Experimental** means the code is
present but validation is partial or undocumented; **Planned** means designed or
scoped but not shipped.

### Stable — shipped and validated in v0.2

| Capability | Status | Notes |
|---|---|---|
| LBM core | Stable | D2Q9 / D3Q19, BGK; D2Q9 MRT; Guo body forces; axisymmetric |
| Thermal | Stable | double-distribution / Boussinesq, 2D and 3D |
| Viscoelastic | Stable | Oldroyd-B (2D, log-conformation) |
| Boundary conditions | Stable | spatial, Zou-He, bounce-back, periodic |
| Geometry | Stable | STL + voxelization, embedded boundaries |
| Units & DSL | Stable | units system; `.krk` DSL + CLI |
| Backends | Stable | CPU, CUDA, Metal via KernelAbstractions |
| Output | Stable | VTK / PNG / GIF |

### Experimental — code present, partial or undocumented validation

| Capability | Status | Notes |
|---|---|---|
| Generalised-Newtonian / other viscoelastic models | Experimental | power-law, Carreau, Cross, Bingham, Herschel–Bulkley, FENE-P, Saramito — 2D, beyond Oldroyd-B |
| Outflow / Neumann / symmetry BCs | Experimental | partial |
| Grid refinement (patch and conservative AMR) | Under development | not validated; not part of the supported feature set |

### Planned — on the roadmap, NOT shipped

| Capability | Status | Notes |
|---|---|---|
| Method-agnostic platform contract | Planned | `solve` / `sample` / `observe` |
| Steady incompressible Navier–Stokes | Planned | structured FV/FD method (SIMPLE) for the elliptic regime ([issue #7](https://github.com/lauguimel/Kraken.jl/issues/7)) |
| Shared GPU linear-solve service | Planned | matrix-free Poisson ([issue #8](https://github.com/lauguimel/Kraken.jl/issues/8)) |
| Geometry/shape automatic differentiation | Planned | steady shape-adjoint |
| Curvilinear / multi-block grids | Planned | — |
| Multiphase | Planned | VOF / phase-field / Shan-Chen / species |
| Additional GPU backends | Planned | ROCm / oneAPI |

**Out of scope:** unstructured FEM/DG; general unstructured-mesh FV CFD; CAD mesh
authoring.

## Choosing a method

In the intended design, the discretization method is a `.krk` choice, not a fork
of the codebase. New physics and new methods drop in as modules behind the shared
problem / solution / observable contract, so a case written for one method needs no
rewrite to target another.

State plainly what that means **today**: the only production method is LBM. The
structured FV/FD steady path and the method-agnostic platform contract itself are
on the roadmap (see the **Planned** tier above), not finished. The framing is the
direction of travel; the **Stable** table is what v0.2 actually ships.
