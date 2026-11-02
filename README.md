# Kraken.jl

[![Build Status](https://github.com/lauguimel/Kraken.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/lauguimel/Kraken.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://lauguimel.github.io/Kraken.jl/stable)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A GPU-portable Lattice Boltzmann (LBM) framework written in Julia, targeting
single-phase incompressible, thermal, and viscoelastic flows on CPU, CUDA, and
Apple Metal backends. Kraken.jl provides a compact kernel core (D2Q9, D3Q19), a
declarative `.krk` configuration DSL, patch-based grid refinement, and
spatial boundary conditions — all behind a small, hackable API.

## Scope (v0.3.0)

- Single-phase LBM in 2D and 3D (BGK and MRT collisions)
- Thermal flows via double-distribution (DDF) coupling
- Patch-based nested grid refinement (Filippova–Hänel rescaling)
- Spatial boundary conditions (Zou–He, bounce-back, periodic, outflow)
- `.krk` configuration DSL for parametric runs
- GPU-portable kernels via `KernelAbstractions.jl` (CPU / CUDA / Metal)
- Viscoelastic Oldroyd-B (log-conformation) — validated to <1% vs RheoTool on the confined cylinder
- Physical ↔ lattice unit-conversion module; STL / analytical immersed-boundary geometry

Multiphase flows and the broader non-Newtonian rheology family are present in
the source tree but remain experimental and are not part of the validated
v0.3.0 surface.

For a complete, up-to-date feature matrix — with status (✓/~/✗), links to
theory pages, examples, and API — see the
[**Capabilities matrix**](docs/src/capabilities.md).

## Installation

Kraken.jl is not yet registered. Install from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/lauguimel/Kraken.jl")
```

For GPU execution, also add the backend of your choice (`CUDA.jl` or
`Metal.jl`).

## Quickstart

Run a 2D lid-driven cavity from a `.krk` configuration file:

```julia
using Kraken

# Parse the .krk file into a SimulationSetup
setup = load_kraken("examples/cavity.krk")

# Run it (CPU by default; a `Backend cuda`/`Backend metal` line in the .krk
# file, or an explicit `backend=CUDABackend()`/`MetalBackend()` kwarg, moves
# it to a GPU)
result = run_simulation(setup)

ux, uy = result.ux, result.uy
```

Output (`examples/cavity.krk`, 128×128, Re = 100, CPU, 60 000 steps):

```
max |u| = 0.10000000000000006   mean |u| = 0.01975436420203213
```

Or build a case directly from Julia, without the `.krk` DSL:

```julia
using Kraken

config = LBMConfig(D2Q9(); Nx=32, Ny=32, ν=0.1, u_lid=0.1, max_steps=100)
result = run_cavity_2d(config)

ux, uy = result.ux, result.uy
```

Output:

```
max |ux| = 0.1000000000000001
```

See `docs/src/getting_started.md` and the `examples/` directory for more
cases (Poiseuille, Couette, Taylor–Green, cylinder flow, Rayleigh–Bénard,
Hagen–Poiseuille, 3D cavity, and grid-refinement demos).

## Features

- D2Q9 and D3Q19 lattices
- BGK and MRT collision operators
- Thermal coupling via double-distribution functions
- Patch-based nested grid refinement with conservative rescaling
- Declarative `.krk` DSL for reproducible parametric studies
- GPU-portable kernels: single source, runs on CPU / CUDA / Metal
- VTK output for ParaView post-processing
- ~2000 unit tests covering kernels, BCs, refinement, and end-to-end drivers

## Documentation

Full documentation (theory, examples, API reference) is built with
Documenter.jl:

```bash
julia --project=docs docs/make.jl
```

## Citation

If you use Kraken.jl in academic work, please cite:

```bibtex
@software{kraken_jl,
  author  = {Maitrejean, Guillaume and Zhao, Jiachen and Sauret, Emilie},
  title   = {Kraken.jl: a GPU-portable Lattice Boltzmann framework in Julia},
  year    = {2026},
  url     = {https://github.com/lauguimel/Kraken.jl},
  version = {0.3.0},
}
```

## License

Kraken.jl is released under the MIT License. See [LICENSE](LICENSE) for
details.
