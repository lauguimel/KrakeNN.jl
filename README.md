# Kraken.jl

[![Build Status](https://github.com/lauguimel/Kraken.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/lauguimel/Kraken.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://lauguimel.github.io/Kraken.jl/stable)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A GPU-portable Lattice Boltzmann (LBM) framework written in Julia, targeting
single-phase incompressible, thermal, and viscoelastic flows on CPU, CUDA, and
Apple Metal backends. Kraken.jl provides a compact kernel core (D2Q9, D3Q19), a
declarative `.krk` configuration DSL and
spatial boundary conditions — all behind a small, hackable API.

## Scope

- Single-phase LBM in 2D and 3D (BGK; MRT in 2D; TRT, see the capabilities matrix for where)
- Thermal flows via double-distribution (DDF) coupling
- Spatial boundary conditions (Zou–He, bounce-back, periodic, outflow)
- `.krk` configuration DSL for parametric runs
- GPU-portable kernels via `KernelAbstractions.jl` (CPU / CUDA / Metal)
- Viscoelastic Oldroyd-B (log-conformation) — drag within 1% of stored RheoTool values on the confined cylinder at Wi ≤ 1 (see the cylinder page for the provenance of those references)
- Physical ↔ lattice unit-conversion module; STL / analytical immersed-boundary geometry

Multiphase flows and the broader non-Newtonian rheology family are present in
the source tree but remain experimental and are not part of the validated
surface.

Grid refinement (patch and conservative AMR) is under development and not validated; it is not part of the supported feature set.

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

From a clone of the repository (`git clone https://github.com/lauguimel/Kraken.jl`),
run `examples/quickstart.jl` on CPU. It runs a lid-driven cavity from the
`examples/cavity.krk` configuration file, then the same kind of case built from
Julia, and stops with an error if the flow is not finite. CI runs this exact file.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/quickstart.jl
```

```julia
# Kraken.jl quickstart: the README shows this file verbatim and CI runs it on CPU.
# From a clone of the repository:
#
#     julia --project=. examples/quickstart.jl
using Kraken

# 1. Run a lid-driven cavity described by a .krk configuration file.
#    `max_steps` overrides the file's `Run 60000 steps` so the script finishes
#    in a few seconds on a laptop CPU.
krk_file = joinpath(@__DIR__, "cavity.krk")
result = run_simulation(krk_file; max_steps = 2000)
println("krk cavity : max |ux| = ", maximum(abs, result.ux))

# 2. The same kind of case built from Julia, without the .krk file.
config = LBMConfig(D2Q9(); Nx = 32, Ny = 32, ν = 0.1, u_lid = 0.1, max_steps = 100)
cavity = run_cavity_2d(config)
println("julia cavity: max |ux| = ", maximum(abs, cavity.ux))

# Sanity checks: the script fails (non-zero exit) if the flow blew up.
@assert all(isfinite, result.ux) && all(isfinite, result.uy)
@assert all(isfinite, cavity.ux) && all(isfinite, cavity.uy)
@assert maximum(abs, cavity.ux) > 0
println("quickstart OK")
```

Expected output (about 20 s the first time, most of it compilation):

```
krk cavity : max |ux| = 0.10000000000000012
julia cavity: max |ux| = 0.1000000000000001
quickstart OK
```

To use a GPU, add a `Backend cuda` (or `Backend metal`) line to the `.krk` file, or
pass `backend = CUDABackend()` to `run_simulation`.

See `docs/src/getting_started.md` and the `examples/` directory for more
cases (Poiseuille, Couette, Taylor–Green, cylinder flow, Rayleigh–Bénard,
Hagen–Poiseuille, and 3D cavity).

## Features

- D2Q9 and D3Q19 lattices
- BGK and MRT collision operators
- Thermal coupling via double-distribution functions
- Declarative `.krk` DSL for reproducible parametric studies
- GPU-portable kernels: single source, runs on CPU / CUDA / Metal
- VTK output for ParaView post-processing
- ~2000 unit tests covering kernels, BCs, and end-to-end drivers

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
  version = {0.6.0},
}
```

## License

Kraken.jl is released under the MIT License. See [LICENSE](LICENSE) for
details.
