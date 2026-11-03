```@raw html
---
layout: home

hero:
  name: "Kraken.jl"
  text: "GPU-native, method-agnostic multiphysics in Julia"
  tagline: "A method-agnostic PDE framework whose mature core method is Lattice Boltzmann. Write a reproducible .krk case or a direct Julia driver, then run it on CUDA, Metal, or CPU backends."
  image:
    src: /logo.png
    alt: Kraken
  actions:
    - theme: brand
      text: Get started
      link: /getting_started
    - theme: alt
      text: KRK reference
      link: /users/krk-reference
    - theme: alt
      text: Validation matrix
      link: /users/benchmarks/validation-matrix
    - theme: alt
      text: Cite
      link: /#citing-kraken

features:
  - title: Reproducible cases
    details: Canonical flows can be launched from declarative .krk files or direct Julia drivers.
    link: /examples/04_cavity_2d
  - title: Validated multiphysics
    details: Cavity, thermal, sphere-drag, and viscoelastic checks are tracked in the v0.2 validation matrix.
    link: /users/benchmarks/validation-matrix
  - title: GPU certification
    details: Backend checks document CPU, CUDA-class, and Apple Silicon execution paths.
    link: /users/benchmarks/gpu-certification
---
```

```@raw html
<div align="center">
  <img src="./assets/showcases/vonkarman_re200.gif" alt="Von Kármán vortex street past a cylinder, simulated with Kraken.jl" width="100%"/>
  <br/><em>Von Kármán vortex street (Re = 200) — D2Q9 LBM on GPU.</em>
</div>
```

## Quick start

```julia
using Kraken

# Lid-driven cavity at Re = 100 on a 128×128 grid
N = 128
ν = 0.1 * N / 100  # ν = u_lid · N / Re
config = LBMConfig(D2Q9(); Nx=N, Ny=N, ν=ν, u_lid=0.1, max_steps=30000)
result = run_cavity_2d(config)
```

The same simulation can be written declaratively in a `.krk` file and launched
from a shell:

```
# cavity.krk — lid-driven cavity at Re = 100
Simulation cavity D2Q9
Domain     L = 1.0 x 1.0   N = 128 x 128
Physics    nu = 0.128
Boundary   north velocity(ux = 0.1, uy = 0)
Boundary   south wall
Boundary   east  wall
Boundary   west  wall
Run        30000 steps
```

```bash
krk cavity.krk
```

## What Kraken can do

```@raw html
<div class="kraken-caps">
  <a class="kraken-cap" href="/examples/04_cavity_2d">
    <span class="kraken-cap-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12.8 19.6A2 2 0 1 0 14 16H2"/><path d="M17.5 8a2.5 2.5 0 1 1 2 4H2"/><path d="M9.8 4.4A2 2 0 1 1 11 8H2"/></svg></span>
    <span class="kraken-cap-title">Newtonian flow</span>
    <span class="kraken-cap-sub">Incompressible 2D · 3D · axisymmetric — BGK &amp; MRT</span>
  </a>
  <a class="kraken-cap" href="/examples/08_rayleigh_benard">
    <span class="kraken-cap-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z"/></svg></span>
    <span class="kraken-cap-title">Thermal convection</span>
    <span class="kraken-cap-sub">Boussinesq natural convection, coupled DDF</span>
  </a>
  <a class="kraken-cap" href="/users/benchmarks/viscoelastic-cylinder">
    <span class="kraken-cap-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2 12q2.5 2 5 0t5 0 5 0 5 0"/><path d="M2 19q2.5 2 5 0t5 0 5 0 5 0"/><path d="M2 5q2.5 2 5 0t5 0 5 0 5 0"/></svg></span>
    <span class="kraken-cap-title">Viscoelastic</span>
    <span class="kraken-cap-sub">Oldroyd-B polymer stress, validated vs RheoTool</span>
  </a>
  <a class="kraken-cap" href="/capabilities#5-geometry-and-obstacles">
    <span class="kraken-cap-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/></svg></span>
    <span class="kraken-cap-title">Complex geometry</span>
    <span class="kraken-cap-sub">STL import, voxelisation, interpolated bounce-back</span>
  </a>
  <a class="kraken-cap" href="/users/benchmarks/gpu-certification">
    <span class="kraken-cap-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 20v2"/><path d="M12 2v2"/><path d="M17 20v2"/><path d="M17 2v2"/><path d="M2 12h2"/><path d="M2 17h2"/><path d="M2 7h2"/><path d="M20 12h2"/><path d="M20 17h2"/><path d="M20 7h2"/><path d="M7 20v2"/><path d="M7 2v2"/><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="8" y="8" width="8" height="8" rx="1"/></svg></span>
    <span class="kraken-cap-title">GPU-native</span>
    <span class="kraken-cap-sub">One kernel layer — CUDA · Apple Metal · CPU</span>
  </a>
  <a class="kraken-cap" href="/users/krk-reference">
    <span class="kraken-cap-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z"/><path d="M14 2v5a1 1 0 0 0 1 1h5"/><path d="M10 12.5 8 15l2 2.5"/><path d="m14 12.5 2 2.5-2 2.5"/></svg></span>
    <span class="kraken-cap-title">.krk language</span>
    <span class="kraken-cap-sub">Declarative cases, no Julia required</span>
  </a>
  <a class="kraken-cap" href="/users/benchmarks/validation-matrix">
    <span class="kraken-cap-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3.85 8.62a4 4 0 0 1 4.78-4.77 4 4 0 0 1 6.74 0 4 4 0 0 1 4.78 4.78 4 4 0 0 1 0 6.74 4 4 0 0 1-4.77 4.78 4 4 0 0 1-6.75 0 4 4 0 0 1-4.78-4.77 4 4 0 0 1 0-6.76Z"/><path d="m9 12 2 2 4-4"/></svg></span>
    <span class="kraken-cap-title">Validated</span>
    <span class="kraken-cap-sub">Every benchmark tracked against a literature reference</span>
  </a>
</div>
```

These are the capabilities of Kraken's mature LBM core. For how LBM fits into the
broader method-agnostic platform — and what is shipped versus on the roadmap — see
the [architecture overview](architecture.md). See the
[full capabilities matrix](capabilities.md) for the complete module breakdown, and
the [validation matrix](users/benchmarks/validation-matrix.md) for every literature
reference.

## Showcase gallery

```@raw html
<div align="center">
  <img src="./assets/showcases/cavity_re1000.gif" alt="Lid-driven cavity at Re = 1000" width="32%"/>
  <img src="./assets/showcases/rayleigh_benard_ra1e5.gif" alt="Rayleigh–Bénard convection at Ra = 1e5" width="32%"/>
  <img src="./assets/showcases/taylor_green_decay.gif" alt="Taylor–Green vortex decay" width="32%"/>
  <br/>
  <em>Left: lid-driven cavity at Re = 1000 (primary vortex + corner eddies).
  Centre: Rayleigh–Bénard convection cells at Ra = 1e5.
  Right: Taylor–Green vortex decay.</em>
</div>
```

Every animation on this page is produced by the validated drivers in the table
above; reproduce them from the [examples](examples/04_cavity_2d.md) and the
[validation matrix](users/benchmarks/validation-matrix.md).

## Citing Kraken

If you use Kraken.jl in your research, please cite it. The repository ships a
[`CITATION.cff`](https://github.com/lauguimel/Kraken.jl/blob/main/CITATION.cff),
so GitHub's **"Cite this repository"** button (top-right of the
[repository](https://github.com/lauguimel/Kraken.jl)) gives a ready-made entry.

```bibtex
@software{kraken_jl,
  author  = {Maitrejean, Guillaume and Zhao, Jiachen and Sauret, Emilie},
  title   = {{Kraken.jl}},
  year    = {2026},
  version = {0.5.1},
  doi     = {10.5281/zenodo.22747126},
  url     = {https://github.com/lauguimel/Kraken.jl}
}
```

Each release is archived on Zenodo under two DOIs. Use
[`10.5281/zenodo.22747126`](https://doi.org/10.5281/zenodo.22747126) — the one
above — to cite the software in general: it always resolves to the most recent
archived release. Use the version-specific DOI instead when the exact code
matters for reproducibility; for `v0.5.1` that is
[`10.5281/zenodo.23598327`](https://doi.org/10.5281/zenodo.23598327), and each
Zenodo record lists its own.
