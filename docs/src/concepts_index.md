# Concepts index

Kraken is a method-agnostic PDE framework; the Lattice Boltzmann Method
is its mature core method, and it is the one this page maps. For how LBM
sits within the broader platform — and what is shipped versus on the
roadmap — see the [architecture overview](architecture.md).

This page is a map of the Lattice Boltzmann concepts implemented in
Kraken.jl v0.2.0. Each entry briefly states what the concept is and why
it matters, links to the theory page that derives it, and points to
example-tutorials that exercise it. If you are looking for a specific
topic, this is the fastest route in.

If you are brand new to LBM, start at
[Getting started](getting_started.md) first, then come back here.

## Core LBM

### Lattices and equilibrium

LBM evolves discrete populations `f_i(x, t)` on a stencil of velocities.
In 2D Kraken uses the nine-velocity D2Q9 stencil; in 3D it uses D3Q19.
Equilibrium is a Mach-expanded Maxwell–Boltzmann distribution that
recovers Navier–Stokes in the hydrodynamic limit.

- Theory: [LBM fundamentals](theory/01_lbm_fundamentals.md),
  [D2Q9 lattice](theory/02_d2q9_lattice.md)
- Examples: [Poiseuille 2D](examples/01_poiseuille_2d.md),
  [Taylor–Green vortex](examples/03_taylor_green_2d.md)

### Streaming

The advection step: post-collision populations are shifted to their
neighbor along each lattice direction. This is the only non-local part
of the algorithm.

- Theory: [Streaming](theory/04_streaming.md)
- Examples: [Couette flow](examples/02_couette_2d.md),
  [Lid-driven cavity 2D](examples/04_cavity_2d.md)

### Collision (BGK)

A single-relaxation-time model that drives the populations toward their
local equilibrium. The relaxation parameter `τ` is set by the kinematic
viscosity.

- Theory: [BGK collision](theory/03_bgk_collision.md)
- Examples: [Lid-driven cavity 2D](examples/04_cavity_2d.md),
  [Flow past a cylinder](examples/06_cylinder_2d.md)

### MRT collision

Multiple-relaxation-time collision relaxes each moment of the
distribution at its own rate. It is more stable than BGK at high
Reynolds number and removes some of BGK's viscosity-dependent boundary
errors.

- Theory: [MRT](theory/12_mrt.md)
- Examples: [Lid-driven cavity 2D](examples/04_cavity_2d.md),
  [Flow past a cylinder](examples/06_cylinder_2d.md)

## Boundary conditions

### Zou–He, bounce-back, periodic

Zou–He reconstructs unknown populations from prescribed macroscopic
moments (velocity or pressure). Bounce-back is the simplest no-slip
wall. Periodic BCs connect opposite faces without any reconstruction.

- Theory: [Boundary conditions](theory/05_boundary_conditions.md)
- Examples: [Poiseuille 2D](examples/01_poiseuille_2d.md),
  [Lid-driven cavity 2D](examples/04_cavity_2d.md)

### Spatially varying BCs

For non-trivial inlet profiles, temperature patches, or moving walls,
Kraken lets you attach a function `f(x, y)` or a STL region to a face
via the `.krk` DSL.

- Theory: [Spatial BCs](theory/19_spatial_bcs.md)
- Examples: [Flow past a cylinder](examples/06_cylinder_2d.md),
  [.krk configuration](examples/10_krk_config.md)

## Body forces

### Guo forcing

Guo et al.'s scheme adds a body force to the LBM step in a way that
stays consistent with the Chapman–Enskog expansion — no spurious
second-order error. Kraken uses it for buoyancy in thermal cases and
for driven periodic flows.

- Theory: [Body forces](theory/07_body_forces.md)
- Examples: [Hagen–Poiseuille](examples/09_hagen_poiseuille.md),
  [Rayleigh–Bénard convection](examples/08_rayleigh_benard.md)

## Thermal coupling

### Double distribution function, Boussinesq

Temperature is evolved with a second LBM distribution on the same
lattice. The temperature field couples back into the flow via a
Boussinesq body force on the velocity populations.

- Theory: [Thermal DDF](theory/08_thermal_ddf.md)
- Examples: [Heat conduction](examples/07_heat_conduction.md),
  [Rayleigh–Bénard convection](examples/08_rayleigh_benard.md)

## Axisymmetric LBM

For pipe flow and other rotationally symmetric problems, Kraken
implements the axisymmetric source-term formulation on a 2D `(r, z)`
grid, avoiding a full 3D run.

- Theory: [Axisymmetric LBM](theory/09_axisymmetric.md)
- Examples: [Hagen–Poiseuille](examples/09_hagen_poiseuille.md)

## Grid refinement (under development)

Grid refinement (patch and conservative AMR) is under development and not validated; it is not part of the supported feature set.
See [Limitations](theory/10_limitations.md).

## From 2D to 3D

The D2Q9 → D3Q19 step is mostly mechanical (same collision, same
streaming, bigger stencil), but the memory footprint and the boundary
bookkeeping grow quickly. This page explains what changes and what
does not.

- Theory: [From 2D to 3D](theory/06_from_2d_to_3d.md)
- Examples: [Lid-driven cavity (2D & 3D)](examples/04_cavity_2d.md)

## Limitations

For the honest list of what is and is not in v0.2.0 — including
multiphase, non-Newtonian, and high density-ratio flows, all of which
are deferred — see [Limitations](theory/10_limitations.md).
