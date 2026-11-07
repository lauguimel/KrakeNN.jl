# # LBM Fundamentals
#
# The Lattice Boltzmann Method (LBM) is a computational fluid dynamics technique
# rooted in kinetic theory rather than the direct discretisation of the
# Navier--Stokes equations. Instead of solving for macroscopic quantities
# (velocity, pressure) directly, LBM tracks **particle distribution functions**
# that describe the probability of finding a fluid particle with a given velocity
# at a given location and time.
#
# ## From Boltzmann to lattice Boltzmann
#
# The continuous Boltzmann equation governs the evolution of the single-particle
# distribution function ``f(\mathbf{x}, \boldsymbol{\xi}, t)``:
#
# ```math
# \frac{\partial f}{\partial t}
# + \boldsymbol{\xi} \cdot \nabla f
# + \frac{\mathbf{F}}{\rho} \cdot \nabla_{\boldsymbol{\xi}} f
# = \Omega(f)
# ```
#
# where ``\boldsymbol{\xi}`` is the microscopic velocity, ``\mathbf{F}`` an
# external force and ``\Omega`` the collision operator.
# [He & Luo (1997)](@cite he1997theory) showed rigorously that the lattice
# Boltzmann equation can be derived from this continuous equation by discretising
# the velocity space onto a small finite set of vectors.
#
# ## Discretisation steps
#
# The passage from continuous kinetic theory to LBM involves three discretisations:
#
# 1. **Velocity space** -- the continuous ``\boldsymbol{\xi}`` is replaced by a
#    finite set of ``Q`` discrete velocities ``\mathbf{e}_q``, chosen so that
#    enough moments (mass, momentum, stress) are exactly recovered. For 2D flows,
#    the D2Q9 lattice with ``Q = 9`` is the standard choice.
#
# 2. **Physical space** -- the domain is covered by a uniform Cartesian grid with
#    spacing ``\Delta x``. The discrete velocities are chosen so that particles
#    hop exactly one lattice spacing per time step: ``\mathbf{e}_q \, \Delta t = \Delta x``.
#
# 3. **Time** -- a single time step ``\Delta t`` couples collision (local, algebraic)
#    and streaming (shift along ``\mathbf{e}_q``).
#
# The result is the **lattice Boltzmann equation** (LBE):
#
# ```math
# f_q(\mathbf{x} + \mathbf{e}_q \Delta t, \, t + \Delta t)
# = f_q(\mathbf{x}, t) + \Omega_q(\mathbf{x}, t)
# ```
#
# !!! note "Key insight"
#     Collision is purely local (each node updates independently) and streaming
#     is a simple memory shift. Both operations parallelise trivially on GPUs,
#     which is why KrakeNN builds on LBM.
#
# ## Chapman--Enskog analysis
#
# Through a multi-scale expansion (the Chapman--Enskog procedure), one can prove
# that the LBE recovers the weakly compressible Navier--Stokes equations in the
# low-Mach-number limit [Chen & Doolen (1998)](@cite chen1998lattice).
# The macroscopic density and momentum are moments of ``f_q``:
#
# ```math
# \rho = \sum_q f_q, \qquad
# \rho \mathbf{u} = \sum_q f_q \, \mathbf{e}_q
# ```
#
# and the kinematic viscosity is set by the collision parameter (see
# the BGK collision chapter).
#
# ## Why LBM for GPU computing?
#
# | Property | Benefit |
# |:---------|:--------|
# | Local collision | No global linear system to solve |
# | Linear streaming | Regular memory access pattern |
# | Explicit time stepping | No iterative pressure solver |
# | Cartesian grid | Trivial domain decomposition |
#
# For a comprehensive textbook treatment, see
# [Kruger et al. (2017)](@cite kruger2017lattice).
#
# ## Quick look at KrakeNN
#
# KrakeNN exposes lattice metadata through simple queries:

using KrakeNN

lattice = D2Q9()

## Spatial dimension and number of discrete velocities
@show lattice_dim(lattice)   # 2
@show lattice_q(lattice)     # 9

## Speed of sound squared (lattice units)
@show cs2(lattice)           # 1/3

# ## See in action
#
# - [Poiseuille channel](../examples/01_poiseuille_2d.md) — the simplest
#   end-to-end LBM case: D2Q9 + BGK + body force.
# - [Lid-driven cavity 2D](../examples/04_cavity_2d.md) — canonical benchmark
#   exercising streaming, collision and Zou-He walls.
# - [Taylor–Green vortex](../examples/03_taylor_green_2d.md) — analytical
#   reference for second-order convergence.
