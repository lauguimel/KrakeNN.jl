# # Limitations of Standard LBM
#
# The Lattice Boltzmann Method is powerful, but it comes with inherent
# limitations that practitioners must understand. This page summarises the
# main constraints and how to work within them
# [Kruger et al. (2017)](@cite kruger2017lattice).
#
# ## Compressibility error
#
# LBM solves a weakly compressible form of the Navier--Stokes equations.
# The pressure relates to density through an equation of state:
#
# ```math
# p = \rho \, c_s^2 = \frac{\rho}{3}
# ```
#
# This means density fluctuations act as a proxy for pressure. The error
# compared to a truly incompressible solver scales as:
#
# ```math
# \text{error} = O(\mathrm{Ma}^2), \qquad \mathrm{Ma} = \frac{u}{c_s}
# ```
#
# !!! warning "The Ma < 0.1 rule"
#     To keep compressibility errors below 1%, the Mach number must satisfy
#     ``\mathrm{Ma} < 0.1``, or equivalently ``u < 0.058`` in lattice units.
#     This is the **single most important constraint** in LBM simulations.
#
# In practice, this means the lattice velocity `u_lid`, `u_max`, or any
# characteristic velocity must remain well below ``0.1``:

using KrakeNN

lattice = D2Q9()
cs = sqrt(cs2(lattice))
@show cs                      # ≈ 0.577

## Good: Ma = 0.1/0.577 ≈ 0.17 (borderline)
u_safe = 0.05
@show u_safe / cs

## Bad: Ma = 0.3/0.577 ≈ 0.52 (too high, results will be wrong)
u_bad = 0.3
@show u_bad / cs

# ## Stability: relaxation frequency bound
#
# The BGK relaxation frequency must satisfy:
#
# ```math
# 0 < \omega < 2
# ```
#
# - ``\omega \to 0``: very high viscosity, overdamped (accurate but slow).
# - ``\omega \to 2``: very low viscosity, high Reynolds number (fast but
#   prone to numerical instability).
#
# In practice, ``\omega > 1.95`` almost always leads to divergence.
# For stable production runs, keep ``\omega \leq 1.9``, which corresponds
# to ``\nu \geq 0.00175`` in lattice units.
#
# ```math
# \nu = \frac{1}{3}\left(\frac{1}{\omega} - \frac{1}{2}\right) \geq 0
# ```

## Minimum safe viscosity
ω_max = 1.9
ν_min = (1.0/3.0) * (1.0/ω_max - 0.5)
@show ν_min  # ≈ 0.00877

# ## Why τ must be strictly greater than 1/2
#
# The relaxation time ``\tau`` is the single most important parameter in LBM.
# It controls the kinematic viscosity through the Chapman--Enskog relation:
#
# ```math
# \nu = \frac{1}{3}\left(\tau - \frac{1}{2}\right) \Delta x^2 / \Delta t
# ```
#
# On a standard lattice where ``\Delta x = \Delta t = 1``, this simplifies to
# ``\nu = (\tau - 1/2)/3``.
#
# **Physical interpretation.** After streaming, the populations at each node
# are out of equilibrium. The collision step relaxes them toward ``f^{\text{eq}}``
# at a rate ``\omega = 1/\tau``. A large ``\tau`` means slow relaxation (high
# viscosity, strongly damped). A small ``\tau`` means fast relaxation (low
# viscosity, weakly damped).
#
# **The ``\tau = 1/2`` singularity.** At ``\tau = 1/2`` (``\omega = 2``),
# the collision step *overshoots* equilibrium and exactly mirrors the
# non-equilibrium part:
#
# ```math
# f^{\star} = f - 2(f - f^{\text{eq}}) = 2f^{\text{eq}} - f
# ```
#
# This is an involution (applying it twice returns to the original ``f``).
# There is no net dissipation — ``\nu = 0``. The simulation has zero viscosity,
# meaning even the smallest perturbation is never damped and grows until the
# low-Mach assumption breaks. In practice, ``\tau < 0.5`` gives **negative
# viscosity** (anti-diffusion, exponential blowup).
#
# **On non-uniform meshes.** When cells have different physical sizes
# ``\Delta x_{\text{local}}``, maintaining a uniform target viscosity ``\nu``
# requires adjusting ``\tau`` per cell. For a global time step ``\Delta t``:
#
# ```math
# \tau_{\text{local}} = \frac{3\nu \, \Delta t}{\Delta x_{\text{local}}^2} + \frac{1}{2}
# = \left(\frac{\Delta x_{\text{ref}}}{\Delta x_{\text{local}}}\right)^2 (\tau_{\text{ref}} - \tfrac{1}{2}) + \frac{1}{2}
# ```
#
# Large cells (``\Delta x_{\text{local}} \gg \Delta x_{\text{ref}}``) push
# ``\tau_{\text{local}} \to 1/2``, making those cells near-inviscid and
# numerically unstable. This is the fundamental challenge of the LBM on
# stretched meshes.
#
# **Mitigations:**
#
# | Strategy | Mechanism | Trade-off |
# |:---------|:----------|:----------|
# | ``\tau_{\text{floor}}`` clamp | Set ``\tau \geq 0.55`` everywhere | Adds numerical viscosity at large cells |
# | Limit stretching ratio | Keep ``\max(\Delta x) / \min(\Delta x) \leq 3`` | Limits mesh flexibility |
# | MRT / cumulant collision | Separate physical and ghost relaxation rates | More complex kernel, ~20% slower |
# | Regularized collision | Filter non-hydrodynamic modes before relaxation | Stable at ``\tau \to 0.5``, same cost |
#
# The regularized collision operator (Latt & Chopard 2006) is especially
# effective: it reconstructs the non-equilibrium part solely from the
# physical stress tensor ``\Pi^{(1)}``, discarding the higher-order
# ghost modes that cause instability. The viscosity relation is unchanged,
# so accuracy is preserved.

# ## Resolution requirements
#
# Under grid refinement, the key constraint is that all non-dimensional
# numbers (Re, Ma, Pr) must remain constant. If we refine by a factor 2
# (double ``N``):
#
# - ``\Delta x \to \Delta x / 2``
# - ``u_{\text{latt}} \to u_{\text{latt}} / 2`` (to keep Ma constant)
# - ``\nu_{\text{latt}} \to \nu_{\text{latt}} / 2`` (to keep Re constant)
# - ``\Delta t \to \Delta t / 4`` (since ``\Delta t \propto \Delta x^2 / \nu``)
#
# This means **doubling the resolution costs 8x in 2D** (4x more nodes,
# 2x more time steps) and **16x in 3D** (8x more nodes, 2x more time steps).
#
# !!! tip "Practical consequence"
#     Always run the coarsest grid that gives acceptable accuracy. Use the
#     Ma < 0.1 constraint and the stability limit on ``\omega`` to choose
#     the lattice velocity and viscosity.
#
# ## Grid refinement (under development)
#
# Standard LBM requires a **uniform Cartesian grid**. Unlike finite volume or
# finite element methods, there is no straightforward way to locally refine the
# mesh near walls or features of interest. Multi-block and adaptive approaches
# exist but break the simplicity that makes LBM attractive.
#
# **Grid refinement (patch and conservative AMR) is under development and not
# validated; it is not part of the supported feature set.** Two families of code
# exist in the repository, with their tests, for development purposes:
#
# - Filippova--Hänel patch refinement (nested patches with rescaling at the
#   coarse/fine interface, `.krk` keyword `Refine`). Patch-refined thermal walls
#   are not half-way: in pure conduction, where the exact Nusselt number is 1.0,
#   the patch-refined runs give Nu of about 1.3 (see issue #98).
# - Conservative-tree AMR. Its drag coefficient is off by a factor between 1.06
#   and 5.6 against the reference.
#
# Until these are fixed and validated, do not use refinement for results. For
# local resolution, use a finer uniform grid.
#
# ## Limited to low-Mach flows
#
# Because the equilibrium distribution is a second-order Taylor expansion of
# the Maxwell--Boltzmann distribution, LBM is restricted to low-Mach-number
# flows. Compressible flows (shocks, supersonic regimes) require either
# higher-order equilibria or fundamentally different approaches.
#
# ## Summary of constraints
#
# | Constraint | Requirement | Consequence |
# |:-----------|:------------|:------------|
# | Low Mach | ``\mathrm{Ma} < 0.1`` | ``u_{\text{latt}} < 0.058`` |
# | Stability | ``\omega < 2`` | ``\nu > 0`` in lattice units |
# | Practical stability | ``\omega \leq 1.9`` | ``\nu \geq 0.009`` |
# | Uniform grid | Cartesian, ``\Delta x = \text{const}`` | No local refinement |
# | Refinement cost | ``\propto N^{D+1}`` | Expensive for high Re |
#
# Despite these limitations, LBM remains an excellent choice for
# low-to-moderate Reynolds number flows on regular geometries, especially
# when GPU acceleration makes the explicit time stepping very fast.

# ## See in action
#
# - [Lid-driven cavity 2D](../examples/04_cavity_2d.md) — low-Mach regime
#   where LBM shines.
# - [Taylor–Green vortex](../examples/03_taylor_green_2d.md) — convergence
#   at second order in space and time.
# - [Rayleigh–Bénard convection](../examples/08_rayleigh_benard.md) —
#   coupled multi-physics within the LBM framework.
