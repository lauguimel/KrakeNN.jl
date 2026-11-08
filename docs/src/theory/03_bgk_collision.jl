# # BGK Collision Operator
#
# The collision operator models molecular interactions that drive the
# distribution function toward local thermodynamic equilibrium.
# The simplest and most widely used model is the **BGK** (Bhatnagar--Gross--Krook)
# single-relaxation-time operator [Bhatnagar et al. (1954)](@cite bgk1954).
#
# ## The BGK relaxation
#
# At each lattice node, the post-collision distribution is:
#
# ```math
# f_q^{\star}(\mathbf{x}, t)
# = f_q(\mathbf{x}, t)
#   - \omega \Big[ f_q(\mathbf{x}, t) - f_q^{\mathrm{eq}}(\rho, \mathbf{u}) \Big]
# ```
#
# where ``\omega = 1/\tau`` is the **relaxation frequency** and ``\tau`` the
# relaxation time (in lattice units). The operator linearly relaxes every
# population toward its equilibrium value at rate ``\omega``.
#
# !!! note "Physical picture"
#     Think of ``\omega`` as a dial: when ``\omega`` is small (large ``\tau``),
#     relaxation is slow and viscosity is high. When ``\omega`` approaches 2,
#     relaxation is fast and viscosity becomes very small.
#
# ## Viscosity--relaxation relation
#
# The Chapman--Enskog expansion (see [LBM Fundamentals](@ref)) yields a direct
# link between ``\omega`` and the kinematic viscosity ``\nu``:
#
# ```math
# \nu = c_s^2 \left( \tau - \frac{1}{2} \right)
#     = \frac{1}{3} \left( \frac{1}{\omega} - \frac{1}{2} \right)
# ```
#
# Inverting:
#
# ```math
# \omega = \frac{2}{6\nu + 1}
# ```
#
# !!! warning "Stability constraint"
#     Since ``\nu > 0`` requires ``\tau > 1/2``, the relaxation frequency must
#     satisfy ``\omega \in (0, 2)``. In practice, ``\omega > 1.9`` leads to
#     severe numerical instability, and values below 1.7 are recommended for
#     production runs.
#
# ## The complete collide-and-stream algorithm
#
# One LBM time step consists of two phases:
#
# 1. **Collision** (local): compute ``f_q^{\star}`` at every node using the BGK
#    formula above.
# 2. **Streaming** (non-local): propagate ``f_q^{\star}`` to neighbouring nodes
#    along ``\mathbf{e}_q``.
#
# ```math
# f_q(\mathbf{x} + \mathbf{e}_q, \, t+1)
# = f_q(\mathbf{x}, t)
#   - \omega \big[ f_q(\mathbf{x}, t) - f_q^{\mathrm{eq}} \big]
# ```
#
# Because collision is purely local (no neighbour access), it maps perfectly
# to GPU threads -- each thread handles one lattice node.
#
# ## Working with KrakeNN
#
# KrakeNN computes ``\omega`` from the viscosity stored in `LBMConfig`:

using KrakeNN

## Create a configuration for a lid-driven cavity
config = LBMConfig(D2Q9(); Nx=64, Ny=64, ν=0.01, u_lid=0.1, max_steps=1000)

## The relaxation frequency
@show omega(config)   # 2 / (6*0.01 + 1) ≈ 1.5385

## The Reynolds number Re = u_lid * Ny / ν
@show reynolds(config)

# The collision kernel `collide_2d!` applies BGK to the entire lattice
# in a single GPU kernel launch. Its signature is:
#
# ```julia
# collide_2d!(f, is_solid, ω)
# ```
#
# Internally, for each node `(i, j)` and each direction `q`:
#
# ```julia
# feq = equilibrium(lattice, ρ[i,j], ux[i,j], uy[i,j], q)
# f_out[i, j, q] = f_in[i, j, q] - ω * (f_in[i, j, q] - feq)
# ```
#
# ## Single-node demonstration
#
# Let us manually perform one BGK collision step on a single node to
# build intuition.

lattice = D2Q9()
ω = omega(config)

## Initial distribution: slightly perturbed from equilibrium
ρ₀, ux₀, uy₀ = 1.0, 0.05, 0.0
f = [equilibrium(lattice, ρ₀, ux₀, uy₀, q) for q in 1:9]
f[2] += 0.01  # add a small perturbation to the East population

## One BGK collision step
feq = [equilibrium(lattice, ρ₀, ux₀, uy₀, q) for q in 1:9]
f_post = f .- ω .* (f .- feq)

## The perturbation has been damped by factor (1 - ω)
@show f[2] - feq[2]       # original perturbation = 0.01
@show f_post[2] - feq[2]  # after collision ≈ 0.01 * (1 - ω)

# ## See in action
#
# - [Lid-driven cavity 2D](../examples/04_cavity_2d.md) — BGK at moderate Re.
# - [Taylor–Green vortex](../examples/03_taylor_green_2d.md) — BGK viscous
#   decay validated against the analytical solution.
