# # Shan-Chen Pseudo-Potential Multiphase
#
# The Shan-Chen model is the simplest approach to multiphase flow in the
# LBM framework.  Instead of tracking an explicit interface, an
# **inter-particle force** is introduced that depends on a
# pseudo-potential function ``\psi(\rho)``.  Phase separation emerges
# spontaneously from the interaction, without any interface reconstruction.
#
# **References**:
# Shan & Chen (1993) [shan1993lattice](@cite shan1993lattice),
# Shan & Chen (1994) [shan1994simulation](@cite shan1994simulation),
# Krüger et al. (2017) [kruger2017lattice](@cite kruger2017lattice) §9.
#
# ## Inter-particle force
#
# The Shan-Chen interaction force at node ``\mathbf{x}`` reads:
# ```math
# \mathbf{F}(\mathbf{x}) = -G\,\psi(\mathbf{x})
#     \sum_\alpha w_\alpha\,\psi(\mathbf{x} + \mathbf{e}_\alpha)\,
#     \mathbf{e}_\alpha
# ```
#
# where ``G`` is the interaction strength, ``\psi(\rho)`` is the
# pseudo-potential, ``\mathbf{e}_\alpha`` are the lattice velocities,
# and ``w_\alpha`` the lattice weights.  The sum runs over nearest and
# next-nearest neighbours (D2Q9 or D3Q19).
#
# - ``G < 0``: attraction → liquid-gas phase separation
# - ``G > 0``: repulsion (used in multi-component models)
#
# The force computation is a **stencil operation** with only nearest-neighbour
# reads, making it ideal for GPU parallelization.
#
# ## Pseudo-potential function
#
# The standard Shan-Chen potential is:
# ```math
# \psi(\rho) = \rho_0 \left(1 - e^{-\rho / \rho_0}\right)
# ```
#
# where ``\rho_0`` is a reference density.  At low density,
# ``\psi \approx \rho`` (ideal gas); at high density, ``\psi`` saturates
# at ``\rho_0``, creating an effective equation of state with a van der
# Waals-like loop.
#
# ```julia
# compute_psi_2d!(ψ, ρ, ρ0)
# ```
#
# ## Equation of state
#
# The macroscopic pressure in the Shan-Chen model is:
# ```math
# p = \rho\,c_s^2 + \frac{G\,c_s^2}{2}\,\psi^2(\rho)
# ```
#
# For ``G < 0`` (attractive interactions), this EOS can exhibit a
# non-monotonic region, leading to **spinodal decomposition**: an initially
# uniform density field spontaneously separates into coexisting liquid
# and gas phases.
#
# The coexistence densities ``\rho_l`` and ``\rho_g`` are determined by
# the **Maxwell equal-area construction** applied to the mechanical EOS.
# They depend on ``G`` and ``\rho_0``, and can be computed numerically
# or looked up from tabulated values.
#
# ## Coexistence curve
#
# For a given ``G`` and ``\rho_0``, the coexistence liquid and gas
# densities can be found by solving the Maxwell equal-area condition:
# ```math
# \int_{\rho_g}^{\rho_l} \left(p(\rho) - p_{\mathrm{sat}}\right)
#     \frac{d\rho}{\rho^2} = 0, \qquad
# p(\rho_l) = p(\rho_g) = p_{\mathrm{sat}}
# ```
#
# In practice, this is done numerically.  For the standard exponential
# pseudo-potential with ``\rho_0 = 1``, the critical ``G`` value is
# approximately ``G_c \approx -4``, below which phase separation occurs.
#
# ## Thermodynamic consistency
#
# The standard Shan-Chen model is **not thermodynamically consistent**:
# the surface tension and coexistence densities cannot be independently
# controlled.  This limits accuracy but makes the model extremely simple
# to implement and efficient to run.
#
# For applications requiring better thermodynamic consistency (e.g. high
# density ratios, controlled surface tension), KrakeNN provides the
# **phase-field** approach (see [Phase-Field LBM](@ref)) as an
# alternative.
#
# ## Force incorporation: Guo scheme
#
# The Shan-Chen force is incorporated into the collision step using the
# **Guo forcing scheme** [guo2002discrete](@cite guo2002discrete).
# The velocity used in the equilibrium is shifted by the force:
# ```math
# \mathbf{u}^{\mathrm{eq}} = \frac{1}{\rho}\sum_\alpha f_\alpha\,
#     \mathbf{e}_\alpha + \frac{\mathbf{F}}{2\rho}
# ```
#
# and an additional source term is added to the collision:
# ```math
# S_\alpha = \left(1 - \frac{\omega}{2}\right)\,w_\alpha\,
#     \left[\frac{\mathbf{e}_\alpha - \mathbf{u}}{c_s^2}
#     + \frac{(\mathbf{e}_\alpha \cdot \mathbf{u})}{c_s^4}\,
#     \mathbf{e}_\alpha \right] \cdot \mathbf{F}
# ```
#
# This ensures second-order accuracy in the recovered Navier-Stokes
# equations.
#
# ## Implementation in KrakeNN
#
# The Shan-Chen pipeline consists of three GPU kernels per time step:
#
# 1. **Pseudo-potential**: ``\psi(\rho)`` at each node
# 2. **Interaction force**: stencil sum for ``\mathbf{F}``
# 3. **BGK collision with Guo forcing**: standard collision + force term
#
# ```julia
# compute_psi_2d!(ψ, ρ, ρ0)
# compute_sc_force_2d!(Fx, Fy, ψ, G, Nx, Ny)
# collide_sc_2d!(f, Fx_sc, Fy_sc, is_solid, ω)
# ```
#
# All three kernels are purely local (stencil reads only) and map
# directly to GPU thread blocks with no synchronization beyond the
# standard kernel-launch barrier.
#
# ## Practical considerations
#
# | Parameter | Typical range | Effect |
# |:---|:---|:---|
# | ``G`` | −6 to −3 | Stronger attraction → higher density ratio |
# | ``\rho_0`` | 1.0 | Reference density for ``\psi`` |
# | ``\omega`` | 0.5 – 1.8 | Relaxation (stability-limited near interfaces) |
#
# !!! warning "Density ratio limitations"
#     The standard Shan-Chen model is limited to density ratios up to
#     roughly 10–20 before numerical instability occurs.  For higher
#     density ratios (e.g. water/air ≈ 1000), use the phase-field model.
#
# !!! note "Solid wetting"
#     Contact angle control in Shan-Chen is achieved by assigning a
#     pseudo-density to solid nodes, creating an effective wetting force.
#     This avoids explicit contact angle boundary conditions.

nothing  # suppress REPL output
