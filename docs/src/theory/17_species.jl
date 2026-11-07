# # Passive Scalar Transport (Species DDF)
#
# Many engineering applications require tracking the transport of a
# dissolved species — temperature, concentration, chemical tracer — that
# does not feed back on the flow.  KrakeNN implements this as a **passive
# scalar** using a separate set of distribution functions (Double
# Distribution Function, DDF) that share the velocity field from the
# hydrodynamic solver but have their own relaxation rate.
#
# **References**:
# He et al. (1998) [he1998novel](@cite he1998novel),
# Krüger et al. (2017) [kruger2017lattice](@cite kruger2017lattice) §6.
#
# ## Governing equation
#
# The target macroscopic equation is the advection-diffusion equation
# for a scalar field ``C(\mathbf{x}, t)``:
# ```math
# \frac{\partial C}{\partial t}
#     + \nabla \cdot (C\,\mathbf{u}) = D\,\nabla^2 C
# ```
#
# where ``D`` is the mass diffusivity (or thermal diffusivity for
# temperature).  The Péclet number ``\mathrm{Pe} = U L / D`` characterizes
# the relative importance of advection to diffusion.
#
# ## Lattice Boltzmann formulation
#
# A second population ``h_\alpha`` is introduced on the same D2Q9 lattice,
# with its own BGK collision:
# ```math
# h_\alpha(\mathbf{x} + \mathbf{e}_\alpha, t+1)
#     = h_\alpha(\mathbf{x}, t)
#     - \omega_D\,\big(h_\alpha - h_\alpha^{\mathrm{eq}}\big)
# ```
#
# The equilibrium for the species population is simpler than the
# hydrodynamic one, since it only needs to recover the correct
# advection-diffusion dynamics:
# ```math
# h_\alpha^{\mathrm{eq}} = w_\alpha\,C\,
#     \left(1 + \frac{\mathbf{e}_\alpha \cdot \mathbf{u}}{c_s^2}\right)
# ```
#
# Note the absence of the ``u^2`` terms present in the hydrodynamic
# equilibrium — the species equilibrium is **first-order** in velocity,
# which is sufficient to recover the advection-diffusion equation at the
# macroscopic level.
#
# ## Relaxation rate and diffusivity
#
# The relaxation rate ``\omega_D`` is related to the mass diffusivity by:
# ```math
# \omega_D = \frac{1}{3\,D + 1/2}
# ```
#
# This is the same form as the viscosity-to-omega relation for the
# hydrodynamic solver, but with ``D`` replacing ``\nu``.
#
# ## Macroscopic concentration
#
# The concentration is recovered as the zeroth moment of the species
# populations:
# ```math
# C = \sum_\alpha h_\alpha
# ```
#
# No velocity or stress information is extracted from ``h_\alpha``.
# The velocity field used in ``h_\alpha^{\mathrm{eq}}`` comes from
# the hydrodynamic solver.
#
# ```julia
# compute_concentration_2d!(C, h)
# ```
#
# ## Chapman-Enskog recovery
#
# Applying the Chapman-Enskog expansion to the species LBE recovers
# the advection-diffusion equation at ``O(\varepsilon^2)``:
#
# At ``O(\varepsilon^1)`` (Euler level):
# ```math
# \partial_t C + \nabla \cdot (C\,\mathbf{u}) = 0
# ```
#
# At ``O(\varepsilon^2)`` (diffusive correction):
# ```math
# \partial_t C + \nabla \cdot (C\,\mathbf{u})
#     = D\,\nabla^2 C + O(\Delta x^2)
# ```
#
# The truncation error is second-order in space, consistent with the
# hydrodynamic LBM.
#
# ## Boundary conditions
#
# ### Fixed concentration (Dirichlet)
#
# Dirichlet boundary conditions are imposed using the
# **anti-bounce-back** method: the unknown incoming populations are set
# to enforce a target concentration ``C_w`` at the wall:
# ```math
# h_{\bar{\alpha}} = -h_\alpha + 2\,w_\alpha\,C_w
# ```
#
# where ``\bar{\alpha}`` is the direction opposite to ``\alpha``.
# This is the concentration analogue of the Zou-He velocity BC.
#
# ```julia
# apply_fixed_conc_south_2d!(h, C_wall, Nx)
# apply_fixed_conc_north_2d!(h, C_wall, Nx, Ny)
# ```
#
# ### Zero-flux (Neumann)
#
# For insulating or impermeable walls, a simple bounce-back of the
# species populations ensures zero normal flux:
# ```math
# h_{\bar{\alpha}} = h_\alpha
# ```
#
# This is implemented automatically when the same `is_solid` mask is
# applied to both hydrodynamic and species populations.
#
# ## Implementation in KrakeNN
#
# The species module operates on a separate 3D array `h[Nx, Ny, Q]`
# using the same D2Q9 streaming kernel as the hydrodynamic populations.
# The collision is implemented as a dedicated kernel:
#
# ```julia
# collide_species_2d!(h, ux, uy, ω_D)
# ```
#
# The algorithm per time step is:
#
# 1. Stream ``h_\alpha`` using the standard `stream_2d!` kernel
# 2. Apply concentration BCs (anti-bounce-back or bounce-back)
# 3. Collide species with `collide_species_2d!`
# 4. Extract ``C = \sum h_\alpha`` with `compute_concentration_2d!`
#
# Since the species populations share the same lattice topology,
# **no additional streaming kernel** is needed — the existing
# `stream_2d!` is reused directly.
#
# !!! note "Multiple species"
#     Multiple scalars can be tracked simultaneously by allocating
#     separate `h` arrays for each species, each with its own
#     diffusivity ``D_k`` and relaxation rate ``\omega_{D,k}``.

nothing  # suppress REPL output
