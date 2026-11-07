# # Phase-Field LBM for Two-Phase Flows
#
# Standard LBM two-phase models encode the density ratio through the
# distributions themselves (``\rho = \sum f_q``).  For density ratios beyond
# ``\sim 10``, the distributions in the light phase become very small,
# leading to numerical instability.  The **phase-field approach** circumvents
# this by using two independent sets of D2Q9 distributions:
#
# 1. ``f_q``: pressure/velocity equation with **modified equilibrium** where
#    the LBM density ``\rho_\text{lbm} \approx 1`` everywhere.
# 2. ``g_q``: Allen--Cahn equation for the **order parameter** ``\varphi \in [-1, 1]``.
#
# This enables density ratios up to 1000:1 while keeping all distributions ``O(1)``.
#
# **Reference**: Fakhari, Mitchell, Bolster & Leonardi (2017), *JCP* 334:620--638
# [Fakhari2017](@cite fakhari2017improved).
#
# ## Order parameter and free energy
#
# The order parameter distinguishes the phases:
# ```math
# \varphi = +1 \;\text{(liquid)}, \qquad \varphi = -1 \;\text{(gas)}
# ```
#
# The volume fraction is ``C = (1 + \varphi)/2`` and the physical density is:
# ```math
# \rho(\varphi) = \frac{\rho_l + \rho_g}{2}
#               + \frac{\rho_l - \rho_g}{2} \, \varphi
# ```
#
# The Ginzburg--Landau free energy functional is:
# ```math
# \mathcal{F}[\varphi] = \int \left[
#     \frac{\beta}{4}(\varphi^2 - 1)^2 + \frac{\kappa}{2} |\nabla\varphi|^2
# \right] \mathrm{d}V
# ```
#
# where ``\beta`` controls the double-well depth and ``\kappa`` the gradient
# energy penalty.  The **chemical potential** is the functional derivative:
# ```math
# \mu = \frac{\delta\mathcal{F}}{\delta\varphi}
#     = \beta \, \varphi (\varphi^2 - 1) - \kappa \, \nabla^2 \varphi
# ```
#
# ## Equilibrium interface profile
#
# At equilibrium (``\mu = 0``), the interface adopts a hyperbolic tangent profile:
# ```math
# \varphi(x) = \tanh\!\left(\frac{x}{W}\right)
# ```
#
# where ``W`` is the **interface width**.  The Cahn--Hilliard parameters relate
# to the surface tension ``\sigma`` and ``W`` via:
# ```math
# \beta = \frac{3\sigma}{2W}, \qquad
# \kappa = \frac{3\sigma W}{4}, \qquad
# \sigma = \frac{2\sqrt{2}}{3}\,\sqrt{\kappa\,\beta}
# ```
#
# In KrakeNN, these are computed by [`phasefield_params`](@ref):

using KrakeNN

σ = 0.01   # surface tension
W = 5.0    # interface width (lattice units)
β, κ = phasefield_params(σ, W)
println("β = $β, κ = $κ")
println("σ_check = ", (2√2/3) * √(κ * β))

# ## Surface tension force
#
# The surface tension enters the Navier--Stokes equation as a body force
# concentrated at the interface:
# ```math
# \mathbf{F}_\sigma = \mu \, \nabla\varphi
# ```
#
# This is equivalent to the continuum surface force (CSF) model but is
# derived consistently from the free energy functional.  No explicit
# curvature computation is needed.
#
# In KrakeNN:
# ```julia
# compute_chemical_potential_2d!(μ, φ, β, κ)
# compute_phasefield_force_2d!(Fx, Fy, μ, φ)
# ```
#
# For **axisymmetric** flows, the Laplacian in cylindrical coordinates has
# an extra ``(1/r)\,\partial\varphi/\partial r`` term.  This is added as a
# correction to ``\mu``:
# ```julia
# add_azimuthal_chemical_potential_2d!(μ, φ, κ, Ny)
# ```
#
# ## Allen--Cahn equation (``g_q``)
#
# The order parameter evolves according to the **conservative Allen--Cahn
# equation** [Chiu2011](@cite chiu2011conservative):
# ```math
# \frac{\partial\varphi}{\partial t} + \nabla \cdot (\varphi\,\mathbf{u})
# = \nabla \cdot \left[
#     D \left(\nabla\varphi + \frac{1}{W}(1-\varphi^2)\,\hat{\mathbf{n}}
#     \right)
# \right]
# ```
#
# The right-hand side consists of two terms:
# - **Diffusion** ``D\,\nabla^2\varphi``: recovered automatically by the LBM
#   with ``D = c_s^2(\tau_g - 1/2) = (\tau_g - 1/2)/3``
# - **Antidiffusion** ``(D/W)\,\nabla\cdot[(1-\varphi^2)\hat{\mathbf{n}}]``:
#   sharpens the interface back to the ``\tanh`` profile
#
# The **conservative** form of the sharpening term uses the interface normal
# ``\hat{\mathbf{n}} = \nabla\varphi/|\nabla\varphi|`` rather than the
# algebraic reaction ``\varphi(1-\varphi^2)``.  This preserves mass for
# curved interfaces, where the non-conservative form causes spurious
# shrinkage.
#
# ### LBM implementation
#
# The D2Q9 equilibrium for ``g_q`` is:
# ```math
# g_q^{\mathrm{eq}} = w_q \, \varphi \left(1 + 3\,\mathbf{e}_q \cdot \mathbf{u}\right)
# ```
#
# The antidiffusion source is added as a post-collision correction:
# ```math
# g_q^{\star} = g_q - \frac{1}{\tau_g}(g_q - g_q^{\mathrm{eq}})
#             + w_q \left(1 - \frac{1}{2\tau_g}\right) R
# ```
#
# where ``R = -(D/W)\,\nabla\cdot\mathbf{A}`` and
# ``\mathbf{A} = (1-\varphi^2)\,\nabla\varphi/|\nabla\varphi|``
# is the antidiffusion flux.
#
# In KrakeNN:
# ```julia
# compute_antidiffusion_flux_2d!(Ax, Ay, φ)
# collide_allen_cahn_2d!(g, ux, uy, Ax, Ay; τ_g=0.6, W=5.0)
# ```
#
# ## Pressure-based Navier--Stokes (``f_q``)
#
# The key idea is to separate the LBM density from the physical density.
# The **modified equilibrium** reads:
# ```math
# f_q^{\mathrm{eq}} = w_q \left[
#     \rho_\text{lbm}
#     + \rho(\varphi) \left(
#         3\,\mathbf{e}_q \cdot \mathbf{u}
#         + \frac{9}{2}(\mathbf{e}_q \cdot \mathbf{u})^2
#         - \frac{3}{2}\mathbf{u}^2
#     \right)
# \right]
# ```
#
# This ensures:
# - **Zeroth moment**: ``\sum f_q^{\mathrm{eq}} = \rho_\text{lbm} \approx 1``
#   (distributions stay ``O(1)`` regardless of density ratio)
# - **First moment**: ``\sum f_q^{\mathrm{eq}}\,\mathbf{e}_q = \rho(\varphi)\,\mathbf{u}``
#   (physical momentum is correct)
#
# The **physical velocity** is computed from the momentum with a half-force
# correction:
# ```math
# \mathbf{u} = \frac{\sum f_q\,\mathbf{e}_q + \mathbf{F}/2}{\rho(\varphi)}
# ```
#
# The collision uses MRT with per-node viscosity interpolated from
# ``\varphi``:
# ```math
# \nu(\varphi) = C\,\nu_l + (1-C)\,\nu_g, \qquad
# s_\nu = \frac{1}{3\nu + 1/2}
# ```
#
# In KrakeNN:
# ```julia
# collide_pressure_phasefield_mrt_2d!(f, φ, Fx, Fy, is_solid;
#     ρ_l=1.0, ρ_g=0.001, ν_l=0.1, ν_g=0.1)
# compute_macroscopic_phasefield_2d!(p, ux, uy, f, φ, Fx, Fy;
#     ρ_l=1.0, ρ_g=0.001)
# ```
#
# ## Static droplet validation
#
# The simplest test is the **Laplace law**: a circular droplet in a periodic
# domain should reach equilibrium with pressure jump
# ``\Delta p = \sigma / R``.

result = run_static_droplet_phasefield_2d(;
    N=80, R=20, W_pf=5.0, σ=0.01,
    ρ_l=1.0, ρ_g=1.0, ν=0.1,
    τ_g=0.6, max_steps=2000)

println("Δp measured = ", round(result.Δp; sigdigits=4))
println("Δp exact    = ", round(result.Δp_exact; sigdigits=4))
println("Error       = ", round(abs(result.Δp - result.Δp_exact) / result.Δp_exact * 100; digits=1), "%")

# At ``\rho_l/\rho_g = 1000``, the pressure-based formulation remains stable:

result_1000 = run_static_droplet_phasefield_2d(;
    N=80, R=20, W_pf=5.0, σ=0.01,
    ρ_l=1.0, ρ_g=0.001, ν=0.1,
    τ_g=0.6, max_steps=2000)

println("\nρ_ratio = 1000:")
println("Δp measured = ", round(result_1000.Δp; sigdigits=4))
println("Stable: ", all(isfinite.(result_1000.p)))

# ## Summary of the algorithm
#
# Each time step proceeds as:
#
# 1. **Stream** both ``f_q`` and ``g_q`` (pull scheme, same streaming kernel)
# 2. **Boundary conditions** for ``f_q`` (Zou-He) and ``g_q`` (equilibrium or extrapolation)
# 3. **Compute** ``\varphi = \sum g_q``
# 4. **Chemical potential** ``\mu = \beta\varphi(\varphi^2-1) - \kappa\nabla^2\varphi``
#    (+ azimuthal correction for axisymmetric)
# 5. **Surface tension force** ``\mathbf{F} = \mu\,\nabla\varphi``
# 6. **Macroscopic** quantities: ``p = c_s^2 \sum f_q``,
#    ``\mathbf{u} = (\sum f_q\mathbf{e}_q + \mathbf{F}/2)/\rho(\varphi)``
# 7. **Allen-Cahn collision** for ``g_q`` (conservative antidiffusion source)
# 8. **Pressure MRT collision** for ``f_q`` (modified equilibrium + Guo forcing)

nothing  # suppress REPL output
