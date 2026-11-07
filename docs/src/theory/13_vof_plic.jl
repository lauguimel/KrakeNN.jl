# # Volume-of-Fluid with PLIC Reconstruction
#
# The **Volume-of-Fluid (VOF)** method tracks a two-phase interface using
# a scalar field ``C(\mathbf{x}, t) \in [0, 1]`` representing the volume
# fraction of the liquid phase in each cell.  Unlike diffuse-interface methods,
# VOF maintains a sharp interface (1--2 cells wide) by combining geometric
# reconstruction with conservative advection.
#
# **References**:
# Scardovelli & Zaleski (1999) [scardovelli1999direct](@cite scardovelli1999direct),
# Rider & Kothe (1998) [rider1998reconstructing](@cite rider1998reconstructing),
# Popinet (2009) [popinet2009accurate](@cite popinet2009accurate).
#
# ## Volume fraction field
#
# Each computational cell is classified based on its volume fraction:
# ```math
# C =
# \begin{cases}
#     1 & \text{liquid (full)} \\
#     0 & \text{gas (empty)} \\
#     0 < C < 1 & \text{interface (mixed cell)}
# \end{cases}
# ```
#
# The physical density and viscosity are interpolated arithmetically:
# ```math
# \rho(C) = C\,\rho_l + (1-C)\,\rho_g, \qquad
# \nu(C)  = C\,\nu_l  + (1-C)\,\nu_g
# ```
#
# The volume fraction is advected by the flow velocity:
# ```math
# \frac{\partial C}{\partial t} + \nabla \cdot (C\,\mathbf{u}) = 0
# ```
#
# Naively discretizing this advection equation with standard LBM or finite
# differences leads to severe numerical diffusion that smears the interface
# over many cells.  The PLIC method avoids this by using geometric
# reconstruction.
#
# ## Interface normal estimation
#
# ### Youngs' method
#
# The interface normal ``\hat{\mathbf{n}}`` in each mixed cell is estimated
# from the gradient of ``C`` using the **Youngs method** (a ``3 \times 3``
# weighted finite-difference stencil):
# ```math
# \hat{\mathbf{n}} = -\frac{\nabla C}{|\nabla C|}
# ```
#
# The sign convention ``\hat{\mathbf{n}} = -\nabla C / |\nabla C|`` gives
# an outward normal from liquid to gas (consistent with Basilisk).
#
# The Youngs stencil weights corner and edge neighbours:
# ```math
# \frac{\partial C}{\partial x} \approx \frac{1}{8}
# \left(C_{i-1,j-1} + 2\,C_{i-1,j} + C_{i-1,j+1}
#      - C_{i+1,j-1} - 2\,C_{i+1,j} - C_{i+1,j+1}\right)
# ```
#
# ### Mixed Youngs--Centred (MYC)
#
# The MYC method selects between Youngs and centred-difference normals
# on a per-cell basis, choosing whichever gives a more accurate PLIC
# reconstruction.  This reduces orientation-dependent errors, especially
# for interfaces aligned with grid diagonals.
#
# In KrakeNN:
# ```julia
# compute_vof_normal_2d!(nx, ny, C, Nx, Ny)
# ```
#
# ## PLIC reconstruction
#
# Given the normal ``\hat{\mathbf{n}}`` and volume fraction ``C``, the
# **Piecewise Linear Interface Calculation (PLIC)** places a line
# ``\hat{\mathbf{n}} \cdot \mathbf{x} = \alpha`` in each mixed cell
# such that the fluid area below the line equals ``C``.
#
# The line parameter ``\alpha`` is found by inverting the exact
# area-fraction relation from Scardovelli & Zaleski (2000).
# In a unit cell ``[-1/2, 1/2]^2`` with normal ``(n_x, n_y)``:
# ```math
# \alpha(C) = \begin{cases}
#   \sqrt{2\,C\,n_1\,n_2}                      & C \leq \frac{n_1}{2\,n_2} \\[4pt]
#   C\,n_2 + \frac{n_1}{2}                      & \frac{n_1}{2\,n_2} < C \leq 1 - \frac{n_1}{2\,n_2} \\[4pt]
#   n_1 + n_2 - \sqrt{2\,n_1\,n_2\,(1 - C)}    & C > 1 - \frac{n_1}{2\,n_2}
# \end{cases}
# ```
#
# where ``n_1 = \min(|n_x|, |n_y|)`` and ``n_2 = \max(|n_x|, |n_y|)``.
# These are the exact Basilisk `line_alpha` / `line_area` formulas,
# reimplemented in KrakeNN as branchless GPU-safe inline functions.
#
# ## VOF advection
#
# ### Operator-split geometric advection
#
# The PLIC-based advection uses **directional splitting** (Strang splitting):
# the volume fraction is advected first in ``x``, then in ``y``
# (alternating order on odd/even steps to reduce directional bias).
#
# For each face, the geometric flux is computed by evaluating the
# `rectangle_fraction`: the portion of the donor cell's PLIC reconstruction
# that is swept through the face during ``\Delta t``:
# ```math
# \Phi_{\text{face}} = C_{\text{face}} \times u_{\text{face}}
# ```
#
# where ``C_{\text{face}}`` is the fraction of the swept strip that is
# filled with liquid, obtained from the PLIC geometry.
#
# The **Weymouth--Yue (2010)** compression correction is applied to
# maintain boundedness:
# ```math
# C_i^{n+1} = C_i^n - (\Phi_R - \Phi_L)
#            + \tilde{C}_i\,(u_R - u_L)
# ```
#
# where ``\tilde{C}_i = \mathbb{1}_{C_i > 0.5}`` is frozen before the
# sweeps to avoid race conditions.
#
# In KrakeNN:
# ```julia
# advect_vof_plic_2d!(C_new, C, nx_n, ny_n, cc_field, ux, uy, Nx, Ny; step=1)
# ```
#
# ### MUSCL-Superbee TVD advection
#
# As an alternative to geometric PLIC advection, KrakeNN provides a
# second-order **MUSCL scheme** with the **Superbee flux limiter**:
# ```math
# \phi(r) = \max\!\big(0,\; \min(2r, 1),\; \min(r, 2)\big)
# ```
#
# Superbee is the most compressive TVD limiter, making it well-suited
# for VOF transport where interface sharpness is critical.  The MUSCL
# advection is algebraic (no geometric reconstruction needed) and
# cheaper per time step, but produces slightly more diffusion than PLIC
# over long integration times.
#
# In KrakeNN:
# ```julia
# advect_vof_2d!(C_new, C, ux, uy, Nx, Ny)
# ```
#
# ## Height-function curvature
#
# Accurate curvature is essential for computing surface tension forces.
# The **height-function (HF)** method accumulates volume fractions along
# columns perpendicular to the interface to form discrete heights ``h``,
# then computes curvature from finite differences
# [popinet2009accurate](@cite popinet2009accurate):
# ```math
# \kappa = -\frac{h''}{(1 + h'^2)^{3/2}}
# ```
#
# For an interface with ``|n_y| \geq |n_x|`` (more horizontal), vertical
# columns of ``\pm 2`` cells are summed to give ``h(x)``:
# ```math
# h_i = \sum_{j=-2}^{2} C(i, j_0 + j)
# ```
#
# The derivatives are approximated with standard central differences:
# ```math
# h' \approx \frac{h_{i+1} - h_{i-1}}{2}, \qquad
# h'' \approx h_{i+1} - 2\,h_i + h_{i-1}
# ```
#
# The height-function method achieves **second-order convergence** of
# curvature, in contrast to first-order methods based on fitting or
# direct differentiation of the volume fraction field.
#
# In KrakeNN:
# ```julia
# compute_hf_curvature_2d!(κ, C, nx_n, ny_n, Nx, Ny)
# ```
#
# ## Continuum Surface Force (CSF)
#
# Surface tension enters the momentum equation as a volume force
# concentrated at the interface [brackbill1992continuum](@cite brackbill1992continuum):
# ```math
# \mathbf{F}_\sigma = \sigma\,\kappa\,\nabla C
# ```
#
# where ``\sigma`` is the surface tension coefficient, ``\kappa`` is the
# curvature from the height-function method, and ``\nabla C`` localizes
# the force to interface cells.  This force is added to the LBM via the
# Guo forcing scheme.
#
# In KrakeNN:
# ```julia
# compute_surface_tension_2d!(Fx, Fy, κ, C, σ, Nx, Ny)
# ```
#
# The two-phase collision kernel handles the momentum equation with
# variable density and viscosity:
# ```julia
# collide_twophase_2d!(f, C, Fx, Fy, gx, gy, is_solid;
#     ρ_l=1.0, ρ_g=0.001, ν_l=0.1, ν_g=0.1)
# ```
#
# ## Axisymmetric extensions
#
# For axisymmetric flows (2D domain representing the ``r``--``z`` plane),
# the curvature has an additional azimuthal component:
# ```math
# \kappa_\text{axi} = \kappa_{2D} + \frac{n_r}{r}
# ```
#
# where ``n_r`` is the radial component of the interface normal and ``r``
# is the distance from the axis of symmetry.  The height-function
# curvature kernel includes this correction for axisymmetric simulations.
#
# Similarly, the VOF advection receives a geometric source term
# ``-C\,u_r / r`` to account for the radial expansion/contraction
# of the cylindrical control volumes.
#
# ## Summary of the VOF-PLIC algorithm
#
# Each time step for the VOF-coupled LBM proceeds as:
#
# 1. **Interface normals**: compute ``\hat{\mathbf{n}}`` from the Youngs stencil
# 2. **PLIC advection**: geometric flux computation with Weymouth--Yue correction
#    (or MUSCL-Superbee alternative)
# 3. **Clamp** ``C`` to ``[0, 1]``
# 4. **Height-function curvature**: ``\kappa`` from column heights
# 5. **Surface tension force**: ``\mathbf{F}_\sigma = \sigma\,\kappa\,\nabla C``
# 6. **LBM collision**: two-phase MRT with variable ``\rho(C)``, ``\nu(C)``
#    and Guo forcing
# 7. **LBM streaming**: standard pull scheme (same kernel as single-phase)
#
# ## Validation examples
#
# The following examples validate the VOF-PLIC implementation:
#
# - **Zalesak disk** (example 11): rigid body rotation of a slotted disk,
#   testing advection accuracy
# - **Reversed vortex** (example 12): deformation and reversal,
#   testing MUSCL-Superbee vs PLIC advection
# - **Capillary wave** (example 13): small-amplitude oscillation,
#   validating surface tension and curvature
#   [prosperetti1981motion](@cite prosperetti1981motion)
# - **Static droplet** (example 14): Laplace law ``\Delta p = \sigma/R``,
#   testing spurious currents
# - **Rayleigh--Plateau instability** (example 15): axisymmetric jet breakup
#   [rayleigh1878instability](@cite rayleigh1878instability)

nothing  # suppress REPL output
