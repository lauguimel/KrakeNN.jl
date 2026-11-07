# # Multiple-Relaxation-Time (MRT) Collision
#
# The BGK (single-relaxation-time) collision operator relaxes all moments
# towards equilibrium at the same rate ``\omega = 1/(3\nu + 1/2)``.
# While simple and efficient, BGK has well-known limitations:
#
# - **Fixed Prandtl number** (``\Pr = 1`` in D2Q9)
# - **Viscosity-dependent stability**: low viscosity requires ``\omega \to 2``,
#   amplifying non-hydrodynamic (ghost) modes
# - **Spurious oscillations** near boundaries at high Reynolds numbers
#
# The **Multiple-Relaxation-Time (MRT)** collision overcomes these issues
# by relaxing each moment independently, providing additional free parameters
# to damp non-physical modes without affecting the hydrodynamic behaviour.
#
# **References**: Lallemand & Luo (2000) [lallemand2000theory](@cite lallemand2000theory),
# d'Humières (2002) [dhumieres2002multiple](@cite dhumieres2002multiple),
# Krüger et al. (2017) [kruger2017lattice](@cite kruger2017lattice) §10.
#
# ## Moment space transform
#
# The D2Q9 distribution vector ``\mathbf{f} = (f_1, \ldots, f_9)^T``
# is mapped to **moment space** via a linear transformation:
# ```math
# \mathbf{m} = \mathsf{M} \, \mathbf{f}
# ```
#
# The nine moments are ordered as:
# ```math
# \mathbf{m} = (\rho,\; e,\; \varepsilon,\;
#               j_x,\; q_x,\; j_y,\; q_y,\;
#               p_{xx},\; p_{xy})^T
# ```
#
# where:
# - ``\rho`` — density (conserved)
# - ``e`` — energy
# - ``\varepsilon`` — energy square
# - ``j_x, j_y`` — momentum components (conserved)
# - ``q_x, q_y`` — energy flux components
# - ``p_{xx}, p_{xy}`` — stress tensor components
#
# The transformation matrix ``\mathsf{M}`` from Lallemand & Luo (2000)
# for D2Q9 with ordering (rest, E, N, W, S, NE, NW, SW, SE) is:
#
# ```math
# \mathsf{M} = \begin{pmatrix}
#  1 &  1 &  1 &  1 &  1 &  1 &  1 &  1 &  1 \\
# -4 & -1 & -1 & -1 & -1 &  2 &  2 &  2 &  2 \\
#  4 & -2 & -2 & -2 & -2 &  1 &  1 &  1 &  1 \\
#  0 &  1 &  0 & -1 &  0 &  1 & -1 & -1 &  1 \\
#  0 & -2 &  0 &  2 &  0 &  1 & -1 & -1 &  1 \\
#  0 &  0 &  1 &  0 & -1 &  1 &  1 & -1 & -1 \\
#  0 &  0 & -2 &  0 &  2 &  1 &  1 & -1 & -1 \\
#  0 &  1 & -1 &  1 & -1 &  0 &  0 &  0 &  0 \\
#  0 &  0 &  0 &  0 &  0 &  1 & -1 &  1 & -1
# \end{pmatrix}
# ```
#
# ## Equilibrium moments
#
# The equilibrium values of the non-conserved moments are functions of
# ``\rho``, ``u_x``, and ``u_y``:
# ```math
# \begin{aligned}
# e^{\mathrm{eq}}   &= -2\rho + 3\rho(u_x^2 + u_y^2) \\
# \varepsilon^{\mathrm{eq}} &= \rho - 3\rho(u_x^2 + u_y^2) \\
# q_x^{\mathrm{eq}} &= -\rho\,u_x, \qquad q_y^{\mathrm{eq}} = -\rho\,u_y \\
# p_{xx}^{\mathrm{eq}} &= \rho(u_x^2 - u_y^2) \\
# p_{xy}^{\mathrm{eq}} &= \rho\,u_x\,u_y
# \end{aligned}
# ```
#
# The conserved moments ``\rho``, ``j_x = \rho u_x``, ``j_y = \rho u_y``
# are unchanged by collision.
#
# ## Relaxation matrix
#
# The collision is performed in moment space with a diagonal relaxation
# matrix ``\mathsf{S}``:
# ```math
# \mathbf{m}^\star = \mathbf{m}
#     - \mathsf{S}\,(\mathbf{m} - \mathbf{m}^{\mathrm{eq}})
# ```
#
# ```math
# \mathsf{S} = \mathrm{diag}(0,\; s_e,\; s_\varepsilon,\;
#              0,\; s_q,\; 0,\; s_q,\; s_\nu,\; s_\nu)
# ```
#
# The **stress relaxation rate** ``s_\nu`` controls the kinematic viscosity,
# exactly as in BGK:
# ```math
# s_\nu = \frac{1}{3\nu + 1/2}
# ```
#
# The other rates (``s_e``, ``s_\varepsilon``, ``s_q``) are **free parameters**
# that do not affect the Navier--Stokes dynamics but control the damping
# of non-hydrodynamic modes.  Typical values from the literature:
#
# | Parameter | Typical value | Role |
# |-----------|:---:|------|
# | ``s_e`` | 1.4 | Energy relaxation |
# | ``s_\varepsilon`` | 1.4 | Energy-square relaxation |
# | ``s_q`` | 1.2 | Energy-flux relaxation |
# | ``s_\nu`` | ``1/(3\nu+0.5)`` | Viscosity (fixed by physics) |
#
# Setting all rates equal to ``s_\nu`` recovers the BGK operator.
#
# The post-collision distributions are obtained by the inverse transform:
# ```math
# \mathbf{f}^\star = \mathsf{M}^{-1} \, \mathbf{m}^\star
# ```
#
# ## Implementation in KrakeNN
#
# In KrakeNN, the MRT collision is implemented as a single GPU kernel
# with the moment-space operations fully unrolled (no matrix multiplications):
#
# ```julia
# collide_mrt_2d!(f, is_solid, ν; s_e=1.4, s_eps=1.4, s_q=1.2)
# ```
#
# The transformation ``\mathsf{M}\,\mathbf{f}`` and inverse
# ``\mathsf{M}^{-1}\,\mathbf{m}^\star`` are written as explicit
# arithmetic on the nine components, avoiding any temporary arrays
# or matrix storage on the GPU.
#
# For **two-phase flows**, a variant with per-node viscosity and Guo
# forcing is provided:
#
# ```julia
# collide_twophase_mrt_2d!(f, C, Fx_st, Fy_st, is_solid;
#     ρ_l=1.0, ρ_g=0.001, ν_l=0.1, ν_g=0.1,
#     s_e=1.4, s_eps=1.4, s_q=1.2)
# ```
#
# Here the local viscosity is interpolated from the volume fraction:
# ``\nu(C) = C\,\nu_l + (1-C)\,\nu_g``, and the momentum is corrected
# by a half-force term following the Guo scheme
# [guo2002discrete](@cite guo2002discrete).
#
# ## Comparison with BGK
#
# | Feature | BGK | MRT |
# |---------|:---:|:---:|
# | Free parameters | 1 (``\omega``) | 4 (``s_e, s_\varepsilon, s_q, s_\nu``) |
# | Stability at low ``\nu`` | Poor | Excellent |
# | Galilean invariance error | ``O(\mathrm{Ma}^3)`` | ``O(\mathrm{Ma}^3)`` (same) |
# | Computational cost | 1× | ~1.1× (moment transforms) |
# | Boundary artefacts | Visible | Reduced |
#
# The MRT collision adds minimal overhead (the ``\mathsf{M}`` and
# ``\mathsf{M}^{-1}`` transforms are ~40 multiply-adds per node) while
# significantly improving stability, especially for flows at low viscosity
# or with large density/viscosity contrasts.
#
# ## When to use MRT
#
# MRT is recommended when:
# - The kinematic viscosity is low (``\tau \lesssim 0.55``, high Re)
# - Two-phase flows with large viscosity ratio (``\nu_l / \nu_g \gg 1``)
# - Flows with complex geometries where boundary stability matters
# - Phase-field simulations (the pressure-based MRT collision is used by default)
#
# In KrakeNN, the CIJ jet simulation (example 16) uses MRT collision
# for the pressure equation in the phase-field formulation.
#
# ## Real source excerpt
#
# The single-phase MRT kernel lives in `src/kernels/collide_mrt_2d.jl`.
# The block below is injected verbatim from that source file at doc-build
# time by the `source_extract` helper, so the theory and the code cannot
# drift apart:
#
# @@EXTRACT src/kernels/collide_mrt_2d.jl collide_mrt_2d!@@
#
# ## See in action
#
# - [Lid-driven cavity 2D](../examples/04_cavity_2d.md) — BGK baseline; swap to
#   MRT via the `collision = :mrt` kwarg.
# - [Taylor–Green vortex](../examples/03_taylor_green_2d.md) — low-viscosity
#   test where MRT stability gains are most visible.

nothing  # suppress REPL output
