# # Body Forces: the Guo Forcing Scheme
#
# Many flows involve external body forces: gravity in natural convection,
# a pressure gradient driving channel flow, or electromagnetic forces in
# magnetohydrodynamics. Adding forces to LBM is not as simple as adding
# ``\mathbf{F}/\rho`` to the velocity -- a naive approach introduces errors
# in the viscosity. The **Guo forcing scheme**
# [Guo, Zheng & Shi (2002)](@cite guo2002discrete) is the standard solution.
#
# ## The problem with naive forcing
#
# One might add a body force by simply modifying the equilibrium velocity:
#
# ```math
# \mathbf{u}^* = \mathbf{u} + \frac{\mathbf{F} \, \Delta t}{\rho}
# ```
#
# However, Chapman--Enskog analysis shows this changes the recovered viscosity
# and introduces a spurious ``O(\mathbf{F})`` term in the momentum equation.
# The Guo scheme corrects this by adding a properly designed forcing term to the
# collision operator.
#
# ## Guo forcing term
#
# The modified collision reads:
#
# ```math
# f_q^{\star} = f_q - \omega(f_q - f_q^{\mathrm{eq}})
#             + \left(1 - \frac{\omega}{2}\right) F_q
# ```
#
# where the discrete forcing term ``F_q`` is:
#
# ```math
# F_q = w_q \left[
#     \frac{(\mathbf{e}_q - \mathbf{u}) \cdot \mathbf{F}}{c_s^2}
#     + \frac{(\mathbf{e}_q \cdot \mathbf{u})(\mathbf{e}_q \cdot \mathbf{F})}{c_s^4}
# \right]
# ```
#
# Expanding with ``c_s^2 = 1/3``:
#
# ```math
# F_q = w_q \Big[
#     3 \, (\mathbf{e}_q - \mathbf{u}) \cdot \mathbf{F}
#     + 9 \, (\mathbf{e}_q \cdot \mathbf{u})(\mathbf{e}_q \cdot \mathbf{F})
# \Big]
# ```
#
# !!! note "Key result"
#     The prefactor ``(1 - \omega/2)`` is essential. Without it, the forcing
#     introduces a viscosity-dependent error in the recovered momentum equation.
#
# ## Corrected macroscopic velocity
#
# With Guo forcing, the physical velocity includes a half-force correction:
#
# ```math
# \rho \, \mathbf{u} = \sum_q f_q \, \mathbf{e}_q + \frac{\mathbf{F}}{2}
# ```
#
# This half-step correction ensures second-order accuracy in time. Without it,
# the velocity field would be ``O(\Delta t)`` off.
#
# The equilibrium is then evaluated at this **corrected** velocity.
#
# ## Application: Poiseuille flow
#
# A textbook validation case is pressure-driven channel flow (Poiseuille flow),
# where a uniform body force ``F_x`` replaces the pressure gradient:
#
# ```math
# F_x = \frac{8 \nu \, u_{\max}}{H^2}
# ```
#
# with ``H`` the channel height and ``u_{\max}`` the target centreline velocity.
# The analytical velocity profile is parabolic:
#
# ```math
# u_x(y) = u_{\max} \left[ 1 - \left(\frac{2y - H}{H}\right)^2 \right]
# ```
#
# ## KrakeNN API
#
# The `collide_guo_2d!` kernel performs BGK collision with Guo forcing:
#
# ```julia
# collide_guo_2d!(f, is_solid, ω, Fx, Fy)
# ```
#
# And `compute_macroscopic_forced_2d!` recovers the corrected velocity:
#
# ```julia
# compute_macroscopic_forced_2d!(ρ, ux, uy, f, Fx, Fy)
# ```

using KrakeNN

## Set up a Poiseuille-like configuration
lattice = D2Q9()
Nx, Ny = 5, 32
ν = 0.01
ω = 2.0 / (6.0 * ν + 1.0)
@show ω

## Body force to drive the flow
u_max = 0.05
H = Ny - 2  # channel height (excluding walls)
Fx = 8.0 * ν * u_max / H^2
Fy = 0.0
@show Fx

# The `run_poiseuille_2d` driver wraps the full simulation loop including
# Guo collision, periodic streaming in x, bounce-back walls in y, and
# macroscopic field recovery with half-force correction.
#
# !!! tip "Verifying the implementation"
#     After convergence, compare the velocity profile against the analytical
#     parabola. The L2 error should decrease as ``O(\Delta x^2)`` under grid
#     refinement.

# ## See in action
#
# - [Poiseuille channel](../examples/01_poiseuille_2d.md) — Guo body-force
#   scheme driving the parabolic profile.
