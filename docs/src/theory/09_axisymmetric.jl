# # Axisymmetric LBM
#
# Many engineering problems (pipe flow, jets, nozzles) have **cylindrical
# symmetry**: the solution depends on the axial coordinate ``z`` and the radial
# coordinate ``r``, but not on the azimuthal angle ``\theta``. Instead of
# running an expensive 3D simulation, we can solve a 2D problem with source
# terms that account for the cylindrical geometry.
#
# ## Coordinate mapping
#
# In KrakeNN, the standard 2D lattice is reinterpreted as:
#
# - ``x \to z`` (axial direction, along the pipe)
# - ``y \to r`` (radial direction, from the axis outward)
#
# The radial position of node ``j`` is:
#
# ```math
# r_j = j - \tfrac{1}{2}
# ```
#
# The half-lattice offset places the axis of symmetry at ``r = 0.5``
# (between ``j = 0`` and ``j = 1``), which avoids the singularity at ``r = 0``.
#
# ## Why source terms?
#
# The Navier--Stokes equations in cylindrical coordinates contain extra terms
# proportional to ``1/r`` that are absent in Cartesian form. For example, the
# continuity equation becomes:
#
# ```math
# \frac{\partial \rho}{\partial t}
# + \frac{\partial (\rho u_z)}{\partial z}
# + \frac{1}{r}\frac{\partial (r \, \rho u_r)}{\partial r} = 0
# ```
#
# These geometric ``1/r`` terms appear as **source terms** in the LBM collision
# operator. Two schemes are implemented in KrakeNN.
#
# ## Zhou (2011) simple scheme
#
# [Zhou (2011)](@cite zhou2011mrt) proposed a straightforward source term added
# to the BGK collision:
#
# ```math
# f_q^{\star} = f_q - \omega(f_q - f_q^{\mathrm{eq}})
#             + S_q
# ```
#
# where:
#
# ```math
# S_q = -\frac{f_q^{\mathrm{eq}} \, u_r}{r}
# ```
#
# !!! note "Simplicity"
#     The Zhou scheme requires only one extra multiplication per direction.
#     It is implemented in KrakeNN as `collide_axisymmetric_2d!`.
#
# However, this simple scheme has limited accuracy: it recovers the
# axisymmetric Navier--Stokes equations only at leading order and introduces
# errors in the viscous stress terms.
#
# ## Li et al. (2010) improved scheme
#
# [Li et al. (2010)](@cite li2010improved) derived a more accurate source term
# by performing the Chapman--Enskog analysis directly in cylindrical coordinates.
# Their scheme modifies three aspects:
#
# ### 1. Direction-dependent source
#
# Instead of a single ``S_q`` proportional to ``f_q^{\mathrm{eq}}``, the source
# depends on the individual velocity components of each direction:
#
# ```math
# S_q = -\frac{w_q}{r} \left[
#   \rho u_r
#   + \rho u_r \frac{(e_{qz} u_z + e_{qr} u_r)}{c_s^2}
#   + \frac{\rho u_r}{2 c_s^4}
#     \Big( (e_{qz} u_z + e_{qr} u_r)^2 - c_s^2 |\mathbf{u}|^2 \Big)
#   - \frac{c_s^2 \rho u_r}{r} (1 - \omega_f)
# \right]
# ```
#
# where ``\omega_f`` is a direction-dependent relaxation correction:
#
# ```math
# \omega_f = w_q \left[ e_{qr}^2 - c_s^2 \right]
#            \cdot \frac{2(\tau - 1)}{2\tau - 1}
# ```
#
# ### 2. Corrected velocity
#
# The physical velocity includes a half-source correction (analogous to Guo
# forcing):
#
# ```math
# \rho u_r^{\text{phys}} = \sum_q f_q \, e_{qr} + \frac{S_r}{2}
# ```
#
# where ``S_r = -\rho u_r / r`` is the radial component of the geometric source.
#
# ### 3. Corrected density
#
# Similarly, the density includes a correction:
#
# ```math
# \rho^{\text{phys}} = \frac{\sum_q f_q}{1 + u_r / (2r)}
# ```
#
# !!! warning "Axis singularity"
#     All axisymmetric schemes have a ``1/r`` factor. Near the axis
#     (``r \to 0``), this requires careful treatment. KrakeNN places the axis
#     at ``r = 0.5`` and uses symmetry conditions on the ``j = 1`` row to
#     avoid division by zero.
#
# ## KrakeNN API

using KrakeNN

lattice = D2Q9()

## The two axisymmetric collision kernels
#
# Simple (Zhou):
# ```julia
# collide_axisymmetric_2d!(f_out, f_in, ρ, ux, uy, ω, lattice;
#                          ndrange=(Nx, Ny))
# ```
#
# Improved (Li et al.):
# ```julia
# collide_li_axisym_2d!(f_out, f_in, ρ, ux, uy, ω, lattice;
#                       ndrange=(Nx, Ny))
# ```

## Example: Hagen-Poiseuille in a pipe
## The analytical solution for axisymmetric Poiseuille flow is:
## u_z(r) = u_max * (1 - (r/R)^2)

R = 15.0       # pipe radius in lattice units
u_max = 0.05   # centreline velocity
ν = 0.01
Fx = 4.0 * ν * u_max / R^2  # required body force
@show Fx

# The `run_hagen_poiseuille_2d` driver validates both axisymmetric kernels
# against the parabolic pipe flow profile. The Li scheme typically achieves
# lower error than the Zhou scheme at the same resolution.

# ## See in action
#
# - [Hagen–Poiseuille pipe flow](../examples/09_hagen_poiseuille.md) —
#   axisymmetric LBM validated against the analytical parabola.
