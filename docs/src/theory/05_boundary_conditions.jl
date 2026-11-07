# # Boundary Conditions
#
# After streaming, some populations at boundary nodes are **unknown** because
# their upstream source lies outside the domain. Boundary conditions provide
# the missing information. This page covers the three main types implemented
# in KrakeNN: bounce-back, Zou--He velocity, and Zou--He pressure.
#
# ## Bounce-back (no-slip walls)
#
# The simplest wall boundary condition reverses incoming populations:
#
# ```math
# f_{\bar{q}}(\mathbf{x}_w, t+1) = f_q^{\star}(\mathbf{x}_w, t)
# ```
#
# where ``\bar{q}`` denotes the direction opposite to ``q``
# [Ladd (1994)](@cite ladd1994numerical).
# Physically, a particle hitting the wall bounces back the way it came, which
# enforces zero velocity at the wall to second-order accuracy (when the wall
# is placed halfway between fluid and solid nodes).
#
# For D2Q9, the opposite pairs are:
#
# | ``q`` | 2 (E) | 3 (N) | 4 (W) | 5 (S) | 6 (NE) | 7 (NW) | 8 (SW) | 9 (SE) |
# |:-----:|:-----:|:-----:|:-----:|:-----:|:------:|:------:|:------:|:------:|
# | ``\bar{q}`` | 4 (W) | 5 (S) | 2 (E) | 3 (N) | 8 (SW) | 9 (SE) | 6 (NE) | 7 (NW) |
#
# ## Zou--He velocity boundary
#
# When we want to **impose a velocity** at a boundary (e.g., the moving lid in a
# cavity), bounce-back alone is not enough. The Zou--He method
# [Zou & He (1997)](@cite zou1997pressure) uses the known velocity to close the
# system of equations for the unknown populations.
#
# ### Derivation for the north wall (lid)
#
# At a north-wall node ``(i, N_y)``, after streaming the three populations
# pointing inward (from above) are unknown: ``f_5`` (S), ``f_8`` (SW),
# ``f_9`` (SE). We know the wall velocity ``(u_w, 0)``.
#
# **Step 1** -- Density from the known populations. Using
# ``\rho = \sum_q f_q`` and ``\rho u_y = \sum_q f_q c_{qy} = 0``
# (zero normal velocity at the wall):
#
# ```math
# \rho = \frac{1}{1 + u_y} \Big[
#   f_1 + f_2 + f_4 + 2(f_3 + f_6 + f_7)
# \Big]
# ```
#
# For a stationary wall with ``u_y = 0``, this simplifies to
# ``\rho = f_1 + f_2 + f_4 + 2(f_3 + f_6 + f_7)``.
#
# **Step 2** -- The unknown populations:
#
# ```math
# f_5 = f_3 - \tfrac{2}{3}\,\rho\,u_y
# ```
# ```math
# f_8 = f_6 - \tfrac{1}{2}(f_2 - f_4)
#       + \tfrac{1}{2}\,\rho\,u_w
#       - \tfrac{1}{6}\,\rho\,u_y
# ```
# ```math
# f_9 = f_7 + \tfrac{1}{2}(f_2 - f_4)
#       - \tfrac{1}{2}\,\rho\,u_w
#       - \tfrac{1}{6}\,\rho\,u_y
# ```
#
# !!! note "Sign convention"
#     The signs in the Zou--He formulae depend on which wall is being treated.
#     KrakeNN provides dedicated kernels for each wall to avoid sign errors.
#
# ## Zou--He pressure boundary
#
# For **pressure (density) outlets**, the density ``\rho_{\text{out}}`` is
# prescribed and the velocity is computed from the known populations. The
# algebra is analogous to the velocity case but with ``\rho`` known and
# ``u_n`` (normal velocity) unknown.
#
# At an east outlet ``(N_x, j)`` with prescribed ``\rho_{\text{out}}``:
#
# ```math
# u_x = 1 - \frac{f_1 + f_3 + f_5 + 2(f_2 + f_6 + f_9)}{\rho_{\text{out}}}
# ```
#
# Then the unknown westward populations (``f_4, f_7, f_8``) are set using
# the same Zou--He framework.
#
# ## Periodic boundaries
#
# Periodic conditions simply wrap the streaming source index:
#
# ```math
# f_q(1, j) = f_q^{\star}(N_x, j) \quad \text{(left ← right)}
# ```
#
# These are handled implicitly inside `stream_fully_periodic_2d!` and
# `stream_periodic_x_wall_y_2d!`.
#
# ## KrakeNN boundary kernels

using KrakeNN

lattice = D2Q9()

## The opposite direction mapping used by bounce-back
opp = opposite(lattice)
for q in 1:9
    println("q=$q  →  opposite = $(opp[q])")
end

# Available Zou--He kernels:
#
# | Kernel | Wall | Prescribes |
# |:-------|:-----|:-----------|
# | `apply_zou_he_north_2d!` | North (top) | Velocity ``(u_w, 0)`` |
# | `apply_zou_he_south_2d!` | South (bottom) | Velocity |
# | `apply_zou_he_west_2d!` | West (inlet) | Velocity |
# | `apply_zou_he_pressure_east_2d!` | East (outlet) | Density ``\rho_{\text{out}}`` |
#
# Each kernel operates on a 1D range (the wall length) and modifies `f`
# in-place. Example call for a lid-driven cavity:
#
# ```julia
# apply_zou_he_north_2d!(f, u_lid, Nx, Ny)
# ```

# ## See in action
#
# - [Lid-driven cavity 2D](../examples/04_cavity_2d.md) — Zou-He velocity
#   BC on the lid, bounce-back on the other walls.
# - [Couette flow](../examples/02_couette_2d.md) — moving wall BC.
# - [Flow past a cylinder](../examples/06_cylinder_2d.md) — curved
#   bounce-back on an immersed obstacle.
