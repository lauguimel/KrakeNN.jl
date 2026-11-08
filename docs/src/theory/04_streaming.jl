# # Streaming Step
#
# After collision updates the distributions locally, the **streaming** step
# propagates them to neighbouring nodes along their respective velocity
# vectors. Streaming is the non-local half of the LBM algorithm and is
# responsible for information transport across the lattice
# [Kruger et al. (2017)](@cite kruger2017lattice).
#
# ## Pull vs push
#
# There are two equivalent ways to implement streaming:
#
# - **Push** (scatter): each node *sends* its post-collision populations to
#   neighbours: ``f_q(\mathbf{x} + \mathbf{e}_q) \leftarrow f_q^{\star}(\mathbf{x})``.
#
# - **Pull** (gather): each node *reads* populations from its upstream
#   neighbours: ``f_q(\mathbf{x}) \leftarrow f_q^{\star}(\mathbf{x} - \mathbf{e}_q)``.
#
# !!! tip "Pull scheme on GPUs"
#     KrakeNN uses the **pull scheme**. It is preferred on GPUs because each
#     thread writes to a single known memory location, avoiding write conflicts
#     and enabling coalesced memory access.
#
# ## The streaming formula
#
# For the D2Q9 lattice with velocities ``\mathbf{e}_q = (c_{qx}, c_{qy})``,
# the pull-scheme streaming reads:
#
# ```math
# f_q(i, j, t+1) = f_q^{\star}(i - c_{qx}, \; j - c_{qy}, \; t)
# ```
#
# For example, the East population (``q = 2``, ``c_x = 1``, ``c_y = 0``) at
# node ``(i, j)`` is pulled from node ``(i-1, j)``. The rest population
# (``q = 1``) stays in place since ``\mathbf{e}_1 = (0,0)``.
#
# ## Concrete example
#
# Consider streaming on a small 5x5 grid. After collision, ``f_2^{\star}``
# (East) at node (2, 3) will be read by node (3, 3) during streaming:
#
# ```
#  Before streaming          After streaming
#  (pull from left)
#
#  f₂*(2,3) = 0.15           f₂(3,3) = 0.15
#      ·---→---·                 ·---→---·
#    (2,3)   (3,3)             (2,3)   (3,3)
# ```
#
# ## Boundary handling
#
# Streaming near domain boundaries requires special treatment because
# upstream nodes may lie outside the domain. Common strategies:
#
# - **Periodic boundaries**: wrap around to the opposite side of the domain.
# - **Wall boundaries**: unknown incoming populations are set by bounce-back
#   or Zou--He conditions (see the Boundary Conditions chapter).
#
# In KrakeNN, different streaming kernels handle different boundary
# configurations:
#
# | Kernel | Walls | Periodic |
# |:-------|:------|:---------|
# | `stream_2d!` | All four sides are walls | None |
# | `stream_periodic_x_wall_y_2d!` | Top/bottom walls | Left/right periodic |
# | `stream_fully_periodic_2d!` | None | All four sides |
#
# ## Interior streaming kernel
#
# The kernel `stream_2d!` operates on interior nodes (away from boundaries).
# Populations pointing into a wall are handled *after* streaming by the
# boundary condition kernels. The call signature is:
#
# ```julia
# stream_2d!(f_out, f_in, Nx, Ny)
# ```
#
# where `f_in` holds the post-collision state and `f_out` receives the
# post-streaming result. The two arrays must be distinct (double-buffering).
#
# ## Streaming with periodicity
#
# For periodic boundaries, `stream_fully_periodic_2d!` uses modular
# arithmetic so that a node at position ``(1, j)`` pulling the West
# population (``q = 4``, ``c_x = -1``) reads from ``(N_x, j)``:
#
# ```math
# i_{\text{src}} = \mathrm{mod1}(i - c_{qx}, \, N_x)
# ```
#
# This avoids explicit ghost layers and is very efficient on GPUs.

using KrakeNN

lattice = D2Q9()

## Show the velocity vectors to understand streaming directions
cx = velocities_x(lattice)
cy = velocities_y(lattice)

for q in 1:9
    println("q=$q: pull from (i-$(cx[q]), j-$(cy[q]))")
end

# ## Double buffering
#
# Because a node may be both a source and a destination during the same
# streaming step, KrakeNN uses **two separate arrays** (`f_in` and `f_out`).
# After streaming, the roles are swapped for the next time step:
#
# ```
# Step n:   collide(f_in → f_tmp) → stream(f_tmp → f_out)
# Step n+1: collide(f_out → f_tmp) → stream(f_tmp → f_in)
# ```
#
# This swap is a simple pointer exchange, not a data copy.

# ## See in action
#
# - [Lid-driven cavity 2D](../examples/04_cavity_2d.md) — exercises the
#   pull-streaming kernel on a closed domain.
