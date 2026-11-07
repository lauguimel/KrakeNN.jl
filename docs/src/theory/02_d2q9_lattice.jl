# # The D2Q9 Lattice
#
# The **D2Q9** lattice is the workhorse of two-dimensional LBM. "D2" means two
# spatial dimensions; "Q9" means nine discrete velocities
# [Qian, d'Humieres & Lallemand (1992)](@cite qian1992lattice).
#
# ## Velocity set
#
# The nine velocity vectors ``\mathbf{e}_q = (e_{qx}, e_{qy})`` are:
#
# | ``q`` | Direction | ``e_{qx}`` | ``e_{qy}`` | Weight ``w_q`` |
# |:-----:|:---------:|:----------:|:----------:|:--------------:|
# |   1   |   rest    |     0      |     0      |     4/9        |
# |   2   |     E     |     1      |     0      |     1/9        |
# |   3   |     N     |     0      |     1      |     1/9        |
# |   4   |     W     |    -1      |     0      |     1/9        |
# |   5   |     S     |     0      |    -1      |     1/9        |
# |   6   |    NE     |     1      |     1      |     1/36       |
# |   7   |    NW     |    -1      |     1      |     1/36       |
# |   8   |    SW     |    -1      |    -1      |     1/36       |
# |   9   |    SE     |     1      |    -1      |     1/36       |
#
# Arranged on the lattice (ASCII art, North = up):
#
# ```
#     NW (7)    N (3)    NE (6)
#         \      |      /
#          \     |     /
#    W (4) --- rest (1) --- E (2)
#          /     |     \
#         /      |      \
#     SW (8)    S (5)    SE (9)
# ```
#
# !!! tip "Julia indexing"
#     In KrakeNN, ``q`` ranges from **1 to 9** (1-indexed). The rest
#     population is ``q = 1``, not ``q = 0`` as in some C/Fortran codes.
#
# ## Speed of sound
#
# The lattice speed of sound is fixed by the weight structure:
#
# ```math
# c_s^2 = \frac{1}{3} \qquad \Longrightarrow \qquad c_s = \frac{1}{\sqrt{3}}
# ```
#
# This value follows from the isotropy conditions on the weight tensor
# ``\sum_q w_q \, e_{q\alpha} \, e_{q\beta} = c_s^2 \, \delta_{\alpha\beta}``.
#
# ## Equilibrium distribution
#
# The Maxwell--Boltzmann equilibrium is discretised into the **lattice
# equilibrium**:
#
# ```math
# f_q^{\mathrm{eq}}(\rho, \mathbf{u})
# = w_q \, \rho \left[
#     1
#     + \frac{\mathbf{e}_q \cdot \mathbf{u}}{c_s^2}
#     + \frac{(\mathbf{e}_q \cdot \mathbf{u})^2}{2 \, c_s^4}
#     - \frac{\mathbf{u} \cdot \mathbf{u}}{2 \, c_s^2}
# \right]
# ```
#
# Expanding with ``c_s^2 = 1/3``:
#
# ```math
# f_q^{\mathrm{eq}} = w_q \, \rho \Big[
#     1 + 3 \, (\mathbf{e}_q \cdot \mathbf{u})
#       + \tfrac{9}{2} \, (\mathbf{e}_q \cdot \mathbf{u})^2
#       - \tfrac{3}{2} \, |\mathbf{u}|^2
# \Big]
# ```
#
# !!! note "Key result"
#     The equilibrium is a second-order polynomial in the Mach number
#     ``\mathrm{Ma} = u / c_s``. Higher-order terms are truncated, which
#     limits LBM accuracy to ``O(\mathrm{Ma}^2)``.
#
# ## KrakeNN API
#
# Let us inspect the D2Q9 lattice and compute equilibrium values.

using KrakeNN

lattice = D2Q9()

## Weights
w = weights(lattice)
@show w

## Velocity components
cx = velocities_x(lattice)
cy = velocities_y(lattice)
@show cx
@show cy

## Opposite direction mapping (used for bounce-back)
@show [opposite(lattice, q) for q in 1:9]

## Compute equilibrium for ρ=1, u=(0.1, 0) for all 9 directions
rho, ux, uy = 1.0, 0.1, 0.0
feq = [equilibrium(lattice, rho, ux, uy, q) for q in 1:9]
@show feq

## Verify mass conservation: Σ feq = ρ
@show sum(feq)

## Verify momentum conservation: Σ feq·e = ρu
@show sum(feq .* cx)  # should be ≈ 0.1
@show sum(feq .* cy)  # should be ≈ 0.0

# ## See in action
#
# - [Poiseuille channel](../examples/01_poiseuille_2d.md) — D2Q9 in its
#   simplest workflow.
# - [Lid-driven cavity 2D](../examples/04_cavity_2d.md) — the reference
#   D2Q9 benchmark.
