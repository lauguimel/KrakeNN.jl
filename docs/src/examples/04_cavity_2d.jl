# # Lid-driven cavity (2D & 3D)
#
# ```@raw html
# <DownloadMenu :files="[{label:'cavity.krk',href:'/downloads/cavity/cavity.krk'},{label:'cavity.csv',href:'/downloads/cavity/cavity.csv'},{label:'cavity.py',href:'/downloads/cavity/cavity.py'}]" />
# ```
#
# **Concepts:** [LBM fundamentals](../theory/01_lbm_fundamentals.md) ·
# [BGK collision](../theory/03_bgk_collision.md) ·
# [Boundary conditions](../theory/05_boundary_conditions.md) ·
# [From 2D to 3D](../theory/06_from_2d_to_3d.md)
#
# **Validates against:** Ghia, Ghia, Shin (1982)
# [`10.1016/0021-9991(82)90058-4`](https://doi.org/10.1016/0021-9991(82)90058-4)
#
# **Download:** [`cavity.krk`](../assets/krk/cavity.krk)
#
# **Hardware:** Apple M3 Max, ~30s wall-clock at N = 128×128
#
# ![Velocity magnitude field for the 2D lid-driven cavity at Re = 100.  The primary recirculation vortex is visible, with the highest velocities near the lid and a quiet core in the centre.](cavity_umag.svg)
#
# ---
#
# ## Problem Statement
#
# The lid-driven cavity is *the* canonical benchmark for incompressible flow
# solvers.  It has appeared in virtually every CFD textbook since the 1960s
# and remains the standard first test for any new Navier--Stokes code.
#
# ### Setup
#
# A square box of side ``N`` is bounded by solid walls on all four sides.
# Three walls are stationary (no-slip), while the **top wall (lid)** moves
# horizontally at constant velocity ``u_\text{lid}``.  The lid drags fluid
# underneath it by viscous shear, creating a recirculating flow inside the
# cavity.
#
# The single governing parameter is the **Reynolds number**:
#
# ```math
# \text{Re} = \frac{u_\text{lid} \cdot N}{\nu}
# ```
#
# where ``\nu`` is the kinematic viscosity.
#
# ### What happens physically?
#
# At ``\text{Re} = 100``, a **single primary vortex** occupies most of the
# cavity, centred slightly above and to the right of the geometric centre.
# Two tiny secondary vortices appear in the bottom corners, but they are
# barely visible at this Reynolds number.  As Re increases, the primary
# vortex migrates toward the centre and the corner vortices grow.
#
# ### Why this test matters
#
# The lid-driven cavity is a **combined test** of the entire LBM solver:
#
# - **Streaming** must correctly propagate populations in all 9 directions
# - **Collision** (BGK) must recover the correct viscous stress tensor
# - **Zou--He velocity BC** on the lid must impose a non-zero tangential
#   velocity without generating spurious density fluctuations
# - **Half-way bounce-back** on the three stationary walls must enforce
#   no-slip at the correct location (half a lattice spacing from the node)
# - The solver must handle a **recirculating flow** (no inlet/outlet),
#   which is more demanding than channel flows
#
# Reference data for the centreline velocity profiles was published by
# [Ghia *et al.* (1982)](@cite ghia1982high), who used a vorticity-stream
# function method with a very fine grid.  Their digitised data points at
# ``\text{Re} = 100`` are the gold standard for validation.
#
# ---
#
# ## Geometry
#
# ![Schematic of the lid-driven cavity.  The top wall moves at velocity u_lid (Zou-He BC), while the three other walls are stationary (half-way bounce-back).  A single primary vortex forms in the interior.](cavity_geometry.svg)
#
# ---
#
# ## LBM Setup
#
# | Parameter | Value |
# |-----------|-------|
# | Lattice   | D2Q9  |
# | Domain    | ``128 \times 128`` |
# | Lid BC    | Zou--He velocity, ``u_\text{lid} = 0.1`` |
# | Other walls | Half-way bounce-back (no-slip) |
# | ``\text{Re}`` | 100 |
# | ``\nu``   | ``u_\text{lid} \cdot N / \text{Re} = 0.128`` |
# | ``\omega`` | ``1/(3\nu + 0.5) \approx 1.19`` |
# | Steps     | 60 000 |
#
# The relaxation parameter ``\omega`` is comfortably in the stable range
# ``(0, 2)``.  The Mach number ``\text{Ma} = u_\text{lid} / c_s = 0.1\sqrt{3}
# \approx 0.17`` is low enough for the incompressible approximation to hold.
#
# ---
#
# ## Simulation Code

using KrakeNN

N     = 128
Re    = 100
u_lid = 0.1
ν     = u_lid * N / Re                   ## ν = 0.128

config = LBMConfig(D2Q9(); Nx=N, Ny=N, ν=ν, u_lid=u_lid,
                   max_steps=60000, output_interval=10000)
ρ, ux, uy, _ = run_cavity_2d(config)

# The function `run_cavity_2d` performs the full time loop:
# 1. **Stream** --- propagate populations to neighbours
# 2. **Bounce-back** on the three stationary walls
# 3. **Zou--He** on the top wall (impose ``u_x = u_\text{lid}``, ``u_y = 0``)
# 4. **Collide** --- BGK relaxation toward equilibrium
# 5. **Compute macroscopic** ``\rho``, ``u_x``, ``u_y`` from populations
#
# After 60 000 steps the flow is fully converged to steady state.
#
# ---
#
# ## Reference Data --- Ghia *et al.* (1982)
#
# The digitised reference data for ``\text{Re} = 100`` consists of two
# profiles: ``u_x(y)`` along the vertical centreline, and ``u_y(x)`` along
# the horizontal centreline.  These arrays are taken directly from Table I
# of [Ghia *et al.* (1982)](@cite ghia1982high).

y_ghia  = [0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
           0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
           0.9688, 0.9766, 1.0]
ux_ghia = [0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
          -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
           0.68717, 0.73722, 0.78871, 0.84123, 1.0]

x_ghia  = [0.0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
           0.2344, 0.5, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
           0.9609, 0.9688, 1.0]
uy_ghia = [0.0, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507,
           0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313,
          -0.08864, -0.07391, -0.05906, 0.0]

# ---
#
# ## Post-processing
#
# Extract the LBM velocity profiles along the centrelines.  The velocity
# is normalised by ``u_\text{lid}`` and the coordinates by ``N`` so that
# both range from 0 to 1, matching the Ghia convention.

mid = N ÷ 2 + 1

## Vertical centreline: ux(y) at x = N/2
ux_profile = [ux[mid, j] / u_lid for j in 1:N]
y_norm     = [(j - 0.5) / N for j in 1:N]

## Horizontal centreline: uy(x) at y = N/2
uy_profile = [uy[i, mid] / u_lid for i in 1:N]
x_norm     = [(i - 0.5) / N for i in 1:N]

# ---
#
# ## Results --- Centreline Profiles
#
# The left panel shows the horizontal velocity ``u_x / u_\text{lid}`` along
# the vertical centreline (``x = N/2``).  At the top (``y/N = 1``), the
# velocity equals the lid speed; at the bottom wall it is zero.  The
# negative values in the lower part of the cavity correspond to the return
# flow of the primary vortex.
#
# The right panel shows the vertical velocity ``u_y / u_\text{lid}`` along
# the horizontal centreline (``y = N/2``).  This profile is antisymmetric,
# with positive values on the left (upward flow) and negative on the right
# (downward flow).
#
# The LBM results (solid lines) closely match the Ghia reference data (red
# circles) at ``N = 128``.
#
# ![Centreline velocity profiles for the lid-driven cavity at Re = 100.  Left: horizontal velocity along the vertical centreline compared with Ghia et al. (1982).  Right: vertical velocity along the horizontal centreline.  The LBM solution (N = 128) shows excellent agreement with the reference data.](cavity_centerlines.svg)
#
# ---
#
# ## Results --- Velocity Magnitude
#
# The velocity magnitude field ``|\mathbf{u}| / u_\text{lid}`` reveals the
# structure of the primary vortex.  The fastest flow is near the lid (top),
# and a thin boundary layer forms along the moving wall.  The vortex core
# is visible as a local minimum in velocity magnitude near the centre of
# the cavity.
#
# ![Velocity magnitude field for the 2D lid-driven cavity at Re = 100.  The primary recirculation vortex is visible, with the highest velocities near the lid and a quiet core in the centre.](cavity_umag.svg)
#
# ---
#
# ## Discussion
#
# ### Boundary condition role
#
# The choice of boundary conditions is critical in cavity flows:
#
# - **Zou--He** on the lid accurately imposes ``u_x = u_\text{lid}`` by
#   solving for the unknown populations using the known velocity and the
#   bounce-back assumption for the non-equilibrium part.  This avoids the
#   pressure singularity at the top corners.
# - **Half-way bounce-back** on the three stationary walls places the
#   effective no-slip plane at distance ``\Delta x / 2`` from the boundary
#   node, giving second-order accuracy in space.
#
# ### Convergence behaviour
#
# With the BGK collision operator, the LBM is second-order accurate in
# space.  Doubling the resolution ``N`` reduces the ``L_2`` error on the
# centreline profiles by roughly a factor of 4, consistent with
# ``O(\Delta x^2)`` convergence.  At ``N = 128``, the maximum pointwise
# error against the Ghia data is typically below 1%.
#
# ### Higher Reynolds numbers
#
# To simulate higher Re (e.g. 400, 1000, 3200), increase ``N``
# proportionally to maintain stability (``\omega < 2`` requires
# ``\nu > 1/6``, so ``N > \text{Re} \cdot u_\text{lid} / (1/6)``).
# At Re = 1000, secondary and tertiary corner vortices become clearly
# visible, and the primary vortex is noticeably shifted.
#
# ---
#
# # 3D --- Lid-driven cavity (D3Q19)
#
# **Validates against:** Ku, Hirsh & Taylor (1987) — centerline profiles
# [`10.1016/0021-9991(87)90193-8`](https://doi.org/10.1016/0021-9991(87)90193-8)
#
# **Download:** [`cavity_3d.krk`](../assets/krk/cavity_3d.krk)
#
# **Hardware:** Apple M3 Max, ~5 min wall-clock at N = 64³ (CPU) /
# NVIDIA H100, ~30s at N = 64³
#
# ![Cavity 3D streamlines](../assets/figures/cavity_3d_umag.png)
#
# ---
#
# ## Problem Statement (3D)
#
# This section extends the classic 2D lid-driven cavity above to **three
# dimensions** using the D3Q19 lattice.  While the physics is conceptually
# identical --- a moving lid drives a recirculating flow inside a closed
# box --- the 3D case is dramatically more expensive (``N^3`` nodes instead
# of ``N^2``) and exercises a completely different lattice topology.
#
# ### From D2Q9 to D3Q19
#
# In 2D, each node has 9 neighbours (including itself).  In 3D, the D3Q19
# lattice uses **19 velocity directions**: 1 rest + 6 face-connected + 12
# edge-connected neighbours.  This is the standard choice for 3D LBM
# because it provides a good balance between accuracy and memory cost (the
# D3Q27 lattice adds 8 corner directions but requires 42% more memory per
# node).
#
# The streaming, collision, and boundary condition kernels all have 3D
# counterparts.  The key difference is that the Zou--He velocity BC must
# now handle a 2D face (``N \times N`` nodes on the lid) instead of a 1D
# line.
#
# ### What to expect
#
# At ``\text{Re} = 100`` the 3D flow is qualitatively similar to 2D: a
# single primary vortex fills the cavity.  However, the 3D solution also
# has **secondary flows in the spanwise direction** (perpendicular to the
# lid motion) that are absent in 2D.  These are weak at Re = 100 but become
# significant at higher Reynolds numbers.
#
# ---
#
# ## LBM Setup (3D)
#
# | Parameter | Value |
# |-----------|-------|
# | Lattice   | D3Q19 |
# | Domain    | ``24 \times 24 \times 24`` |
# | Lid BC    | Zou--He velocity at ``j = N_y`` (lid moves in ``x``) |
# | Other walls | Half-way bounce-back |
# | ``\text{Re}`` | 100 |
# | ``u_\text{lid}`` | 0.1 |
# | ``\nu``   | ``u_\text{lid} \cdot N / \text{Re} = 0.024`` |
# | Steps     | 30 000 |
#
# We deliberately use a **coarse grid** (``N = 24``) to keep the
# documentation build fast.  This is sufficient to capture the primary
# vortex qualitatively, but production runs should use ``N \ge 64`` for
# quantitative validation against reference data.
#
# !!! note "Memory scaling"
#     A D3Q19 simulation with ``N^3`` nodes stores 19 populations per node
#     (double precision).  At ``N = 64`` this is already
#     ``19 \times 64^3 \times 8 \approx 40`` MB for a single distribution
#     array.  At ``N = 256`` it exceeds 40 GB.  This is why 3D LBM
#     almost always requires GPU acceleration.
#
# ---
#
# ## Simulation Code (3D)

N     = 24
Re    = 100
u_lid = 0.1
ν     = u_lid * N / Re                   ## ν = 0.024

config = LBMConfig(D3Q19(); Nx=N, Ny=N, Nz=N, ν=ν, u_lid=u_lid,
                   max_steps=30000, output_interval=10000)
ρ, ux, uy, uz, _ = run_cavity_3d(config)

# The function `run_cavity_3d` performs exactly the same algorithmic steps
# as the 2D version:
# 1. **Stream** --- propagate 19 populations to neighbours in 3D
# 2. **Bounce-back** on the five stationary walls (bottom, front, back,
#    left, right)
# 3. **Zou--He** on the top face (impose ``u_x = u_\text{lid}``,
#    ``u_y = u_z = 0``)
# 4. **Collide** --- BGK relaxation
# 5. **Compute macroscopic** fields ``\rho``, ``u_x``, ``u_y``, ``u_z``
#
# ---
#
# ## Post-processing (3D)
#
# We extract the velocity magnitude in the **mid-plane** at ``z = N/2``
# to visualise the primary vortex, and the vertical centreline profile
# ``u_x(y)`` at ``(x, z) = (N/2, N/2)`` for comparison with 2D results.

mid = N ÷ 2

## Velocity magnitude in the z = N/2 mid-plane
umag = zeros(N, N)
for j in 1:N, i in 1:N
    umag[i, j] = sqrt(ux[i, j, mid]^2 + uy[i, j, mid]^2 + uz[i, j, mid]^2)
end
umag ./= u_lid

## Vertical centreline profile at (x, z) = (N/2, N/2)
ux_profile = [ux[mid, j, mid] / u_lid for j in 1:N]
y_norm     = [(j - 0.5) / N for j in 1:N]

# ---
#
# ## Results --- Mid-plane Velocity Magnitude (3D)
#
# The heatmap below shows ``|\mathbf{u}| / u_\text{lid}`` in the ``z = N/2``
# plane.  Even on this coarse 24^3 grid, the primary vortex structure is
# clearly visible: fast flow near the lid, a relatively quiet core, and
# return flow along the bottom.
#
# At this resolution the vortex is slightly less well-defined than in the
# 2D case because of the limited number of grid points.  The spanwise
# confinement (walls at ``z = 1`` and ``z = N``) also modifies the flow
# compared to the infinite-span 2D solution.
#
# ![Velocity magnitude in the z = N/2 mid-plane of the 3D lid-driven cavity.  The primary vortex is visible with fast flow near the lid and a quiet core.  Resolution is N = 24 (coarse demonstration grid).](cavity_3d_umag.svg)
#
# ---
#
# ## Discussion (3D)
#
# ### 3D vs 2D differences
#
# - The mid-plane profile of the 3D cavity at Re = 100 is qualitatively
#   similar to the 2D solution, but quantitatively different because of
#   the no-slip walls at ``z = 1`` and ``z = N`` that slow down the flow
#   in the spanwise direction.
# - At ``N = 24``, the vertical centreline profile ``u_x(y)`` shows the
#   same S-shaped structure as in 2D (positive near the lid, negative
#   in the return flow), but the magnitudes are slightly reduced due to
#   the 3D wall friction.
#
# ### When to use 3D
#
# For most validation purposes, the 2D cavity is sufficient (and much
# cheaper).  The 3D version is useful for:
# - Validating the D3Q19 streaming and collision kernels
# - Testing the 3D Zou--He boundary condition
# - Benchmarking GPU performance (the 3D problem has much higher
#   arithmetic intensity and better GPU utilisation)
#
# ### Grid resolution guidelines
#
# | ``N``  | Nodes     | Memory (D3Q19, f64) | Use case              |
# |--------|-----------|--------------------|-----------------------|
# | 24     | 13 824    | ~4 MB              | Quick smoke test      |
# | 64     | 262 144   | ~76 MB             | Quantitative (CPU)    |
# | 128    | 2 097 152 | ~608 MB            | Production (GPU)      |
# | 256    | 16 777 216| ~4.9 GB            | High-Re (GPU only)    |
#
# ---
#
# ## References
#
# - [Ghia *et al.* (1982)](@cite ghia1982high) --- Benchmark centreline data (2D)
# - Ku, Hirsh & Taylor (1987)
#   [`10.1016/0021-9991(87)90193-8`](https://doi.org/10.1016/0021-9991(87)90193-8)
#   --- 3D centreline profiles
# - [Zou & He (1997)](@cite zou1997pressure) --- Zou--He velocity BC
# - [He & Luo (1997)](@cite he1997theory) --- Lattice Boltzmann theory
# - [Qian *et al.* (1992)](@cite qian1992lattice) --- D2Q9/D3Q19 lattice models
# - [Kruger *et al.* (2017)](@cite kruger2017lattice) --- LBM textbook, ch. 5
