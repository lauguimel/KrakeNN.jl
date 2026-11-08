using KernelAbstractions

# --- Thermal D3Q19: Double Distribution Function for temperature (3D) ---
#
# Temperature field T tracked by g[i,j,k,q] on D3Q19.
# Equilibrium: g_eq_q = w_q * T * (1 + 3 * c_q·u)
# Collision:   g_q = g_q - ω_T * (g_q - g_eq_q)
# Macroscopic: T = Σ g_q

# --- Collide thermal (BGK on temperature populations, D3Q19) ---

@kernel function collide_thermal_3d_kernel!(g, @Const(ux), @Const(uy), @Const(uz), ω_T)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        T = eltype(g)
        g1=g[i,j,k,1]; g2=g[i,j,k,2]; g3=g[i,j,k,3]; g4=g[i,j,k,4]
        g5=g[i,j,k,5]; g6=g[i,j,k,6]; g7=g[i,j,k,7]; g8=g[i,j,k,8]
        g9=g[i,j,k,9]; g10=g[i,j,k,10]; g11=g[i,j,k,11]; g12=g[i,j,k,12]
        g13=g[i,j,k,13]; g14=g[i,j,k,14]; g15=g[i,j,k,15]; g16=g[i,j,k,16]
        g17=g[i,j,k,17]; g18=g[i,j,k,18]; g19=g[i,j,k,19]

        Temp = g1+g2+g3+g4+g5+g6+g7+g8+g9+g10+g11+g12+g13+g14+g15+g16+g17+g18+g19

        u_x = ux[i,j,k]; u_y = uy[i,j,k]; u_z = uz[i,j,k]
        t3 = T(3)

        # Thermal equilibrium: g_eq = w_q * Temp * (1 + 3 * c_q·u)
        # q=1: (0,0,0), w=1/3
        geq = T(1.0/3.0) * Temp
        g[i,j,k,1] = g1 - ω_T * (g1 - geq)

        # q=2: (+1,0,0), w=1/18
        geq = T(1.0/18.0) * Temp * (one(T) + t3 * u_x)
        g[i,j,k,2] = g2 - ω_T * (g2 - geq)

        # q=3: (-1,0,0), w=1/18
        geq = T(1.0/18.0) * Temp * (one(T) - t3 * u_x)
        g[i,j,k,3] = g3 - ω_T * (g3 - geq)

        # q=4: (0,+1,0), w=1/18
        geq = T(1.0/18.0) * Temp * (one(T) + t3 * u_y)
        g[i,j,k,4] = g4 - ω_T * (g4 - geq)

        # q=5: (0,-1,0), w=1/18
        geq = T(1.0/18.0) * Temp * (one(T) - t3 * u_y)
        g[i,j,k,5] = g5 - ω_T * (g5 - geq)

        # q=6: (0,0,+1), w=1/18
        geq = T(1.0/18.0) * Temp * (one(T) + t3 * u_z)
        g[i,j,k,6] = g6 - ω_T * (g6 - geq)

        # q=7: (0,0,-1), w=1/18
        geq = T(1.0/18.0) * Temp * (one(T) - t3 * u_z)
        g[i,j,k,7] = g7 - ω_T * (g7 - geq)

        # q=8: (+1,+1,0), w=1/36
        geq = T(1.0/36.0) * Temp * (one(T) + t3 * (u_x + u_y))
        g[i,j,k,8] = g8 - ω_T * (g8 - geq)

        # q=9: (-1,+1,0), w=1/36
        geq = T(1.0/36.0) * Temp * (one(T) + t3 * (-u_x + u_y))
        g[i,j,k,9] = g9 - ω_T * (g9 - geq)

        # q=10: (+1,-1,0), w=1/36
        geq = T(1.0/36.0) * Temp * (one(T) + t3 * (u_x - u_y))
        g[i,j,k,10] = g10 - ω_T * (g10 - geq)

        # q=11: (-1,-1,0), w=1/36
        geq = T(1.0/36.0) * Temp * (one(T) + t3 * (-u_x - u_y))
        g[i,j,k,11] = g11 - ω_T * (g11 - geq)

        # q=12: (+1,0,+1), w=1/36
        geq = T(1.0/36.0) * Temp * (one(T) + t3 * (u_x + u_z))
        g[i,j,k,12] = g12 - ω_T * (g12 - geq)

        # q=13: (-1,0,+1), w=1/36
        geq = T(1.0/36.0) * Temp * (one(T) + t3 * (-u_x + u_z))
        g[i,j,k,13] = g13 - ω_T * (g13 - geq)

        # q=14: (+1,0,-1), w=1/36
        geq = T(1.0/36.0) * Temp * (one(T) + t3 * (u_x - u_z))
        g[i,j,k,14] = g14 - ω_T * (g14 - geq)

        # q=15: (-1,0,-1), w=1/36
        geq = T(1.0/36.0) * Temp * (one(T) + t3 * (-u_x - u_z))
        g[i,j,k,15] = g15 - ω_T * (g15 - geq)

        # q=16: (0,+1,+1), w=1/36
        geq = T(1.0/36.0) * Temp * (one(T) + t3 * (u_y + u_z))
        g[i,j,k,16] = g16 - ω_T * (g16 - geq)

        # q=17: (0,-1,+1), w=1/36
        geq = T(1.0/36.0) * Temp * (one(T) + t3 * (-u_y + u_z))
        g[i,j,k,17] = g17 - ω_T * (g17 - geq)

        # q=18: (0,+1,-1), w=1/36
        geq = T(1.0/36.0) * Temp * (one(T) + t3 * (u_y - u_z))
        g[i,j,k,18] = g18 - ω_T * (g18 - geq)

        # q=19: (0,-1,-1), w=1/36
        geq = T(1.0/36.0) * Temp * (one(T) + t3 * (-u_y - u_z))
        g[i,j,k,19] = g19 - ω_T * (g19 - geq)
    end
end

"""
    collide_thermal_3d!(g, ux, uy, uz, ω_T)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.collide_thermal_3d!)
```
"""
function collide_thermal_3d!(g, ux, uy, uz, ω_T)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny, Nz = size(g, 1), size(g, 2), size(g, 3)
    kernel! = collide_thermal_3d_kernel!(backend)
    kernel!(g, ux, uy, uz, eltype(g)(ω_T); ndrange=(Nx, Ny, Nz))
end

# --- Compute temperature from thermal populations (D3Q19) ---

@kernel function compute_temperature_3d_kernel!(Temp, @Const(g))
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        Temp[i,j,k] = g[i,j,k,1]+g[i,j,k,2]+g[i,j,k,3]+g[i,j,k,4]+g[i,j,k,5]+
                       g[i,j,k,6]+g[i,j,k,7]+g[i,j,k,8]+g[i,j,k,9]+g[i,j,k,10]+
                       g[i,j,k,11]+g[i,j,k,12]+g[i,j,k,13]+g[i,j,k,14]+g[i,j,k,15]+
                       g[i,j,k,16]+g[i,j,k,17]+g[i,j,k,18]+g[i,j,k,19]
    end
end

"""
    compute_temperature_3d!(Temp, g)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.compute_temperature_3d!)
```
"""
function compute_temperature_3d!(Temp, g)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny, Nz = size(Temp)
    kernel! = compute_temperature_3d_kernel!(backend)
    kernel!(Temp, g; ndrange=(Nx, Ny, Nz))
end

# --- Thermal streaming of the x-heated box (D3Q19) ---
#
# Pull streaming of the thermal populations of a box whose west and east faces
# carry a fixed temperature and whose four other faces are adiabatic, with
# the slot numbering of `stream_3d!`. At an adiabatic face (y = 0.5,
# y = Ny + 0.5, z = 0.5, z = Nz + 0.5) a population whose source lies across
# the wall is the mirror image, wall-normal component flipped, of the one
# leaving the same wall layer towards it: zero normal heat flux, tangential
# flux kept (bounce-back would reverse the tangential flux of the wall layer).
# A link through an edge of two adiabatic faces is mirrored on both, which is
# the node's own opposite population. A slot whose source lies across the west
# or east face keeps the bounced value g*_opp(x_b), which
# `apply_fixed_temp_west_3d!` / `apply_fixed_temp_east_3d!` turn into the
# Dirichlet rule (edge slots included). On a wall layer jm/jp (km/kp) is the
# layer itself, so the mirrored slot is read there.

@kernel function stream_thermal_cavity_3d_kernel!(g_out, @Const(g_in), Nx, Ny, Nz)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        im = max(i-1, 1); ip = min(i+1, Nx)
        jm = max(j-1, 1); jp = min(j+1, Ny)
        km = max(k-1, 1); kp = min(k+1, Nz)
        jlo = j > 1; jhi = j < Ny; klo = k > 1; khi = k < Nz

        g_out[i,j,k,1] = g_in[i,j,k,1]

        g_out[i,j,k,2] = ifelse(i > 1,  g_in[im,j,k,2], g_in[i,j,k,3])
        g_out[i,j,k,3] = ifelse(i < Nx, g_in[ip,j,k,3], g_in[i,j,k,2])
        g_out[i,j,k,4] = ifelse(jlo, g_in[i,jm,k,4], g_in[i,j,k,5])
        g_out[i,j,k,5] = ifelse(jhi, g_in[i,jp,k,5], g_in[i,j,k,4])
        g_out[i,j,k,6] = ifelse(klo, g_in[i,j,km,6], g_in[i,j,k,7])
        g_out[i,j,k,7] = ifelse(khi, g_in[i,j,kp,7], g_in[i,j,k,6])

        # Edge xy: 8 (+x,+y), 9 (-x,+y), 10 (+x,-y), 11 (-x,-y)
        g_out[i,j,k,8]  = ifelse(i > 1,  g_in[im,jm,k,ifelse(jlo, 8, 10)],  g_in[i,j,k,11])
        g_out[i,j,k,9]  = ifelse(i < Nx, g_in[ip,jm,k,ifelse(jlo, 9, 11)],  g_in[i,j,k,10])
        g_out[i,j,k,10] = ifelse(i > 1,  g_in[im,jp,k,ifelse(jhi, 10, 8)],  g_in[i,j,k,9])
        g_out[i,j,k,11] = ifelse(i < Nx, g_in[ip,jp,k,ifelse(jhi, 11, 9)],  g_in[i,j,k,8])

        # Edge xz: 12 (+x,+z), 13 (-x,+z), 14 (+x,-z), 15 (-x,-z)
        g_out[i,j,k,12] = ifelse(i > 1,  g_in[im,j,km,ifelse(klo, 12, 14)], g_in[i,j,k,15])
        g_out[i,j,k,13] = ifelse(i < Nx, g_in[ip,j,km,ifelse(klo, 13, 15)], g_in[i,j,k,14])
        g_out[i,j,k,14] = ifelse(i > 1,  g_in[im,j,kp,ifelse(khi, 14, 12)], g_in[i,j,k,13])
        g_out[i,j,k,15] = ifelse(i < Nx, g_in[ip,j,kp,ifelse(khi, 15, 13)], g_in[i,j,k,12])

        # Edge yz: 16 (+y,+z), 17 (-y,+z), 18 (+y,-z), 19 (-y,-z); both
        # components can be mirrored.
        g_out[i,j,k,16] = g_in[i,jm,km,ifelse(jlo, ifelse(klo, 16, 18), ifelse(klo, 17, 19))]
        g_out[i,j,k,17] = g_in[i,jp,km,ifelse(jhi, ifelse(klo, 17, 19), ifelse(klo, 16, 18))]
        g_out[i,j,k,18] = g_in[i,jm,kp,ifelse(jlo, ifelse(khi, 18, 16), ifelse(khi, 19, 17))]
        g_out[i,j,k,19] = g_in[i,jp,kp,ifelse(jhi, ifelse(khi, 19, 17), ifelse(khi, 18, 16))]
    end
end

function stream_thermal_cavity_3d!(g_out, g_in, Nx, Ny, Nz)
    backend = KernelAbstractions.get_backend(g_in)
    kernel! = stream_thermal_cavity_3d_kernel!(backend)
    kernel!(g_out, g_in, Nx, Ny, Nz; ndrange=(Nx, Ny, Nz))
end

# --- Fixed temperature BCs (half-way anti-bounce-back Dirichlet, D3Q19) ---
#
# The wall sits half a lattice spacing outside the boundary node (x = 0.5 and
# x = Nx + 0.5 for west/east, likewise in y and z), where the half-way no-slip
# walls of the flow sit. These kernels act on the post-stream array of
# `stream_3d!`, whose bounce-back has already put the post-collision value
# g*_opp(x_b, t) into each unknown slot q (pointing into the domain):
#   g_q(x_b, t+1) = -g*_opp(x_b, t) + 2 w_q T_wall,
# i.e. a sign flip of that slot plus the wall term. Not idempotent: call once
# per step, after streaming and before collision.
#
# Edges. Along an edge, the diagonal slot whose link leaves the domain through
# two faces (e.g. slot 8 = (+x,+y) at the nodes i = 1, j = 1) is a
# wall-crossing slot of both faces. When both faces are Dirichlet it must be
# flipped once, not once per face. Rule: x-faces (west, east) apply their edge
# slots, y-faces (south, north) apply their edges with z-faces and leave their
# edges with x-faces alone, z-faces (bottom, top) leave all their edge slots
# alone; the face that applies an edge slot uses the mean of the two wall
# temperatures (the link midpoint lies on the edge, where both walls meet).
# D3Q19 has no (±1,±1,±1) link, so no slot crosses three faces. Each face
# kernel takes, for its four edge slots, the edge temperature and a skip flag;
# the per-face functions pass the single-face values (own temperature, no
# skip), and `apply_fixed_temp_3d!` sets them from the full set of Dirichlet
# faces.

# Anti-bounce-back of one slot `gq` with doubled weight `w2`. An edge slot
# (`on_edge`) is left unchanged when `skip` is set and otherwise uses the edge
# temperature `T_edge`; every other slot uses the face temperature.
@inline function _abb_slot_3d(gq, w2, T_face, on_edge, T_edge, skip)
    T_link = ifelse(on_edge, T_edge, T_face)
    return ifelse(on_edge & skip, gq, -gq + w2 * T_link)
end

# West wall (i=1, wall at x = 0.5): unknown pops with cx>0 → {2, 8, 10, 12, 14}
# Edge slots: 8 at j = 1 (south), 10 at j = Ny (north), 12 at k = 1 (bottom),
# 14 at k = Nz (top).
@kernel function apply_fixed_temp_west_3d_kernel!(g, T_wall, Ny, Nz, T_e, skip_e)
    j, k = @index(Global, NTuple)
    i = 1
    @inbounds begin
        TT = eltype(g)
        wa = TT(2) * TT(1.0/18.0)
        we = TT(2) * TT(1.0/36.0)
        g[i,j,k,2]  = -g[i,j,k,2] + wa * T_wall   # bounced from 3 (-x)
        g[i,j,k,8]  = _abb_slot_3d(g[i,j,k,8],  we, T_wall, j == 1,  T_e[1], skip_e[1])  # from 11 (-x,-y)
        g[i,j,k,10] = _abb_slot_3d(g[i,j,k,10], we, T_wall, j == Ny, T_e[2], skip_e[2])  # from 9 (-x,+y)
        g[i,j,k,12] = _abb_slot_3d(g[i,j,k,12], we, T_wall, k == 1,  T_e[3], skip_e[3])  # from 15 (-x,-z)
        g[i,j,k,14] = _abb_slot_3d(g[i,j,k,14], we, T_wall, k == Nz, T_e[4], skip_e[4])  # from 13 (-x,+z)
    end
end

"""
    apply_fixed_temp_west_3d!(g, T_wall, Ny, Nz)

Fixed temperature `T_wall` on the west face by half-way anti-bounce-back: the
wall sits at x = 0.5, half a lattice spacing outside the boundary nodes,
where the half-way no-slip wall of the flow sits. `g` must be the post-stream
array of `stream_3d!`; call once per step, after streaming and before
collision (the update flips the sign of the bounced slots, so a second call
undoes the first). For Dirichlet temperatures on two faces that share an edge,
use `apply_fixed_temp_3d!`, which flips each edge slot once.

```julia
using KrakeNN

methods(KrakeNN.apply_fixed_temp_west_3d!)
```
"""
function apply_fixed_temp_west_3d!(g, T_wall, Ny, Nz)
    TT = eltype(g)
    t = TT(T_wall)
    return _fixed_temp_west_3d!(g, t, Ny, Nz, (t, t, t, t),
                                (false, false, false, false))
end

function _fixed_temp_west_3d!(g, T_wall, Ny, Nz, T_e, skip_e)
    backend = KernelAbstractions.get_backend(g)
    kernel! = apply_fixed_temp_west_3d_kernel!(backend)
    kernel!(g, T_wall, Ny, Nz, T_e, skip_e; ndrange=(Ny, Nz))
end

# East wall (i=Nx, wall at x = Nx + 0.5): unknown pops with cx<0 → {3, 9, 11, 13, 15}
# Edge slots: 9 at j = 1 (south), 11 at j = Ny (north), 13 at k = 1 (bottom),
# 15 at k = Nz (top).
@kernel function apply_fixed_temp_east_3d_kernel!(g, T_wall, Nx, Ny, Nz, T_e, skip_e)
    j, k = @index(Global, NTuple)
    i = Nx
    @inbounds begin
        TT = eltype(g)
        wa = TT(2) * TT(1.0/18.0)
        we = TT(2) * TT(1.0/36.0)
        g[i,j,k,3]  = -g[i,j,k,3] + wa * T_wall   # bounced from 2 (+x)
        g[i,j,k,9]  = _abb_slot_3d(g[i,j,k,9],  we, T_wall, j == 1,  T_e[1], skip_e[1])  # from 10 (+x,-y)
        g[i,j,k,11] = _abb_slot_3d(g[i,j,k,11], we, T_wall, j == Ny, T_e[2], skip_e[2])  # from 8 (+x,+y)
        g[i,j,k,13] = _abb_slot_3d(g[i,j,k,13], we, T_wall, k == 1,  T_e[3], skip_e[3])  # from 14 (+x,-z)
        g[i,j,k,15] = _abb_slot_3d(g[i,j,k,15], we, T_wall, k == Nz, T_e[4], skip_e[4])  # from 12 (+x,+z)
    end
end

"""
    apply_fixed_temp_east_3d!(g, T_wall, Nx, Ny, Nz)

Fixed temperature `T_wall` on the east face by half-way anti-bounce-back: the
wall sits at x = Nx + 0.5, half a lattice spacing outside the boundary nodes,
where the half-way no-slip wall of the flow sits. `g` must be the post-stream
array of `stream_3d!`; call once per step, after streaming and before
collision (the update flips the sign of the bounced slots, so a second call
undoes the first). For Dirichlet temperatures on two faces that share an edge,
use `apply_fixed_temp_3d!`, which flips each edge slot once.

```julia
using KrakeNN

methods(KrakeNN.apply_fixed_temp_east_3d!)
```
"""
function apply_fixed_temp_east_3d!(g, T_wall, Nx, Ny, Nz)
    TT = eltype(g)
    t = TT(T_wall)
    return _fixed_temp_east_3d!(g, t, Nx, Ny, Nz, (t, t, t, t),
                                (false, false, false, false))
end

function _fixed_temp_east_3d!(g, T_wall, Nx, Ny, Nz, T_e, skip_e)
    backend = KernelAbstractions.get_backend(g)
    kernel! = apply_fixed_temp_east_3d_kernel!(backend)
    kernel!(g, T_wall, Nx, Ny, Nz, T_e, skip_e; ndrange=(Ny, Nz))
end

# South wall (j=1, wall at y = 0.5): unknown pops with cy>0 → {4, 8, 9, 16, 18}
# Edge slots: 8 at i = 1 (west), 9 at i = Nx (east), 16 at k = 1 (bottom),
# 18 at k = Nz (top).
@kernel function apply_fixed_temp_south_3d_kernel!(g, T_wall, Nx, Nz, T_e, skip_e)
    i, k = @index(Global, NTuple)
    j = 1
    @inbounds begin
        TT = eltype(g)
        wa = TT(2) * TT(1.0/18.0)
        we = TT(2) * TT(1.0/36.0)
        g[i,j,k,4]  = -g[i,j,k,4] + wa * T_wall   # bounced from 5 (-y)
        g[i,j,k,8]  = _abb_slot_3d(g[i,j,k,8],  we, T_wall, i == 1,  T_e[1], skip_e[1])  # from 11 (-x,-y)
        g[i,j,k,9]  = _abb_slot_3d(g[i,j,k,9],  we, T_wall, i == Nx, T_e[2], skip_e[2])  # from 10 (+x,-y)
        g[i,j,k,16] = _abb_slot_3d(g[i,j,k,16], we, T_wall, k == 1,  T_e[3], skip_e[3])  # from 19 (-y,-z)
        g[i,j,k,18] = _abb_slot_3d(g[i,j,k,18], we, T_wall, k == Nz, T_e[4], skip_e[4])  # from 17 (-y,+z)
    end
end

"""
    apply_fixed_temp_south_3d!(g, T_wall, Nx, Nz)

Fixed temperature `T_wall` on the south face by half-way anti-bounce-back: the
wall sits at y = 0.5, half a lattice spacing outside the boundary nodes,
where the half-way no-slip wall of the flow sits. `g` must be the post-stream
array of `stream_3d!`; call once per step, after streaming and before
collision (the update flips the sign of the bounced slots, so a second call
undoes the first). For Dirichlet temperatures on two faces that share an edge,
use `apply_fixed_temp_3d!`, which flips each edge slot once.

```julia
using KrakeNN

methods(KrakeNN.apply_fixed_temp_south_3d!)
```
"""
function apply_fixed_temp_south_3d!(g, T_wall, Nx, Nz)
    TT = eltype(g)
    t = TT(T_wall)
    return _fixed_temp_south_3d!(g, t, Nx, Nz, (t, t, t, t),
                                 (false, false, false, false))
end

function _fixed_temp_south_3d!(g, T_wall, Nx, Nz, T_e, skip_e)
    backend = KernelAbstractions.get_backend(g)
    kernel! = apply_fixed_temp_south_3d_kernel!(backend)
    kernel!(g, T_wall, Nx, Nz, T_e, skip_e; ndrange=(Nx, Nz))
end

# North wall (j=Ny, wall at y = Ny + 0.5): unknown pops with cy<0 → {5, 10, 11, 17, 19}
# Edge slots: 10 at i = 1 (west), 11 at i = Nx (east), 17 at k = 1 (bottom),
# 19 at k = Nz (top).
@kernel function apply_fixed_temp_north_3d_kernel!(g, T_wall, Nx, Ny, Nz, T_e, skip_e)
    i, k = @index(Global, NTuple)
    j = Ny
    @inbounds begin
        TT = eltype(g)
        wa = TT(2) * TT(1.0/18.0)
        we = TT(2) * TT(1.0/36.0)
        g[i,j,k,5]  = -g[i,j,k,5] + wa * T_wall   # bounced from 4 (+y)
        g[i,j,k,10] = _abb_slot_3d(g[i,j,k,10], we, T_wall, i == 1,  T_e[1], skip_e[1])  # from 9 (-x,+y)
        g[i,j,k,11] = _abb_slot_3d(g[i,j,k,11], we, T_wall, i == Nx, T_e[2], skip_e[2])  # from 8 (+x,+y)
        g[i,j,k,17] = _abb_slot_3d(g[i,j,k,17], we, T_wall, k == 1,  T_e[3], skip_e[3])  # from 18 (+y,-z)
        g[i,j,k,19] = _abb_slot_3d(g[i,j,k,19], we, T_wall, k == Nz, T_e[4], skip_e[4])  # from 16 (+y,+z)
    end
end

"""
    apply_fixed_temp_north_3d!(g, T_wall, Nx, Ny, Nz)

Fixed temperature `T_wall` on the north face by half-way anti-bounce-back: the
wall sits at y = Ny + 0.5, half a lattice spacing outside the boundary nodes,
where the half-way no-slip wall of the flow sits. `g` must be the post-stream
array of `stream_3d!`; call once per step, after streaming and before
collision (the update flips the sign of the bounced slots, so a second call
undoes the first). For Dirichlet temperatures on two faces that share an edge,
use `apply_fixed_temp_3d!`, which flips each edge slot once.

```julia
using KrakeNN

methods(KrakeNN.apply_fixed_temp_north_3d!)
```
"""
function apply_fixed_temp_north_3d!(g, T_wall, Nx, Ny, Nz)
    TT = eltype(g)
    t = TT(T_wall)
    return _fixed_temp_north_3d!(g, t, Nx, Ny, Nz, (t, t, t, t),
                                 (false, false, false, false))
end

function _fixed_temp_north_3d!(g, T_wall, Nx, Ny, Nz, T_e, skip_e)
    backend = KernelAbstractions.get_backend(g)
    kernel! = apply_fixed_temp_north_3d_kernel!(backend)
    kernel!(g, T_wall, Nx, Ny, Nz, T_e, skip_e; ndrange=(Nx, Nz))
end

# Bottom wall (k=1, wall at z = 0.5): unknown pops with cz>0 → {6, 12, 13, 16, 17}
# Edge slots: 12 at i = 1 (west), 13 at i = Nx (east), 16 at j = 1 (south),
# 17 at j = Ny (north).
@kernel function apply_fixed_temp_bottom_3d_kernel!(g, T_wall, Nx, Ny, T_e, skip_e)
    i, j = @index(Global, NTuple)
    k = 1
    @inbounds begin
        TT = eltype(g)
        wa = TT(2) * TT(1.0/18.0)
        we = TT(2) * TT(1.0/36.0)
        g[i,j,k,6]  = -g[i,j,k,6] + wa * T_wall   # bounced from 7 (-z)
        g[i,j,k,12] = _abb_slot_3d(g[i,j,k,12], we, T_wall, i == 1,  T_e[1], skip_e[1])  # from 15 (-x,-z)
        g[i,j,k,13] = _abb_slot_3d(g[i,j,k,13], we, T_wall, i == Nx, T_e[2], skip_e[2])  # from 14 (+x,-z)
        g[i,j,k,16] = _abb_slot_3d(g[i,j,k,16], we, T_wall, j == 1,  T_e[3], skip_e[3])  # from 19 (-y,-z)
        g[i,j,k,17] = _abb_slot_3d(g[i,j,k,17], we, T_wall, j == Ny, T_e[4], skip_e[4])  # from 18 (+y,-z)
    end
end

"""
    apply_fixed_temp_bottom_3d!(g, T_wall, Nx, Ny)

Fixed temperature `T_wall` on the bottom face by half-way anti-bounce-back: the
wall sits at z = 0.5, half a lattice spacing outside the boundary nodes,
where the half-way no-slip wall of the flow sits. `g` must be the post-stream
array of `stream_3d!`; call once per step, after streaming and before
collision (the update flips the sign of the bounced slots, so a second call
undoes the first). For Dirichlet temperatures on two faces that share an edge,
use `apply_fixed_temp_3d!`, which flips each edge slot once.

```julia
using KrakeNN

methods(KrakeNN.apply_fixed_temp_bottom_3d!)
```
"""
function apply_fixed_temp_bottom_3d!(g, T_wall, Nx, Ny)
    TT = eltype(g)
    t = TT(T_wall)
    return _fixed_temp_bottom_3d!(g, t, Nx, Ny, (t, t, t, t),
                                  (false, false, false, false))
end

function _fixed_temp_bottom_3d!(g, T_wall, Nx, Ny, T_e, skip_e)
    backend = KernelAbstractions.get_backend(g)
    kernel! = apply_fixed_temp_bottom_3d_kernel!(backend)
    kernel!(g, T_wall, Nx, Ny, T_e, skip_e; ndrange=(Nx, Ny))
end

# Top wall (k=Nz, wall at z = Nz + 0.5): unknown pops with cz<0 → {7, 14, 15, 18, 19}
# Edge slots: 14 at i = 1 (west), 15 at i = Nx (east), 18 at j = 1 (south),
# 19 at j = Ny (north).
@kernel function apply_fixed_temp_top_3d_kernel!(g, T_wall, Nx, Ny, Nz, T_e, skip_e)
    i, j = @index(Global, NTuple)
    k = Nz
    @inbounds begin
        TT = eltype(g)
        wa = TT(2) * TT(1.0/18.0)
        we = TT(2) * TT(1.0/36.0)
        g[i,j,k,7]  = -g[i,j,k,7] + wa * T_wall   # bounced from 6 (+z)
        g[i,j,k,14] = _abb_slot_3d(g[i,j,k,14], we, T_wall, i == 1,  T_e[1], skip_e[1])  # from 13 (-x,+z)
        g[i,j,k,15] = _abb_slot_3d(g[i,j,k,15], we, T_wall, i == Nx, T_e[2], skip_e[2])  # from 12 (+x,+z)
        g[i,j,k,18] = _abb_slot_3d(g[i,j,k,18], we, T_wall, j == 1,  T_e[3], skip_e[3])  # from 17 (-y,+z)
        g[i,j,k,19] = _abb_slot_3d(g[i,j,k,19], we, T_wall, j == Ny, T_e[4], skip_e[4])  # from 16 (+y,+z)
    end
end

"""
    apply_fixed_temp_top_3d!(g, T_wall, Nx, Ny, Nz)

Fixed temperature `T_wall` on the top face by half-way anti-bounce-back: the
wall sits at z = Nz + 0.5, half a lattice spacing outside the boundary nodes,
where the half-way no-slip wall of the flow sits. `g` must be the post-stream
array of `stream_3d!`; call once per step, after streaming and before
collision (the update flips the sign of the bounced slots, so a second call
undoes the first). For Dirichlet temperatures on two faces that share an edge,
use `apply_fixed_temp_3d!`, which flips each edge slot once.

```julia
using KrakeNN

methods(KrakeNN.apply_fixed_temp_top_3d!)
```
"""
function apply_fixed_temp_top_3d!(g, T_wall, Nx, Ny, Nz)
    TT = eltype(g)
    t = TT(T_wall)
    return _fixed_temp_top_3d!(g, t, Nx, Ny, Nz, (t, t, t, t),
                               (false, false, false, false))
end

function _fixed_temp_top_3d!(g, T_wall, Nx, Ny, Nz, T_e, skip_e)
    backend = KernelAbstractions.get_backend(g)
    kernel! = apply_fixed_temp_top_3d_kernel!(backend)
    kernel!(g, T_wall, Nx, Ny, Nz, T_e, skip_e; ndrange=(Nx, Ny))
end

# --- Several Dirichlet faces at once (edge rule) ---

const _FIXED_TEMP_FACES_3D = (:west, :east, :south, :north, :bottom, :top)

# Face temperatures of a `Dict{Symbol}` (missing faces are `nothing`), in the
# form `_apply_fixed_temp_faces_3d!` takes.
_fixed_temp_set_3d(d::AbstractDict) =
    (; west=get(d, :west, nothing), east=get(d, :east, nothing),
       south=get(d, :south, nothing), north=get(d, :north, nothing),
       bottom=get(d, :bottom, nothing), top=get(d, :top, nothing))

# Apply the Dirichlet faces listed in `faces` (a subset of the non-`nothing`
# entries of `temps`), with edge temperatures and ownership taken from the full
# set `temps = (; west, east, south, north, bottom, top)`. `run_refined_3d`
# uses the subset form for its second call on the faces touched by a patch.
# `_corner_temp` (thermal_2d.jl) gives the mean of two wall temperatures.
function _apply_fixed_temp_faces_3d!(g, Nx, Ny, Nz, temps::NamedTuple, faces)
    TT = eltype(g)
    Tw, Te, Ts, Tn = temps.west, temps.east, temps.south, temps.north
    Tb, Tt = temps.bottom, temps.top
    hw, he, hs, hn = Tw !== nothing, Te !== nothing, Ts !== nothing, Tn !== nothing
    hb, ht = Tb !== nothing, Tt !== nothing
    none = (false, false, false, false)
    for face in faces
        if face === :west && hw
            T_e = (_corner_temp(TT, Tw, Ts), _corner_temp(TT, Tw, Tn),
                   _corner_temp(TT, Tw, Tb), _corner_temp(TT, Tw, Tt))
            _fixed_temp_west_3d!(g, TT(Tw), Ny, Nz, T_e, none)
        elseif face === :east && he
            T_e = (_corner_temp(TT, Te, Ts), _corner_temp(TT, Te, Tn),
                   _corner_temp(TT, Te, Tb), _corner_temp(TT, Te, Tt))
            _fixed_temp_east_3d!(g, TT(Te), Nx, Ny, Nz, T_e, none)
        elseif face === :south && hs
            T_e = (TT(Ts), TT(Ts), _corner_temp(TT, Ts, Tb), _corner_temp(TT, Ts, Tt))
            _fixed_temp_south_3d!(g, TT(Ts), Nx, Nz, T_e, (hw, he, false, false))
        elseif face === :north && hn
            T_e = (TT(Tn), TT(Tn), _corner_temp(TT, Tn, Tb), _corner_temp(TT, Tn, Tt))
            _fixed_temp_north_3d!(g, TT(Tn), Nx, Ny, Nz, T_e, (hw, he, false, false))
        elseif face === :bottom && hb
            t = TT(Tb)
            _fixed_temp_bottom_3d!(g, t, Nx, Ny, (t, t, t, t), (hw, he, hs, hn))
        elseif face === :top && ht
            t = TT(Tt)
            _fixed_temp_top_3d!(g, t, Nx, Ny, Nz, (t, t, t, t), (hw, he, hs, hn))
        end
    end
    return g
end

"""
    apply_fixed_temp_3d!(g, Nx, Ny, Nz; west=nothing, east=nothing,
                         south=nothing, north=nothing,
                         bottom=nothing, top=nothing)

Fixed temperatures on several faces at once by half-way anti-bounce-back
(see `apply_fixed_temp_west_3d!` for a single face). A face is Dirichlet when
its keyword is a number. Where two Dirichlet faces share an edge, the diagonal
slot whose link crosses both walls is flipped once, with the mean of the two
wall temperatures; one face alone gives the same result as its per-face
function. `g` must be the post-stream array of `stream_3d!`; call once per
step, after streaming and before collision.

```julia
using KrakeNN

g = zeros(Float64, 6, 6, 6, 19)
apply_fixed_temp_3d!(g, 6, 6, 6; west=1.0, south=1.0, east=0.0, north=0.0)
```
"""
function apply_fixed_temp_3d!(g, Nx, Ny, Nz; west=nothing, east=nothing,
                              south=nothing, north=nothing,
                              bottom=nothing, top=nothing)
    temps = (; west, east, south, north, bottom, top)
    return _apply_fixed_temp_faces_3d!(g, Nx, Ny, Nz, temps, _FIXED_TEMP_FACES_3D)
end

# --- BGK collision with per-node Boussinesq body force (Guo scheme, D3Q19) ---
#
# Buoyancy: Fy = β_g * (T - T_ref), Fx = Fz = 0
# Gravity acts in the y-direction (convention: y is vertical in 3D LBM).

@kernel function collide_boussinesq_3d_kernel!(f, @Const(Temp), @Const(is_solid),
                                                ω, β_g, T_ref)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j, k]
            # Full bounce-back: swap opposite directions
            tmp = f[i,j,k,2]; f[i,j,k,2] = f[i,j,k,3]; f[i,j,k,3] = tmp
            tmp = f[i,j,k,4]; f[i,j,k,4] = f[i,j,k,5]; f[i,j,k,5] = tmp
            tmp = f[i,j,k,6]; f[i,j,k,6] = f[i,j,k,7]; f[i,j,k,7] = tmp
            tmp = f[i,j,k,8]; f[i,j,k,8] = f[i,j,k,11]; f[i,j,k,11] = tmp
            tmp = f[i,j,k,9]; f[i,j,k,9] = f[i,j,k,10]; f[i,j,k,10] = tmp
            tmp = f[i,j,k,12]; f[i,j,k,12] = f[i,j,k,15]; f[i,j,k,15] = tmp
            tmp = f[i,j,k,13]; f[i,j,k,13] = f[i,j,k,14]; f[i,j,k,14] = tmp
            tmp = f[i,j,k,16]; f[i,j,k,16] = f[i,j,k,19]; f[i,j,k,19] = tmp
            tmp = f[i,j,k,17]; f[i,j,k,17] = f[i,j,k,18]; f[i,j,k,18] = tmp
        else
            T = eltype(f)
            f1=f[i,j,k,1]; f2=f[i,j,k,2]; f3=f[i,j,k,3]; f4=f[i,j,k,4]
            f5=f[i,j,k,5]; f6=f[i,j,k,6]; f7=f[i,j,k,7]; f8=f[i,j,k,8]
            f9=f[i,j,k,9]; f10=f[i,j,k,10]; f11=f[i,j,k,11]; f12=f[i,j,k,12]
            f13=f[i,j,k,13]; f14=f[i,j,k,14]; f15=f[i,j,k,15]; f16=f[i,j,k,16]
            f17=f[i,j,k,17]; f18=f[i,j,k,18]; f19=f[i,j,k,19]

            # Buoyancy force: Fy = β_g * (T - T_ref), Fx = Fz = 0
            fy = β_g * (Temp[i,j,k] - T_ref)
            fx = zero(T)
            fz = zero(T)

            ρ = f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14+f15+f16+f17+f18+f19
            inv_ρ = one(T) / ρ
            ux = ((f2-f3+f8-f9+f10-f11+f12-f13+f14-f15) + fx/T(2)) * inv_ρ
            uy = ((f4-f5+f8+f9-f10-f11+f16-f17+f18-f19) + fy/T(2)) * inv_ρ
            uz = ((f6-f7+f12+f13-f14-f15+f16+f17-f18-f19) + fz/T(2)) * inv_ρ
            usq = ux*ux + uy*uy + uz*uz

            wr = T(1.0/3.0); wa = T(1.0/18.0); we = T(1.0/36.0)
            t3 = T(3); t45 = T(4.5); t15 = T(1.5); t9 = T(9)
            guo_pref = one(T) - ω / T(2)

            # q=1: (0,0,0), w=1/3
            feq = wr * ρ * (one(T) - t15*usq)
            Sq = wr * ((-ux)*fx + (-uy)*fy + (-uz)*fz) * t3
            f[i,j,k,1] = f1 - ω*(f1-feq) + guo_pref*Sq

            # q=2: (+1,0,0), w=1/18
            cu = ux
            feq = wa * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = wa * ((one(T)-ux)*fx + (-uy)*fy + (-uz)*fz) * t3 + wa * cu * fx * t9
            f[i,j,k,2] = f2 - ω*(f2-feq) + guo_pref*Sq

            # q=3: (-1,0,0), w=1/18
            cu = -ux
            feq = wa * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = wa * ((-one(T)-ux)*fx + (-uy)*fy + (-uz)*fz) * t3 + wa * cu * (-fx) * t9
            f[i,j,k,3] = f3 - ω*(f3-feq) + guo_pref*Sq

            # q=4: (0,+1,0), w=1/18
            cu = uy
            feq = wa * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = wa * ((-ux)*fx + (one(T)-uy)*fy + (-uz)*fz) * t3 + wa * cu * fy * t9
            f[i,j,k,4] = f4 - ω*(f4-feq) + guo_pref*Sq

            # q=5: (0,-1,0), w=1/18
            cu = -uy
            feq = wa * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = wa * ((-ux)*fx + (-one(T)-uy)*fy + (-uz)*fz) * t3 + wa * cu * (-fy) * t9
            f[i,j,k,5] = f5 - ω*(f5-feq) + guo_pref*Sq

            # q=6: (0,0,+1), w=1/18
            cu = uz
            feq = wa * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = wa * ((-ux)*fx + (-uy)*fy + (one(T)-uz)*fz) * t3 + wa * cu * fz * t9
            f[i,j,k,6] = f6 - ω*(f6-feq) + guo_pref*Sq

            # q=7: (0,0,-1), w=1/18
            cu = -uz
            feq = wa * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = wa * ((-ux)*fx + (-uy)*fy + (-one(T)-uz)*fz) * t3 + wa * cu * (-fz) * t9
            f[i,j,k,7] = f7 - ω*(f7-feq) + guo_pref*Sq

            # q=8: (+1,+1,0), w=1/36
            cu = ux+uy; cdotf = fx+fy
            feq = we * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = we * ((one(T)-ux)*fx + (one(T)-uy)*fy + (-uz)*fz) * t3 + we * cu * cdotf * t9
            f[i,j,k,8] = f8 - ω*(f8-feq) + guo_pref*Sq

            # q=9: (-1,+1,0), w=1/36
            cu = -ux+uy; cdotf = -fx+fy
            feq = we * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = we * ((-one(T)-ux)*fx + (one(T)-uy)*fy + (-uz)*fz) * t3 + we * cu * cdotf * t9
            f[i,j,k,9] = f9 - ω*(f9-feq) + guo_pref*Sq

            # q=10: (+1,-1,0), w=1/36
            cu = ux-uy; cdotf = fx-fy
            feq = we * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = we * ((one(T)-ux)*fx + (-one(T)-uy)*fy + (-uz)*fz) * t3 + we * cu * cdotf * t9
            f[i,j,k,10] = f10 - ω*(f10-feq) + guo_pref*Sq

            # q=11: (-1,-1,0), w=1/36
            cu = -ux-uy; cdotf = -fx-fy
            feq = we * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = we * ((-one(T)-ux)*fx + (-one(T)-uy)*fy + (-uz)*fz) * t3 + we * cu * cdotf * t9
            f[i,j,k,11] = f11 - ω*(f11-feq) + guo_pref*Sq

            # q=12: (+1,0,+1), w=1/36
            cu = ux+uz; cdotf = fx+fz
            feq = we * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = we * ((one(T)-ux)*fx + (-uy)*fy + (one(T)-uz)*fz) * t3 + we * cu * cdotf * t9
            f[i,j,k,12] = f12 - ω*(f12-feq) + guo_pref*Sq

            # q=13: (-1,0,+1), w=1/36
            cu = -ux+uz; cdotf = -fx+fz
            feq = we * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = we * ((-one(T)-ux)*fx + (-uy)*fy + (one(T)-uz)*fz) * t3 + we * cu * cdotf * t9
            f[i,j,k,13] = f13 - ω*(f13-feq) + guo_pref*Sq

            # q=14: (+1,0,-1), w=1/36
            cu = ux-uz; cdotf = fx-fz
            feq = we * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = we * ((one(T)-ux)*fx + (-uy)*fy + (-one(T)-uz)*fz) * t3 + we * cu * cdotf * t9
            f[i,j,k,14] = f14 - ω*(f14-feq) + guo_pref*Sq

            # q=15: (-1,0,-1), w=1/36
            cu = -ux-uz; cdotf = -fx-fz
            feq = we * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = we * ((-one(T)-ux)*fx + (-uy)*fy + (-one(T)-uz)*fz) * t3 + we * cu * cdotf * t9
            f[i,j,k,15] = f15 - ω*(f15-feq) + guo_pref*Sq

            # q=16: (0,+1,+1), w=1/36
            cu = uy+uz; cdotf = fy+fz
            feq = we * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = we * ((-ux)*fx + (one(T)-uy)*fy + (one(T)-uz)*fz) * t3 + we * cu * cdotf * t9
            f[i,j,k,16] = f16 - ω*(f16-feq) + guo_pref*Sq

            # q=17: (0,-1,+1), w=1/36
            cu = -uy+uz; cdotf = -fy+fz
            feq = we * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = we * ((-ux)*fx + (-one(T)-uy)*fy + (one(T)-uz)*fz) * t3 + we * cu * cdotf * t9
            f[i,j,k,17] = f17 - ω*(f17-feq) + guo_pref*Sq

            # q=18: (0,+1,-1), w=1/36
            cu = uy-uz; cdotf = fy-fz
            feq = we * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = we * ((-ux)*fx + (one(T)-uy)*fy + (-one(T)-uz)*fz) * t3 + we * cu * cdotf * t9
            f[i,j,k,18] = f18 - ω*(f18-feq) + guo_pref*Sq

            # q=19: (0,-1,-1), w=1/36
            cu = -uy-uz; cdotf = -fy-fz
            feq = we * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = we * ((-ux)*fx + (-one(T)-uy)*fy + (-one(T)-uz)*fz) * t3 + we * cu * cdotf * t9
            f[i,j,k,19] = f19 - ω*(f19-feq) + guo_pref*Sq
        end
    end
end

"""
    collide_boussinesq_3d!(f, Temp, is_solid, ω, β_g, T_ref)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.collide_boussinesq_3d!)
```
"""
function collide_boussinesq_3d!(f, Temp, is_solid, ω, β_g, T_ref)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny, Nz = size(f, 1), size(f, 2), size(f, 3)
    ET = eltype(f)
    kernel! = collide_boussinesq_3d_kernel!(backend)
    kernel!(f, Temp, is_solid, ET(ω), ET(β_g), ET(T_ref); ndrange=(Nx, Ny, Nz))
end
