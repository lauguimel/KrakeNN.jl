using KernelAbstractions

# --- Thermal D2Q9: Double Distribution Function for temperature ---
#
# Temperature field T is tracked by a separate population g[i,j,q] on D2Q9.
# Equilibrium: g_eq_q = w_q * T * (1 + c_q·u / cs²)
# Collision:   g_q = g_q - ω_T * (g_q - g_eq_q)
# where ω_T = 1 / (3α + 0.5), α = thermal diffusivity
# Macroscopic: T = Σ g_q

# --- Stream thermal populations (reuses same stream kernels as flow) ---
# For walls: adiabatic (bounce-back) or fixed T (anti-bounce-back)

# --- Collide thermal (BGK on temperature populations) ---

@kernel function collide_thermal_2d_kernel!(g, @Const(ux), @Const(uy), ω_T)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(g)
        g1=g[i,j,1]; g2=g[i,j,2]; g3=g[i,j,3]; g4=g[i,j,4]
        g5=g[i,j,5]; g6=g[i,j,6]; g7=g[i,j,7]; g8=g[i,j,8]; g9=g[i,j,9]

        # Macroscopic temperature
        Temp = g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 + g9

        # Local velocity (from flow solver)
        u_x = ux[i,j]
        u_y = uy[i,j]

        # Thermal equilibrium: g_eq = w_q * T * (1 + 3 * c_q · u)
        # (simplified: no u² terms for passive scalar transport)
        t3 = T(3)

        geq = T(4.0/9.0) * Temp
        g[i,j,1] = g1 - ω_T * (g1 - geq)

        geq = T(1.0/9.0) * Temp * (one(T) + t3 * u_x)
        g[i,j,2] = g2 - ω_T * (g2 - geq)

        geq = T(1.0/9.0) * Temp * (one(T) + t3 * u_y)
        g[i,j,3] = g3 - ω_T * (g3 - geq)

        geq = T(1.0/9.0) * Temp * (one(T) - t3 * u_x)
        g[i,j,4] = g4 - ω_T * (g4 - geq)

        geq = T(1.0/9.0) * Temp * (one(T) - t3 * u_y)
        g[i,j,5] = g5 - ω_T * (g5 - geq)

        geq = T(1.0/36.0) * Temp * (one(T) + t3 * (u_x + u_y))
        g[i,j,6] = g6 - ω_T * (g6 - geq)

        geq = T(1.0/36.0) * Temp * (one(T) + t3 * (-u_x + u_y))
        g[i,j,7] = g7 - ω_T * (g7 - geq)

        geq = T(1.0/36.0) * Temp * (one(T) + t3 * (-u_x - u_y))
        g[i,j,8] = g8 - ω_T * (g8 - geq)

        geq = T(1.0/36.0) * Temp * (one(T) + t3 * (u_x - u_y))
        g[i,j,9] = g9 - ω_T * (g9 - geq)
    end
end

"""
    collide_thermal_2d!(g, ux, uy, ω_T)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.collide_thermal_2d!)
```
"""
function collide_thermal_2d!(g, ux, uy, ω_T)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(g, 1), size(g, 2)
    kernel! = collide_thermal_2d_kernel!(backend)
    kernel!(g, ux, uy, ω_T; ndrange=(Nx, Ny))
end

@kernel function collide_thermal_masked_2d_kernel!(g, @Const(ux), @Const(uy), ω_T, @Const(is_skip))
    i, j = @index(Global, NTuple)
    @inbounds if !is_skip[i, j]
        T = eltype(g)
        u_x = T(ux[i, j]); u_y = T(uy[i, j])
        Temp = g[i,j,1] + g[i,j,2] + g[i,j,3] + g[i,j,4] +
               g[i,j,5] + g[i,j,6] + g[i,j,7] + g[i,j,8] + g[i,j,9]
        t3 = T(3)

        for q in 1:9
            w_q = ifelse(q == 1, T(4)/T(9), ifelse(q <= 5, T(1)/T(9), T(1)/T(36)))
            cx = ifelse(q==1,T(0), ifelse(q==2,T(1), ifelse(q==3,T(0), ifelse(q==4,T(-1),
                 ifelse(q==5,T(0), ifelse(q==6,T(1), ifelse(q==7,T(-1), ifelse(q==8,T(-1), T(1)))))))))
            cy = ifelse(q==1,T(0), ifelse(q==2,T(0), ifelse(q==3,T(1), ifelse(q==4,T(0),
                 ifelse(q==5,T(-1), ifelse(q==6,T(1), ifelse(q==7,T(1), ifelse(q==8,T(-1), T(-1)))))))))
            geq = w_q * Temp * (one(T) + t3 * (cx * u_x + cy * u_y))
            g[i, j, q] -= ω_T * (g[i, j, q] - geq)
        end
    end
end

function collide_thermal_masked_2d!(g, ux, uy, ω_T, is_skip)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(g, 1), size(g, 2)
    kernel! = collide_thermal_masked_2d_kernel!(backend)
    kernel!(g, ux, uy, ω_T, is_skip; ndrange=(Nx, Ny))
end

# --- Compute temperature from thermal populations ---

@kernel function compute_temperature_2d_kernel!(Temp, @Const(g))
    i, j = @index(Global, NTuple)
    @inbounds begin
        Temp[i,j] = g[i,j,1] + g[i,j,2] + g[i,j,3] + g[i,j,4] +
                    g[i,j,5] + g[i,j,6] + g[i,j,7] + g[i,j,8] + g[i,j,9]
    end
end

"""
    compute_temperature_2d!(Temp, g)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.compute_temperature_2d!)
```
"""
function compute_temperature_2d!(Temp, g)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(Temp)
    kernel! = compute_temperature_2d_kernel!(backend)
    kernel!(Temp, g; ndrange=(Nx, Ny))
end

# --- Fixed temperature BC (half-way anti-bounce-back for Dirichlet) ---
#
# The wall sits half a lattice spacing outside the boundary node (y = 0.5 for
# the south face, y = Ny + 0.5 for the north face, x = 0.5 and x = Nx + 0.5 for
# west/east), where the half-way no-slip walls of the flow sit. These kernels
# act on the post-stream array of a bounce-back stream kernel (`stream_2d!`,
# `stream_periodic_x_wall_y_2d!`): each wall-crossing slot already holds the
# bounced post-collision value g*_opp(x_b, t), and anti-bounce-back sets
#   g_q(x_b, t+1) = -g*_opp(x_b, t) + 2 w_q T_wall,
# i.e. a sign flip of that slot plus the wall term. The update is not
# idempotent: call it exactly once per time step, after streaming and before
# collision.
#
# Corners. At a corner node, the diagonal slot whose link leaves the domain
# through two faces (slot 6 at the south-west node, 9 at north-west, 7 at
# south-east, 8 at north-east) is a wall-crossing slot of both faces. When both
# faces are Dirichlet it must be flipped once, not once per face. Rule: the
# x-face (west or east) applies it, with the mean of the two wall temperatures
# (the link midpoint is the corner point, where both walls meet); the y-face
# (south or north) leaves it alone. Each face kernel takes, for its two corner
# slots, the corner temperature and a skip flag; the per-face functions pass
# the single-face values (own temperature, no skip), and
# `apply_fixed_temp_2d!` sets them from the full set of Dirichlet faces.

# Anti-bounce-back of one slot `gq` with doubled weight `w2`. A corner slot
# (`on_corner`) is left unchanged when `skip` is set and otherwise uses the
# corner temperature `T_corner`; every other slot uses the face temperature.
@inline function _abb_slot(gq, w2, T_face, on_corner, T_corner, skip)
    T_link = ifelse(on_corner, T_corner, T_face)
    return ifelse(on_corner & skip, gq, -gq + w2 * T_link)
end

# South wall (y = 0.5), unknown populations pointing north: 3, 6, 7.
# Corner slots: 6 at i = 1 (south-west), 7 at i = Nx (south-east).
@kernel function apply_fixed_temp_south_2d_kernel!(g, T_wall, Nx, T_c, skip_c)
    i = @index(Global)
    j = 1
    @inbounds begin
        TT = eltype(g)
        wa = TT(2) * TT(1.0/9.0)
        we = TT(2) * TT(1.0/36.0)
        g[i,j,3] = -g[i,j,3] + wa * T_wall
        g[i,j,6] = _abb_slot(g[i,j,6], we, T_wall, i == 1, T_c[1], skip_c[1])
        g[i,j,7] = _abb_slot(g[i,j,7], we, T_wall, i == Nx, T_c[2], skip_c[2])
    end
end

# North wall (y = Ny + 0.5), unknown populations pointing south: 5, 8, 9.
# Corner slots: 9 at i = 1 (north-west), 8 at i = Nx (north-east).
@kernel function apply_fixed_temp_north_2d_kernel!(g, T_wall, Nx, Ny, T_c, skip_c)
    i = @index(Global)
    j = Ny
    @inbounds begin
        TT = eltype(g)
        wa = TT(2) * TT(1.0/9.0)
        we = TT(2) * TT(1.0/36.0)
        g[i,j,5] = -g[i,j,5] + wa * T_wall
        g[i,j,9] = _abb_slot(g[i,j,9], we, T_wall, i == 1, T_c[1], skip_c[1])
        g[i,j,8] = _abb_slot(g[i,j,8], we, T_wall, i == Nx, T_c[2], skip_c[2])
    end
end

"""
    apply_fixed_temp_south_2d!(g, T_wall, Nx)

Fixed temperature `T_wall` on the south face by half-way anti-bounce-back: the
wall sits at y = 0.5, half a lattice spacing outside the boundary nodes, where
the half-way no-slip wall of the flow sits. `g` must be the post-stream array of
a bounce-back stream kernel; call once per step, after streaming and before
collision (the update flips the sign of the bounced slots, so a second call
undoes the first). For Dirichlet temperatures on two faces that share a corner,
use `apply_fixed_temp_2d!`, which flips each corner slot once.

```julia
using KrakeNN

methods(KrakeNN.apply_fixed_temp_south_2d!)
```
"""
function apply_fixed_temp_south_2d!(g, T_wall, Nx)
    TT = eltype(g)
    return _fixed_temp_south_2d!(g, TT(T_wall), Nx, (TT(T_wall), TT(T_wall)),
                                 (false, false))
end

function _fixed_temp_south_2d!(g, T_wall, Nx, T_c, skip_c)
    backend = KernelAbstractions.get_backend(g)
    kernel! = apply_fixed_temp_south_2d_kernel!(backend)
    kernel!(g, T_wall, Nx, T_c, skip_c; ndrange=(Nx,))
end

"""
    apply_fixed_temp_north_2d!(g, T_wall, Nx, Ny)

Fixed temperature `T_wall` on the north face by half-way anti-bounce-back: the
wall sits at y = Ny + 0.5, half a lattice spacing outside the boundary nodes, where
the half-way no-slip wall of the flow sits. `g` must be the post-stream array of
a bounce-back stream kernel; call once per step, after streaming and before
collision (the update flips the sign of the bounced slots, so a second call
undoes the first). For Dirichlet temperatures on two faces that share a corner,
use `apply_fixed_temp_2d!`, which flips each corner slot once.

```julia
using KrakeNN

methods(KrakeNN.apply_fixed_temp_north_2d!)
```
"""
function apply_fixed_temp_north_2d!(g, T_wall, Nx, Ny)
    TT = eltype(g)
    return _fixed_temp_north_2d!(g, TT(T_wall), Nx, Ny,
                                 (TT(T_wall), TT(T_wall)), (false, false))
end

function _fixed_temp_north_2d!(g, T_wall, Nx, Ny, T_c, skip_c)
    backend = KernelAbstractions.get_backend(g)
    kernel! = apply_fixed_temp_north_2d_kernel!(backend)
    kernel!(g, T_wall, Nx, Ny, T_c, skip_c; ndrange=(Nx,))
end

# --- Fixed temperature BC for west/east walls (natural convection cavity) ---

# West wall (x = 0.5), unknown populations pointing east:
# 2 (+1,0), 6 (+1,+1), 9 (+1,-1).
# Corner slots: 6 at j = 1 (south-west), 9 at j = Ny (north-west).
@kernel function apply_fixed_temp_west_2d_kernel!(g, T_wall, Ny, T_c, skip_c)
    j = @index(Global)
    i = 1
    @inbounds begin
        TT = eltype(g)
        wa = TT(2) * TT(1.0/9.0)
        we = TT(2) * TT(1.0/36.0)
        g[i,j,2] = -g[i,j,2] + wa * T_wall
        g[i,j,6] = _abb_slot(g[i,j,6], we, T_wall, j == 1, T_c[1], skip_c[1])
        g[i,j,9] = _abb_slot(g[i,j,9], we, T_wall, j == Ny, T_c[2], skip_c[2])
    end
end

# East wall (x = Nx + 0.5), unknown populations pointing west:
# 4 (-1,0), 7 (-1,+1), 8 (-1,-1).
# Corner slots: 7 at j = 1 (south-east), 8 at j = Ny (north-east).
@kernel function apply_fixed_temp_east_2d_kernel!(g, T_wall, Nx, Ny, T_c, skip_c)
    j = @index(Global)
    i = Nx
    @inbounds begin
        TT = eltype(g)
        wa = TT(2) * TT(1.0/9.0)
        we = TT(2) * TT(1.0/36.0)
        g[i,j,4] = -g[i,j,4] + wa * T_wall
        g[i,j,7] = _abb_slot(g[i,j,7], we, T_wall, j == 1, T_c[1], skip_c[1])
        g[i,j,8] = _abb_slot(g[i,j,8], we, T_wall, j == Ny, T_c[2], skip_c[2])
    end
end

"""
    apply_fixed_temp_west_2d!(g, T_wall, Ny)

Fixed temperature `T_wall` on the west face by half-way anti-bounce-back: the
wall sits at x = 0.5, half a lattice spacing outside the boundary nodes, where
the half-way no-slip wall of the flow sits. `g` must be the post-stream array of
a bounce-back stream kernel; call once per step, after streaming and before
collision (the update flips the sign of the bounced slots, so a second call
undoes the first). For Dirichlet temperatures on two faces that share a corner,
use `apply_fixed_temp_2d!`, which flips each corner slot once.

```julia
using KrakeNN

methods(KrakeNN.apply_fixed_temp_west_2d!)
```
"""
function apply_fixed_temp_west_2d!(g, T_wall, Ny)
    TT = eltype(g)
    return _fixed_temp_west_2d!(g, TT(T_wall), Ny, (TT(T_wall), TT(T_wall)),
                                (false, false))
end

function _fixed_temp_west_2d!(g, T_wall, Ny, T_c, skip_c)
    backend = KernelAbstractions.get_backend(g)
    kernel! = apply_fixed_temp_west_2d_kernel!(backend)
    kernel!(g, T_wall, Ny, T_c, skip_c; ndrange=(Ny,))
end

"""
    apply_fixed_temp_east_2d!(g, T_wall, Nx, Ny)

Fixed temperature `T_wall` on the east face by half-way anti-bounce-back: the
wall sits at x = Nx + 0.5, half a lattice spacing outside the boundary nodes, where
the half-way no-slip wall of the flow sits. `g` must be the post-stream array of
a bounce-back stream kernel; call once per step, after streaming and before
collision (the update flips the sign of the bounced slots, so a second call
undoes the first). For Dirichlet temperatures on two faces that share a corner,
use `apply_fixed_temp_2d!`, which flips each corner slot once.

```julia
using KrakeNN

methods(KrakeNN.apply_fixed_temp_east_2d!)
```
"""
function apply_fixed_temp_east_2d!(g, T_wall, Nx, Ny)
    TT = eltype(g)
    return _fixed_temp_east_2d!(g, TT(T_wall), Nx, Ny,
                                (TT(T_wall), TT(T_wall)), (false, false))
end

function _fixed_temp_east_2d!(g, T_wall, Nx, Ny, T_c, skip_c)
    backend = KernelAbstractions.get_backend(g)
    kernel! = apply_fixed_temp_east_2d_kernel!(backend)
    kernel!(g, T_wall, Nx, Ny, T_c, skip_c; ndrange=(Ny,))
end

# --- Several Dirichlet faces at once (corner rule) ---

# Temperature of a link that crosses face `a` (temperature `Ta`) and, when
# `Tb !== nothing`, the adjacent Dirichlet face `b`: the mean of the two.
_corner_temp(::Type{TT}, Ta, ::Nothing) where {TT} = TT(Ta)
_corner_temp(::Type{TT}, Ta, Tb) where {TT} = (TT(Ta) + TT(Tb)) / TT(2)

const _FIXED_TEMP_FACES_2D = (:west, :east, :south, :north)

# Face temperatures of a `Dict{Symbol}` (missing faces are `nothing`), in the
# form `_apply_fixed_temp_faces_2d!` takes.
_fixed_temp_set_2d(d::AbstractDict) =
    (; west=get(d, :west, nothing), east=get(d, :east, nothing),
       south=get(d, :south, nothing), north=get(d, :north, nothing))

# Apply the Dirichlet faces listed in `faces` (a subset of the non-`nothing`
# entries of `temps`), with corner temperatures and ownership taken from the
# full set `temps = (; west, east, south, north)`. `run_refined` uses the
# subset form for its second call on the faces touched by a patch.
function _apply_fixed_temp_faces_2d!(g, Nx, Ny, temps::NamedTuple, faces)
    TT = eltype(g)
    Tw, Te, Ts, Tn = temps.west, temps.east, temps.south, temps.north
    for face in faces
        if face === :west && Tw !== nothing
            _fixed_temp_west_2d!(g, TT(Tw), Ny,
                                 (_corner_temp(TT, Tw, Ts), _corner_temp(TT, Tw, Tn)),
                                 (false, false))
        elseif face === :east && Te !== nothing
            _fixed_temp_east_2d!(g, TT(Te), Nx, Ny,
                                 (_corner_temp(TT, Te, Ts), _corner_temp(TT, Te, Tn)),
                                 (false, false))
        elseif face === :south && Ts !== nothing
            _fixed_temp_south_2d!(g, TT(Ts), Nx, (TT(Ts), TT(Ts)),
                                  (Tw !== nothing, Te !== nothing))
        elseif face === :north && Tn !== nothing
            _fixed_temp_north_2d!(g, TT(Tn), Nx, Ny, (TT(Tn), TT(Tn)),
                                  (Tw !== nothing, Te !== nothing))
        end
    end
    return g
end

"""
    apply_fixed_temp_2d!(g, Nx, Ny; west=nothing, east=nothing,
                         south=nothing, north=nothing)

Fixed temperatures on several faces at once by half-way anti-bounce-back
(see `apply_fixed_temp_west_2d!` for a single face). A face is
Dirichlet when its keyword is a number. Where two Dirichlet faces share a
corner, the diagonal slot whose link crosses both walls is flipped once, with
the mean of the two wall temperatures; one face alone gives the same result as
its per-face function. `g` must be the post-stream array of a bounce-back
stream kernel; call once per step, after streaming and before collision.

```julia
using KrakeNN

g = zeros(Float64, 8, 8, 9)
apply_fixed_temp_2d!(g, 8, 8; west=1.0, south=1.0, east=0.0, north=0.0)
```
"""
function apply_fixed_temp_2d!(g, Nx, Ny; west=nothing, east=nothing,
                              south=nothing, north=nothing)
    temps = (; west, east, south, north)
    return _apply_fixed_temp_faces_2d!(g, Nx, Ny, temps, _FIXED_TEMP_FACES_2D)
end

# --- BGK collision with per-node Boussinesq body force (Guo scheme) ---

@kernel function collide_boussinesq_2d_kernel!(f, @Const(Temp), @Const(is_solid),
                                                ω, β_g, T_ref)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j]
            bounce_back_2d!(f, i, j)
        else
            T = eltype(f)
            f1=f[i,j,1]; f2=f[i,j,2]; f3=f[i,j,3]; f4=f[i,j,4]
            f5=f[i,j,5]; f6=f[i,j,6]; f7=f[i,j,7]; f8=f[i,j,8]; f9=f[i,j,9]

            # Per-node buoyancy force: Fy = β·g·(T - T_ref)
            fy = β_g * (Temp[i,j] - T_ref)
            fx = zero(T)

            ρ = f1+f2+f3+f4+f5+f6+f7+f8+f9
            inv_ρ = one(T) / ρ
            ux = ((f2-f4+f6-f7-f8+f9) + fx/T(2)) * inv_ρ
            uy = ((f3-f5+f6+f7-f8-f9) + fy/T(2)) * inv_ρ
            usq = ux*ux + uy*uy

            guo_pref = one(T) - ω / T(2)

            Sq = T(4.0/9.0) * ((-ux)*fx + (-uy)*fy)*T(3)
            f[i,j,1] = f1 - ω*(f1-feq_2d(Val(1), ρ, ux, uy, usq)) + guo_pref*Sq

            Sq = T(1.0/9.0)*((one(T)-ux)*fx+(-uy)*fy)*T(3) + T(1.0/9.0)*ux*fx*T(9)
            f[i,j,2] = f2 - ω*(f2-feq_2d(Val(2), ρ, ux, uy, usq)) + guo_pref*Sq

            Sq = T(1.0/9.0)*((-ux)*fx+(one(T)-uy)*fy)*T(3) + T(1.0/9.0)*uy*fy*T(9)
            f[i,j,3] = f3 - ω*(f3-feq_2d(Val(3), ρ, ux, uy, usq)) + guo_pref*Sq

            Sq = T(1.0/9.0)*((-one(T)-ux)*fx+(-uy)*fy)*T(3) + T(1.0/9.0)*ux*fx*T(9)
            f[i,j,4] = f4 - ω*(f4-feq_2d(Val(4), ρ, ux, uy, usq)) + guo_pref*Sq

            Sq = T(1.0/9.0)*((-ux)*fx+(-one(T)-uy)*fy)*T(3) + T(1.0/9.0)*uy*fy*T(9)
            f[i,j,5] = f5 - ω*(f5-feq_2d(Val(5), ρ, ux, uy, usq)) + guo_pref*Sq

            Sq = T(1.0/36.0)*((one(T)-ux)*fx+(one(T)-uy)*fy)*T(3) + T(1.0/36.0)*(ux+uy)*(fx+fy)*T(9)
            f[i,j,6] = f6 - ω*(f6-feq_2d(Val(6), ρ, ux, uy, usq)) + guo_pref*Sq

            Sq = T(1.0/36.0)*((-one(T)-ux)*fx+(one(T)-uy)*fy)*T(3) + T(1.0/36.0)*(-ux+uy)*(-fx+fy)*T(9)
            f[i,j,7] = f7 - ω*(f7-feq_2d(Val(7), ρ, ux, uy, usq)) + guo_pref*Sq

            Sq = T(1.0/36.0)*((-one(T)-ux)*fx+(-one(T)-uy)*fy)*T(3) + T(1.0/36.0)*(-ux-uy)*(-fx-fy)*T(9)
            f[i,j,8] = f8 - ω*(f8-feq_2d(Val(8), ρ, ux, uy, usq)) + guo_pref*Sq

            Sq = T(1.0/36.0)*((one(T)-ux)*fx+(-one(T)-uy)*fy)*T(3) + T(1.0/36.0)*(ux-uy)*(fx-fy)*T(9)
            f[i,j,9] = f9 - ω*(f9-feq_2d(Val(9), ρ, ux, uy, usq)) + guo_pref*Sq
        end
    end
end

"""
    collide_boussinesq_vt_2d!(f, Temp, is_solid, ν_ref, T_ref, A_arr, β_g)

BGK collision with per-node Boussinesq force AND temperature-dependent viscosity.
ν(T) = ν_ref · exp(A_arr · (1/T - 1/T_ref))  (Arrhenius law)
ω(T) = 1 / (3·ν(T) + 0.5)
"""
@kernel function collide_boussinesq_vt_2d_kernel!(f, @Const(Temp), @Const(is_solid),
                                                    ν_ref, T_ref, A_arr, β_g)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j]
            bounce_back_2d!(f, i, j)
        else
            T = eltype(f)
            f1=f[i,j,1]; f2=f[i,j,2]; f3=f[i,j,3]; f4=f[i,j,4]
            f5=f[i,j,5]; f6=f[i,j,6]; f7=f[i,j,7]; f8=f[i,j,8]; f9=f[i,j,9]

            # Temperature-dependent viscosity (Arrhenius)
            T_local = Temp[i,j]
            ν_local = ν_ref * exp(A_arr * (one(T)/T_local - one(T)/T_ref))
            ω_local = one(T) / (T(3) * ν_local + T(0.5))

            # Per-node buoyancy
            fy = β_g * (T_local - T_ref)
            fx = zero(T)

            ρ = f1+f2+f3+f4+f5+f6+f7+f8+f9
            inv_ρ = one(T) / ρ
            ux = ((f2-f4+f6-f7-f8+f9) + fx/T(2)) * inv_ρ
            uy = ((f3-f5+f6+f7-f8-f9) + fy/T(2)) * inv_ρ
            usq = ux*ux + uy*uy

            guo_pref = one(T) - ω_local / T(2)

            Sq=T(4.0/9.0)*((-ux)*fx+(-uy)*fy)*T(3)
            f[i,j,1]=f1-ω_local*(f1-feq_2d(Val(1), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/9.0)*((one(T)-ux)*fx+(-uy)*fy)*T(3)+T(1.0/9.0)*ux*fx*T(9)
            f[i,j,2]=f2-ω_local*(f2-feq_2d(Val(2), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/9.0)*((-ux)*fx+(one(T)-uy)*fy)*T(3)+T(1.0/9.0)*uy*fy*T(9)
            f[i,j,3]=f3-ω_local*(f3-feq_2d(Val(3), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/9.0)*((-one(T)-ux)*fx+(-uy)*fy)*T(3)+T(1.0/9.0)*ux*fx*T(9)
            f[i,j,4]=f4-ω_local*(f4-feq_2d(Val(4), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/9.0)*((-ux)*fx+(-one(T)-uy)*fy)*T(3)+T(1.0/9.0)*uy*fy*T(9)
            f[i,j,5]=f5-ω_local*(f5-feq_2d(Val(5), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/36.0)*((one(T)-ux)*fx+(one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(ux+uy)*(fx+fy)*T(9)
            f[i,j,6]=f6-ω_local*(f6-feq_2d(Val(6), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/36.0)*((-one(T)-ux)*fx+(one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(-ux+uy)*(-fx+fy)*T(9)
            f[i,j,7]=f7-ω_local*(f7-feq_2d(Val(7), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/36.0)*((-one(T)-ux)*fx+(-one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(-ux-uy)*(-fx-fy)*T(9)
            f[i,j,8]=f8-ω_local*(f8-feq_2d(Val(8), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/36.0)*((one(T)-ux)*fx+(-one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(ux-uy)*(fx-fy)*T(9)
            f[i,j,9]=f9-ω_local*(f9-feq_2d(Val(9), ρ, ux, uy, usq))+guo_pref*Sq
        end
    end
end

"""
    collide_boussinesq_vt_2d!(f, Temp, is_solid, ν_ref, T_ref, A_arr, β_g)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.collide_boussinesq_vt_2d!)
```
"""
function collide_boussinesq_vt_2d!(f, Temp, is_solid, ν_ref, T_ref, A_arr, β_g)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    ET = eltype(f)
    kernel! = collide_boussinesq_vt_2d_kernel!(backend)
    kernel!(f, Temp, is_solid, ET(ν_ref), ET(T_ref), ET(A_arr), ET(β_g); ndrange=(Nx, Ny))
end

"""
    collide_boussinesq_2d!(f, Temp, is_solid, ω, β_g, T_ref)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.collide_boussinesq_2d!)
```
"""
function collide_boussinesq_2d!(f, Temp, is_solid, ω, β_g, T_ref)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    T = eltype(f)
    kernel! = collide_boussinesq_2d_kernel!(backend)
    kernel!(f, Temp, is_solid, ω, T(β_g), T(T_ref); ndrange=(Nx, Ny))
end

# --- BGK collision with modified Arrhenius viscosity (linear exponent) ---

"""
    collide_boussinesq_vt_modified_2d!(f, Temp, is_solid, ν_ref, T0_visc, α_visc, β_g, T_ref_buoy)

BGK collision with per-node Boussinesq force AND temperature-dependent viscosity.
ν(T) = ν_ref · exp(α_visc · (T - T0_visc))  (modified Arrhenius / Frank-Kamenetskii)
ω(T) = 1 / (3·ν(T) + 0.5)
Buoyancy: Fy = β_g · (T - T_ref_buoy)

Used for rheological contrast studies: α_visc = ln(Rc), Rc = η_max/η_min.
"""
@kernel function collide_boussinesq_vt_modified_2d_kernel!(f, @Const(Temp), @Const(is_solid),
                                                            ν_ref, T0_visc, α_visc, β_g, T_ref_buoy)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j]
            bounce_back_2d!(f, i, j)
        else
            T = eltype(f)
            f1=f[i,j,1]; f2=f[i,j,2]; f3=f[i,j,3]; f4=f[i,j,4]
            f5=f[i,j,5]; f6=f[i,j,6]; f7=f[i,j,7]; f8=f[i,j,8]; f9=f[i,j,9]

            # Modified Arrhenius: ν(T) = ν_ref * exp(α * (T - T0_visc))
            T_local = Temp[i,j]
            ν_local = ν_ref * exp(α_visc * (T_local - T0_visc))
            ω_local = one(T) / (T(3) * ν_local + T(0.5))

            # Per-node buoyancy
            fy = β_g * (T_local - T_ref_buoy)
            fx = zero(T)

            ρ = f1+f2+f3+f4+f5+f6+f7+f8+f9
            inv_ρ = one(T) / ρ
            ux = ((f2-f4+f6-f7-f8+f9) + fx/T(2)) * inv_ρ
            uy = ((f3-f5+f6+f7-f8-f9) + fy/T(2)) * inv_ρ
            usq = ux*ux + uy*uy

            guo_pref = one(T) - ω_local / T(2)

            Sq=T(4.0/9.0)*((-ux)*fx+(-uy)*fy)*T(3)
            f[i,j,1]=f1-ω_local*(f1-feq_2d(Val(1), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/9.0)*((one(T)-ux)*fx+(-uy)*fy)*T(3)+T(1.0/9.0)*ux*fx*T(9)
            f[i,j,2]=f2-ω_local*(f2-feq_2d(Val(2), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/9.0)*((-ux)*fx+(one(T)-uy)*fy)*T(3)+T(1.0/9.0)*uy*fy*T(9)
            f[i,j,3]=f3-ω_local*(f3-feq_2d(Val(3), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/9.0)*((-one(T)-ux)*fx+(-uy)*fy)*T(3)+T(1.0/9.0)*ux*fx*T(9)
            f[i,j,4]=f4-ω_local*(f4-feq_2d(Val(4), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/9.0)*((-ux)*fx+(-one(T)-uy)*fy)*T(3)+T(1.0/9.0)*uy*fy*T(9)
            f[i,j,5]=f5-ω_local*(f5-feq_2d(Val(5), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/36.0)*((one(T)-ux)*fx+(one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(ux+uy)*(fx+fy)*T(9)
            f[i,j,6]=f6-ω_local*(f6-feq_2d(Val(6), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/36.0)*((-one(T)-ux)*fx+(one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(-ux+uy)*(-fx+fy)*T(9)
            f[i,j,7]=f7-ω_local*(f7-feq_2d(Val(7), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/36.0)*((-one(T)-ux)*fx+(-one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(-ux-uy)*(-fx-fy)*T(9)
            f[i,j,8]=f8-ω_local*(f8-feq_2d(Val(8), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/36.0)*((one(T)-ux)*fx+(-one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(ux-uy)*(fx-fy)*T(9)
            f[i,j,9]=f9-ω_local*(f9-feq_2d(Val(9), ρ, ux, uy, usq))+guo_pref*Sq
        end
    end
end

"""
    collide_boussinesq_vt_modified_2d!(f, Temp, is_solid, ν_ref, T0_visc, α_visc, β_g, T_ref_buoy)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.collide_boussinesq_vt_modified_2d!)
```
"""
function collide_boussinesq_vt_modified_2d!(f, Temp, is_solid, ν_ref, T0_visc, α_visc, β_g, T_ref_buoy)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    ET = eltype(f)
    kernel! = collide_boussinesq_vt_modified_2d_kernel!(backend)
    kernel!(f, Temp, is_solid, ET(ν_ref), ET(T0_visc), ET(α_visc), ET(β_g), ET(T_ref_buoy); ndrange=(Nx, Ny))
end
