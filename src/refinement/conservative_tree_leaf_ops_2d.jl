function collide_BGK_integrated_D2Q9!(Fcell::AbstractVector, volume, omega)
    _check_d2q9_vector(Fcell, "Fcell")
    volume > zero(volume) || throw(ArgumentError("volume must be positive"))

    m = mass_F(Fcell)
    iszero(m) && throw(ArgumentError("Fcell mass must be nonzero"))
    mx, my = momentum_F(Fcell)
    rho = m / volume
    ux = mx / m
    uy = my / m

    @inbounds for q in 1:9
        f = Fcell[q] / volume
        feq = equilibrium(D2Q9(), rho, ux, uy, q)
        Fcell[q] = (f - omega * (f - feq)) * volume
    end
    return Fcell
end

function collide_BGK_integrated_D2Q9!(F::AbstractArray{<:Any,3}, volume, omega)
    size(F, 3) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations in dimension 3"))
    @inbounds for j in axes(F, 2), i in axes(F, 1)
        collide_BGK_integrated_D2Q9!(@view(F[i, j, :]), volume, omega)
    end
    return F
end

"""
    collide_Guo_integrated_D2Q9!(Fcell::AbstractVector,

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.collide_Guo_integrated_D2Q9!)
```
"""
function collide_Guo_integrated_D2Q9!(Fcell::AbstractVector,
                                      volume,
                                      omega,
                                      Fx,
                                      Fy)
    _check_d2q9_vector(Fcell, "Fcell")
    volume > zero(volume) || throw(ArgumentError("volume must be positive"))

    m = mass_F(Fcell)
    iszero(m) && throw(ArgumentError("Fcell mass must be nonzero"))
    mx, my = momentum_F(Fcell)
    rho = m / volume
    ux = (mx / volume + Fx / 2) / rho
    uy = (my / volume + Fy / 2) / rho
    guo_pref = 1 - omega / 2

    @inbounds for q in 1:9
        cx = d2q9_cx(q)
        cy = d2q9_cy(q)
        w = weights(D2Q9())[q]
        ci_dot_u = cx * ux + cy * uy
        ci_dot_F = cx * Fx + cy * Fy
        Sq = w * (3 * ((cx - ux) * Fx + (cy - uy) * Fy) + 9 * ci_dot_u * ci_dot_F)
        f = Fcell[q] / volume
        feq = equilibrium(D2Q9(), rho, ux, uy, q)
        Fcell[q] = volume * (f - omega * (f - feq) + guo_pref * Sq)
    end
    return Fcell
end

function collide_Guo_integrated_D2Q9!(F::AbstractArray{<:Any,3},
                                      volume,
                                      omega,
                                      Fx,
                                      Fy)
    size(F, 3) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations in dimension 3"))
    @inbounds for j in axes(F, 2), i in axes(F, 1)
        collide_Guo_integrated_D2Q9!(@view(F[i, j, :]), volume, omega, Fx, Fy)
    end
    return F
end

function _check_d2q9_grid_pair(Fout::AbstractArray{<:Any,3},
                               Fin::AbstractArray{<:Any,3})
    size(Fout) == size(Fin) || throw(ArgumentError("Fout and Fin must have the same size"))
    size(Fin, 3) == 9 || throw(ArgumentError("Fin must have 9 D2Q9 populations in dimension 3"))
    return nothing
end

"""
    stream_fully_periodic_F_2d!(Fout, Fin)

Pure pull streaming for a leaf grid stored as integrated D2Q9 populations, with
periodic boundaries in both directions.
"""
function stream_fully_periodic_F_2d!(Fout::AbstractArray{<:Any,3},
                                     Fin::AbstractArray{<:Any,3})
    Fout === Fin && throw(ArgumentError("Fout and Fin must be distinct arrays"))
    _check_d2q9_grid_pair(Fout, Fin)

    nx = size(Fin, 1)
    ny = size(Fin, 2)
    @inbounds for q in 1:9
        cx = d2q9_cx(q)
        cy = d2q9_cy(q)
        for j in 1:ny, i in 1:nx
            isrc = mod1(i - cx, nx)
            jsrc = mod1(j - cy, ny)
            Fout[i, j, q] = Fin[isrc, jsrc, q]
        end
    end
    return Fout
end

"""
    stream_periodic_x_wall_y_F_2d!(Fout, Fin)

Pure pull streaming for a leaf grid stored as integrated D2Q9 populations, with
periodic `x` and stationary bounce-back walls at the south/north `y`
boundaries. This is a leaf-grid boundary condition; inactive parent ledgers
must not be streamed through this operator.
"""
function stream_periodic_x_wall_y_F_2d!(Fout::AbstractArray{<:Any,3},
                                        Fin::AbstractArray{<:Any,3})
    Fout === Fin && throw(ArgumentError("Fout and Fin must be distinct arrays"))
    _check_d2q9_grid_pair(Fout, Fin)

    nx = size(Fin, 1)
    ny = size(Fin, 2)
    @inbounds for j in 1:ny, i in 1:nx
        im = i > 1 ? i - 1 : nx
        ip = i < nx ? i + 1 : 1

        Fout[i, j, 1] = Fin[i, j, 1]
        Fout[i, j, 2] = Fin[im, j, 2]
        Fout[i, j, 4] = Fin[ip, j, 4]

        if j == 1
            Fout[i, j, 3] = Fin[i, j, 5]
            Fout[i, j, 5] = ny == 1 ? Fin[i, j, 3] : Fin[i, j+1, 5]
            Fout[i, j, 6] = Fin[i, j, 8]
            Fout[i, j, 7] = Fin[i, j, 9]
            Fout[i, j, 8] = ny == 1 ? Fin[i, j, 6] : Fin[ip, j+1, 8]
            Fout[i, j, 9] = ny == 1 ? Fin[i, j, 7] : Fin[im, j+1, 9]
        elseif j == ny
            Fout[i, j, 3] = Fin[i, j-1, 3]
            Fout[i, j, 5] = Fin[i, j, 3]
            Fout[i, j, 6] = Fin[im, j-1, 6]
            Fout[i, j, 7] = Fin[ip, j-1, 7]
            Fout[i, j, 8] = Fin[i, j, 6]
            Fout[i, j, 9] = Fin[i, j, 7]
        else
            Fout[i, j, 3] = Fin[i, j-1, 3]
            Fout[i, j, 5] = Fin[i, j+1, 5]
            Fout[i, j, 6] = Fin[im, j-1, 6]
            Fout[i, j, 7] = Fin[ip, j-1, 7]
            Fout[i, j, 8] = Fin[ip, j+1, 8]
            Fout[i, j, 9] = Fin[im, j+1, 9]
        end
    end
    return Fout
end

@inline function _moving_wall_delta(volume, rho_wall, wall_u, q::Int)
    return volume * 2 * (1 / 36) * rho_wall * d2q9_cx(q) * wall_u / (1 / 3)
end

"""
    stream_periodic_x_moving_wall_y_F_2d!(Fout, Fin; u_south=0, u_north=0,
                                          rho_wall=1, volume=1)

Pure pull streaming for a leaf grid with periodic `x` and bounce-back walls at
the south/north `y` boundaries. Tangential wall velocities are included through
the standard moving-wall bounce-back correction on diagonal populations.
"""
function stream_periodic_x_moving_wall_y_F_2d!(Fout::AbstractArray{<:Any,3},
                                               Fin::AbstractArray{<:Any,3};
                                               u_south=0,
                                               u_north=0,
                                               rho_wall=1,
                                               volume=1)
    Fout === Fin && throw(ArgumentError("Fout and Fin must be distinct arrays"))
    _check_d2q9_grid_pair(Fout, Fin)

    nx = size(Fin, 1)
    ny = size(Fin, 2)
    @inbounds for j in 1:ny, i in 1:nx
        im = i > 1 ? i - 1 : nx
        ip = i < nx ? i + 1 : 1

        Fout[i, j, 1] = Fin[i, j, 1]
        Fout[i, j, 2] = Fin[im, j, 2]
        Fout[i, j, 4] = Fin[ip, j, 4]

        if j == 1
            Fout[i, j, 3] = Fin[i, j, 5]
            Fout[i, j, 5] = ny == 1 ? Fin[i, j, 3] : Fin[i, j+1, 5]
            Fout[i, j, 6] = Fin[i, j, 8] + _moving_wall_delta(volume, rho_wall, u_south, 6)
            Fout[i, j, 7] = Fin[i, j, 9] + _moving_wall_delta(volume, rho_wall, u_south, 7)
            Fout[i, j, 8] = ny == 1 ? Fin[i, j, 6] : Fin[ip, j+1, 8]
            Fout[i, j, 9] = ny == 1 ? Fin[i, j, 7] : Fin[im, j+1, 9]
        elseif j == ny
            Fout[i, j, 3] = Fin[i, j-1, 3]
            Fout[i, j, 5] = Fin[i, j, 3]
            Fout[i, j, 6] = Fin[im, j-1, 6]
            Fout[i, j, 7] = Fin[ip, j-1, 7]
            Fout[i, j, 8] = Fin[i, j, 6] + _moving_wall_delta(volume, rho_wall, u_north, 8)
            Fout[i, j, 9] = Fin[i, j, 7] + _moving_wall_delta(volume, rho_wall, u_north, 9)
        else
            Fout[i, j, 3] = Fin[i, j-1, 3]
            Fout[i, j, 5] = Fin[i, j+1, 5]
            Fout[i, j, 6] = Fin[im, j-1, 6]
            Fout[i, j, 7] = Fin[ip, j-1, 7]
            Fout[i, j, 8] = Fin[ip, j+1, 8]
            Fout[i, j, 9] = Fin[im, j+1, 9]
        end
    end
    return Fout
end

"""
    cylinder_solid_mask_leaf_2d(Nx::Int, Ny::Int, cx, cy, radius)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.cylinder_solid_mask_leaf_2d)
```
"""
function cylinder_solid_mask_leaf_2d(Nx::Int, Ny::Int, cx, cy, radius)
    mask = falses(Nx, Ny)
    r2 = radius * radius
    @inbounds for j in 1:Ny, i in 1:Nx
        mask[i, j] = (i - cx)^2 + (j - cy)^2 <= r2
    end
    return mask
end

"""
    square_solid_mask_leaf_2d(Nx::Int, Ny::Int,

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.square_solid_mask_leaf_2d)
```
"""
function square_solid_mask_leaf_2d(Nx::Int, Ny::Int,
                                   i_range::AbstractUnitRange{<:Integer},
                                   j_range::AbstractUnitRange{<:Integer})
    isempty(i_range) && throw(ArgumentError("i_range must be nonempty"))
    isempty(j_range) && throw(ArgumentError("j_range must be nonempty"))
    first(i_range) >= 1 && last(i_range) <= Nx ||
        throw(ArgumentError("i_range must be inside 1:Nx"))
    first(j_range) >= 1 && last(j_range) <= Ny ||
        throw(ArgumentError("j_range must be inside 1:Ny"))

    mask = falses(Nx, Ny)
    @inbounds for j in Int(first(j_range)):Int(last(j_range)),
                  i in Int(first(i_range)):Int(last(i_range))
        mask[i, j] = true
    end
    return mask
end

"""
    backward_facing_step_solid_mask_leaf_2d(Nx::Int, Ny::Int,

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.backward_facing_step_solid_mask_leaf_2d)
```
"""
function backward_facing_step_solid_mask_leaf_2d(Nx::Int, Ny::Int,
                                                 step_i::Int,
                                                 step_height::Int)
    1 <= step_i < Nx || throw(ArgumentError("step_i must be inside 1:Nx-1"))
    1 <= step_height < Ny - 1 ||
        throw(ArgumentError("step_height must leave at least two open rows"))

    mask = falses(Nx, Ny)
    @inbounds for j in 1:step_height, i in 1:step_i
        mask[i, j] = true
    end
    return mask
end

function _check_solid_mask_layout(F::AbstractArray{<:Any,3},
                                  is_solid::AbstractArray{Bool,2})
    size(is_solid) == (size(F, 1), size(F, 2)) ||
        throw(ArgumentError("is_solid must match the first two dimensions of F"))
    return nothing
end

"""
    stream_periodic_x_wall_y_solid_F_2d!(Fout, Fin, is_solid)

Pull streaming on a leaf grid with periodic x, channel walls in y, and
halfway bounce-back on solid links. Solid destination cells are filled with
their previous values and are ignored by the solid-aware collision routines.
"""
function stream_periodic_x_wall_y_solid_F_2d!(Fout::AbstractArray{<:Any,3},
                                              Fin::AbstractArray{<:Any,3},
                                              is_solid::AbstractArray{Bool,2})
    Fout === Fin && throw(ArgumentError("Fout and Fin must be distinct arrays"))
    _check_d2q9_grid_pair(Fout, Fin)
    _check_solid_mask_layout(Fin, is_solid)

    nx = size(Fin, 1)
    ny = size(Fin, 2)
    @inbounds for j in 1:ny, i in 1:nx
        if is_solid[i, j]
            for q in 1:9
                Fout[i, j, q] = Fin[i, j, q]
            end
            continue
        end

        for q in 1:9
            cx = d2q9_cx(q)
            cy = d2q9_cy(q)
            if cx == 0 && cy == 0
                Fout[i, j, q] = Fin[i, j, q]
                continue
            end

            isrc = mod1(i - cx, nx)
            jsrc = j - cy
            if jsrc < 1 || jsrc > ny
                Fout[i, j, q] = Fin[i, j, d2q9_opposite(q)]
            elseif is_solid[isrc, jsrc]
                Fout[i, j, q] = Fin[i, j, d2q9_opposite(q)]
            else
                Fout[i, j, q] = Fin[isrc, jsrc, q]
            end
        end
    end
    return Fout
end

"""
    stream_bounceback_xy_solid_F_2d!(Fout::AbstractArray{<:Any,3},

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.stream_bounceback_xy_solid_F_2d!)
```
"""
function stream_bounceback_xy_solid_F_2d!(Fout::AbstractArray{<:Any,3},
                                          Fin::AbstractArray{<:Any,3},
                                          is_solid::AbstractArray{Bool,2})
    Fout === Fin && throw(ArgumentError("Fout and Fin must be distinct arrays"))
    _check_d2q9_grid_pair(Fout, Fin)
    _check_solid_mask_layout(Fin, is_solid)

    nx = size(Fin, 1)
    ny = size(Fin, 2)
    @inbounds for j in 1:ny, i in 1:nx
        if is_solid[i, j]
            for q in 1:9
                Fout[i, j, q] = Fin[i, j, q]
            end
            continue
        end

        for q in 1:9
            cx = d2q9_cx(q)
            cy = d2q9_cy(q)
            if cx == 0 && cy == 0
                Fout[i, j, q] = Fin[i, j, q]
                continue
            end

            isrc = i - cx
            jsrc = j - cy
            if isrc < 1 || isrc > nx || jsrc < 1 || jsrc > ny
                Fout[i, j, q] = Fin[i, j, d2q9_opposite(q)]
            elseif is_solid[isrc, jsrc]
                Fout[i, j, q] = Fin[i, j, d2q9_opposite(q)]
            else
                Fout[i, j, q] = Fin[isrc, jsrc, q]
            end
        end
    end
    return Fout
end

"""
    apply_zou_he_west_cell_F_2d!(Fcell::AbstractVector{T}, u_in, volume) where T

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.apply_zou_he_west_cell_F_2d!)
```
"""
function apply_zou_he_west_cell_F_2d!(Fcell::AbstractVector{T}, u_in, volume) where T
    _check_d2q9_vector(Fcell, "Fcell")
    u = T(u_in)
    vol = T(volume)
    f1 = Fcell[1] / vol
    f3 = Fcell[3] / vol
    f4 = Fcell[4] / vol
    f5 = Fcell[5] / vol
    f7 = Fcell[7] / vol
    f8 = Fcell[8] / vol
    rho_wall = (f1 + f3 + f5 + T(2) * (f4 + f7 + f8)) / (one(T) - u)
    Fcell[2] = vol * (f4 + T(2//3) * rho_wall * u)
    Fcell[6] = vol * (f8 - T(0.5) * (f3 - f5) + T(1//6) * rho_wall * u)
    Fcell[9] = vol * (f7 + T(0.5) * (f3 - f5) + T(1//6) * rho_wall * u)
    return Fcell
end

"""
    apply_zou_he_west_F_2d!(F::AbstractArray{T,3}, u_in, volume) where T

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.apply_zou_he_west_F_2d!)
```
"""
function apply_zou_he_west_F_2d!(F::AbstractArray{T,3}, u_in, volume) where T
    size(F, 3) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations in dimension 3"))
    @inbounds for j in axes(F, 2)
        apply_zou_he_west_cell_F_2d!(@view(F[1, j, :]), u_in, volume)
    end
    return F
end

function apply_zou_he_west_F_2d!(F::AbstractArray{T,3}, u_in, volume,
                                 is_solid::AbstractArray{Bool,2}) where T
    size(F, 3) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations in dimension 3"))
    _check_solid_mask_layout(F, is_solid)
    @inbounds for j in axes(F, 2)
        is_solid[1, j] && continue
        apply_zou_he_west_cell_F_2d!(@view(F[1, j, :]), u_in, volume)
    end
    return F
end

"""
    apply_zou_he_pressure_east_cell_F_2d!(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.apply_zou_he_pressure_east_cell_F_2d!)
```
"""
function apply_zou_he_pressure_east_cell_F_2d!(
        Fcell::AbstractVector{T},
        volume;
        rho_out=one(T)) where T
    _check_d2q9_vector(Fcell, "Fcell")
    vol = T(volume)
    rho = T(rho_out)
    f1 = Fcell[1] / vol
    f2 = Fcell[2] / vol
    f3 = Fcell[3] / vol
    f5 = Fcell[5] / vol
    f6 = Fcell[6] / vol
    f9 = Fcell[9] / vol
    ux = -one(T) + (f1 + f3 + f5 + T(2) * (f2 + f6 + f9)) / rho
    Fcell[4] = vol * (f2 - T(2//3) * rho * ux)
    Fcell[7] = vol * (f9 - T(0.5) * (f3 - f5) - T(1//6) * rho * ux)
    Fcell[8] = vol * (f6 + T(0.5) * (f3 - f5) - T(1//6) * rho * ux)
    return Fcell
end

"""
    apply_zou_he_pressure_east_F_2d!(F::AbstractArray{T,3}, volume;

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.apply_zou_he_pressure_east_F_2d!)
```
"""
function apply_zou_he_pressure_east_F_2d!(F::AbstractArray{T,3}, volume;
                                          rho_out=one(T)) where T
    size(F, 3) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations in dimension 3"))
    i = last(axes(F, 1))
    @inbounds for j in axes(F, 2)
        apply_zou_he_pressure_east_cell_F_2d!(
            @view(F[i, j, :]), volume; rho_out=rho_out)
    end
    return F
end

function apply_zou_he_pressure_east_F_2d!(F::AbstractArray{T,3}, volume,
                                          is_solid::AbstractArray{Bool,2};
                                          rho_out=one(T)) where T
    size(F, 3) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations in dimension 3"))
    _check_solid_mask_layout(F, is_solid)
    i = last(axes(F, 1))
    @inbounds for j in axes(F, 2)
        is_solid[i, j] && continue
        apply_zou_he_pressure_east_cell_F_2d!(
            @view(F[i, j, :]), volume; rho_out=rho_out)
    end
    return F
end

function collide_BGK_integrated_D2Q9!(F::AbstractArray{<:Any,3},
                                      is_solid::AbstractArray{Bool,2},
                                      volume,
                                      omega)
    size(F, 3) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations in dimension 3"))
    _check_solid_mask_layout(F, is_solid)
    @inbounds for j in axes(F, 2), i in axes(F, 1)
        is_solid[i, j] && continue
        collide_BGK_integrated_D2Q9!(@view(F[i, j, :]), volume, omega)
    end
    return F
end

function collide_Guo_integrated_D2Q9!(F::AbstractArray{<:Any,3},
                                      is_solid::AbstractArray{Bool,2},
                                      volume,
                                      omega,
                                      Fx,
                                      Fy)
    size(F, 3) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations in dimension 3"))
    _check_solid_mask_layout(F, is_solid)
    @inbounds for j in axes(F, 2), i in axes(F, 1)
        is_solid[i, j] && continue
        collide_Guo_integrated_D2Q9!(@view(F[i, j, :]), volume, omega, Fx, Fy)
    end
    return F
end

"""
    compute_drag_mea_solid_F_2d(Fpre::AbstractArray{<:Any,3},

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.compute_drag_mea_solid_F_2d)
```
"""
function compute_drag_mea_solid_F_2d(Fpre::AbstractArray{<:Any,3},
                                     Fpost::AbstractArray{<:Any,3},
                                     is_solid::AbstractArray{Bool,2})
    _check_d2q9_grid_pair(Fpost, Fpre)
    _check_solid_mask_layout(Fpre, is_solid)

    Fx = 0.0
    Fy = 0.0
    nx = size(Fpre, 1)
    ny = size(Fpre, 2)
    @inbounds for j in 1:ny, i in 1:nx
        is_solid[i, j] && continue
        for q in 2:9
            ni = mod1(i + d2q9_cx(q), nx)
            nj = j + d2q9_cy(q)
            1 <= nj <= ny || continue
            is_solid[ni, nj] || continue
            oq = d2q9_opposite(q)
            Fx += d2q9_cx(q) * (Fpre[i, j, q] + Fpost[i, j, oq])
            Fy += d2q9_cy(q) * (Fpre[i, j, q] + Fpost[i, j, oq])
        end
    end
    return (Fx=Fx, Fy=Fy)
end

"""
    ConservativeTreePatch2D

Experimental fixed ratio-2 patch for the conservative tree route.

`fine_F` is the active fine state inside the refined region. `coarse_shadow_F`
is only a parent ledger/agregate over the same physical region; it is not an
active fluid state.
"""
