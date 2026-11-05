@inline function _fvfd_bc_index_1d(idx, n, lower_bc, upper_bc)
    if 1 <= idx <= n
        return idx
    elseif idx < 1 && lower_bc == FVFD_BC_PERIODIC
        return idx + n
    elseif idx > n && upper_bc == FVFD_BC_PERIODIC
        return idx - n
    else
        return 0
    end
end

@inline function _fvfd_solid_bc_derivative_x_2d(
    field, is_solid, i, j, Nx, inv_dx, inv_2dx, west_bc, east_bc,
    polymer_wall_extrap::Val=Val(:quadratic),
)
    T = eltype(field)
    li = _fvfd_bc_index_1d(i - 1, Nx, west_bc, east_bc)
    ri = _fvfd_bc_index_1d(i + 1, Nx, west_bc, east_bc)
    left = li != 0 && !is_solid[li, j]
    right = ri != 0 && !is_solid[ri, j]
    if left && right
        return (field[ri, j] - field[li, j]) * inv_2dx
    elseif right
        if polymer_wall_extrap isa Val{:linear}
            return (field[ri, j] - field[i, j]) * inv_dx
        end
        r2i = _fvfd_bc_index_1d(i + 2, Nx, west_bc, east_bc)
        return (r2i != 0 && !is_solid[r2i, j]) ?
               (-T(3) * field[i, j] + T(4) * field[ri, j] - field[r2i, j]) * inv_2dx :
               (field[ri, j] - field[i, j]) * inv_dx
    elseif left
        if polymer_wall_extrap isa Val{:linear}
            return (field[i, j] - field[li, j]) * inv_dx
        end
        l2i = _fvfd_bc_index_1d(i - 2, Nx, west_bc, east_bc)
        return (l2i != 0 && !is_solid[l2i, j]) ?
               (T(3) * field[i, j] - T(4) * field[li, j] + field[l2i, j]) * inv_2dx :
               (field[i, j] - field[li, j]) * inv_dx
    else
        return zero(T)
    end
end

@inline function _fvfd_solid_bc_derivative_y_2d(
    field, is_solid, i, j, Ny, inv_dy, inv_2dy, south_bc, north_bc,
    polymer_wall_extrap::Val=Val(:quadratic),
)
    T = eltype(field)
    dj = _fvfd_bc_index_1d(j - 1, Ny, south_bc, north_bc)
    uj = _fvfd_bc_index_1d(j + 1, Ny, south_bc, north_bc)
    down = dj != 0 && !is_solid[i, dj]
    up = uj != 0 && !is_solid[i, uj]
    if down && up
        return (field[i, uj] - field[i, dj]) * inv_2dy
    elseif up
        if polymer_wall_extrap isa Val{:linear}
            return (field[i, uj] - field[i, j]) * inv_dy
        end
        u2j = _fvfd_bc_index_1d(j + 2, Ny, south_bc, north_bc)
        return (u2j != 0 && !is_solid[i, u2j]) ?
               (-T(3) * field[i, j] + T(4) * field[i, uj] - field[i, u2j]) * inv_2dy :
               (field[i, uj] - field[i, j]) * inv_dy
    elseif down
        if polymer_wall_extrap isa Val{:linear}
            return (field[i, j] - field[i, dj]) * inv_dy
        end
        d2j = _fvfd_bc_index_1d(j - 2, Ny, south_bc, north_bc)
        return (d2j != 0 && !is_solid[i, d2j]) ?
               (T(3) * field[i, j] - T(4) * field[i, dj] + field[i, d2j]) * inv_2dy :
               (field[i, j] - field[i, dj]) * inv_dy
    else
        return zero(T)
    end
end

"""
    _fvfd_solid_bc_second_derivative_x_2d(field, is_solid, i, j, Nx, inv_dx2, west_bc, east_bc)

x second derivative used by the both-sides-diffusion correction. Next to a solid
cell the one-sided stencil is kept. At the first or last column of an
`FVFD_BC_OPEN` end the result is zero, i.e. it assumes `∂²u/∂x² = 0` there: the
one-sided stencil has the wrong sign on the period-2 mode of the outlet corner
cells and anti-damps it when `nu_s` is small against `nu_p`.
"""
@inline function _fvfd_solid_bc_second_derivative_x_2d(
    field, is_solid, i, j, Nx, inv_dx2, west_bc, east_bc,
)
    T = eltype(field)
    li = _fvfd_bc_index_1d(i - 1, Nx, west_bc, east_bc)
    ri = _fvfd_bc_index_1d(i + 1, Nx, west_bc, east_bc)
    left = li != 0 && !is_solid[li, j]
    right = ri != 0 && !is_solid[ri, j]
    open_end = (li == 0 && i == 1 && west_bc == FVFD_BC_OPEN) ||
               (ri == 0 && i == Nx && east_bc == FVFD_BC_OPEN)
    if left && right
        return (field[ri, j] - T(2) * field[i, j] + field[li, j]) * inv_dx2
    elseif open_end
        return zero(T)
    elseif right
        r2i = _fvfd_bc_index_1d(i + 2, Nx, west_bc, east_bc)
        return (r2i != 0 && !is_solid[r2i, j]) ?
               (field[i, j] - T(2) * field[ri, j] + field[r2i, j]) * inv_dx2 :
               zero(T)
    elseif left
        l2i = _fvfd_bc_index_1d(i - 2, Nx, west_bc, east_bc)
        return (l2i != 0 && !is_solid[l2i, j]) ?
               (field[i, j] - T(2) * field[li, j] + field[l2i, j]) * inv_dx2 :
               zero(T)
    else
        return zero(T)
    end
end

@inline function _fvfd_solid_bc_second_derivative_y_2d(
    field, is_solid, i, j, Ny, inv_dy2, south_bc, north_bc,
)
    T = eltype(field)
    dj = _fvfd_bc_index_1d(j - 1, Ny, south_bc, north_bc)
    uj = _fvfd_bc_index_1d(j + 1, Ny, south_bc, north_bc)
    down = dj != 0 && !is_solid[i, dj]
    up = uj != 0 && !is_solid[i, uj]
    if down && up
        return (field[i, uj] - T(2) * field[i, j] + field[i, dj]) * inv_dy2
    elseif up
        u2j = _fvfd_bc_index_1d(j + 2, Ny, south_bc, north_bc)
        return (u2j != 0 && !is_solid[i, u2j]) ?
               (field[i, j] - T(2) * field[i, uj] + field[i, u2j]) * inv_dy2 :
               zero(T)
    elseif down
        d2j = _fvfd_bc_index_1d(j - 2, Ny, south_bc, north_bc)
        return (d2j != 0 && !is_solid[i, d2j]) ?
               (field[i, j] - T(2) * field[i, dj] + field[i, d2j]) * inv_dy2 :
               zero(T)
    else
        return zero(T)
    end
end

@inline function _fvfd_apply_embedded_wall_gradient_2d(
    gx, gy, phi, wall_nx, wall_ny, wall_inv_distance_to_center, i, j,
)
    inv_distance = wall_inv_distance_to_center[i, j]
    if inv_distance > zero(inv_distance)
        nx = wall_nx[i, j]
        ny = wall_ny[i, j]
        target_normal_derivative = phi[i, j] * inv_distance
        current_normal_derivative = gx * nx + gy * ny
        correction = target_normal_derivative - current_normal_derivative
        return gx + correction * nx, gy + correction * ny
    end
    return gx, gy
end

@inline function _fvfd_xface_average_or_zero_2d(ux, is_solid, i_left, i_right, j)
    T = eltype(ux)
    return (is_solid[i_left, j] || is_solid[i_right, j]) ?
           zero(T) :
           (ux[i_left, j] + ux[i_right, j]) / T(2)
end

@inline function _fvfd_xface_fraction_2d(
    is_solid, west_fraction, east_fraction, i_left, i_right, j,
)
    T = eltype(west_fraction)
    return (is_solid[i_left, j] || is_solid[i_right, j]) ?
           zero(T) :
           min(east_fraction[i_left, j], west_fraction[i_right, j])
end

@inline function _fvfd_yface_average_or_zero_2d(uy, is_solid, i, j_down, j_up)
    T = eltype(uy)
    return (is_solid[i, j_down] || is_solid[i, j_up]) ?
           zero(T) :
           (uy[i, j_down] + uy[i, j_up]) / T(2)
end

@inline function _fvfd_yface_fraction_2d(
    is_solid, south_fraction, north_fraction, i, j_down, j_up,
)
    T = eltype(south_fraction)
    return (is_solid[i, j_down] || is_solid[i, j_up]) ?
           zero(T) :
           min(north_fraction[i, j_down], south_fraction[i, j_up])
end

@inline function _fvfd_xface_scalar_average_or_zero_2d(field, is_solid, i_left, i_right, j)
    T = eltype(field)
    return (is_solid[i_left, j] || is_solid[i_right, j]) ?
           zero(T) :
           (field[i_left, j] + field[i_right, j]) / T(2)
end

@inline function _fvfd_yface_scalar_average_or_zero_2d(field, is_solid, i, j_down, j_up)
    T = eltype(field)
    return (is_solid[i, j_down] || is_solid[i, j_up]) ?
           zero(T) :
           (field[i, j_down] + field[i, j_up]) / T(2)
end

@kernel function fvfd_cell_velocity_to_faces_2d_kernel!(
    ux_face, uy_face,
    @Const(ux), @Const(uy), @Const(is_solid),
    @Const(ux_west), @Const(ux_east),
    @Const(uy_south), @Const(uy_north),
    west_bc, east_bc, south_bc, north_bc, Nx, Ny,
)
    I, J = @index(Global, NTuple)
    @inbounds begin
        if I <= Nx + 1 && J <= Ny
            if I == 1
                if west_bc == FVFD_BC_PERIODIC
                    ux_face[I, J] = _fvfd_xface_average_or_zero_2d(ux, is_solid, Nx, 1, J)
                elseif west_bc == FVFD_BC_OPEN
                    ux_face[I, J] = is_solid[1, J] ? zero(eltype(ux_face)) : ux_west[J]
                else
                    ux_face[I, J] = zero(eltype(ux_face))
                end
            elseif I == Nx + 1
                if east_bc == FVFD_BC_PERIODIC
                    ux_face[I, J] = _fvfd_xface_average_or_zero_2d(ux, is_solid, Nx, 1, J)
                elseif east_bc == FVFD_BC_OPEN
                    ux_face[I, J] = is_solid[Nx, J] ? zero(eltype(ux_face)) : ux_east[J]
                else
                    ux_face[I, J] = zero(eltype(ux_face))
                end
            else
                ux_face[I, J] = _fvfd_xface_average_or_zero_2d(ux, is_solid, I - 1, I, J)
            end
        end

        if I <= Nx && J <= Ny + 1
            if J == 1
                if south_bc == FVFD_BC_PERIODIC
                    uy_face[I, J] = _fvfd_yface_average_or_zero_2d(uy, is_solid, I, Ny, 1)
                elseif south_bc == FVFD_BC_OPEN
                    uy_face[I, J] = is_solid[I, 1] ? zero(eltype(uy_face)) : uy_south[I]
                else
                    uy_face[I, J] = zero(eltype(uy_face))
                end
            elseif J == Ny + 1
                if north_bc == FVFD_BC_PERIODIC
                    uy_face[I, J] = _fvfd_yface_average_or_zero_2d(uy, is_solid, I, Ny, 1)
                elseif north_bc == FVFD_BC_OPEN
                    uy_face[I, J] = is_solid[I, Ny] ? zero(eltype(uy_face)) : uy_north[I]
                else
                    uy_face[I, J] = zero(eltype(uy_face))
                end
            else
                uy_face[I, J] = _fvfd_yface_average_or_zero_2d(uy, is_solid, I, J - 1, J)
            end
        end
    end
end

@kernel function fvfd_cell_velocity_to_faces_embedded_2d_kernel!(
    ux_face, uy_face,
    @Const(ux), @Const(uy), @Const(is_solid),
    @Const(west_fraction), @Const(east_fraction),
    @Const(south_fraction), @Const(north_fraction),
    @Const(ux_west), @Const(ux_east),
    @Const(uy_south), @Const(uy_north),
    west_bc, east_bc, south_bc, north_bc, Nx, Ny,
)
    I, J = @index(Global, NTuple)
    @inbounds begin
        if I <= Nx + 1 && J <= Ny
            if I == 1
                if west_bc == FVFD_BC_PERIODIC
                    frac = _fvfd_xface_fraction_2d(
                        is_solid, west_fraction, east_fraction, Nx, 1, J,
                    )
                    ux_face[I, J] = frac * _fvfd_xface_average_or_zero_2d(ux, is_solid, Nx, 1, J)
                elseif west_bc == FVFD_BC_OPEN
                    ux_face[I, J] = is_solid[1, J] ? zero(eltype(ux_face)) :
                                    west_fraction[1, J] * ux_west[J]
                else
                    ux_face[I, J] = zero(eltype(ux_face))
                end
            elseif I == Nx + 1
                if east_bc == FVFD_BC_PERIODIC
                    frac = _fvfd_xface_fraction_2d(
                        is_solid, west_fraction, east_fraction, Nx, 1, J,
                    )
                    ux_face[I, J] = frac * _fvfd_xface_average_or_zero_2d(ux, is_solid, Nx, 1, J)
                elseif east_bc == FVFD_BC_OPEN
                    ux_face[I, J] = is_solid[Nx, J] ? zero(eltype(ux_face)) :
                                    east_fraction[Nx, J] * ux_east[J]
                else
                    ux_face[I, J] = zero(eltype(ux_face))
                end
            else
                frac = _fvfd_xface_fraction_2d(
                    is_solid, west_fraction, east_fraction, I - 1, I, J,
                )
                ux_face[I, J] = frac * _fvfd_xface_average_or_zero_2d(
                    ux, is_solid, I - 1, I, J,
                )
            end
        end

        if I <= Nx && J <= Ny + 1
            if J == 1
                if south_bc == FVFD_BC_PERIODIC
                    frac = _fvfd_yface_fraction_2d(
                        is_solid, south_fraction, north_fraction, I, Ny, 1,
                    )
                    uy_face[I, J] = frac * _fvfd_yface_average_or_zero_2d(uy, is_solid, I, Ny, 1)
                elseif south_bc == FVFD_BC_OPEN
                    uy_face[I, J] = is_solid[I, 1] ? zero(eltype(uy_face)) :
                                    south_fraction[I, 1] * uy_south[I]
                else
                    uy_face[I, J] = zero(eltype(uy_face))
                end
            elseif J == Ny + 1
                if north_bc == FVFD_BC_PERIODIC
                    frac = _fvfd_yface_fraction_2d(
                        is_solid, south_fraction, north_fraction, I, Ny, 1,
                    )
                    uy_face[I, J] = frac * _fvfd_yface_average_or_zero_2d(uy, is_solid, I, Ny, 1)
                elseif north_bc == FVFD_BC_OPEN
                    uy_face[I, J] = is_solid[I, Ny] ? zero(eltype(uy_face)) :
                                    north_fraction[I, Ny] * uy_north[I]
                else
                    uy_face[I, J] = zero(eltype(uy_face))
                end
            else
                frac = _fvfd_yface_fraction_2d(
                    is_solid, south_fraction, north_fraction, I, J - 1, J,
                )
                uy_face[I, J] = frac * _fvfd_yface_average_or_zero_2d(
                    uy, is_solid, I, J - 1, J,
                )
            end
        end
    end
end

function fvfd_cell_velocity_to_faces_2d!(
    ux_face, uy_face, ux, uy, is_solid,
    ux_west, ux_east, uy_south, uy_north,
    bc::FVFDDomainBC2D;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(ux_face)
    Nx, Ny = size(ux)
    bc.west == FVFD_BC_OPEN &&
        _fvfd_check_boundary_length(:ux_west, ux_west, Ny)
    bc.east == FVFD_BC_OPEN &&
        _fvfd_check_boundary_length(:ux_east, ux_east, Ny)
    bc.south == FVFD_BC_OPEN &&
        _fvfd_check_boundary_length(:uy_south, uy_south, Nx)
    bc.north == FVFD_BC_OPEN &&
        _fvfd_check_boundary_length(:uy_north, uy_north, Nx)
    kernel! = fvfd_cell_velocity_to_faces_2d_kernel!(backend)
    kernel!(
        ux_face, uy_face, ux, uy, is_solid,
        ux_west, ux_east, uy_south, uy_north,
        bc.west, bc.east, bc.south, bc.north, Nx, Ny;
        ndrange=(Nx + 1, Ny + 1),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function fvfd_cell_velocity_to_faces_embedded_2d!(
    ux_face, uy_face, ux, uy, is_solid,
    embedded::FVFDEmbeddedBoundary2D,
    ux_west, ux_east, uy_south, uy_north,
    bc::FVFDDomainBC2D;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(ux_face)
    Nx, Ny = size(ux)
    bc.west == FVFD_BC_OPEN &&
        _fvfd_check_boundary_length(:ux_west, ux_west, Ny)
    bc.east == FVFD_BC_OPEN &&
        _fvfd_check_boundary_length(:ux_east, ux_east, Ny)
    bc.south == FVFD_BC_OPEN &&
        _fvfd_check_boundary_length(:uy_south, uy_south, Nx)
    bc.north == FVFD_BC_OPEN &&
        _fvfd_check_boundary_length(:uy_north, uy_north, Nx)
    kernel! = fvfd_cell_velocity_to_faces_embedded_2d_kernel!(backend)
    kernel!(
        ux_face, uy_face, ux, uy, is_solid,
        embedded.west_fraction, embedded.east_fraction,
        embedded.south_fraction, embedded.north_fraction,
        ux_west, ux_east, uy_south, uy_north,
        bc.west, bc.east, bc.south, bc.north, Nx, Ny;
        ndrange=(Nx + 1, Ny + 1),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function fvfd_cell_velocity_to_faces_embedded_2d!(
    ux_face, uy_face, ux, uy,
    geometry::FVFDGeometry2D,
    ux_bc::FVFDFieldBC2D, uy_bc::FVFDFieldBC2D;
    sync::Bool=true,
)
    return fvfd_cell_velocity_to_faces_embedded_2d!(
        ux_face, uy_face, ux, uy,
        geometry.is_solid, geometry.embedded,
        ux_bc.west, ux_bc.east, uy_bc.south, uy_bc.north,
        geometry.bc; sync,
    )
end

function fvfd_cell_velocity_to_faces_2d!(
    ux_face, uy_face, ux, uy, is_solid,
    ux_bc::FVFDFieldBC2D, uy_bc::FVFDFieldBC2D,
    bc::FVFDDomainBC2D;
    sync::Bool=true,
)
    return fvfd_cell_velocity_to_faces_2d!(
        ux_face, uy_face, ux, uy, is_solid,
        ux_bc.west, ux_bc.east, uy_bc.south, uy_bc.north,
        bc; sync,
    )
end

function fvfd_cell_velocity_to_faces_2d!(
    ux_face, uy_face, ux, uy,
    geometry::FVFDGeometry2D,
    ux_bc::FVFDFieldBC2D, uy_bc::FVFDFieldBC2D;
    sync::Bool=true,
)
    return fvfd_cell_velocity_to_faces_2d!(
        ux_face, uy_face, ux, uy,
        geometry.is_solid, ux_bc, uy_bc, geometry.bc; sync,
    )
end

include("operators_2d_advection.jl")
include("operators_2d_tensor_divergence.jl")
include("operators_2d_velocity_gradient.jl")
