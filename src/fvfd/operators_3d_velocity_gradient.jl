@inline function _fvfd_solid_bc_derivative_x_3d(
    field, is_solid, i, j, k, Nx, inv_dx, inv_2dx, west_bc, east_bc,
)
    T = eltype(field)
    li = _fvfd_bc_index_1d(i - 1, Nx, west_bc, east_bc)
    ri = _fvfd_bc_index_1d(i + 1, Nx, west_bc, east_bc)
    left = li != 0 && !is_solid[li, j, k]
    right = ri != 0 && !is_solid[ri, j, k]
    if left && right
        return (field[ri, j, k] - field[li, j, k]) * inv_2dx
    elseif right
        r2i = _fvfd_bc_index_1d(i + 2, Nx, west_bc, east_bc)
        return (r2i != 0 && !is_solid[r2i, j, k]) ?
               (-T(3) * field[i, j, k] + T(4) * field[ri, j, k] - field[r2i, j, k]) *
               inv_2dx :
               (field[ri, j, k] - field[i, j, k]) * inv_dx
    elseif left
        l2i = _fvfd_bc_index_1d(i - 2, Nx, west_bc, east_bc)
        return (l2i != 0 && !is_solid[l2i, j, k]) ?
               (T(3) * field[i, j, k] - T(4) * field[li, j, k] + field[l2i, j, k]) *
               inv_2dx :
               (field[i, j, k] - field[li, j, k]) * inv_dx
    else
        return zero(T)
    end
end

@inline function _fvfd_solid_bc_derivative_y_3d(
    field, is_solid, i, j, k, Ny, inv_dy, inv_2dy, south_bc, north_bc,
)
    T = eltype(field)
    dj = _fvfd_bc_index_1d(j - 1, Ny, south_bc, north_bc)
    uj = _fvfd_bc_index_1d(j + 1, Ny, south_bc, north_bc)
    down = dj != 0 && !is_solid[i, dj, k]
    up = uj != 0 && !is_solid[i, uj, k]
    if down && up
        return (field[i, uj, k] - field[i, dj, k]) * inv_2dy
    elseif up
        u2j = _fvfd_bc_index_1d(j + 2, Ny, south_bc, north_bc)
        return (u2j != 0 && !is_solid[i, u2j, k]) ?
               (-T(3) * field[i, j, k] + T(4) * field[i, uj, k] - field[i, u2j, k]) *
               inv_2dy :
               (field[i, uj, k] - field[i, j, k]) * inv_dy
    elseif down
        d2j = _fvfd_bc_index_1d(j - 2, Ny, south_bc, north_bc)
        return (d2j != 0 && !is_solid[i, d2j, k]) ?
               (T(3) * field[i, j, k] - T(4) * field[i, dj, k] + field[i, d2j, k]) *
               inv_2dy :
               (field[i, j, k] - field[i, dj, k]) * inv_dy
    else
        return zero(T)
    end
end

@inline function _fvfd_solid_bc_derivative_z_3d(
    field, is_solid, i, j, k, Nz, inv_dz, inv_2dz, back_bc, front_bc,
)
    T = eltype(field)
    bk = _fvfd_bc_index_1d(k - 1, Nz, back_bc, front_bc)
    fk = _fvfd_bc_index_1d(k + 1, Nz, back_bc, front_bc)
    back = bk != 0 && !is_solid[i, j, bk]
    front = fk != 0 && !is_solid[i, j, fk]
    if back && front
        return (field[i, j, fk] - field[i, j, bk]) * inv_2dz
    elseif front
        f2k = _fvfd_bc_index_1d(k + 2, Nz, back_bc, front_bc)
        return (f2k != 0 && !is_solid[i, j, f2k]) ?
               (-T(3) * field[i, j, k] + T(4) * field[i, j, fk] - field[i, j, f2k]) *
               inv_2dz :
               (field[i, j, fk] - field[i, j, k]) * inv_dz
    elseif back
        b2k = _fvfd_bc_index_1d(k - 2, Nz, back_bc, front_bc)
        return (b2k != 0 && !is_solid[i, j, b2k]) ?
               (T(3) * field[i, j, k] - T(4) * field[i, j, bk] + field[i, j, b2k]) *
               inv_2dz :
               (field[i, j, k] - field[i, j, bk]) * inv_dz
    else
        return zero(T)
    end
end

@inline function _fvfd_velocity_gradient_dy_3d(
    field, is_solid, i, j, k, Ny, inv_dy, inv_2dy, south_bc, north_bc,
)
    if south_bc == FVFD_BC_WALL && north_bc == FVFD_BC_WALL
        return _wall_aware_dy_3d(field, is_solid, i, j, k, Ny, eltype(field)) * inv_dy
    end
    return _fvfd_solid_bc_derivative_y_3d(
        field, is_solid, i, j, k, Ny, inv_dy, inv_2dy, south_bc, north_bc,
    )
end

@inline function _fvfd_velocity_gradient_dz_3d(
    field, is_solid, i, j, k, Nz, inv_dz, inv_2dz, back_bc, front_bc,
)
    if back_bc == FVFD_BC_WALL && front_bc == FVFD_BC_WALL
        return _wall_aware_dz_3d(field, is_solid, i, j, k, Nz, eltype(field)) * inv_dz
    end
    return _fvfd_solid_bc_derivative_z_3d(
        field, is_solid, i, j, k, Nz, inv_dz, inv_2dz, back_bc, front_bc,
    )
end

@kernel function fvfd_velocity_gradient_3d_kernel!(
    duxdx, duxdy, duxdz,
    duydx, duydy, duydz,
    duzdx, duzdy, duzdz,
    @Const(ux), @Const(uy), @Const(uz), @Const(is_solid),
    inv_dx, inv_dy, inv_dz, inv_2dx, inv_2dy, inv_2dz,
    west_bc, east_bc, south_bc, north_bc, back_bc, front_bc, Nx, Ny, Nz,
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny && k <= Nz
            if is_solid[i, j, k]
                duxdx[i, j, k] = zero(eltype(duxdx))
                duxdy[i, j, k] = zero(eltype(duxdy))
                duxdz[i, j, k] = zero(eltype(duxdz))
                duydx[i, j, k] = zero(eltype(duydx))
                duydy[i, j, k] = zero(eltype(duydy))
                duydz[i, j, k] = zero(eltype(duydz))
                duzdx[i, j, k] = zero(eltype(duzdx))
                duzdy[i, j, k] = zero(eltype(duzdy))
                duzdz[i, j, k] = zero(eltype(duzdz))
            else
                duxdx[i, j, k] = _fvfd_solid_bc_derivative_x_3d(
                    ux, is_solid, i, j, k, Nx, inv_dx, inv_2dx, west_bc, east_bc,
                )
                duxdy[i, j, k] = _fvfd_velocity_gradient_dy_3d(
                    ux, is_solid, i, j, k, Ny, inv_dy, inv_2dy, south_bc, north_bc,
                )
                duxdz[i, j, k] = _fvfd_velocity_gradient_dz_3d(
                    ux, is_solid, i, j, k, Nz, inv_dz, inv_2dz, back_bc, front_bc,
                )
                duydx[i, j, k] = _fvfd_solid_bc_derivative_x_3d(
                    uy, is_solid, i, j, k, Nx, inv_dx, inv_2dx, west_bc, east_bc,
                )
                duydy[i, j, k] = _fvfd_velocity_gradient_dy_3d(
                    uy, is_solid, i, j, k, Ny, inv_dy, inv_2dy, south_bc, north_bc,
                )
                duydz[i, j, k] = _fvfd_velocity_gradient_dz_3d(
                    uy, is_solid, i, j, k, Nz, inv_dz, inv_2dz, back_bc, front_bc,
                )
                duzdx[i, j, k] = _fvfd_solid_bc_derivative_x_3d(
                    uz, is_solid, i, j, k, Nx, inv_dx, inv_2dx, west_bc, east_bc,
                )
                duzdy[i, j, k] = _fvfd_velocity_gradient_dy_3d(
                    uz, is_solid, i, j, k, Ny, inv_dy, inv_2dy, south_bc, north_bc,
                )
                duzdz[i, j, k] = _fvfd_velocity_gradient_dz_3d(
                    uz, is_solid, i, j, k, Nz, inv_dz, inv_2dz, back_bc, front_bc,
                )
            end
        end
    end
end

function fvfd_velocity_gradient_3d!(
    duxdx, duxdy, duxdz,
    duydx, duydy, duydz,
    duzdx, duzdy, duzdz,
    ux, uy, uz,
    is_solid;
    dx,
    dy,
    dz,
    x_bc=:periodic,
    y_bc=:wall,
    z_bc=:periodic,
    sync::Bool=true,
)
    return fvfd_velocity_gradient_3d!(
        duxdx, duxdy, duxdz,
        duydx, duydy, duydz,
        duzdx, duzdy, duzdz,
        ux, uy, uz, is_solid, dx, dy, dz,
        x_bc, x_bc, y_bc, y_bc, z_bc, z_bc; sync,
    )
end

function fvfd_velocity_gradient_3d!(
    duxdx, duxdy, duxdz,
    duydx, duydy, duydz,
    duzdx, duzdy, duzdz,
    ux, uy, uz, is_solid,
    dx, dy, dz,
    west_bc, east_bc, south_bc, north_bc, back_bc, front_bc;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(ux)
    Nx, Ny, Nz = size(ux)
    west = fvfd_domain_bc_code(west_bc)
    east = fvfd_domain_bc_code(east_bc)
    south = fvfd_domain_bc_code(south_bc)
    north = fvfd_domain_bc_code(north_bc)
    back = fvfd_domain_bc_code(back_bc)
    front = fvfd_domain_bc_code(front_bc)
    kernel! = fvfd_velocity_gradient_3d_kernel!(backend)
    kernel!(
        duxdx, duxdy, duxdz,
        duydx, duydy, duydz,
        duzdx, duzdy, duzdz,
        ux, uy, uz, is_solid,
        inv(dx), inv(dy), inv(dz), inv(2 * dx), inv(2 * dy), inv(2 * dz),
        west, east, south, north, back, front, Nx, Ny, Nz;
        ndrange=(Nx, Ny, Nz),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end
