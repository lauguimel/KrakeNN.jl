function _fvfd_check_boundary_size_3d(name::Symbol, values, expected::Tuple{Int,Int})
    observed = try
        size(values)
    catch
        throw(DimensionMismatch(
            "$(name) boundary does not provide a size; expected $(expected)",
        ))
    end
    observed == expected || throw(DimensionMismatch(
        "$(name) boundary size $(observed) does not match expected $(expected)",
    ))
    return nothing
end

@inline function _fvfd_xface_average_or_zero_3d(ux, is_solid, i_left, i_right, j, k)
    T = eltype(ux)
    return (is_solid[i_left, j, k] || is_solid[i_right, j, k]) ?
           zero(T) :
           (ux[i_left, j, k] + ux[i_right, j, k]) / T(2)
end

@inline function _fvfd_yface_average_or_zero_3d(uy, is_solid, i, j_down, j_up, k)
    T = eltype(uy)
    return (is_solid[i, j_down, k] || is_solid[i, j_up, k]) ?
           zero(T) :
           (uy[i, j_down, k] + uy[i, j_up, k]) / T(2)
end

@inline function _fvfd_zface_average_or_zero_3d(uz, is_solid, i, j, k_back, k_front)
    T = eltype(uz)
    return (is_solid[i, j, k_back] || is_solid[i, j, k_front]) ?
           zero(T) :
           (uz[i, j, k_back] + uz[i, j, k_front]) / T(2)
end

@inline function _fvfd_xface_scalar_average_or_zero_3d(field, is_solid, i_left, i_right, j, k)
    T = eltype(field)
    return (is_solid[i_left, j, k] || is_solid[i_right, j, k]) ?
           zero(T) :
           (field[i_left, j, k] + field[i_right, j, k]) / T(2)
end

@inline function _fvfd_yface_scalar_average_or_zero_3d(field, is_solid, i, j_down, j_up, k)
    T = eltype(field)
    return (is_solid[i, j_down, k] || is_solid[i, j_up, k]) ?
           zero(T) :
           (field[i, j_down, k] + field[i, j_up, k]) / T(2)
end

@inline function _fvfd_zface_scalar_average_or_zero_3d(field, is_solid, i, j, k_back, k_front)
    T = eltype(field)
    return (is_solid[i, j, k_back] || is_solid[i, j, k_front]) ?
           zero(T) :
           (field[i, j, k_back] + field[i, j, k_front]) / T(2)
end

@kernel function fvfd_cell_velocity_to_faces_3d_kernel!(
    ux_face, uy_face, uz_face,
    @Const(ux), @Const(uy), @Const(uz), @Const(is_solid),
    @Const(ux_west), @Const(ux_east),
    @Const(uy_south), @Const(uy_north),
    @Const(uz_back), @Const(uz_front),
    west_bc, east_bc, south_bc, north_bc, back_bc, front_bc, Nx, Ny, Nz,
)
    I, J, K = @index(Global, NTuple)
    @inbounds begin
        if I <= Nx + 1 && J <= Ny && K <= Nz
            if I == 1
                if west_bc == FVFD_BC_PERIODIC
                    ux_face[I, J, K] = _fvfd_xface_average_or_zero_3d(ux, is_solid, Nx, 1, J, K)
                elseif west_bc == FVFD_BC_OPEN
                    ux_face[I, J, K] = is_solid[1, J, K] ? zero(eltype(ux_face)) : ux_west[J, K]
                else
                    ux_face[I, J, K] = zero(eltype(ux_face))
                end
            elseif I == Nx + 1
                if east_bc == FVFD_BC_PERIODIC
                    ux_face[I, J, K] = _fvfd_xface_average_or_zero_3d(ux, is_solid, Nx, 1, J, K)
                elseif east_bc == FVFD_BC_OPEN
                    ux_face[I, J, K] = is_solid[Nx, J, K] ? zero(eltype(ux_face)) : ux_east[J, K]
                else
                    ux_face[I, J, K] = zero(eltype(ux_face))
                end
            else
                ux_face[I, J, K] = _fvfd_xface_average_or_zero_3d(ux, is_solid, I - 1, I, J, K)
            end
        end

        if I <= Nx && J <= Ny + 1 && K <= Nz
            if J == 1
                if south_bc == FVFD_BC_PERIODIC
                    uy_face[I, J, K] = _fvfd_yface_average_or_zero_3d(uy, is_solid, I, Ny, 1, K)
                elseif south_bc == FVFD_BC_OPEN
                    uy_face[I, J, K] = is_solid[I, 1, K] ? zero(eltype(uy_face)) : uy_south[I, K]
                else
                    uy_face[I, J, K] = zero(eltype(uy_face))
                end
            elseif J == Ny + 1
                if north_bc == FVFD_BC_PERIODIC
                    uy_face[I, J, K] = _fvfd_yface_average_or_zero_3d(uy, is_solid, I, Ny, 1, K)
                elseif north_bc == FVFD_BC_OPEN
                    uy_face[I, J, K] = is_solid[I, Ny, K] ? zero(eltype(uy_face)) : uy_north[I, K]
                else
                    uy_face[I, J, K] = zero(eltype(uy_face))
                end
            else
                uy_face[I, J, K] = _fvfd_yface_average_or_zero_3d(uy, is_solid, I, J - 1, J, K)
            end
        end

        if I <= Nx && J <= Ny && K <= Nz + 1
            if K == 1
                if back_bc == FVFD_BC_PERIODIC
                    uz_face[I, J, K] = _fvfd_zface_average_or_zero_3d(uz, is_solid, I, J, Nz, 1)
                elseif back_bc == FVFD_BC_OPEN
                    uz_face[I, J, K] = is_solid[I, J, 1] ? zero(eltype(uz_face)) : uz_back[I, J]
                else
                    uz_face[I, J, K] = zero(eltype(uz_face))
                end
            elseif K == Nz + 1
                if front_bc == FVFD_BC_PERIODIC
                    uz_face[I, J, K] = _fvfd_zface_average_or_zero_3d(uz, is_solid, I, J, Nz, 1)
                elseif front_bc == FVFD_BC_OPEN
                    uz_face[I, J, K] = is_solid[I, J, Nz] ? zero(eltype(uz_face)) : uz_front[I, J]
                else
                    uz_face[I, J, K] = zero(eltype(uz_face))
                end
            else
                uz_face[I, J, K] = _fvfd_zface_average_or_zero_3d(uz, is_solid, I, J, K - 1, K)
            end
        end
    end
end

function fvfd_cell_velocity_to_faces_3d!(
    ux_face, uy_face, uz_face, ux, uy, uz, is_solid,
    ux_west, ux_east, uy_south, uy_north, uz_back, uz_front,
    west_bc, east_bc, south_bc, north_bc, back_bc, front_bc;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(ux_face)
    Nx, Ny, Nz = size(ux)
    west = fvfd_domain_bc_code(west_bc)
    east = fvfd_domain_bc_code(east_bc)
    south = fvfd_domain_bc_code(south_bc)
    north = fvfd_domain_bc_code(north_bc)
    back = fvfd_domain_bc_code(back_bc)
    front = fvfd_domain_bc_code(front_bc)
    west == FVFD_BC_OPEN && _fvfd_check_boundary_size_3d(:ux_west, ux_west, (Ny, Nz))
    east == FVFD_BC_OPEN && _fvfd_check_boundary_size_3d(:ux_east, ux_east, (Ny, Nz))
    south == FVFD_BC_OPEN && _fvfd_check_boundary_size_3d(:uy_south, uy_south, (Nx, Nz))
    north == FVFD_BC_OPEN && _fvfd_check_boundary_size_3d(:uy_north, uy_north, (Nx, Nz))
    back == FVFD_BC_OPEN && _fvfd_check_boundary_size_3d(:uz_back, uz_back, (Nx, Ny))
    front == FVFD_BC_OPEN && _fvfd_check_boundary_size_3d(:uz_front, uz_front, (Nx, Ny))
    kernel! = fvfd_cell_velocity_to_faces_3d_kernel!(backend)
    kernel!(
        ux_face, uy_face, uz_face, ux, uy, uz, is_solid,
        ux_west, ux_east, uy_south, uy_north, uz_back, uz_front,
        west, east, south, north, back, front, Nx, Ny, Nz;
        ndrange=(Nx + 1, Ny + 1, Nz + 1),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

@inline function _fvfd_bc_east_scalar_3d(phi, east_phi, i, j, k, Nx, east_bc)
    if i < Nx
        return phi[i + 1, j, k]
    elseif east_bc == FVFD_BC_PERIODIC
        return phi[1, j, k]
    elseif east_bc == FVFD_BC_OPEN
        return east_phi[j, k]
    else
        return phi[i, j, k]
    end
end

@inline function _fvfd_bc_west_scalar_3d(phi, west_phi, i, j, k, Nx, west_bc)
    if i > 1
        return phi[i - 1, j, k]
    elseif west_bc == FVFD_BC_PERIODIC
        return phi[Nx, j, k]
    elseif west_bc == FVFD_BC_OPEN
        return west_phi[j, k]
    else
        return phi[i, j, k]
    end
end

@inline function _fvfd_bc_north_scalar_3d(phi, north_phi, i, j, k, Ny, north_bc)
    if j < Ny
        return phi[i, j + 1, k]
    elseif north_bc == FVFD_BC_PERIODIC
        return phi[i, 1, k]
    elseif north_bc == FVFD_BC_OPEN
        return north_phi[i, k]
    else
        return phi[i, j, k]
    end
end

@inline function _fvfd_bc_south_scalar_3d(phi, south_phi, i, j, k, Ny, south_bc)
    if j > 1
        return phi[i, j - 1, k]
    elseif south_bc == FVFD_BC_PERIODIC
        return phi[i, Ny, k]
    elseif south_bc == FVFD_BC_OPEN
        return south_phi[i, k]
    else
        return phi[i, j, k]
    end
end

@inline function _fvfd_bc_front_scalar_3d(phi, front_phi, i, j, k, Nz, front_bc)
    if k < Nz
        return phi[i, j, k + 1]
    elseif front_bc == FVFD_BC_PERIODIC
        return phi[i, j, 1]
    elseif front_bc == FVFD_BC_OPEN
        return front_phi[i, j]
    else
        return phi[i, j, k]
    end
end

@inline function _fvfd_bc_back_scalar_3d(phi, back_phi, i, j, k, Nz, back_bc)
    if k > 1
        return phi[i, j, k - 1]
    elseif back_bc == FVFD_BC_PERIODIC
        return phi[i, j, Nz]
    elseif back_bc == FVFD_BC_OPEN
        return back_phi[i, j]
    else
        return phi[i, j, k]
    end
end

function _fvfd_advection_scheme_val_3d(advection_scheme::Symbol)
    scheme = Symbol(replace(lowercase(String(advection_scheme)), '-' => '_'))
    scheme in (:rusanov, :muscl_superbee) ||
        throw(ArgumentError("advection_scheme must be :rusanov or :muscl_superbee"))
    return Val(scheme)
end

@inline _fvfd_superbee_limiter_3d(r) = _fvfd_superbee_limiter_2d(r)
@inline _fvfd_muscl_superbee_face_value_3d(far_upwind, upwind, downwind) =
    _fvfd_muscl_superbee_face_value_2d(far_upwind, upwind, downwind)

@inline function _fvfd_upwind_scalar_advective_rhs_3d(
    phi, west_phi, east_phi, south_phi, north_phi, back_phi, front_phi,
    ux_face, uy_face, uz_face, is_solid, i, j, k, Nx, Ny, Nz, inv_dx, inv_dy, inv_dz,
    west_bc, east_bc, south_bc, north_bc, back_bc, front_bc,
    ::Val{:rusanov},
)
    ue = ux_face[i + 1, j, k]
    uw = ux_face[i, j, k]
    vn = uy_face[i, j + 1, k]
    vs = uy_face[i, j, k]
    wf = uz_face[i, j, k + 1]
    wb = uz_face[i, j, k]

    east_value = _fvfd_bc_east_scalar_3d(phi, east_phi, i, j, k, Nx, east_bc)
    west_value = _fvfd_bc_west_scalar_3d(phi, west_phi, i, j, k, Nx, west_bc)
    north_value = _fvfd_bc_north_scalar_3d(phi, north_phi, i, j, k, Ny, north_bc)
    south_value = _fvfd_bc_south_scalar_3d(phi, south_phi, i, j, k, Ny, south_bc)
    front_value = _fvfd_bc_front_scalar_3d(phi, front_phi, i, j, k, Nz, front_bc)
    back_value = _fvfd_bc_back_scalar_3d(phi, back_phi, i, j, k, Nz, back_bc)

    phie = ifelse(ue >= 0, phi[i, j, k], east_value)
    phiw = ifelse(uw >= 0, west_value, phi[i, j, k])
    phin = ifelse(vn >= 0, phi[i, j, k], north_value)
    phis = ifelse(vs >= 0, south_value, phi[i, j, k])
    phif = ifelse(wf >= 0, phi[i, j, k], front_value)
    phib = ifelse(wb >= 0, back_value, phi[i, j, k])

    flux_div = (ue * phie - uw * phiw) * inv_dx +
               (vn * phin - vs * phis) * inv_dy +
               (wf * phif - wb * phib) * inv_dz
    divu = (ue - uw) * inv_dx + (vn - vs) * inv_dy + (wf - wb) * inv_dz
    return -(flux_div - phi[i, j, k] * divu)
end

@inline function _fvfd_upwind_scalar_advective_rhs_3d(
    phi, west_phi, east_phi, south_phi, north_phi, back_phi, front_phi,
    ux_face, uy_face, uz_face, is_solid, i, j, k, Nx, Ny, Nz, inv_dx, inv_dy, inv_dz,
    west_bc, east_bc, south_bc, north_bc, back_bc, front_bc,
    ::Val{:muscl_superbee},
)
    # Fall back to first-order upwind within two cells of an x or y face, of
    # a non-periodic z face, or of a solid cell. A periodic z wraps its
    # stencil instead, so a thin periodic z (quasi-2D run) stays second order.
    # x and y keep the fallback even when periodic: on a z-slab this operator
    # must equal the 2D one (`operators_2d_advection.jl`), whose hand-written
    # adjoint (`ad/ad_ve_ops.jl`) mirrors that fallback.
    zper = back_bc == FVFD_BC_PERIODIC && front_bc == FVFD_BC_PERIODIC
    if i <= 2 || i >= Nx - 1 || j <= 2 || j >= Ny - 1 ||
       (!zper && (k <= 2 || k >= Nz - 1))
        return _fvfd_upwind_scalar_advective_rhs_3d(
            phi, west_phi, east_phi, south_phi, north_phi, back_phi, front_phi,
            ux_face, uy_face, uz_face, is_solid, i, j, k, Nx, Ny, Nz, inv_dx, inv_dy, inv_dz,
            west_bc, east_bc, south_bc, north_bc, back_bc, front_bc, Val(:rusanov),
        )
    end
    im2, im1, ip1, ip2 = i - 2, i - 1, i + 1, i + 2
    jm2, jm1, jp1, jp2 = j - 2, j - 1, j + 1, j + 2
    km2 = zper ? mod1(k - 2, Nz) : k - 2
    km1 = zper ? mod1(k - 1, Nz) : k - 1
    kp1 = zper ? mod1(k + 1, Nz) : k + 1
    kp2 = zper ? mod1(k + 2, Nz) : k + 2
    if is_solid[im2, j, k] || is_solid[im1, j, k] ||
       is_solid[ip1, j, k] || is_solid[ip2, j, k] ||
       is_solid[i, jm2, k] || is_solid[i, jm1, k] ||
       is_solid[i, jp1, k] || is_solid[i, jp2, k] ||
       is_solid[i, j, km2] || is_solid[i, j, km1] ||
       is_solid[i, j, kp1] || is_solid[i, j, kp2]
        return _fvfd_upwind_scalar_advective_rhs_3d(
            phi, west_phi, east_phi, south_phi, north_phi, back_phi, front_phi,
            ux_face, uy_face, uz_face, is_solid, i, j, k, Nx, Ny, Nz, inv_dx, inv_dy, inv_dz,
            west_bc, east_bc, south_bc, north_bc, back_bc, front_bc, Val(:rusanov),
        )
    end

    ue = ux_face[i + 1, j, k]
    uw = ux_face[i, j, k]
    vn = uy_face[i, j + 1, k]
    vs = uy_face[i, j, k]
    wf = uz_face[i, j, k + 1]
    wb = uz_face[i, j, k]

    phie = ifelse(
        ue >= 0,
        _fvfd_muscl_superbee_face_value_3d(phi[im1, j, k], phi[i, j, k], phi[ip1, j, k]),
        _fvfd_muscl_superbee_face_value_3d(phi[ip2, j, k], phi[ip1, j, k], phi[i, j, k]),
    )
    phiw = ifelse(
        uw >= 0,
        _fvfd_muscl_superbee_face_value_3d(phi[im2, j, k], phi[im1, j, k], phi[i, j, k]),
        _fvfd_muscl_superbee_face_value_3d(phi[ip1, j, k], phi[i, j, k], phi[im1, j, k]),
    )
    phin = ifelse(
        vn >= 0,
        _fvfd_muscl_superbee_face_value_3d(phi[i, jm1, k], phi[i, j, k], phi[i, jp1, k]),
        _fvfd_muscl_superbee_face_value_3d(phi[i, jp2, k], phi[i, jp1, k], phi[i, j, k]),
    )
    phis = ifelse(
        vs >= 0,
        _fvfd_muscl_superbee_face_value_3d(phi[i, jm2, k], phi[i, jm1, k], phi[i, j, k]),
        _fvfd_muscl_superbee_face_value_3d(phi[i, jp1, k], phi[i, j, k], phi[i, jm1, k]),
    )
    phif = ifelse(
        wf >= 0,
        _fvfd_muscl_superbee_face_value_3d(phi[i, j, km1], phi[i, j, k], phi[i, j, kp1]),
        _fvfd_muscl_superbee_face_value_3d(phi[i, j, kp2], phi[i, j, kp1], phi[i, j, k]),
    )
    phib = ifelse(
        wb >= 0,
        _fvfd_muscl_superbee_face_value_3d(phi[i, j, km2], phi[i, j, km1], phi[i, j, k]),
        _fvfd_muscl_superbee_face_value_3d(phi[i, j, kp1], phi[i, j, k], phi[i, j, km1]),
    )

    flux_div = (ue * phie - uw * phiw) * inv_dx +
               (vn * phin - vs * phis) * inv_dy +
               (wf * phif - wb * phib) * inv_dz
    divu = (ue - uw) * inv_dx + (vn - vs) * inv_dy + (wf - wb) * inv_dz
    return -(flux_div - phi[i, j, k] * divu)
end

@kernel function fvfd_advect_upwind_3d_kernel!(
    phi_out, @Const(phi),
    @Const(west_phi), @Const(east_phi),
    @Const(south_phi), @Const(north_phi),
    @Const(back_phi), @Const(front_phi),
    @Const(ux_face), @Const(uy_face), @Const(uz_face), @Const(is_solid),
    dt, inv_dx, inv_dy, inv_dz,
    west_bc, east_bc, south_bc, north_bc, back_bc, front_bc, Nx, Ny, Nz, advection_scheme,
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny && k <= Nz
            if is_solid[i, j, k]
                phi_out[i, j, k] = zero(eltype(phi_out))
            else
                rhs = _fvfd_upwind_scalar_advective_rhs_3d(
                    phi, west_phi, east_phi, south_phi, north_phi, back_phi, front_phi,
                    ux_face, uy_face, uz_face, is_solid, i, j, k, Nx, Ny, Nz,
                    inv_dx, inv_dy, inv_dz,
                    west_bc, east_bc, south_bc, north_bc, back_bc, front_bc,
                    advection_scheme,
                )
                phi_out[i, j, k] = phi[i, j, k] + dt * rhs
            end
        end
    end
end

function fvfd_advect_upwind_3d!(
    phi_out, phi,
    west_phi, east_phi, south_phi, north_phi, back_phi, front_phi,
    ux_face, uy_face, uz_face, is_solid,
    dx, dy, dz,
    west_bc, east_bc, south_bc, north_bc, back_bc, front_bc,
    dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    backend = KernelAbstractions.get_backend(phi_out)
    Nx, Ny, Nz = size(phi_out)
    west = fvfd_domain_bc_code(west_bc)
    east = fvfd_domain_bc_code(east_bc)
    south = fvfd_domain_bc_code(south_bc)
    north = fvfd_domain_bc_code(north_bc)
    back = fvfd_domain_bc_code(back_bc)
    front = fvfd_domain_bc_code(front_bc)
    west == FVFD_BC_OPEN && _fvfd_check_boundary_size_3d(:phi_west, west_phi, (Ny, Nz))
    east == FVFD_BC_OPEN && _fvfd_check_boundary_size_3d(:phi_east, east_phi, (Ny, Nz))
    south == FVFD_BC_OPEN && _fvfd_check_boundary_size_3d(:phi_south, south_phi, (Nx, Nz))
    north == FVFD_BC_OPEN && _fvfd_check_boundary_size_3d(:phi_north, north_phi, (Nx, Nz))
    back == FVFD_BC_OPEN && _fvfd_check_boundary_size_3d(:phi_back, back_phi, (Nx, Ny))
    front == FVFD_BC_OPEN && _fvfd_check_boundary_size_3d(:phi_front, front_phi, (Nx, Ny))
    scheme = _fvfd_advection_scheme_val_3d(advection_scheme)
    kernel! = fvfd_advect_upwind_3d_kernel!(backend)
    kernel!(
        phi_out, phi,
        west_phi, east_phi, south_phi, north_phi, back_phi, front_phi,
        ux_face, uy_face, uz_face, is_solid,
        dt, inv(dx), inv(dy), inv(dz),
        west, east, south, north, back, front, Nx, Ny, Nz, scheme;
        ndrange=(Nx, Ny, Nz),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function fvfd_sym3_advect_upwind_3d!(
    psixx_out, psixy_out, psixz_out, psiyy_out, psiyz_out, psizz_out,
    psixx, psixy, psixz, psiyy, psiyz, psizz,
    psixx_bc, psixy_bc, psixz_bc, psiyy_bc, psiyz_bc, psizz_bc,
    ux_face, uy_face, uz_face, is_solid,
    dx, dy, dz,
    west_bc, east_bc, south_bc, north_bc, back_bc, front_bc,
    dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    fvfd_advect_upwind_3d!(
        psixx_out, psixx, psixx_bc...,
        ux_face, uy_face, uz_face, is_solid, dx, dy, dz,
        west_bc, east_bc, south_bc, north_bc, back_bc, front_bc, dt;
        sync=false, advection_scheme,
    )
    fvfd_advect_upwind_3d!(
        psixy_out, psixy, psixy_bc...,
        ux_face, uy_face, uz_face, is_solid, dx, dy, dz,
        west_bc, east_bc, south_bc, north_bc, back_bc, front_bc, dt;
        sync=false, advection_scheme,
    )
    fvfd_advect_upwind_3d!(
        psixz_out, psixz, psixz_bc...,
        ux_face, uy_face, uz_face, is_solid, dx, dy, dz,
        west_bc, east_bc, south_bc, north_bc, back_bc, front_bc, dt;
        sync=false, advection_scheme,
    )
    fvfd_advect_upwind_3d!(
        psiyy_out, psiyy, psiyy_bc...,
        ux_face, uy_face, uz_face, is_solid, dx, dy, dz,
        west_bc, east_bc, south_bc, north_bc, back_bc, front_bc, dt;
        sync=false, advection_scheme,
    )
    fvfd_advect_upwind_3d!(
        psiyz_out, psiyz, psiyz_bc...,
        ux_face, uy_face, uz_face, is_solid, dx, dy, dz,
        west_bc, east_bc, south_bc, north_bc, back_bc, front_bc, dt;
        sync=false, advection_scheme,
    )
    fvfd_advect_upwind_3d!(
        psizz_out, psizz, psizz_bc...,
        ux_face, uy_face, uz_face, is_solid, dx, dy, dz,
        west_bc, east_bc, south_bc, north_bc, back_bc, front_bc, dt;
        sync=false, advection_scheme,
    )
    sync && KernelAbstractions.synchronize(KernelAbstractions.get_backend(psixx_out))
    return nothing
end
