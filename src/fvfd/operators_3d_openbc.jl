@kernel function _fvfd_fill_planar_extensional_openx_bcs_3d_kernel!(
    ux_west, ux_east, u_profile_west, psi_west, psi_east,
    epsilon_dot, dx, x_center, Nx,
)
    j, k = @index(Global, NTuple)
    T = eltype(ux_west)
    @inbounds begin
        x_west = (T(0.5) - x_center) * dx
        x_east = (T(Nx) + T(0.5) - x_center) * dx
        uw = epsilon_dot * x_west
        ue = epsilon_dot * x_east
        ux_west[j, k] = uw
        ux_east[j, k] = ue
        u_profile_west[j, k] = uw
        psi_west[j, k] = zero(T)
        psi_east[j, k] = zero(T)
    end
end

"""
    fvfd_fill_planar_extensional_openx_bcs_3d!(ux_west, ux_east,
        u_profile_west, psi_west, psi_east, Nx, epsilon_dot; dx=1,
        x_center=(Nx + 1)/2)

Fill the imposed open-x FVFD face velocities and zero log-conformation ghost
values for the 3D planar-extension canary. Coordinates are cell-centered:
`x_i = (i - x_center) * dx`, so the west/east faces lie at `i = 0.5` and
`i = Nx + 0.5`.
"""
function fvfd_fill_planar_extensional_openx_bcs_3d!(
    ux_west, ux_east, u_profile_west, psi_west, psi_east,
    Nx::Int, epsilon_dot::Real;
    dx::Real=1,
    x_center::Real=(Nx + 1) / 2,
    sync::Bool=true,
)
    Ny, Nz = size(ux_west)
    _fvfd_check_boundary_size_3d(:ux_east, ux_east, (Ny, Nz))
    _fvfd_check_boundary_size_3d(:u_profile_west, u_profile_west, (Ny, Nz))
    _fvfd_check_boundary_size_3d(:psi_west, psi_west, (Ny, Nz))
    _fvfd_check_boundary_size_3d(:psi_east, psi_east, (Ny, Nz))

    backend = KernelAbstractions.get_backend(ux_west)
    T = eltype(ux_west)
    kernel! = _fvfd_fill_planar_extensional_openx_bcs_3d_kernel!(backend)
    kernel!(
        ux_west, ux_east, u_profile_west, psi_west, psi_east,
        T(epsilon_dot), T(dx), T(x_center), Nx;
        ndrange=(Ny, Nz),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function _fvfd_fill_planar_extensional_openxy_bcs_3d_kernel!(
    ux_west, ux_east, u_profile_west,
    uy_south, uy_north,
    psi_west, psi_east, psi_south, psi_north,
    epsilon_dot, dx, dy, x_center, y_center, Nx, Ny,
)
    a, k = @index(Global, NTuple)
    T = eltype(ux_west)
    @inbounds begin
        if a <= Ny
            x_west = (T(0.5) - x_center) * dx
            x_east = (T(Nx) + T(0.5) - x_center) * dx
            uw = epsilon_dot * x_west
            ue = epsilon_dot * x_east
            ux_west[a, k] = uw
            ux_east[a, k] = ue
            u_profile_west[a, k] = uw
            psi_west[a, k] = zero(T)
            psi_east[a, k] = zero(T)
        end
        if a <= Nx
            y_south = (T(0.5) - y_center) * dy
            y_north = (T(Ny) + T(0.5) - y_center) * dy
            uy_south[a, k] = -epsilon_dot * y_south
            uy_north[a, k] = -epsilon_dot * y_north
            psi_south[a, k] = zero(T)
            psi_north[a, k] = zero(T)
        end
    end
end

"""
    fvfd_fill_planar_extensional_openxy_bcs_3d!(ux_west, ux_east,
        u_profile_west, uy_south, uy_north, psi_west, psi_east,
        psi_south, psi_north, Nx, Ny, epsilon_dot; dx=1, dy=1,
        x_center=(Nx + 1)/2, y_center=(Ny + 1)/2)

Fill the open x/y face velocities for the 3D planar-extension canary:
`ux = epsilon_dot*x_face` on west/east and
`uy = -epsilon_dot*y_face` on south/north. Log-conformation ghost values
on the open lateral faces are reset to zero.
"""
function fvfd_fill_planar_extensional_openxy_bcs_3d!(
    ux_west, ux_east, u_profile_west,
    uy_south, uy_north,
    psi_west, psi_east, psi_south, psi_north,
    Nx::Int, Ny::Int, epsilon_dot::Real;
    dx::Real=1,
    dy::Real=1,
    x_center::Real=(Nx + 1) / 2,
    y_center::Real=(Ny + 1) / 2,
    sync::Bool=true,
)
    Ny_x, Nz = size(ux_west)
    _fvfd_check_boundary_size_3d(:ux_west, ux_west, (Ny, Nz))
    _fvfd_check_boundary_size_3d(:ux_east, ux_east, (Ny, Nz))
    _fvfd_check_boundary_size_3d(:u_profile_west, u_profile_west, (Ny, Nz))
    _fvfd_check_boundary_size_3d(:uy_south, uy_south, (Nx, Nz))
    _fvfd_check_boundary_size_3d(:uy_north, uy_north, (Nx, Nz))
    _fvfd_check_boundary_size_3d(:psi_west, psi_west, (Ny, Nz))
    _fvfd_check_boundary_size_3d(:psi_east, psi_east, (Ny, Nz))
    _fvfd_check_boundary_size_3d(:psi_south, psi_south, (Nx, Nz))
    _fvfd_check_boundary_size_3d(:psi_north, psi_north, (Nx, Nz))

    backend = KernelAbstractions.get_backend(ux_west)
    T = eltype(ux_west)
    kernel! = _fvfd_fill_planar_extensional_openxy_bcs_3d_kernel!(backend)
    kernel!(
        ux_west, ux_east, u_profile_west,
        uy_south, uy_north,
        psi_west, psi_east, psi_south, psi_north,
        T(epsilon_dot), T(dx), T(dy), T(x_center), T(y_center), Nx, Ny;
        ndrange=(max(Nx, Ny), Nz),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function _fvfd_impose_planar_extensional_velocity_3d_kernel!(
    ux, uy, uz, epsilon_dot, dx, dy, x_center, y_center,
)
    i, j, k = @index(Global, NTuple)
    T = eltype(ux)
    @inbounds begin
        x = (T(i) - x_center) * dx
        y = (T(j) - y_center) * dy
        ux[i, j, k] = epsilon_dot * x
        uy[i, j, k] = -epsilon_dot * y
        uz[i, j, k] = zero(T)
    end
end

"""
    fvfd_impose_planar_extensional_velocity_3d!(ux, uy, uz, epsilon_dot;
        dx=1, dy=1, x_center=(Nx + 1)/2, y_center=(Ny + 1)/2)

Overwrite a cell-centered velocity field with `u=(epsilon_dot*x,
-epsilon_dot*y, 0)`. This is the explicit kinematic fallback used when the
coarse open-flow solve does not provide the analytical stagnation field.
"""
function fvfd_impose_planar_extensional_velocity_3d!(
    ux, uy, uz, epsilon_dot::Real;
    dx::Real=1,
    dy::Real=1,
    x_center::Real=(size(ux, 1) + 1) / 2,
    y_center::Real=(size(ux, 2) + 1) / 2,
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(ux)
    Nx, Ny, Nz = size(ux)
    size(uy) == (Nx, Ny, Nz) || throw(DimensionMismatch("uy must match ux"))
    size(uz) == (Nx, Ny, Nz) || throw(DimensionMismatch("uz must match ux"))
    T = eltype(ux)
    kernel! = _fvfd_impose_planar_extensional_velocity_3d_kernel!(backend)
    kernel!(
        ux, uy, uz, T(epsilon_dot), T(dx), T(dy), T(x_center), T(y_center);
        ndrange=(Nx, Ny, Nz),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function fvfd_planar_extensional_velocity_field_host_3d(
    Nx::Int, Ny::Int, Nz::Int, epsilon_dot::Real;
    dx::Real=1,
    dy::Real=1,
    x_center::Real=(Nx + 1) / 2,
    y_center::Real=(Ny + 1) / 2,
    FT::Type{<:AbstractFloat}=Float64,
)
    ux = zeros(FT, Nx, Ny, Nz)
    uy = zeros(FT, Nx, Ny, Nz)
    uz = zeros(FT, Nx, Ny, Nz)
    epsT = FT(epsilon_dot)
    dxT = FT(dx)
    dyT = FT(dy)
    x0 = FT(x_center)
    y0 = FT(y_center)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ux[i, j, k] = epsT * (FT(i) - x0) * dxT
        uy[i, j, k] = -epsT * (FT(j) - y0) * dyT
        uz[i, j, k] = zero(FT)
    end
    return (; ux, uy, uz)
end

@inline function _fvfd_store_d3q19_cell!(f_out, i, j, k, F)
    @inbounds begin
        f_out[i, j, k, 1]  = F[1];  f_out[i, j, k, 2]  = F[2]
        f_out[i, j, k, 3]  = F[3];  f_out[i, j, k, 4]  = F[4]
        f_out[i, j, k, 5]  = F[5];  f_out[i, j, k, 6]  = F[6]
        f_out[i, j, k, 7]  = F[7];  f_out[i, j, k, 8]  = F[8]
        f_out[i, j, k, 9]  = F[9];  f_out[i, j, k, 10] = F[10]
        f_out[i, j, k, 11] = F[11]; f_out[i, j, k, 12] = F[12]
        f_out[i, j, k, 13] = F[13]; f_out[i, j, k, 14] = F[14]
        f_out[i, j, k, 15] = F[15]; f_out[i, j, k, 16] = F[16]
        f_out[i, j, k, 17] = F[17]; f_out[i, j, k, 18] = F[18]
        f_out[i, j, k, 19] = F[19]
    end
    return nothing
end

@kernel function _fvfd_bc_east_zh_velocity_3d!(f_out, f_in, Nx, profile, s_p, s_m)
    jm1, km1 = @index(Global, NTuple); j = jm1 + 1; k = km1 + 1
    T = eltype(f_out)
    @inbounds begin
        fp1  = f_in[Nx,     j,     k,     1]
        fp4  = f_in[Nx,     j - 1, k,     4]
        fp5  = f_in[Nx,     j + 1, k,     5]
        fp6  = f_in[Nx,     j,     k - 1, 6]
        fp7  = f_in[Nx,     j,     k + 1, 7]
        fp16 = f_in[Nx,     j - 1, k - 1, 16]
        fp17 = f_in[Nx,     j + 1, k - 1, 17]
        fp18 = f_in[Nx,     j - 1, k + 1, 18]
        fp19 = f_in[Nx,     j + 1, k + 1, 19]
        fp2  = f_in[Nx - 1, j,     k,     2]
        fp8  = f_in[Nx - 1, j - 1, k,     8]
        fp10 = f_in[Nx - 1, j + 1, k,     10]
        fp12 = f_in[Nx - 1, j,     k - 1, 12]
        fp14 = f_in[Nx - 1, j,     k + 1, 14]
        u_n  = T(profile[j, k])
        sum_par = fp1 + fp4 + fp5 + fp6 + fp7 + fp16 + fp17 + fp18 + fp19
        sum_out = fp2 + fp8 + fp10 + fp12 + fp14
        ρ_w  = (sum_par + T(2) * sum_out) / (one(T) + u_n)
        fp3  = fp2 - T(1 / 3) * ρ_w * u_n
        tang1_diff = fp4 - fp5
        tang2_diff = fp6 - fp7
        fp9  = fp10 - T(0.5) * tang1_diff - T(1 / 6) * ρ_w * u_n
        fp11 = fp8  + T(0.5) * tang1_diff - T(1 / 6) * ρ_w * u_n
        fp13 = fp14 - T(0.5) * tang2_diff - T(1 / 6) * ρ_w * u_n
        fp15 = fp12 + T(0.5) * tang2_diff - T(1 / 6) * ρ_w * u_n
        F = _trt_collide_local_3d(fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8,
                                  fp9, fp10, fp11, fp12, fp13, fp14,
                                  fp15, fp16, fp17, fp18, fp19, s_p, s_m)
        _fvfd_store_d3q19_cell!(f_out, Nx, j, k, F)
    end
end

@kernel function _fvfd_bc_south_zh_velocity_3d!(f_out, f_in, profile, s_p, s_m)
    im1, km1 = @index(Global, NTuple); i = im1 + 1; k = km1 + 1
    T = eltype(f_out)
    @inbounds begin
        fp1  = f_in[i,     1, k,     1]
        fp2  = f_in[i - 1, 1, k,     2]
        fp3  = f_in[i + 1, 1, k,     3]
        fp6  = f_in[i,     1, k - 1, 6]
        fp7  = f_in[i,     1, k + 1, 7]
        fp12 = f_in[i - 1, 1, k - 1, 12]
        fp13 = f_in[i + 1, 1, k - 1, 13]
        fp14 = f_in[i - 1, 1, k + 1, 14]
        fp15 = f_in[i + 1, 1, k + 1, 15]
        fp5  = f_in[i,     2, k,     5]
        fp11 = f_in[i + 1, 2, k,     11]
        fp10 = f_in[i - 1, 2, k,     10]
        fp19 = f_in[i,     2, k + 1, 19]
        fp17 = f_in[i,     2, k - 1, 17]
        u_n  = T(profile[i, k])
        sum_par = fp1 + fp2 + fp3 + fp6 + fp7 + fp12 + fp13 + fp14 + fp15
        sum_out = fp5 + fp11 + fp10 + fp19 + fp17
        ρ_w  = (sum_par + T(2) * sum_out) / (one(T) - u_n)
        fp4  = fp5 + T(1 / 3) * ρ_w * u_n
        tang1_diff = fp2 - fp3
        tang2_diff = fp6 - fp7
        fp8  = fp11 - T(0.5) * tang1_diff + T(1 / 6) * ρ_w * u_n
        fp9  = fp10 + T(0.5) * tang1_diff + T(1 / 6) * ρ_w * u_n
        fp16 = fp19 - T(0.5) * tang2_diff + T(1 / 6) * ρ_w * u_n
        fp18 = fp17 + T(0.5) * tang2_diff + T(1 / 6) * ρ_w * u_n
        F = _trt_collide_local_3d(fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8,
                                  fp9, fp10, fp11, fp12, fp13, fp14,
                                  fp15, fp16, fp17, fp18, fp19, s_p, s_m)
        _fvfd_store_d3q19_cell!(f_out, i, 1, k, F)
    end
end

@kernel function _fvfd_bc_north_zh_velocity_3d!(f_out, f_in, Ny, profile, s_p, s_m)
    im1, km1 = @index(Global, NTuple); i = im1 + 1; k = km1 + 1
    T = eltype(f_out)
    @inbounds begin
        fp1  = f_in[i,     Ny, k,     1]
        fp2  = f_in[i - 1, Ny, k,     2]
        fp3  = f_in[i + 1, Ny, k,     3]
        fp6  = f_in[i,     Ny, k - 1, 6]
        fp7  = f_in[i,     Ny, k + 1, 7]
        fp12 = f_in[i - 1, Ny, k - 1, 12]
        fp13 = f_in[i + 1, Ny, k - 1, 13]
        fp14 = f_in[i - 1, Ny, k + 1, 14]
        fp15 = f_in[i + 1, Ny, k + 1, 15]
        fp4  = f_in[i,     Ny - 1, k,     4]
        fp8  = f_in[i - 1, Ny - 1, k,     8]
        fp9  = f_in[i + 1, Ny - 1, k,     9]
        fp16 = f_in[i,     Ny - 1, k - 1, 16]
        fp18 = f_in[i,     Ny - 1, k + 1, 18]
        u_n  = T(profile[i, k])
        sum_par = fp1 + fp2 + fp3 + fp6 + fp7 + fp12 + fp13 + fp14 + fp15
        sum_out = fp4 + fp8 + fp9 + fp16 + fp18
        ρ_w  = (sum_par + T(2) * sum_out) / (one(T) + u_n)
        fp5  = fp4 - T(1 / 3) * ρ_w * u_n
        tang1_diff = fp2 - fp3
        tang2_diff = fp6 - fp7
        fp10 = fp9  - T(0.5) * tang1_diff - T(1 / 6) * ρ_w * u_n
        fp11 = fp8  + T(0.5) * tang1_diff - T(1 / 6) * ρ_w * u_n
        fp17 = fp18 - T(0.5) * tang2_diff - T(1 / 6) * ρ_w * u_n
        fp19 = fp16 + T(0.5) * tang2_diff - T(1 / 6) * ρ_w * u_n
        F = _trt_collide_local_3d(fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8,
                                  fp9, fp10, fp11, fp12, fp13, fp14,
                                  fp15, fp16, fp17, fp18, fp19, s_p, s_m)
        _fvfd_store_d3q19_cell!(f_out, i, Ny, k, F)
    end
end

function fvfd_apply_extensional_straining_bc_3d!(
    f_out, f_in, ux_east, uy_south, uy_north, ν::Real,
    Nx::Int, Ny::Int, Nz::Int;
    sync::Bool=false,
)
    backend = KernelAbstractions.get_backend(f_out)
    T = eltype(f_out)
    s_p_r, s_m_r = trt_rates(ν; Λ=3/16)
    s_p = T(s_p_r)
    s_m = T(s_m_r)
    _fvfd_check_boundary_size_3d(:ux_east, ux_east, (Ny, Nz))
    _fvfd_check_boundary_size_3d(:uy_south, uy_south, (Nx, Nz))
    _fvfd_check_boundary_size_3d(:uy_north, uy_north, (Nx, Nz))
    _fvfd_bc_east_zh_velocity_3d!(backend)(f_out, f_in, Nx, ux_east, s_p, s_m;
                                           ndrange=(Ny - 2, Nz - 2))
    _fvfd_bc_south_zh_velocity_3d!(backend)(f_out, f_in, uy_south, s_p, s_m;
                                            ndrange=(Nx - 2, Nz - 2))
    _fvfd_bc_north_zh_velocity_3d!(backend)(f_out, f_in, Ny, uy_north, s_p, s_m;
                                            ndrange=(Nx - 2, Nz - 2))
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end
