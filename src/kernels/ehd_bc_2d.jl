using KernelAbstractions

# Opt-in #73 closure. Preserve outgoing normal aggregate and tangential
# difference, not each individual outgoing population. D2Q9 order is R,E,N,W,S,
# NE,NW,SW,SE. The collector remains the existing zero-gradient NEE condition.
@inline function _ehd_moment_eq(k, q, vx, vy)
    T = typeof(q)
    cx = (0,1,0,-1,0,1,-1,-1,1)[k]
    cy = (0,0,1,0,-1,1,1,-1,-1)[k]
    w = k == 1 ? T(4)/T(9) : k <= 5 ? T(1)/T(9) : T(1)/T(36)
    c = cx*vx + cy*vy
    return q*w*(one(T)+T(3)*c+T(4.5)*c*c-T(1.5)*(vx*vx+vy*vy))
end

@inline function _ehd_moment_wall!(f, i, j, qb, vx, vy, top)
    T = eltype(f)
    @inbounds begin
        if top
            g = f[i,j,3]+f[i,j,6]+f[i,j,7]
            d = f[i,j,6]-f[i,j,7]
            jy = T(2)*(g-qb*(one(T)+T(3)*vy+T(3)*vy*vy)/T(6))
            jx = T(6)*d-qb*vx*(one(T)+T(3)*vy)
        else
            g = f[i,j,5]+f[i,j,8]+f[i,j,9]
            d = f[i,j,9]-f[i,j,8]
            jy = T(2)*(qb*(one(T)-T(3)*vy+T(3)*vy*vy)/T(6)-g)
            jx = T(6)*d-qb*vx*(one(T)-T(3)*vy)
        end
        for k in 1:9
            cx = (0,1,0,-1,0,1,-1,-1,1)[k]
            cy = (0,0,1,0,-1,1,1,-1,-1)[k]
            w = k == 1 ? T(4)/T(9) : k <= 5 ? T(1)/T(9) : T(1)/T(36)
            f[i,j,k] = _ehd_moment_eq(k,qb,vx,vy)+T(3)*w*(cx*jx+cy*jy)
        end
    end
end

@kernel function _ehd_moment_electrodes_kernel!(f, @Const(q), @Const(ux), @Const(uy),
        @Const(Ex), @Const(Ey), qb, K, Nx, Ny, periodic, moment)
    i = @index(Global)
    @inbounds if periodic || (1 < i < Nx)
        if moment
            _ehd_moment_wall!(f,i,1,qb,ux[i,1]+K*Ex[i,1],uy[i,1]+K*Ey[i,1],false)
        else
            _ehd_apply_charge_nee!(f,q,ux,uy,Ex,Ey,K,i,1,i,2,qb)
        end
        if periodic && moment
            _ehd_moment_wall!(f,i,Ny,qb,ux[i,Ny]+K*Ex[i,Ny],uy[i,Ny]+K*Ey[i,Ny],true)
        else
            qt = periodic ? qb : q[i,Ny-1]
            _ehd_apply_charge_nee!(f,q,ux,uy,Ex,Ey,K,i,Ny,i,Ny-1,qt)
        end
    end
end

@kernel function _ehd_moment_sides_kernel!(f, @Const(q), @Const(ux), @Const(uy),
        @Const(Ex), @Const(Ey), K, Nx)
    j = @index(Global)
    @inbounds begin
        _ehd_apply_charge_nee!(f,q,ux,uy,Ex,Ey,K,1,j,2,j,q[2,j])
        _ehd_apply_charge_nee!(f,q,ux,uy,Ex,Ey,K,Nx,j,Nx-1,j,q[Nx-1,j])
    end
end

# Internal control switches allow periodic verification and an ordered NEE
# control (moment=false) on the same race-free schedule as the moment closure.
# This control is NOT bit-identical to the production apply_charge_nee_box_2d!:
# it recomputes q after electrodes, before sidewalls read completed corner data.
# ECState calls this wrapper only for the box moment option. For Nx>=4, the
# side writes never read another side destination. Separate launches with a
# barrier complete electrodes and q before corner reads: no #64 scheduling race.
function _apply_charge_moment_box_2d!(f,q,ux,uy,Ex,Ey,qb,K,Nx,Ny;
                                     periodic=false, moment=true)
    Nx >= 4 && Ny >= 3 || throw(ArgumentError("moment walls require Nx>=4, Ny>=3"))
    b = KernelAbstractions.get_backend(f)
    _ehd_moment_electrodes_kernel!(b)(f,q,ux,uy,Ex,Ey,eltype(f)(qb),eltype(f)(K),
        Nx,Ny,periodic,moment; ndrange=Nx)
    KernelAbstractions.synchronize(b)
    compute_ehd_scalar_2d!(q,f)
    KernelAbstractions.synchronize(b)
    if !periodic
        _ehd_moment_sides_kernel!(b)(f,q,ux,uy,Ex,Ey,eltype(f)(K),Nx; ndrange=Ny)
        KernelAbstractions.synchronize(b)
        compute_ehd_scalar_2d!(q,f)
    end
    return nothing
end

# EHD-local sidewall and coupling kernels for the electroconvection canary.
# Direction order: rest,E,N,W,S,NE,NW,SW,SE.

@kernel function stream_wall_x_wall_y_2d_kernel!(f_out, @Const(f_in), Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        im = ifelse(i > 1, i - 1, i)
        ip = ifelse(i < Nx, i + 1, i)
        jm = ifelse(j > 1, j - 1, j)
        jp = ifelse(j < Ny, j + 1, j)

        f_out[i,j,1] = f_in[i, j, 1]
        f_out[i,j,2] = ifelse(i > 1, f_in[im, j, 2], f_in[i, j, 4])
        f_out[i,j,3] = ifelse(j > 1, f_in[i, jm, 3], f_in[i, j, 5])
        f_out[i,j,4] = ifelse(i < Nx, f_in[ip, j, 4], f_in[i, j, 2])
        f_out[i,j,5] = ifelse(j < Ny, f_in[i, jp, 5], f_in[i, j, 3])
        f_out[i,j,6] = ifelse(i > 1 && j > 1, f_in[im, jm, 6], f_in[i, j, 8])
        f_out[i,j,7] = ifelse(i < Nx && j > 1, f_in[ip, jm, 7], f_in[i, j, 9])
        f_out[i,j,8] = ifelse(i < Nx && j < Ny, f_in[ip, jp, 8], f_in[i, j, 6])
        f_out[i,j,9] = ifelse(i > 1 && j < Ny, f_in[im, jp, 9], f_in[i, j, 7])
    end
end

function stream_wall_x_wall_y_2d!(f_out, f_in, Nx, Ny)
    backend = KernelAbstractions.get_backend(f_in)
    kernel! = stream_wall_x_wall_y_2d_kernel!(backend)
    kernel!(f_out, f_in, Nx, Ny; ndrange=(Nx, Ny))
end

@kernel function collide_electric_charge_srt_u_2d_kernel!(f, @Const(ux), @Const(uy),
                                                          @Const(Ex), @Const(Ey),
                                                          tau_q, K)
    i, j = @index(Global, NTuple)
    @inbounds begin
        q = f[i,j,1] + f[i,j,2] + f[i,j,3] + f[i,j,4] +
            f[i,j,5] + f[i,j,6] + f[i,j,7] + f[i,j,8] + f[i,j,9]
        ueqx = ux[i, j] + K * Ex[i, j]
        ueqy = uy[i, j] + K * Ey[i, j]
        ω = one(eltype(f)) / tau_q
        f[i,j,1] -= ω * (f[i,j,1] - ehd_charge_feq(Val(1), q, ueqx, ueqy))
        f[i,j,2] -= ω * (f[i,j,2] - ehd_charge_feq(Val(2), q, ueqx, ueqy))
        f[i,j,3] -= ω * (f[i,j,3] - ehd_charge_feq(Val(3), q, ueqx, ueqy))
        f[i,j,4] -= ω * (f[i,j,4] - ehd_charge_feq(Val(4), q, ueqx, ueqy))
        f[i,j,5] -= ω * (f[i,j,5] - ehd_charge_feq(Val(5), q, ueqx, ueqy))
        f[i,j,6] -= ω * (f[i,j,6] - ehd_charge_feq(Val(6), q, ueqx, ueqy))
        f[i,j,7] -= ω * (f[i,j,7] - ehd_charge_feq(Val(7), q, ueqx, ueqy))
        f[i,j,8] -= ω * (f[i,j,8] - ehd_charge_feq(Val(8), q, ueqx, ueqy))
        f[i,j,9] -= ω * (f[i,j,9] - ehd_charge_feq(Val(9), q, ueqx, ueqy))
    end
end

function collide_electric_charge_srt_2d!(f, ux, uy, Ex, Ey, tau_q, K)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    kernel! = collide_electric_charge_srt_u_2d_kernel!(backend)
    kernel!(f, ux, uy, Ex, Ey, eltype(f)(tau_q), eltype(f)(K); ndrange=(Nx, Ny))
end

@kernel function collide_electric_charge_regularized_u_2d_kernel!(f, @Const(ux), @Const(uy),
                                                                  @Const(Ex), @Const(Ey),
                                                                  tau_q, K)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f)
        q = f[i,j,1] + f[i,j,2] + f[i,j,3] + f[i,j,4] +
            f[i,j,5] + f[i,j,6] + f[i,j,7] + f[i,j,8] + f[i,j,9]
        ueqx = ux[i, j] + K * Ex[i, j]
        ueqy = uy[i, j] + K * Ey[i, j]
        feq1 = ehd_charge_feq(Val(1), q, ueqx, ueqy)
        feq2 = ehd_charge_feq(Val(2), q, ueqx, ueqy)
        feq3 = ehd_charge_feq(Val(3), q, ueqx, ueqy)
        feq4 = ehd_charge_feq(Val(4), q, ueqx, ueqy)
        feq5 = ehd_charge_feq(Val(5), q, ueqx, ueqy)
        feq6 = ehd_charge_feq(Val(6), q, ueqx, ueqy)
        feq7 = ehd_charge_feq(Val(7), q, ueqx, ueqy)
        feq8 = ehd_charge_feq(Val(8), q, ueqx, ueqy)
        feq9 = ehd_charge_feq(Val(9), q, ueqx, ueqy)
        jx = (f[i,j,2] - feq2) - (f[i,j,4] - feq4) +
             (f[i,j,6] - feq6) - (f[i,j,7] - feq7) -
             (f[i,j,8] - feq8) + (f[i,j,9] - feq9)
        jy = (f[i,j,3] - feq3) - (f[i,j,5] - feq5) +
             (f[i,j,6] - feq6) + (f[i,j,7] - feq7) -
             (f[i,j,8] - feq8) - (f[i,j,9] - feq9)
        pref = one(T) - one(T) / tau_q
        f[i,j,1] = feq1
        f[i,j,2] = feq2 + pref * ehd_w(Val(2), T) * T(3) * jx
        f[i,j,3] = feq3 + pref * ehd_w(Val(3), T) * T(3) * jy
        f[i,j,4] = feq4 - pref * ehd_w(Val(4), T) * T(3) * jx
        f[i,j,5] = feq5 - pref * ehd_w(Val(5), T) * T(3) * jy
        f[i,j,6] = feq6 + pref * ehd_w(Val(6), T) * T(3) * (jx + jy)
        f[i,j,7] = feq7 + pref * ehd_w(Val(7), T) * T(3) * (-jx + jy)
        f[i,j,8] = feq8 + pref * ehd_w(Val(8), T) * T(3) * (-jx - jy)
        f[i,j,9] = feq9 + pref * ehd_w(Val(9), T) * T(3) * (jx - jy)
    end
end

function collide_electric_charge_regularized_2d!(f, ux, uy, Ex, Ey, tau_q, K)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    kernel! = collide_electric_charge_regularized_u_2d_kernel!(backend)
    kernel!(f, ux, uy, Ex, Ey, eltype(f)(tau_q), eltype(f)(K); ndrange=(Nx, Ny))
end

@inline function _ehd_apply_charge_nee!(f, qfield, ux, uy, Ex, Ey, K, ib, jb, inb, jnb, qb)
    @inbounds begin
        uxb = ux[ib, jb] + K * Ex[ib, jb]
        uyb = uy[ib, jb] + K * Ey[ib, jb]
        uxn = ux[inb, jnb] + K * Ex[inb, jnb]
        uyn = uy[inb, jnb] + K * Ey[inb, jnb]
        qn = qfield[inb, jnb]
        f[ib, jb, 1] = f[inb, jnb, 1] + ehd_charge_feq(Val(1), qb, uxb, uyb) - ehd_charge_feq(Val(1), qn, uxn, uyn)
        f[ib, jb, 2] = f[inb, jnb, 2] + ehd_charge_feq(Val(2), qb, uxb, uyb) - ehd_charge_feq(Val(2), qn, uxn, uyn)
        f[ib, jb, 3] = f[inb, jnb, 3] + ehd_charge_feq(Val(3), qb, uxb, uyb) - ehd_charge_feq(Val(3), qn, uxn, uyn)
        f[ib, jb, 4] = f[inb, jnb, 4] + ehd_charge_feq(Val(4), qb, uxb, uyb) - ehd_charge_feq(Val(4), qn, uxn, uyn)
        f[ib, jb, 5] = f[inb, jnb, 5] + ehd_charge_feq(Val(5), qb, uxb, uyb) - ehd_charge_feq(Val(5), qn, uxn, uyn)
        f[ib, jb, 6] = f[inb, jnb, 6] + ehd_charge_feq(Val(6), qb, uxb, uyb) - ehd_charge_feq(Val(6), qn, uxn, uyn)
        f[ib, jb, 7] = f[inb, jnb, 7] + ehd_charge_feq(Val(7), qb, uxb, uyb) - ehd_charge_feq(Val(7), qn, uxn, uyn)
        f[ib, jb, 8] = f[inb, jnb, 8] + ehd_charge_feq(Val(8), qb, uxb, uyb) - ehd_charge_feq(Val(8), qn, uxn, uyn)
        f[ib, jb, 9] = f[inb, jnb, 9] + ehd_charge_feq(Val(9), qb, uxb, uyb) - ehd_charge_feq(Val(9), qn, uxn, uyn)
    end
end

@kernel function apply_phi_nee_box_2d_kernel!(f, @Const(phi), phi_bottom, phi_top, Nx, Ny)
    k, = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f)
        if k <= Nx
            i = k
            if i > 1 && i < Nx
                for qdir in 1:9
                    wq = ifelse(qdir == 1, T(4)/T(9), ifelse(qdir <= 5, T(1)/T(9), T(1)/T(36)))
                    f[i, 1, qdir] = f[i, 2, qdir] + wq * (phi_bottom - phi[i, 2])
                    f[i, Ny, qdir] = f[i, Ny - 1, qdir] + wq * (phi_top - phi[i, Ny - 1])
                end
            end
        elseif k <= Nx + Ny
            j = k - Nx
            for qdir in 1:9
                wq = ifelse(qdir == 1, T(4)/T(9), ifelse(qdir <= 5, T(1)/T(9), T(1)/T(36)))
                f[1, j, qdir] = f[2, j, qdir] + wq * (phi[2, j] - phi[2, j])
                f[Nx, j, qdir] = f[Nx - 1, j, qdir] + wq * (phi[Nx - 1, j] - phi[Nx - 1, j])
            end
        end
    end
end

function apply_phi_nee_box_2d!(f, phi, phi_bottom, phi_top, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = apply_phi_nee_box_2d_kernel!(backend)
    kernel!(f, phi, eltype(f)(phi_bottom), eltype(f)(phi_top), Nx, Ny; ndrange=(Nx + Ny,))
end

@kernel function apply_charge_nee_box_2d_kernel!(f, @Const(qfield), @Const(ux), @Const(uy),
                                                 @Const(Ex), @Const(Ey), q_bottom,
                                                 grad_top, K, Nx, Ny)
    k, = @index(Global, NTuple)
    @inbounds begin
        if k <= Nx
            i = k
            if i > 1 && i < Nx
                _ehd_apply_charge_nee!(f, qfield, ux, uy, Ex, Ey, K, i, 1, i, 2, q_bottom)
                _ehd_apply_charge_nee!(f, qfield, ux, uy, Ex, Ey, K, i, Ny, i, Ny - 1,
                                       qfield[i, Ny - 1] - grad_top)
            end
        elseif k <= Nx + Ny
            j = k - Nx
            _ehd_apply_charge_nee!(f, qfield, ux, uy, Ex, Ey, K, 1, j, 2, j, qfield[2, j])
            _ehd_apply_charge_nee!(f, qfield, ux, uy, Ex, Ey, K, Nx, j, Nx - 1, j, qfield[Nx - 1, j])
        end
    end
end

function apply_charge_nee_box_2d!(f, qfield, ux, uy, Ex, Ey, q_bottom, grad_top, K, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = apply_charge_nee_box_2d_kernel!(backend)
    kernel!(f, qfield, ux, uy, Ex, Ey, eltype(f)(q_bottom), eltype(f)(grad_top),
            eltype(f)(K), Nx, Ny; ndrange=(Nx + Ny,))
end

@kernel function compute_coulomb_force_2d_kernel!(Fx, Fy, @Const(qfield), @Const(Ex), @Const(Ey), Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if j == 1 || j == Ny
            Fx[i, j] = zero(eltype(Fx))
            Fy[i, j] = zero(eltype(Fy))
        else
            Fx[i, j] = qfield[i, j] * Ex[i, j]
            Fy[i, j] = qfield[i, j] * Ey[i, j]
        end
    end
end

function compute_coulomb_force_2d!(Fx, Fy, qfield, Ex, Ey, Nx, Ny)
    backend = KernelAbstractions.get_backend(Fx)
    kernel! = compute_coulomb_force_2d_kernel!(backend)
    kernel!(Fx, Fy, qfield, Ex, Ey, Ny; ndrange=(Nx, Ny))
end

@kernel function project_coulomb_force_rows_2d_kernel!(Fx, Fy, @Const(is_solid), mode_code, Nx)
    j, = @index(Global, NTuple)
    @inbounds begin
        T = eltype(Fx)
        count = zero(Int)
        sx = zero(T)
        sy = zero(T)
        for i in 1:Nx
            if !is_solid[i, j]
                count += 1
                sx += Fx[i, j]
                sy += Fy[i, j]
            end
        end
        if count > 0
            inv_count = one(T) / T(count)
            mx = sx * inv_count
            my = sy * inv_count
            for i in 1:Nx
                if !is_solid[i, j]
                    mode_code == 1 && (Fx[i, j] -= mx)
                    mode_code <= 2 && (Fy[i, j] -= my)
                end
            end
        end
    end
end

function project_coulomb_force_rows_2d!(Fx, Fy, is_solid, mode_code, Nx, Ny)
    mode_code == 0 && return nothing
    backend = KernelAbstractions.get_backend(Fx)
    kernel! = project_coulomb_force_rows_2d_kernel!(backend)
    kernel!(Fx, Fy, is_solid, mode_code, Nx; ndrange=(Ny,))
    return nothing
end

@kernel function compute_macroscopic_guo_field_2d_kernel!(rho, ux, uy, @Const(f),
                                                          @Const(Fx), @Const(Fy), Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f)
        f1 = f[i,j,1]; f2 = f[i,j,2]; f3 = f[i,j,3]
        f4 = f[i,j,4]; f5 = f[i,j,5]; f6 = f[i,j,6]
        f7 = f[i,j,7]; f8 = f[i,j,8]; f9 = f[i,j,9]
        r = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
        rho[i, j] = r
        if j == 1 || j == Ny
            ux[i, j] = zero(T)
            uy[i, j] = zero(T)
        else
            invr = one(T) / r
            ux[i, j] = ((f2 - f4 + f6 - f7 - f8 + f9) + Fx[i, j] / T(2)) * invr
            uy[i, j] = ((f3 - f5 + f6 + f7 - f8 - f9) + Fy[i, j] / T(2)) * invr
        end
    end
end

function compute_macroscopic_guo_field_2d!(rho, ux, uy, f, Fx, Fy, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = compute_macroscopic_guo_field_2d_kernel!(backend)
    kernel!(rho, ux, uy, f, Fx, Fy, Ny; ndrange=(Nx, Ny))
end

@kernel function enforce_free_side_macros_2d_kernel!(ux, uy, Nx, Ny)
    k, = @index(Global, NTuple)
    j = k + 1
    @inbounds begin
        if j <= Ny - 1
            ux[1, j] = zero(eltype(ux))
            ux[Nx, j] = zero(eltype(ux))
            uy[1, j] = uy[2, j]
            uy[Nx, j] = uy[Nx - 1, j]
        end
    end
end

function enforce_free_side_macros_2d!(ux, uy, Nx, Ny)
    backend = KernelAbstractions.get_backend(ux)
    kernel! = enforce_free_side_macros_2d_kernel!(backend)
    kernel!(ux, uy, Nx, Ny; ndrange=(max(Ny - 2, 1),))
end

@kernel function apply_free_slip_sidewalls_2d_kernel!(f, Nx, Ny)
    k, = @index(Global, NTuple)
    j = k + 1
    @inbounds begin
        if j <= Ny - 1
            f[1, j, 2] = f[1, j, 4]
            f[1, j, 6] = f[1, j, 7]
            f[1, j, 9] = f[1, j, 8]
            f[Nx, j, 4] = f[Nx, j, 2]
            f[Nx, j, 7] = f[Nx, j, 6]
            f[Nx, j, 8] = f[Nx, j, 9]
        end
    end
end

function apply_free_slip_sidewalls_2d!(f, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = apply_free_slip_sidewalls_2d_kernel!(backend)
    kernel!(f, Nx, Ny; ndrange=(max(Ny - 2, 1),))
end

@kernel function apply_no_slip_sidewalls_2d_kernel!(f, @Const(Fx), @Const(Fy), Nx, Ny)
    k, = @index(Global, NTuple)
    j = k + 1
    @inbounds begin
        if j <= Ny - 1
            T = eltype(f)
            # Guo velocity is (sum(c*f) + F/2)/rho. At a stationary wall
            # the required RAW momentum is therefore J = -F/2, not zero.
            # The axial non-equilibrium reflection closes the three unknowns;
            # the remaining two equations impose both momentum components.
            jx = -Fx[1, j] / T(2)
            jy = -Fy[1, j] / T(2)
            shear = (f[1, j, 5] - f[1, j, 3]) / T(2)
            f[1, j, 2] = f[1, j, 4] + T(2) * jx / T(3)
            f[1, j, 6] = f[1, j, 8] + shear + jx / T(6) + jy / T(2)
            f[1, j, 9] = f[1, j, 7] - shear + jx / T(6) - jy / T(2)

            jx = -Fx[Nx, j] / T(2)
            jy = -Fy[Nx, j] / T(2)
            shear = (f[Nx, j, 5] - f[Nx, j, 3]) / T(2)
            f[Nx, j, 4] = f[Nx, j, 2] - T(2) * jx / T(3)
            f[Nx, j, 7] = f[Nx, j, 9] + shear - jx / T(6) + jy / T(2)
            f[Nx, j, 8] = f[Nx, j, 6] - shear - jx / T(6) - jy / T(2)
        end
    end
end

"""
    apply_no_slip_sidewalls_2d!(f, Fx, Fy, Nx, Ny)

Reconstruct the three incoming populations after streaming at each stationary
lateral wall. The wall nodes are i=1 and i=Nx (separation Nx-1), collide as
fluid, and satisfy sum(c*f)=-F/2 with the supplied force density. Only j=2:Ny-1
is touched: plate rows, including corners, retain their existing treatment.
Use the same force fields for reconstruction and subsequent Guo macroscopic
evaluation. There are no neighbour reads or cross-thread write dependencies.
"""
function apply_no_slip_sidewalls_2d!(f, Fx, Fy, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = apply_no_slip_sidewalls_2d_kernel!(backend)
    kernel!(f, Fx, Fy, Nx, Ny; ndrange=(max(Ny - 2, 1),))
end
