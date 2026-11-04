using KernelAbstractions

# D2Q9 EHD scalar kernels. Direction order matches Kraken.D2Q9 and Jiachen's
# MATLAB Lattice.m: rest, E, N, W, S, NE, NW, SW, SE.

@inline ehd_w(::Val{1}, T) = T(4) / T(9)
@inline ehd_w(::Val{2}, T) = T(1) / T(9)
@inline ehd_w(::Val{3}, T) = T(1) / T(9)
@inline ehd_w(::Val{4}, T) = T(1) / T(9)
@inline ehd_w(::Val{5}, T) = T(1) / T(9)
@inline ehd_w(::Val{6}, T) = T(1) / T(36)
@inline ehd_w(::Val{7}, T) = T(1) / T(36)
@inline ehd_w(::Val{8}, T) = T(1) / T(36)
@inline ehd_w(::Val{9}, T) = T(1) / T(36)

@inline ehd_cx(::Val{1}, T) = zero(T)
@inline ehd_cx(::Val{2}, T) = one(T)
@inline ehd_cx(::Val{3}, T) = zero(T)
@inline ehd_cx(::Val{4}, T) = -one(T)
@inline ehd_cx(::Val{5}, T) = zero(T)
@inline ehd_cx(::Val{6}, T) = one(T)
@inline ehd_cx(::Val{7}, T) = -one(T)
@inline ehd_cx(::Val{8}, T) = -one(T)
@inline ehd_cx(::Val{9}, T) = one(T)

@inline ehd_cy(::Val{1}, T) = zero(T)
@inline ehd_cy(::Val{2}, T) = zero(T)
@inline ehd_cy(::Val{3}, T) = one(T)
@inline ehd_cy(::Val{4}, T) = zero(T)
@inline ehd_cy(::Val{5}, T) = -one(T)
@inline ehd_cy(::Val{6}, T) = one(T)
@inline ehd_cy(::Val{7}, T) = one(T)
@inline ehd_cy(::Val{8}, T) = -one(T)
@inline ehd_cy(::Val{9}, T) = -one(T)

@inline function ehd_charge_feq(::Val{Q}, q, ux, uy) where {Q}
    T = typeof(q)
    cx = ehd_cx(Val(Q), T)
    cy = ehd_cy(Val(Q), T)
    cu3 = T(3) * (cx * ux + cy * uy)
    usq = ux * ux + uy * uy
    return q * ehd_w(Val(Q), T) * (one(T) + cu3 + T(0.5) * cu3 * cu3 - T(1.5) * usq)
end

@inline function ehd_phi_feq(::Val{Q}, phi) where {Q}
    return ehd_w(Val(Q), typeof(phi)) * phi
end

@kernel function compute_ehd_scalar_2d_kernel!(field, @Const(f))
    i, j = @index(Global, NTuple)
    @inbounds field[i, j] = f[i,j,1] + f[i,j,2] + f[i,j,3] + f[i,j,4] +
                            f[i,j,5] + f[i,j,6] + f[i,j,7] + f[i,j,8] + f[i,j,9]
end

function compute_ehd_scalar_2d!(field, f)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(field)
    kernel! = compute_ehd_scalar_2d_kernel!(backend)
    kernel!(field, f; ndrange=(Nx, Ny))
end

@kernel function ehd_rel_change_2d_kernel!(out, @Const(field), @Const(prev), Nx, Ny)
    k, = @index(Global, NTuple)
    @inbounds begin
        if k == 1
            T = eltype(field)
            maxdiff = zero(T)
            maxabs = zero(T)
            finite_flag = one(T)
            for j in 1:Ny, i in 1:Nx
                v = field[i, j]
                old = prev[i, j]
                d = abs(v - old)
                a = abs(v)
                maxdiff = max(maxdiff, d)
                maxabs = max(maxabs, a)
                finite_flag = ifelse(isfinite(v), finite_flag, zero(T))
            end
            out[1] = maxdiff / max(maxabs, floatmin(T))
            out[2] = finite_flag
        end
    end
end

function ehd_rel_change_2d!(out, field, prev, Nx, Ny)
    backend = KernelAbstractions.get_backend(field)
    kernel! = ehd_rel_change_2d_kernel!(backend)
    kernel!(out, field, prev, Nx, Ny; ndrange=(1,))
end

@kernel function ehd_field_change_2d_kernel!(out, @Const(Ex), @Const(Ey), @Const(Ex_prev),
                                             @Const(Ey_prev), E_ref, Nx, Ny, offset)
    k, = @index(Global, NTuple)
    @inbounds begin
        if k == 1
            T = eltype(Ex)
            maxdiff = zero(T)
            maxabs = zero(T)
            finite_flag = one(T)
            for j in 1:Ny, i in 1:Nx
                ex = Ex[i, j]
                ey = Ey[i, j]
                maxdiff = max(maxdiff, abs(ex - Ex_prev[i, j]), abs(ey - Ey_prev[i, j]))
                maxabs = max(maxabs, abs(ex), abs(ey))
                finite_flag = ifelse(isfinite(ex) & isfinite(ey), finite_flag, zero(T))
            end
            out[offset + 1] = maxdiff / max(maxabs, E_ref, floatmin(T))
            out[offset + 2] = finite_flag
        end
    end
end

"""
    ehd_field_change_2d!(out, Ex, Ey, Ex_prev, Ey_prev, E_ref, Nx, Ny, offset=0)

Relative change of the electric field `(Ex, Ey)` since `(Ex_prev, Ey_prev)`, reduced
by a single work item like `ehd_rel_change_2d!`:

- `out[offset + 1] = max|E - E_prev| / max(max|E|, E_ref, floatmin)`, where both
  maxima run over every node and both components;
- `out[offset + 2]` is `1` when every component of `E` is finite, `0` otherwise.

This is the change of `E` since the previous check, relative to `max(max|E|, E_ref)`,
with `E_ref = |phi_bottom − phi_top|/(Ny − 1)` the applied field (`1/(Ny − 1)` in all
public drivers). The applied field is a minimum scale: it applies while `E` is below
it everywhere, as at the first checks of a cold start, where `E` starts at 0. A
`floatmin` guard avoids a division by zero when the plates are at the same potential
and the field is zero. `E_ref` is converted to the element type. The caller computes
it; measured values of `max|E| / E_ref` at acceptance are in the docstring of
`ehd_phi_ddf_solve!`. No allocation beyond the kernel launch.
"""
function ehd_field_change_2d!(out, Ex, Ey, Ex_prev, Ey_prev, E_ref, Nx, Ny, offset=0)
    backend = KernelAbstractions.get_backend(Ex)
    kernel! = ehd_field_change_2d_kernel!(backend)
    kernel!(out, Ex, Ey, Ex_prev, Ey_prev, eltype(Ex)(E_ref), Nx, Ny, Int(offset); ndrange=(1,))
end

@kernel function ehd_maxspeed_2d_kernel!(out, @Const(ux), @Const(uy), Nx, Ny)
    k, = @index(Global, NTuple)
    @inbounds begin
        if k == 1
            T = eltype(ux)
            umax = zero(T)
            finite_flag = one(T)
            for j in 1:Ny, i in 1:Nx
                sx = ux[i, j]
                sy = uy[i, j]
                speed = sqrt(sx * sx + sy * sy)
                umax = max(umax, speed)
                finite_flag = ifelse(isfinite(speed), finite_flag, zero(T))
            end
            out[1] = umax
            out[2] = finite_flag
        end
    end
end

function ehd_maxspeed_2d!(out, ux, uy, Nx, Ny)
    backend = KernelAbstractions.get_backend(ux)
    kernel! = ehd_maxspeed_2d_kernel!(backend)
    kernel!(out, ux, uy, Nx, Ny; ndrange=(1,))
end

@kernel function collide_electric_potential_2d_kernel!(f, @Const(q), eps, ω_U, nu_U)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f)
        phi = f[i,j,1] + f[i,j,2] + f[i,j,3] + f[i,j,4] +
              f[i,j,5] + f[i,j,6] + f[i,j,7] + f[i,j,8] + f[i,j,9]
        source_scale = nu_U * q[i, j] / eps
        f[i,j,1] -= ω_U * (f[i,j,1] - ehd_phi_feq(Val(1), phi)) - ehd_w(Val(1), T) * source_scale
        f[i,j,2] -= ω_U * (f[i,j,2] - ehd_phi_feq(Val(2), phi)) - ehd_w(Val(2), T) * source_scale
        f[i,j,3] -= ω_U * (f[i,j,3] - ehd_phi_feq(Val(3), phi)) - ehd_w(Val(3), T) * source_scale
        f[i,j,4] -= ω_U * (f[i,j,4] - ehd_phi_feq(Val(4), phi)) - ehd_w(Val(4), T) * source_scale
        f[i,j,5] -= ω_U * (f[i,j,5] - ehd_phi_feq(Val(5), phi)) - ehd_w(Val(5), T) * source_scale
        f[i,j,6] -= ω_U * (f[i,j,6] - ehd_phi_feq(Val(6), phi)) - ehd_w(Val(6), T) * source_scale
        f[i,j,7] -= ω_U * (f[i,j,7] - ehd_phi_feq(Val(7), phi)) - ehd_w(Val(7), T) * source_scale
        f[i,j,8] -= ω_U * (f[i,j,8] - ehd_phi_feq(Val(8), phi)) - ehd_w(Val(8), T) * source_scale
        f[i,j,9] -= ω_U * (f[i,j,9] - ehd_phi_feq(Val(9), phi)) - ehd_w(Val(9), T) * source_scale
    end
end

function collide_electric_potential_2d!(f, q, eps, ω_U, nu_U)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    kernel! = collide_electric_potential_2d_kernel!(backend)
    kernel!(f, q, eltype(f)(eps), eltype(f)(ω_U), eltype(f)(nu_U); ndrange=(Nx, Ny))
end

@kernel function compute_electric_field_2d_kernel!(Ex, Ey, @Const(f), tau_U)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f)
        inv = one(T) / (tau_U * (one(T) / T(3)))
        Ex[i, j] = (f[i,j,2] - f[i,j,4] + f[i,j,6] - f[i,j,7] - f[i,j,8] + f[i,j,9]) * inv
        Ey[i, j] = (f[i,j,3] - f[i,j,5] + f[i,j,6] + f[i,j,7] - f[i,j,8] - f[i,j,9]) * inv
    end
end

function compute_electric_field_2d!(Ex, Ey, f, tau_U)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(Ex)
    kernel! = compute_electric_field_2d_kernel!(backend)
    kernel!(Ex, Ey, f, eltype(f)(tau_U); ndrange=(Nx, Ny))
end

@kernel function compute_electric_field_fd_2d_kernel!(Ex, Ey, @Const(phi), xbc_code, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(phi)
        if xbc_code == 1
            Ex[i, j] = (i == 1 || i == Nx) ? zero(T) :
                       -(phi[i + 1, j] - phi[i - 1, j]) / T(2)
        else
            im = i == 1 ? Nx : i - 1
            ip = i == Nx ? 1 : i + 1
            Ex[i, j] = -(phi[ip, j] - phi[im, j]) / T(2)
        end

        if j == 1
            Ey[i, j] = (T(3) * phi[i, 1] - T(4) * phi[i, 2] + phi[i, 3]) / T(2)
        elseif j == Ny
            Ey[i, j] = (-T(3) * phi[i, Ny] + T(4) * phi[i, Ny - 1] - phi[i, Ny - 2]) / T(2)
        else
            Ey[i, j] = -(phi[i, j + 1] - phi[i, j - 1]) / T(2)
        end
    end
end

"""
    compute_electric_field_fd_2d!(Ex, Ey, phi, xbc, Nx, Ny)

Recover `E = -grad(phi)` from the direct Poisson potential on the unit lattice.
The y-plates use second-order one-sided differences; x uses either mirror
Neumann (`Ex=0` on side nodes) or periodic central wrap.
"""
function compute_electric_field_fd_2d!(Ex, Ey, phi, xbc::Symbol, Nx, Ny)
    xbc in (:neumann, :periodic) ||
        throw(ArgumentError("xbc must be :neumann or :periodic."))
    backend = KernelAbstractions.get_backend(phi)
    kernel! = compute_electric_field_fd_2d_kernel!(backend)
    xbc_code = xbc === :neumann ? 1 : 2
    kernel!(Ex, Ey, phi, xbc_code, Nx, Ny; ndrange=(Nx, Ny))
end

@kernel function collide_electric_charge_srt_2d_kernel!(f, @Const(Ex), @Const(Ey), tau_q, K)
    i, j = @index(Global, NTuple)
    @inbounds begin
        q = f[i,j,1] + f[i,j,2] + f[i,j,3] + f[i,j,4] +
            f[i,j,5] + f[i,j,6] + f[i,j,7] + f[i,j,8] + f[i,j,9]
        ux = K * Ex[i, j]
        uy = K * Ey[i, j]
        ω = one(eltype(f)) / tau_q
        f[i,j,1] -= ω * (f[i,j,1] - ehd_charge_feq(Val(1), q, ux, uy))
        f[i,j,2] -= ω * (f[i,j,2] - ehd_charge_feq(Val(2), q, ux, uy))
        f[i,j,3] -= ω * (f[i,j,3] - ehd_charge_feq(Val(3), q, ux, uy))
        f[i,j,4] -= ω * (f[i,j,4] - ehd_charge_feq(Val(4), q, ux, uy))
        f[i,j,5] -= ω * (f[i,j,5] - ehd_charge_feq(Val(5), q, ux, uy))
        f[i,j,6] -= ω * (f[i,j,6] - ehd_charge_feq(Val(6), q, ux, uy))
        f[i,j,7] -= ω * (f[i,j,7] - ehd_charge_feq(Val(7), q, ux, uy))
        f[i,j,8] -= ω * (f[i,j,8] - ehd_charge_feq(Val(8), q, ux, uy))
        f[i,j,9] -= ω * (f[i,j,9] - ehd_charge_feq(Val(9), q, ux, uy))
    end
end

function collide_electric_charge_srt_2d!(f, Ex, Ey, tau_q, K)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    kernel! = collide_electric_charge_srt_2d_kernel!(backend)
    kernel!(f, Ex, Ey, eltype(f)(tau_q), eltype(f)(K); ndrange=(Nx, Ny))
end

@kernel function collide_electric_charge_regularized_2d_kernel!(f, @Const(Ex), @Const(Ey), tau_q, K)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f)
        q = f[i,j,1] + f[i,j,2] + f[i,j,3] + f[i,j,4] +
            f[i,j,5] + f[i,j,6] + f[i,j,7] + f[i,j,8] + f[i,j,9]
        ux = K * Ex[i, j]
        uy = K * Ey[i, j]
        feq1 = ehd_charge_feq(Val(1), q, ux, uy)
        feq2 = ehd_charge_feq(Val(2), q, ux, uy)
        feq3 = ehd_charge_feq(Val(3), q, ux, uy)
        feq4 = ehd_charge_feq(Val(4), q, ux, uy)
        feq5 = ehd_charge_feq(Val(5), q, ux, uy)
        feq6 = ehd_charge_feq(Val(6), q, ux, uy)
        feq7 = ehd_charge_feq(Val(7), q, ux, uy)
        feq8 = ehd_charge_feq(Val(8), q, ux, uy)
        feq9 = ehd_charge_feq(Val(9), q, ux, uy)
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

function collide_electric_charge_regularized_2d!(f, Ex, Ey, tau_q, K)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    kernel! = collide_electric_charge_regularized_2d_kernel!(backend)
    kernel!(f, Ex, Ey, eltype(f)(tau_q), eltype(f)(K); ndrange=(Nx, Ny))
end

@kernel function apply_phi_nee_walls_2d_kernel!(f, @Const(phi), phi_bottom, phi_top, Ny)
    i, = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f)
        for qdir in 1:9
            wq = ifelse(qdir == 1, T(4)/T(9), ifelse(qdir <= 5, T(1)/T(9), T(1)/T(36)))
            f[i, 1, qdir] = f[i, 2, qdir] + wq * (phi_bottom - phi[i, 2])
            f[i, Ny, qdir] = f[i, Ny - 1, qdir] + wq * (phi_top - phi[i, Ny - 1])
        end
    end
end

function apply_phi_nee_walls_2d!(f, phi, phi_bottom, phi_top, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = apply_phi_nee_walls_2d_kernel!(backend)
    kernel!(f, phi, eltype(f)(phi_bottom), eltype(f)(phi_top), Ny; ndrange=(Nx,))
end

@kernel function apply_charge_nee_walls_2d_kernel!(f, @Const(qfield), @Const(Ex), @Const(Ey),
                                                   q_bottom, grad_top, K, Ny)
    i, = @index(Global, NTuple)
    @inbounds begin
        qb = q_bottom
        qt = qfield[i, Ny - 1] - grad_top
        ux1 = K * Ex[i, 1]
        uy1 = K * Ey[i, 1]
        ux2 = K * Ex[i, 2]
        uy2 = K * Ey[i, 2]
        uxt = K * Ex[i, Ny]
        uyt = K * Ey[i, Ny]
        uxn = K * Ex[i, Ny - 1]
        uyn = K * Ey[i, Ny - 1]
        f[i, 1, 1] = f[i, 2, 1] + ehd_charge_feq(Val(1), qb, ux1, uy1) - ehd_charge_feq(Val(1), qfield[i, 2], ux2, uy2)
        f[i, 1, 2] = f[i, 2, 2] + ehd_charge_feq(Val(2), qb, ux1, uy1) - ehd_charge_feq(Val(2), qfield[i, 2], ux2, uy2)
        f[i, 1, 3] = f[i, 2, 3] + ehd_charge_feq(Val(3), qb, ux1, uy1) - ehd_charge_feq(Val(3), qfield[i, 2], ux2, uy2)
        f[i, 1, 4] = f[i, 2, 4] + ehd_charge_feq(Val(4), qb, ux1, uy1) - ehd_charge_feq(Val(4), qfield[i, 2], ux2, uy2)
        f[i, 1, 5] = f[i, 2, 5] + ehd_charge_feq(Val(5), qb, ux1, uy1) - ehd_charge_feq(Val(5), qfield[i, 2], ux2, uy2)
        f[i, 1, 6] = f[i, 2, 6] + ehd_charge_feq(Val(6), qb, ux1, uy1) - ehd_charge_feq(Val(6), qfield[i, 2], ux2, uy2)
        f[i, 1, 7] = f[i, 2, 7] + ehd_charge_feq(Val(7), qb, ux1, uy1) - ehd_charge_feq(Val(7), qfield[i, 2], ux2, uy2)
        f[i, 1, 8] = f[i, 2, 8] + ehd_charge_feq(Val(8), qb, ux1, uy1) - ehd_charge_feq(Val(8), qfield[i, 2], ux2, uy2)
        f[i, 1, 9] = f[i, 2, 9] + ehd_charge_feq(Val(9), qb, ux1, uy1) - ehd_charge_feq(Val(9), qfield[i, 2], ux2, uy2)
        f[i, Ny, 1] = f[i, Ny - 1, 1] + ehd_charge_feq(Val(1), qt, uxt, uyt) - ehd_charge_feq(Val(1), qfield[i, Ny - 1], uxn, uyn)
        f[i, Ny, 2] = f[i, Ny - 1, 2] + ehd_charge_feq(Val(2), qt, uxt, uyt) - ehd_charge_feq(Val(2), qfield[i, Ny - 1], uxn, uyn)
        f[i, Ny, 3] = f[i, Ny - 1, 3] + ehd_charge_feq(Val(3), qt, uxt, uyt) - ehd_charge_feq(Val(3), qfield[i, Ny - 1], uxn, uyn)
        f[i, Ny, 4] = f[i, Ny - 1, 4] + ehd_charge_feq(Val(4), qt, uxt, uyt) - ehd_charge_feq(Val(4), qfield[i, Ny - 1], uxn, uyn)
        f[i, Ny, 5] = f[i, Ny - 1, 5] + ehd_charge_feq(Val(5), qt, uxt, uyt) - ehd_charge_feq(Val(5), qfield[i, Ny - 1], uxn, uyn)
        f[i, Ny, 6] = f[i, Ny - 1, 6] + ehd_charge_feq(Val(6), qt, uxt, uyt) - ehd_charge_feq(Val(6), qfield[i, Ny - 1], uxn, uyn)
        f[i, Ny, 7] = f[i, Ny - 1, 7] + ehd_charge_feq(Val(7), qt, uxt, uyt) - ehd_charge_feq(Val(7), qfield[i, Ny - 1], uxn, uyn)
        f[i, Ny, 8] = f[i, Ny - 1, 8] + ehd_charge_feq(Val(8), qt, uxt, uyt) - ehd_charge_feq(Val(8), qfield[i, Ny - 1], uxn, uyn)
        f[i, Ny, 9] = f[i, Ny - 1, 9] + ehd_charge_feq(Val(9), qt, uxt, uyt) - ehd_charge_feq(Val(9), qfield[i, Ny - 1], uxn, uyn)
    end
end

function apply_charge_nee_walls_2d!(f, qfield, Ex, Ey, q_bottom, grad_top, K, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = apply_charge_nee_walls_2d_kernel!(backend)
    kernel!(f, qfield, Ex, Ey, eltype(f)(q_bottom), eltype(f)(grad_top), eltype(f)(K), Ny; ndrange=(Nx,))
end

# Experimental static-only closure, issue #73. Do not use for a transverse or
# hydrodynamically coupled problem: the tangential nonequilibrium moment is zero.
@inline function _ehd_static_injector_population(::Val{Q}, qb, v, outgoing) where {Q}
    T = typeof(qb)
    jy = T(2) * (qb * (one(T) - T(3)*v + T(3)*v*v) / T(6) - outgoing)
    return ehd_charge_feq(Val(Q), qb, zero(T), v) +
           T(3) * ehd_w(Val(Q), T) * ehd_cy(Val(Q), T) * jy
end

@kernel function _apply_charge_static_moment_walls_kernel!(f, @Const(qfield),
                                                            @Const(Ex), @Const(Ey),
                                                            qb, K, Ny)
    i, = @index(Global, NTuple)
    @inbounds begin
        # Read the post-stream outgoing aggregate before overwriting any row.
        outgoing = f[i,1,5] + f[i,1,8] + f[i,1,9]
        v = K * Ey[i,1]
        f[i,1,1] = _ehd_static_injector_population(Val(1), qb, v, outgoing)
        f[i,1,2] = _ehd_static_injector_population(Val(2), qb, v, outgoing)
        f[i,1,3] = _ehd_static_injector_population(Val(3), qb, v, outgoing)
        f[i,1,4] = _ehd_static_injector_population(Val(4), qb, v, outgoing)
        f[i,1,5] = _ehd_static_injector_population(Val(5), qb, v, outgoing)
        f[i,1,6] = _ehd_static_injector_population(Val(6), qb, v, outgoing)
        f[i,1,7] = _ehd_static_injector_population(Val(7), qb, v, outgoing)
        f[i,1,8] = _ehd_static_injector_population(Val(8), qb, v, outgoing)
        f[i,1,9] = _ehd_static_injector_population(Val(9), qb, v, outgoing)
        # Same zero-gradient collector NEE expressions as the legacy kernel.
        qt = qfield[i,Ny-1]
        uxt = K*Ex[i,Ny]; uyt = K*Ey[i,Ny]
        uxn = K*Ex[i,Ny-1]; uyn = K*Ey[i,Ny-1]
        f[i,Ny,1] = f[i,Ny-1,1] + ehd_charge_feq(Val(1),qt,uxt,uyt) - ehd_charge_feq(Val(1),qt,uxn,uyn)
        f[i,Ny,2] = f[i,Ny-1,2] + ehd_charge_feq(Val(2),qt,uxt,uyt) - ehd_charge_feq(Val(2),qt,uxn,uyn)
        f[i,Ny,3] = f[i,Ny-1,3] + ehd_charge_feq(Val(3),qt,uxt,uyt) - ehd_charge_feq(Val(3),qt,uxn,uyn)
        f[i,Ny,4] = f[i,Ny-1,4] + ehd_charge_feq(Val(4),qt,uxt,uyt) - ehd_charge_feq(Val(4),qt,uxn,uyn)
        f[i,Ny,5] = f[i,Ny-1,5] + ehd_charge_feq(Val(5),qt,uxt,uyt) - ehd_charge_feq(Val(5),qt,uxn,uyn)
        f[i,Ny,6] = f[i,Ny-1,6] + ehd_charge_feq(Val(6),qt,uxt,uyt) - ehd_charge_feq(Val(6),qt,uxn,uyn)
        f[i,Ny,7] = f[i,Ny-1,7] + ehd_charge_feq(Val(7),qt,uxt,uyt) - ehd_charge_feq(Val(7),qt,uxn,uyn)
        f[i,Ny,8] = f[i,Ny-1,8] + ehd_charge_feq(Val(8),qt,uxt,uyt) - ehd_charge_feq(Val(8),qt,uxn,uyn)
        f[i,Ny,9] = f[i,Ny-1,9] + ehd_charge_feq(Val(9),qt,uxt,uyt) - ehd_charge_feq(Val(9),qt,uxn,uyn)
    end
end

# Internal seam for the flat, periodic-x, u=0 stationary-injection driver only.
function _apply_charge_static_moment_walls!(f, qfield, Ex, Ey, qb, K, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    _apply_charge_static_moment_walls_kernel!(backend)(
        f, qfield, Ex, Ey, eltype(f)(qb), eltype(f)(K), Ny; ndrange=(Nx,))
end
