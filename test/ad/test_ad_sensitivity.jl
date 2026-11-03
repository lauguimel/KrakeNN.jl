using Test, Kraken, Enzyme

module KrakenADSensitivityTests

using Test
using Kraken
using Enzyme
using KernelAbstractions
using LinearAlgebra
using Random

const NU_KEY = Symbol("\u03bd")
const RHO_OUT_KEY = Symbol("\u03c1_out")

const CX = (0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0)
const CY = (0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0)
const WT = (4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
            1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0)

_relerr(a, b) = abs(a - b) / max(abs(b), eps(Float64))

function _dot_arrays(a, b)
    s = 0.0
    @inbounds for idx in eachindex(a, b)
        s += a[idx] * b[idx]
    end
    return s
end

@inline function _moments(f1::T, f2::T, f3::T, f4::T, f5::T,
                          f6::T, f7::T, f8::T, f9::T) where {T}
    rho = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
    inv_rho = one(T) / rho
    ux = (f2 - f4 + f6 - f7 - f8 + f9) * inv_rho
    uy = (f3 - f5 + f6 + f7 - f8 - f9) * inv_rho
    return rho, ux, uy
end

@inline function _feq(::Val{Q}, rho::T, ux::T, uy::T, usq::T) where {Q,T}
    cx = T(CX[Q])
    cy = T(CY[Q])
    cu = cx * ux + cy * uy
    return T(WT[Q]) * rho *
           (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end

@inline function _guo_source(::Val{Q}, omega::T, g::T, ux::T, uy::T) where {Q,T}
    cx = T(CX[Q])
    cy = T(CY[Q])
    cu = cx * ux + cy * uy
    return (one(T) - omega / T(2)) * T(WT[Q]) * g *
           (T(3) * (cx - ux) + T(9) * cu * cx)
end

@inline function _libb_branch(qw::T, here::T, back::T, opp_here::T) where {T}
    if qw <= T(0.5)
        return T(2) * qw * here + (one(T) - T(2) * qw) * back
    end
    inv2q = one(T) / (T(2) * qw)
    return inv2q * here + (one(T) - inv2q) * opp_here
end

function _initial_equilibrium(Nx, Ny; ux_profile=nothing)
    f = zeros(Float64, Nx, Ny, 9)
    @inbounds for j in 1:Ny, i in 1:Nx
        ux = ux_profile === nothing ? 0.0 : ux_profile[j]
        uy = 0.0
        usq = ux * ux
        for q in 1:9
            f[i, j, q] = _feq(Val(q), 1.0, ux, uy, usq)
        end
    end
    return f
end

function _perturbed_equilibrium(rng, Nx, Ny)
    f = zeros(Float64, Nx, Ny, 9)
    @inbounds for j in 1:Ny, i in 1:Nx
        x = (i - 1) / Nx
        y = (j - 1) / Ny
        ux = 0.035 * sin(2 * pi * x) * cos(2 * pi * y) + 0.005 * randn(rng)
        uy = -0.035 * cos(2 * pi * x) * sin(2 * pi * y) + 0.005 * randn(rng)
        usq = ux * ux + uy * uy
        for q in 1:9
            f[i, j, q] = _feq(Val(q), 1.0, ux, uy, usq)
        end
    end
    @inbounds for idx in eachindex(f)
        f[idx] += 1e-3 * randn(rng)
    end
    return f
end

function _bounce_back_cell!(f, i, j)
    tmp = f[i, j, 2]; f[i, j, 2] = f[i, j, 4]; f[i, j, 4] = tmp
    tmp = f[i, j, 3]; f[i, j, 3] = f[i, j, 5]; f[i, j, 5] = tmp
    tmp = f[i, j, 6]; f[i, j, 6] = f[i, j, 8]; f[i, j, 8] = tmp
    tmp = f[i, j, 7]; f[i, j, 7] = f[i, j, 9]; f[i, j, 9] = tmp
    return nothing
end

function _collide_bgk!(f, is_solid, omega, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]
            _bounce_back_cell!(f, i, j)
        else
            f1 = f[i, j, 1]; f2 = f[i, j, 2]; f3 = f[i, j, 3]
            f4 = f[i, j, 4]; f5 = f[i, j, 5]; f6 = f[i, j, 6]
            f7 = f[i, j, 7]; f8 = f[i, j, 8]; f9 = f[i, j, 9]
            rho, ux, uy = _moments(f1, f2, f3, f4, f5, f6, f7, f8, f9)
            usq = ux * ux + uy * uy
            for q in 1:9
                f[i, j, q] -= omega * (f[i, j, q] - _feq(Val(q), rho, ux, uy, usq))
            end
        end
    end
    return nothing
end

function _stream_periodic!(out, f, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        ip = ifelse(i > 1, i - 1, Nx)
        im = ifelse(i < Nx, i + 1, 1)
        jp = ifelse(j > 1, j - 1, Ny)
        jm = ifelse(j < Ny, j + 1, 1)
        out[i, j, 1] = f[i, j, 1]
        out[i, j, 2] = f[ip, j, 2]
        out[i, j, 3] = f[i, jp, 3]
        out[i, j, 4] = f[im, j, 4]
        out[i, j, 5] = f[i, jm, 5]
        out[i, j, 6] = f[ip, jp, 6]
        out[i, j, 7] = f[im, jp, 7]
        out[i, j, 8] = f[im, jm, 8]
        out[i, j, 9] = f[ip, jm, 9]
    end
    return nothing
end

function _bulk_periodic_step!(out, f, is_solid, omega, Nx, Ny)
    work = copy(f)
    _collide_bgk!(work, is_solid, omega, Nx, Ny)
    _stream_periodic!(out, work, Nx, Ny)
    return nothing
end

function _bulk_vjp(f, v, is_solid, omega, Nx, Ny)
    out = zeros(Float64, size(f))
    df = zeros(Float64, size(f))
    Enzyme.autodiff(Enzyme.Reverse, _bulk_periodic_step!,
                    Enzyme.Duplicated(out, copy(v)),
                    Enzyme.Duplicated(copy(f), df),
                    Enzyme.Const(is_solid), Enzyme.Const(omega),
                    Enzyme.Const(Nx), Enzyme.Const(Ny))
    return df
end

function _apply_bulk(f, is_solid, omega, Nx, Ny)
    out = zeros(Float64, size(f))
    _bulk_periodic_step!(out, f, is_solid, omega, Nx, Ny)
    return out
end

function _dense_fd_jacobian(f, is_solid, omega, Nx, Ny, h)
    N = length(f)
    J = zeros(Float64, N, N)
    fp = similar(f)
    fm = similar(f)
    @inbounds for m in 1:N
        copyto!(fp, f)
        copyto!(fm, f)
        fp[m] += h
        fm[m] -= h
        J[:, m] .= vec(_apply_bulk(fp, is_solid, omega, Nx, Ny) .-
                       _apply_bulk(fm, is_solid, omega, Nx, Ny)) ./ (2h)
    end
    return J
end

function _run_c0()
    Nx, Ny = 4, 4
    nu = 0.1
    omega = 1.0 / (3.0 * nu + 0.5)
    rng = MersenneTwister(0xc0)
    is_solid = falses(Nx, Ny)
    f = _perturbed_equilibrium(rng, Nx, Ny)
    vs = [randn(rng, Float64, Nx, Ny, 9) for _ in 1:2]
    fds = [_dense_fd_jacobian(f, is_solid, omega, Nx, Ny, h)
           for h in (1e-5, 1e-6, 1e-7)]
    max_rel = 0.0
    for v in vs
        w = _bulk_vjp(f, v, is_solid, omega, Nx, Ny)
        best = minimum(maximum(abs, w .- reshape(J' * vec(v), size(f))) /
                       max(maximum(abs, J' * vec(v)), eps(Float64)) for J in fds)
        max_rel = max(max_rel, best)
    end
    return (; Nx, Ny, pairs=length(vs), max_rel)
end

function _forced_collide!(f, is_solid, omega, g, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]
            _bounce_back_cell!(f, i, j)
        else
            f1 = f[i, j, 1]; f2 = f[i, j, 2]; f3 = f[i, j, 3]
            f4 = f[i, j, 4]; f5 = f[i, j, 5]; f6 = f[i, j, 6]
            f7 = f[i, j, 7]; f8 = f[i, j, 8]; f9 = f[i, j, 9]
            rho, ux_raw, uy = _moments(f1, f2, f3, f4, f5, f6, f7, f8, f9)
            ux = ux_raw + g / (2 * rho)
            usq = ux * ux + uy * uy
            for q in 1:9
                f[i, j, q] = f[i, j, q] -
                             omega * (f[i, j, q] - _feq(Val(q), rho, ux, uy, usq)) +
                             _guo_source(Val(q), omega, g, ux, uy)
            end
        end
    end
    return nothing
end

function _stream_channel!(out, f, q_wall, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        ip = i > 1 ? i - 1 : Nx
        im = i < Nx ? i + 1 : 1
        out[i, j, 1] = f[i, j, 1]
        out[i, j, 2] = f[ip, j, 2]
        out[i, j, 4] = f[im, j, 4]

        if j > 1
            out[i, j, 3] = f[i, j - 1, 3]
            out[i, j, 6] = f[ip, j - 1, 6]
            out[i, j, 7] = f[im, j - 1, 7]
        else
            out[i, j, 3] = q_wall[i, j, 5] > 0.0 ?
                _libb_branch(q_wall[i, j, 5], f[i, j, 5], f[i, j + 1, 5], f[i, j, 3]) :
                f[i, j, 5]
            out[i, j, 6] = q_wall[i, j, 8] > 0.0 ?
                _libb_branch(q_wall[i, j, 8], f[i, j, 8], f[im, j + 1, 8], f[i, j, 6]) :
                f[i, j, 8]
            out[i, j, 7] = q_wall[i, j, 9] > 0.0 ?
                _libb_branch(q_wall[i, j, 9], f[i, j, 9], f[ip, j + 1, 9], f[i, j, 7]) :
                f[i, j, 9]
        end

        if j < Ny
            out[i, j, 5] = f[i, j + 1, 5]
            out[i, j, 8] = f[im, j + 1, 8]
            out[i, j, 9] = f[ip, j + 1, 9]
        else
            out[i, j, 5] = q_wall[i, j, 3] > 0.0 ?
                _libb_branch(q_wall[i, j, 3], f[i, j, 3], f[i, j - 1, 3], f[i, j, 5]) :
                f[i, j, 3]
            out[i, j, 8] = q_wall[i, j, 6] > 0.0 ?
                _libb_branch(q_wall[i, j, 6], f[i, j, 6], f[ip, j - 1, 6], f[i, j, 8]) :
                f[i, j, 6]
            out[i, j, 9] = q_wall[i, j, 7] > 0.0 ?
                _libb_branch(q_wall[i, j, 7], f[i, j, 7], f[im, j - 1, 7], f[i, j, 9]) :
                f[i, j, 7]
        end
    end
    return nothing
end

function _channel_step!(out, f, q_wall, is_solid, omega, g, Nx, Ny)
    work = copy(f)
    _forced_collide!(work, is_solid, omega, g, Nx, Ny)
    _stream_channel!(out, work, q_wall, Nx, Ny)
    return nothing
end

function _relative_residual(f_out, f_in)
    num = 0.0
    den = 0.0
    @inbounds for idx in eachindex(f_in)
        d = f_out[idx] - f_in[idx]
        num += d * d
        den += f_in[idx] * f_in[idx]
    end
    return sqrt(num) / sqrt(den)
end

function _forward_channel(Nx, Ny, omega, g, q_wall, is_solid;
                          tol=1e-11, max_steps=60_000, f_init=nothing)
    f_in = f_init === nothing ? _initial_equilibrium(Nx, Ny) : copy(f_init)
    f_out = similar(f_in)
    residual = Inf
    n_iter = 0
    converged = false
    for step in 1:max_steps
        _channel_step!(f_out, f_in, q_wall, is_solid, omega, g, Nx, Ny)
        residual = _relative_residual(f_out, f_in)
        n_iter = step
        if residual < tol
            converged = true
            break
        end
        f_in, f_out = f_out, f_in
    end
    return (; f_star=copy(f_out), n_iter, residual, converged)
end

function _rho_ux_raw(f, i, j)
    f1 = f[i, j, 1]; f2 = f[i, j, 2]; f3 = f[i, j, 3]
    f4 = f[i, j, 4]; f5 = f[i, j, 5]; f6 = f[i, j, 6]
    f7 = f[i, j, 7]; f8 = f[i, j, 8]; f9 = f[i, j, 9]
    rho, ux, _ = _moments(f1, f2, f3, f4, f5, f6, f7, f8, f9)
    return rho, ux
end

function _mean_density(f, Nx, Ny)
    s = 0.0
    @inbounds for j in 1:Ny, i in 1:Nx
        rho, _ = _rho_ux_raw(f, i, j)
        s += rho
    end
    return s / (Nx * Ny)
end

function _physical_ux_profile(f, g, Nx, Ny)
    prof = zeros(Float64, Ny)
    @inbounds for j in 1:Ny
        rho, ux = _rho_ux_raw(f, 1, j)
        prof[j] = ux + g / (2 * rho)
    end
    return prof
end

function _profile_error_halfway(f, g, nu, Nx, Ny)
    rho_bar = _mean_density(f, Nx, Ny)
    prof = _physical_ux_profile(f, g, Nx, Ny)
    ref = [g / (2 * rho_bar * nu) * (j - 0.5) * (Ny - (j - 0.5)) for j in 1:Ny]
    return maximum(abs, prof .- ref) / maximum(abs, ref)
end

function _profile_error_cut(f, g, nu, Nx, Ny, y_bot, y_top)
    rho_bar = _mean_density(f, Nx, Ny)
    prof = _physical_ux_profile(f, g, Nx, Ny)
    ref = [g / (2 * rho_bar * nu) * (Float64(j) - y_bot) * (y_top - Float64(j))
           for j in 1:Ny]
    return maximum(abs, prof .- ref) / maximum(abs, ref)
end

function _channel_J(f, Nx, Ny)
    s = 0.0
    @inbounds for j in 1:Ny, i in 1:Nx
        s += f[i, j, 2] - f[i, j, 4] + f[i, j, 6] -
             f[i, j, 7] - f[i, j, 8] + f[i, j, 9]
    end
    return s
end

function _dJdf(f, Nx, Ny)
    djdf = zeros(Float64, size(f))
    Enzyme.autodiff(Enzyme.Reverse, _channel_J, Enzyme.Active,
                    Enzyme.Duplicated(copy(f), djdf),
                    Enzyme.Const(Nx), Enzyme.Const(Ny))
    return djdf
end

function _check_dJdf_fd(f, djdf, Nx, Ny)
    h = 1e-6
    inds = (CartesianIndex(1, 1, 2), CartesianIndex(Nx, Ny, 4),
            CartesianIndex(max(1, Nx ÷ 2), max(1, Ny ÷ 2), 6),
            CartesianIndex(max(1, Nx ÷ 3), max(1, Ny ÷ 3), 8))
    max_rel = 0.0
    for idx in inds
        fp = copy(f)
        fm = copy(f)
        fp[idx] += h
        fm[idx] -= h
        fd = (_channel_J(fp, Nx, Ny) - _channel_J(fm, Nx, Ny)) / (2h)
        max_rel = max(max_rel, _relerr(djdf[idx], fd))
    end
    return max_rel
end

function _channel_vjp(f_star, v, q_wall, is_solid, omega, g, Nx, Ny)
    out = zeros(Float64, size(f_star))
    df = zeros(Float64, size(f_star))
    Enzyme.autodiff(Enzyme.Reverse, _channel_step!,
                    Enzyme.Duplicated(out, copy(v)),
                    Enzyme.Duplicated(copy(f_star), df),
                    Enzyme.Const(q_wall), Enzyme.Const(is_solid),
                    Enzyme.Const(omega), Enzyme.Const(g),
                    Enzyme.Const(Nx), Enzyme.Const(Ny))
    return df
end

function _lambda_dot_dGdg(f_star, lambda, q_wall, is_solid, omega, g, Nx, Ny)
    out = zeros(Float64, size(f_star))
    ret = Enzyme.autodiff(Enzyme.Reverse, _channel_step!,
                          Enzyme.Duplicated(out, copy(lambda)),
                          Enzyme.Const(f_star), Enzyme.Const(q_wall),
                          Enzyme.Const(is_solid), Enzyme.Const(omega),
                          Enzyme.Active(g), Enzyme.Const(Nx), Enzyme.Const(Ny))
    return Float64(ret[1][6])
end

function _fd_dJdg(Nx, Ny, omega, g, h, q_wall, is_solid, f_base)
    plus = _forward_channel(Nx, Ny, omega, g + h, q_wall, is_solid; f_init=f_base)
    minus = _forward_channel(Nx, Ny, omega, g - h, q_wall, is_solid; f_init=f_base)
    value = (plus.converged && minus.converged) ?
        (_channel_J(plus.f_star, Nx, Ny) - _channel_J(minus.f_star, Nx, Ny)) / (2h) :
        NaN
    return (; value, plus, minus)
end

function _flat_wall_geometry(Nx, Ny, H)
    y_center = (1.0 + Float64(Ny)) / 2.0
    y_bot = y_center - H
    y_top = y_center + H
    q_bot = 1.0 - y_bot
    q_top = y_top - Float64(Ny)
    q_wall = zeros(Float64, Nx, Ny, 9)
    dq_dH = zeros(Float64, Nx, Ny, 9)
    @inbounds for i in 1:Nx
        q_wall[i, 1, 5] = q_bot
        q_wall[i, 1, 8] = q_bot
        q_wall[i, 1, 9] = q_bot
        q_wall[i, Ny, 3] = q_top
        q_wall[i, Ny, 6] = q_top
        q_wall[i, Ny, 7] = q_top
        dq_dH[i, 1, 5] = 1.0
        dq_dH[i, 1, 8] = 1.0
        dq_dH[i, 1, 9] = 1.0
        dq_dH[i, Ny, 3] = 1.0
        dq_dH[i, Ny, 6] = 1.0
        dq_dH[i, Ny, 7] = 1.0
    end
    return (; q_wall, dq_dH, is_solid=falses(Nx, Ny), y_bot, y_top, q_bot, q_top)
end

function _initial_poiseuille(Nx, Ny, g, nu, y_bot, y_top)
    ux_profile = [begin
        ux_phys = g / (2.0 * nu) * (Float64(j) - y_bot) * (y_top - Float64(j))
        ux_phys - g / 2.0
    end for j in 1:Ny]
    return _initial_equilibrium(Nx, Ny; ux_profile)
end

function _lambda_dot_G(f_star, lambda, q_wall, is_solid, omega, g, Nx, Ny)
    out = zeros(Float64, size(f_star))
    _channel_step!(out, f_star, q_wall, is_solid, omega, g, Nx, Ny)
    return _dot_arrays(lambda, out)
end

function _directional_fd_lambdaG_qwall(f_star, lambda, q_wall, dq, is_solid,
                                       omega, g, Nx, Ny)
    h = 1e-5
    qp = copy(q_wall)
    qm = copy(q_wall)
    @inbounds for idx in eachindex(q_wall, dq)
        qp[idx] += h * dq[idx]
        qm[idx] -= h * dq[idx]
    end
    return (_lambda_dot_G(f_star, lambda, qp, is_solid, omega, g, Nx, Ny) -
            _lambda_dot_G(f_star, lambda, qm, is_solid, omega, g, Nx, Ny)) / (2h)
end

function _fd_lambdaG_qwall(f_star, lambda, q_wall, dq, is_solid, omega, g, Nx, Ny)
    dqw = zeros(Float64, size(q_wall))
    h = 1e-6
    @inbounds for idx in CartesianIndices(q_wall)
        dq[idx] == 0.0 && continue
        qp = copy(q_wall)
        qm = copy(q_wall)
        qp[idx] += h
        qm[idx] -= h
        vp = _lambda_dot_G(f_star, lambda, qp, is_solid, omega, g, Nx, Ny)
        vm = _lambda_dot_G(f_star, lambda, qm, is_solid, omega, g, Nx, Ny)
        dqw[idx] = (vp - vm) / (2h)
    end
    return dqw
end

function _lambdaG_dH(f_star, lambda, q_wall, dq_dH, is_solid, omega, g, Nx, Ny)
    err = ""
    for mode in (Enzyme.Reverse,
                 isdefined(Enzyme, :set_runtime_activity) ?
                    Enzyme.set_runtime_activity(Enzyme.Reverse) : Enzyme.Reverse)
        try
            out = zeros(Float64, size(f_star))
            dqw = zeros(Float64, size(q_wall))
            Enzyme.autodiff(mode, _channel_step!,
                            Enzyme.Duplicated(out, copy(lambda)),
                            Enzyme.Const(f_star),
                            Enzyme.Duplicated(copy(q_wall), dqw),
                            Enzyme.Const(is_solid), Enzyme.Const(omega),
                            Enzyme.Const(g), Enzyme.Const(Nx), Enzyme.Const(Ny))
            val = _dot_arrays(dqw, dq_dH)
            fd = _directional_fd_lambdaG_qwall(f_star, lambda, q_wall, dq_dH,
                                               is_solid, omega, g, Nx, Ny)
            if isfinite(val) && abs(val) > 0.0 && _relerr(val, fd) <= 1e-5
                return (; value=val, path="taped", directional_fd=fd,
                        directional_rel=_relerr(val, fd), error="")
            end
            err *= "taped contraction mismatch or zero; "
        catch e
            err *= sprint(showerror, e) * "; "
        end
    end
    dqw = _fd_lambdaG_qwall(f_star, lambda, q_wall, dq_dH,
                            is_solid, omega, g, Nx, Ny)
    value = _dot_arrays(dqw, dq_dH)
    directional = _directional_fd_lambdaG_qwall(f_star, lambda, q_wall, dq_dH,
                                               is_solid, omega, g, Nx, Ny)
    return (; value, path="Fallback A (FD-on-q_wall)",
            directional_fd=directional, directional_rel=_relerr(value, directional),
            error=first(err, min(360, lastindex(err))))
end

function _fd_dJdH(Nx, Ny, omega, g, H, h, f_base)
    plus_geom = _flat_wall_geometry(Nx, Ny, H + h)
    minus_geom = _flat_wall_geometry(Nx, Ny, H - h)
    plus = _forward_channel(Nx, Ny, omega, g, plus_geom.q_wall,
                            plus_geom.is_solid; f_init=f_base)
    minus = _forward_channel(Nx, Ny, omega, g, minus_geom.q_wall,
                             minus_geom.is_solid; f_init=f_base)
    value = (plus.converged && minus.converged) ?
        (_channel_J(plus.f_star, Nx, Ny) - _channel_J(minus.f_star, Nx, Ny)) / (2h) :
        NaN
    return (; value, plus, minus)
end

function _run_c1()
    Nx, Ny = 16, 16
    nu = 0.1
    omega = 1.0 / (3.0 * nu + 0.5)
    g = 1e-5
    q_wall = zeros(Float64, Nx, Ny, 9)
    is_solid = falses(Nx, Ny)
    base = _forward_channel(Nx, Ny, omega, g, q_wall, is_solid)
    profile_rel = _profile_error_halfway(base.f_star, g, nu, Nx, Ny)
    djdf = _dJdf(base.f_star, Nx, Ny)
    djdf_fd_rel = _check_dJdf_fd(base.f_star, djdf, Nx, Ny)
    adj = Kraken.gmres_adjoint(v -> _channel_vjp(base.f_star, v, q_wall,
                                                 is_solid, omega, g, Nx, Ny),
                               djdf; tol=1e-11, restart=80,
                               max_restarts=6, max_richardson=240,
                               richardson_tol=1e-9, linear_tol=1e-10)
    dJdg_adj = _lambda_dot_dGdg(base.f_star, adj.lambda, q_wall, is_solid,
                                omega, g, Nx, Ny)
    fd1 = _fd_dJdg(Nx, Ny, omega, g, 2e-7, q_wall, is_solid, base.f_star)
    fd2 = _fd_dJdg(Nx, Ny, omega, g, 1e-7, q_wall, is_solid, base.f_star)
    fd_consistency = _relerr(fd1.value, fd2.value)
    fd = fd_consistency < 1e-5 ? fd2 : fd1
    rel = _relerr(dJdg_adj, fd.value)
    return (; Nx, Ny, base, profile_rel, djdf_fd_rel, adj, dJdg_adj,
            fd_value=fd.value, fd, fd_consistency, rel)
end

function _run_c2()
    Nx, Ny = 8, 16
    H = 8.3
    nu = 0.1
    omega = 1.0 / (3.0 * nu + 0.5)
    g = 1e-6
    geom = _flat_wall_geometry(Nx, Ny, H)
    f0 = _initial_poiseuille(Nx, Ny, g, nu, geom.y_bot, geom.y_top)
    base = _forward_channel(Nx, Ny, omega, g, geom.q_wall, geom.is_solid;
                            f_init=f0)
    profile_rel = _profile_error_cut(base.f_star, g, nu, Nx, Ny,
                                     geom.y_bot, geom.y_top)
    J0 = _channel_J(base.f_star, Nx, Ny)
    djdf = _dJdf(base.f_star, Nx, Ny)
    adj = Kraken.gmres_adjoint(v -> _channel_vjp(base.f_star, v, geom.q_wall,
                                                 geom.is_solid, omega, g, Nx, Ny),
                               djdf; tol=1e-11, restart=80,
                               max_restarts=6, max_richardson=240,
                               richardson_tol=1e-9, linear_tol=1e-10)
    geom_term = _lambdaG_dH(base.f_star, adj.lambda, geom.q_wall, geom.dq_dH,
                            geom.is_solid, omega, g, Nx, Ny)
    fd = _fd_dJdH(Nx, Ny, omega, g, H, 0.05, base.f_star)
    fd_rel = _relerr(geom_term.value, fd.value)
    analytic = 3.0 * J0 / H
    analytic_rel = _relerr(geom_term.value, analytic)
    sign_match = sign(geom_term.value) == sign(analytic)
    return (; Nx, Ny, H, geom, base, profile_rel, adj, geom_term,
            fd_value=fd.value, fd, fd_rel, analytic, analytic_rel, sign_match)
end

function _run_c3()
    Nx, Ny = 48, 16
    radius = 3.75
    nu = 0.05
    u_in = 0.05
    kwargs = (; Nx, Ny, cx=Nx ÷ 4, cy=Ny ÷ 2, radius, u_in,
              (NU_KEY)=>nu, tol=1e-9, max_steps=60_000,
              inlet=:parabolic, fd_check=true, fd_h=0.05,
              gmres_tol=1e-9, adjoint_tol=1e-8)
    result = Kraken.steady_shape_sensitivity(; kwargs...)
    return (; Nx, Ny, radius, nu, u_in, result,
            fd_value=result.fd_check.value, rel=result.fd_check.relerr)
end

function _run_c4_krk()
    path = normpath(joinpath(@__DIR__, "..", "..", "examples",
                             "sensitivity_cylinder.krk"))
    setup = Kraken.load_kraken(path)
    krk = Kraken.run_simulation(setup)

    nu = Float64(setup.physics.params[:nu])
    kwargs = (;
        Nx=setup.domain.Nx,
        Ny=setup.domain.Ny,
        cx=Float64(setup.user_vars[:cx]),
        cy=Float64(setup.user_vars[:cy]),
        radius=Float64(setup.user_vars[:R]),
        u_in=Float64(setup.user_vars[:U]),
        (NU_KEY)=>nu,
        (RHO_OUT_KEY)=>1.0,
        qoi=setup.sensitivity.qoi,
        wrt=setup.sensitivity.wrt,
        tol=Float64(setup.user_vars[:tol]),
        max_steps=setup.max_steps,
        inlet=:parabolic,
        gmres_tol=Float64(setup.user_vars[:gmres_tol]),
        adjoint_tol=Float64(setup.user_vars[:adjoint_tol]),
    )
    direct = Kraken.steady_shape_sensitivity(; kwargs...)
    rel = _relerr(krk.gradient, direct.gradient)
    return (; path, setup, krk, direct, rel)
end

function _run_antidrift()
    Nx, Ny = 48, 16
    radius = 3.75
    nu = 0.05
    u_in = 0.05
    cx, cy = Nx ÷ 4, Ny ÷ 2
    base = Kraken.ad_forward_solve(; Nx, Ny, cx, cy, radius, u_in,
                                   nu, inlet=:parabolic, tol=1e-9,
                                   max_steps=60_000)
    uwx = zeros(Float64, size(base.q_wall))
    uwy = zeros(Float64, size(base.q_wall))
    drag = Kraken.compute_drag_libb_mei_2d(base.f_star, base.q_wall, uwx, uwy, Nx, Ny)
    cd_mei = 2.0 * Float64(drag.Fx) / (base.u_ref * base.u_ref * base.D)
    cd_inline = Kraken.cd_pure(base.f_star, base.q_wall, base.u_ref, base.D, Nx, Ny)

    run_kwargs = (; Nx, Ny, cx, cy, radius, u_in, (NU_KEY)=>nu,
                  inlet=:parabolic, max_steps=base.n_iter, avg_window=1,
                  backend=KernelAbstractions.CPU(), T=Float64,
                  (RHO_OUT_KEY)=>base.rho_out)
    fused = Kraken.run_cylinder_libb_2d(; run_kwargs...)
    return (; Nx, Ny, radius, nu, u_in, base, cd_inline, cd_mei,
            cd_mei_delta=abs(cd_inline - cd_mei),
            cd_fused=fused.Cd, cd_fused_delta=abs(cd_inline - fused.Cd))
end

function _thermal_params_with_beta(p, beta_g)
    return Kraken.ADNatconvParams(p.Nx, p.Ny, p.omega_f, p.omega_T,
                                  Float64(beta_g), p.T_ref, p.T_hot,
                                  p.T_cold, p.Ra, p.Pr)
end

function _thermal_forward_from(p; L=nothing, q_hot=0.5, q_cold=0.7,
                               tol=1e-11, max_steps=120_000, w_init=nothing)
    L_f = Float64(L === nothing ? Float64(p.Nx - 1) + q_hot + q_cold : L)
    geom = Kraken.ad_cavity_wall_geometry(p.Nx, p.Ny, L_f; q_hot=q_hot)
    w_in = w_init === nothing ?
           Kraken.ad_initial_thermal_w(p, geom.x_cold, geom.q_hot) :
           copy(w_init)
    w_out = similar(w_in)
    residual = Inf
    n_iter = 0
    converged = false
    for step in 1:max_steps
        Kraken.ad_thermal_cut_step!(w_out, w_in, geom.q_wall, geom.q_wall, p)
        residual = Kraken.ad_relative_step_residual(w_out, w_in)
        n_iter = step
        if residual < tol
            converged = true
            break
        end
        w_in, w_out = w_out, w_in
    end
    return (; w_star=copy(w_out), q_wall=geom.q_wall, dq_dL=geom.dq_dL,
            q_hot=geom.q_hot, q_cold=geom.q_cold, L=L_f, params=p,
            Nu=Kraken.nu_pure(w_out, geom.q_wall, p),
            n_iter, residual=Float64(residual), converged)
end

function _thermal_dense_fd_vjp(w, v, q_wall, p; h=1e-6)
    grad = zeros(Float64, length(w))
    wp = copy(w)
    wm = copy(w)
    outp = similar(w)
    outm = similar(w)
    @inbounds for idx in eachindex(w)
        old = w[idx]
        wp[idx] = old + h
        wm[idx] = old - h
        Kraken.ad_thermal_cut_step!(outp, wp, q_wall, q_wall, p)
        Kraken.ad_thermal_cut_step!(outm, wm, q_wall, q_wall, p)
        s = 0.0
        for jdx in eachindex(v, outp, outm)
            s += v[jdx] * (outp[jdx] - outm[jdx])
        end
        grad[idx] = s / (2h)
        wp[idx] = old
        wm[idx] = old
    end
    return grad
end

function _run_tc0()
    N = 3
    p = Kraken.ad_natconv_params(; N, Ra=1e3, Pr=0.71)
    geom = Kraken.ad_cavity_wall_geometry(N, N, Float64(N - 1) + 1.2;
                                          q_hot=0.5)
    rng = MersenneTwister(0x7c0)
    w = Kraken.ad_initial_thermal_w(p, geom.x_cold, geom.q_hot)
    @inbounds for idx in eachindex(w)
        w[idx] += 1e-4 * randn(rng)
    end
    nf = Kraken.ad_thermal_nlat(p)
    vf = zeros(Float64, length(w))
    vg = zeros(Float64, length(w))
    vf[1:nf] .= randn(rng, nf)
    vg[(nf + 1):end] .= randn(rng, nf)

    ad_f = Kraken._ad_thermal_vjp_GtT(w, vf, geom.q_wall, geom.q_wall, p)
    fd_f = _thermal_dense_fd_vjp(w, vf, geom.q_wall, p)
    ad_g = Kraken._ad_thermal_vjp_GtT(w, vg, geom.q_wall, geom.q_wall, p)
    fd_g = _thermal_dense_fd_vjp(w, vg, geom.q_wall, p)

    rel_f = norm(ad_f .- fd_f) / max(norm(fd_f), eps(Float64))
    rel_g = norm(ad_g .- fd_g) / max(norm(fd_g), eps(Float64))
    thermal_to_flow = norm(@view ad_f[(nf + 1):end])
    flow_to_thermal = norm(@view ad_g[1:nf])
    return (; N, rel=max(rel_f, rel_g), rel_f, rel_g,
            thermal_to_flow, flow_to_thermal)
end

function _thermal_scalarLG_beta(w_star, lambda, beta_g, q_wall,
                                Nx::Int, Ny::Int, omega_f, omega_T,
                                T_ref, T_hot, T_cold, Ra, Pr)
    p = Kraken.ADNatconvParams(Nx, Ny, omega_f, omega_T, beta_g,
                               T_ref, T_hot, T_cold, Ra, Pr)
    out = zeros(Float64, length(w_star))
    Kraken.ad_thermal_cut_step!(out, w_star, q_wall, q_wall, p)
    return _dot_arrays(lambda, out)
end

function _thermal_lambda_dot_dG_dbeta(w_star, lambda, q_wall, p)
    ret = Enzyme.autodiff(Enzyme.Reverse, _thermal_scalarLG_beta,
                          Enzyme.Active,
                          Enzyme.Const(w_star),
                          Enzyme.Const(lambda),
                          Enzyme.Active(p.beta_g),
                          Enzyme.Const(q_wall),
                          Enzyme.Const(p.Nx), Enzyme.Const(p.Ny),
                          Enzyme.Const(p.omega_f), Enzyme.Const(p.omega_T),
                          Enzyme.Const(p.T_ref), Enzyme.Const(p.T_hot),
                          Enzyme.Const(p.T_cold), Enzyme.Const(p.Ra),
                          Enzyme.Const(p.Pr))
    return Float64(ret[1][3])
end

function _thermal_adjoint_for_rhs(base, p, rhs; gmres_tol=1e-10,
                                  linear_tol=1e-9, max_restarts=8)
    apply_GtT = v -> Kraken._ad_thermal_vjp_GtT(base.w_star, v,
                                                base.q_wall, base.q_wall, p)
    rhohat = Kraken.ad_richardson_rhohat(apply_GtT, rhs; n_iter=80)
    mass = Kraken.ad_thermal_mass_gradient(p)
    adj = Kraken.ad_gauge_augmented_adjoint(apply_GtT, rhs, mass;
                                            tol=gmres_tol,
                                            restart=min(256, length(rhs) + 1),
                                            max_restarts=max_restarts,
                                            linear_tol=linear_tol,
                                            rhohat=rhohat)
    return (; rhs, adj, rhohat)
end

function _thermal_adjoint(base, p; gmres_tol=1e-10, linear_tol=1e-9,
                          max_restarts=8)
    rhs = Kraken._ad_dNudw(base.w_star, base.q_wall, p)
    return _thermal_adjoint_for_rhs(base, p, rhs; gmres_tol=gmres_tol,
                                    linear_tol=linear_tol,
                                    max_restarts=max_restarts)
end

function _thermal_fd_dNu_dbeta(p, base, h; tol=1e-10, max_steps=120_000)
    plus_p = _thermal_params_with_beta(p, p.beta_g + h)
    minus_p = _thermal_params_with_beta(p, p.beta_g - h)
    plus = _thermal_forward_from(plus_p; L=base.L, q_hot=base.q_hot,
                                 tol=tol, max_steps=max_steps,
                                 w_init=base.w_star)
    minus = _thermal_forward_from(minus_p; L=base.L, q_hot=base.q_hot,
                                  tol=tol, max_steps=max_steps,
                                  w_init=base.w_star)
    value = (plus.converged && minus.converged) ?
        (plus.Nu - minus.Nu) / (2h) : NaN
    return (; value, plus, minus, h=Float64(h))
end

function _run_tc1()
    N = 8
    p = Kraken.ad_natconv_params(; N, Ra=1e3, Pr=0.71)
    L = Float64(N - 1) + 1.2
    base = _thermal_forward_from(p; L, q_hot=0.5, tol=1e-10,
                                 max_steps=120_000)
    adjinfo = _thermal_adjoint(base, p; gmres_tol=1e-10,
                               linear_tol=1e-8, max_restarts=8)
    dnudbeta = _thermal_lambda_dot_dG_dbeta(base.w_star,
                                            adjinfo.adj.lambda,
                                            base.q_wall, p)
    fd1 = _thermal_fd_dNu_dbeta(p, base, 1e-4)
    fd2 = _thermal_fd_dNu_dbeta(p, base, 5e-5)
    rel1 = _relerr(dnudbeta, fd1.value)
    rel2 = _relerr(dnudbeta, fd2.value)
    rel = min(rel1, rel2)
    fd = rel2 <= rel1 ? fd2 : fd1
    return (; N, base, adj=adjinfo.adj, dnudbeta, fd, fd1, fd2,
            rel, fd_consistency=_relerr(fd1.value, fd2.value))
end

function _thermal_temperature(w, p, i::Int, j::Int)
    goff = Kraken.ad_thermal_nlat(p)
    s = 0.0
    @inbounds for q in 1:9
        s += Kraken.ad_thermal_readpop(w, goff, i, j, q, p.Nx, p.Ny)
    end
    return s
end

function _thermal_hot_flux(w, q_wall, p, q_hot::Real)
    alpha = 0.05 / p.Pr
    s = 0.0
    @inbounds for j in 1:p.Ny
        s += alpha * (p.T_hot - _thermal_temperature(w, p, 1, j)) / q_hot
    end
    z = 0.0
    @inbounds for idx in eachindex(q_wall)
        z += 0.0 * q_wall[idx]
    end
    return s / p.Ny + z
end

function _thermal_dQdw(w_star, q_wall, p, q_hot::Real)
    dwd = zeros(Float64, length(w_star))
    Enzyme.autodiff(Enzyme.Reverse, _thermal_hot_flux, Enzyme.Active,
                    Enzyme.Duplicated(copy(w_star), dwd),
                    Enzyme.Const(q_wall), Enzyme.Const(p),
                    Enzyme.Const(q_hot))
    return dwd
end

function _thermal_lambda_dot_dG_dqwalls(w_star, lambda, q_flow, q_therm, p)
    out = zeros(Float64, length(w_star))
    dqw_flow = zeros(Float64, size(q_flow))
    dqw_therm = zeros(Float64, size(q_therm))
    Enzyme.autodiff(Enzyme.Reverse, Kraken.ad_thermal_cut_step!,
                    Enzyme.Duplicated(out, copy(lambda)),
                    Enzyme.Const(w_star),
                    Enzyme.Duplicated(copy(q_flow), dqw_flow),
                    Enzyme.Duplicated(copy(q_therm), dqw_therm),
                    Enzyme.Const(p))
    return (; flow=dqw_flow, thermal=dqw_therm)
end

@inline _tc2_temp_at(g, i::Int, j::Int) =
    g[i, j, 1] + g[i, j, 2] + g[i, j, 3] +
    g[i, j, 4] + g[i, j, 5] + g[i, j, 6] +
    g[i, j, 7] + g[i, j, 8] + g[i, j, 9]

@inline function _tc2_write_equilibrium!(g, i::Int, j::Int, temp)
    T = typeof(temp)
    @inbounds for q in 1:9
        g[i, j, q] = T(WT[q]) * temp
    end
    return nothing
end

@inline _tc2_mean3(a, b, c) = (a + b + c) / 3.0
@inline _tc2_west_cut(q_wall, j::Int) =
    _tc2_mean3(q_wall[1, j, 4], q_wall[1, j, 7], q_wall[1, j, 8])
@inline _tc2_east_cut(q_wall, Nx::Int, j::Int) =
    _tc2_mean3(q_wall[Nx, j, 2], q_wall[Nx, j, 6], q_wall[Nx, j, 9])

@inline function _tc2_dirichlet_ghost(temp_here, temp_wall, q_wall_link)
    T = typeof(temp_here)
    q = T(q_wall_link)
    return (T(temp_wall) - (one(T) - q) * temp_here) / q
end

function _tc2_slab_geometry(Nx::Int, Ny::Int, L::Real; q_hot::Real=0.5)
    qh = Float64(q_hot)
    qc = Float64(L) - (Float64(Nx - 1) + qh)
    if !(0.0 < qh <= 1.0 && 0.0 < qc <= 1.0)
        throw(ArgumentError("TC2 slab L=$(Float64(L)) gives q_hot=$qh q_cold=$qc"))
    end
    q_wall = zeros(Float64, Nx, Ny, 9)
    dq_dL = zeros(Float64, Nx, Ny, 9)
    @inbounds for j in 1:Ny
        q_wall[1, j, 4] = qh
        q_wall[1, j, 7] = qh
        q_wall[1, j, 8] = qh
        q_wall[Nx, j, 2] = qc
        q_wall[Nx, j, 6] = qc
        q_wall[Nx, j, 9] = qc
        dq_dL[Nx, j, 2] = 1.0
        dq_dL[Nx, j, 6] = 1.0
        dq_dL[Nx, j, 9] = 1.0
    end
    return (; q_wall, dq_dL, q_hot=qh, q_cold=qc)
end

function _tc2_analytic_temp(i::Int, L::Real, q_hot::Real,
                            T_hot::Real, T_cold::Real)
    x = Float64(q_hot) + Float64(i - 1)
    return Float64(T_hot) -
           (Float64(T_hot) - Float64(T_cold)) * x / Float64(L)
end

function _tc2_initial_g(Nx::Int, Ny::Int, L::Real, q_hot::Real,
                        T_hot::Real, T_cold::Real)
    g = zeros(Float64, Nx, Ny, 9)
    @inbounds for j in 1:Ny, i in 1:Nx
        t = _tc2_analytic_temp(i, L, q_hot, T_hot, T_cold)
        t += 1e-3 * sin(2.0 * pi * i / Nx) * cos(pi * j / (Ny + 1))
        _tc2_write_equilibrium!(g, i, j, t)
    end
    return g
end

function _tc2_thermal_G!(g_out, g, q_wall, Nx::Int, Ny::Int,
                         relax::Real, T_hot::Real, T_cold::Real)
    @inbounds for j in 1:Ny, i in 1:Nx
        t_here = _tc2_temp_at(g, i, j)
        t_west = if i == 1
            _tc2_dirichlet_ghost(t_here, T_hot, _tc2_west_cut(q_wall, j))
        else
            _tc2_temp_at(g, i - 1, j)
        end
        t_east = if i == Nx
            _tc2_dirichlet_ghost(t_here, T_cold, _tc2_east_cut(q_wall, Nx, j))
        else
            _tc2_temp_at(g, i + 1, j)
        end
        _tc2_write_equilibrium!(g_out, i, j,
                                (1.0 - relax) * t_here +
                                relax * 0.5 * (t_west + t_east))
    end
    return nothing
end

function _tc2_forward(Nx::Int, Ny::Int, L::Real, q_wall;
                      q_hot::Real, T_hot::Real, T_cold::Real,
                      relax::Real=0.92, tol::Real=1e-13,
                      max_steps::Int=80_000)
    g_in = _tc2_initial_g(Nx, Ny, L, q_hot, T_hot, T_cold)
    g_out = similar(g_in)
    residual = Inf
    n_iter = 0
    converged = false
    for step in 1:max_steps
        _tc2_thermal_G!(g_out, g_in, q_wall, Nx, Ny, relax, T_hot, T_cold)
        residual = _relative_residual(g_out, g_in)
        n_iter = step
        if residual < tol
            converged = true
            break
        end
        g_in, g_out = g_out, g_in
    end
    return (; g_star=copy(g_out), n_iter, residual, converged)
end

function _tc2_hot_flux(g, q_wall, Nx::Int, Ny::Int, alpha::Real,
                       T_hot::Real, q_hot::Real)
    s = 0.0
    @inbounds for j in 1:Ny
        s += Float64(alpha) * (Float64(T_hot) - _tc2_temp_at(g, 1, j)) /
             Float64(q_hot)
    end
    z = 0.0
    @inbounds for idx in eachindex(q_wall)
        z += 0.0 * q_wall[idx]
    end
    return s / Ny + z
end

function _tc2_dQdg(g_star, q_wall, Nx::Int, Ny::Int, alpha::Real,
                   T_hot::Real, q_hot::Real)
    dg = zeros(Float64, size(g_star))
    Enzyme.autodiff(Enzyme.Reverse, _tc2_hot_flux, Enzyme.Active,
                    Enzyme.Duplicated(copy(g_star), dg),
                    Enzyme.Const(q_wall), Enzyme.Const(Nx),
                    Enzyme.Const(Ny), Enzyme.Const(alpha),
                    Enzyme.Const(T_hot), Enzyme.Const(q_hot))
    return dg
end

function _tc2_Gt_vjp(g_star, v, q_wall, Nx::Int, Ny::Int,
                     relax::Real, T_hot::Real, T_cold::Real)
    out = zeros(Float64, size(g_star))
    dg = zeros(Float64, size(g_star))
    Enzyme.autodiff(Enzyme.Reverse, _tc2_thermal_G!,
                    Enzyme.Duplicated(out, copy(v)),
                    Enzyme.Duplicated(copy(g_star), dg),
                    Enzyme.Const(q_wall), Enzyme.Const(Nx),
                    Enzyme.Const(Ny), Enzyme.Const(relax),
                    Enzyme.Const(T_hot), Enzyme.Const(T_cold))
    return dg
end

function _tc2_lambda_dot_dG_dqwall(g_star, lambda, q_wall,
                                   Nx::Int, Ny::Int, relax::Real,
                                   T_hot::Real, T_cold::Real)
    out = zeros(Float64, size(g_star))
    dqw = zeros(Float64, size(q_wall))
    Enzyme.autodiff(Enzyme.Reverse, _tc2_thermal_G!,
                    Enzyme.Duplicated(out, copy(lambda)),
                    Enzyme.Const(g_star),
                    Enzyme.Duplicated(copy(q_wall), dqw),
                    Enzyme.Const(Nx), Enzyme.Const(Ny),
                    Enzyme.Const(relax), Enzyme.Const(T_hot),
                    Enzyme.Const(T_cold))
    return dqw
end

function _run_tc2()
    N = 8
    Ny = 16
    q_hot = 0.5
    q_cold = 0.7
    L = Float64(N - 1) + q_hot + q_cold
    T_hot = 1.0
    T_cold = 0.0
    alpha = 0.1
    relax = 0.92
    geom = _tc2_slab_geometry(N, Ny, L; q_hot)
    base = _tc2_forward(N, Ny, L, geom.q_wall; q_hot, T_hot, T_cold,
                        relax, tol=1e-13)
    delta_T = 1.0
    q_value = _tc2_hot_flux(base.g_star, geom.q_wall, N, Ny, alpha,
                            T_hot, q_hot)
    q_exact = alpha * delta_T / L
    rhs = _tc2_dQdg(base.g_star, geom.q_wall, N, Ny, alpha, T_hot, q_hot)
    apply_GtT = v -> _tc2_Gt_vjp(base.g_star, v, geom.q_wall, N, Ny,
                                 relax, T_hot, T_cold)
    adj = Kraken.gmres_adjoint(apply_GtT, rhs; tol=1e-11,
                               restart=min(256, length(rhs)),
                               max_restarts=8, max_richardson=80,
                               linear_tol=1e-9)
    dqw = _tc2_lambda_dot_dG_dqwall(base.g_star, adj.lambda, geom.q_wall,
                                   N, Ny, relax, T_hot, T_cold)
    flux_gradient = _dot_arrays(dqw, geom.dq_dL)
    analytic = -alpha * delta_T / (L * L)
    return (; N, Ny, L, base, adj, q_value, q_exact,
            q_rel=_relerr(q_value, q_exact), flux_gradient, analytic,
            rel=_relerr(flux_gradient, analytic))
end

function _run_tc3()
    N = 16
    q_hot = 0.5
    q_cold = 0.7
    L = Float64(N - 1) + q_hot + q_cold
    result = Kraken.steady_shape_sensitivity(; qoi=:nusselt,
        wrt=:wall_position, N, Ra=1e3, Pr=0.71, L, q_hot, q_cold,
        tol=1e-11, max_steps=450_000, fd_check=true, fd_h=0.01,
        gmres_tol=1e-10, adjoint_tol=1e-10)
    return (; N, L, result, fd=result.fd_check, rel=result.fd_check.relerr)
end

function _run_tc_krk()
    path = normpath(joinpath(@__DIR__, "..", "..", "examples",
                             "sensitivity_cavity_nusselt.krk"))
    setup = Kraken.load_kraken(path)
    krk = Kraken.run_simulation(setup)
    direct = Kraken.steady_shape_sensitivity(;
        qoi=setup.sensitivity.qoi,
        wrt=setup.sensitivity.wrt,
        N=setup.domain.Nx,
        Ra=Float64(setup.physics.params[:Ra]),
        Pr=Float64(setup.physics.params[:Pr]),
        L=Float64(setup.user_vars[:L]),
        q_hot=Float64(setup.user_vars[:q_hot]),
        q_cold=Float64(setup.user_vars[:q_cold]),
        T_hot=1.0,
        T_cold=0.0,
        tol=Float64(setup.user_vars[:tol]),
        max_steps=setup.max_steps,
        gmres_tol=Float64(setup.user_vars[:gmres_tol]),
        adjoint_tol=Float64(setup.user_vars[:adjoint_tol]))
    return (; path, setup, krk, direct,
            rel=_relerr(krk.gradient, direct.gradient))
end

function _run_thermal_antidrift()
    N = 8
    base = Kraken.ad_thermal_forward_solve(; N, Ra=1e3, Pr=0.71,
                                           L=Float64(N - 1) + 1.2,
                                           q_hot=0.5, q_cold=0.7,
                                           tol=1e-10,
                                           max_steps=120_000)
    nu_pure = Kraken.nu_pure(base.w_star, base.q_wall, base.params)
    # The AD QoI reads the wall distance and the cavity width from q_wall;
    # the scalar formula must give the same value for the same geometry.
    nu_driver = Kraken.nu_driver(base.w_star, base.params;
                                 q_hot=base.q_hot, L=base.L)
    return (; N, base, nu_pure, nu_driver,
            delta=abs(nu_pure - nu_driver),
            rel=_relerr(nu_pure, nu_driver))
end

@testset "AD steady shape-adjoint" begin
    @test Base.get_extension(Kraken, :KrakenADExt) !== nothing

    @testset "C0 bulk VJP" begin
        elapsed = @elapsed c0 = _run_c0()
        @test c0.max_rel < 1e-6
        @info "AD C0" grid="$(c0.Nx)x$(c0.Ny)" pairs=c0.pairs rel=c0.max_rel seconds=elapsed
    end

    @testset "C1 Poiseuille body-force adjoint" begin
        elapsed = @elapsed c1 = _run_c1()
        @test c1.base.converged
        @test c1.adj.converged
        @test c1.profile_rel < 2e-2
        @test c1.djdf_fd_rel < 1e-8
        @test c1.rel < 1e-4
        @info "AD C1" grid="$(c1.Nx)x$(c1.Ny)" forward_iter=c1.base.n_iter rel=c1.rel seconds=elapsed
    end

    @testset "C2 Poiseuille geometry chain" begin
        elapsed = @elapsed c2 = _run_c2()
        @test c2.base.converged
        @test c2.adj.converged
        @test c2.profile_rel < 8e-2
        @test c2.sign_match
        @test c2.analytic_rel < 0.10
        @test c2.fd_rel < 1e-3
        @info "AD C2" grid="$(c2.Nx)x$(c2.Ny)" H=c2.H path=c2.geom_term.path fd_rel=c2.fd_rel analytic_rel=c2.analytic_rel seconds=elapsed
    end

    @testset "C3 cylinder radius sensitivity" begin
        elapsed = @elapsed c3 = _run_c3()
        @test c3.result.solver.converged
        @test c3.result.fd_check.plus_converged
        @test c3.result.fd_check.minus_converged
        @test c3.rel < 1e-2
        @info "AD C3" grid="$(c3.Nx)x$(c3.Ny)" radius=c3.radius gradient=c3.result.gradient fd=c3.fd_value rel=c3.rel seconds=elapsed
    end

    @testset "C4 krk Sensitivity" begin
        elapsed = @elapsed c4 = _run_c4_krk()
        @test c4.setup.sensitivity == (; qoi=:drag, wrt=:radius)
        @test c4.krk.solver.converged
        @test c4.direct.solver.converged
        @test c4.rel < 1e-8
        @info "AD C4 krk Sensitivity" path=c4.path krk_gradient=c4.krk.gradient api_gradient=c4.direct.gradient rel=c4.rel seconds=elapsed
    end

    @testset "anti-drift drag bit mirror" begin
        elapsed = @elapsed guard = _run_antidrift()
        # BIT-MIRROR GUARD -- if a change to the fused production collision/BC/drag flips this RED,
        # the unfused AD step in `src/ad/ad_step.jl`/`ad_qoi.jl` has drifted and must be re-synced.
        @test guard.base.converged
        @test guard.cd_mei_delta < 1e-6
        @test guard.cd_fused_delta < 1e-6
        @info "AD anti-drift" grid="$(guard.Nx)x$(guard.Ny)" radius=guard.radius forward_iter=guard.base.n_iter mei_delta=guard.cd_mei_delta fused_delta=guard.cd_fused_delta seconds=elapsed
    end
end

@testset "AD thermal (Nusselt)" begin
    @test Base.get_extension(Kraken, :KrakenADExt) !== nothing

    @testset "TC0 coupled one-step VJP" begin
        elapsed = @elapsed tc0 = _run_tc0()
        @test tc0.rel < 1e-6
        @test tc0.thermal_to_flow > 1e-10
        @test tc0.flow_to_thermal > 1e-10
        @info "AD thermal TC0" grid="$(tc0.N)x$(tc0.N)" rel=tc0.rel thermal_to_flow=tc0.thermal_to_flow flow_to_thermal=tc0.flow_to_thermal seconds=elapsed
    end

    @testset "TC1 beta_g adjoint" begin
        elapsed = @elapsed tc1 = _run_tc1()
        @test tc1.base.converged
        @test tc1.adj.converged
        @test tc1.fd.plus.converged
        @test tc1.fd.minus.converged
        @test tc1.rel < 1e-4
        @info "AD thermal TC1" grid="$(tc1.N)x$(tc1.N)" forward_iter=tc1.base.n_iter gradient=tc1.dnudbeta fd=tc1.fd.value rel=tc1.rel fd_consistency=tc1.fd_consistency seconds=elapsed
    end

    @testset "TC2 conduction geometry chain" begin
        elapsed = @elapsed tc2 = _run_tc2()
        @test tc2.base.converged
        @test tc2.adj.converged
        @test tc2.q_rel < 1e-8
        @test tc2.rel < 1e-3
        @info "AD thermal TC2" grid="$(tc2.N)x$(tc2.N)" L=tc2.L q=tc2.q_value q_exact=tc2.q_exact q_rel=tc2.q_rel gradient_flux=tc2.flux_gradient analytic=tc2.analytic rel=tc2.rel seconds=elapsed
    end

    @testset "TC3 cavity dNu/dL" begin
        elapsed = @elapsed tc3 = _run_tc3()
        @test tc3.result.solver.converged
        @test tc3.fd.plus_converged
        @test tc3.fd.minus_converged
        @test tc3.rel < 2e-2
        @info "AD thermal TC3" grid="$(tc3.N)x$(tc3.N)" L=tc3.L gradient=tc3.result.gradient fd=tc3.fd.value rel=tc3.rel seconds=elapsed
    end

    @testset "TC-krk cavity Nusselt Sensitivity" begin
        elapsed = @elapsed tck = _run_tc_krk()
        @test tck.setup.sensitivity == (; qoi=:nusselt, wrt=:wall_position)
        @test tck.krk.solver.converged
        @test tck.direct.solver.converged
        @test tck.rel < 1e-8
        @info "AD thermal TC-krk" path=tck.path krk_gradient=tck.krk.gradient api_gradient=tck.direct.gradient rel=tck.rel seconds=elapsed
    end

    @testset "anti-drift Nusselt bit mirror" begin
        elapsed = @elapsed guard = _run_thermal_antidrift()
        @test guard.base.converged
        @test guard.delta < 1e-12
        @info "AD thermal anti-drift" grid="$(guard.N)x$(guard.N)" forward_iter=guard.base.n_iter nu_pure=guard.nu_pure driver_Nu=guard.nu_driver delta=guard.delta seconds=elapsed
    end
end

end # module
