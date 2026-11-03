# BIT-MIRROR of production compute_drag_libb_mei_2d for stationary walls.

function cd_pure(f, q_wall, u_ref, D, Nx::Int, Ny::Int)
    Fx = 0.0
    @inbounds for j in 1:Ny, i in 1:Nx
        for q in 2:9
            q_w = q_wall[i, j, q]
            q_w > 0.0 || continue
            qbar = AD_OPP[q]
            im = i - Int(AD_CXV[q])
            jm = j - Int(AD_CYV[q])
            fp_q_back = (1 <= im <= Nx && 1 <= jm <= Ny) ? f[im, jm, q] : f[i, j, qbar]
            fq_here = f[i, j, q]
            fqbar_here = f[i, j, qbar]
            arriving = if q_w <= 0.5
                2.0 * q_w * fq_here + (1.0 - 2.0 * q_w) * fp_q_back
            else
                inv2q = 1.0 / (2.0 * q_w)
                inv2q * fq_here + (1.0 - inv2q) * fqbar_here
            end
            Fx += AD_CXV[q] * (fq_here + arriving)
        end
    end
    return 2.0 * Fx / (u_ref * u_ref * D)
end

function cd_production(f, q_wall, u_ref, D, Nx::Int, Ny::Int)
    uw_x = zeros(Float64, size(q_wall))
    uw_y = zeros(Float64, size(q_wall))
    drag = compute_drag_libb_mei_2d(f, q_wall, uw_x, uw_y, Nx, Ny)
    return 2.0 * Float64(drag.Fx) / (u_ref * u_ref * D)
end

function fd_dCd_dqwall(f, q_wall, u_ref, D, Nx::Int, Ny::Int)
    dqw = zeros(Float64, size(q_wall))
    eps_q = 1e-6
    @inbounds for idx in CartesianIndices(q_wall)
        q_wall[idx] > 0.0 || continue
        qp = copy(q_wall)
        qm = copy(q_wall)
        qp[idx] += eps_q
        qm[idx] -= eps_q
        vp = cd_pure(f, qp, u_ref, D, Nx, Ny)
        vm = cd_pure(f, qm, u_ref, D, Nx, Ny)
        dqw[idx] = (vp - vm) / (2.0 * eps_q)
    end
    return dqw
end

function ad_lambda_dot_G_value(f_star, lambda, q_wall, is_solid, u_profile,
                               rho_out, s_plus, s_minus, Nx::Int, Ny::Int)
    out = zeros(Float64, size(f_star))
    ad_step!(out, f_star, q_wall, is_solid, u_profile, rho_out,
             s_plus, s_minus, Nx, Ny)
    return ad_dot_arrays(lambda, out)
end

function fd_lambda_dot_dG_dqwall(f_star, lambda, q_wall, is_solid, u_profile,
                                 rho_out, s_plus, s_minus, Nx::Int, Ny::Int)
    dqw = zeros(Float64, size(q_wall))
    eps_q = 1e-6
    @inbounds for idx in CartesianIndices(q_wall)
        q_wall[idx] > 0.0 || continue
        qp = copy(q_wall)
        qm = copy(q_wall)
        qp[idx] += eps_q
        qm[idx] -= eps_q
        vp = ad_lambda_dot_G_value(f_star, lambda, qp, is_solid, u_profile,
                                   rho_out, s_plus, s_minus, Nx, Ny)
        vm = ad_lambda_dot_G_value(f_star, lambda, qm, is_solid, u_profile,
                                   rho_out, s_plus, s_minus, Nx, Ny)
        dqw[idx] = (vp - vm) / (2.0 * eps_q)
    end
    return dqw
end

function directional_fd_Cd_qwall(f, q_wall, dq_dR, u_ref, D, Nx::Int, Ny::Int)
    eps_R = 1e-5
    qp = copy(q_wall)
    qm = copy(q_wall)
    @inbounds for idx in eachindex(q_wall, dq_dR)
        qp[idx] += eps_R * dq_dR[idx]
        qm[idx] -= eps_R * dq_dR[idx]
    end
    return (cd_pure(f, qp, u_ref, D, Nx, Ny) -
            cd_pure(f, qm, u_ref, D, Nx, Ny)) / (2.0 * eps_R)
end

function directional_fd_lambdaG_qwall(f_star, lambda, q_wall, dq_dR,
                                      is_solid, u_profile, rho_out,
                                      s_plus, s_minus, Nx::Int, Ny::Int)
    eps_R = 1e-5
    qp = copy(q_wall)
    qm = copy(q_wall)
    @inbounds for idx in eachindex(q_wall, dq_dR)
        qp[idx] += eps_R * dq_dR[idx]
        qm[idx] -= eps_R * dq_dR[idx]
    end
    vp = ad_lambda_dot_G_value(f_star, lambda, qp, is_solid, u_profile,
                               rho_out, s_plus, s_minus, Nx, Ny)
    vm = ad_lambda_dot_G_value(f_star, lambda, qm, is_solid, u_profile,
                               rho_out, s_plus, s_minus, Nx, Ny)
    return (vp - vm) / (2.0 * eps_R)
end

function cut_cotangent_norm(dqw, q_wall)
    s = 0.0
    @inbounds for idx in eachindex(dqw, q_wall)
        if q_wall[idx] > 0.0
            s += dqw[idx] * dqw[idx]
        end
    end
    return sqrt(s)
end

function moving_cut_cotangent_norm(dqw, dq_dL)
    s = 0.0
    @inbounds for idx in eachindex(dqw, dq_dL)
        if dq_dL[idx] != 0.0
            s += dqw[idx] * dqw[idx]
        end
    end
    return sqrt(s)
end

# Hot-wall Nusselt number with the formula of `run_natural_convection_2d`:
# the hot wall is half-way (q_hot = 0.5, the default of
# `ad_cavity_wall_geometry`), and the second-order one-sided gradient goes
# through the wall value and the first two nodes,
# dT/dx|wall = (-8 T_hot + 9 T_1 - T_2) / 3.
function nu_driver(w, p::ADNatconvParams)
    Nx = p.Nx
    Ny = p.Ny
    goff = ad_thermal_nlat(p)
    H = Float64(Nx)
    delta_T = p.T_hot - p.T_cold
    s = 0.0
    @inbounds for j in 2:(Ny - 1)
        T1 = 0.0
        T2 = 0.0
        for q in 1:9
            T1 += ad_thermal_readpop(w, goff, 1, j, q, Nx, Ny)
            T2 += ad_thermal_readpop(w, goff, 2, j, q, Nx, Ny)
        end
        s += -H * (-8.0 * p.T_hot + 9.0 * T1 - T2) / 3.0 / delta_T
    end
    return s / (Ny - 2)
end

function nu_pure(w, q_wall, p::ADNatconvParams)
    z = 0.0
    @inbounds for idx in eachindex(q_wall)
        z += 0.0 * q_wall[idx]
    end
    return nu_driver(w, p) + z
end

function ad_thermal_lambda_dot_G_value(w_star, lambda, q_flow, q_therm,
                                       p::ADNatconvParams)
    out = zeros(Float64, length(w_star))
    ad_thermal_cut_step!(out, w_star, q_flow, q_therm, p)
    return ad_dot_arrays(lambda, out)
end

function directional_fd_thermal_lambdaG_qwalls(w_star, lambda, q_flow, q_therm,
                                               dq_dL, p::ADNatconvParams)
    eps_L = 1e-5
    qfp = copy(q_flow)
    qfm = copy(q_flow)
    qtp = copy(q_therm)
    qtm = copy(q_therm)
    @inbounds for idx in eachindex(dq_dL)
        step = eps_L * dq_dL[idx]
        qfp[idx] += step
        qfm[idx] -= step
        qtp[idx] += step
        qtm[idx] -= step
    end
    vp = ad_thermal_lambda_dot_G_value(w_star, lambda, qfp, qtp, p)
    vm = ad_thermal_lambda_dot_G_value(w_star, lambda, qfm, qtm, p)
    return (vp - vm) / (2.0 * eps_L)
end
