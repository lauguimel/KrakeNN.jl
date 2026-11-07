using Test
using LinearAlgebra
using KrakeNN

const CX = (0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0)
const CY = (0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0)
const W = (4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36)
const TMRT = [
     1  1  1  1  1  1  1  1  1;
    -4 -1 -1 -1 -1  2  2  2  2;
     4 -2 -2 -2 -2  1  1  1  1;
     0  1  0 -1  0  1 -1 -1  1;
     0 -2  0  2  0  1 -1 -1  1;
     0  0  1  0 -1  1  1 -1 -1;
     0  0 -2  0  2  1  1 -1 -1;
     0  1 -1  1 -1  0  0  0  0;
     0  0  0  0  0  1 -1  1 -1
]

function hydro_b_bisect(C; tol=1e-14)
    f(b) = (4C/3) * sqrt(b) * ((1 + b)^(3/2) - b^(3/2)) - 1
    lo = 1e-12
    hi = 1.0
    flo = f(lo)
    fhi = f(hi)
    while sign(flo) == sign(fhi) && hi < 1e12
        hi *= 2
        fhi = f(hi)
    end
    sign(flo) == sign(fhi) && error("Could not bracket hydrostatic b.")
    while hi - lo > tol
        mid = (lo + hi) / 2
        fm = f(mid)
        if sign(flo) == sign(fm)
            lo = mid
            flo = fm
        else
            hi = mid
        end
    end
    return (lo + hi) / 2
end

feq_ns(rho, ux, uy) = begin
    usq = ux^2 + uy^2
    [rho * W[q] * (1 + 3(CX[q] * ux + CY[q] * uy) +
                   4.5(CX[q] * ux + CY[q] * uy)^2 - 1.5usq) for q in 1:9]
end

function matlab_mrt_force(a_M, tau)
    # MATLAB Lattice.m:126-153: N = T\(eye(9)-Ak/2)*T.
    Ak = Diagonal([1.0, 1.64, 1.54, 1.0, 1.0, 1.0, 1.0, 1/tau, 1/tau])
    fEq = feq_ns(1.0, 0.0, 0.0)
    rhs = fEq .* (a_M .* collect(CY))
    return (1 / (1/3)) .* (TMRT \ ((I - Ak / 2) * (TMRT * rhs)))
end

function kraken_bgk_force(a_M, tau)
    # KrakeNN src/kernels/collide_guo_2d.jl:107-140:
    # force-corrected u=(sum(fc)+F/2)/rho, then textbook Guo source with prefactor.
    omega = 1 / tau
    fx = 0.0
    fy = a_M
    guo_pref = 1 - omega / 2
    out = zeros(9)
    for q in 1:9
        dot1 = CX[q] * fx + CY[q] * fy
        out[q] = guo_pref * W[q] * 3dot1
    end
    return out
end

function kraken_mrt_force(a_M, tau)
    # KrakeNN src/kernels/ehd_mrt_2d.jl:48-79 forms textbook Guo g_i, transforms
    # to moments, applies (1-s_k/2), then inverse-transforms using lines 85-93.
    fx = 0.0
    fy = a_M
    g = zeros(9)
    for q in 1:9
        dot1 = CX[q] * fx + CY[q] * fy
        g[q] = W[q] * 3dot1
    end
    s = [1.0, 1.64, 1.54, 1.0, 1.0, 1.0, 1.0, 1/tau, 1/tau]
    return TMRT \ ((I - Diagonal(s) / 2) * (TMRT * g))
end

@testset "EHD mapping parity" begin
    # MATLAB run_standard_LBM_electroconvection.m:26-31, 80-91, 132-150.
    Ny = 321
    H = Ny - 1
    C = 10.0
    T_ehd = 190.0
    M = 10.0
    alpha = 1e-4
    Ma_E = 0.01
    delta_U = 1.0
    gamma = 0.3
    rho_0 = 1.0
    delta_t = 1.0
    c = 1.0
    cs = sqrt(c^2 / 3)

    K = Ma_E * H * cs / delta_U
    nu = M^2 * K * delta_U / T_ehd
    tau = 0.5 + nu / (cs^2 * delta_t)
    dt_star = K * delta_U / H^2
    eps_e = rho_0 * (M * K)^2
    q_inj = C * eps_e * delta_U / H^2
    D = alpha * K * delta_U
    T_check = eps_e * delta_U / (rho_0 * nu * K)
    C_check = q_inj * H^2 / (eps_e * delta_U)
    M_check = sqrt(eps_e / rho_0) / K
    alpha_check = D / (K * delta_U)
    tau_U = 3 * gamma + 0.5
    tau_q = 3 * D + 0.5
    Ma_electric = K * delta_U / (H * cs)

    p = KrakeNN._ehd_ec_lattice_params(Ny, C, M, T_ehd, Ma_E, alpha, delta_U, gamma; FT=Float64)
    @test p.tau > 0.5
    @test p.tau_q > 0.5

    mapped = (
        H=Float64(H), cs=cs, K=K, nu=nu, tau=tau, omega=1 / tau,
        eps=eps_e, q_inj=q_inj, D=D, tau_U=tau_U, nu_U=gamma,
        omega_U=1 / tau_U, tau_q=tau_q, omega_q=1 / tau_q,
        dt_star=dt_star, T_check=T_check, C_check=C_check,
        M_check=M_check, alpha_check=alpha_check,
    )
    for name in propertynames(mapped)
        @test isapprox(getproperty(p, name), getproperty(mapped, name); rtol=1e-12)
    end
    @test isapprox(p.K * delta_U / (p.H * p.cs), Ma_electric; rtol=1e-12)

    # MATLAB run_standard_LBM_electroconvection.m:157-175 vs KrakeNN src/drivers/ehd_ec.jl:86-88, 115-127.
    b = hydro_b_bisect(C)
    hydro_a = 2C * sqrt(b)
    ystar = 0.5
    hydro_q_star = hydro_a / (2C * sqrt(ystar + b))
    hydro_E_star = hydro_a * sqrt(ystar + b)
    q_phys = q_inj * hydro_q_star
    E_lat = hydro_E_star / H
    analytic = KrakeNN.ehd_hydrostatic_profiles(C, Ny; FT=Float64)
    @test isapprox(analytic.b, b; rtol=1e-12)
    @test isapprox(analytic.a, hydro_a; rtol=1e-12)

    mid_j = 1 + Int(round(ystar * H))
    q_init_K = p.q_inj * analytic.q_star[mid_j]
    Ey_init_K = delta_U * analytic.E_star[mid_j] / p.H
    @test isapprox(q_init_K, q_phys; rtol=1e-12)
    @test isapprox(Ey_init_K, E_lat; rtol=1e-12)

    # MATLAB Lattice.m:1238-1283 divides by rho; KrakeNN src/kernels/ehd_bc_2d.jl:182-199 stores q*E.
    a_M = q_phys * E_lat / rho_0
    force_density_K = q_init_K * Ey_init_K
    force_accel_K = force_density_K / rho_0
    @test isapprox(force_accel_K, a_M; rtol=1e-12)
    @test isapprox(force_density_K, q_phys * E_lat; rtol=1e-12)

    q_profile_M = [q_inj * hydro_a / (2C * sqrt((j - 1) / H + b)) for j in 1:Ny]
    E_profile_M = [hydro_a * sqrt((j - 1) / H + b) / H for j in 1:Ny]
    peak_force_M = maximum(abs.(q_profile_M .* E_profile_M ./ rho_0))
    peak_force_K = maximum(abs.(p.q_inj .* analytic.q_star .* (delta_U .* analytic.E_star ./ p.H) ./ rho_0))
    @test isapprox(peak_force_K, peak_force_M; rtol=1e-12)

    # MATLAB Lattice.m:291-318; KrakeNN src/kernels/collide_guo_2d.jl:107-140.
    S_srt_M = (1 / cs^2) * (1 - 1 / (2tau)) .* feq_ns(1.0, 0.0, 0.0) .* (a_M .* collect(CY))
    S_mrt_M = matlab_mrt_force(a_M, tau)
    S_bgk_K = kraken_bgk_force(a_M, tau)
    S_mrt_K = kraken_mrt_force(a_M, tau)
    @test isapprox(maximum(abs.(S_bgk_K)), maximum(abs.(S_srt_M)); rtol=1e-12)
    @test isapprox(maximum(abs.(S_mrt_K)), maximum(abs.(S_mrt_M)); rtol=1e-12)
end
