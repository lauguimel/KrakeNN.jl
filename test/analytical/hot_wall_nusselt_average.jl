# Hot-wall Nusselt number of the cavity drivers (v0.7). The hot wall is
# half-way (x = 0.5), each row covers one spacing of it, so the wall average
# is the mean over all rows 1:N; before the change rows 1 and N were left out.
#
# All gates are analytical identities (round-off only), CPU Float64 unless
# stated. Runtime: a few seconds.

using Test
using Kraken

@testset "Hot-wall Nusselt average (half-way walls)" begin
    rel(a, b) = abs(a - b) / abs(b)

    @testset "hot-wall Nusselt: conduction gives Nu = 1 on every row" begin
        # Linear conduction profile between the half-way walls x = 0.5 and
        # x = N + 0.5 (node i at x = i): T = T_hot - ΔT (x - 0.5) / N.
        for N in (3, 8, 33, 64), (T_hot, T_cold) in ((1.0, 0.0), (0.75, -0.25))
            ΔT = T_hot - T_cold
            T = [T_hot - ΔT * (i - 0.5) / N for i in 1:N, j in 1:N]
            rows = Kraken._hot_wall_nusselt_rows_2d(T, N, T_hot, ΔT)
            @test length(rows) == N
            @test maximum(abs.(rows .- 1)) <= 1e-12
            @test abs(Kraken._hot_wall_nusselt_2d(T, N, T_hot, ΔT) - 1) <= 1e-12
            @test abs(Kraken._gnf_hot_wall_nusselt(T, Float64(N), T_hot, ΔT) - 1) <= 1e-12
        end
        T32 = Float32[1 - (i - 0.5f0) / 16 for i in 1:16, j in 1:16]
        @test abs(Kraken._hot_wall_nusselt_2d(T32, 16, 1.0, 1.0) - 1) <= 1e-5
    end

    @testset "hot-wall Nusselt: second-order gradient, all rows averaged" begin
        # Row j: quadratic T = T_hot + b_j s + c_j s^2, s = x - 0.5, whose wall
        # gradient is b_j; the one-sided formula is exact for it, so
        # Nu_j = -N b_j / ΔT. The slopes differ on every row, end rows included.
        N = 12
        T_hot, ΔT = 1.0, 1.0
        b = [-(1 + 0.3 * j + 0.05 * j^2) / N for j in 1:N]
        c = [0.02 * sin(j) for j in 1:N]
        T = [T_hot + b[j] * (i - 0.5) + c[j] * (i - 0.5)^2 for i in 1:N, j in 1:N]
        expected = [-N * b[j] / ΔT for j in 1:N]
        rows = Kraken._hot_wall_nusselt_rows_2d(T, N, T_hot, ΔT)
        @test maximum(abs.(rows .- expected)) <= 1e-12 * maximum(abs, expected)
        Nu = Kraken._hot_wall_nusselt_2d(T, N, T_hot, ΔT)
        @test rel(Nu, sum(expected) / N) <= 1e-12
        # The end rows are in the average: dropping them changes it.
        @test rel(sum(expected[2:N-1]) / (N - 2), sum(expected) / N) > 1e-3
        @test rel(Nu, sum(expected[2:N-1]) / (N - 2)) > 1e-3
    end

    @testset "AD Nusselt QoIs evaluate the forward driver quantity" begin
        # nu_driver (defaults q_hot = 1/2, L = Nx) and nu_pure (wall fractions
        # read from the cut-link geometry) must equal the forward helper on
        # the same temperature field, including rows 1 and Ny.
        N = 10
        p = Kraken.ad_natconv_params(; N=N, Ra=1e3, Pr=0.71)
        geom = Kraken.ad_cavity_wall_geometry(p.Nx, p.Ny, Float64(N); q_hot=0.5)
        w = Kraken.ad_initial_thermal_w(p, geom.x_cold, geom.q_hot)
        goff = Kraken.ad_thermal_nlat(p)
        # Row-dependent perturbation of the thermal populations, largest on
        # the end rows, so that leaving them out would show.
        for j in 1:p.Ny, i in 1:p.Nx, q in 1:9
            w[goff + Kraken.ad_thermal_popidx(i, j, q, p.Nx, p.Ny)] +=
                1e-2 * (1 + abs(j - (p.Ny + 1) / 2)) * cos(0.7 * i + 0.3 * q)
        end
        T = [sum(w[goff + Kraken.ad_thermal_popidx(i, j, q, p.Nx, p.Ny)] for q in 1:9)
             for i in 1:p.Nx, j in 1:p.Ny]
        nu_fwd = Kraken._hot_wall_nusselt_2d(T, p.Nx, p.T_hot, p.T_hot - p.T_cold)
        nu_drv = Kraken.nu_driver(w, p)
        nu_pur = Kraken.nu_pure(w, geom.q_wall, p)
        @test rel(nu_drv, nu_fwd) <= 1e-12
        @test rel(nu_pur, nu_fwd) <= 1e-12
        rows = Kraken._hot_wall_nusselt_rows_2d(T, p.Nx, p.T_hot, p.T_hot - p.T_cold)
        @test rel(sum(rows[2:end-1]) / (p.Ny - 2), nu_fwd) > 1e-6
    end
end
