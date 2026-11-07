using Test
using KrakeNN

# ===========================================================================
# Taylor-Green vortex decay on the SLBM path (Week 3 validation).
#
# Analytical solution on a doubly-periodic box of size L × L:
#
#   ux(x, y, t) = -u0 · cos(k·x) · sin(k·y) · exp(-2ν·k² · t)
#   uy(x, y, t) =  u0 · sin(k·x) · cos(k·y) · exp(-2ν·k² · t)
#
# with k = 2π / L. Decay is exponential in kinetic energy at rate
# Γ = 2ν·k²·2 = 4ν·k² (for both components, k² summed).
#
# This tests the *physics* of the SLBM path end-to-end: metric,
# departure, bilinear interpolation, collision, and periodic wrap.
# No boundary conditions are involved.
# ===========================================================================

function run_slbm_taylor_green(; N::Int=64, ν::Float64=0.01,
                                 u0::Float64=0.01,
                                 max_steps::Int=1000,
                                 collision::Symbol=:bgk)
    mesh = KrakeNN.build_mesh((ξ, η) -> (ξ * N, η * N);
                              Nξ=N, Nη=N,
                              periodic_ξ=true, periodic_η=true,
                              type=:custom)
    geom = build_slbm_geometry(mesh)

    k = 2π / N
    f_in = zeros(Float64, N, N, 9)
    ρ = ones(N, N); ux = zeros(N, N); uy = zeros(N, N)
    is_solid = zeros(Bool, N, N)

    for j in 1:N, i in 1:N
        x = Float64(i - 1); y = Float64(j - 1)
        ux_ij = -u0 * cos(k * x) * sin(k * y)
        uy_ij =  u0 * sin(k * x) * cos(k * y)
        ux[i, j] = ux_ij; uy[i, j] = uy_ij
        for q in 1:9
            f_in[i, j, q] = KrakeNN.equilibrium(D2Q9(), 1.0, ux_ij, uy_ij, q)
        end
    end

    f_out = similar(f_in)
    ω = 1.0 / (3ν + 0.5)

    for _ in 1:max_steps
        if collision === :bgk
            slbm_bgk_step!(f_out, f_in, ρ, ux, uy, is_solid, geom, ω)
        else
            slbm_mrt_step!(f_out, f_in, ρ, ux, uy, is_solid, geom, ν)
        end
        f_in, f_out = f_out, f_in
    end

    return (ρ=ρ, ux=ux, uy=uy)
end

@testset "SLBM Taylor-Green decay" begin

    @testset "BGK — L2 error < 5% after 1000 steps" begin
        N = 64
        ν = 0.01
        u0 = 0.01
        steps = 1000
        k = 2π / N
        decay = exp(-2 * ν * k^2 * steps)

        out = run_slbm_taylor_green(; N=N, ν=ν, u0=u0,
                                      max_steps=steps, collision=:bgk)

        ux_ana = zeros(N, N); uy_ana = zeros(N, N)
        for j in 1:N, i in 1:N
            x = Float64(i - 1); y = Float64(j - 1)
            ux_ana[i, j] = -u0 * cos(k * x) * sin(k * y) * decay
            uy_ana[i, j] =  u0 * sin(k * x) * cos(k * y) * decay
        end
        norm_num = sqrt(sum(out.ux .^ 2 .+ out.uy .^ 2))
        norm_ana = sqrt(sum(ux_ana .^ 2 .+ uy_ana .^ 2))
        l2 = sqrt(sum((out.ux .- ux_ana) .^ 2 .+ (out.uy .- uy_ana) .^ 2)) / norm_ana

        @info "SLBM Taylor-Green BGK: L2 = $(round(l2, digits=5)), |u|/|u|_ana = $(round(norm_num / norm_ana, digits=4))"
        @test l2 < 0.05

        # Velocity has actually decayed, not too little nor too much
        max_u = maximum(sqrt.(out.ux .^ 2 .+ out.uy .^ 2))
        @test max_u < u0
        @test max_u > 0.5 * u0 * decay
    end

    @testset "MRT — L2 error < 5% after 1000 steps" begin
        N = 64
        ν = 0.01
        u0 = 0.01
        steps = 1000
        k = 2π / N
        decay = exp(-2 * ν * k^2 * steps)

        out = run_slbm_taylor_green(; N=N, ν=ν, u0=u0,
                                      max_steps=steps, collision=:mrt)

        ux_ana = zeros(N, N); uy_ana = zeros(N, N)
        for j in 1:N, i in 1:N
            x = Float64(i - 1); y = Float64(j - 1)
            ux_ana[i, j] = -u0 * cos(k * x) * sin(k * y) * decay
            uy_ana[i, j] =  u0 * sin(k * x) * cos(k * y) * decay
        end
        norm_ana = sqrt(sum(ux_ana .^ 2 .+ uy_ana .^ 2))
        l2 = sqrt(sum((out.ux .- ux_ana) .^ 2 .+ (out.uy .- uy_ana) .^ 2)) / norm_ana

        @info "SLBM Taylor-Green MRT: L2 = $(round(l2, digits=5))"
        @test l2 < 0.05
    end

    @testset "Mildly stretched mesh — decay still bounded" begin
        # Mild stretching (tanh s=0.3, :both in y). On a non-uniform mesh
        # the effective viscosity varies locally (Krämer 2017 §4) and
        # a per-cell rescaling is needed for high-accuracy; see the
        # follow-up "variable-viscosity SLBM" work. For mild distortion
        # the error remains bounded and this regression test guards
        # against outright instability.
        N = 64
        ν = 0.02
        u0 = 0.01
        steps = 200
        k = 2π / N

        mesh = KrakeNN.build_mesh(
            (ξ, η) -> (ξ * N,
                       N * (tanh(0.3 * (2η - 1)) / tanh(0.3) + 1) / 2);
            Nξ=N, Nη=N,
            periodic_ξ=true, periodic_η=true)
        geom = build_slbm_geometry(mesh)

        f_in = zeros(N, N, 9)
        ρ = ones(N, N); ux = zeros(N, N); uy = zeros(N, N)
        is_solid = zeros(Bool, N, N)
        for j in 1:N, i in 1:N
            # Initialise at physical coordinates from the mesh (not indices)
            x = mesh.X[i, j]; y = mesh.Y[i, j]
            ux_ij = -u0 * cos(k * x) * sin(k * y)
            uy_ij =  u0 * sin(k * x) * cos(k * y)
            ux[i, j] = ux_ij; uy[i, j] = uy_ij
            for q in 1:9
                f_in[i, j, q] = KrakeNN.equilibrium(D2Q9(), 1.0, ux_ij, uy_ij, q)
            end
        end
        f_out = similar(f_in)
        ω = 1.0 / (3ν + 0.5)
        for _ in 1:steps
            slbm_bgk_step!(f_out, f_in, ρ, ux, uy, is_solid, geom, ω)
            f_in, f_out = f_out, f_in
        end

        decay = exp(-2 * ν * k^2 * steps)
        ux_ana = zeros(N, N); uy_ana = zeros(N, N)
        for j in 1:N, i in 1:N
            x = mesh.X[i, j]; y = mesh.Y[i, j]
            ux_ana[i, j] = -u0 * cos(k * x) * sin(k * y) * decay
            uy_ana[i, j] =  u0 * sin(k * x) * cos(k * y) * decay
        end
        norm_ana = sqrt(sum(ux_ana .^ 2 .+ uy_ana .^ 2))
        l2 = sqrt(sum((ux .- ux_ana) .^ 2 .+ (uy .- uy_ana) .^ 2)) / norm_ana
        @info "SLBM Taylor-Green mild-stretched: L2 = $(round(l2, digits=5))"
        # Not instability — mild distortion, bounded error
        @test l2 < 0.30
        @test all(isfinite.(ux))
        @test all(isfinite.(uy))
        # Velocity has decayed (not blown up)
        @test maximum(sqrt.(ux .^ 2 .+ uy .^ 2)) < 1.5 * u0
    end

    @testset "Biquadratic beats bilinear on stretched periodic" begin
        # Smooth flow without walls — the setting where biquadratic
        # should deliver its O(Δx³) advantage over bilinear O(Δx²).
        function l2_error(interp)
            N = 64; ν = 0.02; u0 = 0.01; steps = 200
            mesh = KrakeNN.build_mesh(
                (ξ, η) -> (ξ * N,
                           N * (tanh(0.3 * (2η - 1)) / tanh(0.3) + 1) / 2);
                Nξ=N, Nη=N, periodic_ξ=true, periodic_η=true)
            geom = build_slbm_geometry(mesh)
            f_in = zeros(Float64, N, N, 9)
            ρ = ones(N, N); ux = zeros(N, N); uy = zeros(N, N)
            is_solid = zeros(Bool, N, N)
            k = 2π / N
            for j in 1:N, i in 1:N
                x = mesh.X[i, j]; y = mesh.Y[i, j]
                uxi = -u0 * cos(k * x) * sin(k * y)
                uyi =  u0 * sin(k * x) * cos(k * y)
                for q in 1:9
                    f_in[i, j, q] = KrakeNN.equilibrium(D2Q9(), 1.0, uxi, uyi, q)
                end
            end
            f_out = similar(f_in)
            ω = 1.0 / (3ν + 0.5)
            for _ in 1:steps
                slbm_bgk_step!(f_out, f_in, ρ, ux, uy, is_solid, geom, ω;
                               interp=interp)
                f_in, f_out = f_out, f_in
            end
            decay = exp(-2 * ν * k^2 * steps)
            ux_ana = [-u0 * cos(k * mesh.X[i, j]) * sin(k * mesh.Y[i, j]) * decay
                       for i in 1:N, j in 1:N]
            uy_ana = [ u0 * sin(k * mesh.X[i, j]) * cos(k * mesh.Y[i, j]) * decay
                       for i in 1:N, j in 1:N]
            nrm = sqrt(sum(ux_ana .^ 2 .+ uy_ana .^ 2))
            return sqrt(sum((ux .- ux_ana) .^ 2 .+ (uy .- uy_ana) .^ 2)) / nrm
        end
        l2_bi = l2_error(:bilinear)
        l2_bq = l2_error(:biquadratic)
        @info "Taylor-Green stretched: bilinear L2=$(round(l2_bi, digits=5)) vs biquadratic L2=$(round(l2_bq, digits=5))"
        @test l2_bq < l2_bi
        @test l2_bq < 0.5 * l2_bi    # biquadratic ≥ 2× better than bilinear
    end

end
