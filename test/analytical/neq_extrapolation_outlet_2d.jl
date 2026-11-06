# East pressure outlet of the 2D rebuild path (issue #118).
#
# Empty channel: half-way walls on south/north, Zou-He velocity inlet with the
# Poiseuille profile of walls at j = 1/2 and j = Ny + 1/2, pressure outlet at
# i = Nx, production fused TRT step + `apply_bc_rebuild_2d!`. The initial state
# is that profile with a seeded random perturbation of density and velocity.
# A steady outlet must forget the perturbation: the consecutive-step change of
# the outlet column, max |f(Nx, :, :)^{n+1} - f(Nx, :, :)^n|, must fall to
# round-off.
#
# Checked:
#   (a) `NeqExtrapolationPressure`: the outlet change after 10 000 steps is at
#       round-off (gate 1e-13; measured 6.2e-14, CPU, 2026-11-06, macOS arm64,
#       Julia 1.12.5); the outlet density equals ρ_out and the outlet raw
#       velocity equals that of column Nx-1, including rows 1 and Ny
#       (gate 1e-14 each; measured <= 2.3e-16); the per-node-rate dispatch (`sp_field`,
#       `sm_field`) gives the same state as the uniform-rate one (bitwise,
#       uniform fields).
#   (b) Sensitivity of the check: with `ZouHePressure` east, regularized TRT
#       weights of #118 included, the same outlet change stays at 3.4e-4 after
#       10 000 steps and grows (3.2e-4 at 5 000 steps), the sign-flipping
#       outlet mode documented on `ZouHePressure`. The assertion below only
#       requires it to stay above 1e-6. If that closure is ever made to
#       converge on this path, update (b), not (a).
#
# Cost: under 2 s on CPU (32 x 12 nodes, 2 x 10 000 steps).

using Test
using Random
using Kraken

const _NEQOUT_GATE = 1e-13
const _NEQOUT_MOMENT_GATE = 1e-14

function _neqout_channel(east; Nx=32, Ny=12, ν=0.02, Umax=0.04, pert=1e-2,
                         nsteps=10_000, sp_field=nothing, sm_field=nothing)
    T = Float64
    is_solid = falses(Nx, Ny)
    q_wall = zeros(T, Nx, Ny, 9)
    uw = zeros(T, Nx, Ny, 9)
    u_prof = T[4Umax * (j - 0.5) * (Ny + 0.5 - j) / Ny^2 for j in 1:Ny]
    rng = MersenneTwister(118)
    f_in = zeros(T, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        r = 1 + pert * Umax * (2rand(rng) - 1)
        a = u_prof[j] * (1 + pert * (2rand(rng) - 1))
        b = pert * Umax * (2rand(rng) - 1)
        usq = a * a + b * b
        for q in 1:9
            f_in[i, j, q] = Kraken.feq_2d(Val(q), r, a, b, usq)
        end
    end
    f_out = zeros(T, Nx, Ny, 9)
    ρ = ones(T, Nx, Ny); ux = zeros(T, Nx, Ny); uy = zeros(T, Nx, Ny)
    bc = BCSpec2D(; west=ZouHeVelocity(u_prof), east=east)
    d_half = 0.0
    for step in 1:nsteps
        fused_trt_libb_v2_step!(f_out, f_in, ρ, ux, uy, is_solid, q_wall, uw, uw,
                                Nx, Ny, T(ν))
        apply_bc_rebuild_2d!(f_out, f_in, bc, ν, Nx, Ny;
                             sp_field=sp_field, sm_field=sm_field)
        step == nsteps ÷ 2 &&
            (d_half = maximum(abs.(view(f_out, Nx, :, :) .- view(f_in, Nx, :, :))))
        step < nsteps && ((f_in, f_out) = (f_out, f_in))
    end
    d_end = maximum(abs.(view(f_out, Nx, :, :) .- view(f_in, Nx, :, :)))
    return (; f=f_out, d_half, d_end, Nx, Ny, ν)
end

function _neqout_moments(f, i, j)
    g = ntuple(q -> f[i, j, q], 9)
    return Kraken.moments_2d(g...)
end

@testset "East pressure outlet of the 2D rebuild path (#118)" begin
    @testset "NeqExtrapolationPressure reaches round-off" begin
        r = _neqout_channel(NeqExtrapolationPressure(1.0))
        @test r.d_end <= _NEQOUT_GATE
        drho = 0.0; du = 0.0
        for j in 1:r.Ny
            ρo, uxo, uyo = _neqout_moments(r.f, r.Nx, j)
            _, uxm, uym = _neqout_moments(r.f, r.Nx - 1, j)
            drho = max(drho, abs(ρo - 1.0))
            du = max(du, abs(uxo - uxm), abs(uyo - uym))
        end
        @test drho <= _NEQOUT_MOMENT_GATE
        @test du <= _NEQOUT_MOMENT_GATE
        @info "NeqExtrapolationPressure outlet" d_end = r.d_end drho du

        # Per-node-rate dispatch, uniform fields: same state bit for bit.
        sp, sm = Kraken.trt_rates(r.ν; Λ=3/16)
        rl = _neqout_channel(NeqExtrapolationPressure(1.0); nsteps=200,
                             sp_field=fill(sp, r.Nx, r.Ny), sm_field=fill(sm, r.Nx, r.Ny))
        ru = _neqout_channel(NeqExtrapolationPressure(1.0); nsteps=200)
        @test rl.f == ru.f
    end

    @testset "ZouHePressure east keeps the outlet mode (sensitivity)" begin
        r = _neqout_channel(ZouHePressure(1.0))
        @test r.d_end > 1e-6
        @info "ZouHePressure outlet" d_half = r.d_half d_end = r.d_end
    end

    @testset "Post-collision source, no collision, included end rows" begin
        Nx, Ny = 8, 6
        rng = MersenneTwister(1182)
        post = 0.05 .+ 0.01 .* rand(rng, Nx, Ny, 9)
        # Distinct pre-step data catches reading the wrong time level.
        pre = 0.1 .+ 0.01 .* rand(rng, Nx, Ny, 9)
        rho_out = 1.01
        expected = copy(post)
        for j in 1:Ny
            rho, ux, uy = _neqout_moments(post, Nx - 1, j)
            for q in 1:9
                expected[Nx, j, q] = post[Nx - 1, j, q] +
                    Kraken.feq_2d(Val(q), rho_out, ux, uy, ux^2 + uy^2) -
                    Kraken.feq_2d(Val(q), rho, ux, uy, ux^2 + uy^2)
            end
        end
        # Half-way walls and block interfaces both include the end rows.
        # The round-off bound follows the analytic extrapolation formula above.
        for adjacent in (HalfwayBB(), InterfaceBC())
            bc = BCSpec2D(; east=NeqExtrapolationPressure(rho_out),
                           south=adjacent, north=adjacent)
            actual = copy(post)
            apply_bc_rebuild_2d!(actual, pre, bc, 0.02, Nx, Ny)
            @test maximum(abs.(actual .- expected)) <= _NEQOUT_MOMENT_GATE
            @test maximum(abs.(actual[Nx, [1, Ny], :] .-
                               expected[Nx, [1, Ny], :])) <= _NEQOUT_MOMENT_GATE
            @test actual[1:Nx-1, :, :] == post[1:Nx-1, :, :]
            local_rates = copy(post)
            apply_bc_rebuild_2d!(local_rates, pre, bc, 0.1, Nx, Ny;
                sp_field=0.2 .+ rand(rng, Nx, Ny), sm_field=0.3 .+ rand(rng, Nx, Ny))
            @test local_rates == actual
        end
    end

    @testset "Corner rows next to a Zou-He face stay excluded" begin
        Nx, Ny = 8, 6
        rng = MersenneTwister(126)
        post = 0.05 .+ 0.01 .* rand(rng, Nx, Ny, 9)
        pre = 0.1 .+ 0.01 .* rand(rng, Nx, Ny, 9)
        east = NeqExtrapolationPressure(1.01)
        backend = Kraken.KernelAbstractions.get_backend(post)
        # Exercise each endpoint independently, including mixed face types.
        for (south, north, rows) in (
            (ZouHePressure(1.0), HalfwayBB(), 2:Ny),
            (InterfaceBC(), ZouHeVelocity(zeros(Nx)), 1:Ny-1),
            (ZouHeVelocity(zeros(Nx)), ZouHePressure(1.0), 2:Ny-1),
        )
            expected = copy(post)
            for j in rows
                rho, ux, uy = _neqout_moments(post, Nx - 1, j)
                for q in 1:9
                    expected[Nx, j, q] = post[Nx - 1, j, q] +
                        Kraken.feq_2d(Val(q), east.ρ_out, ux, uy, ux^2 + uy^2) -
                        Kraken.feq_2d(Val(q), rho, ux, uy, ux^2 + uy^2)
                end
            end
            # Isolate the east dispatch so the adjacent face's own rebuild
            # cannot alter the reference source column after extrapolation.
            actual = copy(post)
            Kraken._apply_bc_2d_east!(backend, actual, pre, east, 0.8, 1.1, Nx, Ny;
                south_bc=south, north_bc=north)
            @test maximum(abs.(actual .- expected)) <= _NEQOUT_MOMENT_GATE
            for j in setdiff([1, Ny], rows)
                @test actual[Nx, j, :] == post[Nx, j, :]
            end
            @test actual[1:Nx-1, :, :] == post[1:Nx-1, :, :]
            local_rates = copy(post)
            Kraken._apply_bc_2d_east_local!(backend, local_rates, pre, east,
                fill(0.7, Nx, Ny), fill(1.2, Nx, Ny), Nx, Ny;
                south_bc=south, north_bc=north)
            @test local_rates == actual
        end
    end

    @testset "NeqExtrapolationPressure argument checks" begin
        f = zeros(8, 6, 9)
        @test_throws ArgumentError apply_bc_rebuild_2d!(
            copy(f), f, BCSpec2D(; west=NeqExtrapolationPressure(1.0)), 0.1, 8, 6)
        @test_throws ArgumentError apply_bc_rebuild_2d!(
            copy(f), f, BCSpec2D(; north=NeqExtrapolationPressure(1.0)), 0.1, 8, 6)
        g = zeros(2, 6, 9)
        @test_throws ArgumentError apply_bc_rebuild_2d!(
            copy(g), g, BCSpec2D(; east=NeqExtrapolationPressure(1.0)), 0.1, 2, 6)
    end
end
