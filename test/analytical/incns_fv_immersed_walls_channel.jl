# Inclined plane channel through generic immersed profile walls (CPU Float64).
#
# Frozen gates (2026-11-12, analytical solutions in the helper header):
# - velocity order >= 1.8 between consecutive grids h = 1/16 ... 1/128, both
#   L-infinity and RMS, Stokes and Navier-Stokes, quadratic ghost + M3 (the
#   C5 acceptance gate of incns_fv_immersed_m3_convergence.jl);
# - temperature order >= 1.8 on the two finer pairs (>= 1.6 on the coarsest,
#   pre-asymptotic pair) for the manufactured Dirichlet field, wall temperature
#   given as a function on the cut points;
# - discrete thermal balance: sum of the thermal rows equals the side fluxes +
#   fv_immersed_wall_heat - sources, to 1e-10 relative (identity of the
#   residual, any state);
# - fv_immersed_wall_heat vs the exact diffusive wall heat: error decreases
#   with h (no rate frozen: the Cartesian face sum is a measurement).
isdefined(@__MODULE__, :_iw_problem) || include(joinpath(@__DIR__, "incns_fv_immersed_walls_helpers.jl"))

function _iw_solve(p)
    U = _iw_exact_state(p)
    s = fv_newton_solver(p)
    stats = p.energy === nothing ? fv_newton!(s, U; atol=1e-12, rtol=1e-13, maxit=10) :
                                   fv_newton!(s, U; maxit=20)
    return U, stats
end

# Side (W/E Dirichlet) thermal fluxes, written from the residual's side branch.
function _iw_side_heat(p, U)
    g = p.grid; e = p.energy; total = 0.0
    for j in 1:g.Ny, i in 1:g.Nx
        p.tag[i, j] == FV_CELL_SOLID && continue
        for d in 1:4
            KrakeNN._fv_energy_side_face(g, i, j, d) || continue
            kind, _, _ = KrakeNN._fv_face_kind(g, p.tag, i, j, d)
            kind == KrakeNN._FV_F_DIRICHLET || continue
            uw, vw = KrakeNN._fv_dirichlet_velocity(g, p.bvel, i, j, d)
            sx = (d == 1 || d == 3) ? 1 : -1
            F = g.h * sx * (d <= 2 ? uw : vw)
            thP = U[fv_index(g, i, j, 4)]
            thb = e.dirichlet[d] ? e.boundary[d] - e.Tref : thP
            total += max(F, 0) * thP + min(F, 0) * thb
            e.dirichlet[d] && (total += 2e.alpha * (thP - thb))
        end
    end
    return total
end

@testset "FV immersed profile walls: inclined channel" begin
    @testset "geometry contract" begin
        ib = _iw_geometry(0)
        Nx, Ny, h = _iw_grid(0)
        @test ib.surface !== nothing && ib.surface.open_walls
        cuts = count(>(0), ib.qcut)
        @test cuts > 0
        @test all(q -> 0 <= q <= 1, ib.qcut)
        # Every fluid/solid face carries a cut, with an in-domain fluid opposite cell.
        for j in 1:Ny, i in 1:Nx
            ib.tag[i, j] == FV_CELL_SOLID && continue
            for d in 1:4
                ii, jj = i + (d == 1) - (d == 2), j + (d == 3) - (d == 4)
                (1 <= ii <= Nx && 1 <= jj <= Ny && ib.tag[ii, jj] == FV_CELL_SOLID) || continue
                @test ib.qcut[d, i, j] > 0
                oi, oj = i - (d == 1) + (d == 2), j - (d == 3) + (d == 4)
                @test 1 <= oi <= Nx && 1 <= oj <= Ny && ib.tag[oi, oj] == FV_CELL_FLUID
            end
        end
        # cut points lie on the walls
        for j in 1:Ny, i in 1:Nx, d in 1:4
            q = ib.qcut[d, i, j]; q > 0 || continue
            x = (i - 0.5 + q * ((d == 1) - (d == 2))) * h
            y = (j - 0.5 + q * ((d == 3) - (d == 4))) * h
            k = ib.surface.wall[d, i, j]
            @test abs(y - ((k == 1 ? 0.25 : 0.75) + 0.25x)) <= 1e-13
        end
        p = _iw_problem(0)
        @test KrakeNN._fv_npairs(p.immersed) == 2
        # Wall too close to the north side is rejected (>= 3h contract).
        @test_throws Exception fv_immersed_walls(32, 24; h=1 / 16,
            walls=(FVProfileWall([0.0, 2.0], [1.3, 1.4]; solid=:above),))
    end

    for conv in (false, true)
        @testset "velocity order, conv=$conv" begin
            errors = []
            for r in 0:3
                p = _iw_problem(r; conv)
                U, stats = _iw_solve(p)
                @test stats.converged
                vel, _ = _iw_errors(p, U)
                push!(errors, vel)
                @info "inclined channel velocity" conv r h=p.grid.h linf=vel.inf rms=vel.l2
            end
            for r in 1:3, metric in (:inf, :l2)
                order = _iw_order(getproperty(errors[r], metric), getproperty(errors[r+1], metric))
                @info "inclined channel velocity order" conv grids=(r - 1, r) metric order
                @test order >= 1.8
            end
        end
    end

    @testset "temperature order and wall heat" begin
        errors = []; heat_errors = Float64[]
        exact_heat = [_iw_wall_heat_exact(1), _iw_wall_heat_exact(2)]
        for r in 0:3
            p = _iw_problem(r; conv=true, thermal=true)
            U, stats = _iw_solve(p)
            @test stats.converged
            vel, temp = _iw_errors(p, U)
            push!(errors, temp)
            heat = fv_immersed_wall_heat(U, p)
            @test length(heat) == 2
            push!(heat_errors, norm(heat - exact_heat) / norm(exact_heat))
            # Thermal balance identity of the residual (converged and perturbed state).
            g = p.grid
            rows = [fv_index(g, i, j, 4) for j in 1:g.Ny for i in 1:g.Nx if p.tag[i, j] != FV_CELL_SOLID]
            source = sum(g.h^2 * _iw_heat_source((i - 0.5) * g.h, (j - 0.5) * g.h)
                         for j in 1:g.Ny for i in 1:g.Nx if p.tag[i, j] != FV_CELL_SOLID)
            for V in (U, U .+ 1e-3 .* randn(MersenneTwister(31 + r), length(U)))
                R = fv_residual(V, p)
                balance = _iw_side_heat(p, V) + sum(fv_immersed_wall_heat(V, p)) - source
                scale = abs(source) + sum(abs, fv_immersed_wall_heat(V, p)) + abs(_iw_side_heat(p, V))
                @test abs(sum(R[rows]) - balance) <= 1e-10 * scale
            end
            @info "inclined channel temperature" r h=g.h linf=temp.inf rms=temp.l2 velocity_rms=vel.l2 wall_heat=Tuple(heat) exact_heat=Tuple(exact_heat) heat_rel=last(heat_errors)
        end
        for r in 1:3, metric in (:inf, :l2)
            order = _iw_order(getproperty(errors[r], metric), getproperty(errors[r+1], metric))
            @info "inclined channel temperature order" grids=(r - 1, r) metric order
            # Measured 2026-11-12 (Aqua CPU): 1.70/1.64, 1.86/1.83, 1.93/1.92 (Linf/RMS),
            # rising to 2: the coarsest pair (8 cells across the gap) is pre-asymptotic.
            # Coarsest-pair gate 1.6 set after measurement, approved by the maintainer 2026-11-12.
            @test order >= (r == 1 ? 1.6 : 1.8)
        end
        @test all(diff(heat_errors) .< 0)
    end
end
