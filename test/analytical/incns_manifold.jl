# Analytical validation: localized inlet/outlet SIMPLE manifold flow.
#
# Rung (a): west-inlet/east-outlet channel, checked in the developed region
# against plane Poiseuille:
#   u(y) = 6 U_in (y/H)(1 - y/H),  Δp = 12 μ U_in Lx / H².
# Rung (b): same channel with one centred full-cell plate, checking symmetry,
# divergence, and zero velocity in solid cells.
# Rung (c): smoke handoff of the returned face fields to solve_scalar_transport.

using Test
using KernelAbstractions   # CPU() — no longer inherited once the includes below auto-skip

if !isdefined(@__MODULE__, :solve_incns_manifold)
    include(joinpath(@__DIR__, "..", "..", "src", "methods", "inc_ns", "manifold_flow.jl"))
end
if !isdefined(@__MODULE__, :solve_scalar_transport)
    include(joinpath(@__DIR__, "..", "..", "src", "methods", "scalar_transport", "thermal_transport.jl"))
end

const INCNS_MANIFOLD_RESULTS = Dict{Symbol,Any}()

function incns_manifold_poiseuille_case(; ny::Integer=16, aspect::Integer=8,
                                        Lx::Real=16.0, Ly::Real=1.0,
                                        U_in::Real=1.0, mu::Real=10.0,
                                        tol::Real=1e-7, maxiter::Integer=200,
                                        relax=(u=0.7, p=0.3),
                                        scheme::Symbol=:simplec,
                                        momentum_advection::Symbol=:linear_upwind,
                                        backend=CPU())
    ny = Int(ny)
    nx = Int(aspect) * ny
    res = solve_incns_manifold(; nx, ny, Lx, Ly, Re=U_in * Ly / mu,
                               U_in, mu,
                               inlet=(; side=:west, j0=1, j1=ny, u=U_in),
                               outlet=(; side=:east, j0=1, j1=ny),
                               relax, scheme, momentum_advection, tol, maxiter, backend)

    cols = Int(floor(0.75 * res.nx)):Int(floor(0.90 * res.nx))
    uprof = vec(sum(res.u[cols, :]; dims=1)) ./ length(cols)
    uan = [6.0 * U_in * (y / Ly) * (1.0 - y / Ly) for y in res.ycenters]
    l2_rel = sqrt(sum(abs2, uprof .- uan) / res.ny) /
             sqrt(sum(abs2, uan) / res.ny)
    maxv_developed = maximum(abs, res.v[cols, :])
    dp_an = 12.0 * mu * U_in * Lx / (Ly * Ly)
    dp_rel = abs(res.dp - dp_an) / dp_an

    return (; res, uprof, uan, l2_rel, maxv_developed, dp_an, dp_rel)
end

function incns_manifold_plate_case(; nx::Integer=64, ny::Integer=32,
                                   Lx::Real=4.0, Ly::Real=1.0,
                                   U_in::Real=1.0, mu::Real=10.0,
                                   scheme::Symbol=:simplec,
                                   momentum_advection::Symbol=:linear_upwind,
                                   backend=CPU())
    plates = [(; x0=1.75, x1=2.25, y0=0.375, y1=0.625)]
    is_solid = manifold_full_cell_mask(nx, ny, Lx, Ly, plates)
    res = solve_incns_manifold(; nx, ny, Lx, Ly, Re=U_in * Ly / mu,
                               U_in, mu, is_solid,
                               inlet=(; side=:west, j0=1, j1=ny, u=U_in),
                               outlet=(; side=:east, j0=1, j1=ny),
                               relax=(u=0.7, p=0.3), tol=1e-7,
                               maxiter=300, scheme, momentum_advection, backend)

    uwest = fill(Float64(U_in), ny)
    div = zeros(Float64, nx, ny)
    # Dual-mode: standalone include defines the helper in this module; under
    # runtests the include auto-skips and the helper lives in Kraken.
    facediv! = isdefined(@__MODULE__, :_mf_face_divergence!) ?
        _mf_face_divergence! : Kraken._mf_face_divergence!
    facediv!(div, res.uf, res.vf, uwest, res.dx, res.dy,
             nx, ny, res.is_solid)
    fluid = findall(!, res.is_solid)
    div_l2 = sqrt(sum(abs2, div[fluid]) / length(fluid))
    solid_speed = maximum(abs.(res.u[res.is_solid])) +
                  maximum(abs.(res.v[res.is_solid]))

    symmetry = let s = 0.0
        for j in 1:ny, i in 1:nx
            jm = ny + 1 - j
            if !res.is_solid[i, j] && !res.is_solid[i, jm]
                s = max(s, abs(res.u[i, j] - res.u[i, jm]),
                        abs(res.v[i, j] + res.v[i, jm]))
            end
        end
        s
    end

    return (; res, div_l2, solid_speed, symmetry)
end

@testset "IncNS manifold inlet/outlet Poiseuille" begin
    backend = CPU()
    c = incns_manifold_poiseuille_case(; backend)
    INCNS_MANIFOLD_RESULTS[:poiseuille] = c

    @test c.res.converged
    @test c.res.scheme === :simplec
    @test c.res.momentum_advection === :linear_upwind
    @test c.res.iters <= 200
    @test c.res.mass_imbalance < 1e-10
    @test c.l2_rel <= 0.01
    @test c.dp_rel <= 0.02
    @test c.maxv_developed < 1e-3
    @test c.res.checkerboard < 0.5

    coarse = incns_manifold_poiseuille_case(; ny=8, backend)
    order = log2(coarse.l2_rel / c.l2_rel)
    INCNS_MANIFOLD_RESULTS[:order] = order
    @test 1.7 <= order <= 2.3

    up = incns_manifold_poiseuille_case(; backend, momentum_advection=:upwind)
    INCNS_MANIFOLD_RESULTS[:poiseuille_upwind] = up
    @test up.res.momentum_advection === :upwind
    @test up.res.converged
    @test up.res.iters <= 200
end

@testset "IncNS manifold full-cell plate sanity and scalar handoff" begin
    backend = CPU()
    c = incns_manifold_plate_case(; backend)
    INCNS_MANIFOLD_RESULTS[:plate] = c

    @test c.res.converged
    @test c.res.momentum_advection === :linear_upwind
    @test c.res.mass_imbalance < 1e-10
    @test c.solid_speed == 0.0
    @test c.div_l2 < 1e-10
    @test c.symmetry < 1e-8

    st = solve_scalar_transport(; nx=c.res.nx, ny=c.res.ny,
                                dx=c.res.dx, dy=c.res.dy,
                                uf=c.res.uf, vf=c.res.vf,
                                DT=1e-2, is_solid=c.res.is_solid,
                                bc=(west=(kind=:dirichlet, value=1.0),
                                    east=(kind=:outflow, value=0.0),
                                    south=(kind=:flux, value=0.0),
                                    north=(kind=:flux, value=0.0)),
                                backend=nothing)
    INCNS_MANIFOLD_RESULTS[:scalar_transport] = st
    @test st.converged
    @test all(isfinite, st.T)
end

@testset "IncNS manifold legacy SIMPLE parity" begin
    c = incns_manifold_poiseuille_case(; ny=8, aspect=4, Lx=4.0,
                                       scheme=:simple,
                                       momentum_advection=:upwind, backend=CPU())
    @test c.res.scheme === :simple
    @test c.res.momentum_advection === :upwind
    @test c.res.converged
    @test c.res.iters == 30
    @test c.res.dp ≈ 492.7252026635531 rtol=1e-13
    @test sum(c.res.u) ≈ 255.75709873039895 rtol=1e-13
    @test sum(abs, c.res.v) ≈ 4.188177731722439 rtol=1e-13
    # 0.0 on macOS arm64, one ulp (2.2e-16) on Linux x86_64: round-off, not physics.
    @test c.res.mass_imbalance <= 1e-14

    @test_throws ArgumentError incns_manifold_poiseuille_case(;
        ny=4, aspect=2, Lx=2.0, momentum_advection=:quick)
end

let p = INCNS_MANIFOLD_RESULTS[:poiseuille],
    order = INCNS_MANIFOLD_RESULTS[:order],
    up = INCNS_MANIFOLD_RESULTS[:poiseuille_upwind],
    plate = INCNS_MANIFOLD_RESULTS[:plate]
    @info "incns manifold validation" profile_l2=p.l2_rel dp_rel=p.dp_rel order=order iters=p.res.iters upwind_iters=up.res.iters dp=p.res.dp upwind_dp=up.res.dp dp_shift=(p.res.dp - up.res.dp) / up.res.dp mass=p.res.mass_imbalance plate_div=plate.div_l2 plate_sym=plate.symmetry
end
