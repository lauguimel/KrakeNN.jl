# Optional central momentum convection (fv_steady_problem(...; convection = :central)), CPU Float64.
#
# Frozen gates (2026-11-13, set before the first run):
# - option contract: convection = :upwind is the default grid and gives a
#   residual equal (==) to the default problem (cavity, inclined immersed
#   channel, slip); unknown symbols throw; with conv = false (Stokes) the
#   central and upwind residuals are equal (==);
# - inclined immersed channel, Navier-Stokes, quadratic ghost + M3, central:
#   fixed walls and walls moving tangentially at U_s = 0.2: velocity order
#   >= 1.9 between consecutive grids h = 1/16 ... 1/128, L-infinity and RMS
#   (the upwind scheme measured 1.88 / 1.87 on the finest pair, 2026-11-12,
#   Aqua job 28097405; central override 1.99, job 28102707);
# - coloured dual Jacobian = dense ForwardDiff Jacobian <= 1e-12 max|J|, every
#   nonzero inside fv_jacobian_pattern, J v vs central FD <= 1e-6 (lid cavity
#   with power-law GNF; inclined immersed channel with cubic slip);
# - adjoint gradient vs central full-resolve FD, best relative <= 1e-6:
#   slip mobility b (inclined immersed channel) and uniform viscosity (lid
#   cavity), adjoint residual |J' λ - g_U| <= 1e-10 |g_U|;
# - stability: lid cavity 32², Re = 100 (cell Péclet U h / ν ≈ 3.1 > 2),
#   Newton from rest converges in <= 30 iterations. Re = 400 and 1000
#   (cell Péclet 12.5 and 31) are reported with continuation, not gated: the
#   central scheme has no numerical diffusion, so convergence and the absence
#   of cell-to-cell oscillation are not guaranteed there.
isdefined(@__MODULE__, :_isl_problem) || include(joinpath(@__DIR__, "incns_fv_slip_helpers.jl"))

const _CC_US = 0.2

function _cc_moving(r; uw = _CC_US, convection = :central, conv = true)
    Nx, Ny, h = _iw_grid(r)
    ib = fv_immersed_walls(Nx, Ny; walls = _iw_walls(), h, closure = :quadratic, mass = :M3,
        wall_velocity = (w, x, y) -> (uw * _IW_C, uw * _IW_S))
    return fv_steady_problem(Nx, Ny; h, conv, immersed = ib, wall_flux = 0.0, nu = _IW_NU,
        bc = (W = :wall, E = :wall, S = :wall, N = :wall),
        boundary_velocity = (side, x, y) -> _isl_velocity(x, y, uw),
        source = (x, y) -> _iw_force(x, y), convection)
end

_cc_cavity(N; Re = 100.0, convection = :central, kw...) =
    fv_steady_problem(N, N; h = 1 / N, nu = 1 / Re, convection,
        bc = (W = :wall, E = :wall, S = :wall, N = :wall),
        boundary_velocity = (side, x, y) -> (side === :N ? 1.0 : 0.0, 0.0), kw...)

function _cc_check_jacobian(p, U, rng; label)
    S = Matrix(fv_jacobian_pattern(p))
    J = Matrix(fv_jacobian!(fv_newton_solver(p), U))
    Jref = _sl_dense_jacobian(p, U)
    scale = max(1.0, maximum(abs, Jref))
    @test isempty(findall((.!iszero.(Jref)) .& .!S))
    @test maximum(abs, J - Jref) <= 1e-12 * scale
    v = randn(rng, length(U)); v ./= norm(v, Inf)
    hstep = 1e-6
    fd = (fv_residual(U + hstep * v, p) - fv_residual(U - hstep * v, p)) / (2hstep)
    Jv = J * v
    @test norm(Jv - fd, Inf) <= 1e-6 * max(norm(Jv, Inf), 1)
    @info "central Jacobian" label dual_vs_dense = maximum(abs, J - Jref) / scale fd = norm(Jv - fd, Inf) / norm(Jv, Inf)
end

function _cc_adjoint_fd(p, U0, θ, obj, pmap; label)
    s = fv_newton_solver(p)
    U = copy(U0)
    st = fv_newton!(s, U; atol = 1e-13, rtol = 1e-14, maxit = 30)
    @test st.converged || st.stagnated
    r = fv_steady_gradient(s, U, θ, obj; params = pmap)
    @test norm(transpose(s.J) * r.λ - r.gU) <= 1e-10 * max(norm(r.gU), eps())
    sfd = fv_newton_solver(p)
    best = Inf
    for hstep in (1e-3, 1e-4) .* max(norm(θ, Inf), 1e-3)
        vals = map((-1.0, 1.0)) do sgn
            fv_set_params!(sfd, pmap, θ .+ sgn * hstep)
            V = copy(U)
            fv_newton!(sfd, V; atol = 1e-13, rtol = 1e-14, maxit = 30)
            fv_objective(obj, V, sfd.prob)
        end
        fd = (vals[2] - vals[1]) / (2hstep)
        rel = abs(fd - r.grad[1]) / max(abs(r.grad[1]), eps())
        best = min(best, rel)
        @info "central adjoint FD" label hstep exact = r.grad[1] fd rel
    end
    @test best <= 1e-6
end

# vertical-centreline u of a cavity solution and its count of local extrema
function _cc_centreline(p, U)
    g = p.grid
    f = fv_fields(p, U)
    u = f[1][g.Nx ÷ 2, :]
    du = diff(u)
    return u, count(k -> du[k] * du[k+1] < 0, 1:length(du)-1)
end

@testset "FV central momentum convection" begin
    @testset "option contract (default bit-identical)" begin
        rng = MersenneTwister(4)
        for (p0, pu) in ((_cc_cavity(9; convection = :upwind), fv_steady_problem(9, 9; h = 1 / 9, nu = 0.01,
                              bc = (W = :wall, E = :wall, S = :wall, N = :wall),
                              boundary_velocity = (side, x, y) -> (side === :N ? 1.0 : 0.0, 0.0))),
                         (_isl_problem(0; conv = true, convection = :upwind), _isl_problem(0; conv = true)))
            @test p0.grid.convection == KrakeNN.FV_CONV_UPWIND
            U = randn(rng, fv_nunknowns(pu))
            @test p0.grid == pu.grid
            @test fv_residual(U, p0) == fv_residual(U, pu)
        end
        # the moving-wall problem with uw = 0 is the default inclined channel
        pz = _cc_moving(0; uw = 0.0, convection = :upwind); pd = _iw_problem(0; conv = true)
        U = _iw_exact_state(pd) .+ 0.05 .* randn(rng, fv_nunknowns(pd))
        @test fv_residual(U, pz) == fv_residual(U, pd)
        @test_throws ArgumentError _cc_cavity(8; convection = :quick)
        ps_c = _isl_problem(0; conv = false, convection = :central)
        ps_u = _isl_problem(0; conv = false)
        @test ps_c.grid.convection == KrakeNN.FV_CONV_CENTRAL
        U = randn(rng, fv_nunknowns(ps_u))
        @test fv_residual(U, ps_c) == fv_residual(U, ps_u)
        @test fv_jacobian_pattern(_isl_problem(0; conv = true, convection = :central)) ==
              fv_jacobian_pattern(_isl_problem(0; conv = true))
    end

    @testset "inclined channel order, $name walls" for (name, uw) in (("fixed", 0.0), ("moving", _CC_US))
        errors = []
        for r in 0:3
            p = _cc_moving(r; uw)
            U = _isl_state(p, uw)
            st = fv_newton!(fv_newton_solver(p), U; atol = 1e-12, rtol = 1e-13, maxit = 15)
            @test st.converged
            push!(errors, _isl_errors(p, U, uw))
            @info "central inclined channel" name r h = p.grid.h linf = errors[end].inf rms = errors[end].l2 iterations = st.iterations
        end
        for r in 1:3, metric in (:inf, :l2)
            order = _iw_order(getproperty(errors[r], metric), getproperty(errors[r+1], metric))
            @info "central inclined channel order" name grids = (r - 1, r) metric order
            @test order >= 1.9
        end
    end

    @testset "dual Jacobian" begin
        rng = MersenneTwister(21)
        p = _cc_cavity(10; Re = 50.0, rheology = PowerLaw(0.02, 0.6; nu_min = 0.0, nu_max = Inf), gnf_eps = 1e-2)
        _cc_check_jacobian(p, 0.3 .* randn(rng, fv_nunknowns(p)), rng; label = "cavity GNF")
        law = _SlipCubic(0.02, 0.002)
        p = _isl_problem(0; law, conv = true, convection = :central)
        U = _isl_state(p, _isl_us(law)) .+ 0.05 .* randn(rng, fv_nunknowns(p))
        _cc_check_jacobian(p, U, rng; label = "immersed cubic slip")
    end

    @testset "adjoint vs FD" begin
        b = 0.04
        p = _isl_problem(0; law = FVNavierSlip(b), conv = true, convection = :central)
        g = p.grid; i = g.Nx ÷ 2
        idx = Int[]; w = Float64[]
        for j in 1:g.Ny
            p.tag[i, j] == FV_CELL_SOLID && continue
            push!(idx, fv_index(g, i, j, 1)); push!(w, g.h * ((j - 0.5) * g.h - 0.75)^2)
        end
        _cc_adjoint_fd(p, _isl_state(p, _isl_us(FVNavierSlip(b))), [b], FVLinearFunctional(idx, w),
            FVProblemMap(fv_with_slip_parameters, 1); label = "slip b")
        pc = _cc_cavity(16; Re = 100.0)
        gc = pc.grid
        idc = [fv_index(gc, gc.Nx ÷ 2, j, 1) for j in 1:gc.Ny]
        U0 = zeros(fv_nunknowns(pc))
        fv_newton!(fv_newton_solver(pc), U0; atol = 1e-12, rtol = 1e-13, maxit = 30)
        _cc_adjoint_fd(pc, U0, [0.01], FVLinearFunctional(idc, fill(gc.h, gc.Ny)),
            FVUniformViscosity(); label = "cavity nu")
    end

    @testset "cell Péclet > 2: lid cavity 32²" begin
        U = nothing
        for Re in (100.0, 400.0, 1000.0)
            p = _cc_cavity(32; Re)
            s = fv_newton_solver(p)
            V = U === nothing ? zeros(fv_nunknowns(p)) : copy(U)
            st = fv_newton!(s, V; atol = 1e-10, rtol = 1e-12, maxit = 30)
            ok = st.converged && all(isfinite, V)
            _, extrema = ok ? _cc_centreline(p, V) : (nothing, -1)
            pu = _cc_cavity(32; Re, convection = :upwind)
            Vu = solve_incns_fv_newton(; problem = pu, maxit = 40)
            _, extrema_u = Vu.converged ? _cc_centreline(pu, Vu.U) : (nothing, -1)
            @info "central cavity" Re cell_peclet = Re / 32 converged = st.converged iterations = st.iterations centreline_extrema = extrema upwind_extrema = extrema_u
            Re == 100.0 && @test st.converged
            ok && (U = V)
        end
    end
end
