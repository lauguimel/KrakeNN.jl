# FV wall slip: dual Jacobian and slip-parameter adjoint (CPU Float64).
#
# Frozen gates (2026-11-12, set before the first run):
# - coloured dual Jacobian = dense ForwardDiff Jacobian <= 1e-12 * max|J|, and
#   every nonzero of the dense Jacobian lies in fv_jacobian_pattern (same
#   contract as incns_fv_immersed_walls_derivatives.jl): grid-aligned slip
#   (domain sides + masked solid, moving lid, power-law GNF, Navier-Stokes) and
#   immersed slip (inclined channel, quadratic+M3 / linear+M1, Navier and cubic
#   laws, Newtonian and power-law GNF);
# - central FD of the residual along a random direction: J*v - FD <= 1e-6 |J*v|;
# - adjoint gradient of the slip parameters vs central full-resolve FD: best
#   relative <= 1e-6; adjoint residual |J' λ - g_U| <= 1e-10 |g_U|; plane
#   channel flow rate: dQ/db = G H²/2 exactly (<= 1e-10 relative).
isdefined(@__MODULE__, :_isl_problem) || include(joinpath(@__DIR__, "incns_fv_slip_helpers.jl"))

function _sld_check_jacobian(p, U, rng; label)
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
    @info "slip Jacobian" label unknowns = length(U) dual_vs_dense = maximum(abs, J - Jref) / scale fd = norm(Jv - fd, Inf) / norm(Jv, Inf)
end

function _sld_solve!(s, U)
    st = fv_newton!(s, U; atol = 1e-13, rtol = 1e-14, maxit = 30)
    st.converged || st.stagnated || error("Newton did not converge")
    return st
end

# adjoint gradient vs central FD of the converged objective along `direction`
function _sld_adjoint_fd(p, U0, θ, obj, pmap, directions; label)
    s = fv_newton_solver(p)
    U = copy(U0)
    _sld_solve!(s, U)
    r = fv_steady_gradient(s, U, θ, obj; params = pmap)
    @test norm(transpose(s.J) * r.λ - r.gU) <= 1e-10 * max(norm(r.gU), eps())
    sfd = fv_newton_solver(p)
    for (dname, direction) in directions
        exact = dot(r.grad, direction)
        best = Inf
        for hstep in (1e-3, 1e-4) .* max(norm(θ, Inf), 1e-3)
            vals = map((-1.0, 1.0)) do sgn
                fv_set_params!(sfd, pmap, θ .+ sgn * hstep .* direction)
                V = copy(U)
                _sld_solve!(sfd, V)
                fv_objective(obj, V, sfd.prob)
            end
            fd = (vals[2] - vals[1]) / (2hstep)
            rel = abs(fd - exact) / max(abs(exact), eps())
            best = min(best, rel)
            @info "slip adjoint FD" label dname hstep exact fd rel
        end
        fv_set_params!(sfd, pmap, θ)
        @test best <= 1e-6
    end
    return r
end

@testset "FV wall slip: derivatives" begin
    @testset "dual Jacobian, grid-aligned slip (sides, solid, lid, GNF, NS)" begin
        solid = falses(12, 10); solid[6:7, 1:4] .= true
        p = fv_steady_problem(12, 10; h = 0.1, conv = true, solid,
            bc = (W = :wall, E = :wall, S = :wall, N = :wall),
            boundary_velocity = (side, x, y) -> (side === :N ? 1.0 : 0.0, 0.0),
            rheology = PowerLaw(0.1, 0.5; nu_min = 0.0, nu_max = Inf), gnf_eps = 1e-2,
            slip = FVSlipWalls(_SlipCubic(0.3, 0.5); sides = (:N, :S, :W), solid = true))
        rng = MersenneTwister(11)
        U = 0.3 .* randn(rng, fv_nunknowns(p))
        _sld_check_jacobian(p, U, rng; label = "aligned")
    end

    @testset "dual Jacobian, immersed slip $label" for (label, kw) in (
            ("quadratic M3 Navier", (; law = FVNavierSlip(0.04), conv = true)),
            ("linear M1 cubic", (; law = _SlipCubic(0.02, 0.002), conv = true, closure = :linear, mass = :M1)),
            ("quadratic M3 cubic GNF", (; law = _SlipCubic(0.02, 0.002), conv = true,
                                        rheology = PowerLaw(0.5, 0.7; nu_min = 0.0, nu_max = Inf), eps = 1e-2)))
        p = _isl_problem(0; kw...)
        rng = MersenneTwister(2027)
        U = _isl_state(p, _isl_us(kw.law)) .+ 0.05 .* randn(rng, fv_nunknowns(p))
        _sld_check_jacobian(p, U, rng; label)
    end

    @testset "adjoint, plane channel flow rate" begin
        b = 0.1
        p = _sl_channel(16; law = FVNavierSlip(b), conv = true)
        obj = fv_flow_rate(p, 1)
        pm = FVProblemMap(fv_with_slip_parameters, 1)
        r = _sld_adjoint_fd(p, zeros(fv_nunknowns(p)), [b], obj, pm, (("b", [1.0]),); label = "channel Navier")
        @test abs(r.grad[1] - _SL_G * _SL_H^2 / 2) <= 1e-10 * _SL_G * _SL_H^2 / 2
        θ = [0.05, 0.4]
        pc = _sl_channel(16; law = _SlipCubic(θ...), conv = true)
        mapc = FVProblemMap(fv_with_slip_parameters, 2)
        _sld_adjoint_fd(pc, zeros(fv_nunknowns(pc)), θ, fv_flow_rate(pc, 1), mapc,
            (("b", [1.0, 0.0]), ("random", normalize(randn(MersenneTwister(8), 2))));
            label = "channel cubic")
    end

    @testset "adjoint, inclined immersed channel" begin
        b = 0.04
        p = _isl_problem(0; law = FVNavierSlip(b), conv = true)
        g = p.grid; i = g.Nx ÷ 2
        idx = Int[]; w = Float64[]
        for j in 1:g.Ny
            p.tag[i, j] == FV_CELL_SOLID && continue
            push!(idx, fv_index(g, i, j, 1)); push!(w, g.h * ((j - 0.5) * g.h - 0.75)^2)
        end
        obj = FVLinearFunctional(idx, w)
        pm = FVProblemMap(fv_with_slip_parameters, 1)
        _sld_adjoint_fd(p, _isl_state(p, _isl_us(FVNavierSlip(b))), [b], obj, pm, (("b", [1.0]),);
            label = "inclined Navier")
    end
end

# CPU inference guard of the slip residual kernel (not a GPU validation): the
# local slip solve used to iterate on duals nested inside the state duals,
# which does not infer and made the GPU compiler reject the kernel
# (InvalidIRError, Aqua A100 2026-11-14). Same check as the M3 kernels.
isdefined(@__MODULE__, :_fvc4_kernel_issues) ||
    include(joinpath(@__DIR__, "incns_fv_adjoint_compiler.jl"))
@testset "FV wall slip residual kernel inference" begin
    DJ = ForwardDiff.Dual{FVJacobianTag,Float64,4}
    DP = ForwardDiff.Dual{KrakeNN.FVParamTag,Float64,2}
    cases = (("channel Navier", _sl_channel(16; law = FVNavierSlip(0.1), conv = true)),
             ("immersed cubic central", _isl_problem(0; law = _SlipCubic(0.02, 0.002), conv = true,
                                                     convection = :central)))
    for (label, p) in cases
        qcut, wvel, closure = KrakeNN._fv_immersed_args(p.immersed)
        fp = KrakeNN._fv_face_pair(p.immersed)
        n = fv_nunknowns(p)
        θ = fv_slip_parameters(p.slip.law)
        pθ = fv_with_slip_parameters(p, DP.(θ, Ref(ForwardDiff.Partials((1.0, 0.0)))))
        # state duals (Jacobian columns), then slip-law parameter duals (adjoint)
        for (payload, U, q) in (("state", fill(DJ(0.1), n), p), ("law parameters", fill(0.1, n), pθ))
            @testset "$label $payload" begin
                T = promote_type(eltype(U), KrakeNN._fv_slip_eltype(q.slip))
                args = (zeros(T, n), U, q.tag, q.nu, q.bvel, q.src, q.grid, q.visc,
                        qcut, wvel, closure, fp, q.slip)
                @test isempty(_fvc4_kernel_issues(KrakeNN._fv_slip_residual_kernel!, args,
                                                  (q.grid.Nx, q.grid.Ny)))
            end
        end
    end
end
