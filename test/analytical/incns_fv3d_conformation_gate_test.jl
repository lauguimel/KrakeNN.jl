# Derivative gate of the steady 3D FV conformation adjoint (plan "discrete
# viscoelastic adjoints", step F2), square-duct slice 1 x N x N, Float64, with
# FVRelaxationParameters(:lambda, :nu_p, :nu_s[, :theta]).
#
# Coverage matrix (only these combinations are claimed verified):
#   law               parameters               CPU N=8   GPU N=8 (G4)   GPU N=24
#   Oldroyd-B 0.5     lambda, nu_p, nu_s          yes       yes            no
#   Giesekus 0.1, 1   lambda, nu_p, nu_s, alpha   yes       yes            lambda, alpha (full)
#   linear PTT 0.5    lambda, nu_p, nu_s, eps     yes       yes            no
#   exp. PTT 0.5      lambda, nu_p, nu_s, eps     yes       yes            no
#   FENE-P 0.5        lambda, nu_p, nu_s, L2      yes       yes            no
#   learned law 0.5   4 weights + 1 random dir.   yes       yes            1 weight (full)
# (the number after the law is De = lambda U / H)
#
# Objective (plan R2, fv3d_conformation_gate_setup.jl): FVPIVObjective of a
# stereo x-normal plane (axial and secondary velocities weighted separately) +
# (Fx - Fx*)² term + (N1/G - N1*/G)² term (section mean of c_xx - c_yy, an
# FVLinearFunctional), data = state converged at θ* = θ + 0.1 s (additive).
# States converged from rest (continuation in lambda), ||R||_inf <= 1e-12 on the
# CPU, conformation domain checked. Per row:
#   precheck  every sensitivity expected nonzero is: |g_k| > 1e-8 |J| / s_k;
#   G1        tangent vs adjoint along each parameter and a random direction:
#             |<g, dθ> - (gU . dU + explicit . dθ)| <= 1e-10 |<g, dθ>| + 1e-14
#             (consistency only: both share J and dR/dθ);
#   G2        centred FD of the reconverged objective, steps VE_G2_STEPS s_k,
#             min error <= VE_G2_REL |g_k| + VE_G2_ABS |J| / s_k (constants of the
#             setup file, set to the plan's target and confirmed by the step
#             study benchmarks/fv3d/ve_adjoint_step_study.jl);
#   G3        dual chunk 2 and 8 vs 4: <= 1e-12 |g_k| + 1e-15 (with 3-4
#             parameters chunk 4 and 8 are both one pass, chunk 2 splits it);
#   G4 (GPU)  state solved on the GPU from rest, same host data:
#             |g_GPU - g_CPU| <= 1e-9 |g_CPU| + 1e-13 |J| / s_k.
# Learned law (step F4, NeuralRelaxation{4} around Giesekus alpha0 = 0.3, all
# 28 weights mapped, s = 1): at the start point (zero output weights, the law is
# Giesekus exactly: residual bit-identical) the output derivatives are nonzero
# (> 1e-6 ||grad J||) and the hidden ones exactly zero (expected), G1-G3 along
# two output weights; at the second point (output weights N(0, 0.05²), same
# seed) G1-G3 along two hidden and two output weights and a random direction.
# Oldroyd-B (R10): in the developed straight slice velocity and Fx depend on
# nu_s + nu_p only, so (a) without the conformation term |dJ/dlambda| <= 1e-10
# ||grad J|| (expected zero) while nu_p and nu_s stay nonzero; (b) the
# conformation term makes lambda visible and the row passes G1-G3.
#
# KRAKEN_VE_ADJOINT_FULL=1 adds the GPU N = 24 rows: Giesekus (lambda, alpha at
# De 0.1 and 1) and the learned law (one hidden weight, second point), G1-G3. GPU parts are self-gated like
# incns_fv3d_conformation_adjoint_test.jl (KRAKEN_REQUIRE_GPU=1 turns the skip
# into a failure).
using Test
using LinearAlgebra
using Random
using KrakeNN

include(joinpath(@__DIR__, "fv3d_conformation_gate_setup.jl"))

const _CG_GPU_OK = (try
    @eval Main using CUDA, CUDSS
    @eval Main CUDA.functional()
catch
    false
end) === true
const _CG_REQUIRE_GPU = get(ENV, "KRAKEN_REQUIRE_GPU", "0") in ("1", "true")
const _CG_FULL = get(ENV, "KRAKEN_VE_ADJOINT_FULL", "0") in ("1", "true")

# G1 + G2 + G3 of one (row, map) on one backend; returns the gradient.
# `checked`: parameter indices of the precheck, G1 and G2 (default all);
# G1 also runs along a random direction over all parameters, and G2 along it
# with `g2_random` (step h max(s), direction scaled to unit max norm).
function _cg_check(label, s, U, m, θ, sk, names, obj; to = Array, linsolve = CPUBackendTag(),
                   g3 = true, expect_zero = Symbol[], checked = eachindex(θ), g2_random = false)
    ws = fv_adjoint_workspace(s, m; chunk = 4)
    res = fv_steady_gradient(s, U, θ, obj; params = m, ws)
    grad = copy(res.grad)
    gU = Array(copy(res.gU))                  # obj.gU is overwritten by later evaluations
    explicit = copy(res.explicit)
    J = res.J
    @test all(isfinite, grad)
    @test res.residual_inf <= 1e-12
    @test J > 0
    # precheck: nonzero sensitivities
    for k in checked
        names[k] in expect_zero && continue
        ok = abs(grad[k]) > 1e-8 * J / sk[k]
        ok || @info "zero sensitivity" label name = names[k] g = grad[k] J
        @test ok
    end
    # G1
    rd = randn(MersenneTwister(13), length(θ))
    rd ./= norm(rd, Inf)
    dirs = [[(Float64.(1:length(θ) .== k), String(names[k])) for k in checked]; (rd, "random")]
    for (δθ, name) in dirs
        dU = fv_steady_tangent(s, U, θ, δθ; params = m, ws)
        lhs = dot(grad, δθ)
        e = abs(lhs - (dot(gU, Array(dU)) + dot(explicit, δθ)))
        @info "G1 tangent vs adjoint" label name lhs err = e
        @test e <= 1e-10 * abs(lhs) + 1e-14
    end
    # G2
    sf = fv_newton_solver(KrakeNN._fv_host_problem(s.prob); to, linsolve)
    g2 = [(Float64.(1:length(θ) .== k), String(names[k]), sk[k]) for k in checked if !(names[k] in expect_zero)]
    g2_random && push!(g2, (rd, "random", maximum(sk)))
    for (d, name, sd) in g2
        gd = dot(grad, d)
        errs = Float64[]
        for h in VE_G2_STEPS
            fd, _ = ve_fd(sf, m, θ, d, h, sd, U, obj)
            push!(errs, abs(fd - gd))
        end
        tol = VE_G2_REL * abs(gd) + VE_G2_ABS * J / sd
        @info "G2 adjoint vs centred FD" label name g = gd err = minimum(errs) rel = minimum(errs) / abs(gd) tol
        @test minimum(errs) <= tol
    end
    # G3
    if g3
        for c in (2, 8)
            gc = fv_steady_gradient(s, U, θ, obj; params = m, ws = fv_adjoint_workspace(s, m; chunk = c)).grad
            @test all(abs.(gc .- grad) .<= 1e-12 .* abs.(grad) .+ 1e-15)
        end
    end
    return grad, J
end

@testset "FV 3D conformation: derivative gate (F2)" begin
    rows = ve_rows()
    cpu = Dict{String,Any}()
    @testset "CPU N=8: $(r.label)" for r in rows
        s, U = ve_state(r)
        @test maximum(abs, fv_residual(U, s.prob)) <= 1e-12
        Ud = ve_data_state(r, s, U)
        m = ve_map(r); θ = ve_theta(r); sk = ve_scales(r); names = ve_names(r)
        obj, info = ve_objective(s.prob, Ud)
        @test info.sa > 0 && info.Fs != 0
        if r.par === nothing
            # Oldroyd-B (R10 a): lambda invisible to velocity + Fx
            objv, _ = ve_objective(s.prob, Ud; conformation = false)
            gv = fv_steady_gradient(s, U, θ, objv; params = m).grad
            @info "Oldroyd-B velocity + Fx gradient (lambda expected zero)" gv
            @test abs(gv[1]) <= 1e-10 * norm(gv)
            @test all(abs.(gv[2:3]) .> 1e-8 * ve_J(objv, U, s.prob) ./ sk[2:3])
        elseif r.par[1] === :alpha
            @test info.ss > 1e-12 * info.sa          # Giesekus: secondary flow in the data
        end
        grad, J = _cg_check(r.label, s, U, m, θ, sk, names, obj)
        cpu[r.label] = (; grad, J, Ud, sk)
    end

    @testset "learned law (F4): coefficients, storage, start = Giesekus" begin
        law = ve_neural_law(); H = VE_NN_H
        @test relaxation_nparams(law) == 7H && relaxation_theta_array(law)
        @test isbitstype(typeof(law))
        θ0 = ve_neural_theta(:start); θ2 = ve_neural_theta(:second)
        @test θ0[1:4H] == θ2[1:4H] && all(!iszero, θ0[1:4H])      # same hidden layer, nonzero
        @test all(iszero, θ0[(4H + 1):end]) && all(!iszero, θ2[(4H + 1):end])
        # rest: (0, 1, alpha0) for any parameters; start: Giesekus at any invariants
        @test relaxation_coefficients(law, θ2, 3.0, 3.0, 1.0) == (0.0, 1.0, VE_ALPHA)
        @test relaxation_coefficients(law, θ0, 7.0, 9.0, 2.5) == (0.0, 1.0, VE_ALPHA)
        # hand formula (W: H x 3, b: H, V: 3 x H, no output bias)
        W = reshape(θ2[1:3H], H, 3); b = θ2[(3H + 1):4H]; V = permutedims(reshape(θ2[(4H + 1):7H], H, 3))
        M(x) = V * tanh.(W * x .+ b)
        ref = [0.0, 1.0, VE_ALPHA] .+ M([0.4, 0.6, 0.15]) .- M(zeros(3))
        @test collect(relaxation_coefficients(law, θ2, 7.0, 9.0, 2.5)) ≈ ref rtol = 1e-13 atol = 1e-15
        # parameters stored as a vector; residual at the start == Giesekus alpha0
        cN = ve_neural_conf(VE_NN_DE, θ0)
        @test cN.theta isa Vector{Float64}
        pN = ve_slice(cN)
        pG = ve_slice(FVConformation3D(; lambda = VE_NN_DE, nu_p = 1.0, alpha = VE_ALPHA))
        X = ve_rest(pG) .+ 1e-2 .* randn(MersenneTwister(4), fv_nunknowns(pG))
        @test fv_residual(X, pN) == fv_residual(X, pG)
        # coefficients update keeps the array storage; wrong length is an error
        s = fv_newton_solver(pN)
        fv_set_conformation!(s, ve_neural_conf(0.2, θ2))
        @test s.prob.conformation.theta == θ2 && s.prob.conformation.theta !== θ2
        @test_throws DimensionMismatch fv_set_conformation!(s, FVConformation3D(; lambda = 0.2, nu_p = 1.0,
            law = NeuralRelaxation{4}(; alpha0 = VE_ALPHA, scale = 10.0), theta = zeros(27)))
        # parameter map: subset of slots, mapped values land in the problem's array
        m2 = FVRelaxationParameters(:theta; theta_slots = VE_NN_OUTPUT)
        @test fv_nparams(m2, pN) == 2
        P2 = fv_param_problem(m2, pN, [0.7, -0.2])
        t2 = copy(θ0); t2[VE_NN_OUTPUT[1]] = 0.7; t2[VE_NN_OUTPUT[2]] = -0.2
        @test P2.conformation.theta == t2 && pN.conformation.theta == θ0
        @test fv_nparams(ve_neural_map(), pN) == 7H
    end

    @testset "CPU N=8: learned law, $point point" for point in (:start, :second)
        θnet = ve_neural_theta(point)
        s, U = ve_neural_state(θnet)
        @test maximum(abs, fv_residual(U, s.prob)) <= 1e-12
        Ud = ve_neural_data(s, U, θnet)
        m = ve_neural_map(); P = length(θnet); sk = ones(P); names = [Symbol("w$q") for q in 1:P]
        obj, _ = ve_objective(s.prob, Ud)
        H = VE_NN_H
        if point === :start
            # zero output weights: output derivatives nonzero, hidden ones exactly zero (expected)
            g = fv_steady_gradient(s, U, θnet, obj; params = m).grad
            @test all(abs.(g[(4H + 1):7H]) .> 1e-6 * norm(g))
            @test maximum(abs, g[1:4H]) <= 1e-14 * norm(g)
            grad, J = _cg_check("learned law start", s, U, m, θnet, sk, names, obj; checked = VE_NN_OUTPUT)
        else
            grad, J = _cg_check("learned law second point", s, U, m, θnet, sk, names, obj;
                                checked = (VE_NN_HIDDEN..., VE_NN_OUTPUT...), g2_random = true)
        end
        cpu["learned $point"] = (; grad, J, Ud, sk)
    end

    @testset "GPU N=8 (G4): $(r.label)" for r in rows
        if !_CG_GPU_OK
            if _CG_REQUIRE_GPU
                @error "KRAKEN_REQUIRE_GPU=1 but CUDA/CUDSS is not functional"
                @test _CG_GPU_OK
            else
                @test_skip "G4 requires CUDA.functional()"
            end
        else
            to = Main.CUDA.CuArray
            c = cpu[r.label]
            s, U = ve_state(r; to, linsolve = CUDABackendTag())
            obj, _ = ve_objective(s.prob, c.Ud; to)
            res = fv_steady_gradient(s, U, ve_theta(r), obj; params = ve_map(r))
            e = abs.(res.grad .- c.grad)
            @info "G4 GPU vs CPU" label = r.label rel = e ./ abs.(c.grad)
            @test all(e .<= 1e-9 .* abs.(c.grad) .+ 1e-13 .* c.J ./ c.sk)
            @test abs(res.J - c.J) <= 1e-10 * c.J
        end
    end

    @testset "GPU N=8 (G4): learned law, $point point" for point in (:start, :second)
        if !_CG_GPU_OK
            _CG_REQUIRE_GPU ? (@test _CG_GPU_OK) : (@test_skip "G4 requires CUDA.functional()")
        else
            to = Main.CUDA.CuArray
            c = cpu["learned $point"]
            θnet = ve_neural_theta(point)
            s, U = ve_neural_state(θnet; to, linsolve = CUDABackendTag())
            @test s.prob.conformation.theta isa Main.CUDA.CuArray
            obj, _ = ve_objective(s.prob, c.Ud; to)
            res = fv_steady_gradient(s, U, θnet, obj; params = ve_neural_map())
            e = abs.(res.grad .- c.grad)
            @info "G4 GPU vs CPU, learned law" point maxrel = maximum(e ./ max.(abs.(c.grad), 1e-300))
            @test all(e .<= 1e-9 .* abs.(c.grad) .+ 1e-13 .* c.J ./ c.sk)
            @test abs(res.J - c.J) <= 1e-10 * c.J
        end
    end

    if _CG_FULL
        @testset "GPU N=24 Giesekus (lambda, alpha): De=$De" for De in (0.1, 1.0)
            if !_CG_GPU_OK
                _CG_REQUIRE_GPU ? (@test _CG_GPU_OK) : (@test_skip "N=24 row requires CUDA.functional()")
            else
                to = Main.CUDA.CuArray
                r = ve_giesekus(De)
                s, U = ve_state(r; N = 24, to, linsolve = CUDABackendTag())
                @test maximum(abs, Array(fv_residual(U, s.prob))) <= 1e-11
                Ud = ve_data_state(r, s, U; to, linsolve = CUDABackendTag())
                # map of the row restricted to (lambda, alpha); nu_p, nu_s at their base values
                m = FVRelaxationParameters(:lambda, :theta)
                θ = [De, VE_ALPHA]; sk = [ve_scale(:lambda, De), 1.0]
                obj, _ = ve_objective(s.prob, Ud; to)
                _cg_check("GPU N=24 $(r.label)", s, U, m, θ, sk, [:lambda, :alpha], obj;
                          to, linsolve = CUDABackendTag())
            end
        end
        @testset "GPU N=24 learned law, second point, one hidden weight" begin
            if !_CG_GPU_OK
                _CG_REQUIRE_GPU ? (@test _CG_GPU_OK) : (@test_skip "N=24 row requires CUDA.functional()")
            else
                to = Main.CUDA.CuArray
                θnet = ve_neural_theta(:second)
                s, U = ve_neural_state(θnet; N = 24, to, linsolve = CUDABackendTag())
                @test maximum(abs, Array(fv_residual(U, s.prob))) <= 1e-11
                Ud = ve_neural_data(s, U, θnet; to, linsolve = CUDABackendTag())
                obj, _ = ve_objective(s.prob, Ud; to)
                P = length(θnet)
                _cg_check("GPU N=24 learned law", s, U, ve_neural_map(), θnet, ones(P), [Symbol("w$q") for q in 1:P],
                          obj; to, linsolve = CUDABackendTag(), checked = (VE_NN_HIDDEN[1],))
            end
        end
    else
        @info "GPU N=24 rows skipped (set KRAKEN_VE_ADJOINT_FULL=1)"
    end
end
