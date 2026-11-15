# Derivative gate of the steady 3D FV conformation adjoint (plan "discrete
# viscoelastic adjoints", step F2), square-duct slice 1 x N x N, Float64, with
# FVRelaxationParameters(:lambda, :nu_p, :nu_s[, :theta]).
#
# Coverage matrix (only these combinations are claimed verified; the learned
# relaxation law of step F4 is not in this file yet):
#   law               parameters               CPU N=8   GPU N=8 (G4)   GPU N=24
#   Oldroyd-B 0.5     lambda, nu_p, nu_s          yes       yes            no
#   Giesekus 0.1, 1   lambda, nu_p, nu_s, alpha   yes       yes            lambda, alpha (full)
#   linear PTT 0.5    lambda, nu_p, nu_s, eps     yes       yes            no
#   exp. PTT 0.5      lambda, nu_p, nu_s, eps     yes       yes            no
#   FENE-P 0.5        lambda, nu_p, nu_s, L2      yes       yes            no
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
# Oldroyd-B (R10): in the developed straight slice velocity and Fx depend on
# nu_s + nu_p only, so (a) without the conformation term |dJ/dlambda| <= 1e-10
# ||grad J|| (expected zero) while nu_p and nu_s stay nonzero; (b) the
# conformation term makes lambda visible and the row passes G1-G3.
#
# KRAKEN_VE_ADJOINT_FULL=1 adds the GPU N = 24 Giesekus row (lambda, alpha at De
# 0.1 and 1: G1, G2, G3). GPU parts are self-gated like
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

# G1 + G2 + G3 of one (row, map) on one backend; returns the gradient
function _cg_check(label, s, U, m, θ, sk, names, obj; to = Array, linsolve = CPUBackendTag(),
                   g3 = true, expect_zero = Symbol[])
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
    for k in eachindex(θ)
        names[k] in expect_zero && continue
        ok = abs(grad[k]) > 1e-8 * J / sk[k]
        ok || @info "zero sensitivity" label name = names[k] g = grad[k] J
        @test ok
    end
    # G1
    rd = randn(MersenneTwister(13), length(θ))
    dirs = [[(Float64.(1:length(θ) .== k), String(names[k])) for k in eachindex(θ)]; (rd, "random")]
    for (δθ, name) in dirs
        dU = fv_steady_tangent(s, U, θ, δθ; params = m, ws)
        lhs = dot(grad, δθ)
        e = abs(lhs - (dot(gU, Array(dU)) + dot(explicit, δθ)))
        @info "G1 tangent vs adjoint" label name lhs err = e
        @test e <= 1e-10 * abs(lhs) + 1e-14
    end
    # G2
    sf = fv_newton_solver(KrakeNN._fv_host_problem(s.prob); to, linsolve)
    for k in eachindex(θ)
        names[k] in expect_zero && continue
        errs = Float64[]
        for h in VE_G2_STEPS
            fd, _ = ve_fd(sf, m, θ, k, h, sk[k], U, obj)
            push!(errs, abs(fd - grad[k]))
        end
        tol = VE_G2_REL * abs(grad[k]) + VE_G2_ABS * J / sk[k]
        @info "G2 adjoint vs centred FD" label name = names[k] g = grad[k] err = minimum(errs) rel = minimum(errs) / abs(grad[k]) tol
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
    else
        @info "GPU N=24 Giesekus row skipped (set KRAKEN_VE_ADJOINT_FULL=1)"
    end
end
