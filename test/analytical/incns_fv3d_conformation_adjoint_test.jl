# Relaxation-parameter adjoint of the steady 3D FV conformation problem (plan
# "discrete viscoelastic adjoints", step F1 and the transposed-solve check that
# opens step F2), on the square-duct slice 1 x 8 x 8 (Nx = 1 periodic, walls on
# y and z, imposed mean axial velocity, eta_s = eta_p = 1), Float64:
#   (a) F1.1/F1.2 residual derivative at a fixed state: dR/dtheta_k by
#       parameter duals (FVRelaxationParameters through fv_param_problem and the
#       conformation residual kernel) against the centred difference of the
#       residual, each parameter alone (lambda real and theta dual, and the
#       converse), for Oldroyd-B (lambda, nu_p, nu_s), Giesekus (+ alpha),
#       linear and exponential PTT (+ epsilon), FENE-P (+ L2); step 1e-6 s_k,
#       s_k = 1 for alpha and epsilon, the value for lambda, nu_p, nu_s, and
#       L2 - 3 for L2. Gate (plan F1.2): max |dual - FD| <= 1e-8 max |dR/dtheta_k|
#       + 1e-12. Also: eltype of the residual is the expected dual type, the
#       mapped problem at the base values has the bit-identical residual, the
#       four-parameter map gives the same columns as the single maps. CPU, and
#       GPU when available;
#   (b) F1.3 fv_set_params! carries the conformation coefficients (before the
#       fix only the viscosity model was copied): residual bit-identical to the
#       problem built directly, same Newton state, no new symbolic analysis;
#   (c) F1.5 domains: parameter errors (alpha outside [0, 1/2], lambda, nu_p,
#       nu_s <= 0, epsilon < 0, L2 <= 3, non-finite, wrong map or problem), no
#       solver change on a rejected fv_set_params!, state errors (c not
#       positive definite, FENE-P tr c >= L2) also raised by the adjoint;
#   (d) F2 transposed-solve check on the Giesekus slice, CPU and GPU:
#       ||J^T lambda - g|| / ||g|| <= 1e-10 and, for random x,
#       |<lambda, J x> - <g, x>| <= 1e-12 ||lambda|| ||J x|| (lambda from the
#       transposed factor solve, J x from the assembled matrix), plus the duality
#       of the forward and transposed factor solves |<lambda, b> - <g, J^-1 b>|
#       <= 1e-12 ||lambda|| ||b||;
#   (e) G1 tangent vs adjoint, Giesekus (lambda, alpha), CPU: velocity + Fx
#       misfit to data at theta + 0.1 s (additive), every sensitivity nonzero
#       (|g_k| > 1e-8 |J| / s_k), |<grad, dtheta> - gU . dU| <= 1e-10
#       |<grad, dtheta>| + 1e-14 along each parameter and a random direction.
#       A consistency check only (both share J and dR/dtheta); the centred FD
#       matrix of F2 comes later.
#   (f) iterative refinement of the direct solves (adjoint transposed solve and
#       tangent): with exact factors, refine = 0 reports the same residual before
#       and after and the default (one step) stays <= 1e-12; with the factors of
#       a perturbed J (1e-8 relative on every nonzero) and the assembled J
#       restored, one solve leaves a true residual above the 1e-10 warning
#       threshold (warning logged), 8 steps bring it <= 1e-12 and lambda / dU
#       back to the values of the exact factors. CPU, and GPU (cuDSS) when
#       available.
#
# GPU: self-gated like incns_fv_adjoint_gpu.jl (skipped when CUDA/CUDSS are not
# loadable or CUDA is not functional; KRAKEN_REQUIRE_GPU=1 turns the skip into a
# failure); the transposed cuDSS solve needs CUDSS.jl >= 1.0.
using Test
using LinearAlgebra
using Random
using ForwardDiff
using KrakeNN
using KrakeNN.SparseArrays

const _CA_GPU_OK = (try
    @eval Main using CUDA, CUDSS
    @eval Main CUDA.functional()
catch
    false
end) === true
const _CA_REQUIRE_GPU = get(ENV, "KRAKEN_REQUIRE_GPU", "0") in ("1", "true")

const _CA_BC = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall)
const _CA_NEWTON = (; atol = 1e-14, rtol = 1e-15, maxit = 40)
const _CA_N = 8
const _CA_LAMBDA = 0.3
const _CA_ALPHA = 0.3

_ca_slice(conf; nu_s = 1.0, N = _CA_N) =
    fv_steady_problem(1, N, N; h = 1 / N, nu = nu_s, bc = _CA_BC, conformation = conf,
                      source = (x, y, z) -> (50.0, 0.0, 0.0, 0.0), u_mean_target = 1.0)
_ca_rest(p) = (z = zeros(1, p.grid.Ny, p.grid.Nz); fv_state(p, z, z, z, z))
_ca_giesekus(lam = _CA_LAMBDA, alpha = _CA_ALPHA; nu_p = 1.0) =
    FVConformation3D(; lambda = lam, nu_p, alpha)

function _ca_solve!(s, U)
    st = fv_newton!(s, U; _CA_NEWTON...)
    st.converged || (st.stagnated && last(st.inf) <= 1e-11) ||
        error("Newton did not converge (residual $(last(st.inf)))")
    return st
end

# Giesekus slice converged at lambda = _CA_LAMBDA by continuation from rest at 0.1
function _ca_giesekus_state(; to = Array, linsolve = CPUBackendTag())
    p = _ca_slice(_ca_giesekus(0.1))
    s = fv_newton_solver(p; to, linsolve)
    U = to(_ca_rest(p))
    for lam in (0.1, 0.2, _CA_LAMBDA)
        fv_set_conformation!(s, _ca_giesekus(lam))
        _ca_solve!(s, U)
    end
    return s, U
end

# host copy of a device CSR Jacobian (fields of the cuDSS matrix)
_ca_host_J(J::SparseMatrixCSC) = J
_ca_host_J(J) = sparse(transpose(SparseMatrixCSC(size(J)..., Array(J.rowPtr), Array(J.colVal),
                                                 Array(J.nzVal))))

# step scale s_k of the plan (F2 "Échelles de pas")
_ca_scale(name, value) = name in (:alpha, :epsilon) ? 1.0 : name === :L2 ? value - 3 : value

# (label, conformation, nu_s, [(parameter name, map, base value)])
function _ca_law_cases()
    rel(w) = FVRelaxationParameters(w)
    flow(nu_p) = [(:lambda, rel(:lambda), _CA_LAMBDA), (:nu_p, rel(:nu_p), nu_p), (:nu_s, rel(:nu_s), 1.0)]
    return [
        ("Oldroyd-B", FVConformation3D(; lambda = _CA_LAMBDA, nu_p = 1.0), flow(1.0)),
        ("Giesekus", _ca_giesekus(), [flow(1.0); (:alpha, rel(:theta), _CA_ALPHA)]),
        ("linear PTT", FVConformation3D(; lambda = _CA_LAMBDA, nu_p = 1.0, law = LinearPTTRelaxation(),
                                        theta = (0.2,)), [flow(1.0); (:epsilon, rel(:theta), 0.2)]),
        ("exponential PTT", FVConformation3D(; lambda = _CA_LAMBDA, nu_p = 1.0,
                                             law = ExponentialPTTRelaxation(), theta = (0.2,)),
         [flow(1.0); (:epsilon, rel(:theta), 0.2)]),
        ("FENE-P", FVConformation3D(; lambda = _CA_LAMBDA, nu_p = 1.0, law = FENEPRelaxation(),
                                    theta = (50.0,)), [flow(1.0); (:L2, rel(:theta), 50.0)]),
    ]
end

const _CA_D1 = ForwardDiff.Dual{FVParamTag,Float64,1}
_ca_d1(x) = ForwardDiff.Dual{FVParamTag}(Float64(x), 1.0)   # one seeded parameter

# (a) on one backend: p on that backend, X a state on that backend
function _ca_residual_derivatives(label, p, X, entries)
    R0 = Array(fv_residual(X, p))
    for (name, m, θ0) in entries
        @testset "$label: $name" begin
            @test fv_nparams(m, p) == 1
            # base values: bit-identical residual
            @test Array(fv_residual(X, fv_param_problem(m, p, [θ0]))) == R0
            pd = fv_param_problem(m, p, [_ca_d1(θ0)])
            Rd = fv_residual(X, pd)
            @test eltype(Rd) === _CA_D1
            if name === :nu_s
                @test eltype(pd.nu) === _CA_D1 && pd.conformation === p.conformation
            else
                # every coefficient promoted, the unmapped ones with zero partials
                c = pd.conformation
                @test typeof(c.lambda) === _CA_D1 && eltype(c.theta) === _CA_D1 && eltype(pd.nu) === Float64
                name === :lambda || @test iszero(ForwardDiff.partials(c.lambda)[1])
                name in (:alpha, :epsilon, :L2) || @test all(t -> iszero(ForwardDiff.partials(t)[1]), c.theta)
                @test ForwardDiff.value(c.G) == p.conformation.G
            end
            dR = [ForwardDiff.partials(r)[1] for r in Array(Rd)]
            ε = 1e-6 * _ca_scale(name, θ0)
            Rp = Array(fv_residual(X, fv_param_problem(m, p, [θ0 + ε])))
            Rm = Array(fv_residual(X, fv_param_problem(m, p, [θ0 - ε])))
            fd = (Rp .- Rm) ./ (2ε)
            scale = maximum(abs, dR)
            err = maximum(abs, dR .- fd)
            @info "dR/dtheta duals vs centred FD" label name scale err
            @test scale > 0
            @test err <= 1e-8 * scale + 1e-12
        end
    end
    return nothing
end

# perturbed copy of a converged state (c != I, nonzero velocity gradient everywhere)
_ca_perturbed(U; seed = 5) = (Uh = Array(U); Uh .+ 1e-3 .* randn(MersenneTwister(seed), length(Uh)))

# (f) on one backend
_ca_nzval(J::SparseMatrixCSC) = nonzeros(J)
_ca_nzval(J) = J.nzVal
function _ca_refine_check(s, U, to)
    m = FVRelaxationParameters(:lambda, :theta)
    θ = [_CA_LAMBDA, _CA_ALPHA]
    δθ = [1.0, 0.5]
    n = length(U)
    gU = to(randn(MersenneTwister(17), n))
    fv_newton_finalize!(s, U)
    _, sex = fv_steady_adjoint(s, U, gU; params = m, θ, finalize = false, refine = 0)
    λex = Array(copy(sex.λ))
    dUex = Array(fv_steady_tangent(s, U, θ, δθ; params = m, finalize = false, refine = 0))
    @test sex.transpose_relres_initial == sex.transpose_relres
    @test sex.transpose_relres <= 1e-10
    _, s1 = fv_steady_adjoint(s, U, gU; params = m, θ, finalize = false)        # default: one step
    @test s1.transpose_relres <= 1e-12
    # factors of a perturbed J, assembled J restored
    nz = _ca_nzval(s.J); nz0 = copy(nz)
    nz .*= 1 .+ 1e-8 .* to(randn(MersenneTwister(19), length(nz)))
    KrakeNN._fv_factor!(s)
    copyto!(nz, nz0)
    _, sa = @test_logs (:warn, r"above") match_mode = :any fv_steady_adjoint(s, U, gU; params = m, θ,
                                                                               finalize = false, refine = 0)
    λa = Array(copy(sa.λ))
    _, sb = fv_steady_adjoint(s, U, gU; params = m, θ, finalize = false, refine = 8)
    λb = Array(copy(sb.λ))
    dUa = @test_logs (:warn, r"above") match_mode = :any Array(fv_steady_tangent(s, U, θ, δθ; params = m,
                                                                                   finalize = false, refine = 0))
    dUb = Array(fv_steady_tangent(s, U, θ, δθ; params = m, finalize = false, refine = 8))
    @info "iterative refinement" backend = (to === Array ? "CPU" : "GPU") exact = sex.transpose_relres one_step = s1.transpose_relres stale = sa.transpose_relres stale_refined = sb.transpose_relres lambda_err_stale = norm(λa - λex) / norm(λex) lambda_err_refined = norm(λb - λex) / norm(λex) dU_err_stale = norm(dUa - dUex) / norm(dUex) dU_err_refined = norm(dUb - dUex) / norm(dUex)
    @test sa.transpose_relres > 1e-10
    @test sb.transpose_relres <= 1e-12
    @test norm(λb - λex) <= 1e-3 * norm(λa - λex)
    @test norm(dUb - dUex) <= 1e-3 * norm(dUa - dUex)
    @test norm(λb - λex) <= 1e-7 * norm(λex)
    @test norm(dUb - dUex) <= 1e-7 * norm(dUex)
    fv_newton_finalize!(s, U)                  # consistent factors again
    return nothing
end

# (d) on one backend
function _ca_transpose_check(s, U, to)
    fv_newton_finalize!(s, U)
    @test lin_supports_transpose(s.cache)
    n = length(U)
    rng = MersenneTwister(7)
    g = randn(rng, n); x = randn(rng, n); b = randn(rng, n)
    λ = similar(U); lin_solve!(λ, s.cache, to(g); transpose = true)
    y = similar(U); lin_solve!(y, s.cache, to(b))
    λh = Array(λ); yh = Array(y)
    J = _ca_host_J(s.J)
    r = norm(transpose(J) * λh .- g) / norm(g)
    Jx = J * x
    e1 = abs(dot(λh, Jx) - dot(g, x)) / (norm(λh) * norm(Jx))
    e2 = abs(dot(λh, b) - dot(g, yh)) / (norm(λh) * norm(b))
    @info "transposed solve" backend = (to === Array ? "CPU" : "GPU") n residual = r duality_product = e1 duality_solves = e2
    @test r <= 1e-10
    @test e1 <= 1e-12
    @test e2 <= 1e-12
    return nothing
end

@testset "FV 3D conformation: relaxation-parameter adjoint (F1, F2 transposed solve)" begin
    s, U = _ca_giesekus_state()
    pG = _ca_slice(_ca_giesekus())
    X = _ca_perturbed(U)

    @testset "(a) dR/dtheta duals vs centred FD (CPU)" begin
        for (label, conf, entries) in _ca_law_cases()
            _ca_residual_derivatives(label, _ca_slice(conf), X, entries)
        end
        # four-parameter map: columns of the single maps, parameter order of `which`
        m4 = FVRelaxationParameters(:nu_s, :lambda, :theta, :nu_p)
        @test fv_nparams(m4, pG) == 4
        θ4 = [1.0, _CA_LAMBDA, _CA_ALPHA, 1.0]
        D4 = ForwardDiff.Dual{FVParamTag,Float64,4}
        θd = [D4(θ4[k], ForwardDiff.Partials(ntuple(c -> Float64(c == k), 4))) for k in 1:4]
        Rd4 = fv_residual(X, fv_param_problem(m4, pG, θd))
        @test eltype(Rd4) === D4
        for (k, w) in enumerate((:nu_s, :lambda, :theta, :nu_p))
            m1 = FVRelaxationParameters(w)
            d1 = [ForwardDiff.partials(r)[1] for r in fv_residual(X, fv_param_problem(m1, pG, [_ca_d1(θ4[k])]))]
            dk = [ForwardDiff.partials(r)[k] for r in Rd4]
            @test maximum(abs, dk .- d1) <= 1e-14 * maximum(abs, d1)
        end
        @test fv_nparams(FVRelaxationParameters((:lambda, :theta)), pG) == 2
        @test fv_nparams(FVRelaxationParameters(:theta; theta_slots = (1, 2)), pG) == 2
    end

    @testset "(b) fv_set_params! carries the conformation coefficients" begin
        m = FVRelaxationParameters(:lambda, :nu_p, :theta, :nu_s)
        θ1 = [0.25, 0.8, 0.4, 1.2]
        p1 = _ca_slice(_ca_giesekus(0.25, 0.4; nu_p = 0.8); nu_s = 1.2)
        s1 = fv_newton_solver(pG)
        fv_set_params!(s1, m, θ1)
        @test s1.prob.conformation === p1.conformation
        @test s1.prob.nu == p1.nu
        Y = X .+ 1e-3 .* randn(MersenneTwister(9), length(X))
        @test fv_residual(Y, s1.prob) == fv_residual(Y, p1)
        U1 = copy(U); _ca_solve!(s1, U1)
        sref = fv_newton_solver(p1)
        Uref = copy(U); _ca_solve!(sref, Uref)
        @test norm(U1 .- Uref, Inf) <= 1e-12 * norm(Uref, Inf)
        # back to the base values on the same solver: same analysis, base state
        nf = s1.n_factorize
        fv_set_params!(s1, m, [_CA_LAMBDA, 1.0, _CA_ALPHA, 1.0])
        @test s1.prob.conformation === pG.conformation
        _ca_solve!(s1, U1)
        @test s1.n_factorize == nf
        @test norm(U1 .- U, Inf) <= 1e-10 * norm(U, Inf)
    end

    @testset "(c) admissible domains (errors, no clipping)" begin
        mt = FVRelaxationParameters(:theta)
        for α in (-0.01, 0.51, NaN)
            @test_throws ArgumentError fv_param_problem(mt, pG, [α])
        end
        @test fv_param_problem(mt, pG, [0.0]).conformation.theta == (1.0, 0.0)
        @test fv_param_problem(mt, pG, [0.5]).conformation.theta == (1.0, 0.5)
        for w in (:lambda, :nu_p, :nu_s), v in (0.0, -1.0, Inf)
            @test_throws ArgumentError fv_param_problem(FVRelaxationParameters(w), pG, [v])
        end
        # b1 free: alpha = b2 / b1
        @test_throws ArgumentError fv_param_problem(FVRelaxationParameters(:theta; theta_slots = (1, 2)), pG,
                                                    [0.5, 0.3])
        @test_throws ArgumentError fv_param_problem(FVRelaxationParameters(:theta; theta_slots = (1, 2)), pG,
                                                    [-1.0, 0.0])
        ptt = _ca_slice(FVConformation3D(; lambda = 0.3, nu_p = 1.0, law = LinearPTTRelaxation(), theta = (0.2,)))
        @test_throws ArgumentError fv_param_problem(mt, ptt, [-0.1])
        fenep = _ca_slice(FVConformation3D(; lambda = 0.3, nu_p = 1.0, law = FENEPRelaxation(), theta = (50.0,)))
        @test_throws ArgumentError fv_param_problem(mt, fenep, [3.0])
        # dual entries are checked on their values
        @test_throws ArgumentError fv_param_problem(mt, pG, [_ca_d1(0.6)])
        # map construction and problem
        @test_throws ArgumentError FVRelaxationParameters()
        @test_throws ArgumentError FVRelaxationParameters(:alpha)
        @test_throws ArgumentError FVRelaxationParameters(:lambda, :lambda)
        @test_throws ArgumentError FVRelaxationParameters(:lambda; theta_slots = (1,))
        @test_throws ArgumentError fv_nparams(FVRelaxationParameters(:theta; theta_slots = (3,)), pG)
        @test_throws ArgumentError fv_nparams(mt, fv_steady_problem(1, 4, 4; h = 0.25, bc = _CA_BC))
        @test_throws DimensionMismatch fv_param_problem(mt, pG, [0.3, 0.1])
        # a rejected fv_set_params! leaves the solver untouched
        s2 = fv_newton_solver(pG)
        @test_throws ArgumentError fv_set_params!(s2, FVRelaxationParameters(:theta, :nu_s), [0.7, 2.0])
        @test s2.prob.conformation === pG.conformation && all(==(1.0), s2.prob.nu)
        # states
        d = fv_check_conformation_domain(pG, U)
        @test d.min_eigenvalue > 0 && d.max_trace > 3 && d.fenep_margin == Inf
        Ubad = copy(U)
        Ubad[fv_index(pG.grid, 1, 4, 5, 5)] = -0.5        # c_xx < 0 in one cell
        @test fv_conformation_domain(pG, Ubad).min_eigenvalue < 0
        @test_throws DomainError fv_check_conformation_domain(pG, Ubad)
        gU = zeros(length(U)); gU[1] = 1.0
        @test_throws DomainError fv_steady_adjoint(s, Ubad, gU; params = mt, θ = [_CA_ALPHA])
        @test_throws DomainError fv_steady_tangent(s, Ubad, [_CA_ALPHA], [1.0]; params = mt)
        df = fv_check_conformation_domain(fenep, U)
        @test df.fenep_margin ≈ 50.0 - df.max_trace
        Uf = copy(U)
        Uf[fv_index(fenep.grid, 1, 4, 4, 5)] = 60.0       # tr c > L2 = 50, still positive definite
        @test fv_conformation_domain(fenep, Uf).min_eigenvalue > 0
        @test fv_conformation_domain(fenep, Uf).fenep_margin < 0
        @test_throws DomainError fv_check_conformation_domain(fenep, Uf)
    end

    @testset "(d) transposed solve with the Newton factors (CPU)" begin
        _ca_transpose_check(s, U, Array)
    end

    @testset "(e) G1 tangent vs adjoint, Giesekus (lambda, alpha), CPU" begin
        m = FVRelaxationParameters(:lambda, :theta)
        θ = [_CA_LAMBDA, _CA_ALPHA]
        sk = [_ca_scale(:lambda, θ[1]), _ca_scale(:alpha, θ[2])]
        # data: converged state at theta* = theta + 0.1 s (warm start from U)
        sd = fv_newton_solver(pG)
        fv_set_params!(sd, m, θ .+ 0.1 .* sk)
        Ud = copy(U); _ca_solve!(sd, Ud)
        fv_check_conformation_domain(sd.prob, Ud)
        # weights: axial and secondary velocities separately normalised, plus Fx
        g = pG.grid
        n = length(U)
        W = zeros(n)
        u, v, w, _ = fv_fields(pG, Ud)
        sa = sum(abs2, u); ss = sum(abs2, v) + sum(abs2, w)
        for k in 1:g.Nz, j in 1:g.Ny
            W[fv_index(g, 1, j, k, 1)] = 1 / sa
            W[fv_index(g, 1, j, k, 2)] = 1 / ss
            W[fv_index(g, 1, j, k, 3)] = 1 / ss
        end
        iF = KrakeNN._fv_force_index(pG)
        W[iF] = 1 / Ud[iF]^2
        @test ss > 0
        Jv = sum(W .* (U .- Ud) .^ 2) / 2
        gU = W .* (U .- Ud)
        @test fv_check_conformation_domain(s.prob, U).min_eigenvalue > 0
        grad, st = fv_steady_adjoint(s, U, gU; params = m, θ)
        @test st.residual_inf <= 1e-12
        @test length(grad) == 2 && all(isfinite, grad)
        @test all(abs.(grad) .> 1e-8 .* Jv ./ sk)
        rd = randn(MersenneTwister(13), 2)
        for (δθ, name) in (([1.0, 0.0], "lambda"), ([0.0, 1.0], "alpha"), (rd, "random"))
            dU = fv_steady_tangent(s, U, θ, δθ; params = m)
            lhs = dot(grad, δθ)
            rhs = dot(gU, Array(dU))
            @info "G1 tangent vs adjoint" name lhs err = abs(lhs - rhs)
            @test abs(lhs - rhs) <= 1e-10 * abs(lhs) + 1e-14
        end
        # informative only (the FD gate of F2 is fixed after its step study)
        sf = fv_newton_solver(pG)
        function J_at(θp)
            fv_set_params!(sf, m, θp)
            Up = copy(U); _ca_solve!(sf, Up)
            return sum(W .* (Up .- Ud) .^ 2) / 2
        end
        for k in 1:2
            ε = 1e-4 * sk[k]
            e = Float64.(1:2 .== k)
            fd = (J_at(θ .+ ε .* e) - J_at(θ .- ε .* e)) / (2ε)
            @info "adjoint vs centred FD (not gated here)" k grad = grad[k] fd rel = abs(fd - grad[k]) / abs(grad[k])
        end
    end

    @testset "(f) iterative refinement of the direct solves (CPU)" begin
        _ca_refine_check(s, U, Array)
    end

    @testset "GPU: (a) dR/dtheta, (d) transposed solve, (f) refinement" begin
        if !_CA_GPU_OK
            if _CA_REQUIRE_GPU
                @error "KRAKEN_REQUIRE_GPU=1 but CUDA/CUDSS is not functional"
                @test _CA_GPU_OK
            else
                @info "Skipping the GPU part of the relaxation-parameter adjoint tests (CUDA not functional here)"
                @test_skip "relaxation-parameter adjoint GPU checks require CUDA.functional()"
            end
        else
            CU = Main.CUDA
            to = CU.CuArray
            sg, Ug = _ca_giesekus_state(; to, linsolve = CUDABackendTag())
            @test norm(Array(Ug) .- U, Inf) <= 1e-10 * norm(U, Inf)
            Xg = to(X)
            for (label, conf, entries) in _ca_law_cases()
                _ca_residual_derivatives("GPU $label", fv_adapt(to, _ca_slice(conf)), Xg, entries)
            end
            _ca_transpose_check(sg, Ug, to)
            _ca_refine_check(sg, Ug, to)
        end
    end
end
