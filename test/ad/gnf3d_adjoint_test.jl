# Steady adjoint of the 3D D3Q19 GNF step (src/ad/gnf3d_adjoint*.jl), CPU.
# Small periodic lattice 4×6×3, one solid wall plane (j = 1) and one solid
# voxel, DenseNet law through LBMLawViscosity (eps > 0), halfway bounce-back.
#
#   T1  velocity and opposite tables of the adjoint == the D3Q19 lattice
#   T2  the stream map is a permutation of the fluid populations
#   T3  host cell map (adjoint model) == production kernel step
#   T4  JVP of the linearised step == central differences of the production
#       step (state direction), and the ν column == differences in θ
#   T5  transposition <Aᵀx, y> = <x, A y> (unprojected and projected)
#   T6  mass gauge: P Aᵀ P ĉ = ĉ; ν column has zero mass
#   T7  steady state, bordered GMRES solve, explicit residual, K-step solve
#       (shared 2D solver) == GMRES gradient, gradient of
#       J = mean u_x vs central differences (V shape)
using Test, KrakeNN, Random, LinearAlgebra

const _G3T_DIMS = (4, 6, 3)

function _g3t_case()
    nx, ny, nz = _G3T_DIMS
    solid = falses(nx, ny, nz)
    solid[:, 1, :] .= true
    solid[2, 4, 2] = true
    net = DenseNet(1, (6,); activation=:tanh, output=:exp, scale=(1.0,))
    θ = law_init(net; rng=MersenneTwister(5))
    θ[end] = log(0.12)
    return Array{Bool}(solid), net, θ
end

_g3t_closure(net, θ) = LBMLawViscosity(net, θ; eps=1e-3)

function _g3t_state(solid; seed=7)
    rng = MersenneTwister(seed)
    nx, ny, nz = size(solid)
    f = zeros(nx, ny, nz, 19); tau = zeros(nx, ny, nz)
    for I in CartesianIndices(solid)
        ρ = 1 + 0.01 * randn(rng)
        u = 0.02 .* randn(rng, 3)
        for q in 1:19
            f[I, q] = KrakeNN.equilibrium(D3Q19(), ρ, u[1], u[2], u[3], q) * (1 + 0.01 * randn(rng))
        end
        tau[I] = 0.7 + 0.2 * rand(rng)
    end
    return f, tau
end

_g3t_rel(a, b) = norm(a .- b) / max(norm(b), eps(Float64))

@testset "3D GNF steady adjoint" begin
    solid, net, θ = _g3t_case()
    cl = _g3t_closure(net, θ)
    nx, ny, nz = size(solid); N = nx * ny * nz; n = 20N
    F = (2e-5, 0.0, 0.0)
    f, tau = _g3t_state(solid)
    w = gnf3d_pack_state(f, tau)
    act = gnf3d_active_mask(solid)
    fl = .!solid

    @testset "T1 lattice tables" begin
        lat = D3Q19()
        @test collect(KrakeNN._G3_CX) == collect(KrakeNN.velocities_x(lat)) &&
              collect(KrakeNN._G3_CY) == collect(KrakeNN.velocities_y(lat)) &&
              collect(KrakeNN._G3_CZ) == collect(KrakeNN.velocities_z(lat))
        o = KrakeNN._G3_OPP
        @test all(KrakeNN._G3_CX[o[q]] == -KrakeNN._G3_CX[q] &&
                  KrakeNN._G3_CY[o[q]] == -KrakeNN._G3_CY[q] &&
                  KrakeNN._G3_CZ[o[q]] == -KrakeNN._G3_CZ[q] for q in 1:19)
    end

    @testset "T2 stream permutation" begin
        srcs = Int[]
        for k in 1:nz, j in 1:ny, i in 1:nx, q in 1:19
            solid[i, j, k] && continue
            push!(srcs, KrakeNN._gnf3d_src(i, j, k, q, nx, ny, nz, solid))
        end
        @test allunique(srcs)
        @test length(srcs) == 19 * count(fl) && all(act[srcs]) && all(srcs .<= 19N)
    end

    @testset "T3 host cell map == production kernel step" begin
        wk = gnf3d_step_state(w, solid, cl, F)
        wh = gnf3d_host_step(w, solid, cl, F)
        err = _g3t_rel(wk[act], wh[act])
        @info "GNF3D T3 host vs kernel step" relative = err
        @test err <= 1e-13
    end

    lin = gnf3d_linearise(f, tau, solid, cl, F)
    op = GNF3DStepOperator(lin)

    @testset "T4 JVP vs central differences of the production step" begin
        rng = MersenneTwister(21)
        v = randn(rng, n) .* act
        h = 1e-6
        fd = (gnf3d_step_state(w .+ h .* v, solid, cl, F) .-
              gnf3d_step_state(w .- h .* v, solid, cl, F)) ./ (2h)
        y = similar(v); gnf3d_jvp!(y, op, v)
        err = _g3t_rel(y[act], fd[act])
        # ν column: d(step)/dθ along d = Σ_c B[c, :, 21] dν_c/dθ·d
        d = normalize(randn(rng, length(θ)))
        hθ = 1e-6
        fdθ = (gnf3d_step_state(w, solid, _g3t_closure(net, θ .+ hθ .* d), F) .-
               gnf3d_step_state(w, solid, _g3t_closure(net, θ .- hθ .* d), F)) ./ (2hθ)
        pred = zeros(n)
        clp = _g3t_closure(net, θ .+ hθ .* d); clm = _g3t_closure(net, θ .- hθ .* d)
        for k in 1:nz, j in 1:ny, i in 1:nx
            solid[i, j, k] && continue
            I1c = lin.I1[i, j, k]
            dν = (gnf_viscosity(clp, I1c) - gnf_viscosity(clm, I1c)) / (2hθ)
            for r in 1:20
                pred[KrakeNN._gnf3d_lin(i, j, k, r, nx, ny, nz)] = lin.B[i, j, k, r, 21] * dν
            end
        end
        errθ = _g3t_rel(pred[act], fdθ[act])
        @info "GNF3D T4 JVP vs FD" state_direction = err theta_direction = errθ
        @test err <= 1e-7
        @test errθ <= 1e-6
    end

    @testset "T5 transposition" begin
        rng = MersenneTwister(4)
        x = randn(rng, n); y = randn(rng, n)
        ax = similar(x); ay = similar(y)
        for project in (false, true)
            gnf3d_vjp!(ax, op, x; project); gnf3d_jvp!(ay, op, y; project)
            a = dot(ax, y); b = dot(x, ay)
            err = abs(a - b) / (norm(x) * norm(ay))
            @info "GNF3D T5 transposition" project a b scaled_error = err
            @test err <= 1e-12
        end
    end

    @testset "T6 mass gauge" begin
        y = similar(op.chat)
        gnf3d_vjp!(y, op, op.chat)
        @test norm(y .- op.chat) <= 1e-13
        colmass = [sum(lin.B[i, j, k, 1:19, 21]) for k in 1:nz, j in 1:ny, i in 1:nx if !solid[i, j, k]]
        @test maximum(abs, colmass) <= 1e-13
    end

    @testset "T7 steady adjoint, residual, gradient vs central differences" begin
        Fs = (1e-3, 0.0, 0.0)              # u ~ 0.03, shear rates above eps
        r = run_gnf3d_periodic_steady(; is_solid=solid, closure=cl, F=Fs, nu_init=0.12,
                                      tol=1e-12, max_steps=200_000)
        @test r.converged
        _, ux, uy, uz = gnf3d_post_moments(r.f, solid, Fs)
        @test _g3t_rel(ux[fl], r.ux[fl]) <= 1e-12   # kernel u == post-collision moment
        nfl = count(fl)
        J = rr -> sum(rr.ux[fl]) / nfl
        gux = zeros(size(solid)); gux[fl] .= 1 / nfl
        z = zeros(size(solid))
        res = gnf3d_steady_law_gradient(r.f, r.tau, (gux, z, z); is_solid=solid, closure=cl,
                                        F=Fs, solver=:gmres, tol=1e-12, restart=1000, maxiter=20_000)
        @test res.stats.converged
        λ = res.stats.λ
        @test abs(dot(res.op.chat, λ)) <= 1e-10 * norm(λ)
        b = res.op.active .* gnf3d_velocity_rhs(res.lin, (gux, z, z))
        y = similar(λ); gnf3d_vjp!(y, res.op, λ)
        resid = norm(b .- (λ .- y .+ res.op.chat .* res.stats.eta)) / norm(b)
        @info "GNF3D T7 adjoint" steps = r.steps iters = res.stats.iters relres = res.stats.relres explicit = resid t_lin = res.t_lin t_adjoint = res.t_adjoint
        @test resid <= 1e-10
        @test all(isfinite, res.grad) && norm(res.grad) > 0
        # shared K-step solver of the 2D adjoint on the 3D operator (default solver)
        rk = gnf3d_steady_law_gradient(r.f, r.tau, (gux, z, z); is_solid=solid, closure=cl,
                                       F=Fs, tol=1e-11)
        kgap = _g3t_rel(rk.grad, res.grad)
        @info "GNF3D T7 kstep vs gmres" converged = rk.stats.converged iters = rk.stats.iters grad_gap = kgap
        @test rk.stats.converged && kgap <= 1e-8
        rng = MersenneTwister(13)
        for kd in 1:2
            d = normalize(randn(rng, length(θ)))
            exact = dot(res.grad, d)
            errs = Float64[]
            for h in (1e-2, 1e-3, 1e-4, 1e-5)
                Jh = t -> J(run_gnf3d_periodic_steady(; is_solid=solid,
                    closure=_g3t_closure(net, θ .+ t .* d), F=Fs, tol=0.0, f_init=r.f,
                    tau_init=r.tau, max_steps=3000))
                push!(errs, abs((Jh(h) - Jh(-h)) / (2h) - exact) / abs(exact))
            end
            @info "GNF3D T7 adjoint vs FD" direction = kd steps = (1e-2, 1e-3, 1e-4, 1e-5) rel_err = Tuple(errs)
            @test minimum(errs) <= 1e-6
            @test errs[1] > 10 * errs[3]
        end
    end
end
