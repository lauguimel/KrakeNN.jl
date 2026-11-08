# C4 of the FV Newton roadmap (src/methods/inc_ns/fv_adjoint.jl): steady
# discrete adjoint on the Newton factors. CPU checks (the parameter duals run
# through the same KernelAbstractions residual kernel as on GPU, on the KA CPU
# backend):
#   1. building blocks against dense ForwardDiff: −λᵀ ∂R/∂θ for every built-in
#      parameter map (chunked scalar maps, coloured viscosity field, including a
#      2-cell periodic grid where a cell is its own E and W neighbour) and the
#      objective gradients ∂𝒥/∂U, ∂𝒥/∂θ;
#   2. Navier-Stokes cavity: transposed solve (Jᵀλ = g) with the Newton
#      factors; adjoint gradient vs central finite differences of the converged
#      objective (Taylor test, two step sizes, ≥ 3 directions for the vector
#      parameters, roadmap V3a: 1e-6 relative); adjoint vs forward tangent;
#      the gradient reuses the Newton factorization (one symbolic analysis);
#   3. masked inlet/outlet channel (obstacle drag, pressure drop), Stokes + NS;
#   4. exact identity dQ/dν = −Q/ν at fixed body force (V3a, 1e-8);
#   5. platform contract: SteadyAdjoint + AssembledJacobian on :fv_newton only.
# GPU contract: incns_fv_adjoint_gpu.jl.

using Test
using LinearAlgebra
using Random
using ForwardDiff
using KrakeNN
using KrakeNN.SparseArrays

include(joinpath(@__DIR__, "incns_fv_adjoint_compiler.jl"))

if !isdefined(@__MODULE__, :_fvc2_ghia_error)
    include(joinpath(@__DIR__, "fv_newton_helpers.jl"))
end

const _FVC4_NEWTON = (; atol = 1e-14, rtol = 1e-15, maxit = 40)
const _FVC4_BASE = Ref{Any}(nothing)   # problem the FD solver starts from
const _FVC4_LOG = Dict{Symbol,Float64}(:fd => 0.0, :tangent => 0.0, :dense => 0.0)
_fvc4_log!(k, e) = (_FVC4_LOG[k] = max(_FVC4_LOG[k], e); e)

# converged objective at θ: the FD solver is reused through fv_set_params!
# (same pattern, colouring and symbolic analysis), warm start from U0
function _fvc4_objective(sfd, map, θ, obj, U0)
    base = _FVC4_BASE[]               # undo the parameters of earlier calls
    copyto!(sfd.prob.nu, base.nu); copyto!(sfd.prob.bvel, base.bvel); copyto!(sfd.prob.src, base.src)
    fv_set_params!(sfd, map, θ)
    U = copy(U0)
    st = fv_newton!(sfd, U; _FVC4_NEWTON...)
    st.converged || st.stagnated || error("FD Newton did not converge")
    return fv_objective(obj, U, sfd.prob)
end

# relative errors of central differences along d with steps eps·|θ|-scaled
function _fvc4_taylor(sfd, map, θ, d, obj, U0, gd; steps = (1e-3, 1e-4))
    scale = max(norm(θ, Inf), 1e-12) / norm(d, Inf)
    errs = Float64[]
    for e in steps
        ε = e * scale
        fd = (_fvc4_objective(sfd, map, θ .+ ε .* d, obj, U0) -
              _fvc4_objective(sfd, map, θ .- ε .* d, obj, U0)) / (2ε)
        push!(errs, abs(fd - gd) / max(abs(gd), 1e-300))
    end
    _fvc4_log!(:fd, minimum(errs))
    return errs
end

_fvc4_dense_cot(p, map, U, θ, λ) =
    -(transpose(ForwardDiff.jacobian(t -> fv_residual(U, fv_param_problem(map, p, t)), θ)) * λ)

@testset "FV parameter maps preserve the GNF viscosity model" begin
    p = fv_steady_problem(4, 3; rheology = CarreauYasuda(0.1, 0.01, 2.0, 2.0, 0.6),
                          gnf_eps = 1e-2, boundary_velocity = _fvc2_lid)
    U = randn(MersenneTwister(31), fv_nunknowns(p))
    maps = ((FVUniformViscosity(), [0.03]),
            (FVViscosityField(), fill(0.04, length(p.nu))),
            (FVSideVelocity(p, :N), [1.3]),
            (FVUniformSource(1), [0.7]))
    for (map, θ) in maps
        q = fv_param_problem(map, p, θ)
        @test q.visc === p.visc
        # Exact preservation contract: only the selected arrays may change.
        ref = FVSteadyProblem2D(p.grid, p.tag, q.nu, q.bvel, q.src, p.visc)
        @test fv_residual(U, q) == fv_residual(U, ref)
    end

    θ = fv_gnf_parameters(p)
    θ[1] *= 1.5
    map = FVProblemMap(fv_with_gnf_parameters, length(θ))
    s = fv_newton_solver(p)
    arrays = (s.prob.nu, s.prob.bvel, s.prob.src)
    asm, J = s.asm, s.J
    ref = fv_with_gnf_parameters(s.prob, θ)
    fv_newton_finalize!(s, U)
    @test all(isfinite, s.fac_sig)
    fv_set_params!(s, map, θ)
    @test fv_gnf_parameters(s.prob) == θ
    @test s.prob.visc === ref.visc
    @test all(a === b for (a, b) in zip(arrays, (s.prob.nu, s.prob.bvel, s.prob.src)))
    @test s.asm === asm && s.J === J
    @test fv_residual(U, s.prob) == fv_residual(U, ref)
    @test fv_gnf_parameters(p) != θ
    @test all(isnan, s.fac_sig)
    @test_throws ArgumentError fv_steady_gradient(s, U, θ, FVWallForce(1; region = :N);
                                                 params = map, finalize = false)
    # Reject both a different law and a different scalar type before any
    # array write. Either can invalidate the existing solver's concrete type.
    oldarrays = Base.map(copy, arrays)
    oldvisc = s.prob.visc
    for visc in (KrakeNN.FVNewtonianViscosity(),
                 FVGNFViscosity(CarreauYasuda(0.1f0, 0.01f0, 2f0, 2f0, 0.6f0), 1f-2))
        incompatible = FVProblemMap((q, t) -> FVSteadyProblem2D(q.grid, q.tag,
            q.nu .+ 1, q.bvel .+ 1, q.src .+ 1, visc), 0)
        @test_throws ArgumentError fv_set_params!(s, incompatible, Float64[])
        @test all(a == b for (a, b) in zip(arrays, oldarrays))
        @test s.prob.visc === oldvisc
    end
    changedgrid = fv_steady_problem(4, 3; h = 2p.grid.h).grid
    changedtag = copy(s.prob.tag); changedtag[2, 2] = FV_CELL_SOLID
    badproblems = ((ArgumentError, FVSteadyProblem2D(changedgrid, s.prob.tag,
                      arrays[1] .+ 1, arrays[2] .+ 1, arrays[3] .+ 1, oldvisc)),
                   (ArgumentError, FVSteadyProblem2D(s.prob.grid, changedtag,
                      arrays[1] .+ 1, arrays[2] .+ 1, arrays[3] .+ 1, oldvisc)),
                   (DimensionMismatch, FVSteadyProblem2D(s.prob.grid, s.prob.tag,
                      arrays[1] .+ 1, arrays[2] .+ 1, vcat(arrays[3], 1.0), oldvisc)))
    for (exception, bad) in badproblems
        @test_throws exception fv_set_params!(s, FVProblemMap((q, t) -> bad, 0), Float64[])
        @test all(a == b for (a, b) in zip(arrays, oldarrays))
        @test s.prob.visc === oldvisc
    end
    # A different model can require a different Jacobian pattern; reject it
    # before copying any parameter arrays into the existing solver.
    oldsrc = copy(s.prob.src)
    incompatible = FVProblemMap((q, t) ->
        FVSteadyProblem2D(q.grid, q.tag, q.nu, q.bvel, ones(length(q.src))), 0)
    @test_throws ArgumentError fv_set_params!(s, incompatible, Float64[])
    @test s.prob.src == oldsrc
end

# Small nonimmersed C3/C4 integration cases. Thresholds are the existing C4
# gates: 1e-12 for dense derivatives, 1e-6 for central finite differences.
# Infinite PowerLaw clamp bounds are excluded from FD directions; the clamp
# is inactive, and its two parameter derivatives must be exactly zero.
for law in (CarreauYasuda(0.1, 0.01, 2.0, 2.0, 0.6),
            PowerLaw(0.1, 0.7; nu_min = 0.0, nu_max = Inf))
    @testset "GNF adjoint integration: $(nameof(typeof(law)))" begin
        elapsed = @elapsed begin
            p = fv_steady_problem(4, 3; rheology = law, gnf_eps = 1e-2,
                                  boundary_velocity = _fvc2_lid)
            s = fv_newton_solver(p)
            U = zeros(fv_nunknowns(p))
            @test fv_newton!(s, U; _FVC4_NEWTON...).converged
            Jdense = ForwardDiff.jacobian(X -> fv_residual(X, p), U)
            lawmap = FVProblemMap(fv_with_gnf_parameters, length(fv_gnf_parameters(p)))
            maps = (("side velocity", FVSideVelocity(p, :N), [1.0]),
                    ("law", lawmap, fv_gnf_parameters(p)))
            objs = (FVWallForce(1; region = :W, pressure = :cell),
                    FVWallForce(1; region = :W, pressure = :linear),
                    FVStateObs(p, LineProfile(:u, [(2, j) for j in 1:3]), zeros(3)))
            worst_dense, worst_fd = 0.0, 0.0
            for obj in objs, (mname, parammap, θ) in maps
                r = fv_steady_gradient(s, U, θ, obj; params = parammap)
                gU = ForwardDiff.gradient(X -> fv_objective(obj, X, p), U)
                explicit = ForwardDiff.gradient(t -> fv_objective(obj, U,
                    fv_param_problem(parammap, p, t)), θ)
                Rθ = ForwardDiff.jacobian(t -> fv_residual(U,
                    fv_param_problem(parammap, p, t)), θ)
                dense = explicit - transpose(Rθ) * (transpose(Jdense) \ gU)
                err = norm(r.grad - dense, Inf) / max(norm(dense, Inf), 1e-12)
                worst_dense = max(worst_dense, err)
                @test err <= 1e-12
                @test norm(r.explicit - explicit, Inf) <= 1e-12 * max(norm(explicit, Inf), 1)
                active = law isa PowerLaw && mname == "law" ? (1:2) : eachindex(θ)
                if law isa PowerLaw && mname == "law"
                    @test all(iszero, r.grad[3:4])
                end
                sfd = fv_newton_solver(p)
                for k in active
                    # Parameter-wise central differences avoid scaling by an
                    # infinite inactive bound in the power-law vector.
                    ε = 1e-4 * max(abs(θ[k]), 0.01)
                    vals = Float64[]
                    for sign in (-1, 1)
                        t = copy(θ); t[k] += sign * ε
                        fv_set_params!(sfd, parammap, t)
                        X = copy(U)
                        st = fv_newton!(sfd, X; _FVC4_NEWTON...)
                        @test st.converged
                        push!(vals, fv_objective(obj, X, sfd.prob))
                    end
                    fd = (vals[2] - vals[1]) / (2ε)
                    # A component can vanish by symmetry; the full gradient
                    # supplies its natural scale without an arbitrary floor.
                    errfd = abs(fd - r.grad[k]) / max(norm(r.grad, Inf), 1e-12)
                    worst_fd = max(worst_fd, errfd)
                    @test errfd <= 1e-6
                end
            end
            # The cell viscosity field is unused by a GNF law, in both the
            # residual and the explicit wall-force derivative.
            for obj in objs[1:2]
                r = fv_steady_gradient(s, U, nothing, obj; params = FVViscosityField())
                @test all(iszero, r.grad)
                ru = fv_steady_gradient(s, U, [p.nu[1]], obj; params = FVUniformViscosity())
                @test all(iszero, ru.grad)
                @test all(iszero, KrakeNN._fv_wall_nu_gradient(obj, FVViscosityField(),
                                                           U, p, vec(p.nu)))
                @test all(iszero, KrakeNN._fv_wall_nu_gradient(obj, FVUniformViscosity(),
                                                           U, p, [p.nu[1]]))
            end
        end
        @info "C3/C4 GNF adjoint errors" law = nameof(typeof(law)) worst_dense worst_fd elapsed
    end
end

@testset "GNF wall force: accumulator derivatives and Newtonian limits" begin
    rng = MersenneTwister(32)
    kwargs = (; boundary_velocity = (side, x, y) -> (x + y, x - y))
    p = fv_steady_problem(4, 3; nu = 0.1, kwargs...)
    U = randn(rng, fv_nunknowns(p))
    solid = falses(4, 3); solid[2, 2] = true
    for mask in (falses(4, 3), solid),
        law in (Newtonian(0.1), PowerLaw(0.1, 1.0; nu_min = 0.0, nu_max = Inf))
        pn = fv_steady_problem(4, 3; nu = 0.1, solid = mask, kwargs...)
        q = fv_steady_problem(4, 3; rheology = law, gnf_eps = 1e-2, solid = mask, kwargs...)
        for dir in 1:2, region in (:E, :W, :N, :S, :solid), pressure in (:cell, :linear)
            obj = FVWallForce(dir; region, pressure)
            @test isequal(fv_objective(obj, U, q), fv_objective(obj, U, pn))
        end
    end
    for law in (CarreauYasuda(0.1, 0.01, 2.0, 2.0, 0.6),
                PowerLaw(0.1, 0.7; nu_min = 0.0, nu_max = Inf))
        q = fv_steady_problem(4, 3; rheology = law, gnf_eps = 1e-2, kwargs...)
        θ = fv_gnf_parameters(q)
        parammap = FVProblemMap(fv_with_gnf_parameters, length(θ))
        for dir in 1:2, region in (:E, :W, :N, :S), pressure in (:cell, :linear)
            obj = FVWallForce(dir; region, pressure)
            faces = KrakeNN._fv_wall_faces(obj, q)
            # Exercise the accumulator with dual law parameters directly.
            gθ = ForwardDiff.gradient(t -> KrakeNN._fv_wall_force_sum(obj, U,
                fv_with_gnf_parameters(q, t), faces), θ)
            @test norm(gθ, Inf) > 0
            @test KrakeNN._fv_objective_grad_theta(obj, parammap, U, q, θ) ≈ gθ rtol = 1e-12
            ε = 1e-6 * θ[1]
            tp, tm = copy(θ), copy(θ); tp[1] += ε; tm[1] -= ε
            fd = (fv_objective(obj, U, fv_with_gnf_parameters(q, tp)) -
                  fv_objective(obj, U, fv_with_gnf_parameters(q, tm))) / (2ε)
            @test abs(fd - gθ[1]) <= 1e-6 * max(abs(gθ[1]), 1e-12)
            _, gU = KrakeNN._fv_objective_grad_u(obj, U, q)
            ref = ForwardDiff.gradient(X -> fv_objective(obj, X, q), U)
            @test norm(gU - ref, Inf) <= 1e-12 * max(norm(ref, Inf), 1)
        end
    end
end

@testset "GNF boundary force equals the summed momentum residual" begin
    rng = MersenneTwister(34)
    for law in (CarreauYasuda(0.1, 0.01, 2.0, 2.0, 0.6),
                PowerLaw(0.1, 0.7; nu_min = 0.0, nu_max = Inf))
        p = fv_steady_problem(4, 3; conv = false, rheology = law, gnf_eps = 1e-2,
                              boundary_velocity = (side, x, y) -> (x + y, x - y))
        U = randn(rng, fv_nunknowns(p)); U[3:3:end] .= 0.7
        R = fv_residual(U, p)
        for dir in 1:2, pressure in (:cell, :linear)
            force = sum(fv_objective(FVWallForce(dir; region, pressure), U, p)
                        for region in (:E, :W, :N, :S))
            # Interior tractions cancel pairwise. At constant pressure the
            # two wall-pressure closures coincide; no source or convection
            # contributes. This checks force/residual physics independently
            # of differentiating the force implementation against itself.
            @test abs(force - sum(R[dir:3:end])) <= 1e-12 * max(sum(abs, R[dir:3:end]), 1)
        end
    end
end

@testset "IncNS steady FV adjoint (C4, CPU)" begin

    @testset "building blocks vs dense ForwardDiff" begin
        rng = MersenneTwister(4)
        solid = zeros(Bool, 14, 8); solid[5:7, 1:3] .= true; solid[10:11, 6:8] .= true
        cases = (
            ("cavity NS nu(x,y) 12x10",
             fv_steady_problem(12, 10; nu = (x, y) -> 0.01 * (1 + x * y), boundary_velocity = _fvc2_lid)),
            ("masked inlet/outlet channel 14x8",
             fv_steady_problem(14, 8; nu = (x, y) -> 0.05 * (1 + x * y), solid,
                               bc = (W = :inlet, E = :outlet, S = :wall, N = :wall),
                               boundary_velocity = (s, x, y) -> s === :W ? (4y * (1 - y), 0.0) : (0.0, 0.0))),
            ("x-periodic channel 12x7",
             fv_steady_problem(12, 7; nu = (x, y) -> 0.1 + 0.05 * y,
                               bc = (W = :periodic, E = :periodic, S = :wall, N = :wall),
                               source = (x, y) -> (1.0, 0.0, 0.0))),
            ("2-cell x-periodic channel 2x6 (E = W neighbour)",
             fv_steady_problem(2, 6; nu = (x, y) -> 0.1 + 0.05 * y + 0.01 * x,
                               bc = (W = :periodic, E = :periodic, S = :wall, N = :wall),
                               boundary_velocity = (s, x, y) -> s === :N ? (1.0, 0.0) : (0.0, 0.0))),
        )
        for (name, p) in cases
            @testset "$name" begin
                n = fv_nunknowns(p)
                s = fv_newton_solver(p)
                U = randn(rng, n)
                λ = randn(rng, n)
                side = p.grid.bcN == FV_BC_WALL ? :N : :S
                maps = ((FVUniformViscosity(), [0.03]),
                        (FVViscosityField(), vec(copy(p.nu))),
                        (FVSideVelocity(p, side), [1.3]),
                        (FVUniformSource(1), [0.7]),
                        (FVProblemMap((q, t) -> fv_param_problem(FVViscosityField(), q,
                                          vec([t[1] + t[2] * j + t[3] * i for i in 1:q.grid.Nx, j in 1:q.grid.Ny])), 3),
                         [0.05, 0.01, 0.002]))
                for (map, θ) in maps, chunk in (1, 4)
                    ws = fv_adjoint_workspace(s, map; chunk)
                    copyto!(ws.λ, λ)
                    # the solver's problem must be the problem at θ
                    s.prob = fv_param_problem(map, p, θ)
                    cot = KrakeNN._fv_param_cotangent!(ws, map, s, U, θ)
                    ref = _fvc4_dense_cot(p, map, U, θ, λ)
                    @test _fvc4_log!(:dense, norm(vec(Array(cot)) - ref, Inf) / max(norm(ref, Inf), 1)) <= 1e-12
                end
                s.prob = p
                # field colouring: a handful of colours (greedy largest-first;
                # 8 on the masked channel, measured 2026-11-05)
                _, nnc = fv_nu_coloring(p)
                @test nnc <= 9
            end
        end

        # objective gradients (∂𝒥/∂U, explicit ∂𝒥/∂θ) vs dense ForwardDiff
        p = cases[2][2]
        U = randn(rng, fv_nunknowns(p))
        objs = (FVWallForce(1; region = :solid), FVWallForce(2; region = :N),
                FVWallForce(1; region = :solid, pressure = :linear),
                FVStateObs(p, LineProfile(:u, [(9, j) for j in 4:8]), randn(rng, 5)),
                FVStateObs(p, LineProfile(:p, [(2, j) for j in 1:8]), randn(rng, 8); scale = 0.5),
                fv_flow_rate(p, 9), fv_pressure_drop(p, 2, 13))
        for obj in objs
            J, gU = KrakeNN._fv_objective_grad_u(obj, U, p)
            @test J ≈ fv_objective(obj, U, p) rtol = 1e-14
            ref = ForwardDiff.gradient(X -> fv_objective(obj, X, p), U)
            @test norm(gU - ref, Inf) <= 1e-13 * max(norm(ref, Inf), 1)
            for (map, θ) in ((FVViscosityField(), vec(copy(p.nu))), (FVSideVelocity(p, :W), [1.0]))
                ge = KrakeNN._fv_objective_grad_theta(obj, map, U, p, θ)
                refθ = ForwardDiff.gradient(t -> fv_objective(obj, U, fv_param_problem(map, p, t)), θ)
                @test norm(ge - refθ, Inf) <= 1e-13 * max(norm(refθ, Inf), 1)
            end
        end
        # flow rate: h Σ u over the fluid cells of the column
        @test fv_objective(fv_flow_rate(p, 9), U, p) ≈
              p.grid.h * sum(U[fv_index(p.grid, 9, j, 1)] for j in 1:8)
    end

    @testset "local wall-force derivatives preserve every face closure" begin
        rng = MersenneTwister(33)
        solid = falses(5, 4); solid[3, 2:3] .= true
        cases = (
            fv_steady_problem(5, 4; solid, nu = (x, y) -> 0.03 + 0.02x + 0.01y,
                bc = (W = :inlet, E = :outlet, S = :wall, N = :wall),
                boundary_velocity = (side, x, y) -> (0.1 + x * y, 0.2 - y)),
            fv_steady_problem(5, 4; nu = (x, y) -> 0.04 + 0.01x,
                boundary_velocity = (side, x, y) -> (x + y, x - y)),
            fv_steady_problem(2, 4; nu = (x, y) -> 0.04 + 0.01y,
                bc = (W = :periodic, E = :periodic, S = :wall, N = :wall)),
            fv_steady_problem(1, 4; nu = (x, y) -> 0.04 + 0.01y,
                bc = (W = :periodic, E = :periodic, S = :wall, N = :wall)),
            fv_steady_problem(4, 2; nu = (x, y) -> 0.04 + 0.01x,
                bc = (W = :wall, E = :wall, S = :periodic, N = :periodic)),
            fv_steady_problem(5, 4; solid,
                rheology = CarreauYasuda(0.1, 0.01, 2.0, 2.0, 0.6), gnf_eps = 1e-2,
                boundary_velocity = (side, x, y) -> (x + y, x - y)),
        )
        for (case, p) in enumerate(cases), dir in 1:2,
            region in (:solid, :E, :W, :N, :S), pressure in (:cell, :linear)
            @testset "case=$case dir=$dir region=$region pressure=$pressure" begin
                obj = FVWallForce(dir; region, pressure)
                U = randn(rng, fv_nunknowns(p))
                J, gU = KrakeNN._fv_objective_grad_u(obj, U, p)
                ref = ForwardDiff.gradient(X -> fv_objective(obj, X, p), U)
                @test J ≈ fv_objective(obj, U, p) rtol = 1e-14 atol = 1e-14
                @test norm(gU - ref, Inf) <= 1e-13 * max(norm(ref, Inf), 1)
                for (map, θ) in ((FVViscosityField(), vec(copy(p.nu))),
                                 (FVSideVelocity(p, :W), [1.3]),
                                 (FVUniformViscosity(), [0.05]),
                                 (FVUniformSource(1), [0.7]))
                    ge = KrakeNN._fv_objective_grad_theta(obj, map, U, p, θ)
                    refθ = ForwardDiff.gradient(t -> fv_objective(obj, U, fv_param_problem(map, p, t)), θ)
                    @test norm(ge - refθ, Inf) <= 1e-13 * max(norm(refθ, Inf), 1)
                end
            end
        end
    end

    @testset "cavity NS 24²: adjoint vs FD, tangent, factor reuse" begin
        p = fv_steady_problem(24, 24; nu = (x, y) -> 0.02 * (1 + 0.5 * x * y), boundary_velocity = _fvc2_lid)
        s = fv_newton_solver(p)
        U = zeros(fv_nunknowns(p))
        st = fv_newton!(s, U; _FVC4_NEWTON...)
        @test st.converged
        sfd = fv_newton_solver(p); _FVC4_BASE[] = p
        U0 = copy(U)

        # transposed solve with the finalized Newton factors
        fv_newton_finalize!(s, U)
        rng = MersenneTwister(5)
        gv = randn(rng, length(U)); λ = similar(gv)
        lin_solve!(λ, s.cache, gv; transpose = true)
        @test norm(transpose(s.J) * λ - gv) / norm(gv) <= 1e-12
        v = randn(rng, length(U))
        @test abs(dot(λ, s.J * v) - dot(gv, v)) <= 1e-12 * norm(λ) * norm(s.J * v)

        # one symbolic analysis: the gradient refactorizes, never re-analyses
        F = s.cache.factor
        sym = F.symbolic.p
        nf, nr = s.n_factorize, s.n_refactorize
        lp = LineProfile(:u, [(12, j) for j in 2:2:24])
        prof = FVStateObs(p, lp, zeros(12))
        objs = (("lid drag", FVWallForce(1; region = :N)), ("u profile misfit", prof))
        Ny = p.grid.Ny
        nuxy(θ) = [θ[1] + θ[2] * ((j - 0.5) / Ny) + θ[3] * ((i - 0.5) / Ny)^2 for i in 1:24, j in 1:24]
        poly = FVProblemMap((q, θ) -> fv_param_problem(FVViscosityField(), q, vec(nuxy(θ))), 3)
        for (oname, obj) in objs
            @testset "$oname" begin
                # scalar parameters: uniform ν (replaces the field), lid speed
                for (mname, map, θ) in (("uniform nu", FVUniformViscosity(), [0.02]),
                                        ("lid speed", FVSideVelocity(p, :N), [1.0]),
                                        ("nu = a + b y + c x^2", poly, [0.02, 0.01, 0.005]))
                    # the gradient is taken at the problem of θ: solve there
                    sθ = fv_newton_solver(p); fv_set_params!(sθ, map, θ)
                    Uθ = copy(U0); fv_newton!(sθ, Uθ; _FVC4_NEWTON...)
                    r = fv_steady_gradient(sθ, Uθ, θ, obj; params = map)
                    @test r.residual_inf <= 1e-13
                    dirs = length(θ) == 1 ? [[1.0]] :
                           [[1.0, 0, 0], [0, 1.0, 0], [0.3, -1.0, 2.0], randn(rng, 3)]
                    for d in dirs
                        errs = _fvc4_taylor(sfd, map, θ, d, obj, Uθ, dot(r.grad, d))
                        @test minimum(errs) <= 1e-6                     # V3a
                        @test errs[2] <= errs[1] || errs[2] <= 1e-9     # FD error shrinks with the step
                    end
                    # adjoint vs forward tangent along one direction
                    d = length(θ) == 1 ? [1.0] : [0.3, -1.0, 2.0]
                    dU = fv_steady_tangent(sθ, Uθ, θ, d; params = map)
                    tg = dot(r.gU, dU) + dot(r.explicit, d)
                    @test _fvc4_log!(:tangent, abs(tg - dot(r.grad, d)) / max(abs(tg), norm(r.grad) * norm(d))) <= 1e-12
                end
                # viscosity field: ≥ 3 directions + single cells
                map = FVViscosityField()
                θ = vec(copy(p.nu))
                r = fv_steady_gradient(s, U, θ, obj; params = map)
                @test size(r.s) == size(p.nu)
                for d in (randn(rng, length(θ)), vec([sin(3x) * cos(2y) for x in 1:24, y in 1:24]),
                          vec([exp(-((i - 20)^2 + (j - 22)^2) / 8) for i in 1:24, j in 1:24]),
                          Float64.(1:length(θ) .== 24 * 23 + 12))
                    errs = _fvc4_taylor(sfd, map, θ, d, obj, U0, dot(r.grad, d))
                    @test minimum(errs) <= 1e-6
                    dU = fv_steady_tangent(s, U, θ, d; params = map)
                    tg = dot(r.gU, dU) + dot(r.explicit, d)
                    @test _fvc4_log!(:tangent, abs(tg - dot(r.grad, d)) / max(abs(tg), norm(r.grad) * norm(d))) <= 1e-12
                end
            end
        end
        # all of the above on `s`: still the analysis of its first Newton step
        @test s.n_factorize == nf == 1
        @test s.n_refactorize > nr
        @test s.cache.factor === F && F.symbolic.p == sym
        # finalize = false reuses the current factors as they are
        nr2 = s.n_refactorize
        r1 = fv_steady_gradient(s, U, nothing, prof; params = FVViscosityField())
        r2 = fv_steady_gradient(s, U, nothing, prof; params = FVViscosityField(), finalize = false)
        @test s.n_refactorize == nr2 + 1
        @test norm(r1.grad - r2.grad, Inf) <= 1e-14 * norm(r1.grad, Inf)
        # guard: finalize = false refuses factors that are not the unshifted J at U
        Upert = copy(U); Upert[1] += 1e-3
        @test_throws ArgumentError fv_steady_gradient(s, Upert, nothing, prof; params = FVViscosityField(), finalize = false)
        fv_jacobian!(s, U; shift = 1e-2); KrakeNN._fv_factor!(s)     # pseudo-transient factors left in the cache
        @test_throws ArgumentError fv_steady_gradient(s, U, nothing, prof; params = FVViscosityField(), finalize = false)
        fv_newton_finalize!(s, U)                      # back to the normal path
        r4 = fv_steady_gradient(s, U, nothing, prof; params = FVViscosityField(), finalize = false)
        @test r4.grad == r1.grad
        fv_set_params!(s, FVViscosityField(), vec(Array(s.prob.nu)))
        @test_throws ArgumentError fv_steady_gradient(s, U, nothing, prof; params = FVViscosityField(), finalize = false)
        fv_newton_finalize!(s, U)
        # the FD solver went through many fv_set_params! + Newton solves: one analysis
        @test sfd.n_factorize == 1
        # a preallocated workspace gives the same gradient
        ws = fv_adjoint_workspace(s, FVViscosityField())
        r3 = fv_steady_gradient(s, U, nothing, prof; params = FVViscosityField(), ws)
        @test r3.grad == r1.grad
        @test_throws ArgumentError fv_steady_gradient(s, U, nothing, prof; params = FVUniformViscosity())
    end

    @testset "masked inlet/outlet channel: drag and pressure drop" begin
        solid = zeros(Bool, 32, 8); solid[11:14, 3:6] .= true
        for conv in (false, true)
            p = fv_steady_problem(32, 8; h = 1 / 8, solid, conv,
                                  nu = (x, y) -> 0.05 * (1 + 0.5 * sin(2x) * y),
                                  bc = (W = :inlet, E = :outlet, S = :wall, N = :wall),
                                  boundary_velocity = (s, x, y) -> s === :W ? (4y * (1 - y), 0.0) : (0.0, 0.0))
            s = fv_newton_solver(p); sfd = fv_newton_solver(p); _FVC4_BASE[] = p
            U = zeros(fv_nunknowns(p))
            @test fv_newton!(s, U; _FVC4_NEWTON...).converged
            rng = MersenneTwister(9)
            for obj in (FVWallForce(1; region = :solid), fv_pressure_drop(p, 4, 24))
                r = fv_steady_gradient(s, U, nothing, obj; params = FVViscosityField())
                @test all(iszero, Array(r.s)[solid])
                θ = vec(copy(p.nu))
                for d in (randn(rng, length(θ)), vec(Float64.(.!solid)))
                    errs = _fvc4_taylor(sfd, FVViscosityField(), θ, d, obj, U, dot(r.grad, d))
                    @test minimum(errs) <= 1e-6
                end
                # inlet speed (FVSideVelocity on the inlet side, parabolic shape)
                map = FVSideVelocity(p, :W; shape = (x, y) -> (4y * (1 - y), 0.0))
                r = fv_steady_gradient(s, U, [1.0], obj; params = map)
                errs = _fvc4_taylor(sfd, map, [1.0], [1.0], obj, U, r.grad[1])
                @test minimum(errs) <= 1e-6
                if !conv
                    # Stokes: the state is linear in the inlet speed, J too
                    @test r.grad[1] ≈ fv_objective(obj, U, p) rtol = 1e-9
                end
            end
        end
    end

    @testset "dQ/dν = −Q/ν at fixed body force (exact discrete identity)" begin
        for conv in (false, true)
            p = fv_steady_problem(8, 16; nu = 0.1, conv,
                                  bc = (W = :periodic, E = :periodic, S = :wall, N = :wall),
                                  source = (x, y) -> (1.0, 0.0, 0.0))
            s = fv_newton_solver(p)
            U = zeros(fv_nunknowns(p))
            @test fv_newton!(s, U; _FVC4_NEWTON...).converged
            Q = fv_flow_rate(p, 3)
            r = fv_steady_gradient(s, U, [0.1], Q; params = FVUniformViscosity())
            @test abs(r.grad[1] + r.J / 0.1) <= 1e-8 * abs(r.J / 0.1)
            r = fv_steady_gradient(s, U, [1.0], Q; params = FVUniformSource(1))
            @test abs(r.grad[1] - r.J) <= 1e-8 * abs(r.J)          # Q linear in the force
        end
    end

    @info "C4 adjoint error levels (max over cases)" fd_best_step_rel = _FVC4_LOG[:fd] tangent_rel = _FVC4_LOG[:tangent] param_cotangent_vs_dense_rel = _FVC4_LOG[:dense]

    @testset "platform contract" begin
        @test SteadyAdjoint in capabilities(IncNS(:fv_newton))
        @test AssembledJacobian in capabilities(IncNS(:fv_newton))
        for d in (:simple, :cavity, :cavity_mg, :projection, :manifold)
            @test AssembledJacobian ∉ capabilities(IncNS(d))
            @test SteadyAdjoint ∉ capabilities(IncNS(d))
        end
        @test AssembledJacobian ∉ capabilities(LBM())
    end
end
