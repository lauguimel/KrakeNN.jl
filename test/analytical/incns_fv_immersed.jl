# C5-S012 closure and C1-C4 seam checks, CPU unit scale only.
# Exact identities below come from polynomial interpolation and preservation
# of the same discrete operator. Floating comparisons use 1e-12 relative to
# the dense ForwardDiff reference, allowing only reduction round-off.
using Test
using LinearAlgebra
using Random
using ForwardDiff
using Kraken
using Kraken.SparseArrays

include(joinpath(@__DIR__, "incns_fv_immersed_geometry.jl"))

_fvc5_lid(side, x, y) = side === :N ? (1.0, 0.0) : (0.0, 0.0)
_fvc5_geom(; kwargs...) = fv_immersed_circle(12, 12, 6.13, 5.87, 2.21; h = 1.0, kwargs...)
function _fvc5_problem(; closure = :quadratic, mass = :M1, pressure = :linear,
                       rheology = nothing, conv = true)
    immersed = _fvc5_geom(; closure, mass, pressure)
    kw = rheology === nothing ? (; nu = (x, y) -> 0.1 * (1 + 0.03x + 0.02y)) :
                               (; rheology, gnf_eps = 1e-2)
    return fv_steady_problem(12, 12; h = 1.0, immersed, conv,
                              boundary_velocity = _fvc5_lid, kw...)
end
_fvc5_with_geom(p, geom; grid = p.grid, tag = p.tag, nu = p.nu) =
    FVSteadyProblem2D(grid, tag, nu, p.bvel, p.src, p.visc, geom)
_fvc5_geom_copy(g; qcut = copy(g.qcut), wvel = copy(g.wvel), tag = copy(g.tag),
                  h = g.h, closure = g.closure) =
    FVImmersedBoundary2D(qcut, wvel, tag, h, closure)
_fvc5_dense_jac(p, U) = ForwardDiff.jacobian(X -> fv_residual(X, p), U)

@testset "Immersed ghost coefficients reproduce affine and quadratic profiles" begin
    for q in (1//7, 1//2, 4//5, 1//1), ghost in (0x01, 0x02)
        f(x) = 3//2 - 2//3 * x + (ghost == 0x02 ? 5//7 * x^2 : 0)
        @test Kraken._fv_ghost_value(f(-1), f(0), f(q), q, ghost) == f(1)
        @test @inferred(Kraken._fv_ghost_value(Float64(f(-1)), Float64(f(0)),
            Float64(f(q)), Float64(q), ghost)) isa Float64
    end
end

@testset "Immersed C1 exact compatibility" begin
    p = _fvc5_problem(; closure = :c1)
    old = fv_steady_problem(12, 12; h = 1.0, solid = p.tag .== FV_CELL_SOLID,
                            nu = (x, y) -> 0.1 * (1 + 0.03x + 0.02y),
                            boundary_velocity = _fvc5_lid)
    U = 0.3 .* randn(MersenneTwister(501), fv_nunknowns(p))
    @test fv_residual(U, p) == fv_residual(U, old)
    s, s0 = fv_newton_solver(p), fv_newton_solver(old)
    @test fv_jacobian!(s, U) == fv_jacobian!(s0, U)
    for dir in (1, 2), pressure in (:cell, :linear)
        obj = FVWallForce(dir; pressure)
        @test fv_objective(obj, U, p) == fv_objective(obj, U, old)
    end
    # A small linear Stokes solve checks the existing C4 gradient path too.
    ps = _fvc5_problem(; closure = :c1, conv = false)
    os = fv_steady_problem(12, 12; h = 1.0, solid = ps.tag .== FV_CELL_SOLID,
                           conv = false, nu = (x, y) -> 0.1 * (1 + 0.03x + 0.02y),
                           boundary_velocity = _fvc5_lid)
    ss, so = fv_newton_solver(ps), fv_newton_solver(os)
    Us, Uo = zeros(fv_nunknowns(ps)), zeros(fv_nunknowns(os))
    @test fv_newton!(ss, Us; atol = 1e-12, maxit = 5).converged
    @test fv_newton!(so, Uo; atol = 1e-12, maxit = 5).converged
    @test Us == Uo
    obj = FVWallForce(1)
    rs = fv_steady_gradient(ss, Us, nothing, obj; params = FVViscosityField())
    ro = fv_steady_gradient(so, Uo, nothing, obj; params = FVViscosityField())
    @test rs.J == ro.J
    @test rs.grad == ro.grad
    law = CarreauYasuda(0.1, 0.01, 2.0, 2.0, 0.6)
    pg = _fvc5_problem(; closure = :c1, rheology = law)
    og = fv_steady_problem(12, 12; h = 1.0, solid = pg.tag .== FV_CELL_SOLID,
                           rheology = law, gnf_eps = 1e-2, boundary_velocity = _fvc5_lid)
    @test fv_residual(U, pg) == fv_residual(U, og)
    @test fv_jacobian!(fv_newton_solver(pg), U) == fv_jacobian!(fv_newton_solver(og), U)
end

@testset "Immersed given viscosity reaches the nominal wall face" begin
    p = _fvc5_problem()
    ib, g = p.immersed, p.grid
    G = (0.2, 0.3, -0.4, 0.5, 0.0, 0.0)
    dun, dvn = 0.7, -0.2
    for d in 1:4
        i, j = Tuple(findfirst(!iszero, view(ib.qcut, d, :, :)))
        ex, ey = (d == 1) - (d == 2), (d == 3) - (d == 4)
        xface, yface = (i - 0.5 + ex / 2) * g.h, (j - 0.5 + ey / 2) * g.h
        # The prescribed linear viscosity is exact at the nominal face.
        nuf = 0.1 * (1 + 0.03xface + 0.02yface)
        sx = d in (1, 3) ? 1 : -1
        tx, ty = d <= 2 ? (2nuf * dun, nuf * (dvn + sx * G[2])) :
                          (nuf * (dun + sx * G[3]), 2nuf * dvn)
        actual = Kraken._fv_wall_stress(p.nu, p.visc, g, i, j, d, G,
                                        dun, dvn, ib.qcut, ib.closure)
        @test actual[1] ≈ tx rtol = 1e-14
        @test actual[2] ≈ ty rtol = 1e-14
        oi, oj = i - ex, j - ey
        U = randn(MersenneTwister(510 + d), fv_nunknowns(p))
        obj = FVWallForce(d <= 2 ? 1 : 2)
        field = FVViscosityField()
        face(q) = Kraken._fv_wall_force_face(obj, U, q, i, j, d,
            ib.qcut, ib.wvel, ib.closure)
        dense = ForwardDiff.gradient(t -> face(fv_param_problem(field, p, t)), vec(p.nu))
        cp, co = i + (j - 1) * g.Nx, oi + (oj - 1) * g.Nx
        @test !iszero(dense[co])
        @test dense[cp] ≈ -3dense[co] rtol = 1e-14
        # The optional local P dual must not replace the O value.
        nd = ForwardDiff.Dual(p.nu[i, j], 1.0)
        local_face = Kraken._fv_wall_force_face(obj, U, p, i, j, d,
            ib.qcut, ib.wvel, ib.closure, nd)
        @test ForwardDiff.partials(local_face)[1] ≈ dense[cp] rtol = 1e-14
    end
end

@testset "Immersed mixed quadratic stress reaches the nominal face" begin
    # Independent polynomial gradients, including both mixed derivatives.
    # The frozen bound is 512 machine epsilons times the tensor scale: only
    # roundoff is allowed for quadratic interpolation and affine gradients.
    velocity(x, y) = (x/4 - y/2 + x^2/16 + x*y/8 - y^2/32,
                      -x/8 + y/4 - x^2/32 + x*y/16 + y^2/32)
    gradient(x, y) = (1/4 + x/8 + y/8, -1/2 + x/8 - y/16,
                      -1/8 - x/16 + y/16, 1/4 + x/16 + y/16)
    viscosity(x, y) = 1 + x/32 + y/16
    for T in (Float64, Float32), gnf in (false, true)
        ib = fv_immersed_circle(12, 12, T(6.13), T(5.87), T(2.21); h = one(T), mass=:M1,
            wall_velocity = (wall, x, y) -> velocity(x, y))
        law = CarreauYasuda(T(0.1), T(0.01), T(2), T(2), T(0.6))
        kw = gnf ? (; rheology = law, gnf_eps = T(1e-2)) : (; nu = viscosity)
        p = fv_steady_problem(12, 12; h = one(T), T, immersed = ib, conv = false, kw...)
        g = p.grid
        U = fv_state(p, (x,y) -> velocity(x,y)[1], (x,y) -> velocity(x,y)[2],
                         (x,y) -> zero(x))
        for (d,i,j) in Tuple.(findall(>(0), ib.qcut))
            ex, ey = (d == 1) - (d == 2), (d == 3) - (d == 4)
            x, y = T(i - 0.5 + ex/2), T(j - 0.5 + ey/2)
            ux, uy, vx, vy = gradient(x, y)
            G = Kraken._fv_cell_grad(U,p.tag,p.bvel,g,i,j,ib.qcut,ib.wvel,ib.closure)
            F = Kraken._fv_wall_gradient(U,p.tag,p.bvel,g,i,j,d,G,ib.qcut,ib.wvel,ib.closure)
            wall = Kraken._fv_wall_data(U,p.bvel,g,i,j,d,ib.qcut,ib.wvel,ib.closure)
            actual = Kraken._fv_wall_stress(p.nu,p.visc,g,i,j,d,F,wall[8],wall[9],ib.qcut,ib.closure)
            nuf = gnf ? effective_viscosity(law, sqrt(2ux^2 + 2vy^2 + (uy+vx)^2 + p.visc.eps^2)) :
                        viscosity(x,y)
            exact = (nuf * (2ux*ex + (uy+vx)*ey), nuf * ((uy+vx)*ex + 2vy*ey))
            @test maximum(abs.(actual .- exact)) <= 512eps(T) * max(maximum(abs, exact), 1)
            tangent = d <= 2 ? (2,4) : (1,3)
            @test maximum(abs(F[k] - (ux,uy,vx,vy)[k]) for k in tangent) <=
                  512eps(T) * max(abs(ux),abs(uy),abs(vx),abs(vy),1)
            @test Kraken._fv_wall_gradient(U,p.tag,p.bvel,g,i,j,d,G,ib.qcut,ib.wvel,
                                            FVImmersedClosure(:c1)) === G
        end
    end
end

@testset "Immersed L/Q Jacobian and adjoint support" begin
    rng = MersenneTwister(502)
    colors_seen = Int[]
    for closure in (:linear, :quadratic), mass in (:M1, :M2),
        pressure in (:zero_gradient, :linear),
        law in (nothing, CarreauYasuda(0.1, 0.01, 2.0, 2.0, 0.6))
        p = _fvc5_problem(; closure, mass, pressure, rheology = law)
        U = 0.3 .* randn(rng, fv_nunknowns(p))
        S = fv_jacobian_pattern(p)
        _, nc = fv_jacobian_coloring(S)
        push!(colors_seen, nc)
        dense = _fvc5_dense_jac(p, U)
        @test all(S[k] for k in findall(!iszero, dense))
        # All actual dependencies remain inside the frozen radius-two box.
        @test all(findall(!iszero, dense)) do k
            row, col = Tuple(k)
            ri, rj = mod1((row - 1) ÷ 3 + 1, p.grid.Nx), (row - 1) ÷ (3p.grid.Nx) + 1
            ci, cj = mod1((col - 1) ÷ 3 + 1, p.grid.Nx), (col - 1) ÷ (3p.grid.Nx) + 1
            max(abs(ri - ci), abs(rj - cj)) <= 2
        end
        s = fv_newton_solver(p)
        @test Matrix(fv_jacobian!(s, U)) == dense
        @test all(isfinite, dense)
        # Every ghost O and its tangential neighbours are in the local state
        # stencil. Check all cut faces explicitly, including corner cuts.
        faces = Kraken._fv_wall_faces(FVWallForce(1), p)
        @test length(faces) == count(!iszero, p.immersed.qcut)
        @test all(faces) do (i, j, d)
            oi = i - (d == 1) + (d == 2)
            oj = j - (d == 3) + (d == 4)
            idx = Kraken._fv_wall_state(U, p, i, j).index
            tangents = d <= 2 ? (3,4) : (1,2)
            cells = [(oi,oj)]
            for t in tangents
                kind, ti, tj = Kraken._fv_face_kind(p.grid,p.tag,oi,oj,t)
                kind == Kraken._FV_F_INT && push!(cells,(ti,tj))
            end
            all(fv_index(p.grid, a, b, c) in idx for (a,b) in cells for c in 1:3)
        end
        # Local state and viscosity derivatives must match full-grid duals,
        # including O in the extrapolated given viscosity and all GNF terms.
        for dir in (1, 2), force_pressure in (:cell, :linear)
            obj = FVWallForce(dir; pressure = force_pressure)
            value, gu = Kraken._fv_objective_grad_u(obj, U, p)
            gd = ForwardDiff.gradient(X -> fv_objective(obj, X, p), U)
            @test all(k in fv_objective_support(obj,p) for k in findall(!iszero,gd))
            @test value ≈ fv_objective(obj, U, p) rtol = 1e-14 atol = 1e-14
            @test norm(gu - gd, Inf) <= 1e-12 * max(norm(gd, Inf), 1)
            for (map, θ) in ((FVViscosityField(), vec(copy(p.nu))),
                             (FVUniformViscosity(), [0.17]))
                ge = Kraken._fv_objective_grad_theta(obj, map, U, p, θ)
                ref = ForwardDiff.gradient(t -> fv_objective(obj, U,
                    fv_param_problem(map, p, t)), θ)
                @test norm(ge - ref, Inf) <= 1e-12 * max(norm(ref, Inf), 1)
                law === nothing || @test all(iszero, ge) && all(iszero, ref)
            end
            if law !== nothing
                θ = fv_gnf_parameters(p)
                map = FVProblemMap(fv_with_gnf_parameters, length(θ))
                ge = Kraken._fv_objective_grad_theta(obj, map, U, p, θ)
                ref = ForwardDiff.gradient(t -> fv_objective(obj, U,
                    fv_with_gnf_parameters(p, t)), θ)
                @test norm(ge - ref, Inf) <= 1e-12 * max(norm(ref, Inf), 1)
                @test any(!iszero, ge)
            end
        end
        if law === nothing
            map = FVViscosityField()
            θ = vec(copy(p.nu))
            λ = randn(rng, length(U))
            ws = fv_adjoint_workspace(s, map)
            copyto!(ws.λ, λ)
            cot = Kraken._fv_param_cotangent!(ws, map, s, U, θ)
            densecot = ForwardDiff.gradient(t -> -dot(λ, fv_residual(U,
                fv_param_problem(map, p, t))), θ)
            @test norm(vec(cot) - densecot, Inf) <= 1e-12 * max(norm(densecot, Inf), 1)
        end
    end
    @info "C5 immersed Jacobian colours" minimum = minimum(colors_seen) maximum = maximum(colors_seen)
end

@testset "Immersed geometry survives parameter and adaptation seams" begin
    p = _fvc5_problem()
    g = p.immersed
    adapted = fv_adapt(Array, p)
    @test adapted.immersed.qcut == g.qcut
    @test adapted.immersed.wvel == g.wvel
    @test adapted.immersed.tag == g.tag
    @test adapted.immersed.closure === g.closure
    @test_throws DimensionMismatch fv_steady_problem(13, 12; h = 1.0, immersed = g)
    @test_throws ArgumentError fv_steady_problem(12, 12; h = 0.5, immersed = g)
    for (map, θ) in ((FVUniformViscosity(), [0.2]),
                     (FVViscosityField(), fill(0.2, length(p.nu))),
                     (FVSideVelocity(p, :N), [1.3]), (FVUniformSource(1), [0.7]))
        @test fv_param_problem(map, p, θ).immersed === g
    end
    pg = _fvc5_problem(; rheology = CarreauYasuda(0.1, 0.01, 2.0, 2.0, 0.6))
    θg = fv_gnf_parameters(pg); θg[1] *= 1.1
    @test fv_with_gnf_parameters(pg, θg).immersed === pg.immersed

    map = FVProblemMap((q, θ) -> _fvc5_with_geom(q,
        _fvc5_geom_copy(q.immersed; qcut = q.immersed.qcut .* (1 + θ[1]),
            wvel = q.immersed.wvel .+ θ[2])), 2)
    # Arbitrary cut-array updates describe only the residual. Analytic surface
    # metadata must instead be rebuilt from the analytic geometry parameters.
    s = fv_newton_solver(_fvc5_with_geom(p, _fvc5_geom_copy(p.immersed)))
    arrays = (s.prob.nu, s.prob.bvel, s.prob.src, s.prob.immersed.qcut, s.prob.immersed.wvel)
    ref = fv_param_problem(map, s.prob, [1e-4, 0.03])
    fv_set_params!(s, map, [1e-4, 0.03])
    @test all(a === b for (a, b) in zip(arrays,
        (s.prob.nu, s.prob.bvel, s.prob.src, s.prob.immersed.qcut, s.prob.immersed.wvel)))
    @test s.prob.immersed.qcut == ref.immersed.qcut
    @test s.prob.immersed.wvel == ref.immersed.wvel

    oldnu = copy(s.prob.nu)
    oldq, oldw = copy(s.prob.immersed.qcut), copy(s.prob.immersed.wvel)
    tag = copy(s.prob.tag); tag[2, 2] = FV_CELL_SOLID
    gtag = copy(s.prob.immersed.tag); gtag[2, 2] = FV_CELL_SOLID
    grid = s.prob.grid
    changed_grid = FVGrid2D(grid.Nx, grid.Ny, 1.01 * grid.h, grid.conv,
                            grid.bcE, grid.bcW, grid.bcN, grid.bcS)
    bads = (_fvc5_with_geom(s.prob, _fvc5_geom_copy(s.prob.immersed; tag = gtag);
                            tag, nu = fill(99.0, size(oldnu))),
            _fvc5_with_geom(s.prob, _fvc5_geom_copy(s.prob.immersed; h = changed_grid.h);
                            grid = changed_grid, nu = fill(99.0, size(oldnu))),
            _fvc5_with_geom(s.prob, _fvc5_geom_copy(s.prob.immersed;
                            closure = FVImmersedClosure(:linear)); nu = fill(99.0, size(oldnu))))
    for bad in bads
        @test_throws ArgumentError fv_set_params!(s, FVProblemMap((q, θ) -> bad, 0), Float64[])
        @test s.prob.nu == oldnu
        @test s.prob.immersed.qcut == oldq && s.prob.immersed.wvel == oldw
    end
end

struct _FVC5PayloadTag end
@testset "Immersed q-only and wall-only dual payloads" begin
    p = _fvc5_problem()
    U = randn(MersenneTwister(503), fv_nunknowns(p))
    g = p.immersed
    D = ForwardDiff.Dual{_FVC5PayloadTag,Float64,1}
    duals(A) = map(x -> D(x, ForwardDiff.Partials((1.0,))), A)
    @test isbitstype(typeof(g.closure))
    @test isbitstype(D)
    d, i, j = Tuple(first(findall(>(0), g.qcut)))
    Ud = duals(U)
    @test @inferred(Kraken._fv_wall_data(Ud, p.bvel, p.grid, i, j, d,
        g.qcut, g.wvel, g.closure)) isa NTuple{9,D}
    @test @inferred(Kraken._fv_cell_grad(Ud, p.tag, p.bvel, p.grid, i, j,
        g.qcut, g.wvel, g.closure)) isa NTuple{6,D}
    @test @inferred(fv_cell_residual(Ud, p.tag, p.nu, p.bvel, p.src, p.grid,
        i, j, p.visc, g.qcut, g.wvel, g.closure)) isa NTuple{3,D}
    for gd in (_fvc5_geom_copy(g; qcut = duals(g.qcut)),
               _fvc5_geom_copy(g; wvel = duals(g.wvel)))
        pd = _fvc5_with_geom(p, gd)
        @test @inferred(Kraken._fv_wall_data(U, p.bvel, p.grid, i, j, d,
            gd.qcut, gd.wvel, gd.closure)) isa NTuple{9,D}
        @test @inferred(Kraken._fv_cell_grad(U, p.tag, p.bvel, p.grid, i, j,
            gd.qcut, gd.wvel, gd.closure)) isa NTuple{6,D}
        @test @inferred(fv_cell_residual(U, p.tag, p.nu, p.bvel, p.src, p.grid,
            i, j, p.visc, gd.qcut, gd.wvel, gd.closure)) isa NTuple{3,D}
        R = fv_residual(U, pd)
        @test eltype(R) === D
        @test ForwardDiff.value.(R) == fv_residual(U, p)
        @test all(isfinite, R)
        for dir in (1, 2)
            obj = FVWallForce(dir)
            force = fv_objective(obj, U, pd)
            @test force isa D
            @test ForwardDiff.value(force) ≈ fv_objective(obj, U, p) rtol = 1e-14
            @test isfinite(ForwardDiff.partials(force)[1])
        end
        partial = [ForwardDiff.partials(r)[1] for r in R]
        @test any(!iszero, partial)
        for j in 1:p.grid.Ny, i in 1:p.grid.Nx
            if p.tag[i, j] == FV_CELL_SOLID
                @test all(iszero, partial[fv_index(p.grid, i, j, 1):fv_index(p.grid, i, j, 3)])
            elseif p.tag[i, j] == FV_CELL_PINNED
                @test iszero(partial[fv_index(p.grid, i, j, 3)])
            end
        end
    end
end

@testset "Immersed Float32 residual" begin
    g = fv_immersed_circle(12, 12, 6.13f0, 5.87f0, 2.21f0; h = 1f0, mass=:M1)
    p = fv_steady_problem(12, 12; h = 1f0, nu = 0.1f0, immersed = g, T = Float32)
    U = randn(MersenneTwister(505), Float32, fv_nunknowns(p))
    R = fv_residual(U, p)
    @test eltype(R) === Float32
    @test all(isfinite, R)
end

# Independent constitutive evaluation at the wall gradient. Geometry and
# interpolation are shared, but the strain invariant and stress are formed
# here directly from the analytical tensor definition, not _fv_wall_stress.
function _fvc5_gnf_wall_force(U, p, dir)
    g, ib = p.grid, p.immersed
    acc = zero(eltype(U)) + zero(eltype(fv_gnf_parameters(p)))
    for j in 1:g.Ny, i in 1:g.Nx, d in 1:4
        ib.qcut[d, i, j] > 0 || continue
        _, _, _, _, pf, _, _, dun, dvn = Kraken._fv_wall_data(U, p.bvel, g,
            i, j, d, ib.qcut, ib.wvel, ib.closure)
        G = Kraken._fv_cell_grad(U, p.tag, p.bvel, g, i, j, ib.qcut, ib.wvel, ib.closure)
        oi, oj = i - (d == 1) + (d == 2), j - (d == 3) + (d == 4)
        O = Kraken._fv_cell_grad(U, p.tag, p.bvel, g, oi, oj, ib.qcut, ib.wvel, ib.closure)
        # Independent linear extrapolation of the cell gradients to the
        # nominal face; normal entries are overwritten by the ghost slope.
        G = ntuple(k -> 1.5G[k] - 0.5O[k], 6)
        sx = d in (1, 3) ? 1 : -1
        ux, uy, vx, vy = d <= 2 ? (sx * dun, G[2], sx * dvn, G[4]) :
                                  (G[1], sx * dun, G[3], sx * dvn)
        shear = sqrt(2ux^2 + 2vy^2 + (uy + vx)^2 + p.visc.eps^2)
        nuw = effective_viscosity(p.visc.law, shear)
        tx, ty = d <= 2 ? (2nuw * dun, nuw * (dvn + sx * G[2])) :
                          (nuw * (dun + sx * G[3]), 2nuw * dvn)
        acc += -g.h * (dir == 1 ? tx : ty) +
               (dir == 1 && d <= 2 || dir == 2 && d > 2 ? g.h * sx * pf : zero(pf))
    end
    return acc
end

@testset "GNF immersed force uses the residual wall constitutive law" begin
    p = _fvc5_problem(; rheology = CarreauYasuda(0.1, 0.01, 2.0, 2.0, 0.6))
    U = 0.3 .* randn(MersenneTwister(504), fv_nunknowns(p))
    θ = fv_gnf_parameters(p)
    for dir in (1, 2)
        obj = FVWallForce(dir)
        ref = _fvc5_gnf_wall_force(U, p, dir)
        @test fv_objective(obj, U, p) ≈ ref rtol = 1e-12
        gu = ForwardDiff.gradient(X -> fv_objective(obj, X, p), U)
        gr = ForwardDiff.gradient(X -> _fvc5_gnf_wall_force(X, p, dir), U)
        @test norm(gu - gr, Inf) <= 1e-12 * max(norm(gr, Inf), 1)
        gp = ForwardDiff.gradient(t -> fv_objective(obj, U, fv_with_gnf_parameters(p, t)), θ)
        gpr = ForwardDiff.gradient(t -> _fvc5_gnf_wall_force(U, fv_with_gnf_parameters(p, t), dir), θ)
        @test norm(gp - gpr, Inf) <= 1e-12 * max(norm(gpr, Inf), 1)
    end
end
