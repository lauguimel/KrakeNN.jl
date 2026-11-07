# C5 objective contracts. CPU unit cases only; no convergence campaign.
# Algebraic identities use 2e-11 absolute roundoff allowance for O(10^2)
# accumulated face terms. Adjoint/centred-FD gate: 1e-6 relative, 2e-7 absolute
# (C4 derivative gate, with a fixed two-step truncation/cancellation check).
using Test, KrakeNN, ForwardDiff, LinearAlgebra, Random

_c5obj_cross(a, b) = a[1] * b[2] - a[2] * b[1]
function _c5obj_interior_flux(U, p, i, j, d)
    g, h = p.grid, p.grid.h
    ex, ey = (d == 1) - (d == 2), (d == 3) - (d == 4)
    ni, nj = i + ex, j + ey
    q, w, c = KrakeNN._fv_immersed_args(p.immersed)
    a = KrakeNN._fv_cell_grad(U, p.tag, p.bvel, g, i, j, q, w, c)
    b = KrakeNN._fv_cell_grad(U, p.tag, p.bvel, g, ni, nj, q, w, c)
    u, v, pressure = ntuple(k -> U[fv_index(g, i, j, k)], 3)
    un, vn, pn = ntuple(k -> U[fv_index(g, ni, nj, k)], 3)
    isx, sx = d <= 2, ex + ey
    du, dv = (un - u) / h, (vn - v) / h
    nu = isx ? KrakeNN._fv_face_nu(p.visc, p.nu[i, j], p.nu[ni, nj],
        sx * du, (a[2] + b[2]) / 2, sx * dv, (a[4] + b[4]) / 2) :
        KrakeNN._fv_face_nu(p.visc, p.nu[i, j], p.nu[ni, nj],
        (a[1] + b[1]) / 2, sx * du, (a[3] + b[3]) / 2, sx * dv)
    speed = sx * (isx ? (u + un) / 2 : (v + vn) / 2)
    df = h^2 / (4nu + g.conv * abs(speed) * h)
    gp = sx * (isx ? (a[5] + b[5]) / 2 : (a[6] + b[6]) / 2)
    mass = h * (speed - df * ((pn - pressure) / h - gp))
    tx = isx ? 2nu * du : nu * (du + sx * (a[3] + b[3]) / 2)
    ty = isx ? nu * (dv + sx * (a[2] + b[2]) / 2) : 2nu * dv
    return (g.conv * (max(mass, 0) * u + min(mass, 0) * un) - h * tx + h * ex * (pressure + pn) / 2,
            g.conv * (max(mass, 0) * v + min(mass, 0) * vn) - h * ty + h * ey * (pressure + pn) / 2)
end

@testset "Immersed residual-flux momentum and moment identities" begin
    for conv in (false, true)
        ib = fv_immersed_annulus(20, 20, 10.13, 9.87, 2.6, 6.2; h=1.0, mass=:M1,
            wall_velocity=(k, x, y) -> (0.2x + 0.1y, -0.3x + 0.1y))
        p = fv_steady_problem(20, 20; h=1.0, immersed=ib, conv,
            source=(x, y) -> (0.1x, -0.2y, 0.0))
        U = randn(MersenneTwister(47), fv_nunknowns(p))
        R = fv_residual(U, p)
        force, moment, internal, shift = zeros(2), 0.0, 0.0, 0.0
        g, center = p.grid, (9.2, 10.3)
        for j in 1:g.Ny, i in 1:g.Nx
            p.tag[i, j] == FV_CELL_SOLID && continue
            r = ((i - 0.5) * g.h - center[1], (j - 0.5) * g.h - center[2])
            f = ntuple(k -> R[fv_index(g, i, j, k)] + g.h^2 * p.src[fv_index(g, i, j, k)], 2)
            force .+= f
            moment += _c5obj_cross(r, f)
            for d in (1, 3)
                kind, _, _ = KrakeNN._fv_face_kind(g, p.tag, i, j, d)
                kind == KrakeNN._FV_F_INT || continue
                e = d == 1 ? (-g.h, 0.0) : (0.0, -g.h)
                internal += _c5obj_cross(e, _c5obj_interior_flux(U, p, i, j, d))
            end
        end
        q, w, c = KrakeNN._fv_immersed_args(ib)
        for (i, j, d) in KrakeNN._fv_wall_faces(FVWallForce(1), p)
            f = ntuple(k -> KrakeNN._fv_wall_force_face(FVWallForce(k; kind=:residual_flux), U, p, i, j, d, q, w, c), 2)
            shift += _c5obj_cross((((d == 1) - (d == 2)) * g.h / 2,
                                   ((d == 3) - (d == 4)) * g.h / 2), f)
        end
        for k in 1:2
            @test force[k] ≈ fv_objective(FVWallForce(k; kind=:residual_flux), U, p) atol=2e-11 rtol=0
        end
        @test moment - internal + shift ≈ fv_objective(FVWallTorque(center; kind=:residual_flux), U, p) atol=2e-11 rtol=0
    end
end

@testset "Constant symmetric stress on closed immersed contours" begin
    velocity(k, x, y) = (0.2x + 0.3y, -0.4x - 0.2y)
    ib = fv_immersed_circle(12, 12, 6.13, 5.87, 2.21; h=1.0, mass=:M1, wall_velocity=velocity)
    p = fv_steady_problem(12, 12; h=1.0, immersed=ib, conv=false, nu=0.7,
                         boundary_velocity=(s, x, y) -> velocity(1, x, y))
    U = [k == 3 ? 1.2 : velocity(1, i - 0.5, j - 0.5)[k]
         for k in 1:3, i in 1:12, j in 1:12][:]
    for obj in (FVWallForce(1; kind=:residual_flux), FVWallForce(2; kind=:residual_flux),
                FVWallTorque((4.2, 4.7); kind=:residual_flux))
        @test abs(fv_objective(obj, U, p)) < 2e-11
    end
    ds = ib.surface.values[5, :, :, :]
    @test sum(ds) ≈ 2pi * 2.21 atol=1e-13
    # R10 requires midpoint arc lengths and normals at the cut. They are not
    # an exact constant-stress quadrature on an irregular contour. Keep and
    # report the discrepancy rather than renormalising weights or normals.
    surface = [fv_objective(FVWallForce(k; kind=:surface), U, p) for k in 1:2]
    torque = fv_objective(FVWallTorque((4.2, 4.7); kind=:surface), U, p)
    exact = zeros(3)
    for (i, j, d) in KrakeNN._fv_wall_faces(FVWallForce(1), p)
        x, y, nx, ny, weight = ib.surface.values[1:5, d, i, j]
        f = weight .* (1.2nx - 0.7 * (0.4nx - 0.1ny),
                       1.2ny - 0.7 * (-0.1nx - 0.4ny))
        exact[1:2] .+= f
        exact[3] += _c5obj_cross((x - 4.2, y - 4.7), f)
    end
    @test [surface; torque] ≈ exact atol=2e-11 rtol=0
    @info "C5 midpoint-arc constant-stress quadrature defect" surface_force=surface surface_torque=torque
    @test_throws ArgumentError fv_immersed_circle(12, 12, 6.13, 5.87, 2.21;
                                                 h=1.0, min_transversality=0.99)
    @test_throws ArgumentError FVWallForce(1; kind=:residual_flux, pressure=:linear)
end

function _c5obj_problem(theta; frozen_tag=nothing, rheology=nothing)
    R, cx, omega = theta
    ib = fv_immersed_circle(12, 12, cx, 5.87, R; h=1.0, mass=:M1, frozen_tag,
        wall_velocity=(k, x, y) -> (-omega * (y - 5.87), omega * (x - cx)))
    return fv_steady_problem(12, 12; h=1.0, immersed=ib, nu=(x, y) -> 0.3 + 0.01x,
        conv=false, rheology, gnf_eps=1e-2,
        boundary_velocity=(s, x, y) -> s == :N ? (0.1, 0.0) : (0.0, 0.0))
end

@testset "Immersed objective local derivatives and metadata guards" begin
    p = _c5obj_problem([2.21, 6.13, 0.2])
    U = randn(MersenneTwister(11), fv_nunknowns(p))
    for kind in (:residual_flux, :surface),
        obj in (FVWallForce(1; kind), FVWallTorque((6.13, 5.87); kind))
        value, gu = KrakeNN._fv_objective_grad_u(obj, U, p)
        ref = ForwardDiff.gradient(x -> fv_objective(obj, x, p), U)
        @test value ≈ fv_objective(obj, U, p) rtol=1e-13 atol=1e-13
        @test norm(gu - ref, Inf) <= 1e-11
        for (map, theta) in ((FVUniformViscosity(), [0.4]),
                             (FVViscosityField(), vec(copy(p.nu))))
            grad = KrakeNN._fv_objective_grad_theta(obj, map, U, p, theta)
            direct = ForwardDiff.gradient(t -> fv_objective(obj, U, fv_param_problem(map, p, t)), theta)
            @test norm(grad - direct, Inf) <= 1e-10
        end
    end
    s = fv_newton_solver(p)
    before = (copy(s.prob.nu), copy(s.prob.immersed.qcut), copy(s.prob.immersed.surface.values))
    raw = FVImmersedBoundary2D(p.immersed.qcut, p.immersed.wvel, p.immersed.tag, p.immersed.h, p.immersed.closure)
    bad = FVProblemMap((q, t) -> KrakeNN._fv_with(q; nu=fill(99.0, size(q.nu)), immersed=raw), 0)
    @test_throws ArgumentError fv_set_params!(s, bad, Float64[])
    @test before == (s.prob.nu, s.prob.immersed.qcut, s.prob.immersed.surface.values)
    data = p.immersed.surface
    invalid = KrakeNN.FVImmersedSurface(data.values, data.wall, data.given_viscosity, 0.5)
    invalid_ib = FVImmersedBoundary2D(p.immersed.qcut, p.immersed.wvel,
        p.immersed.tag, p.immersed.h, p.immersed.closure, invalid)
    bad = FVProblemMap((q, t) -> KrakeNN._fv_with(q; nu=fill(99.0, size(q.nu)), immersed=invalid_ib), 0)
    @test_throws ArgumentError fv_set_params!(s, bad, Float64[])
    @test before == (s.prob.nu, s.prob.immersed.qcut, s.prob.immersed.surface.values)
    adapted = fv_adapt(Array, p)
    @test adapted.immersed.surface.values == p.immersed.surface.values
end

@testset "Physical cut viscosity and GNF surface law" begin
    velocity(k, x, y) = (0.2x + 0.3y, -0.4x - 0.2y)
    physical(k, x, y) = 0.3 + 0.01x^2 + 0.02y^2
    law = CarreauYasuda(0.8, 0.1, 1.0, 2.0, 0.6)
    for rheology in (nothing, law)
        ib = fv_immersed_circle(12, 12, 6.13, 5.87, 2.21; h=1.0, mass=:M1,
            wall_velocity=velocity, wall_viscosity=physical)
        p = fv_steady_problem(12, 12; h=1.0, immersed=ib, nu=99.0, conv=false,
                              rheology, gnf_eps=1e-2)
        U = [k == 3 ? 1.2 : velocity(1, i - 0.5, j - 0.5)[k]
             for k in 1:3, i in 1:12, j in 1:12][:]
        exact = zeros(2)
        for (i, j, d) in KrakeNN._fv_wall_faces(FVWallForce(1), p)
            x, y, nx, ny, weight = ib.surface.values[1:5, d, i, j]
            nu = KrakeNN._fv_wall_nu(p.visc, physical(1, x, y), 0.2, 0.3, -0.4, -0.2)
            exact .+= weight .* (1.2nx - nu * (0.4nx - 0.1ny), 1.2ny - nu * (-0.1nx - 0.4ny))
        end
        for k in 1:2
            @test fv_objective(FVWallForce(k; kind=:surface), U, p) ≈ exact[k] atol=1e-12 rtol=0
        end
        if rheology !== nothing
            for kind in (:surface, :residual_flux)
                grad = KrakeNN._fv_objective_grad_theta(FVWallForce(1; kind),
                    FVUniformViscosity(), U, p, [99.0])
                @test iszero(only(grad))
            end
        end
    end
end

@testset "Immersed force and torque adjoint vs centred FD" begin
    theta = [2.21, 6.13, 0.2]
    p = _c5obj_problem(theta)
    # This unit case is host-only; production maps choose their backend explicitly.
    map = FVProblemMap((q, t) -> _c5obj_problem(t; frozen_tag=q.tag), 3)
    s = fv_newton_solver(p)
    U = zeros(fv_nunknowns(p))
    @test fv_newton!(s, U; atol=1e-12, rtol=1e-13, maxit=5).converged
    sfd = fv_newton_solver(p)
    for kind in (:residual_flux, :surface),
        obj in (FVWallForce(1; kind), FVWallTorque((6.13, 5.87); kind))
        result = fv_steady_gradient(s, U, theta, obj; params=map)
        abs_errors, rel_errors = Float64[], Float64[]
        for k in eachindex(theta)
            errors = Float64[]
            for eps in (1e-4, 2e-5)
                values = Float64[]
                for sign in (-1, 1)
                    trial = copy(theta); trial[k] += sign * eps
                    fv_set_params!(sfd, map, trial)
                    V = copy(U)
                    st = fv_newton!(sfd, V; atol=1e-12, rtol=1e-13, maxit=5)
                    @test st.converged || st.stagnated
                    @test last(st.inf) < 2e-11
                    push!(values, fv_objective(obj, V, sfd.prob))
                end
                push!(errors, abs((values[2] - values[1]) / (2eps) - result.grad[k]))
            end
            best = minimum(errors)
            push!(abs_errors, best)
            push!(rel_errors, best / max(abs(result.grad[k]), 1e-12))
            @test best <= 2e-7 + 1e-6 * abs(result.grad[k])
        end
        @info "C5 objective adjoint vs centred FD" kind objective=typeof(obj) gradient=Tuple(result.grad) abs_errors=Tuple(abs_errors) rel_errors=Tuple(rel_errors)
    end
end
