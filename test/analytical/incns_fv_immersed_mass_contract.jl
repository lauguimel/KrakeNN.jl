# M3 mode boundaries and augmented GNF sparsity, CPU unit cases only.
# Exact equality enforces unchanged C1/domain arithmetic. The GNF matrix
# bound is the existing C2 1e-12 dense-dual parity gate.
using Test, Kraken, ForwardDiff, LinearAlgebra, Random

@testset "M3 default and exact C1/domain compatibility" begin
    @test FVImmersedClosure().mass == 0x03
    @test FVImmersedClosure(:linear).mass == 0x03
    @test FVImmersedClosure(:quadratic; mass=:M1).mass == 0x01
    @test FVImmersedClosure(:c1; mass=:M3) == FVImmersedClosure(:c1; mass=:M1)
    tag = fill(FV_CELL_FLUID, 4, 4)
    reference = fv_steady_problem(4, 4; h=1.0,
        boundary_velocity=(s,x,y)->s==:N ? (0.2,0.0) : (0.0,0.0))
    U = randn(MersenneTwister(136), fv_nunknowns(reference))
    reference_solver = fv_newton_solver(reference)
    J = copy(fv_jacobian!(reference_solver, U))
    for closure in (:c1, :linear, :quadratic)
        ib = FVImmersedBoundary2D(zeros(4,4,4), zeros(2,4,4,4), copy(tag),
            1.0, FVImmersedClosure(closure; mass=:M3))
        p = fv_steady_problem(4, 4; h=1.0, immersed=ib,
            boundary_velocity=(s,x,y)->s==:N ? (0.2,0.0) : (0.0,0.0))
        @test fv_nunknowns(p) == 48
        @test fv_residual(U, p) == fv_residual(U, reference)
        @test fv_jacobian!(fv_newton_solver(p), U) == J
    end
end

@testset "M3 rejects immersed wall contact rather than solid contact" begin
    # A vertical immersed face has endpoints on the domain's north/south
    # sides. The raw constructor bypasses analytic-circle gap checks, so the
    # M3 topology constructor itself must reject it explicitly.
    N = 6
    tag = [i >= 4 ? FV_CELL_SOLID : FV_CELL_FLUID for i in 1:N,j in 1:N]
    q = zeros(4,N,N)
    q[1,3,:] .= 0.5
    for closure in (:linear, :quadratic)
        ib = FVImmersedBoundary2D(q, zeros(2,4,N,N), tag, 1.0,
            FVImmersedClosure(closure; mass=:M3))
        caught = try
            fv_steady_problem(N,N; h=1.0, immersed=ib, wall_flux=0.0)
            nothing
        catch err
            err
        end
        @test caught isa ArgumentError
        @test occursin("domain boundary", sprint(showerror,caught))
    end
    annulus = fv_immersed_annulus(20,20,10.13,9.87,2.6,6.2; h=1.0)
    p = fv_steady_problem(20,20; h=1.0, immersed=annulus, wall_flux=0.0)
    @test all(==(FV_CELL_SOLID), p.tag[1,:])
    @test length(p.immersed.mass.Q) == 2
end

@testset "M3 GNF augmented Jacobian parity" begin
    law = CarreauYasuda(0.8, 0.1, 1.0, 2.0, 0.6)
    for closure in (:linear,:quadratic), conv in (false,true)
        ib = fv_immersed_circle(12,12,6.13,5.87,2.21; h=1.0,closure)
        p = fv_steady_problem(12,12; h=1.0, immersed=ib, wall_flux=0.0,
            rheology=law, gnf_eps=1e-2, conv)
        U = 0.2randn(MersenneTwister(137),fv_nunknowns(p))
        reference = ForwardDiff.jacobian(x->fv_residual(x,p),U)
        s = fv_newton_solver(p)
        @test norm(Matrix(fv_jacobian!(s,U))-reference,Inf) <= 1e-12*max(norm(reference,Inf),1)
    end
end
