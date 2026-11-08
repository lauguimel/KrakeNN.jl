# C5-M3 mandatory gates 1 and 2, before any nonlinear solve.
# Identity bound: 2048eps times the sum of absolute flux contributions.
# Jacobian bound: 1e-12 times the dense ForwardDiff reference norm (C2 gate).
# Rank bound: n*eps*sigma_max, the standard backward-error numerical rank.
include(joinpath(@__DIR__,"incns_fv_immersed_m3_helpers.jl"))

let evidence = NamedTuple[]
    for shape in (:circle,:annulus),closure in (:linear,:quadratic),conv in (false,true)
        @testset "M3 arbitrary-state flux identity $shape $closure conv=$conv" begin
            p = _c5m3_problem(shape;closure,conv)
            g,m = p.grid,p.immersed.mass
            ncell = 3g.Nx*g.Ny
            U = 0.2randn(MersenneTwister(831),fv_nunknowns(p))
            raw,absolute = _c5m3_raw_flux(U,p)
            corrected = raw-m.A.*U[ncell+1:end]
            R = fv_residual(U,p)
            restored = fv_residual(U,_c5m3_restore(p))
            pins = [fv_index(g,Tuple(c)...,3) for c in findall(==(FV_CELL_PINNED),p.tag)]
            others = [fv_index(g,Tuple(c)...,3) for c in findall(==(FV_CELL_FLUID),p.tag)]
            Qext = _c5m3_external_flux(p)
            lhs = sum(restored[pins])
            rhs = sum(R[ncell+1:end])+sum(m.Q)+Qext-sum(R[others])
            scale = max(absolute,sum(abs,R[others]),sum(abs,R[ncell+1:end]),1)
            @test R[ncell+1:end] ≈ corrected-m.Q atol=2048eps()*scale rtol=0
            @test lhs ≈ rhs atol=2048eps()*scale rtol=0
            # Satisfy pair rows algebraically without solving momentum/continuity.
            U[ncell+1:end] .= (raw-m.Q)./m.A
            Rzero = fv_residual(U,p)
            corrected_zero = raw-m.A.*U[ncell+1:end]
            @test norm(Rzero[ncell+1:end],Inf) <= 2048eps()*scale
            @test sum(corrected_zero) ≈ sum(m.Q) atol=2048eps()*scale rtol=0
            push!(evidence,(;shape,closure,conv,pairs=length(m.Q),
                pin_continuity=lhs,identity_rhs=rhs,identity_error=abs(lhs-rhs),
                prescribed_flux=sum(m.Q),corrected_flux=sum(corrected_zero),
                pair_residual=norm(Rzero[ncell+1:end],Inf)))
        end
    end
    @testset "M3 prescribed nonzero physical flux signs" begin
        p = _c5m3_problem(:annulus;radial=true)
        m = p.immersed.mass
        @test sort(m.Q) ≈ [-2pi*0.17,2pi*0.17] atol=2eps() rtol=0
        U = randn(MersenneTwister(832),fv_nunknowns(p))
        raw,absolute = _c5m3_raw_flux(U,p)
        ncell = 3p.grid.Nx*p.grid.Ny
        U[ncell+1:end] .= (raw-m.Q)./m.A
        @test norm(fv_residual(U,p)[ncell+1:end],Inf) <= 2048eps()*max(absolute,1)
        @test raw-m.A.*U[ncell+1:end] ≈ m.Q atol=2048eps()*max(absolute,1) rtol=0
    end
    _c5m3_csv("closed_mass_m3_identity",evidence)
end

@testset "M3 one solid body separates two fluid regions" begin
    N = 18
    tag = [5 <= i <= 14 && 5 <= j <= 14 && !(8 <= i <= 11 && 8 <= j <= 11) ?
           FV_CELL_SOLID : FV_CELL_FLUID for i in 1:N,j in 1:N]
    qcut,wvel = zeros(4,N,N),zeros(2,4,N,N)
    for j in 1:N,i in 1:N,d in 1:4
        tag[i,j] == FV_CELL_SOLID && continue
        ni,nj = i+(d==1)-(d==2),j+(d==3)-(d==4)
        1 <= ni <= N && 1 <= nj <= N || continue
        tag[ni,nj] == FV_CELL_SOLID && (qcut[d,i,j] = 0.5)
    end
    ib = FVImmersedBoundary2D(qcut,wvel,tag,1.0,FVImmersedClosure(:quadratic;mass=:M3))
    for outlet in (false,true)
        p = fv_steady_problem(N,N;h=1.0,immersed=ib,wall_flux=0.0,conv=false,
            bc=(W=:wall,E=outlet ? :outlet : :wall,S=:wall,N=:wall))
        m = p.immersed.mass
        @test length(m.Q) == 2
        @test length(unique(m.body)) == 1
        @test length(unique(m.region)) == 2
        @test count(==(FV_CELL_PINNED),p.tag) == (outlet ? 1 : 2)
        @test count(identity,m.region_open) == (outlet ? 1 : 0)
    end
end

let evidence = NamedTuple[]
    for shape in (:circle,:annulus),conv in (false,true)
        @testset "M3 colored/direct Jacobian and pinned rank $shape conv=$conv" begin
            p = _c5m3_problem(shape;conv)
            U = 0.2randn(MersenneTwister(833),fv_nunknowns(p))
            dense = ForwardDiff.jacobian(x->fv_residual(x,p),U)
            s = fv_newton_solver(p)
            historical = fv_newton_solver(_c5m3_problem(shape;conv,mass=:M1))
            actual = Matrix(fv_jacobian!(s,U))
            err = norm(actual-dense,Inf)
            @test err <= 1e-12*max(norm(dense,Inf),1)
            m,ncell = p.immersed.mass,3p.grid.Nx*p.grid.Ny
            @test norm(actual[ncell+1:end,:]-dense[ncell+1:end,:],Inf) <= 1e-12*max(norm(dense,Inf),1)
            active = [fv_index(p.grid,Tuple(c)...,k) for c in findall(!=(FV_CELL_SOLID),p.tag) for k in 1:3]
            append!(active,ncell+1:fv_nunknowns(p))
            # Removed solid rows are independent identities. Their columns may
            # enter fluid rows; this block triangular elimination preserves rank.
            sigma = svdvals(dense[active,active])
            threshold = length(active)*eps()*first(sigma)
            numerical_rank = count(>(threshold),sigma)
            full_rank = numerical_rank == length(active)
            push!(evidence,(;shape,conv,pairs=length(m.Q),unknowns=length(U),
                active_unknowns=length(active),colors=s.asm.ncolors,pattern_nnz=nnz(s.J),
                m1_unknowns=fv_nunknowns(historical.prob),m1_colors=historical.asm.ncolors,
                m1_pattern_nnz=nnz(historical.J),jacobian_error=err,
                sigma_max=first(sigma),sigma_min=last(sigma),rank_threshold=threshold,
                numerical_rank,full_rank))
            _c5m3_csv("closed_mass_m3_jacobian",evidence)
            @info "M3 pinned Jacobian rank" shape conv colors=s.asm.ncolors numerical_rank size=length(active) sigma_min=last(sigma) sigma_max=first(sigma)
            @test full_rank
            full_rank || error("M3 STOP: singular pinned augmented Jacobian; no adjoint or solve gates may run")
        end
    end
end

@testset "M3 physical target alone has an exact negative unit derivative" begin
    p = _c5m3_problem(:annulus)
    U = randn(MersenneTwister(835),fv_nunknowns(p))
    ncell = 3p.grid.Nx*p.grid.Ny
    derivative = ForwardDiff.jacobian([0.0]) do target
        q = fv_steady_problem(p.grid.Nx,p.grid.Ny;h=p.grid.h,immersed=p.immersed,
            conv=false,wall_flux=only(target))
        fv_residual(U,q)
    end
    @test all(iszero,derivative[1:ncell,:])
    @test all(==(-1),derivative[ncell+1:end,:])
end

@testset "M3 all appended lambda entries enter norms and state signature" begin
    circles = ((7.13,7.87,2.21),(18.13,7.87,2.21),(12.13,18.87,2.21))
    ib = fv_immersed_boundary(27,27;h=1.0,circles,mass=:M3)
    p = fv_steady_problem(27,27;h=1.0,immersed=ib,conv=false,wall_flux=0.0)
    @test length(p.immersed.mass.Q) == 3
    solver = fv_newton_solver(p)
    ncell = 3p.grid.Nx*p.grid.Ny
    U = zeros(fv_nunknowns(p))
    before = KrakeNN._fv_state_sig(solver.norms,0.0,U)
    for pair in 1:3
        fill!(U,0)
        U[ncell+pair] = 2.0+pair
        mom,cont,inf = KrakeNN._fv_norms(solver.norms,U)
        @test mom == 0
        @test cont == 2.0+pair
        @test inf == 2.0+pair
        @test KrakeNN._fv_state_sig(solver.norms,0.0,U) != before
    end
end
