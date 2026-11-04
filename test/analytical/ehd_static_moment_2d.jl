using Test, Kraken, KernelAbstractions, LinearAlgebra

# Issue #73 PR 1: prospective Float64 gates. Algebra tolerance 2e-14;
# manufactured residual 1e-11 and order >1.8 are frozen from the independent
# 2026-10-19 static candidate protocol. This is not the finite-alpha ES-003 gate.
@testset "Experimental static moment injector" begin
    cx = (0,1,0,-1,0,1,-1,-1,1)
    cy = (0,0,1,0,-1,1,1,-1,-1)
    w = (4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36)
    for v in (0.0, 0.0005, 0.01/sqrt(3), 0.009), qb in (0.01, 1.0, 2.0)
        original = [0.03*sin(k) for k in 1:9]
        outgoing = sum(original[k] for k in (5,8,9))
        f = [Kraken._ehd_static_injector_population(Val(k), qb, v, outgoing) for k in 1:9]
        @test abs(sum(f)-qb) <= 2e-14
        @test abs(sum(f[k] for k in (5,8,9))-outgoing) <= 2e-14
        @test abs(sum(f .* cx)) <= 2e-14
        eq = [qb*w[k]*(1+3cy[k]*v+4.5*(cy[k]*v)^2-1.5v^2) for k in 1:9]
        geq = sum(eq[k] for k in (5,8,9))
        @test maximum(abs(Kraken._ehd_static_injector_population(Val(k),qb,v,geq)-eq[k]) for k in 1:9) <= 2e-14
    end

    @testset "Collector/interior unchanged and injector kernel identities" begin
        nx,ny = 4,8
        f = reshape([0.02+0.001*sin(k) for k in 1:nx*ny*9],nx,ny,9)
        before = copy(f); legacy = copy(f)
        q = dropdims(sum(f; dims=3); dims=3)
        ex = zeros(nx,ny); ey = fill(0.003,nx,ny)
        Kraken.apply_charge_nee_walls_2d!(legacy,q,ex,ey,0.2,0.0,1.0,nx,ny)
        Kraken._apply_charge_static_moment_walls!(f,q,ex,ey,0.2,1.0,nx,ny)
        KernelAbstractions.synchronize(CPU())
        @test f[:,2:ny-1,:] == before[:,2:ny-1,:]
        @test f[:,ny,:] == legacy[:,ny,:]
        @test maximum(abs.(sum(f[:,1,:];dims=2).-0.2)) <= 2e-14
        @test maximum(abs.(sum(f[:,1,[5,8,9]];dims=2).-sum(before[:,1,[5,8,9]];dims=2))) <= 2e-14
    end

    @testset "Driver capability refusals before allocation" begin
        base = (charge_scheme=:regularized,phi_scheme=:direct,injector_closure=:static_moment)
        for change in ((charge_scheme=:srt,), (phi_scheme=:lbm,), (FT=Float32,),
                       (delta_U=2.0,), (injector_closure=:unknown,), (alpha=NaN,))
            @test_throws ArgumentError Kraken.run_ehd_hydrostatic_2d(;merge(base,change)...)
        end
        @test_throws ArgumentError Kraken.init_state(Kraken.ECState; base...)
        @test_throws ArgumentError Kraken.run_electroconvection_2d(; base...)
        fixture = joinpath(@__DIR__,"..","..","benchmarks","krk","ehd","hydrostatic_fast.krk")
        setup = Kraken.load_kraken(fixture)
        setup.physics.params[:injector_closure] = :static_moment
        setup.physics.params[:charge_scheme] = :srt
        @test_throws ArgumentError Kraken.run_simulation(setup; T=Float64)
    end

    @testset "Ny96 public-driver canary and explicit legacy parity" begin
        # One-step admission canary, deliberately not a converged physical test.
        # Converged H256/512/1024 cases run separately on Aqua with this source.
        args = (Nx=4,Ny=96,charge_scheme=:regularized,phi_scheme=:direct,
                charge_tol=Inf,max_steps=1,backend=CPU(),FT=Float64)
        default = Kraken.run_ehd_hydrostatic_2d(;args...)
        legacy = Kraken.run_ehd_hydrostatic_2d(;args...,injector_closure=:nee)
        candidate = Kraken.run_ehd_hydrostatic_2d(;args...,injector_closure=:static_moment)
        @test isequal(default,legacy)
        @test candidate.steps == 1
        @test all(isfinite,candidate.q) && minimum(candidate.q) > 0
        @test maximum(abs.(candidate.q[:,1].-candidate.params.q_inj)) <= 2e-14
        @test candidate.y_mapping == :wall_nodes
        @test candidate.analytic.y == collect(0:95)./95
        @test default.y_mapping == :wall_nodes_interior_half_link
        @test candidate.q != legacy.q
    end

    @testset "Converged public-driver all-node current regression" begin
        # PR78 review, 2026-10-23, issuecomment-6362790750: maintainer CPU
        # measurement at Nx=4, Ny=96 is 0.52% for :static_moment versus 26.6%
        # for :nee. Freeze a 1% regression ceiling BEFORE the follow-up run.
        # This guards the integrated call order, not a new profile-accuracy claim.
        result = Kraken.run_ehd_hydrostatic_2d(; Nx=4, Ny=96,
            C=10.0, M=10.0, Ma_E=0.01, alpha=1e-4, delta_U=1.0,
            charge_scheme=:regularized, phi_scheme=:direct,
            injector_closure=:static_moment, charge_tol=1e-10,
            max_steps=100000, backend=CPU(), FT=Float64)
        @test 1 < result.steps < 100000
        @test result.q_rel_change <= 1e-10
        @test all(isfinite, result.q) && minimum(result.q) > 0
        # q(y=0)=q_inj to summation roundoff, not an accuracy tolerance.
        @test all(isapprox.(result.q[:,1], result.params.q_inj;
                            rtol=32eps(Float64), atol=0.0))
        q, ey, p = result.q_profile, result.Ey_profile, result.params
        # Lattice spacing is one. Include BOTH electrodes with second-order
        # one-sided derivatives; the interior uses centred differences.
        dq = similar(q)
        dq[1] = (-3q[1] + 4q[2] - q[3])/2
        dq[2:end-1] .= (q[3:end] .- q[1:end-2])./2
        dq[end] = (3q[end] - 4q[end-1] + q[end-2])/2
        current = p.K .* q .* ey .- p.D .* dq
        mean_current = sum(current)/length(current)
        @test all(isfinite, current) && minimum(current) > 0
        variation = maximum(abs.(current .- mean_current))/abs(mean_current)
        @test variation <= 0.01
        @info "Static injector converged regression" steps=result.steps current_variation=variation charge_rel_change=result.q_rel_change
    end

    @testset "Manufactured stationary recurrence, link balance and actual kernel" begin
        errors = Float64[]
        tau = 0.5017736200269506
        for h in (16,32,64,128)
            # High-precision recurrence avoids the near-singular Float64 dense
            # (I-A) subtraction documented in the original external screen.
            q,v,flux,err = setprecision(BigFloat,200) do
                t=BigFloat(string(tau)); p=1-1/t; beta=1-1/(2t)
                dlb=(t-BigFloat(".5"))/3
                y=BigFloat.(0:h)./h
                exact=1 .- BigFloat(".4").*(y.-y.^2/2)
                grad=-BigFloat(".4").*(1 .- y)
                vv=(BigFloat(".2").+BigFloat(".05").*grad)./exact.*dlb/BigFloat(".05")/h
                em=(1 .-3vv.+3vv.^2)./6
                aa=ones(BigFloat,h+1); bb=zeros(BigFloat,h+1)
                c=(1-p)/(2beta)
                for j in 1:h
                    left=em[j]+vv[j]/(2beta)
                    right=em[j+1]+p*vv[j+1]/(2beta)
                    aa[j+1]=left/right*aa[j]
                    bb[j+1]=left/right*bb[j]-c/right
                end
                flux=(BigFloat(".8")-aa[end])/bb[end]
                qq=aa.+bb.*flux
                Float64.(qq),Float64.(vv),Float64(flux),Float64(norm(qq-exact)/norm(exact))
            end
            push!(errors,err)
            nx=3; ny=h+1; beta=1-1/(2tau)
            jne=(flux.-q.*v)./beta
            f=[q[j]*w[k]*(1+3cy[k]*v[j]+4.5*(cy[k]*v[j])^2-1.5v[j]^2)+3w[k]*cy[k]*jne[j]
               for i in 1:nx,j in 1:ny,k in 1:9]
            before=copy(f); out=similar(f); ex=zeros(nx,ny); ey=repeat(reshape(v,1,ny),nx,1)
            Kraken.collide_electric_charge_regularized_2d!(f,ex,ey,tau,1.0)
            KernelAbstractions.synchronize(CPU())
            links=[sum(f[1,j,k] for k in (3,6,7))-sum(f[1,j+1,k] for k in (5,8,9)) for j in 1:h]
            @test maximum(abs.(links.-flux)) <= 1e-11
            Kraken.stream_periodic_x_wall_y_2d!(out,f,nx,ny)
            qfield=zeros(nx,ny); Kraken.compute_ehd_scalar_2d!(qfield,out)
            Kraken._apply_charge_static_moment_walls!(out,qfield,ex,ey,1.0,1.0,nx,ny)
            KernelAbstractions.synchronize(CPU())
            # Isolation reference prescribes top q=.8; exclude the legacy
            # Neumann collector row from this manufactured fixed-point check.
            @test maximum(abs.(out[:,1:ny-1,:].-before[:,1:ny-1,:])) <= 1e-11
        end
        @test all(log2.(errors[1:end-1]./errors[2:end]) .> 1.8)
    end
end
