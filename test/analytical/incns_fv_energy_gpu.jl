# T2 CUDA/cuDSS energy contract. Same analytical thresholds as the CPU file.
# This gate is skipped locally without CUDA; KRAKEN_REQUIRE_GPU=1 makes
# missing CUDA/CUDSS an error, so a cluster receipt cannot silently skip it.
using Test, Kraken, LinearAlgebra, Random, SparseArrays

const _FVE_GPU_OK=(try
    @eval Main using CUDA, CUDSS
    @eval Main CUDA.functional()
catch
    false
end)===true
const _FVE_REQUIRE_GPU=get(ENV,"KRAKEN_REQUIRE_GPU","0") in ("1","true")
if _FVE_GPU_OK
    @eval _fve_gpu_allocated(f)=Main.CUDA.@allocated f()
end
_fve_gpu_host(J)=sparse(transpose(SparseMatrixCSC(size(J)...,
    Array(J.rowPtr),Array(J.colVal),Array(J.nzVal))))

@testset "FV one-way energy (T2, CUDA Float64)" begin
    if !_FVE_GPU_OK
        if _FVE_REQUIRE_GPU
            @error "KRAKEN_REQUIRE_GPU=1 but CUDA/CUDSS is not functional"
            @test _FVE_GPU_OK
        else
            @info "Skipping T2 energy GPU tests: CUDA/CUDSS not functional"
            @test_skip "T2 energy GPU contract requires CUDA.functional()"
        end
    else
        CU=Main.CUDA; rng=MersenneTwister(22091)
        bc=(W=:periodic,E=:periodic,S=:wall,N=:wall)
        solid=falses(9,7); solid[4:5,3:4].=true; solid[[9,1],4:5].=true
        law=LearnedViscosity(TinyMLP{2,2,1}();nu_ref=0.3,monotone=false)
        theta=[0.3,-0.2,0.4,-0.3,0.1,-0.15,0.2,-0.15,0.04]
        field=[0.13i+0.09j+0.03sin(i*j) for i in 1:9,j in 1:7]
        models=((:newtonian,(;nu=0.3)),
            (:gnf,(;rheology=CarreauYasuda(0.3,0.05,1.2,2.0,0.7),gnf_eps=0.02)),
            (:law,(;viscosity=FVLawViscosity(law,theta,(field,);eps=0.02))))
        for (model,options) in models, scheme in (:linear_upwind,:upwind)
            p=fv_steady_problem(9,7;solid,bc,options...,
                energy=FVEnergy(;alpha=0.2,sigma=0.4,bottom=1.0,scheme))
            sc=fv_newton_solver(p)
            sg=fv_newton_solver(p;to=CU.CuArray,linsolve=CUDABackendTag())
            base=0.3randn(rng,fv_nunknowns(p)); base[3:4:end].*=0.1
            base[4:4:end].+=1.0
            for direction in (1.0,-1.0)
                U=copy(base)
                U[1:4:end].*=direction; U[2:4:end].*=direction
                U[3:4:end].*=direction; Ug=CU.CuArray(U)
                Rc=fv_residual(U,p); Rg=similar(Ug)
                fv_residual!(Rg,Ug,sg.prob)
                residual_error=norm(Array(Rg)-Rc,Inf)/max(norm(Rc,Inf),eps())
                @test residual_error<=1e-12
                Jc=fv_jacobian!(sc,U); Jg=_fve_gpu_host(fv_jacobian!(sg,Ug))
                @test Jc.colptr==Jg.colptr && rowvals(Jc)==rowvals(Jg)
                jacobian_error=norm(nonzeros(Jg)-nonzeros(Jc),Inf)/max(norm(nonzeros(Jc),Inf),eps())
                @test jacobian_error<=1e-12
                @info "T2 GPU thermal parity" model scheme direction residual_error jacobian_error
            end
        end

        for kind in (:couette,:poiseuille)
            errors=Float64[]
            for N in (32,64,128)
                nu,alpha,sigma,Tw,force=0.3,0.2,0.4,1.0,0.6
                speed=kind===:couette ? 1.0 : 0.0
                forcing=kind===:poiseuille ? force : 0.0
                p=fv_steady_problem(2,N;nu,bc,
                    boundary_velocity=(side,x,y)->(side===:N ? speed : 0.0,0.0),
                    source=(x,y)->(forcing,0.0,0.0),
                    energy=FVEnergy(;alpha,sigma,bottom=Tw))
                U0=fv_state(p,(x,y)->0.0,(x,y)->0.0,(x,y)->0.0,(x,y)->Tw)
                rc=solve_incns_fv_newton(;problem=p,U0,atol=1e-12,rtol=0.0,maxit=12)
                sg=fv_newton_solver(p;to=CU.CuArray,linsolve=CUDABackendTag())
                U=CU.CuArray(U0); st=fv_newton!(sg,U;atol=1e-12,rtol=0.0,maxit=12)
                @test rc.converged && st.converged
                @test norm(Array(U)-rc.U,Inf)<=1e-10
                @test maximum(last(st.equation_inf))<=1e-10
                @test last(st.temperature_increment)<=1e-10*st.thermal_scales.deltaT
                T=fv_fields(p,U)[4]
                q=kind===:couette ? sigma*nu : sigma*force^2/(4nu)
                exact=[Tw+q/alpha*(kind===:couette ? y-y^2/2 :
                    y/3-y^2/2+2y^3/3-y^4/3) for i in 1:2,
                    y in ((j-0.5)/N for j in 1:N)]
                push!(errors,maximum(abs,T-exact))
                if N==128
                    bound=(kind===:couette ? 0.25 : 1.0)*q/(alpha*N^2)
                    @test errors[end]<=bound
                end
                if N==32
                    # Warm all paths and retain factorization/work buffers.
                    fv_jacobian!(sg,U); lin_refactorize!(sg.cache)
                    lin_solve!(sg.dU,sg.cache,sg.R); CU.synchronize()
                    @test _fve_gpu_allocated(()->fv_jacobian!(sg,U))==0
                    @test _fve_gpu_allocated(()->lin_refactorize!(sg.cache))==0
                    @test _fve_gpu_allocated(()->(lin_solve!(sg.dU,sg.cache,sg.R);CU.synchronize()))==0
                    @test _fve_gpu_allocated(()->fv_residual!(sg.R,U,sg.prob))==0
                    U .+= 1e-4
                    @test _fve_gpu_allocated(()->fv_newton!(sg,U;maxit=4))==0

                    # Frozen-flow solve parity and its complete warmed hot loop.
                    energy_state=copy(rc.U); energy_state[4:4:end].=Tw
                    ec=fv_energy_solve(p,energy_state)
                    eg=fv_energy_solve(p,energy_state;to=CU.CuArray,linsolve=CUDABackendTag())
                    @test ec.converged && eg.converged
                    @test norm(eg.T-ec.T,Inf)<=1e-10
                    temperature=CU.CuArray(vec(eg.T)); temperature .+= 1e-4
                    @test _fve_gpu_allocated(()->fv_energy_solve!(eg.solver,temperature))==0
                end
            end
            @test minimum(log2.(errors[1:2]./errors[2:3]))>=1.8
            @info "T2 GPU coupled heating" kind errors=Tuple(errors)
        end
    end
end
