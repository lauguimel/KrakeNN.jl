# T2 CUDA/cuDSS energy contract. Same analytical thresholds as the CPU file.
# This gate is skipped locally without CUDA; KRAKEN_REQUIRE_GPU=1 makes
# missing CUDA/CUDSS an error, so a cluster receipt cannot silently skip it.
using Test, KrakeNN, LinearAlgebra, Random
using KrakeNN.SparseArrays

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

# T4 additions: the full thermal adjoint uses the existing Newton/cuDSS
# factors, and its five viscosity slots allocate no warmed device storage.
@testset "FV thermal adjoint (T4, CUDA Float64)" begin
    if !_FVE_GPU_OK
        if _FVE_REQUIRE_GPU
            @test _FVE_GPU_OK
        else
            @test_skip "T4 thermal adjoint requires CUDA.functional()"
        end
    else
        CU=Main.CUDA;Tw=468.15
        bc=(W=:periodic,E=:periodic,S=:wall,N=:wall)
        solid=falses(4,5);solid[2,3]=true;solid[3,4]=true
        truth=[2000.,.4,.035,.30,.006,1.5,465.15,2.5]
        e5=E5ThermalLaw(truth;T_wall=Tw,nu_ref=.3,shear_scale=2.)
        learned=LearnedViscosity(TinyMLP{2,2,1}(;shift=(0.,Tw),scale=(1.,2.));
            nu_ref=.3,monotone=false)
        learnedtheta=[.3,-.2,.4,-.3,.1,-.15,.2,-.15,.04]
        for (name,law,theta) in ((:e5,e5,truth),(:learned,learned,learnedtheta)),target in (nothing,.12)
            p=fv_steady_problem(4,5;bc,solid,conv=true,u_mean_target=target,
                source=(x,y)->(1.4,.07sin(2x+3y),0.),
                energy=FVEnergy(alpha=.08,sigma=.6,bottom=Tw),
                viscosity=FVLawViscosity(law,theta,(FVStateTemperature(),);eps=.02))
            sc=fv_newton_solver(p)
            sg=fv_newton_solver(p;to=CU.CuArray,linsolve=CUDABackendTag())
            Uc=fv_state(p,(x,y)->0.,(x,y)->0.,(x,y)->0.,Tw);Ug=CU.CuArray(Uc)
            for (s,U) in ((sc,Uc),(sg,Ug))
                @test fv_newton!(s,U;maxit=20).converged
                for _ in 1:2
                    fv_newton_finalize!(s,U);fv_residual!(s.R,U,s.prob)
                    lin_solve!(s.dU,s.cache,s.R);U.-=s.dU
                end
            end
            @test norm(Array(Ug)-Uc,Inf)<=1e-10
            map=FVLawParameters(length(theta))
            wc=fv_adjoint_workspace(sc,map);wg=fv_adjoint_workspace(sg,map)
            objs=FVObjective[fv_mean_temperature(p;T_wall=Tw,target=.25),
                fv_temperature_l2(p,Tw+.25;weights=.7)]
            target===nothing || push!(objs,fv_flow_force_objective(p))
            for obj in objs
                rc=fv_steady_gradient(sc,Uc,theta,obj;params=map,ws=wc)
                rg=fv_steady_gradient(sg,Ug,theta,obj;params=map,ws=wg)
                relative=norm(rg.grad-rc.grad)/max(norm(rc.grad),eps())
                @test relative<=1e-10
                @test sg.n_factorize==1
                @test norm(transpose(_fve_gpu_host(sg.J))*Array(rg.λ)-rg.gU)/norm(rg.gU)<=1e-10
                KrakeNN._fv_param_cotangent!(wg,map,sg,Ug,theta)
                @test _fve_gpu_allocated(()->KrakeNN._fv_param_cotangent!(wg,map,sg,Ug,theta))==0
                @info "T4 GPU gradient parity" name target objective=string(typeof(obj)) relative
            end
        end
    end
end

# T3 additions retain every T2 gate below. These exercise state-T inputs in
# device residual/Jacobian traversal and the global fixed-flow border.
@testset "FV two-way energy (T3, CUDA Float64)" begin
    if !_FVE_GPU_OK
        if _FVE_REQUIRE_GPU
            @test _FVE_GPU_OK
        else
            @test_skip "T3 energy GPU contract requires CUDA.functional()"
        end
    else
        CU=Main.CUDA
        bc=(W=:periodic,E=:periodic,S=:wall,N=:wall)
        solid=falses(7,5); solid[3:4,3].=true
        truth=[2000.0,0.4,0.035,0.30,0.006,1.5,465.15,2.5]
        e5=E5ThermalLaw(truth;T_wall=468.15,nu_ref=0.3,shear_scale=2.0)
        learned=LearnedViscosity(TinyMLP{2,2,1}(;shift=(0.0,468.15),scale=(1.0,2.0));
                                 nu_ref=0.3,monotone=false)
        theta=[0.3,-0.2,0.4,-0.3,0.1,-0.15,0.2,-0.15,0.04]
        models=((:e5,FVLawViscosity(e5,truth,(FVStateTemperature(),);eps=0)),
            (:learned,FVLawViscosity(learned,theta,(FVStateTemperature(),);eps=0.02)),
            (:classical,FVGNFViscosity(Newtonian(0.3;thermal=ArrheniusCoupling(10.0,468.15)),
                                      0.0;temperature=FVStateTemperature())))
        for (model,viscosity) in models, scheme in (:linear_upwind,:upwind), target in (nothing,0.1)
            p=fv_steady_problem(7,5;solid,bc,viscosity,u_mean_target=target,
                energy=FVEnergy(;alpha=0.2,sigma=0.1,bottom=468.15,scheme))
            sc=fv_newton_solver(p)
            sg=fv_newton_solver(p;to=CU.CuArray,linsolve=CUDABackendTag())
            base=fv_state(p,(x,y)->0.2+0.03sin(3x+2y),(x,y)->0.04cos(x-y),
                            (x,y)->0.02sin(x+y),(x,y)->468.15+0.2sin(x+3y))
            for direction in (1.0,-1.0)
                U=copy(base)
                for j in 1:5,i in 1:7,k in 1:3
                    U[fv_index(p.grid,i,j,k)]*=direction
                end
                target===nothing || (U[end]=direction*0.6)
                Ug=CU.CuArray(U); Rc=fv_residual(U,p); Rg=similar(Ug)
                fv_residual!(Rg,Ug,sg.prob)
                @test norm(Array(Rg)-Rc,Inf)<=1e-12*max(norm(Rc,Inf),1.0)
                Jc=fv_jacobian!(sc,U); Jg=_fve_gpu_host(fv_jacobian!(sg,Ug))
                @test norm(Jc[1:4:140,4:4:140],Inf)>0
                @test Jc.colptr==Jg.colptr && rowvals(Jc)==rowvals(Jg)
                @test norm(nonzeros(Jg)-nonzeros(Jc),Inf)<=1e-12*max(norm(nonzeros(Jc),Inf),1.0)
                # Both complete traversals are warmed before measuring device storage.
                @test _fve_gpu_allocated(()->fv_residual!(sg.R,Ug,sg.prob))==0
                @test _fve_gpu_allocated(()->fv_jacobian!(sg,Ug))==0
            end
        end

        viscosity=FVGNFViscosity(Newtonian(0.3;thermal=ArrheniusCoupling(0.4,1.0)),
                                 0.0;temperature=FVStateTemperature())
        for target in (nothing,0.15)
            p=fv_steady_problem(2,16;bc,viscosity,u_mean_target=target,
                source=(x,y)->(0.6,0.0,0.0),energy=FVEnergy(;alpha=0.2,sigma=0.1,bottom=1.0))
            U0=fv_state(p,(x,y)->0.0,(x,y)->0.0,(x,y)->0.0,1.0)
            rc=solve_incns_fv_newton(;problem=p,U0,maxit=30)
            sg=fv_newton_solver(p;to=CU.CuArray,linsolve=CUDABackendTag())
            Ug=CU.CuArray(U0); result=fv_newton!(sg,Ug;maxit=30)
            @test rc.converged && result.converged
            @test norm(Array(Ug)-rc.U,Inf)<=1e-10
            @test maximum(last(result.equation_inf))<=1e-10
            @test last(result.temperature_increment)<=1e-10*result.thermal_scales.deltaT
            # Reuse the symbolic and numeric factorization on a nearby state.
            Ug .+= 1e-5
            @test _fve_gpu_allocated(()->fv_newton!(sg,Ug;maxit=4))==0
        end
    end
end

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
