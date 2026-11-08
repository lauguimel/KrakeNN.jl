# T3 CPU Float64. Frozen before the first FV measurement, 2026-11-08:
# continuum order >=1.8 at N=32,64,128; finest max(|du|,|dT|,|dnu|)
# <=8*C/128^2, C=max continuum curvature from independent collocation.
# C and its numerical convergence are recorded with the reference CSV files.
# Derivative consistency <=1e-12 relative, Newton/Picard and force/flow
# agreement <=1e-10 relative. No acceptance constant is fitted to FV results.
# Run: sh tmp/thermal_julia.sh --project=. test/analytical/incns_fv_thermal_coupling_test.jl
using Test, Kraken, ForwardDiff, LinearAlgebra, Random, SparseArrays

struct _FVTExpLaw <: AbstractScalarLaw end
Kraken.law_ninputs(::_FVTExpLaw)=2
Kraken.law_nparams(::_FVTExpLaw)=3
@inline Kraken.law_value(::_FVTExpLaw,x::Tuple,theta::AbstractVector)=
    theta[1]*exp(-theta[2]*(x[2]-theta[3]))
const _FVT_BC=(W=:periodic,E=:periodic,S=:wall,N=:wall)
const _FVT_SIGMA=1.9127858635347974
const _FVT_FOLD=2.611023571651698
const _FVT_BOUNDS=(poiseuille=0.002227206179125562,couette=0.000496048208396679)
_fvt_zero(x,y)=0.0
_fvt_orders(e)=log2.(e[1:end-1]./e[2:end])
function _fvt_e5_eta(gamma,T,theta,shear_scale)
    eta0,lambda,beta,n0,slope,amp,Tc,width=theta
    a=exp(-beta*(T-483.15)); n=clamp(n0+slope*(T-463.15),.2,.75)
    eta0*a*(1+amp/(1+exp((T-Tc)/width)))*
        (1+(lambda*a*shear_scale*gamma)^2)^((n-1)/2)
end

function _fvt_reference(kind,N)
    path=joinpath(@__DIR__,"..","reference","fv_thermal_nahme",string(kind)*".csv")
    rows=[parse.(Float64,split(line,',')) for line in readlines(path)[2:end]]
    return [only(filter(r->r[1]==(j-.5)/N,rows))[k] for j in 1:N,k in 2:4]
end
function _fvt_channel(N;kind=:poiseuille,sigma=nothing,target=nothing,
                      fields=(FVStateTemperature(),),conv=false)
    sigma===nothing && (sigma=kind===:poiseuille ? _FVT_SIGMA : 0.8)
    visc=FVLawViscosity(_FVTExpLaw(),[.3,1.,1.],fields;eps=.02)
    return fv_steady_problem(2,N;viscosity=visc,bc=_FVT_BC,conv,
        u_mean_target=target,
        boundary_velocity=(side,x,y)->(kind===:couette && side===:N ? 1. : 0.,0.),
        source=(x,y)->(kind===:poiseuille ? .6 : 0.,0.,0.),
        energy=FVEnergy(alpha=.2,sigma=sigma,bottom=1.))
end
function _fvt_initial(p;kind=:poiseuille)
    fv_state(p,(x,y)->kind===:poiseuille ? y*(1-y) : y,
        _fvt_zero,_fvt_zero,(x,y)->1.)
end
function _fvt_dense_jacobian(p,U)
    D=ForwardDiff.Dual{Kraken.FVJacobianTag,Float64,4}
    X=D.(U);R=similar(X);J=zeros(length(U),length(U))
    for col in eachindex(U)
        X[col]=D(U[col],ForwardDiff.Partials((1.,0.,0.,0.)))
        fv_residual!(R,X,p)
        for row in eachindex(U)
            J[row,col]=ForwardDiff.partials(R[row])[1]
        end
        X[col]=D(U[col])
    end
    J
end
function _fvt_fluxes(p,U)
    g=p.grid;flux=Float64[];axis=Int[]
    visc=Kraken._fv_bind_viscosity(p.visc,U,g)
    for j in 1:g.Ny,i in 1:g.Nx
        p.tag[i,j]==FV_CELL_SOLID && continue
        gp=Kraken._fv_cell_grad(U,p.tag,p.bvel,g,i,j)
        for d in (1,3)
            kind,ii,jj=Kraken._fv_face_kind(g,p.tag,i,j,d)
            kind==Kraken._FV_F_INT || continue
            gn=Kraken._fv_cell_grad(U,p.tag,p.bvel,g,ii,jj)
            up,vp,pp=ntuple(c->U[fv_index(g,i,j,c)],3)
            un,vn,pn=ntuple(c->U[fv_index(g,ii,jj,c)],3)
            grad=d==1 ? ((un-up)/g.h,(gp[2]+gn[2])/2,(vn-vp)/g.h,(gp[4]+gn[4])/2) :
                ((gp[1]+gn[1])/2,(un-up)/g.h,(gp[3]+gn[3])/2,(vn-vp)/g.h)
            nu=Kraken._fv_face_nu(visc,p.nu[i,j],p.nu[ii,jj],grad...,i,j,ii,jj)
            push!(flux,Kraken._fv_interior_mass_flux(g.h,g.conv,up,vp,pp,un,vn,pn,
                gp,gn,nu,d==1,1)); push!(axis,d)
        end
    end
    flux,axis
end
function _fvt_picard(p,U0)
    Tfield=copy(fv_fields(p,U0)[4])
    frozen=_fvt_channel(p.grid.Ny;fields=(Tfield,))
    solver=fv_newton_solver(frozen); U=copy(U0)
    Tfield=solver.prob.visc.fields[1] # solver owns its adapted field copy
    delta=Inf; residual=Inf; it=0
    for k in 1:100
        it=k; old=copy(U); Tfield.=fv_fields(p,U)[4]
        stats=fv_newton!(solver,U;maxit=12)
        @test stats.converged
        delta=norm(U-old,Inf)
        residual=norm(fv_residual(U,p),Inf)
        delta<=1e-12 && residual<=1e-13 && break
    end
    (;U,iterations=it,delta,residual)
end

@testset "FV thermal coupling (T3, CPU Float64)" begin
    @testset "independent Nahme continuum reference" begin
        for kind in (:poiseuille,:couette)
            errors=Float64[]; components=Tuple[]
            for N in (32,64,128)
                p=_fvt_channel(N;kind); r=solve_incns_fv_newton(problem=p,
                    U0=_fvt_initial(p;kind),maxit=20)
                @test r.converged
                @test maximum(last(r.stats.equation_inf))<=1e-10
                @test last(r.stats.temperature_increment)<=1e-10*r.stats.thermal_scales.deltaT
                ref=_fvt_reference(kind,N)
                nu=.3exp.(-(r.T.-1))
                e=(maximum(abs,r.u.-transpose(ref[:,1])),
                   maximum(abs,r.T.-transpose(ref[:,2])),
                   maximum(abs,nu.-transpose(ref[:,3])))
                push!(components,e);push!(errors,maximum(e))
                @test 1-exp(-(sum(r.T[:,N÷2:N÷2+1])/4-1))>=.2
            end
            @info "T3 Nahme reference" kind errors=Tuple(errors) components=Tuple(components) orders=Tuple(_fvt_orders(errors)) finest_bound=getproperty(_FVT_BOUNDS,kind)
            @test minimum(_fvt_orders(errors))>=1.8
            @test errors[end]<=getproperty(_FVT_BOUNDS,kind)
        end
    end

    @testset "smooth Newton history and same-equation Picard" begin
        p=_fvt_channel(8); U0=_fvt_initial(p)
        r=solve_incns_fv_newton(;problem=p,U0,maxit=20)
        @test r.converged
        # Roundoff is excluded by a threshold fixed before measurement. Last
        # three resolvable residuals must exhibit the Newton quadratic law.
        resolved_indices=findall(>(1e-8),r.stats.history)
        resolved=r.stats.history[resolved_indices]
        @test length(resolved)>=3
        tail=resolved[end-2:end]
        order=log(tail[3]/tail[2])/log(tail[2]/tail[1])
        ratios=(tail[2]/tail[1]^2,tail[3]/tail[2]^2)
        @test order>=1.8
        @test .2<=ratios[2]/ratios[1]<=5.
        @test all(isone,r.stats.alpha[resolved_indices[end-2:end-1]])
        # The final iterates sit on the roundoff plateau (measured 4.8159e-14
        # twice: the T-increment stop needs one correction after it), so the
        # order is measured on resolvable residuals only.
        picard=_fvt_picard(p,U0)
        rel=norm(picard.U-r.U,Inf)/norm(r.U,Inf)
        @test picard.delta<=1e-12
        @test picard.residual<=1e-13
        @test rel<=1e-10
        @info "T3 Newton and Picard" history=Tuple(r.stats.history) resolved_last_three=Tuple(tail) order ratios strict_last_three=Tuple(strict_tail) strict_order strict_ratios picard_iterations=picard.iterations picard_delta=picard.delta picard_residual=picard.residual relative_state=rel
    end

    @testset "physical E5 normalization and thermal classical laws" begin
        theta=[2000.,.4,.035,.30,.006,1.5,465.15,2.5]
        law=E5ThermalLaw(theta;T_wall=468.15,nu_ref=.3,shear_scale=2.)
        reference_eta=_fvt_e5_eta(0.,468.15,theta,2.)
        @test law.eta_ref==reference_eta
        for (gamma,T) in ((0.,468.15),(.2,470.),(5.,490.))
            expected=.3/reference_eta*_fvt_e5_eta(gamma,T,theta,2.)
            @test law_value(law,(gamma,T),theta)≈expected rtol=2e-15
        end
        perturbed=copy(theta);perturbed[1]*=2
        @test law_value(law,(0.,468.15),perturbed)≈.6 rtol=2e-15
        @test law.eta_ref==reference_eta && law.nu_ref==.3 && law.shear_scale==2.
        f32=E5ThermalLaw(Float32.(theta);T_wall=468.15f0,nu_ref=.3f0)
        @test law_value(f32,(.2f0,470f0),Float32.(theta)) isa Float32
        models=(FVLawViscosity(law,theta,(FVStateTemperature(),);eps=0),
            FVGNFViscosity(Newtonian(.3;thermal=ArrheniusCoupling(10.,468.15)),0.;
                temperature=FVStateTemperature()),
            FVGNFViscosity(CarreauYasuda(.3,.05,1.,2.,.7;
                thermal=ArrheniusCoupling(10.,468.15)),0.;temperature=FVStateTemperature()))
        for viscosity in models
            @test_throws ArgumentError fv_steady_problem(3,4;bc=_FVT_BC,viscosity)
            p=fv_steady_problem(3,4;bc=_FVT_BC,viscosity,
                energy=FVEnergy(alpha=.2,sigma=.1,bottom=468.15))
            U=fv_state(p,(x,y)->.2sin(y),(x,y)->.05cos(x),(x,y)->.02sin(x*y),
                (x,y)->468.15+.2sin(x+y))
            s=fv_newton_solver(p);J=Matrix(fv_jacobian!(s,U));Jref=_fvt_dense_jacobian(p,U)
            @test all(isfinite,J)
            @test maximum(abs,J-Jref)<=1e-12*maximum(abs,Jref)
            @test maximum(abs,J[1:4:end,4:4:end])>0
            d=zeros(length(U));d[4:4:end].=range(-1.,1.;length=12)
            h=1e-4;fd=(fv_residual(U+h*d,p)-fv_residual(U-h*d,p))/(2h)
            @test norm(J*d-fd,Inf)<=2e-8+2e-7*norm(J*d,Inf)
            rest=fv_state(p,_fvt_zero,_fvt_zero,_fvt_zero,468.15)
            @test all(isfinite,Matrix(fv_jacobian!(s,rest)))
            @test all(isfinite,Kraken._fv_energy_scales(p).equations)
        end
        thermal=_fvt_channel(4);s=fv_newton_solver(thermal);U=_fvt_initial(thermal)
        @test_throws ArgumentError fv_adjoint_workspace(s,FVLawParameters(3))
        @test_throws ArgumentError fv_steady_gradient(s,U,[.3,1.,1.],
            FVLinearFunctional([4],[1.]);params=FVLawParameters(3))
        # A state-temperature Arrhenius model must initialize at physical Tw
        # when U0 is omitted; a zero-kelvin seed makes its viscosity infinite.
        rest_problem=fv_steady_problem(3,4;bc=_FVT_BC,viscosity=models[2],
            energy=FVEnergy(alpha=.2,sigma=.1,bottom=468.15))
        default=solve_incns_fv_newton(problem=rest_problem)
        @test default.converged
        @test all(isfinite,default.T)
        @test all(==(468.15),default.T)
        @test_throws ArgumentError fv_adjoint_workspace(default.solver,FVUniformViscosity())
    end

    @testset "T columns: odd periodic masks, mixed and reversed states" begin
        rng=MersenneTwister(38193)
        theta=[2000.,.4,.035,.30,.006,1.5,465.15,2.5]
        models=((:exponential,_FVTExpLaw(),[.3,1.,1.],1.),
                (:E5,E5ThermalLaw(theta;T_wall=468.15,nu_ref=.3),theta,468.15))
        solid=falses(5,5);solid[3,3]=true;solid[[5,1],3:4].=true
        for (name,law,parameters,Tw) in models, masked in (false,true)
            visc=FVLawViscosity(law,parameters,(FVStateTemperature(),);eps=.02)
            p=fv_steady_problem(5,5;bc=_FVT_BC,solid=masked ? solid : falses(5,5),
                viscosity=visc,energy=FVEnergy(alpha=.2,sigma=.3,bottom=Tw))
            base=.3randn(rng,fv_nunknowns(p));base[3:4:end].*=.1;base[4:4:end].+=Tw+1.
            S=Matrix(fv_jacobian_pattern(p));solver=fv_newton_solver(p)
            for direction in (1.,-1.)
                U=copy(base);U[1:4:end].*=direction;U[2:4:end].*=direction;U[3:4:end].*=direction
                flux,axis=_fvt_fluxes(p,U)
                for d in (1,3)
                    @test minimum(flux[axis.==d])<0<maximum(flux[axis.==d])
                end
                Jref=_fvt_dense_jacobian(p,U);J=Matrix(fv_jacobian!(solver,U))
                scale=maximum(abs,Jref);missing=count((.!iszero.(Jref)).&.!S)
                @test missing==0
                @test maximum(abs,J-Jref)<=1e-12*scale
                # Explicitly nonzero T columns in momentum and continuity.
                @test maximum(abs,Jref[1:4:end,4:4:end])>0
                @test maximum(abs,Jref[3:4:end,4:4:end])>0
                v=zeros(length(U));v[4:4:end].=randn(rng,length(U)÷4);v./=norm(v,Inf)
                errors=Float64[];relative=Float64[]
                for h in (1e-4,1e-5,1e-6,1e-7)
                    @test sign.(first(_fvt_fluxes(p,U+h*v)))==sign.(flux)
                    @test sign.(first(_fvt_fluxes(p,U-h*v)))==sign.(flux)
                    fd=(fv_residual(U+h*v,p)-fv_residual(U-h*v,p))/(2h)
                    push!(errors,norm(J*v-fd,Inf));push!(relative,errors[end]/max(norm(fd,Inf),eps()))
                end
                @test any(errors .<=2e-8 .+2e-7*norm(J*v,Inf))
                @info "T3 thermal Jacobian" name masked direction missing dual_difference=maximum(abs,J-Jref) fd_absolute=Tuple(errors) fd_relative=Tuple(relative)
            end
        end
    end

    @testset "fixed force and global flow border" begin
        p=_fvt_channel(16);r=solve_incns_fv_newton(problem=p,U0=_fvt_initial(p),maxit=20)
        @test r.converged
        target=sum(r.u)/length(r.u);flow=_fvt_channel(16;target)
        rflow=solve_incns_fv_newton(problem=flow,U0=_fvt_initial(flow),maxit=20)
        @test rflow.converged
        rel=norm(rflow.U[1:length(r.U)]-r.U,Inf)/norm(r.U,Inf)
        force=fv_flow_force(flow,rflow.U)
        @test rel<=1e-10
        @test abs(force-.6)/.6<=1e-10
        @test abs(sum(rflow.u)/length(rflow.u)-target)/target<=1e-10
        # Independent dual traverses every row and column, including the border.
        small=_fvt_channel(4;target)
        U=_fvt_initial(small);U[end]=.73;U[4:4:end-1].+=.13
        Jref=_fvt_dense_jacobian(small,U);J=Matrix(fv_jacobian!(fv_newton_solver(small),U))
        @test maximum(abs,J-Jref)<=1e-12*maximum(abs,Jref)
        @test count((.!iszero.(Jref)).&.!Matrix(fv_jacobian_pattern(small)))==0
        @test count(!iszero,Jref[end,:])==8
        @test Jref[end,end]==0
        @test all(Jref[1:4:end-1,end].==-small.grid.h^2)
        fixedsmall=_fvt_channel(4)
        cf,nf=fv_jacobian_coloring(fv_jacobian_pattern(fixedsmall),fixedsmall)
        cq,nq=fv_jacobian_coloring(fv_jacobian_pattern(small),small)
        @test cq[1:end-1]==cf
        @test nq==nf
        @info "T3 global flow border" target recovered_force=force relative_state=rel dual_difference=maximum(abs,J-Jref)
    end

    @testset "immersed constraints precede the global flow border" begin
        ib=fv_immersed_circle(10,10,5.,5.,2.;h=1.,closure=:quadratic,
            mass=:M3,pressure=:linear)
        p=fv_steady_problem(10,10;h=1.,immersed=ib,wall_flux=0.,nu=.3,conv=false,
            bc=_FVT_BC,source=(x,y)->(.006,0.,0.),u_mean_target=.2)
        U=.1randn(MersenneTwister(8173),fv_nunknowns(p));U[end]=.006
        @test length(U)==fv_nunknowns(p.grid)+Kraken._fv_npairs(p.immersed)+1
        Jref=_fvt_dense_jacobian(p,U);J=Matrix(fv_jacobian!(fv_newton_solver(p),U))
        @test maximum(abs,J-Jref)<=1e-12*maximum(abs,Jref)
        @test count((.!iszero.(Jref)).&.!Matrix(fv_jacobian_pattern(p)))==0
        @test J[end,end]==0.
        @test all(iszero,J[end,fv_nunknowns(p.grid)+1:end])
        @test fv_flow_force(p,U)==U[end]
        @info "T3 immersed plus flow border" unknowns=length(U) immersed_constraints=Kraken._fv_npairs(p.immersed) dual_difference=maximum(abs,J-Jref)
    end

    @testset "independent fold and heating continuation" begin
        p=_fvt_channel(32;sigma=3.)
        r=fv_thermal_continuation(p;initial_step=.25,max_halvings=5,maxit=30)
        @test !r.converged
        @test r.reason===:step_halving_limit
        @test r.last_sigma<_FVT_FOLD
        @test r.last_sigma>0
        singular=[entry.singular_min for entry in r.history]
        @test all(isfinite,singular)
        @test last(singular)<first(singular)
        @test all(diff(singular[end-min(3,length(singular)-1):end]).<0)
        @info "T3 fixed-force fold" reason=r.reason last_sigma=r.last_sigma reference_fold=_FVT_FOLD singular=Tuple(singular)
        flow=_fvt_channel(32;sigma=3.,target=1/6)
        rf=fv_thermal_continuation(flow;initial_step=.25,max_halvings=5,maxit=30)
        @test rf.converged
        @test rf.last_sigma==3.
        @info "T3 fixed-flow continuation" converged=rf.converged last_sigma=rf.last_sigma force=fv_flow_force(flow,rf.U)
    end
end
