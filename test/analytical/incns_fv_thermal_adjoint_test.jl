# T4 CPU Float64 discrete derivative contract, frozen 2026-11-08:
# contraction/parameter dual relative <= 1e-12; adjoint residual <= 1e-10;
# central full-resolve FD best relative <= 1e-6 and absolute <= 1e-9 +
# 1e-6*|derivative|. These check differentiation, not continuum accuracy.
# R_theta holds all of U=(u,v,p,T[,Fx]) and E5 normalization fixed.
using Test, Kraken, ForwardDiff, LinearAlgebra, Random

const _FVT4_BC=(W=:periodic,E=:periodic,S=:wall,N=:wall)
const _FVT4_TW=468.15
const _FVT4_STEPS=(1e-3,3e-4,1e-4)
_fvt4_rel(a,b)=norm(a-b)/max(norm(a),norm(b),eps(Float64))
function _fvt4_model(kind)
    if kind===:e5
        theta=[2000.,.4,.035,.30,.006,1.5,465.15,2.5]
        return E5ThermalLaw(theta;T_wall=_FVT4_TW,nu_ref=.3,shear_scale=2.),theta
    end
    law=LearnedViscosity(TinyMLP{2,2,1}(;shift=(0.,_FVT4_TW),scale=(1.,2.));
        nu_ref=.3,monotone=false)
    return law,[.3,-.2,.4,-.3,.1,-.15,.2,-.15,.04]
end
function _fvt4_problem(law,theta;target=nothing,sigma=.6,thermal=true,fields=(FVStateTemperature(),))
    solid=falses(4,5);solid[2,3]=true;solid[3,4]=true
    energy=thermal ? FVEnergy(alpha=.08,sigma=sigma,bottom=_FVT4_TW) : nothing
    fv_steady_problem(4,5;bc=_FVT4_BC,solid,conv=true,u_mean_target=target,
        source=(x,y)->(1.4,.07sin(2x+3y),0.),energy,
        viscosity=FVLawViscosity(law,theta,fields;eps=.02))
end
_fvt4_initial(p)=fv_state(p,(x,y)->0.,(x,y)->0.,(x,y)->0.,_FVT4_TW)
_fvt4_tindices(p)=[fv_index(p.grid,i,j,4) for j in 1:p.grid.Ny for i in 1:p.grid.Nx
    if p.tag[i,j]!=FV_CELL_SOLID]

# Observe each unique interior/periodic face, including every energy upwind
# decision. Boundary fluxes are prescribed zero and cannot switch with theta.
function _fvt4_fluxes(p,U)
    g=p.grid;flux=Float64[]
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
                gp,gn,nu,d==1,1))
        end
    end
    flux
end

# Thermal stopping is the production 1e-10 scaled gate; atol does not change
# it. Two explicit Newton corrections resolve the much smaller FD signals.
# The final remaining correction is measured, never assumed from atol.
function _fvt4_solve!(s,U)
    stats=fv_newton!(s,U;maxit=20)
    @test stats.converged
    for _ in 1:2
        fv_newton_finalize!(s,U)
        fv_residual!(s.R,U,s.prob)
        lin_solve!(s.dU,s.cache,s.R)
        U.-=s.dU
    end
    fv_newton_finalize!(s,U)
    fv_residual!(s.R,U,s.prob)
    lin_solve!(s.dU,s.cache,s.R)
    return copy(s.dU),norm(s.R,Inf)
end

function _fvt4_fd(sfd,map,theta,U,objectives,gradients,direction;label)
    baseflux=_fvt4_fluxes(sfd.prob,U)
    exact=[dot(g,direction) for g in gradients]
    absolute=zeros(length(objectives),length(_FVT4_STEPS));relative=similar(absolute)
    signals=similar(absolute);noise=similar(absolute);switches=Int[];residuals=Float64[]
    for (k,h) in enumerate(_FVT4_STEPS)
        values=zeros(length(objectives),2);errors=similar(values)
        for (side,sgn) in enumerate((-1.,1.))
            fv_set_params!(sfd,map,theta.+sgn*h.*direction)
            V=copy(U);correction,residual=_fvt4_solve!(sfd,V)
            push!(residuals,residual)
            push!(switches,count(sign.(_fvt4_fluxes(sfd.prob,V)).!=sign.(baseflux)))
            for (j,obj) in enumerate(objectives)
                values[j,side]=fv_objective(obj,V,sfd.prob)
                errors[j,side]=abs(fv_objective(obj,V-correction,sfd.prob)-values[j,side])
            end
        end
        for j in eachindex(objectives)
            fd=(values[j,2]-values[j,1])/(2h)
            absolute[j,k]=abs(fd-exact[j])
            relative[j,k]=absolute[j,k]/max(abs(exact[j]),eps(Float64))
            signals[j,k]=abs(values[j,2]-values[j,1])
            noise[j,k]=sum(errors[j,:])/max(signals[j,k],eps(Float64))
        end
    end
    fv_set_params!(sfd,map,theta)
    @test all(iszero,switches)
    for j in eachindex(objectives)
        best=argmin(view(relative,j,:))
        @test relative[j,best]<=1e-6
        @test absolute[j,best]<=1e-9+1e-6*abs(exact[j])
        @test noise[j,best]<=1e-4
        @info "T4 full-resolve FD" label objective=j exact=exact[j] steps=_FVT4_STEPS absolute=Tuple(absolute[j,:]) relative=Tuple(relative[j,:]) signal=Tuple(signals[j,:]) correction_to_signal=Tuple(noise[j,:]) switches=Tuple(switches) residual_max=maximum(residuals) minimum_flux=minimum(abs,baseflux)
    end
end

@testset "FV coupled thermal adjoint (T4, CPU Float64)" begin
    @testset "objective values and fluid-only support" begin
        law,theta=_fvt4_model(:e5);p=_fvt4_problem(law,theta;target=.1)
        U=fv_state(p,(x,y)->x,(x,y)->y,(x,y)->x*y,(x,y)->_FVT4_TW+x+2y)
        idx=_fvt4_tindices(p);T=fv_fields(p,U)[4];fluid=p.tag.!=FV_CELL_SOLID
        weights=[.1i+.2j for i in 1:4,j in 1:5];weights[1,1]=0.
        target=[_FVT4_TW+.03i-.02j for i in 1:4,j in 1:5]
        meanobj=fv_mean_temperature(p;T_wall=_FVT4_TW,target=.2)
        @test fv_objective(meanobj,U,p)≈.5*(sum(T[fluid].-_FVT4_TW)/length(idx)-.2)^2 atol=1e-12
        @test sort(fv_objective_support(meanobj,p))==sort(idx)
        l2=fv_temperature_l2(p,target;weights)
        @test fv_objective(l2,U,p)≈.5sum(weights[fluid].*(T[fluid].-target[fluid]).^2) atol=1e-12
        @test all(in(idx),fv_objective_support(l2,p))
        @test fv_objective(fv_temperature_l2(p,_FVT4_TW;weights=2),U,p)≈sum(abs2,T[fluid].-_FVT4_TW)
        U[end]=1.7;force=fv_flow_force_objective(p)
        @test fv_objective(force,U,p)==1.7
        @test fv_objective_support(force,p)==[length(U)]
        @test_throws ArgumentError fv_flow_force_objective(_fvt4_problem(law,theta))
        @test_throws ArgumentError fv_temperature_l2(p,target;weights=-1)
        @test_throws DimensionMismatch fv_temperature_l2(p,zeros(2,2))
    end

    @testset "fixed-state heating and energy-transport cotangents" begin
        for kind in (:e5,:learned),target in (nothing,.12)
            law,theta=_fvt4_model(kind);p=_fvt4_problem(law,theta;target)
            s=fv_newton_solver(p);map=FVLawParameters(length(theta));ws=fv_adjoint_workspace(s,map)
            rng=MersenneTwister(8403);U=.15randn(rng,fv_nunknowns(p))
            U[_fvt4_tindices(p)].+=1. # raw state is theta=T-T_ref (T_ref=T_wall): T=T_wall+1
            bound=@inferred Kraken._fv_bind_viscosity(p.visc,U,p.grid)
            @test (@inferred Kraken._fv_visc_eltype(bound))===Float64
            capture=@inferred Kraken._FVLawCapture(law,zeros(20,5,2),1,1,Val(5))
            captured=FVLawViscosity(capture,theta,bound.fields,bound.eps)
            @test (@inferred Kraken._fv_visc_eltype(captured))===ForwardDiff.Dual{Kraken.FVParamTag,Float64,5}
            @test (@inferred Kraken._fv_heating_viscosity(captured)).law.face==5
            @test isfinite(@inferred Kraken._fv_gnf_nu(captured,.1,.2,.3,.4,(_FVT4_TW,)))
            lambda=randn(rng,length(U));copyto!(ws.λ,lambda)
            a=Kraken._fv_param_cotangent!(ws,map,s,U,theta)
            b=Kraken._fv_param_cotangent!(ws,map,s,U,theta;law_gradient=:duals)
            @test _fvt4_rel(a,b)<=1e-12
            # Retain only thermal cotangents; switch off heating to expose the
            # viscosity-dependent Rhie--Chow transport contribution separately.
            lambda.=0.;lambda[_fvt4_tindices(p)].=randn(rng,length(_fvt4_tindices(p)))
            copyto!(ws.λ,lambda)
            allenergy=Kraken._fv_param_cotangent!(ws,map,s,U,theta)
            p0=_fvt4_problem(law,theta;target,sigma=0.);s0=fv_newton_solver(p0)
            w0=fv_adjoint_workspace(s0,map);copyto!(w0.λ,lambda)
            transport=Kraken._fv_param_cotangent!(w0,map,s0,U,theta)
            transportdual=Kraken._fv_param_cotangent!(w0,map,s0,U,theta;law_gradient=:duals)
            @test _fvt4_rel(transport,transportdual)<=1e-12
            @test norm(transport)>1e-6
            @test norm(allenergy-transport)>1e-6
            # Independent fixed-U dual of cell heating, without energy transport.
            function heating(t)
                q=fv_param_problem(map,p,t)
                sum(lambda[fv_index(p.grid,i,j,4)]*Kraken._fv_energy_heating(U,q.tag,q.nu,q.bvel,q.grid,i,j,q.visc,q.energy.sigma)
                    for j in 1:5,i in 1:4)
            end
            heatingdual=ForwardDiff.gradient(heating,theta)
            @test _fvt4_rel(allenergy-transport,heatingdual)<=1e-12
            @info "T4 fixed-state contributions" kind target contraction_relative=_fvt4_rel(a,b) transport_norm=norm(transport) heating_norm=norm(heatingdual) heating_relative=_fvt4_rel(allenergy-transport,heatingdual)
            # Existing residual-wall force must differentiate its state T input.
            obj=FVWallForce(1;kind=:residual_flux)
            _,g=Kraken._fv_objective_grad_u(obj,U,p)
            d=zeros(length(U));d[_fvt4_tindices(p)].=randn(rng,length(_fvt4_tindices(p)))
            h=1e-4;fd=(fv_objective(obj,U+h*d,p)-fv_objective(obj,U-h*d,p))/(2h)
            @test abs(dot(g,d))>1e-8
            @test abs(fd-dot(g,d))<=1e-9+1e-6*abs(dot(g,d))
        end
    end

    @testset "explicit upwind switching point, both one-sided derivatives" begin
        # h=1, nuf=.25, ubn=1, conv=1, dp=2 gives F=0 exactly.
        # The T reconstructions are unequal, so no unique derivative exists.
        gradient=ntuple(_->0.,6)
        flux(nu)=Kraken._fv_interior_mass_flux(1.,1.,1.,0.,0.,1.,0.,2.,gradient,gradient,nu,true,1)
        transport(nu)=Kraken._fv_posp(flux(nu))*2.0 + Kraken._fv_negp(flux(nu))*5.0
        @test flux(.25)==0.
        expected=(left=10.,right=4.) # dF/dnu=2 at the switching point.
        errors=Float64[]
        for h in (1e-4,1e-5,1e-6)
            left=(transport(.25)-transport(.25-h))/h
            right=(transport(.25+h)-transport(.25))/h
            @test flux(.25-h)<0<flux(.25+h)
            push!(errors,max(abs(left-expected.left),abs(right-expected.right)))
            @info "T4 one-sided switch" h left right expected
        end
        @test last(errors)<=3e-5
        @test errors[3]<errors[2]<errors[1]
    end

    @testset "coupled full-resolve parameter derivatives" begin
        for kind in (:e5,:learned),target in (nothing,.12)
            law,theta=_fvt4_model(kind);p=_fvt4_problem(law,theta;target)
            s=fv_newton_solver(p);U=_fvt4_initial(p);_fvt4_solve!(s,U)
            map=FVLawParameters(length(theta));ws=fv_adjoint_workspace(s,map)
            F=s.cache.factor;sym=F.symbolic.p;nf=s.n_factorize
            objectives=FVObjective[fv_mean_temperature(p;T_wall=_FVT4_TW,target=.25),
                fv_temperature_l2(p,_FVT4_TW+.25;weights=.7)]
            target===nothing || push!(objectives,fv_flow_force_objective(p))
            gradients=Vector{Float64}[]
            for obj in objectives
                r=fv_steady_gradient(s,U,theta,obj;params=map,ws)
                nr=s.n_refactorize
                d=fv_steady_gradient(s,U,theta,obj;params=map,ws,law_gradient=:duals,finalize=false)
                @test _fvt4_rel(r.grad,d.grad)<=1e-12
                @test norm(transpose(s.J)*r.λ-r.gU)/max(norm(r.gU),eps())<=1e-10
                @test s.n_refactorize==nr
                @test s.n_factorize==nf==1 && s.cache.factor===F && F.symbolic.p==sym
                push!(gradients,r.grad)
                @info "T4 gradient" kind target objective=string(typeof(obj)) gradient=Tuple(r.grad) contraction_relative=_fvt4_rel(r.grad,d.grad) adjoint_relative=norm(transpose(s.J)*r.λ-r.gU)/norm(r.gU) residual=r.residual_inf
            end
            sfd=fv_newton_solver(p)
            if kind===:e5
                scales=[2000.,.4,.035,.30,.006,1.5,2.5,2.5]
                for k in 1:8
                    direction=zeros(8);direction[k]=scales[k]
                    _fvt4_fd(sfd,map,theta,U,objectives,gradients,direction;label=(kind,target,k))
                end
                @test sfd.prob.visc.law.eta_ref==law.eta_ref
                @test sfd.prob.visc.law.nu_ref==law.nu_ref && sfd.prob.visc.law.shear_scale==law.shear_scale
            end
            rng=MersenneTwister(4018)
            for k in 1:2
                direction=normalize(randn(rng,length(theta)))
                kind===:e5 && (direction.*=[2000.,.4,.035,.30,.006,1.5,2.5,2.5])
                _fvt4_fd(sfd,map,theta,U,objectives,gradients,direction;label=(kind,target,:random,k))
            end
            @test sfd.n_factorize==1
            if kind===:e5
                wall=FVWallForce(1;region=:S,kind=:residual_flux)
                rw=fv_steady_gradient(s,U,theta,wall;params=map,ws)
                dense=ForwardDiff.gradient(V->fv_objective(wall,V,p),U)
                @test _fvt4_rel(rw.gU,dense)<=1e-12
                @test norm(rw.gU[_fvt4_tindices(p)])>1e-8
                direction=zeros(8);direction[3]=.035
                _fvt4_fd(sfd,map,theta,U,[wall],[rw.grad],direction;label=(kind,target,:wall))
            end
        end
    end

    @testset "zero heating and temperature-independent isothermal reduction" begin
        law=ClassicalLaw(CarreauYasuda(.3,.05,1.2,2.,.7));theta=law_init(law)
        for target in (nothing,.12)
            pc=_fvt4_problem(law,theta;target,sigma=0.,fields=())
            pi=_fvt4_problem(law,theta;target,thermal=false,fields=())
            sc=fv_newton_solver(pc);si=fv_newton_solver(pi)
            Uc=_fvt4_initial(pc);Ui=zeros(fv_nunknowns(pi));Kraken._fv_initial_force!(Ui,pi)
            _fvt4_solve!(sc,Uc);_fvt4_solve!(si,Ui)
            ic=[fv_index(pc.grid,i,j,k) for j in 1:5 for i in 1:4 for k in 1:3]
            ii=[fv_index(pi.grid,i,j,k) for j in 1:5 for i in 1:4 for k in 1:3]
            @test _fvt4_rel(Uc[ic],Ui[ii])<=1e-12
            oc=target===nothing ? fv_flow_rate(pc,1) : fv_flow_force_objective(pc)
            oi=target===nothing ? fv_flow_rate(pi,1) : fv_flow_force_objective(pi)
            map=FVLawParameters(length(theta))
            gc=fv_steady_gradient(sc,Uc,theta,oc;params=map)
            gi=fv_steady_gradient(si,Ui,theta,oi;params=map)
            @test _fvt4_rel(gc.grad,gi.grad)<=1e-12
            @info "T4 isothermal reduction" target state_relative=_fvt4_rel(Uc[ic],Ui[ii]) gradient_relative=_fvt4_rel(gc.grad,gi.grad)
        end
    end
end
