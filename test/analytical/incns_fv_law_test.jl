# L2 scalar-law FV integration. CPU unit fixtures, Float64 unless stated.
# Frozen algebraic gates from the L2 contract: classical residual/J <=1e-13,
# Newton state <=1e-12, full implicit gradient vs dense duals <=1e-12 and
# centred finite differences in two random directions <=1e-6 (relative).
# These test discretization/derivative consistency, not continuum accuracy.
using Test, KrakeNN, ForwardDiff, Random, LinearAlgebra

struct _FVLFieldLaw <: AbstractScalarLaw end
KrakeNN.law_ninputs(::_FVLFieldLaw) = 2
KrakeNN.law_nparams(::_FVLFieldLaw) = 1
@inline KrakeNN.law_value(::_FVLFieldLaw, x::Tuple, theta::AbstractVector) = theta[1] + x[1] + x[2]
struct _FVLForeignBackend <: KrakeNN.KernelAbstractions.Backend end
struct _FVLForeignMatrix <: AbstractMatrix{Float64}
    values::Matrix{Float64}
end
Base.size(f::_FVLForeignMatrix) = size(f.values)
KrakeNN.KernelAbstractions.get_backend(::_FVLForeignMatrix) = _FVLForeignBackend()

_fvl_rel(a, b) = norm(a - b) / max(norm(b), eps(Float64))
_fvl_temperature() = [0.2 * (i-0.5) + 0.1 * (j-0.5) for i in 1:12, j in 1:12]
function _fvl_problem(visc; immersed=true)
    ib = immersed ? fv_immersed_circle(12, 12, 6.13, 5.87, 2.21; h=1.0, mass=:M1) : nothing
    return fv_steady_problem(12, 12; h=1.0, immersed=ib, viscosity=visc, conv=false,
        bc=(W=:periodic, E=:periodic, S=:wall, N=:wall), source=(x,y)->(0.02, 0.0, 0.0))
end
function _fvl_learned(monotone)
    mlp = TinyMLP{2,2,1}()
    law = LearnedViscosity(mlp; nu_ref=0.3, monotone)
    theta = law_init(law; rng=MersenneTwister(33))
    # Non-negligible thermal dependence in both network modes. Indices follow
    # the public column-major TinyMLP parameter layout, without solver state.
    theta[3:4] .= (0.3, -0.2)
    if monotone
        offset = law_nparams(mlp)
        theta[offset+1:offset+2] .= (0.5, -0.3)
        theta[offset+5:offset+6] .= (0.15, -0.1)
    else
        theta[7:8] .= (0.2, -0.15)
    end
    return law, theta
end

@testset "FV law: field convention and validation" begin
    field = _fvl_temperature()
    visc = FVLawViscosity(_FVLFieldLaw(), [0.3], (field,); eps=0.02)
    p = _fvl_problem(visc)
    grad = (0.2, 0.3, -0.1, -0.2)
    sxy = (grad[2]+grad[3])/2
    gamma = sqrt(2*(grad[1]^2 + grad[4]^2 + 2sxy^2 + visc.eps^2/2))
    for (i,j,d) in ((2,3,1), (12,3,1)) # interior, periodic wrap
        kind, ii, jj = KrakeNN._fv_face_kind(p.grid, p.tag, i,j,d)
        @test kind == KrakeNN._FV_F_INT
        got = KrakeNN._fv_face_nu(visc, 99.0, 99.0, grad..., i,j,ii,jj)
        @test got == 0.3 + gamma + (field[i,j]+field[ii,jj])/2
    end
    @test KrakeNN._fv_wall_nu(visc, 99.0, grad..., 2,1) == 0.3+gamma+field[2,1]
    # Inlet shares the same Dirichlet stress hook and takes cell P.
    inlet = fv_steady_problem(12,12; h=1.0, viscosity=visc,
        bc=(W=:inlet,E=:outlet,S=:wall,N=:wall))
    @test first(KrakeNN._fv_face_kind(inlet.grid,inlet.tag,1,3,2)) == KrakeNN._FV_F_DIRICHLET
    tx, ty = KrakeNN._fv_wall_stress(inlet.nu,visc,inlet.grid,1,3,2,grad,0.2,-0.1,nothing,nothing)
    nuw = law_value(visc.law, (sqrt(2*((-0.2)^2+grad[4]^2+2*((grad[2]+0.1)/2)^2+visc.eps^2/2)),field[1,3]),visc.theta)
    @test tx == 2nuw*0.2 && ty == nuw*(-0.1-grad[2])
    # Immersed nominal and physical-cut evaluations both use P, not O or an
    # extrapolated field. Given Newtonian viscosity retains its own convention.
    U = fv_state(p,(x,y)->0.2x+0.3y,(x,y)->-0.1x-0.2y,(x,y)->0.1x)
    q,w,c = KrakeNN._fv_immersed_args(p.immersed)
    for (i,j,d) in KrakeNN._fv_wall_faces(FVWallForce(1),p)
        sx = d in (1,3) ? 1 : -1
        nuw = KrakeNN._fv_wall_nu(visc,99.0,grad...,i,j)
        @test nuw == 0.3+gamma+field[i,j]
        # Nominal stress: choose normal differences matching this gradient.
        dun,dvn = d <= 2 ? (sx*grad[1],sx*grad[3]) : (sx*grad[2],sx*grad[4])
        got = KrakeNN._fv_wall_stress(p.nu,visc,p.grid,i,j,d,grad,dun,dvn,q,c)
        ref = d <= 2 ? (2nuw*dun,nuw*(dvn+sx*grad[2])) : (nuw*(dun+sx*grad[3]),2nuw*dvn)
        @test got == ref
        surface = p.immersed.surface.values
        r = KrakeNN._fv_surface_reconstruction(U,p,i,j,d,
            (surface[3,d,i,j],surface[4,d,i,j]),(0.0,0.0),99.0)
        ux,uy,vx,vy = r.gradient
        rate = sqrt(2*(ux*ux+vy*vy+2*((uy+vx)/2)^2+visc.eps^2/2))
        @test r.viscosity == law_value(visc.law,(rate,field[i,j]),visc.theta)
    end
    @test_throws ArgumentError FVLawViscosity(visc.law,visc.theta,())
    @test_throws ArgumentError FVLawViscosity(visc.law,visc.theta,(field,field))
    @test_throws ArgumentError FVLawViscosity(visc.law,visc.theta,(_FVLForeignMatrix(field),))
    @test_throws DimensionMismatch fv_steady_problem(6,6; viscosity=visc)
    bad = KrakeNN._fv_with(p;visc=FVLawViscosity(visc.law,visc.theta,(ones(2,2),),visc.eps))
    @test_throws DimensionMismatch fv_newton_solver(bad)
    @test_throws ArgumentError FVLawViscosity(visc.law,visc.theta,(field,);eps=0.0)
    @test_throws ArgumentError fv_steady_problem(12,12;viscosity=visc,rheology=Newtonian(0.3))
    @test fv_nunknowns(p) == 3*12*12
end

@testset "FV law: classical equivalence and unchanged structure" begin
    model = CarreauYasuda(0.3,0.05,1.2,2.0,0.7)
    law = ClassicalLaw(model)
    theta = law_init(law)
    solid = falses(6,6); solid[3:4,3:4] .= true
    ib = fv_immersed_circle(12,12,6.13,5.87,2.21;h=1.0,mass=:M1)
    configs = ((6,6,(;)),(6,6,(;solid)),(12,12,(;immersed=ib)))
    for eps0 in (0.0,1e-3), (nx,ny,kw) in configs
        common = (;h=1.0,conv=false,bc=(W=:periodic,E=:periodic,S=:wall,N=:wall),
                   source=(x,y)->(0.02,0.0,0.0))
        ref = fv_steady_problem(nx,ny;rheology=model,gnf_eps=eps0,common...,kw...)
        p = fv_steady_problem(nx,ny;viscosity=FVLawViscosity(law,theta;eps=eps0),common...,kw...)
        @test fv_jacobian_pattern(p) == fv_jacobian_pattern(ref)
        @test fv_jacobian_coloring(fv_jacobian_pattern(p)) == fv_jacobian_coloring(fv_jacobian_pattern(ref))
        s,sref = fv_newton_solver(p),fv_newton_solver(ref)
        for U in (zeros(fv_nunknowns(p)),0.1randn(MersenneTwister(12),fv_nunknowns(p)))
            @test _fvl_rel(fv_residual(U,p),fv_residual(U,ref)) <= 1e-13
            fv_jacobian!(s,U); fv_jacobian!(sref,U)
            @test all(isfinite,s.J)
            @test _fvl_rel(s.J,sref.J) <= 1e-13
        end
        U,V = zeros(fv_nunknowns(p)),zeros(fv_nunknowns(ref))
        @test fv_newton!(s,U;atol=1e-13,rtol=0.0,maxit=20).converged
        @test fv_newton!(sref,V;atol=1e-13,rtol=0.0,maxit=20).converged
        @test _fvl_rel(U,V) <= 1e-12
        if eps0 == 0
            badtheta = copy(theta); badtheta[4] = 1.5
            @test_throws ArgumentError FVLawViscosity(law,badtheta;eps=0.0)
            old = s.prob; sig = s.fac_sig
            @test_throws ArgumentError fv_set_params!(s,FVLawParameters(5),badtheta)
            @test s.prob === old && s.fac_sig === sig
        end
    end
    # The unused ClassicalLaw template/thermal metadata must not promote
    # actual Float32 parameters and eps to Float64.
    thermal = ArrheniusCoupling(300.0,1000.0)
    c = ClassicalLaw(CarreauYasuda(0.3,0.05,1.2,2.0,0.7;thermal))
    v = FVLawViscosity(c,Float32.(theta);eps=1f-3)
    @test (@inferred KrakeNN._fv_visc_eltype(v)) === Float32
    @test (@inferred KrakeNN._fv_face_nu(v,0f0,0f0,0.2f0,0.3f0,-0.1f0,0.4f0,1,1,2,1)) isa Float32
end

@testset "FV law: concrete scalar and kernel inference" begin
    D = ForwardDiff.Dual{FVParamTag,Float64,4}
    S = ForwardDiff.Dual{FVJacobianTag,Float64,4}
    for monotone in (true,false)
        law,theta = _fvl_learned(monotone)
        for (PT,ST) in ((Float64,Float64),(D,Float64),(Float64,S))
            v = FVLawViscosity(law,PT.(theta),(_fvl_temperature(),))
            p = _fvl_problem(v;immersed=false)
            T = promote_type(PT,ST)
            @test (@inferred KrakeNN._fv_visc_eltype(v)) === PT
            @test (@inferred KrakeNN._fv_visc_zero(v)) === zero(PT)
            @test (@inferred KrakeNN._fv_face_nu(v,0.0,0.0,ST(0.2),ST(0.3),ST(-0.1),ST(0.4),1,1,2,1)) isa T
            U = fill(ST(0.1),fv_nunknowns(p))
            @test (@inferred fv_cell_residual(U,p.tag,p.nu,p.bvel,p.src,p.grid,2,2,v)) isa NTuple{3,T}
            # Reflect the two derivative lanes of the larger monotone law;
            # scalar/cell inference above still checks every numeric/mode pair.
            if monotone && (PT === D || ST === S)
                KA = KrakeNN.KernelAbstractions
                kernel = KrakeNN._fv_residual_kernel!(KA.CPU(),64)
                args = (zeros(T,length(U)),U,p.tag,p.nu,p.bvel,p.src,p.grid,v,nothing,nothing,nothing,nothing)
                ir = only(KA.ka_code_typed(kernel,Tuple{map(typeof,args)...};ndrange=(12,12),optimize=true)).first
                dynamic = [stmt for (stmt,typ) in zip(ir.code,ir.ssavaluetypes)
                           if stmt isa Expr && stmt.head in (:call,:invoke) && typ === Any]
                @test isempty(dynamic)
            end
        end
    end
end

@testset "FV learned law: Newton, full objective gradients and parameter reuse" begin
    for monotone in (true,false)
        law,theta = _fvl_learned(monotone)
        temperature = _fvl_temperature()
        p = _fvl_problem(FVLawViscosity(law,theta,(temperature,);eps=1e-3))
        s = fv_newton_solver(p)
        U = zeros(fv_nunknowns(p))
        @test fv_newton!(s,U;atol=1e-13,rtol=0.0,maxit=20).converged
        @test norm(fv_residual(U,p),Inf) < 1e-13
        params = FVLawParameters(length(theta))
        ws = fv_adjoint_workspace(s,params;chunk=4)
        probe = FVLinearFunctional([fv_index(p.grid,2,4,1)],[1.0])
        objectives = monotone ?
            (probe,FVWallForce(1;kind=:residual_flux),FVWallForce(1;kind=:surface)) : (probe,)
        dR = ForwardDiff.jacobian(t->fv_residual(U,fv_param_problem(params,p,t)),theta)
        denseJ = ForwardDiff.jacobian(x->fv_residual(x,p),U)
        fv_jacobian!(s,U)
        @test _fvl_rel(Matrix(s.J),denseJ) <= 1e-12
        rng = MersenneTwister(42)
        directions = [normalize(randn(rng,length(theta))) for _ in 1:2]
        sfd = fv_newton_solver(p)
        for obj in objectives
            result = fv_steady_gradient(s,U,theta,obj;params,ws)
            gu = ForwardDiff.gradient(x->fv_objective(obj,x,p),U)
            direct = ForwardDiff.gradient(t->fv_objective(obj,U,fv_param_problem(params,p,t)),theta)
            dense = direct - transpose(dR)*(transpose(denseJ)\gu)
            @test _fvl_rel(result.grad,dense) <= 1e-12
            @test norm(result.gU-gu,Inf) <= 1e-12
            @test norm(result.explicit-direct,Inf) <= 1e-12
            if obj isa FVWallForce
                @test norm(direct) > 1e-8 # exercises direct objective-law dependence
                i,j,d = first(KrakeNN._fv_wall_faces(obj,p))
                @test length(KrakeNN._fv_wall_state(U,p,i,j,d).index) == 27 # Q9 state path
                changed = KrakeNN._fv_with(p;visc=FVLawViscosity(law,theta,(temperature.+0.5,);eps=1e-3))
                @test abs(fv_objective(obj,U,changed)-result.J) > 1e-6
            end
            errors = Float64[]
            for direction in directions
                step = 2e-4
                values = Float64[]
                for sign in (-1,1)
                    fv_set_params!(sfd,params,theta.+sign*step.*direction)
                    V = copy(U)
                    @test fv_newton!(sfd,V;atol=1e-13,rtol=0.0,maxit=12).converged
                    push!(values,fv_objective(obj,V,sfd.prob))
                end
                fd = (values[2]-values[1])/(2step)
                exact = dot(result.grad,direction)
                push!(errors,abs(fd-exact)/max(abs(exact),eps(Float64)))
                @test last(errors) <= 1e-6
            end
            @info "L2 full objective gradient" monotone objective=typeof(obj) kind=(obj isa FVWallForce ? obj.kind : :probe) dense_relative=_fvl_rel(result.grad,dense) fd_relative=Tuple(errors)
        end
        asm,J,cache = s.asm,s.J,s.cache
        changedtheta = theta.+0.01
        fields = (temperature.+0.1,)
        fv_set_params!(s,params,changedtheta;fields)
        @test s.prob.visc.theta == changedtheta && s.prob.visc.fields == fields
        @test isnan(first(s.fac_sig))
        @test s.asm === asm && s.J === J && s.cache === cache
        @test fv_newton!(s,U;atol=1e-13,rtol=0.0,maxit=15).converged
        @test s.n_factorize == 1
        before = s.prob; sig = s.fac_sig
        @test_throws ArgumentError fv_set_params!(s,params,theta;fields=(temperature,temperature))
        @test_throws DimensionMismatch fv_set_params!(s,params,theta;fields=(ones(2,2),))
        @test_throws ArgumentError fv_set_params!(s,params,theta;fields=(Float32.(temperature),))
        @test_throws ArgumentError fv_set_params!(s,params,theta;fields=(_FVLForeignMatrix(temperature),))
        other = FVProblemMap((q,t)->KrakeNN._fv_with(q;visc=FVLawViscosity(_FVLFieldLaw(),[0.3],(temperature,))),length(theta))
        @test_throws ArgumentError fv_set_params!(s,other,theta)
        @test s.prob === before && s.fac_sig === sig
        @test s.n_factorize == 1
        adapted = fv_adapt(Array,p)
        @test adapted.visc.theta == theta && adapted.visc.fields == p.visc.fields
    end
end
