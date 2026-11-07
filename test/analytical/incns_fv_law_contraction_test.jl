# L3 algebraic gates, frozen by the 2026-11-07 implementation contract:
# Float64 relative contraction/parameter-dual agreement <= 1e-12 and centred
# differences of full converged objectives in two random directions <= 1e-6.
# These are discrete derivative checks, not continuum-accuracy tolerances.
using Test, Kraken, Random, LinearAlgebra

_fvl3_rel(a, b) = norm(a-b) / max(norm(a), norm(b), eps(Float64))
function _fvl3_law(kind)
    kind == :classical && return ClassicalLaw(CarreauYasuda(0.3,0.05,1.2,2.0,0.7))
    return LearnedViscosity(TinyMLP{2,2,1}(); nu_ref=0.3, monotone=kind == :monotone)
end
function _fvl3_problem(law, theta; immersed=false, inlet=false, conv=false)
    N = 12
    fields = law_ninputs(law) == 1 ? () :
        ([0.2*(i-0.5)+0.1*(j-0.5) for i in 1:N, j in 1:N],)
    ib = immersed ? fv_immersed_circle(N,N,6.13,5.87,2.21;h=1.0,mass=:M1) : nothing
    bc = inlet ? (W=:inlet,E=:outlet,S=:wall,N=:wall) :
                 (W=:periodic,E=:periodic,S=:wall,N=:wall)
    return fv_steady_problem(N,N;h=1.0,conv,immersed=ib,bc,
        source=(x,y)->(0.02,0.0,0.0),
        viscosity=FVLawViscosity(law,theta,fields;eps=1e-3))
end

@testset "FV law contraction: full gradients and central differences" begin
    for kind in (:classical,:monotone,:free)
        law = _fvl3_law(kind)
        theta = law_init(law;rng=MersenneTwister(33))
        if kind != :classical
            theta[3:4] .= (0.3,-0.2) # Prescribed-field dependence is not zero.
            if kind == :monotone
                offset = law_nparams(law.mlp)
                theta[offset+1:offset+2] .= (0.5,-0.3)
                theta[offset+5:offset+6] .= (0.15,-0.1)
            else
                theta[7:8] .= (0.2,-0.15)
            end
        end
        for immersed in (false,true)
            p = _fvl3_problem(law,theta;immersed)
            s = fv_newton_solver(p)
            U = zeros(fv_nunknowns(p))
            @test fv_newton!(s,U;atol=1e-13,rtol=0.0,maxit=20).converged
            map = FVLawParameters(length(theta))
            ws = fv_adjoint_workspace(s,map;chunk=4)
            probe = FVLinearFunctional([fv_index(p.grid,2,4,1)],[1.0])
            objectives = immersed ?
                (FVWallForce(1;kind=:residual_flux),FVWallForce(1;kind=:surface)) : (probe,)
            for obj in objectives
                contracted = fv_steady_gradient(s,U,theta,obj;params=map,ws)
                duals = fv_steady_gradient(s,U,theta,obj;params=map,ws,law_gradient=:duals,
                    finalize=false)
                err = _fvl3_rel(contracted.grad,duals.grad)
                @test err <= 1e-12
                @test _fvl3_rel(contracted.s,duals.s) <= 1e-12
                @test contracted.explicit == duals.explicit
                @test contracted.gU == duals.gU
                @test all(isfinite,contracted.grad)
                # Two FD directions for each law in both geometries. Both force
                # definitions have parity gates; FD uses one force per law.
                fd_case = !immersed ||
                    obj.kind == (kind == :free ? :surface : :residual_flux)
                fd_errors = Float64[]
                if fd_case
                    sfd = fv_newton_solver(p)
                    rng = MersenneTwister(42)
                    for _ in 1:2
                        direction = normalize(randn(rng,length(theta)))
                        step = 2e-4
                        values = Float64[]
                        for sign in (-1,1)
                            fv_set_params!(sfd,map,theta.+sign*step.*direction)
                            V = copy(U)
                            @test fv_newton!(sfd,V;atol=1e-13,rtol=0.0,maxit=12).converged
                            push!(values,fv_objective(obj,V,sfd.prob))
                        end
                        fd = (values[2]-values[1])/(2step)
                        exact = dot(contracted.grad,direction)
                        push!(fd_errors,abs(fd-exact)/max(abs(exact),eps(Float64)))
                        @test last(fd_errors) <= 1e-6
                    end
                end
                @info "L3 full gradient" kind immersed objective=(obj isa FVWallForce ? obj.kind : :probe) relative=err fd_relative=Tuple(fd_errors)
            end
            @test_throws ArgumentError fv_steady_gradient(s,U,theta,probe;params=map,ws,
                law_gradient=:unknown)
        end
        # Fixed state/cotangent checks include inlet/outlet and nonzero normal
        # velocities, independently of the converged channel momentum balance.
        p = _fvl3_problem(law,theta;inlet=true,conv=true)
        s = fv_newton_solver(p)
        map = FVLawParameters(length(theta))
        ws = fv_adjoint_workspace(s,map;chunk=4)
        U = 0.1randn(MersenneTwister(71),fv_nunknowns(p))
        copyto!(ws.λ,randn(MersenneTwister(72),length(U)))
        for supplied in (theta,theta.+0.01)
            a = Kraken._fv_param_cotangent!(ws,map,s,U,supplied;law_gradient=:contraction)
            b = Kraken._fv_param_cotangent!(ws,map,s,U,supplied;law_gradient=:duals)
            @test _fvl3_rel(a,b) <= 1e-12
        end
    end
end
