# DenseNet inside FVLawViscosity: contraction weight gradient == :duals
# reference (<= 1e-12 relative) and central differences of the converged
# objective on random weight directions (<= 1e-6). Same setup as
# incns_fv_law_contraction_test.jl.
using Test, KrakeNN, Random, LinearAlgebra

_fvd_rel(a, b) = norm(a-b) / max(norm(a), norm(b), eps(Float64))

function _fvd_problem(law, theta; immersed=false)
    N = 12
    fields = ([0.2*(i-0.5)+0.1*(j-0.5) for i in 1:N, j in 1:N],)
    ib = immersed ? fv_immersed_circle(N,N,6.13,5.87,2.21;h=1.0,mass=:M1) : nothing
    return fv_steady_problem(N,N;h=1.0,immersed=ib,
        bc=(W=:periodic,E=:periodic,S=:wall,N=:wall),
        source=(x,y)->(0.02,0.0,0.0),
        viscosity=FVLawViscosity(law,theta,fields;eps=1e-3))
end

@testset "FV law contraction with DenseNet" begin
    for (hidden, act) in (((4,3), (:tanh,:softplus)), ((5,), :sigmoid))
        law = DenseNet(2, hidden; activation=act, output=:exp)
        theta = law_init(law; rng=MersenneTwister(5))
        theta[end] = log(0.3)                      # output bias: nu ~ 0.3
        for immersed in (false, true)
            p = _fvd_problem(law, theta; immersed)
            s = fv_newton_solver(p)
            U = zeros(fv_nunknowns(p))
            @test fv_newton!(s,U;atol=1e-13,rtol=0.0,maxit=20).converged
            map = FVLawParameters(length(theta))
            ws = fv_adjoint_workspace(s,map;chunk=4)
            obj = immersed ? FVWallForce(1;kind=:surface) :
                  FVLinearFunctional([fv_index(p.grid,2,4,1)],[1.0])
            contracted = fv_steady_gradient(s,U,theta,obj;params=map,ws)
            duals = fv_steady_gradient(s,U,theta,obj;params=map,ws,law_gradient=:duals,
                finalize=false)
            err = _fvd_rel(contracted.grad,duals.grad)
            @test err <= 1e-12
            @test all(isfinite,contracted.grad)
            @test norm(contracted.grad) > 0
            sfd = fv_newton_solver(p)
            rng = MersenneTwister(42)
            fd_errs = Float64[]
            for _ in 1:3
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
                push!(fd_errs,abs(fd-exact)/max(abs(exact),eps(Float64)))
                @test last(fd_errs) <= 1e-6
            end
            @info "DenseNet FV gradient" hidden immersed contraction_vs_duals=err fd=Tuple(fd_errs)
        end
    end
end
