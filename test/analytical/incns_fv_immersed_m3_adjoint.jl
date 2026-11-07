# C5-M3 gate 3, executed only after identity and full-rank gates are green.
# Frozen C4 centred-FD bound: 2e-7 absolute + 1e-6 relative, two fixed steps.
# Dense objective state-gradient reference permits 1e-11 absolute roundoff.
isdefined(@__MODULE__,:_c5m3_csv) || include(joinpath(@__DIR__,"incns_fv_immersed_m3_helpers.jl"))

function _c5m3_shape_problem(theta;conv=false,frozen_tag=nothing)
    radius,cx,omega = theta
    ib = fv_immersed_circle(12,12,cx,5.87,radius;h=1.0,frozen_tag,mass=:M3,
        wall_velocity=(k,x,y)->(-omega*(y-5.87),omega*(x-cx)))
    return fv_steady_problem(12,12;h=1.0,immersed=ib,wall_flux=0.0,conv,
        nu=(x,y)->0.3+0.01x,
        boundary_velocity=(s,x,y)->s==:N ? (0.1,0.0) : (0.0,0.0))
end

let evidence = NamedTuple[]
    for conv in (false,true),kind in (:surface,:residual_flux),objective in (:force_x,:force_y,:torque)
        @testset "M3 adjoint $objective $kind conv=$conv" begin
            theta = [2.21,6.13,0.2]
            p = _c5m3_shape_problem(theta;conv)
            map = FVProblemMap((q,t)->_c5m3_shape_problem(t;conv,frozen_tag=q.tag),3)
            obj = objective == :torque ? FVWallTorque((6.13,5.87);kind) :
                                        FVWallForce(objective == :force_x ? 1 : 2;kind)
            arbitrary = 0.3randn(MersenneTwister(834),fv_nunknowns(p))
            _,local_grad = KrakeNN._fv_objective_grad_u(obj,arbitrary,p)
            dense_grad = ForwardDiff.gradient(x->fv_objective(obj,x,p),arbitrary)
            @test norm(local_grad-dense_grad,Inf) <= 1e-11
            lambda_derivative = dense_grad[end]
            if kind == :residual_flux && conv
                @test !iszero(lambda_derivative)
            else
                @test iszero(lambda_derivative)
            end
            s = fv_newton_solver(p)
            U = zeros(fv_nunknowns(p))
            status = fv_newton!(s,U;atol=1e-12,rtol=1e-13,maxit=12)
            @test status.converged
            status.converged || error("M3 STOP: primal solve failed before adjoint")
            result = fv_steady_gradient(s,U,theta,obj;params=map)
            sfd = fv_newton_solver(p)
            for k in eachindex(theta)
                errors,derivatives = Float64[],Float64[]
                for step in (1e-4,2e-5)
                    values = Float64[]
                    for sign in (-1,1)
                        trial = copy(theta); trial[k] += sign*step
                        fv_set_params!(sfd,map,trial)
                        V = copy(U)
                        status_fd = fv_newton!(sfd,V;atol=1e-12,rtol=1e-13,maxit=12)
                        @test status_fd.converged || status_fd.stagnated
                        @test last(status_fd.inf) < 2e-11
                        push!(values,fv_objective(obj,V,sfd.prob))
                    end
                    derivative = (values[2]-values[1])/(2step)
                    push!(derivatives,derivative)
                    push!(errors,abs(derivative-result.grad[k]))
                end
                best = argmin(errors)
                @test errors[best] <= 2e-7+1e-6*abs(result.grad[k])
                push!(evidence,(;conv,kind,objective,parameter=(:R,:cx,:omega)[k],
                    adjoint=result.grad[k],fd_large=derivatives[1],fd_small=derivatives[2],
                    absolute_error=errors[best],relative_error=errors[best]/max(abs(result.grad[k]),1e-12),
                    lambda_derivative,colors=s.asm.ncolors))
            end
            _c5m3_csv("closed_mass_m3_adjoint",evidence)
        end
    end
end
