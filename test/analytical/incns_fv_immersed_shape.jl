# C5-M3 shape derivative gate, after the open-channel gate.
# Mandatory discrete gate inherited from C4: central FD step sweep agrees
# with the adjoint within 2e-7 absolute + 1e-6 relative (best fixed step).
# Physical convergence threshold is frozen at observed order >=0.8 over
# four grids, all consecutive pairs plus endpoint and logarithmic fit.
# Failure of that physical threshold classifies the result as a discrete
# derivative without continuous validation, as permitted by the C5 plan.
# No step adaptation, topology filtering, or tolerance relaxation is used.
include(joinpath(@__DIR__,"incns_fv_immersed_shape_helpers.jl"))
isdefined(@__MODULE__,:_FV_IMMERSED_KNOWN112) || include(joinpath(@__DIR__,"incns_fv_immersed_known112.jl"))

@testset "C5 shape analytic derivative and frozen topology" begin
    theta = [_C5SHAPE_BASE.ri,_C5SHAPE_BASE.cx]
    @test ForwardDiff.derivative(_c5shape_torque,theta[1]) ≈ _c5shape_dtorque(theta[1]) rtol=32eps()
    p = _c5shape_problem(24,theta)
    map = FVShapeMap(t->_c5shape_problem(24,t;frozen_tag=p.tag),p,2)
    @test map isa FVProblemMap
    @test fv_nparams(map,p) == 2
    @test fv_param_problem(map,p,theta).tag == p.tag
    invalid = _c5shape_error(()->fv_param_problem(map,p,[0.01,theta[2]]))
    @test invalid isa ArgumentError
    @test occursin("radius",sprint(showerror,invalid))
    @test occursin("cells",sprint(showerror,invalid))
    changed = first(findall(==(FV_CELL_FLUID),p.tag))
    bad_map = FVShapeMap(p,2) do t
        q = _c5shape_problem(24,t)
        q.tag[changed] = FV_CELL_SOLID
        q
    end
    mismatch = _c5shape_error(()->fv_param_problem(bad_map,p,theta))
    @test mismatch isa ArgumentError
    @test occursin("frozen fluid/solid/pin tags",sprint(showerror,mismatch))
    @test occursin(string(Tuple(changed)),sprint(showerror,mismatch))
end

let theta = [_C5SHAPE_BASE.ri,_C5SHAPE_BASE.cx], geometry=NamedTuple[]
    # All four nominal grids and every fixed FD step must first pass geometry;
    # nothing is discarded or replaced by a more favorable grid or step.
    for N in _C5SHAPE_GRIDS
        @testset "C5 shape fixed topology N=$N" begin
            p = _c5shape_problem(N,theta)
            map = FVShapeMap(t->_c5shape_problem(N,t;frozen_tag=p.tag),p,2)
            for parameter in 1:2, step in _C5SHAPE_STEPS, sign in (-1,1)
                trial = copy(theta); trial[parameter] += sign*step
                error = _c5shape_error(()->fv_param_problem(map,p,trial))
                push!(geometry,(;N,parameter=(:Ri,:cx)[parameter],step,sign,accepted=error === nothing))
                _c5shape_csv("geometry",geometry)
                @test error === nothing
                error === nothing || throw(error)
            end
        end
    end
end

if get(ENV,"KRAKEN_C5_SHAPE_GEOMETRY_ONLY","false") != "true"
    let theta = [_C5SHAPE_BASE.ri,_C5SHAPE_BASE.cx], evidence=NamedTuple[], sweep=NamedTuple[]
        for N in _C5SHAPE_GRIDS
            p = _c5shape_problem(N,theta)
            map = FVShapeMap(t->_c5shape_problem(N,t;frozen_tag=p.tag),p,2)
            s = fv_newton_solver(p)
            U = _c5shape_initial(p,theta)
            @testset "C5 shape primal N=$N" begin
                status = fv_newton!(s,U;atol=2e-12,rtol=1e-13,maxit=4)
                @test status.converged
                @test norm(fv_residual(U,p),Inf) <= 2e-11
                status.converged || error("C5 STOP: shape primal failed at N=$N")
            end
            objectives = [FVWallTorque((_C5SHAPE_BASE.cx,_C5SHAPE_BASE.cy);kind,wall=1)
                          for kind in (:surface,:residual_flux)]
            adjoints = Any[]
            for obj in objectives
                @testset "C5 shape adjoint $(obj.kind) N=$N" begin
                    result = fv_steady_gradient(s,U,theta,obj;params=map)
                    @test all(isfinite,result.grad)
                    push!(adjoints,result)
                end
            end
            sfd = fv_newton_solver(p)
            for parameter in 1:2, step in _C5SHAPE_STEPS
                @testset "C5 shape FD N=$N parameter=$parameter step=$step" begin
                    values = zeros(2,2)
                    for (side,sign) in enumerate((-1,1))
                        trial = copy(theta); trial[parameter] += sign*step
                        fv_set_params!(sfd,map,trial)
                        V = copy(U)
                        status = fv_newton!(sfd,V;atol=2e-12,rtol=1e-13,maxit=4)
                        @test status.converged || status.stagnated
                        @test norm(fv_residual(V,sfd.prob),Inf) <= 2e-11
                        for (k,obj) in enumerate(objectives)
                            values[k,side] = fv_objective(obj,V,sfd.prob)
                        end
                    end
                    for (k,obj) in enumerate(objectives)
                        fd = (values[k,2]-values[k,1])/(2step)
                        ad = adjoints[k].grad[parameter]
                        push!(sweep,(;N,kind=obj.kind,parameter=(:Ri,:cx)[parameter],step,
                            adjoint=ad,finite_difference=fd,absolute_error=abs(ad-fd),
                            relative_error=abs(ad-fd)/max(abs(ad),1e-12)))
                    end
                    _c5shape_csv("fd_sweep",sweep)
                end
            end
            for (k,obj) in enumerate(objectives)
                @testset "C5 shape discrete derivative $(obj.kind) N=$N" begin
                    for parameter in (:Ri,:cx)
                        samples = filter(r->r.N==N && r.kind==obj.kind && r.parameter==parameter,sweep)
                        best = samples[argmin(getproperty.(samples,:absolute_error))]
                        @fv_immersed_gate best.absolute_error <= 2e-7+1e-6*abs(best.adjoint)
                    end
                    torque = fv_objective(obj,U,p)
                    derivative = adjoints[k].grad[1]
                    exact = _c5shape_dtorque(theta[1])
                    push!(evidence,(;N,kind=obj.kind,ri=theta[1],cx=theta[2],cy=_C5SHAPE_BASE.cy,
                        ro=_C5SHAPE_BASE.ro,torque,analytic_torque=_c5shape_torque(theta[1]),
                        derivative,analytic_derivative=exact,absolute_error=abs(derivative-exact),
                        relative_error=abs(derivative-exact)/abs(exact),cx_derivative=adjoints[k].grad[2],
                        colors=s.asm.ncolors,pairs=length(p.immersed.mass.Q)))
                    _c5shape_csv("derivatives",evidence)
                end
            end
        end
        orders = NamedTuple[]
        @testset "C5 shape continuous derivative classification" begin
            for kind in (:surface,:residual_flux)
                values = sort(filter(r->r.kind==kind,evidence);by=r->r.N)
                @test length(values) == 4
                local_orders = [log(values[k].absolute_error/values[k+1].absolute_error)/
                    log(values[k+1].N/values[k].N) for k in 1:3]
                endpoint = log(first(values).absolute_error/last(values).absolute_error)/
                    log(last(values).N/first(values).N)
                x = log.(Float64.(getproperty.(values,:N)))
                y = log.(getproperty.(values,:absolute_error))
                xc,yc = x .- sum(x)/length(x),y .- sum(y)/length(y)
                fit = -dot(xc,yc)/dot(xc,xc)
                validated = all(>=(0.8),local_orders) && endpoint >= 0.8 && fit >= 0.8
                for (k,order) in enumerate(local_orders)
                    push!(orders,(;kind,Na=values[k].N,Nb=values[k+1].N,measure=:consecutive,order,
                        threshold=0.8,continuous_validated=validated))
                end
                push!(orders,(;kind,Na=first(values).N,Nb=last(values).N,measure=:endpoint,
                    order=endpoint,threshold=0.8,continuous_validated=validated))
                push!(orders,(;kind,Na=first(values).N,Nb=last(values).N,measure=:fit,
                    order=fit,threshold=0.8,continuous_validated=validated))
                @test all(isfinite,(local_orders...,endpoint,fit))
                @info "C5 shape derivative classification" kind local_orders endpoint fit continuous_validated=validated
                validated || @warn "Discrete shape derivative passed FD; continuous derivative is not validated" kind threshold=0.8
            end
            _c5shape_csv("orders",orders)
        end
    end
end
