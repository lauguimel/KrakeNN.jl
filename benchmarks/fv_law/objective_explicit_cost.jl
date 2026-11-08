# Host explicit-objective cost at fixed state: no Newton or adjoint solve.
# Match the L3 248-parameter model; use an immersed circle with 92 wall faces.
using KrakeNN, Random, LinearAlgebra, Statistics

function objective_explicit_cost()
    N = 64
    law = LearnedViscosity(TinyMLP{2,35,1}();nu_ref=0.3,monotone=true)
    theta = law_init(law;rng=MersenneTwister(33))
    field = [0.2*(i-0.5)/N+0.1*(j-0.5)/N for i in 1:N,j in 1:N]
    ib = fv_immersed_circle(N,N,0.511,0.489,0.184;h=1/N,mass=:M1)
    p = fv_steady_problem(N,N;h=1/N,conv=false,immersed=ib,
        bc=(W=:periodic,E=:periodic,S=:wall,N=:wall),
        viscosity=FVLawViscosity(law,theta,(field,);eps=1e-2))
    s = fv_newton_solver(p)
    map = FVLawParameters(length(theta))
    ws = fv_adjoint_workspace(s,map;chunk=4)
    U = 0.03randn(MersenneTwister(71),fv_nunknowns(p))
    copyto!(ws.λ,randn(MersenneTwister(72),length(U)))
    for kind in (:contraction,:residual_flux,:surface)
        operation = kind == :contraction ?
            () -> KrakeNN._fv_param_cotangent!(ws,map,s,U,theta;law_gradient=:contraction) :
            () -> KrakeNN._fv_objective_grad_theta(FVWallForce(1;kind),map,U,p,theta)
        cold = @timed operation()
        all(isfinite,cold.value) || error("Nonfinite result")
        warm = [@timed operation() for _ in 1:3]
        all(r -> r.value == cold.value,warm) || error("Non-reproducible result")
        println((;kind,N,parameters=length(theta),
            wall_faces=length(KrakeNN._fv_wall_faces(FVWallForce(1),p)),cold_s=cold.time,
            compile_s=cold.compile_time,warm_s=median(r.time for r in warm),
            host_bytes=minimum(r.bytes for r in warm),gradient_norm=norm(cold.value)))
        flush(stdout)
    end
end

objective_explicit_cost()
