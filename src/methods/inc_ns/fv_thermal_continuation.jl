"""Rebuild only the heating coefficient; keep nominal scales and arrays fixed."""
function _fv_heating_problem(p, sigma)
    e = p.energy
    energy = _FVEnergyData(e.alpha, typeof(e.sigma)(sigma), e.sigma_nom,
                           e.boundary, e.dirichlet, e.scheme, e.source)
    return _fv_with(p; energy)
end

"""Inverse iteration on J^(-T)J^(-1), using the existing Newton factorization.

The returned value estimates the smallest singular value of the physical,
unscaled residual Jacobian. Both normal and transpose solves use the same
factor cache; no dense matrix, new analysis, or GPU reduction storage is used.
It is a diagnostic estimate, not a certified bound or proof of a fold.
"""
function _fv_smallest_singular_estimate!(s, x, y, z; iterations=12)
    x ./= sqrt(_fv_state_sig(s.norms,0.0,x)[3])
    for _ in 1:iterations
        lin_solve!(y,s.cache,x)
        lin_solve!(z,s.cache,y;transpose=true)
        magnitude = sqrt(_fv_state_sig(s.norms,0.0,z)[3])
        isfinite(magnitude) && magnitude > 0 || return 0.0
        x .= z ./ magnitude
    end
    lin_solve!(y,s.cache,x)
    magnitude = sqrt(_fv_state_sig(s.norms,0.0,y)[3])
    return isfinite(magnitude) && magnitude > 0 ? inv(magnitude) : 0.0
end

function _fv_continuation_failure(error)
    error isa LinearAlgebra.SingularException && return true
    error isa LinearAlgebra.ZeroPivotException && return true
    error isa LinearAlgebra.LAPACKException && return true
    # cuDSS reports numerical singularity through backend-specific exceptions.
    name = string(typeof(error)); message = lowercase(sprint(showerror,error))
    return (occursin("CUDSS",uppercase(name)) || error isa ErrorException) &&
           (occursin("singular",message) || occursin("factorization",message) ||
            occursin("not finite",message) || occursin("nonfinite",message))
end

function _fv_continuation_attempt!(s,U,scales;maxit,verbose)
    try
        stats = fv_newton!(s,U;maxit,verbose,thermal_scales=scales,
                            alpha_min=1e-4,abort_small_step=true)
        reason = stats.converged ? :converged : stats.failure_reason == :line_search ?
                 :line_search : stats.iterations >= maxit ? :max_iterations : :stagnated
        return (; converged=stats.converged,reason,stats,error=nothing)
    catch error
        _fv_continuation_failure(error) || rethrow()
        return (;converged=false,reason=:singular_factorization,stats=nothing,
                 error=sprint(showerror,error))
    end
end

"""
    fv_thermal_continuation(prob; U0=nothing, initial_step=0.25,
                            max_halvings=5, maxit=30, to=Array,
                            linsolve=CPUBackendTag(), chunk=4, verbose=false)

Continue heating from zero to `prob.energy.sigma_nom`. `initial_step` is a
fraction of that nominal coefficient. The initial solve has uniform imposed
wall temperature and zero heating; thermal walls must share that temperature
and the independent energy source must vanish. `U0` supplies an optional flow
guess; its temperatures are reset to this isothermal state.

Pass-3 equation scales are computed once from nominal case data and reused
for every Newton solve and line search. A step fails on a line-search factor
below 1e-4 or after 30 iterations (a smaller `maxit` is useful for diagnostics).
Failed steps restore the last converged state and halve the heating increment,
at most five times in the whole run. No arclength continuation is attempted.

Returns fields, device `U`, solver, `converged`, `reason`, `last_sigma`,
`last_failure`, `halvings`, converged-step `history`, and all `attempts`.
Each history entry includes `sigma`, Newton `stats`, and `singular_min`, an
inverse-iteration estimate for the physical Jacobian using normal/transpose
solves of its existing factorization. Exhaustion is a structured return;
`reason=:step_halving_limit` may indicate thermal runaway but does not prove
a fold. The returned state is always the last converged state if one exists.
"""
function fv_thermal_continuation(prob::FVSteadyProblem2D; U0=nothing,
        initial_step::Real=0.25, max_halvings::Int=5, maxit::Int=30,
        singular_iterations::Int=12, to=Array,
        linsolve::LinearSolveBackend=CPUBackendTag(),chunk::Int=4,verbose::Bool=false)
    prob.energy === nothing && throw(ArgumentError("heating continuation requires FVEnergy"))
    0 < initial_step <= 1 || throw(ArgumentError("initial_step must lie in (0,1]"))
    0 <= max_halvings <= 5 || throw(ArgumentError("max_halvings must lie in 0:5"))
    1 <= maxit <= 30 || throw(ArgumentError("maxit must lie in 1:30"))
    singular_iterations > 0 || throw(ArgumentError("singular_iterations must be positive"))
    p = prob.nu isa Array ? prob : fv_adapt(Array,prob)
    e = p.energy
    walls = [e.boundary[d] for d in 1:4 if e.dirichlet[d]]
    Tw = first(walls)
    all(==(Tw),walls) || throw(ArgumentError("isothermal continuation start requires equal imposed temperatures"))
    all(iszero,e.source) || throw(ArgumentError("isothermal continuation start requires zero independent energy source"))
    scales = _fv_energy_scales(p)
    initial = U0 === nothing ? _fv_initial_force!(zeros(Float64,fv_nunknowns(p)),p) : Float64.(Array(U0))
    length(initial) == fv_nunknowns(p) || throw(DimensionMismatch("U0 does not match the problem"))
    for cell in eachindex(p.tag)
        initial[_fv_cell_index(p.grid,cell,4)] = p.tag[cell] == FV_CELL_SOLID ? 0.0 : Tw
    end
    s = fv_newton_solver(_fv_heating_problem(p,0);to,linsolve,chunk)
    U = to(initial); saved = copy(U)
    x = to([sin(Float64(k)) + cos(sqrt(2.0)*k) for k in eachindex(initial)])
    y = similar(U); z = similar(U)
    history = NamedTuple[]; attempts = NamedTuple[]
    last_sigma = nothing; last_failure = nothing; halvings = 0
    nominal = Float64(e.sigma_nom); step = initial_step*nominal; trial_sigma = 0.0
    reason = :initial_failure
    while true
        s.prob = _fv_heating_problem(s.prob,trial_sigma)
        copyto!(U,saved)
        attempt = _fv_continuation_attempt!(s,U,scales;maxit,verbose)
        estimate = NaN
        if attempt.converged
            try
                fv_newton_finalize!(s,U)
                estimate = _fv_smallest_singular_estimate!(s,x,y,z;iterations=singular_iterations)
            catch error
                _fv_continuation_failure(error) || rethrow()
                attempt = (;converged=false,reason=:singular_factorization,stats=attempt.stats,
                             error=sprint(showerror,error))
            end
        end
        push!(attempts,(;sigma=trial_sigma,step,halvings,attempt...))
        if attempt.converged
            push!(history,(;sigma=trial_sigma,singular_min=estimate,stats=attempt.stats))
            copyto!(saved,U); last_sigma = trial_sigma
            if trial_sigma >= nominal
                reason = :converged
                break
            end
            trial_sigma = min(nominal,last_sigma+step)
        else
            last_failure = attempt.reason
            last_sigma === nothing && break
            if halvings >= max_halvings
                reason = :step_halving_limit
                break
            end
            halvings += 1; step /= 2
            trial_sigma = min(nominal,last_sigma+step)
        end
    end
    copyto!(U,saved)
    if last_sigma !== nothing
        s.prob = _fv_heating_problem(s.prob,last_sigma)
        # Failed trials overwrite the numeric factors; expose matching final factors.
        reason == :converged || fv_newton_finalize!(s,U)
    end
    u,v,pressure,temperature = fv_fields(p,U)
    return (;converged=reason==:converged,reason,last_sigma,last_failure,halvings,
             history,attempts,U,u,v,p=pressure,T=temperature,solver=s,thermal_scales=scales)
end
