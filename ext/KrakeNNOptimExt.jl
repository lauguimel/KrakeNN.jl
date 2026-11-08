module KrakeNNOptimExt

using KrakeNN
using Optim
using LinearAlgebra: norm

import KrakeNN: _fit_lbfgs

"""
    _fit_lbfgs(problem, ::LBM, data, p0, pspace; ...) -> CalibResult

L-BFGS-B field-fit driver using `Optim.Fminbox(Optim.LBFGS())`.
Implements the `fit(...; method=:lbfgs)` optimizer hook; loaded when `using Optim`
triggers `KrakeNNOptimExt`.

Cache-last-forward pattern: `compute_fg!` performs one forward+adjoint per evaluation
point; `f` and `g` are always consistent (same point). The objective closure caches the
last result keyed by the parameter vector, so Optim's f and g calls at the same point
do not trigger a double forward solve.

For `LBMFieldParams`-shaped pspaces (names `ν_1…ν_Ny`), `reg_weight > 0` adds Tikhonov
smoothness regularization (α/2 ‖D·ν‖²) and its analytic gradient. Enzyme-free for the
reg term. Returns a `CalibResult` with the same structure as the BB+Armijo (`:pgd`) path.
"""
function _fit_lbfgs(problem, ::KrakeNN.LBM, data, p0::NamedTuple,
                    pspace::KrakeNN.ParameterSpace;
                    observables,
                    weights=nothing,
                    max_iter::Int=200,
                    gtol::Real=1e-6,
                    ftol::Real=1e-12,
                    forward_tol::Real=1e-12,
                    forward_max_steps::Int=200_000,
                    gmres_tol::Real=1e-11,
                    adjoint_tol::Real=KrakeNN.AD_LINEAR_RES_TOL,
                    gmres_restart::Int=240,
                    gmres_max_restarts::Int=20,
                    reg_weight::Real=0.0,
                    verbose::Bool=false,
                    kwargs...)
    x0 = KrakeNN.project!(pspace, copy(KrakeNN.to_flat(pspace, p0)))
    isempty(x0) && throw(ArgumentError("ParameterSpace has no free parameters"))

    loss_trace = Float64[]
    grad_trace = Float64[]

    is_field = KrakeNN._is_nufield_pspace(pspace)

    # Cache-last-forward: avoid double forward solve when f and g are queried separately
    cache_x    = fill(NaN, length(x0))
    cache_val  = Ref(0.0)
    cache_grad = zeros(Float64, length(x0))

    function compute_fg!(x_flat)
        # Return cached result if x_flat unchanged
        if x_flat == cache_x
            return (cache_val[], copy(cache_grad))
        end

        p_named = KrakeNN.from_flat(pspace, x_flat, p0)

        # Forward solve
        kw = Dict{Symbol,Any}(pairs(problem))
        KrakeNN._insert_fit_params!(kw, p_named)
        kw[:tol] = Float64(forward_tol)
        kw[:max_steps] = forward_max_steps

        local fwd
        if is_field
            Ny_val = Int(get(kw, :Ny, problem[:Ny]))
            nu_fld = KrakeNN._extract_nufield(p_named, Ny_val)
            haskey(kw, :nu) && delete!(kw, :nu)
            kw[:nu_field] = nu_fld
            fwd = KrakeNN.ad_forward_solve_nufield(; kw...)
            fwd.converged || @warn "_fit_lbfgs: nufield forward did not converge" residual=fwd.residual
        else
            haskey(kw, :nu) || throw(ArgumentError("_fit_lbfgs: needs :nu in problem or :ν in pspace"))
            fwd = KrakeNN.ad_forward_solve(; kw...)
            fwd.converged || @warn "_fit_lbfgs: forward did not converge" residual=fwd.residual
        end

        # Build predictions
        preds = KrakeNN.Prediction[]
        for obs in observables
            if obs isa KrakeNN.LineProfile && obs.field === :ux
                push!(preds, KrakeNN._obs_lineprofile_ux_from_f(fwd.f_star, obs))
            elseif KrakeNN._is_cd_observable(obs)
                push!(preds, KrakeNN.Prediction(obs, fwd.Cd))
            else
                error("_fit_lbfgs: unsupported observable $(typeof(obs))")
            end
        end

        # Data loss + Tikhonov regularization
        L_data = KrakeNN.loss(preds, data; weights=weights)
        L_reg  = (reg_weight > 0.0 && is_field) ?
                 KrakeNN._reg_loss(fwd.nu_field, reg_weight) : 0.0
        L = L_data + L_reg

        # dL/df assembly
        dLdf = zeros(Float64, size(fwd.f_star))
        cd_residual = 0.0
        for i in eachindex(observables)
            obs = observables[i]
            w   = weights === nothing ? 1.0 : Float64(weights[i])
            if obs isa KrakeNN.LineProfile && obs.field === :ux
                dLdf .+= w .* KrakeNN._dJ_df_lineprofile_ux(fwd.f_star, preds[i].value,
                                                            data[i].value, obs.indices)
            elseif KrakeNN._is_cd_observable(obs)
                res = w * (Float64(preds[i].value) - Float64(data[i].value))
                dLdf .+= res .* KrakeNN._ad_dJdf(fwd.f_star, fwd.q_wall, fwd.u_ref,
                                                 fwd.D, fwd.Nx, fwd.Ny)
                cd_residual += res
            end
        end

        # Adjoint solve — field or scalar state VJP
        if is_field
            nu_fld    = fwd.nu_field
            apply_GtT = v -> KrakeNN._ad_vjp_GtT_nufield(fwd.f_star, v, fwd.q_wall,
                                                         fwd.is_solid, fwd.u_profile,
                                                         fwd.rho_out, nu_fld,
                                                         fwd.Nx, fwd.Ny)
        else
            apply_GtT = v -> KrakeNN._ad_vjp_GtT(fwd.f_star, v, fwd.q_wall, fwd.is_solid,
                                                  fwd.u_profile, fwd.rho_out,
                                                  fwd.s_plus, fwd.s_minus, fwd.Nx, fwd.Ny)
        end
        adj = KrakeNN.gmres_adjoint(apply_GtT, dLdf; tol=gmres_tol,
                                   restart=gmres_restart,
                                   max_restarts=gmres_max_restarts,
                                   linear_tol=adjoint_tol)
        adj.converged || @warn "_fit_lbfgs: adjoint did not converge" linres=adj.linres
        λ = adj.lambda

        # Parameter gradient assembly
        g_flat = zeros(Float64, length(x0))
        if is_field
            nu_fld = fwd.nu_field
            dnu    = KrakeNN._ad_pvjp_nufield(fwd.f_star, λ, nu_fld, fwd.q_wall,
                                             fwd.is_solid, fwd.u_profile, fwd.rho_out,
                                             fwd.Nx, fwd.Ny)
            if reg_weight > 0.0
                dnu .+= KrakeNN._reg_grad(nu_fld, reg_weight)
            end
            gidx = 0
            for i in eachindex(pspace.names)
                pspace.fixed[i] && continue
                gidx += 1
                name = pspace.names[i]
                m = match(r"^ν_(\d+)$", string(name))
                m === nothing && error("_fit_lbfgs: unexpected name $(name) in nufield pspace")
                j = parse(Int, m.captures[1])
                g_flat[gidx] = pspace.log_scale[i] ? dnu[j] * nu_fld[j] : dnu[j]
            end
        else
            gidx = 0
            for i in eachindex(pspace.names)
                pspace.fixed[i] && continue
                gidx += 1
                name = pspace.names[i]
                if name === :ν
                    geom   = KrakeNN.LBMGeomParams(fwd.q_wall, fwd.is_solid, fwd.u_profile,
                                                   fwd.rho_out, fwd.s_plus, fwd.s_minus,
                                                   fwd.Nx, fwd.Ny)
                    dL_dν  = KrakeNN._ad_pvjp_nu(fwd.f_star, λ,
                                                 KrakeNN.LBMScalarParams(geom, p_named[:ν]))
                    g_flat[gidx] = pspace.log_scale[i] ? dL_dν * p_named[:ν] : dL_dν
                else
                    error("_fit_lbfgs: gradient not implemented for parameter :$(name)")
                end
            end
        end

        # Update cache
        copyto!(cache_x, x_flat)
        cache_val[]  = L
        copyto!(cache_grad, g_flat)

        push!(loss_trace, L)
        push!(grad_trace, norm(g_flat))
        verbose && @info "_fit_lbfgs iter=$(length(loss_trace))" L gnorm=norm(g_flat)

        return (L, g_flat)
    end

    # Box bounds in the optimizer's native space (log-scale or natural)
    lower_box = zeros(Float64, length(x0))
    upper_box = zeros(Float64, length(x0))
    jj = 0
    for i in eachindex(pspace.names)
        pspace.fixed[i] && continue
        jj += 1
        if pspace.log_scale[i]
            lower_box[jj] = log(pspace.lower[i])
            upper_box[jj] = log(pspace.upper[i])
        else
            lower_box[jj] = pspace.lower[i]
            upper_box[jj] = pspace.upper[i]
        end
    end

    # Separate f and g! closures — safe across Optim versions
    f_only(x) = compute_fg!(x)[1]
    function g!(G, x)
        _, gv = compute_fg!(x)
        copyto!(G, gv)
        return nothing
    end

    opts = Optim.Options(
        iterations   = max_iter,
        g_tol        = Float64(gtol),
        f_tol        = Float64(ftol),
        show_trace   = false,
        store_trace  = false,
    )

    od     = Optim.OnceDifferentiable(f_only, g!, x0)
    result = Optim.optimize(od, lower_box, upper_box,
                            x0, Optim.Fminbox(Optim.LBFGS()), opts)

    x_opt    = Optim.minimizer(result)
    L_final  = Optim.minimum(result)
    converged = Optim.converged(result)
    n_iter   = length(loss_trace)
    message  = converged ?
               "Optim.Fminbox(LBFGS) converged" :
               "Optim.Fminbox(LBFGS) did not converge ($(Optim.summary(result)))"

    if isempty(loss_trace) || loss_trace[end] != L_final
        push!(loss_trace, L_final)
        push!(grad_trace, 0.0)
    end

    return KrakeNN.CalibResult(KrakeNN.from_flat(pspace, x_opt, p0),
                              L_final, loss_trace, grad_trace,
                              n_iter, converged, message)
end

end # module KrakeNNOptimExt
