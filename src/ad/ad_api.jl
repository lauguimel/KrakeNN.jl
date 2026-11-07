const _AD_ENZYME_LOAD_ERROR = "Load Enzyme to enable AD: `using " * "Enzyme`"

function _ad_dJdf(args...)
    error(_AD_ENZYME_LOAD_ERROR)
end

function _ad_vjp_GtT(args...)
    error(_AD_ENZYME_LOAD_ERROR)
end

function _ad_dqwall_terms(args...)
    error(_AD_ENZYME_LOAD_ERROR)
end

function _ad_dNudw(args...)
    error(_AD_ENZYME_LOAD_ERROR)
end

function _ad_thermal_vjp_GtT(args...)
    error(_AD_ENZYME_LOAD_ERROR)
end

function _ad_thermal_dqwall_terms(args...)
    error(_AD_ENZYME_LOAD_ERROR)
end

# --- Viscoelastic (Cd_polymer) AD seams (methods live in ext/KrakeNNADExt.jl) ---
# `_ad_ve_dJdw`     : reverse of `ad_ve_J_fx` wrt the stacked state w -> dJ/dw.
# `_ad_ve_vjp_GtT`  : reverse of `ad_ve_coupled_step!` -> dG^T . v (adjoint matvec).
# `_ad_ve_dGdR_jvp` : forward-JVP of `ad_ve_coupled_step!` seeded by d(geom)/dR
#                     (EmbeddedGeom + q_wall Duplicated) -> dG/dR (the analytic chain).
function _ad_ve_dJdw(args...)
    error(_AD_ENZYME_LOAD_ERROR)
end

function _ad_ve_vjp_GtT(args...)
    error(_AD_ENZYME_LOAD_ERROR)
end

function _ad_ve_dGdR_jvp(args...)
    error(_AD_ENZYME_LOAD_ERROR)
end

# --- ν parameter VJP seam (M-P2b-1) ---
# `_ad_pvjp_nu` : computes dL/dν = λᵀ (∂G/∂ν) via Enzyme Reverse over
#                  ad_step_nu! with Active(ν). Returns a scalar Float64.
#                  Impl lives in ext/KrakeNNADExt.jl.
function _ad_pvjp_nu(args...)
    error(_AD_ENZYME_LOAD_ERROR)
end

"""
    steady_shape_sensitivity(; qoi=:drag, ...)

Compute a steady shape sensitivity on the validated CPU Float64 AD paths.
Load Enzyme before calling this function.

For `qoi=:drag`, pass `Nx, Ny, radius, u_in, ν` (or `nu`), with
`wrt=:radius` and optional `cx, cy, ρ_out, tol, max_steps, inlet,
gmres_tol, adjoint_tol, fd_check, fd_h`.

For `qoi=:nusselt`, pass `N, Ra, Pr`, with `wrt=:wall_position` and
optional `L` (or `wall_param`/`wall_position`), `q_hot, q_cold, T_hot,
T_cold, tol, max_steps, gmres_tol, adjoint_tol, fd_check, fd_h`.

For `qoi=:polymer_drag`, pass `Nx, Ny, radius, cx, cy, Wi, beta`, the
polymer viscosity (`ν_p`/`nu_p`) and solvent viscosity (`ν_s`/`nu_s`),
and `Fx_body`, with `wrt=:radius` and optional `n_substeps, dt, samples,
u_mean, ρ_out, fwd_tol, max_steps, patience, gmres_tol, adjoint_tol,
bc, fd_check, fd_h`. The QoI is the polymeric x-drag `Fx`. The forward
reconverges to `fwd_tol=1e-13` (mandatory: the net `d(Fx)/dR` is a ~20x
catastrophic cancellation that a looser floor poisons). For `bc=:open`
(cylinder west-velocity / east-pressure ZouHe) the adjoint is the
UNGAUGED `(I-dGᵀ)λ=dJ/dw` solve; `bc∈{:closed,:periodic}` reuses the
mass-gauged path.
"""
function steady_shape_sensitivity(; qoi::Symbol=:drag,
                                    wrt::Symbol=(qoi === :nusselt ?
                                                 :wall_position : :radius),
                                    kwargs...)
    if qoi === :drag
        return _steady_drag_sensitivity(; wrt=wrt, kwargs...)
    elseif qoi === :nusselt
        return _steady_nusselt_sensitivity(; wrt=wrt, kwargs...)
    elseif qoi === :polymer_drag
        return _steady_polymer_drag_sensitivity(; wrt=wrt, kwargs...)
    end
    throw(ArgumentError("unsupported qoi=$(qoi); supported qoi values are :drag, :nusselt and :polymer_drag"))
end

function _ad_pick_viscosity(ν, nu)
    if ν !== nothing && nu !== nothing
        throw(ArgumentError("pass only one of `ν` or `nu`"))
    end
    value = ν === nothing ? nu : ν
    value === nothing && throw(ArgumentError("missing required viscosity keyword `ν` or `nu`"))
    return Float64(value)
end

function _steady_drag_sensitivity(; Nx::Int, Ny::Int, radius::Real,
                                    u_in::Real,
                                    ν=nothing,
                                    nu=nothing,
                                    ρ_out::Real=1.0,
                                    cx::Union{Nothing,Real}=nothing,
                                    cy::Union{Nothing,Real}=nothing,
                                    wrt::Symbol=:radius,
                                    tol::Real=1e-12,
                                    fd_check::Bool=false,
                                    fd_h::Real=0.05,
                                    max_steps::Int=120_000,
                                    inlet::Symbol=:parabolic,
                                    gmres_tol::Real=1e-11,
                                    adjoint_tol::Real=AD_LINEAR_RES_TOL,
                                    kwargs...)
    isempty(kwargs) ||
        throw(ArgumentError("unsupported keyword(s): $(collect(keys(kwargs)))"))
    wrt === :radius ||
        throw(ArgumentError("unsupported wrt=$(wrt); supported wrt is :radius"))
    nu_f = _ad_pick_viscosity(ν, nu)

    forward = ad_forward_solve(; Nx=Nx, Ny=Ny, cx=cx, cy=cy, radius=radius,
                               u_in=u_in, nu=nu_f, inlet=inlet, rho_out=ρ_out,
                               tol=tol, max_steps=max_steps)
    forward.converged ||
        error("steady AD forward did not converge: residual=$(forward.residual), n_iter=$(forward.n_iter)")

    qoi_value = forward.Cd
    rhs = _ad_dJdf(forward.f_star, forward.q_wall, forward.u_ref,
                  forward.D, forward.Nx, forward.Ny)
    apply_GtT = v -> _ad_vjp_GtT(forward.f_star, v, forward.q_wall,
                                 forward.is_solid, forward.u_profile,
                                 forward.rho_out, forward.s_plus,
                                 forward.s_minus, forward.Nx, forward.Ny)
    adj = gmres_adjoint(apply_GtT, rhs; tol=gmres_tol, linear_tol=adjoint_tol)
    adj.converged ||
        error("steady AD adjoint did not converge: solver=$(adj.solver), linres=$(adj.linres)")

    dq_dR = dq_wall_dR_cylinder(forward.Nx, forward.Ny, forward.cx, forward.cy,
                                forward.radius; FT=Float64)
    qwall_terms = _ad_dqwall_terms(forward.f_star, adj.lambda, forward.q_wall,
                                  forward.is_solid, forward.u_profile,
                                  forward.rho_out, forward.s_plus,
                                  forward.s_minus, forward.Nx, forward.Ny,
                                  forward.u_ref, forward.D, dq_dR)
    terms = ad_assemble_radius_terms(qoi_value, forward.radius, dq_dR,
                                     qwall_terms.explicit, qwall_terms.implicit)

    fd = if fd_check
        fd_res = ad_fd_dCd_dR(forward; h=fd_h, tol=tol, max_steps=max_steps)
        (; fd_res..., relerr=abs(terms.gradient - fd_res.value) /
                          max(abs(fd_res.value), eps(Float64)))
    else
        nothing
    end

    solver = (;
        method=adj.solver,
        n_iter=adj.n_iter,
        rhohat=adj.rhohat,
        converged=adj.converged,
        linres=adj.linres,
        note=adj.note,
    )
    forward_info = (;
        n_iter=forward.n_iter,
        residual=forward.residual,
        converged=forward.converged,
        cut_links=ad_cut_link_count(forward.q_wall),
        solid_nodes=ad_solid_count(forward.is_solid),
    )

    return (;
        value=qoi_value,
        gradient=terms.gradient,
        qoi_value=qoi_value,
        solver=solver,
        terms=terms,
        n_iter=forward.n_iter,
        residual=forward.residual,
        forward=forward_info,
        fd_check=fd,
    )
end

function _ad_pick_wall_position(N::Int, L, wall_param, wall_position,
                                q_hot::Real, q_cold::Real)
    supplied = 0
    value = nothing
    for candidate in (L, wall_param, wall_position)
        if candidate !== nothing
            supplied += 1
            value = candidate
        end
    end
    supplied > 1 &&
        throw(ArgumentError("pass only one of `L`, `wall_param`, or `wall_position`"))
    return Float64(value === nothing ?
                   Float64(N - 1) + Float64(q_hot) + Float64(q_cold) :
                   value)
end

function _steady_nusselt_sensitivity(; N::Int,
                                       Ra::Real,
                                       Pr::Real,
                                       L=nothing,
                                       wall_param=nothing,
                                       wall_position=nothing,
                                       q_hot::Real=0.5,
                                       q_cold::Real=0.7,
                                       T_hot::Real=1.0,
                                       T_cold::Real=0.0,
                                       wrt::Symbol=:wall_position,
                                       tol::Real=1e-11,
                                       fd_check::Bool=false,
                                       fd_h::Real=0.01,
                                       max_steps::Int=450_000,
                                       gmres_tol::Real=1e-10,
                                       adjoint_tol::Real=AD_LINEAR_RES_TOL,
                                       gmres_restart::Int=640,
                                       gmres_max_restarts::Int=16,
                                       kwargs...)
    isempty(kwargs) ||
        throw(ArgumentError("unsupported keyword(s): $(collect(keys(kwargs)))"))
    wrt === :wall_position ||
        throw(ArgumentError("unsupported wrt=$(wrt); supported wrt is :wall_position"))
    L_f = _ad_pick_wall_position(N, L, wall_param, wall_position, q_hot, q_cold)

    forward = ad_thermal_forward_solve(; N=N, Ra=Ra, Pr=Pr, L=L_f,
                                       q_hot=q_hot, q_cold=q_cold,
                                       T_hot=T_hot, T_cold=T_cold,
                                       tol=tol, max_steps=max_steps)
    forward.converged ||
        error("steady thermal AD forward did not converge: residual=$(forward.residual), n_iter=$(forward.n_iter)")

    qoi_value = forward.Nu
    rhs = _ad_dNudw(forward.w_star, forward.q_wall, forward.params)
    apply_GtT = v -> _ad_thermal_vjp_GtT(forward.w_star, v,
                                         forward.q_wall, forward.q_wall,
                                         forward.params)
    rhohat = ad_richardson_rhohat(apply_GtT, rhs; n_iter=100)
    mass = ad_thermal_mass_gradient(forward.params)
    adj = ad_gauge_augmented_adjoint(apply_GtT, rhs, mass;
                                     tol=gmres_tol,
                                     restart=min(gmres_restart, length(rhs) + 1),
                                     max_restarts=gmres_max_restarts,
                                     linear_tol=adjoint_tol,
                                     rhohat=rhohat)
    adj.converged ||
        error("steady thermal AD adjoint did not converge: solver=$(adj.solver), linres=$(adj.linres), original_linres=$(adj.original_linres)")

    qwall_terms = _ad_thermal_dqwall_terms(forward.w_star, adj.lambda,
                                           forward.q_wall, forward.q_wall,
                                           forward.params, forward.dq_dL)
    terms = ad_assemble_wall_position_terms(qwall_terms.explicit,
                                            qwall_terms.implicit,
                                            forward.dq_dL)

    fd = if fd_check
        fd_res = ad_fd_dNu_dL(forward; h=fd_h, tol=tol, max_steps=max_steps)
        (; fd_res..., relerr=abs(terms.gradient - fd_res.value) /
                          max(abs(fd_res.value), eps(Float64)))
    else
        nothing
    end

    solver = (;
        method=adj.solver,
        n_iter=adj.n_iter,
        rhohat=adj.rhohat,
        converged=adj.converged,
        linres=adj.linres,
        original_linres=adj.original_linres,
        gauge=adj.gauge,
        eta=adj.eta,
        note=adj.note,
    )
    forward_info = (;
        n_iter=forward.n_iter,
        residual=forward.residual,
        converged=forward.converged,
        cut_links=ad_cut_link_count(forward.q_wall),
        q_hot=forward.q_hot,
        q_cold=forward.q_cold,
        L=forward.L,
    )

    return (;
        value=qoi_value,
        gradient=terms.gradient,
        qoi_value=qoi_value,
        solver=solver,
        terms=terms,
        n_iter=forward.n_iter,
        residual=forward.residual,
        forward=forward_info,
        fd_check=fd,
    )
end

# --- Viscoelastic (polymeric drag) shape sensitivity --------------------------

function _ad_ve_pick(value, alias, label::AbstractString)
    if value !== nothing && alias !== nothing
        throw(ArgumentError("pass only one of the two aliases for $label"))
    end
    v = value === nothing ? alias : value
    v === nothing && throw(ArgumentError("missing required keyword for $label"))
    return Float64(v)
end

"""
    _steady_polymer_drag_sensitivity(; ...) -> NamedTuple

Assemble d(Fx_polymer)/dR for the Oldroyd-B confined cylinder via the validated
CPU-F64 VE shape-adjoint chain: build the matched geometry + warm start → tight
forward `ad_ve_forward_solve` (fwd_tol=1e-13) → dJ/dw (`_ad_ve_dJdw`) → adjoint
(UNGAUGED for bc=:open; mass-gauged for :closed/:periodic) → analytic dG/dR
(`ad_ve_assemble_dGdR`) → `dJ/dR = ∂J/∂R|geom + λᵀ·dG/dR`. Returns the same
NamedTuple shape as the Newtonian/thermal tracks.
"""
function _steady_polymer_drag_sensitivity(; Nx::Int, Ny::Int, radius::Real,
                                            cx::Real, cy::Real,
                                            Wi::Real, beta::Real=0.5,
                                            ν_p=nothing, nu_p=nothing,
                                            ν_s=nothing, nu_s=nothing,
                                            Fx_body::Real,
                                            u_mean::Union{Nothing,Real}=nothing,
                                            n_substeps::Int=4,
                                            dt::Real=0.05,
                                            samples::Int=16,
                                            ρ_out::Real=1.0,
                                            rho_out::Union{Nothing,Real}=nothing,
                                            wrt::Symbol=:radius,
                                            fwd_tol::Real=AD_VE_FWD_TOL,
                                            max_steps::Int=AD_VE_FWD_MAX_STEPS,
                                            patience::Int=AD_VE_FWD_PATIENCE,
                                            gmres_tol::Real=1e-11,
                                            adjoint_tol::Real=AD_LINEAR_RES_TOL,
                                            gmres_restart::Int=640,
                                            gmres_max_restarts::Int=30,
                                            bc::Symbol=:open,
                                            epsR::Real=1e-5,
                                            fd_check::Bool=false,
                                            fd_h::Real=2e-5,
                                            kwargs...)
    isempty(kwargs) ||
        throw(ArgumentError("unsupported keyword(s): $(collect(keys(kwargs)))"))
    wrt === :radius ||
        throw(ArgumentError("unsupported wrt=$(wrt); supported wrt is :radius"))
    bc in (:open, :closed, :periodic) ||
        throw(ArgumentError("unsupported bc=$(bc); supported bc values are :open, :closed and :periodic"))

    nu_p_f = _ad_ve_pick(ν_p, nu_p, "polymer viscosity `ν_p`/`nu_p`")
    nu_s_f = _ad_ve_pick(ν_s, nu_s, "solvent viscosity `ν_s`/`nu_s`")
    R = Float64(radius)
    Fx = Float64(Fx_body)
    um = u_mean === nothing ? Fx : Float64(u_mean)
    # ρ_out / rho_out accepted for API parity; the coupled VE operator + forward
    # hardcode the east-pressure ZouHe outlet at rho_out=1.0 (production).
    rout = rho_out === nothing ? Float64(ρ_out) : Float64(rho_out)
    rout == 1.0 || throw(ArgumentError("ρ_out must be 1.0 for the VE polymer-drag path (got $rout)"))

    lambda = Float64(Wi)
    prefactor = nu_p_f / lambda
    s_plus, s_minus = ad_ve_trt_rates(nu_s_f)
    p = ADVECoupledParams(Nx, Ny, lambda, Float64(dt), n_substeps, prefactor,
                          nu_s_f, Fx, s_plus, s_minus)

    geom = ad_ve_build_geom(Nx, Ny, Float64(cx), Float64(cy), R;
                            samples=samples, u_mean=um)
    w0 = ad_ve_initial_state(geom.g, Nx, Ny, 0.05)
    forward = ad_ve_forward_solve(w0, geom, p; fwd_tol=fwd_tol,
                                  max_steps=max_steps, patience=patience)
    forward.converged ||
        error("steady VE AD forward did not converge: residual=$(forward.residual), n_iter=$(forward.n_iter)")

    qoi_value = ad_ve_J_fx(forward.w_star, geom.pts, geom.g, p)

    # adjoint dJ/dw, then (I - dGᵀ) λ = dJ/dw (ungauged for open BC)
    rhs = _ad_ve_dJdw(forward.w_star, geom.pts, geom.g, p)
    adj = if bc === :open
        ad_ve_ungauged_adjoint(forward.w_star, geom, p, rhs;
                               gmres_tol=gmres_tol,
                               restart=min(gmres_restart, length(rhs) + 1),
                               max_restarts=gmres_max_restarts)
    else
        apply_GtT = v -> _ad_ve_vjp_GtT(forward.w_star, v, geom.g,
                                        geom.q_wall, geom.u_profile, p)
        rhohat = ad_richardson_rhohat(apply_GtT, rhs; n_iter=100)
        mass = ad_ve_mass_gradient(p)
        ad_gauge_augmented_adjoint(apply_GtT, rhs, mass;
                                   tol=gmres_tol,
                                   restart=min(gmres_restart, length(rhs) + 1),
                                   max_restarts=gmres_max_restarts,
                                   linear_tol=adjoint_tol,
                                   rhohat=rhohat)
    end
    adj.converged ||
        error("steady VE AD adjoint did not converge: linres=$(adj.linres)")

    # analytic dG/dR (FD-free) + the explicit geometry partial ∂J/∂R|geom
    dGdR = ad_ve_assemble_dGdR(forward.w_star, geom, p;
                               cx=Float64(cx), cy=Float64(cy),
                               samples=samples)
    state_response = ad_dot_arrays(adj.lambda, dGdR)
    explicit = ad_ve_dJ_dR_geom_explicit(forward.w_star, Nx, Ny,
                                         Float64(cx), Float64(cy), R, p,
                                         Float64(epsR); samples=samples,
                                         u_mean=um)
    gradient = explicit + state_response

    terms = (;
        explicit=explicit,
        state_response=state_response,
        gradient=gradient,
        bc=bc,
    )

    fd = if fd_check
        fd_res = ad_ve_fd_dCdpoly_dR(forward.w_star, Nx, Ny, Float64(cx),
                                     Float64(cy), R, p, Float64(fd_h);
                                     samples=samples, u_mean=um,
                                     fwd_tol=fwd_tol, max_steps=max_steps,
                                     patience=patience)
        (;
            value=fd_res.value, h=Float64(fd_h),
            Jp=fd_res.Jp, Jm=fd_res.Jm,
            plus_converged=fd_res.fp.converged,
            minus_converged=fd_res.fm.converged,
            topo_fixed=(fd_res.cutp == fd_res.cutm &&
                        fd_res.solp == fd_res.solm),
            relerr=abs(gradient - fd_res.value) /
                   max(abs(fd_res.value), eps(Float64)),
        )
    else
        nothing
    end

    solver = (;
        method=:gmres,
        n_iter=adj.n_iter,
        converged=adj.converged,
        linres=adj.linres,
        original_linres=hasproperty(adj, :original_linres) ?
                         adj.original_linres : adj.linres,
        gauge=(bc === :open ? :ungauged : :mass),
    )
    forward_info = (;
        n_iter=forward.n_iter,
        residual=forward.residual,
        converged=forward.converged,
        reached_tol=forward.reached_tol,
        cut_links=count(>(0.0), geom.q_wall),
        solid_nodes=count(geom.g.is_solid),
    )

    return (;
        value=qoi_value,
        gradient=gradient,
        qoi_value=qoi_value,
        solver=solver,
        terms=terms,
        n_iter=forward.n_iter,
        residual=forward.residual,
        forward=forward_info,
        fd_check=fd,
    )
end

# --- ν-field VJP seams (M-P2c-1) ---
# `_ad_pvjp_nufield` : computes dL/dν_j for each row j = λᵀ (∂G/∂ν_j) via Enzyme Reverse
#                       over ad_step_nufield! with Duplicated(nu_field, dnu).
#                       Returns Vector{Float64} of length Ny. Impl in ext/KrakeNNADExt.jl.
# `_ad_vjp_GtT_nufield` : state VJP for field ν — same as _ad_vjp_GtT but
#                          differentiates ad_step_nufield! with Const(nu_field).
#                          Returns df (cotangent wrt f_star). Impl in ext/KrakeNNADExt.jl.
function _ad_pvjp_nufield(args...)
    error(_AD_ENZYME_LOAD_ERROR)
end

function _ad_vjp_GtT_nufield(args...)
    error(_AD_ENZYME_LOAD_ERROR)
end

# --- Optim.jl L-BFGS field-fit seam (M-P2c-2) ---
# `_fit_lbfgs` : delegates `fit(...; method=:lbfgs, ...)` to the Fminbox(LBFGS()) driver
#                in ext/KrakeNNOptimExt.jl. Without Optim loaded, raises a documented error.
const _AD_OPTIM_LOAD_ERROR = "Load Optim to enable L-BFGS: `using " * "Optim`"

function _fit_lbfgs(args...; kwargs...)
    error(_AD_OPTIM_LOAD_ERROR)
end
