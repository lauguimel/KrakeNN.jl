module KrakeNNADExt

using KrakeNN
using Enzyme

import KrakeNN: _ad_dJdf, _ad_vjp_GtT, _ad_dqwall_terms
import KrakeNN: _ad_dNudw, _ad_thermal_vjp_GtT, _ad_thermal_dqwall_terms
import KrakeNN: _ad_ve_dJdw, _ad_ve_vjp_GtT, _ad_ve_dGdR_jvp
import KrakeNN: _ad_pvjp_nu, ad_step_nu!
import KrakeNN: _ad_pvjp_nufield, _ad_vjp_GtT_nufield, ad_step_nufield!

_short_error(err) = first(sprint(showerror, err), min(360, lastindex(sprint(showerror, err))))
_short_string(msg) = first(msg, min(360, lastindex(msg)))

function _ad_dJdf(f, q_wall, u_ref, D, Nx::Int, Ny::Int)
    djdf = zeros(Float64, size(f))
    Enzyme.autodiff(Enzyme.Reverse, KrakeNN.cd_pure, Enzyme.Active,
                    Enzyme.Duplicated(copy(f), djdf),
                    Enzyme.Const(q_wall), Enzyme.Const(u_ref),
                    Enzyme.Const(D), Enzyme.Const(Nx), Enzyme.Const(Ny))
    return djdf
end

function _ad_vjp_GtT(f_star, v, q_wall, is_solid, u_profile, rho_out,
                     s_plus, s_minus, Nx::Int, Ny::Int)
    out = zeros(Float64, size(f_star))
    dout = copy(v)
    df = zeros(Float64, size(f_star))
    Enzyme.autodiff(Enzyme.Reverse, KrakeNN.ad_step!,
                    Enzyme.Duplicated(out, dout),
                    Enzyme.Duplicated(copy(f_star), df),
                    Enzyme.Const(q_wall), Enzyme.Const(is_solid),
                    Enzyme.Const(u_profile), Enzyme.Const(rho_out),
                    Enzyme.Const(s_plus), Enzyme.Const(s_minus),
                    Enzyme.Const(Nx), Enzyme.Const(Ny))
    return df
end

function _dCd_dqwall_taped(f, q_wall, u_ref, D, Nx::Int, Ny::Int)
    dqw = zeros(Float64, size(q_wall))
    Enzyme.autodiff(Enzyme.Reverse, KrakeNN.cd_pure, Enzyme.Active,
                    Enzyme.Const(f), Enzyme.Duplicated(copy(q_wall), dqw),
                    Enzyme.Const(u_ref), Enzyme.Const(D),
                    Enzyme.Const(Nx), Enzyme.Const(Ny))
    return dqw
end

function _lambda_dot_dG_dqwall_with_mode(mode, f_star, lambda, q_wall,
                                         is_solid, u_profile, rho_out,
                                         s_plus, s_minus, Nx::Int, Ny::Int)
    out = zeros(Float64, size(f_star))
    dout = copy(lambda)
    dqw = zeros(Float64, size(q_wall))
    Enzyme.autodiff(mode, KrakeNN.ad_step!,
                    Enzyme.Duplicated(out, dout),
                    Enzyme.Const(f_star),
                    Enzyme.Duplicated(copy(q_wall), dqw),
                    Enzyme.Const(is_solid), Enzyme.Const(u_profile),
                    Enzyme.Const(rho_out), Enzyme.Const(s_plus),
                    Enzyme.Const(s_minus), Enzyme.Const(Nx), Enzyme.Const(Ny))
    return dqw
end

function _dCd_dqwall(f, q_wall, u_ref, D, Nx::Int, Ny::Int)
    try
        dqw = _dCd_dqwall_taped(f, q_wall, u_ref, D, Nx, Ny)
        if all(isfinite, dqw) && KrakeNN.cut_cotangent_norm(dqw, q_wall) > 0.0
            return (; dqw=dqw, path="taped", error="")
        end
        return (; dqw=KrakeNN.fd_dCd_dqwall(f, q_wall, u_ref, D, Nx, Ny),
                path="Fallback A (FD-on-q_wall)",
                error="taped explicit Cd/q_wall returned nonfinite or zero cut-link cotangent")
    catch err
        return (; dqw=KrakeNN.fd_dCd_dqwall(f, q_wall, u_ref, D, Nx, Ny),
                path="Fallback A (FD-on-q_wall)", error=_short_error(err))
    end
end

function _lambda_dot_dG_dqwall(f_star, lambda, q_wall, is_solid, u_profile,
                               rho_out, s_plus, s_minus, Nx::Int, Ny::Int)
    plain_err = ""
    runtime_err = ""
    try
        dqw = _lambda_dot_dG_dqwall_with_mode(Enzyme.Reverse, f_star, lambda,
                                              q_wall, is_solid, u_profile,
                                              rho_out, s_plus, s_minus, Nx, Ny)
        if all(isfinite, dqw) && KrakeNN.cut_cotangent_norm(dqw, q_wall) > 0.0
            return (; dqw=dqw, path="taped", error="")
        end
        plain_err = "plain Reverse returned nonfinite or zero cut-link cotangent"
    catch err
        plain_err = sprint(showerror, err)
    end

    try
        if !isdefined(Enzyme, :set_runtime_activity)
            error("Enzyme.set_runtime_activity is unavailable")
        end
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
        dqw = _lambda_dot_dG_dqwall_with_mode(mode, f_star, lambda,
                                              q_wall, is_solid, u_profile,
                                              rho_out, s_plus, s_minus, Nx, Ny)
        if all(isfinite, dqw) && KrakeNN.cut_cotangent_norm(dqw, q_wall) > 0.0
            return (; dqw=dqw, path="runtime-activity", error=plain_err)
        end
        runtime_err = "runtime-activity returned nonfinite or zero cut-link cotangent"
    catch err
        runtime_err = sprint(showerror, err)
    end

    dqw = KrakeNN.fd_lambda_dot_dG_dqwall(f_star, lambda, q_wall, is_solid,
                                         u_profile, rho_out, s_plus, s_minus,
                                         Nx, Ny)
    msg = "plain Reverse: " * _short_string(plain_err) *
          " | runtime-activity: " * _short_string(runtime_err)
    return (; dqw=dqw, path="Fallback A (FD-on-q_wall)", error=msg)
end

function _ad_dqwall_terms(f_star, lambda, q_wall, is_solid, u_profile,
                          rho_out, s_plus, s_minus, Nx::Int, Ny::Int,
                          u_ref, D, dq_dR)
    explicit = _dCd_dqwall(f_star, q_wall, u_ref, D, Nx, Ny)
    implicit = _lambda_dot_dG_dqwall(f_star, lambda, q_wall, is_solid,
                                     u_profile, rho_out, s_plus, s_minus,
                                     Nx, Ny)

    explicit_q_R = KrakeNN.ad_dot_arrays(explicit.dqw, dq_dR)
    explicit_dirfd = KrakeNN.directional_fd_Cd_qwall(f_star, q_wall, dq_dR,
                                                    u_ref, D, Nx, Ny)
    explicit_rel = KrakeNN.ad_rel_delta(explicit_q_R, explicit_dirfd)
    if explicit_rel > 1e-5
        explicit = (; dqw=KrakeNN.fd_dCd_dqwall(f_star, q_wall, u_ref, D, Nx, Ny),
                    path="Fallback A (FD-on-q_wall after directional mismatch)",
                    error="directional contraction mismatch",
                    directional_fd=explicit_dirfd,
                    directional_rel=explicit_rel)
    else
        explicit = (; explicit..., directional_fd=explicit_dirfd,
                    directional_rel=explicit_rel)
    end

    implicit_R = KrakeNN.ad_dot_arrays(implicit.dqw, dq_dR)
    implicit_dirfd = KrakeNN.directional_fd_lambdaG_qwall(f_star, lambda,
                                                         q_wall, dq_dR,
                                                         is_solid, u_profile,
                                                         rho_out, s_plus,
                                                         s_minus, Nx, Ny)
    implicit_rel = KrakeNN.ad_rel_delta(implicit_R, implicit_dirfd)
    if implicit_rel > 1e-5
        implicit = (; dqw=KrakeNN.fd_lambda_dot_dG_dqwall(f_star, lambda,
                                                         q_wall, is_solid,
                                                         u_profile, rho_out,
                                                         s_plus, s_minus,
                                                         Nx, Ny),
                    path="Fallback A (FD-on-q_wall after directional mismatch)",
                    error="directional contraction mismatch",
                    directional_fd=implicit_dirfd,
                    directional_rel=implicit_rel)
    else
        implicit = (; implicit..., directional_fd=implicit_dirfd,
                    directional_rel=implicit_rel)
    end

    return (; explicit=explicit, implicit=implicit)
end

function _ad_dNudw(w_star, q_wall, p)
    dwd = zeros(Float64, length(w_star))
    Enzyme.autodiff(Enzyme.Reverse, KrakeNN.nu_pure, Enzyme.Active,
                    Enzyme.Duplicated(copy(w_star), dwd),
                    Enzyme.Const(q_wall),
                    Enzyme.Const(p))
    return dwd
end

function _ad_thermal_vjp_GtT(w_star, v, q_flow, q_therm, p)
    out = zeros(Float64, length(w_star))
    dout = copy(v)
    dw = zeros(Float64, length(w_star))
    Enzyme.autodiff(Enzyme.Reverse, KrakeNN.ad_thermal_cut_step!,
                    Enzyme.Duplicated(out, dout),
                    Enzyme.Duplicated(copy(w_star), dw),
                    Enzyme.Const(q_flow),
                    Enzyme.Const(q_therm),
                    Enzyme.Const(p))
    return dw
end

function _ad_dNu_dqwall(w_star, q_wall, p)
    dqw = zeros(Float64, size(q_wall))
    Enzyme.autodiff(Enzyme.Reverse, KrakeNN.nu_pure, Enzyme.Active,
                    Enzyme.Const(w_star),
                    Enzyme.Duplicated(copy(q_wall), dqw),
                    Enzyme.Const(p))
    return (; dqw=dqw, path="taped", error="")
end

function _lambda_dot_thermal_dG_dqwalls(w_star, lambda, q_flow, q_therm, p)
    out = zeros(Float64, length(w_star))
    dout = copy(lambda)
    dqw_flow = zeros(Float64, size(q_flow))
    dqw_therm = zeros(Float64, size(q_therm))
    Enzyme.autodiff(Enzyme.Reverse, KrakeNN.ad_thermal_cut_step!,
                    Enzyme.Duplicated(out, dout),
                    Enzyme.Const(w_star),
                    Enzyme.Duplicated(copy(q_flow), dqw_flow),
                    Enzyme.Duplicated(copy(q_therm), dqw_therm),
                    Enzyme.Const(p))
    return (; flow=dqw_flow, thermal=dqw_therm, path="taped", error="")
end

function _ad_thermal_dqwall_terms(w_star, lambda, q_flow, q_therm, p, dq_dL)
    explicit = _ad_dNu_dqwall(w_star, q_therm, p)
    implicit = _lambda_dot_thermal_dG_dqwalls(w_star, lambda, q_flow, q_therm, p)
    implicit_L = KrakeNN.ad_dot_arrays(implicit.flow, dq_dL) +
                 KrakeNN.ad_dot_arrays(implicit.thermal, dq_dL)
    dirfd = KrakeNN.directional_fd_thermal_lambdaG_qwalls(w_star, lambda,
                                                         q_flow, q_therm,
                                                         dq_dL, p)
    rel = KrakeNN.ad_rel_delta(implicit_L, dirfd)
    return (; explicit=explicit,
            implicit=(; implicit..., directional_fd=dirfd,
                      directional_rel=rel))
end

# ============================================================================
# Viscoelastic (Cd_polymer) shape-adjoint seams. Mirror of the validated scratch
# chain (ve_ad_c1.jl dJ_dw_enzyme, ve_ad_c3_matched.jl c3m_vjp, ve_ad_c3_analytic.jl
# dGdR_analytic_jvp). The tapeable plain-Julia operators live in src/ad/ad_ve_*.jl;
# `using KrakeNN` stays Enzyme-free. `ADVECoupledParams`, `ADVEEmbeddedGeom`,
# `ADVEWallPoint` are Const (the EmbeddedGeom struct of arrays / params never carry
# a state cotangent here). The geometry forward-JVP seeds the EmbeddedGeom shadow
# (`Duplicated(g, dg)`) + the q_wall shadow (`Duplicated(q_wall, dq_wall)`).
# ============================================================================

# Reverse of the QoI ad_ve_J_fx wrt the stacked state w -> dJ/dw (psi-only; the
# f-block cotangent is structurally zero since J reads only the psi block).
function _ad_ve_dJdw(w_star, pts, g, p)
    djdw = zeros(Float64, length(w_star))
    Enzyme.autodiff(Enzyme.Reverse, KrakeNN.ad_ve_J_fx, Enzyme.Active,
                    Enzyme.Duplicated(copy(w_star), djdw),
                    Enzyme.Const(pts), Enzyme.Const(g), Enzyme.Const(p))
    return djdw
end

# dG^T . v : reverse over the coupled VE one-step operator ad_ve_coupled_step!.
# set_runtime_activity(Reverse) (the scratch c3m_vjp default) handles the
# is_solid-masked branches / mixed-activity geometry struct.
function _ad_ve_vjp_GtT(w_star, v, g, q_wall, u_profile, p)
    out = zeros(Float64, length(w_star))
    dout = copy(v)
    dw = zeros(Float64, length(w_star))
    mode = isdefined(Enzyme, :set_runtime_activity) ?
           Enzyme.set_runtime_activity(Enzyme.Reverse) : Enzyme.Reverse
    Enzyme.autodiff(mode, KrakeNN.ad_ve_coupled_step!,
                    Enzyme.Duplicated(out, dout),
                    Enzyme.Duplicated(copy(w_star), dw),
                    Enzyme.Const(g), Enzyme.Const(q_wall),
                    Enzyme.Const(p), Enzyme.Const(u_profile),
                    Enzyme.Const(1.0), Enzyme.Const(nothing))
    return dw
end

# dG/dR : forward-JVP of ad_ve_coupled_step! with the state seed = 0 and the
# geometry/q_wall shadows seeded by the analytic d(geom)/dR. One forward pass.
function _ad_ve_dGdR_jvp(w_star, g, dg, q_wall, dq_wall, u_profile, p)
    out = zeros(Float64, length(w_star))
    dout = zeros(Float64, length(w_star))
    dw = zeros(Float64, length(w_star))          # geometry-only JVP: no state seed
    mode = isdefined(Enzyme, :set_runtime_activity) ?
           Enzyme.set_runtime_activity(Enzyme.Forward) : Enzyme.Forward
    Enzyme.autodiff(mode, KrakeNN.ad_ve_coupled_step!,
                    Enzyme.Duplicated(out, dout),
                    Enzyme.Duplicated(copy(w_star), dw),
                    Enzyme.Duplicated(g, dg),
                    Enzyme.Duplicated(copy(q_wall), copy(dq_wall)),
                    Enzyme.Const(p), Enzyme.Const(u_profile),
                    Enzyme.Const(1.0), Enzyme.Const(nothing))
    return dout
end

"""
    _ad_pvjp_nu(f_star, lambda, p::LBMScalarParams) -> Float64

Compute dL/dν = λᵀ (∂G/∂ν) via Enzyme Reverse over `ad_step_nu!` with `Active(ν)`.
Strategy B: differentiate the ν → (s_plus, s_minus) → G chain in one Enzyme call.
Returns a scalar — the cotangent of the loss wrt the free viscosity.
"""
function _ad_pvjp_nu(f_star, lambda, p::KrakeNN.LBMScalarParams)
    out = zeros(Float64, size(f_star))
    dout = copy(lambda)
    df = zeros(Float64, size(f_star))
    ret = Enzyme.autodiff(Enzyme.Reverse, KrakeNN.ad_step_nu!,
                          Enzyme.Duplicated(out, dout),
                          Enzyme.Duplicated(copy(f_star), df),
                          Enzyme.Const(p.q_wall),
                          Enzyme.Const(p.is_solid),
                          Enzyme.Const(p.u_profile),
                          Enzyme.Const(p.rho_out),
                          Enzyme.Active(p.ν),
                          Enzyme.Const(p.Nx), Enzyme.Const(p.Ny))
    # Enzyme versions differ between compact Active-only and per-argument
    # cotangent tuples; ν is the only Active scalar in this call.
    for cotangent in ret[1]
        cotangent isa Real && return Float64(cotangent)
    end
    error("_ad_pvjp_nu: Enzyme returned no scalar cotangent for ν")
end

"""
    _ad_pvjp_nufield(f_star, lambda, nu_field::Vector{Float64},
                     q_wall, is_solid, u_profile, rho_out, Nx::Int, Ny::Int) -> Vector{Float64}

Field VJP: dL/dν_j = λᵀ (∂G/∂ν_j) for all j simultaneously, via Enzyme Reverse
over `ad_step_nufield!` with `Duplicated(nu_field, dnu)` array shadow.
Returns `dnu` — Vector{Float64} of length Ny — the field cotangent.
"""
function _ad_pvjp_nufield(f_star, lambda, nu_field::Vector{Float64},
                          q_wall, is_solid, u_profile, rho_out, Nx::Int, Ny::Int)
    out  = zeros(Float64, size(f_star))
    dout = copy(lambda)
    df   = zeros(Float64, size(f_star))
    dnu  = zeros(Float64, length(nu_field))

    Enzyme.autodiff(Enzyme.Reverse, KrakeNN.ad_step_nufield!,
                    Enzyme.Duplicated(out, dout),
                    Enzyme.Duplicated(copy(f_star), df),
                    Enzyme.Const(q_wall),
                    Enzyme.Const(is_solid),
                    Enzyme.Const(u_profile),
                    Enzyme.Const(rho_out),
                    Enzyme.Duplicated(copy(nu_field), dnu),
                    Enzyme.Const(Nx), Enzyme.Const(Ny))
    return dnu
end

"""
    _ad_vjp_GtT_nufield(f_star, v, q_wall, is_solid, u_profile, rho_out,
                         nu_field, Nx::Int, Ny::Int) -> Array{Float64,3}

State VJP for field ν: dG^T · v differentiating `ad_step_nufield!` with
`Const(nu_field)` (nu_field NOT differentiated — this is the state adjoint, not
the parameter VJP). Returns `df` — the cotangent wrt `f_star`.
"""
function _ad_vjp_GtT_nufield(f_star, v, q_wall, is_solid, u_profile, rho_out,
                             nu_field, Nx::Int, Ny::Int)
    out  = zeros(Float64, size(f_star))
    dout = copy(v)
    df   = zeros(Float64, size(f_star))

    Enzyme.autodiff(Enzyme.Reverse, KrakeNN.ad_step_nufield!,
                    Enzyme.Duplicated(out, dout),
                    Enzyme.Duplicated(copy(f_star), df),
                    Enzyme.Const(q_wall),
                    Enzyme.Const(is_solid),
                    Enzyme.Const(u_profile),
                    Enzyme.Const(rho_out),
                    Enzyme.Const(nu_field),
                    Enzyme.Const(Nx), Enzyme.Const(Ny))
    return df
end

end # module KrakeNNADExt
