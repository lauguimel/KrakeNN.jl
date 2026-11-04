# Finite-difference cross-check of the viscoelastic polymer-drag shape adjoint (#41).
#
# test/ad/test_ad_ve_sensitivity.jl checks the one-step VJP against the
# one-step JVP (transpose identity) and the operator-level adjoint identity,
# but never compares the assembled gradient dJ/dθ with a finite difference of
# the converged primal QoI on the fast case (its FD gate is env-gated behind
# KRAKEN_AD_VE_FULL=1 and only covers the radius). All 19 of its assertions
# pass under the Enzyme miscompile of #41 (wrong gradient on arm64 macOS under
# --check-bounds=yes): they cannot tell a right gradient from a wrong one.
# This file can. Cost: 52 s on Julia 1.11.9, 96 s on 1.12.7 (CI ubuntu x64,
# run 35621077285, 2026-09-21). Enzyme-driven: runs in the AD tier, and on CI
# in its own step with --check-bounds=auto like the other Enzyme files.
#
# This file takes the FAST 24x24 cut-cylinder case of that suite, solves the
# adjoint once (lambda from (I - dG^T) lambda = dJ/dw, with dG^T from Enzyme
# reverse mode over ad_ve_coupled_step!), and compares, for four design
# directions, the adjoint gradient
#     dJ/dθ = ∂J/∂θ + lambda' * ∂G/∂θ
# with a central finite difference of J(θ) where the forward is reconverged at
# θ ± h to fwd_tol = 1e-13. The step is checked by three-point Richardson
# (h, 2h, 4h): fd_err = |D(h) - D(2h)| / 3 estimates the error of D(h), and
# fd_ratio = |D(2h) - D(4h)| / |D(h) - D(2h)| is ~4 when truncation dominates
# (a value far from 4 means the FD is noise-limited at this step).
#
# Directions:
#   1  R       cylinder radius: the production chain (explicit ∂J/∂R by frozen-
#              state FD of the QoI over the geometry, plus lambda' * dG/dR with
#              the analytic geometry seed through an Enzyme forward JVP). Also
#              compared with the public steady_shape_sensitivity(qoi=:polymer_drag).
#   2  lambda  relaxation time (prefactor nu_p/lambda follows), fixed geometry.
#   3  Fx      body force (fixed inlet profile), fixed geometry.
#   4  nu_s    solvent viscosity (TRT rates follow), fixed geometry.
# For 2-4, ∂G/∂θ and ∂J/∂θ are Enzyme FORWARD mode over the step / QoI with the
# parameter as a Duplicated scalar (forward mode is exact in every variant of
# #41); ∂G/∂θ is additionally cross-checked by a one-step central FD. The only
# reverse-mode ingredient in all four directions is lambda.
#
# Output, one machine-readable line per direction:
#   === ve_fd dir=<k> name=<θ> adj=<v> fd=<v> rel=<v> fd_err=<v> fd_ratio=<v>
# plus the C0 transpose-identity residual of the existing suite and the
# environment (Julia, arch, check-bounds mode, Enzyme version).

using Test, Kraken, Enzyme, LinearAlgebra, Printf

module VEFDCheck

using Test, Kraken, Enzyme, LinearAlgebra, Printf
const K = Kraken

_relerr(a, b) = abs(a - b) / max(abs(b), eps(Float64))

const FAST = (; Nx=24, Ny=24, cx=12.35, cy=11.65, R=5.13,
              Wi=0.5, beta=0.5, nu_p=0.02, nu_s=0.08, Fx_body=2e-4, samples=16)

const FWD_TOL = 1e-13

_fwd_mode() = isdefined(Enzyme, :set_runtime_activity) ?
              Enzyme.set_runtime_activity(Enzyme.Forward) : Enzyme.Forward

# θ -> ADVECoupledParams.  which: 2 = lambda, 3 = Fx_body, 4 = nu_s.
function _params(c, theta, which::Int)
    lambda = which == 2 ? theta : c.Wi
    Fx     = which == 3 ? theta : c.Fx_body
    nu_s   = which == 4 ? theta : c.nu_s
    pref = c.nu_p / lambda
    s_plus, s_minus = K.ad_ve_trt_rates(nu_s)
    return K.ADVECoupledParams(c.Nx, c.Ny, lambda, 0.05, 4, pref, nu_s,
                               Fx, s_plus, s_minus)
end

_theta0(c, which) = which == 2 ? c.Wi : which == 3 ? c.Fx_body : c.nu_s
_name(which) = which == 1 ? "R" : which == 2 ? "lambda" : which == 3 ? "Fx" : "nu_s"

function _build(c)
    p = _params(c, 0.0, 0)          # which=0 -> all base values
    geom = K.ad_ve_build_geom(c.Nx, c.Ny, c.cx, c.cy, c.R;
                              samples=c.samples, u_mean=c.Fx_body)
    return p, geom
end

function _forward(w0, geom, p)
    return K.ad_ve_forward_solve(w0, geom, p; fwd_tol=FWD_TOL)
end

# --- the reverse-mode ingredient: adjoint lambda --------------------------------
function _adjoint(w_star, geom, p)
    dJdw = K._ad_ve_dJdw(w_star, geom.pts, geom.g, p)
    adj = K.ad_ve_ungauged_adjoint(w_star, geom, p, dJdw; gmres_tol=1e-11)
    return dJdw, adj
end

# C0 transpose identity of the existing suite (same seeds).
function _jvp(w_star, u, geom, p)
    out = zeros(Float64, length(w_star)); dout = zeros(Float64, length(w_star))
    Enzyme.autodiff(_fwd_mode(), K.ad_ve_coupled_step!,
                    Enzyme.Duplicated(out, dout),
                    Enzyme.Duplicated(copy(w_star), copy(u)),
                    Enzyme.Const(geom.g), Enzyme.Const(geom.q_wall),
                    Enzyme.Const(p), Enzyme.Const(geom.u_profile),
                    Enzyme.Const(1.0), Enzyme.Const(nothing))
    return dout
end

function _transpose_residual(w_star, geom, p)
    n = length(w_star)
    u = zeros(Float64, n); v = zeros(Float64, n)
    @inbounds for idx in 1:n
        u[idx] = sin(0.137 * idx + 0.3)
        v[idx] = cos(0.211 * idx + 0.7)
    end
    u ./= norm(u); v ./= norm(v)
    Ju = _jvp(w_star, u, geom, p)
    Jtv = K._ad_ve_vjp_GtT(w_star, v, geom.g, geom.q_wall, geom.u_profile, p)
    return _relerr(dot(v, Ju), dot(Jtv, u))
end

# --- forward-mode ∂G/∂θ and ∂J/∂θ (directions 2-4) ------------------------------
function _step_theta!(out, w, theta, which, g, qw, up, c)
    p = _params(c, theta, which)
    K.ad_ve_coupled_step!(out, w, g, qw, p, up, 1.0, nothing)
    return nothing
end

function _J_theta(w, theta, which, pts, g, c)
    return K.ad_ve_J_fx(w, pts, g, _params(c, theta, which))
end

function _dG_dtheta_fwd(w_star, geom, c, which)
    n = length(w_star)
    out = zeros(Float64, n); dout = zeros(Float64, n)
    Enzyme.autodiff(_fwd_mode(), _step_theta!,
                    Enzyme.Duplicated(out, dout),
                    Enzyme.Duplicated(copy(w_star), zeros(Float64, n)),
                    Enzyme.Duplicated(_theta0(c, which), 1.0),
                    Enzyme.Const(which), Enzyme.Const(geom.g),
                    Enzyme.Const(geom.q_wall), Enzyme.Const(geom.u_profile),
                    Enzyme.Const(c))
    return dout
end

function _dG_dtheta_fd(w_star, geom, c, which)
    th = _theta0(c, which); h = 1e-4 * abs(th)
    n = length(w_star)
    op = zeros(Float64, n); om = zeros(Float64, n)
    _step_theta!(op, copy(w_star), th + h, which, geom.g, geom.q_wall, geom.u_profile, c)
    _step_theta!(om, copy(w_star), th - h, which, geom.g, geom.q_wall, geom.u_profile, c)
    return (op .- om) ./ (2h)
end

function _dJ_dtheta_fwd(w_star, geom, c, which)
    r = Enzyme.autodiff(_fwd_mode(), _J_theta,
                        Enzyme.Const(w_star),
                        Enzyme.Duplicated(_theta0(c, which), 1.0),
                        Enzyme.Const(which), Enzyme.Const(geom.pts),
                        Enzyme.Const(geom.g), Enzyme.Const(c))
    return r[1]
end

# --- primal QoI at a perturbed design (forward reconverged from w_star) -----------
function _J_primal_R(w_star, c, Rh)
    gh = K.ad_ve_build_geom(c.Nx, c.Ny, c.cx, c.cy, Rh; samples=c.samples, u_mean=c.Fx_body)
    p = _params(c, 0.0, 0)
    fwd = _forward(w_star, gh, p)
    J = K.ad_ve_J_fx(fwd.w_star, gh.pts, gh.g, p)
    return (; J, fwd, cut=count(>(0.0), gh.q_wall), sol=count(gh.g.is_solid))
end

function _J_primal_theta(w_star, geom, c, which, th)
    p = _params(c, th, which)
    fwd = _forward(w_star, geom, p)
    J = K.ad_ve_J_fx(fwd.w_star, geom.pts, geom.g, p)
    return (; J, fwd, cut=count(>(0.0), geom.q_wall), sol=count(geom.g.is_solid))
end

# Three-point Richardson check of a central FD.  Jfun(δ) returns the primal record
# at design + δ.  Returns D(h), the error estimate |D(h)-D(2h)|/3, the ratio
# |D(2h)-D(4h)|/|D(h)-D(2h)|, the extrapolated value and convergence flags.
function _richardson(Jfun, h)
    D = Float64[]; ok = true; topo = true; base_cut = -1; base_sol = -1
    for hk in (h, 2h, 4h)
        rp = Jfun(+hk); rm = Jfun(-hk)
        ok &= rp.fwd.reached_tol && rm.fwd.reached_tol
        if base_cut < 0
            base_cut = rp.cut; base_sol = rp.sol
        end
        topo &= rp.cut == base_cut && rm.cut == base_cut &&
                rp.sol == base_sol && rm.sol == base_sol
        push!(D, (rp.J - rm.J) / (2hk))
    end
    d12 = abs(D[1] - D[2]); d24 = abs(D[2] - D[3])
    err = d12 / 3
    ratio = d24 / max(d12, eps(Float64))
    extrap = D[1] + (D[1] - D[2]) / 3
    return (; fd=D[1], D2=D[2], D4=D[3], err, ratio, extrap, reached_tol=ok, topo)
end

function run()
    c = FAST
    jl = Base.JLOptions()
    cb = jl.check_bounds == 0 ? "auto" : jl.check_bounds == 1 ? "yes" : "no"
    enz = pkgversion(Enzyme)
    println("=== ve_fd env julia=$(VERSION) arch=$(Sys.MACHINE) check_bounds=$(cb) enzyme=$(enz)")
    flush(stdout)

    t0 = time()
    p, geom = _build(c)
    w0 = K.ad_ve_initial_state(geom.g, c.Nx, c.Ny, 0.05)
    fwd = _forward(w0, geom, p)
    w_star = fwd.w_star
    J0 = K.ad_ve_J_fx(w_star, geom.pts, geom.g, p)
    println("=== ve_fd forward n_iter=$(fwd.n_iter) residual=$(fwd.residual) reached_tol=$(fwd.reached_tol) J0=$(J0) cut=$(count(>(0.0), geom.q_wall)) solid=$(count(geom.g.is_solid)) seconds=$(round(time()-t0; digits=1))")
    flush(stdout)

    # transpose identity (C0 of the existing suite)
    t1 = time()
    trel = _transpose_residual(w_star, geom, p)
    println("=== ve_fd transpose_rel=$(trel) seconds=$(round(time()-t1; digits=1))")
    flush(stdout)

    # adjoint (reverse mode inside GMRES)
    t2 = time()
    _, adj = _adjoint(w_star, geom, p)
    println("=== ve_fd adjoint converged=$(adj.converged) n_iter=$(adj.n_iter) linres=$(adj.original_linres) norm_lambda=$(norm(adj.lambda)) seconds=$(round(time()-t2; digits=1))")
    flush(stdout)
    lam = adj.lambda

    results = NamedTuple[]

    # --- direction 1: radius, the production chain -------------------------------
    t3 = time()
    dGdR = K.ad_ve_assemble_dGdR(w_star, geom, p; cx=c.cx, cy=c.cy, samples=c.samples)
    state_response = dot(lam, dGdR)
    explicit = K.ad_ve_dJ_dR_geom_explicit(w_star, c.Nx, c.Ny, c.cx, c.cy, c.R, p, 1e-5;
                                           samples=c.samples, u_mean=c.Fx_body)
    adj_R = explicit + state_response
    rich = _richardson(δ -> _J_primal_R(w_star, c, c.R + δ), 2e-5)
    rel = _relerr(adj_R, rich.fd)
    @printf("=== ve_fd dir=1 name=R adj=%.10e fd=%.10e rel=%.3e fd_err=%.3e fd_ratio=%.2f fd_extrap=%.10e explicit=%.6e state_response=%.6e reached_tol=%s topo_fixed=%s seconds=%.1f\n",
            adj_R, rich.fd, rel, rich.err, rich.ratio, rich.extrap, explicit,
            state_response, rich.reached_tol, rich.topo, time() - t3)
    flush(stdout)
    push!(results, (; which=1, adj=adj_R, rich..., rel))

    # public API on the same case (its own forward + adjoint), gradient only
    t4 = time()
    api = K.steady_shape_sensitivity(; qoi=:polymer_drag, wrt=:radius,
        Nx=c.Nx, Ny=c.Ny, radius=c.R, cx=c.cx, cy=c.cy, Wi=c.Wi, beta=c.beta,
        nu_p=c.nu_p, nu_s=c.nu_s, Fx_body=c.Fx_body, samples=c.samples,
        fwd_tol=FWD_TOL, fd_check=false)
    @printf("=== ve_fd api gradient=%.10e inline=%.10e rel_api_inline=%.3e rel_api_fd=%.3e converged=%s seconds=%.1f\n",
            api.gradient, adj_R, _relerr(api.gradient, adj_R),
            _relerr(api.gradient, rich.fd), api.solver.converged, time() - t4)
    flush(stdout)

    # --- directions 2-4: parameters at fixed geometry ----------------------------
    for which in (2, 3, 4)
        t5 = time()
        th = _theta0(c, which)
        dG = _dG_dtheta_fwd(w_star, geom, c, which)
        dG_fd = _dG_dtheta_fd(w_star, geom, c, which)
        dG_chk = norm(dG .- dG_fd) / max(norm(dG_fd), eps(Float64))
        dJ_explicit = _dJ_dtheta_fwd(w_star, geom, c, which)
        adj_th = dJ_explicit + dot(lam, dG)
        rich = _richardson(δ -> _J_primal_theta(w_star, geom, c, which, th + δ), 1e-3 * abs(th))
        rel = _relerr(adj_th, rich.fd)
        @printf("=== ve_fd dir=%d name=%s adj=%.10e fd=%.10e rel=%.3e fd_err=%.3e fd_ratio=%.2f fd_extrap=%.10e explicit=%.6e state_response=%.6e dG_fwd_vs_fd=%.2e reached_tol=%s topo_fixed=%s seconds=%.1f\n",
                which, _name(which), adj_th, rich.fd, rel, rich.err, rich.ratio,
                rich.extrap, dJ_explicit, dot(lam, dG), dG_chk, rich.reached_tol,
                rich.topo, time() - t5)
        flush(stdout)
        push!(results, (; which, adj=adj_th, rich..., rel))
    end

    println("=== ve_fd total_seconds=$(round(time()-t0; digits=1))")
    return (; trel, adj, results, api)
end

end # module

# Gates. Measured 2026-09-21 with Enzyme 0.13.204 on ubuntu x64 (Julia 1.11.9
# and 1.12.7, --check-bounds=auto, run 35621077285) and macOS arm64 (Julia
# 1.12.5, both bounds modes), identical to every printed digit:
#   R       rel 9.2e-4  (FD noise-limited: Richardson ratio 0.01; the gap is the
#                        known accuracy of the analytic geometry seed, C2 of the
#                        sibling suite, amplified by a ~20x cancellation)
#   lambda  rel 9.9e-7,  Fx rel 1.4e-7,  nu_s rel 9.1e-7
#   After the TRT Guo source fix (#67), macOS arm64, Julia 1.11.9, 2026-10-08:
#   R rel 1.05e-3; lambda 8.7e-7, Fx 1.2e-7, nu_s 1.17e-6 (gates unchanged)
#   transpose residual 8.5e-16 / 6.4e-16;  API vs inline gradient 1.9e-15 / 6.4e-16
# The miscompiled reduced construct of #41 gives a 2.3e-2 gradient error or NaN.
# Directions 2-4 depend on the reverse-mode adjoint only through lambda, so
# their 1e-4 gate (100x the measured error) is the detector; the radius gate
# 5e-3 (5x) covers the production chain end to end.
const GATE_REL_R = 5e-3
const GATE_REL_PARAM = 1e-4

@testset "VE polymer-drag adjoint vs central FD (#41)" begin
    @test Base.get_extension(Kraken, :KrakenADExt) !== nothing
    r = VEFDCheck.run()
    @test r.trel < 1e-10
    @test r.adj.converged
    @test r.api.solver.converged
    @test abs(r.api.gradient - r.results[1].adj) / abs(r.results[1].adj) < 1e-12
    for res in r.results
        @test res.reached_tol
        @test res.topo
        @test isfinite(res.adj) && isfinite(res.fd)
        @test res.rel < (res.which == 1 ? GATE_REL_R : GATE_REL_PARAM)
    end
end
