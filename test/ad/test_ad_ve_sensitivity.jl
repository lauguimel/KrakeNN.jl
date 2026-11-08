# VE shape-adjoint AD test suite (steady_shape_sensitivity qoi=:polymer_drag).
# Mirrors test/ad/test_ad_sensitivity.jl. Runs only when Enzyme is loadable.
#
# Ladder (the validated C0-C3 chain from bench/scratch/ve_ad_c3_*.jl):
#   C0  one-step VJP == JVPᵀ to FP floor (the tapeable ad_ve_coupled_step!)
#   C1  the ROUTE-1 operator-consistent adjoint identity λᵀ(I-J)u == dJ/dw·u
#       (cancellation-IMMUNE, fast; no 1e-13 R±h forwards needed)
#   C2  analytic d(geom)/dR chain: per-field face fractions vs clean central-FD
#       (A1) + λᵀ·dG/dR vs frozen-solid one-step FD (A2)
#   anti-drift  inline QoI Fx == production compute_polymeric_drag_2d Fx (Δ=0)
#   .krk  the Sensitivity{qoi=polymer_drag} dispatch == direct API (bit-exact)
#
# FULL gate behind KRAKEN_AD_VE_FULL=1: the 32×32 case, public-API dJ/dR vs
# central-FD <1% (forward reconverged to fwd_tol=1e-13 at R±h) PLUS the tol-sweep
# guard (1e-11 forward poisons the FD reference; 1e-13 recovers <1%) — documents
# why fwd_tol=1e-13 is mandatory (the net d(Fx)/dR is a ~20× cancellation).

using Test, KrakeNN, Enzyme

module KrakenADVESensitivityTests

using Test
using KrakeNN
using Enzyme
using LinearAlgebra

const K = KrakeNN

_relerr(a, b) = abs(a - b) / max(abs(b), eps(Float64))

function _dot_arrays(a, b)
    s = 0.0
    @inbounds for idx in eachindex(a, b)
        s += a[idx] * b[idx]
    end
    return s
end

# --- shared case parameters -------------------------------------------------
# Fast off-lattice (no q_w=0.5 kinks) cut cylinder for C0/C1/C2; converges to
# 1e-13 in ~7k iters (~3s CPU-F64). The full C3 case is the 32×32 below.
const FAST = (; Nx=24, Ny=24, cx=12.35, cy=11.65, R=5.13,
              Wi=0.5, beta=0.5, nu_p=0.02, nu_s=0.08, Fx_body=2e-4, samples=16)
const FULL = (; Nx=32, Ny=32, cx=16.350, cy=15.650, R=8.130,
              Wi=0.5, beta=0.5, nu_p=0.02, nu_s=0.08, Fx_body=2e-4, samples=16)

function _build(c)
    lambda = c.Wi
    pref = c.nu_p / lambda
    s_plus, s_minus = K.ad_ve_trt_rates(c.nu_s)
    p = K.ADVECoupledParams(c.Nx, c.Ny, lambda, 0.05, 4, pref, c.nu_s,
                            c.Fx_body, s_plus, s_minus)
    geom = K.ad_ve_build_geom(c.Nx, c.Ny, c.cx, c.cy, c.R;
                              samples=c.samples, u_mean=c.Fx_body)
    return p, geom
end

function _forward(c, p, geom)
    w0 = K.ad_ve_initial_state(geom.g, c.Nx, c.Ny, 0.05)
    return K.ad_ve_forward_solve(w0, geom, p; fwd_tol=1e-13)
end

# exact state-JVP via Enzyme.Forward over the tapeable coupled step (test-side).
function _jvp(w_star, u, geom, p)
    out_len = length(w_star)
    out = zeros(Float64, out_len)
    dout = zeros(Float64, out_len)
    mode = isdefined(Enzyme, :set_runtime_activity) ?
           Enzyme.set_runtime_activity(Enzyme.Forward) : Enzyme.Forward
    Enzyme.autodiff(mode, K.ad_ve_coupled_step!,
                    Enzyme.Duplicated(out, dout),
                    Enzyme.Duplicated(copy(w_star), copy(u)),
                    Enzyme.Const(geom.g), Enzyme.Const(geom.q_wall),
                    Enzyme.Const(p), Enzyme.Const(geom.u_profile),
                    Enzyme.Const(1.0), Enzyme.Const(nothing))
    return dout
end

# ---------------------------------------------------------------------------
# C0: VJP == JVPᵀ to FP floor (exact, no FD).
function _run_c0()
    c = FAST
    p, geom = _build(c)
    fwd = _forward(c, p, geom)
    w_star = fwd.w_star
    out_len = length(w_star)
    u = zeros(Float64, out_len); v = zeros(Float64, out_len)
    @inbounds for idx in 1:out_len
        u[idx] = sin(0.137 * idx + 0.3)
        v[idx] = cos(0.211 * idx + 0.7)
    end
    u ./= norm(u); v ./= norm(v)
    Ju = _jvp(w_star, u, geom, p)
    Jtv = K._ad_ve_vjp_GtT(w_star, v, geom.g, geom.q_wall, geom.u_profile, p)
    rel = _relerr(_dot_arrays(v, Ju), _dot_arrays(Jtv, u))
    return (; c.Nx, c.Ny, fwd, transpose_rel=rel,
            cut=count(>(0.0), geom.q_wall), npts=length(geom.pts))
end

# C1: ROUTE-1 operator-consistent adjoint identity (cancellation-immune).
function _run_c1()
    c = FAST
    p, geom = _build(c)
    fwd = _forward(c, p, geom)
    w_star = fwd.w_star
    out_len = length(w_star)
    dJdw = K._ad_ve_dJdw(w_star, geom.pts, geom.g, p)
    adj = K.ad_ve_ungauged_adjoint(w_star, geom, p, dJdw; gmres_tol=1e-11)
    u = zeros(Float64, out_len)
    @inbounds for idx in 1:out_len
        u[idx] = sin(0.091 * idx + 1.1)
    end
    u ./= norm(u)
    ImJ_u = u .- _jvp(w_star, u, geom, p)
    sr_id = _dot_arrays(adj.lambda, ImJ_u)   # λᵀ(I-J)u
    target = _dot_arrays(dJdw, u)            # dJ/dw·u
    return (; c.Nx, c.Ny, fwd, adj, identity_rel=_relerr(sr_id, target))
end

# C2: analytic geometry chain.  A1 per-field smooth-face FD; A2 operator-level
# λᵀ·dG/dR vs frozen-solid one-step FD (cell_fraction is validated ONLY through
# the operator — its field-level FD is staircase noise, by construction).
function _run_c2()
    c = FAST
    p, geom = _build(c)
    fwd = _forward(c, p, geom)
    w_star = fwd.w_star
    base_solid = copy(geom.g.is_solid)

    # A1: smooth face-fraction fields vs clean central-FD (frozen topology).
    dg = K.ad_ve_build_dcircle_geom_dR(c.Nx, c.Ny, c.cx, c.cy, c.R, geom.g;
                                       samples=c.samples)
    h = 1e-4
    gp = K.ad_ve_build_geom(c.Nx, c.Ny, c.cx, c.cy, c.R + h;
                            samples=c.samples, u_mean=c.Fx_body)
    gm = K.ad_ve_build_geom(c.Nx, c.Ny, c.cx, c.cy, c.R - h;
                            samples=c.samples, u_mean=c.Fx_body)
    topo = count(>(0.0), gp.q_wall) == count(>(0.0), geom.q_wall) &&
           count(>(0.0), gm.q_wall) == count(>(0.0), geom.q_wall) &&
           gp.g.is_solid == base_solid && gm.g.is_solid == base_solid
    a1_rel = -Inf
    for fld in (:west_fraction, :east_fraction, :south_fraction, :north_fraction)
        fd = (getproperty(gp.g, fld) .- getproperty(gm.g, fld)) ./ (2h)
        an = getproperty(dg, fld)
        a1_rel = max(a1_rel, norm(an .- fd) / max(norm(fd), eps(Float64)))
    end

    # A2: operator-level λᵀ·dG/dR (analytic, FD-free) vs frozen-solid one-step FD.
    dJdw = K._ad_ve_dJdw(w_star, geom.pts, geom.g, p)
    adj = K.ad_ve_ungauged_adjoint(w_star, geom, p, dJdw; gmres_tol=1e-11)
    dGdR = K.ad_ve_assemble_dGdR(w_star, geom, p; cx=c.cx, cy=c.cy,
                                 samples=c.samples)
    sr_an = _dot_arrays(adj.lambda, dGdR)

    out_len = length(w_star)
    function _G_at(g_, q_)
        o = zeros(Float64, out_len)
        K.ad_ve_coupled_step!(o, w_star, g_, q_, p, geom.u_profile, 1.0, nothing)
        return o
    end
    function _frozen(Rh)
        gf = K.ad_ve_build_circle_geom(c.Nx, c.Ny, c.cx, c.cy, Rh;
                                       samples=c.samples)
        g = K.ad_ve_build_matched_geom(gf, base_solid)
        qw, _ = K.precompute_q_wall_cylinder(c.Nx, c.Ny, c.cx - 0.5, c.cy - 0.5,
                                             Rh; FT=Float64)
        return g, qw
    end
    hf = 2e-5
    gpf, qpf = _frozen(c.R + hf); gmf, qmf = _frozen(c.R - hf)
    dGdR_fd = (_G_at(gpf, qpf) .- _G_at(gmf, qmf)) ./ (2hf)
    sr_fd = _dot_arrays(adj.lambda, dGdR_fd)

    return (; c.Nx, c.Ny, fwd, adj, topo_fixed=topo, a1_rel,
            a2_rel=_relerr(sr_an, sr_fd))
end

# anti-drift: inline QoI Fx == production compute_polymeric_drag_2d Fx (Δ ≈ 0).
function _run_antidrift()
    c = FAST
    p, geom = _build(c)
    fwd = _forward(c, p, geom)
    ad = K.ad_ve_antidrift_delta(fwd.w_star, geom, p; cx=c.cx, cy=c.cy,
                                 radius=c.R)
    return (; c.Nx, c.Ny, fwd, delta=ad.delta, ad.prod_Fx, ad.inline_Fx)
end

# .krk dispatch (fast 24×24 fixture) == direct API call (bit-exact, same kwargs).
function _run_krk()
    path = normpath(joinpath(@__DIR__, "sensitivity_cylinder_polymer_fast.krk"))
    setup = K.load_kraken(path)
    krk = K.run_krk_sensitivity(setup)
    kwargs = K._krk_sensitivity_polymer_drag_kwargs(setup, setup.sensitivity)
    direct = K.steady_shape_sensitivity(; kwargs...)
    return (; path, setup, krk, direct,
            rel=_relerr(krk.gradient, direct.gradient))
end

# FULL C3 gate (env-gated): public-API dJ/dR vs central-FD <1% + tol-sweep guard.
function _run_c3_full()
    c = FULL
    base = K.steady_shape_sensitivity(; qoi=:polymer_drag, wrt=:radius,
        Nx=c.Nx, Ny=c.Ny, radius=c.R, cx=c.cx, cy=c.cy, Wi=c.Wi, beta=c.beta,
        nu_p=c.nu_p, nu_s=c.nu_s, Fx_body=c.Fx_body, samples=c.samples,
        fwd_tol=1e-13, fd_check=true, fd_h=2e-5)
    return base
end

function _run_tol_sweep()
    c = FULL
    poisoned = K.steady_shape_sensitivity(; qoi=:polymer_drag, wrt=:radius,
        Nx=c.Nx, Ny=c.Ny, radius=c.R, cx=c.cx, cy=c.cy, Wi=c.Wi, beta=c.beta,
        nu_p=c.nu_p, nu_s=c.nu_s, Fx_body=c.Fx_body, samples=c.samples,
        fwd_tol=1e-11, fd_check=true, fd_h=2e-5)
    correct = K.steady_shape_sensitivity(; qoi=:polymer_drag, wrt=:radius,
        Nx=c.Nx, Ny=c.Ny, radius=c.R, cx=c.cx, cy=c.cy, Wi=c.Wi, beta=c.beta,
        nu_p=c.nu_p, nu_s=c.nu_s, Fx_body=c.Fx_body, samples=c.samples,
        fwd_tol=1e-13, fd_check=true, fd_h=2e-5)
    return (; poisoned_rel=poisoned.fd_check.relerr,
            correct_rel=correct.fd_check.relerr)
end

end # module KrakenADVESensitivityTests

using .KrakenADVESensitivityTests: _run_c0, _run_c1, _run_c2, _run_antidrift,
                                   _run_krk, _run_c3_full, _run_tol_sweep

@testset "AD VE shape-adjoint (polymer drag)" begin
    @test Base.get_extension(KrakeNN, :KrakeNNADExt) !== nothing

    @testset "C0 coupled one-step VJP == JVPᵀ" begin
        elapsed = @elapsed c0 = _run_c0()
        @test c0.fwd.converged
        @test c0.fwd.reached_tol
        @test c0.transpose_rel < 1e-10
        @info "AD VE C0" grid="$(c0.Nx)x$(c0.Ny)" cut=c0.cut npts=c0.npts forward_iter=c0.fwd.n_iter transpose_rel=c0.transpose_rel seconds=elapsed
    end

    @testset "C1 ROUTE-1 adjoint identity" begin
        elapsed = @elapsed c1 = _run_c1()
        @test c1.fwd.converged
        @test c1.adj.converged
        @test c1.adj.original_linres < 1e-8
        @test c1.identity_rel < 1e-3
        @info "AD VE C1" grid="$(c1.Nx)x$(c1.Ny)" adj_iter=c1.adj.n_iter linres=c1.adj.original_linres identity_rel=c1.identity_rel seconds=elapsed
    end

    @testset "C2 analytic geometry chain" begin
        elapsed = @elapsed c2 = _run_c2()
        @test c2.fwd.converged
        @test c2.adj.converged
        @test c2.topo_fixed
        @test c2.a1_rel < 1e-3      # smooth face fractions vs clean FD
        @test c2.a2_rel < 1e-3      # λᵀ·dG/dR vs frozen-solid one-step FD
        @info "AD VE C2" grid="$(c2.Nx)x$(c2.Ny)" a1_rel=c2.a1_rel a2_rel=c2.a2_rel seconds=elapsed
    end

    @testset "anti-drift QoI bit mirror" begin
        elapsed = @elapsed g = _run_antidrift()
        # BIT-MIRROR GUARD: if a change flips this RED, the inline ad_ve_J_fx has
        # drifted from the production compute_polymeric_drag_2d Fx and must re-sync.
        @test g.fwd.converged
        @test g.delta < 1e-12
        @info "AD VE anti-drift" grid="$(g.Nx)x$(g.Ny)" prod_Fx=g.prod_Fx inline_Fx=g.inline_Fx delta=g.delta seconds=elapsed
    end

    @testset "krk Sensitivity dispatch" begin
        elapsed = @elapsed k = _run_krk()
        @test k.setup.sensitivity == (; qoi=:polymer_drag, wrt=:radius)
        @test k.krk.solver.converged
        @test k.direct.solver.converged
        @test k.rel < 1e-10
        @info "AD VE krk" krk_gradient=k.krk.gradient api_gradient=k.direct.gradient rel=k.rel seconds=elapsed
    end

    if get(ENV, "KRAKEN_AD_VE_FULL", "0") == "1"
        @testset "C3 full net dJ/dR vs FD (<1%)" begin
            elapsed = @elapsed c3 = _run_c3_full()
            @test c3.solver.converged
            @test c3.forward.reached_tol
            @test c3.fd_check.plus_converged
            @test c3.fd_check.minus_converged
            @test c3.fd_check.topo_fixed
            @test c3.fd_check.relerr < 1e-2
            @info "AD VE C3 full" gradient=c3.gradient fd=c3.fd_check.value relerr=c3.fd_check.relerr forward_iter=c3.forward.n_iter seconds=elapsed
        end

        @testset "C3 tol-sweep guard (1e-13 mandatory)" begin
            elapsed = @elapsed ts = _run_tol_sweep()
            # 1e-11 forward POISONS the FD reference (net is a ~20× cancellation);
            # 1e-13 recovers <1%. Documents why fwd_tol=1e-13 is the default.
            @test ts.poisoned_rel > 1e-2     # loose forward -> wrong FD (> 1%)
            @test ts.correct_rel < 1e-2      # tight forward  -> right FD (< 1%)
            @info "AD VE tol-sweep" poisoned_rel_1e11=ts.poisoned_rel correct_rel_1e13=ts.correct_rel seconds=elapsed
        end
    else
        @info "Skipping VE full C3 net-gradient + tol-sweep gate (set KRAKEN_AD_VE_FULL=1 to run)"
    end
end
