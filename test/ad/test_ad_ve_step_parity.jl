# Step equality of the LBM viscoelastic CPU copy against the production driver.
#
# src/ad/ad_ve_step.jl declares `ad_ve_coupled_step!` a step-for-step mirror of
# the production coupled step (drivers/viscoelastic_logfv_coupled_step_2d.jl).
# Until this file, no test compared the two: test/ad/test_ad_ve_sensitivity.jl
# checks the copy against itself (transpose identity, adjoint identity,
# geometry chain) and only compares the drag QoI with production on the copy's
# own state. This file runs both from the same state with the same parameters
# and compares, after 1 step and after each of 10 steps, the populations f, the
# log-conformation psi, the polymer stress tau and rho/ux/uy.
#
# Production entry point: `_run_viscoelastic_logfv_step_channel_coupled_2d`
# (the driver behind run_viscoelastic_logfv_cylinder_coupled_2d), called on a
# StepChannelGeometry2D built from the copy's own q_wall / LBM mask, with its
# `step_callback` recording the fields after every step. The driver starts from
# its own initial state (f at equilibrium with the inlet profile on fluid cells
# that are not cut, rest elsewhere; psi = 0); the copy is started from that same
# state, built here with the same formula.
#
# Production settings forced to the copy's model:
#   Oldroyd-B, bsd_fraction = 0 (the copy has no both-sides-diffusion term and
#   uses nu_lbm = nu_s), polymer_substeps = 20, embedded_geometry = :circle with
#   16 samples (the copy's circle builder), embedded_gradient / embedded_advection
#   / embedded_force = true (wall-gradient correction, west face = inlet profile,
#   embedded div tau with cell-fraction rescale), advection_scheme =
#   :muscl_superbee_relax (Rusanov near edges and solid, MUSCL-Superbee relax on
#   the cylinder band), wall_bc = :halfwayBB, west Zou-He parabolic inlet and
#   east NeqExtrapolationPressure(1) (default_step_bcspec_2d with open faces
#   1:Ny), drag disabled (drag_u_ref = nothing).
#
# Constitutive time step. Production fixes the substep to dt_poly = 1/n
# (one LBM step of polymer time, split in n substeps); it has no dt keyword.
# The copy follows the same rule: ad_ve_coupled_params takes n_substeps and sets
# dt = 1/n. The FAST case of the adjoint tests is 20 substeps of dt = 0.05
# (1.0/20 and the literal 0.05 are the same Float64). It used to be 4 substeps
# of dt = 0.05 (0.2 polymer time per LBM step), which production cannot run;
# that case had a different steady state (23 % in the drag objective). The
# "unreachable pair" testset below pins that the helper cannot express it.
#
# Comparison: fluid cells only (the copy writes rest populations and zero psi
# in solid cells, which nothing downstream reads). Per field, the max absolute
# difference divided by the max absolute production value. Gate 1e-13
# (round-off). No Enzyme: plain CPU tier.
#
# A last testset converges the copy to its steady fixed point (the state the
# adjoint differentiates), runs production for the same number of steps, and
# compares the two.

using Test, KrakeNN

module VEStepParity

using KrakeNN
const K = KrakeNN

# FAST 24x24 cut cylinder of test_ad_ve_fd_check.jl / test_ad_ve_sensitivity.jl.
const FAST = (; Nx=24, Ny=24, cx=12.35, cy=11.65, R=5.13,
              Wi=0.5, nu_p=0.02, nu_s=0.08, Fx_body=2e-4, samples=16,
              n_substeps=20)
const NSTEPS = 10

_copy_params(c) = K.ad_ve_coupled_params(; Nx=c.Nx, Ny=c.Ny, lambda=c.Wi,
                                         n_substeps=c.n_substeps,
                                         prefactor=c.nu_p / c.Wi, nu_lbm=c.nu_s,
                                         Fx_body=c.Fx_body)

# Production geometry carrying exactly the copy's q_wall and LBM mask
# (node frame cx - 0.5, cy - 0.5), with inlet and outlet open on 1:Ny.
function _prod_geometry(c, q_wall, is_solid)
    Ny = c.Ny
    hydro = fill(false, Ny)
    Ny > 2 && (hydro[2:(Ny - 1)] .= true)
    conf = fill(true, Ny)
    return K.StepChannelGeometry2D{Float64,Array{Float64,3},Matrix{Bool},Vector{Bool}}(
        :cylinder, c.Nx, Ny, round(Int, c.cx - 0.5) + 1, 1:Ny, 1:Ny,
        max(1, round(Int, 2 * c.R)), Ny, Ny,
        copy(q_wall), Matrix{Bool}(is_solid),
        hydro, copy(hydro), conf, copy(conf),
    )
end

# Production run of `nsteps` steps; records the fields after every step, or only
# after the last one when `last_only`.
function _run_production(c, geom_h; nsteps=NSTEPS, last_only=false)
    snaps = NamedTuple[]
    cb = function (step, fl)
        last_only && step != nsteps && return nothing
        push!(snaps, (; step, f=copy(Array(fl.f_out)),
                      rho=copy(Array(fl.rho)), ux=copy(Array(fl.ux)), uy=copy(Array(fl.uy)),
                      psixx=copy(Array(fl.psixx)), psixy=copy(Array(fl.psixy)),
                      psiyy=copy(Array(fl.psiyy)),
                      tauxx=copy(Array(fl.tauxx)), tauxy=copy(Array(fl.tauxy)),
                      tauyy=copy(Array(fl.tauyy))))
        return nothing
    end
    res = K._run_viscoelastic_logfv_step_channel_coupled_2d(
        geom_h;
        shear_length=c.Ny,
        nu_s=c.nu_s, nu_p=c.nu_p, lambda=c.Wi, polymer_model=:oldroydb,
        u_mean=c.Fx_body, Fx_body=c.Fx_body, bsd_fraction=0.0,
        polymer_substeps=c.n_substeps, max_steps=nsteps,
        embedded_gradient=true, embedded_advection=true, embedded_force=true,
        embedded_drag=false, embedded_geometry=:circle,
        embedded_circle_samples=c.samples,
        force_boundary_fill=:bc_aware, advection_scheme=:muscl_superbee_relax,
        wall_bc=:halfwayBB, step_callback=cb, stop=:none,
        drag_cx=c.cx - 0.5, drag_cy=c.cy - 0.5, drag_radius=c.R, drag_u_ref=nothing,
        T=Float64,
    )
    return res, snaps
end

# The driver's initial state, rebuilt with the same formula, as a stacked w.
function _initial_state(c, is_solid, u_profile)
    Nx, Ny = c.Nx, c.Ny
    n = Nx * Ny
    emb = K.fvfd_embedded_boundary_from_circle_2d(Nx, Ny, c.cx, c.cy, c.R;
                                                  FT=Float64, samples=c.samples)
    w = zeros(Float64, 12n)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        ux0 = (is_solid[i, j] || emb.cut_count[i, j] > 0) ? 0.0 : u_profile[j]
        w[K.ad_ve_fpop(i, j, q, Nx, Ny)] = K.equilibrium(K.D2Q9(), 1.0, ux0, 0.0, q)
    end
    return w, emb   # psi block stays 0, as in the driver
end

# Fields of a stacked copy state in the production layout.
function _copy_fields(w, p)
    Nx, Ny = p.Nx, p.Ny
    n = Nx * Ny
    f = reshape(w[1:9n], Nx, Ny, 9)
    rho = zeros(Nx, Ny); ux = zeros(Nx, Ny); uy = zeros(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        f1, f2, f3, f4, f5, f6, f7, f8, f9 = (f[i, j, q] for q in 1:9)
        r = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
        inv_r = 1.0 / r
        rho[i, j] = r
        ux[i, j] = (f2 - f4 + f6 - f7 - f8 + f9) * inv_r
        uy[i, j] = (f3 - f5 + f6 + f7 - f8 - f9) * inv_r
    end
    psixx = reshape(w[9n+1:10n], Nx, Ny)
    psixy = reshape(w[10n+1:11n], Nx, Ny)
    psiyy = reshape(w[11n+1:12n], Nx, Ny)
    tauxx, tauxy, tauyy = K.ad_ve_extract_tau(w, p)
    return (; f, rho, ux, uy, psixx, psixy, psiyy, tauxx, tauxy, tauyy)
end

const FIELDS = (:f, :rho, :ux, :uy, :psixx, :psixy, :psiyy, :tauxx, :tauxy, :tauyy)

# max |a - b| / max |b| over fluid cells (f: all 9 populations of fluid cells).
function _field_rel(a, b, fluid)
    num = 0.0; den = 0.0
    if ndims(b) == 3
        for q in axes(b, 3), j in axes(b, 2), i in axes(b, 1)
            fluid[i, j] || continue
            num = max(num, abs(a[i, j, q] - b[i, j, q])); den = max(den, abs(b[i, j, q]))
        end
    else
        for j in axes(b, 2), i in axes(b, 1)
            fluid[i, j] || continue
            num = max(num, abs(a[i, j] - b[i, j])); den = max(den, abs(b[i, j]))
        end
    end
    return den > 0 ? num / den : num
end

_rels(cp, pr, fluid) = NamedTuple{FIELDS}(Tuple(_field_rel(getproperty(cp, s),
                                                           getproperty(pr, s), fluid)
                                                for s in FIELDS))

function run()
    c = FAST
    geom = K.ad_ve_build_geom(c.Nx, c.Ny, c.cx, c.cy, c.R;
                              samples=c.samples, u_mean=c.Fx_body)
    fluid = .!geom.g.is_solid
    geom_h = _prod_geometry(c, geom.q_wall, geom.g.is_solid)

    # geometry: copy circle builder vs the production builder the driver calls
    w0, emb = _initial_state(c, geom.g.is_solid, geom.u_profile)
    geo_fields = (:cell_fraction, :west_fraction, :east_fraction, :south_fraction,
                  :north_fraction, :wall_nx, :wall_ny, :wall_inv_distance_to_center)
    geo_diff = maximum(maximum(abs.(getproperty(geom.g, s) .- getproperty(emb, s)))
                       for s in geo_fields)
    cut_equal = geom.g.cut_count == emb.cut_count
    profile_equal = geom.u_profile ==
                    K.parabolic_face_profile_2d(geom_h; face=:west,
                                                mean_velocity=c.Fx_body, FT=Float64)

    res, snaps = _run_production(c, geom_h)

    # copy with the FAST setting: 20 substeps of dt = 1/20, the production rule
    p = _copy_params(c)
    w = copy(w0)
    rels = NamedTuple[]
    for k in 1:NSTEPS
        w_out = zeros(Float64, length(w))
        K.ad_ve_coupled_step!(w_out, w, geom.g, geom.q_wall, p, geom.u_profile, 1.0, nothing)
        w = w_out
        push!(rels, _rels(_copy_fields(w, p), snaps[k], fluid))
    end

    return (; res, p, snaps, rels, geo_diff, cut_equal, profile_equal,
            nsnap=length(snaps))
end


# --- steady fixed point -----------------------------------------------------
# The adjoint differentiates the fixed point w* = G(w*), so the converged state
# is what matters. (a) The copy is converged from the driver's initial state to
# the fd_check forward tolerance (2e-15 one-step change) in k steps; production
# is run for the same k steps and compared (same map, same start: round-off).
# (b) The copy is also converged from ad_ve_initial_state, the start used by
# test_ad_ve_fd_check.jl; both converged states must be the same fixed point,
# fields and the QoI ad_ve_J_fx, to well below the gate.
const FWD_TOL = 2e-15

function run_steady()
    c = FAST
    geom = K.ad_ve_build_geom(c.Nx, c.Ny, c.cx, c.cy, c.R;
                              samples=c.samples, u_mean=c.Fx_body)
    fluid = .!geom.g.is_solid
    geom_h = _prod_geometry(c, geom.q_wall, geom.g.is_solid)
    p = _copy_params(c)
    w_drv, _ = _initial_state(c, geom.g.is_solid, geom.u_profile)
    w_fd = K.ad_ve_initial_state(geom.g, c.Nx, c.Ny, 0.05)   # u0 = 0.05, as in fd_check
    t_drv = @elapsed fwd_drv = K.ad_ve_forward_solve(w_drv, geom, p; fwd_tol=FWD_TOL)
    t_fd = @elapsed fwd_fd = K.ad_ve_forward_solve(w_fd, geom, p; fwd_tol=FWD_TOL)
    k = fwd_drv.n_iter
    t_prod = @elapsed prod_out = _run_production(c, geom_h; nsteps=k, last_only=true)
    snaps = prod_out[2]
    fields_drv = _copy_fields(fwd_drv.w_star, p)
    fields_fd = _copy_fields(fwd_fd.w_star, p)
    rels_prod = _rels(fields_drv, snaps[end], fluid)
    rels_start = _rels(fields_fd, fields_drv, fluid)
    J_drv = K.ad_ve_J_fx(fwd_drv.w_star, geom.pts, geom.g, p)
    J_fd = K.ad_ve_J_fx(fwd_fd.w_star, geom.pts, geom.g, p)
    J_rel = abs(J_fd - J_drv) / max(abs(J_drv), eps(Float64))
    return (; fwd_drv, fwd_fd, k, nsnap=length(snaps), last_step=snaps[end].step,
            rels_prod, rels_start, J_drv, J_fd, J_rel,
            seconds=(; copy_drv=t_drv, copy_fd=t_fd, production=t_prod))
end
end # module

const VE_STEP_PARITY_GATE = 1e-13

@testset "LBM VE CPU copy vs production step" begin
    elapsed = @elapsed r = VEStepParity.run()
    @info "VE step parity geometry" geo_max_abs_diff=r.geo_diff cut_equal=r.cut_equal profile_equal=r.profile_equal seconds=elapsed
    for (k, rel) in enumerate(r.rels)
        @info "VE step parity step $(k)" rel=rel
    end

    @testset "same inputs" begin
        @test r.nsnap == VEStepParity.NSTEPS
        @test r.res.polymer_substeps == VEStepParity.FAST.n_substeps
        @test r.p.dt == 1.0 / VEStepParity.FAST.n_substeps
        @test r.profile_equal
        @test r.cut_equal
        @test r.geo_diff <= VE_STEP_PARITY_GATE
    end

    @testset "one step, field $(s)" for s in VEStepParity.FIELDS
        @test getproperty(r.rels[1], s) <= VE_STEP_PARITY_GATE
    end

    @testset "$(VEStepParity.NSTEPS) steps, field $(s)" for s in VEStepParity.FIELDS
        @test maximum(getproperty(rel, s) for rel in r.rels) <= VE_STEP_PARITY_GATE
    end

    # The earlier FAST case, 4 substeps of dt = 0.05 (0.2 polymer time per LBM
    # step), cannot be run by production (dt_poly = 1/n, no dt keyword) and had
    # a different steady state. The helper has no dt keyword, so it cannot build
    # that pair; the positional constructor still can, and the predicate flags it.
    @testset "unreachable pair is not expressible through the helper" begin
        c = VEStepParity.FAST
        kw = (; Nx=c.Nx, Ny=c.Ny, lambda=c.Wi, prefactor=c.nu_p / c.Wi,
              nu_lbm=c.nu_s, Fx_body=c.Fx_body)
        @test_throws MethodError KrakeNN.ad_ve_coupled_params(; kw..., n_substeps=4, dt=0.05)
        @test KrakeNN.ad_ve_coupled_params(; kw..., n_substeps=4).dt == 0.25
        @test KrakeNN.ad_ve_production_reachable(KrakeNN.ad_ve_coupled_params(; kw..., n_substeps=4))
        @test KrakeNN.ad_ve_production_reachable(r.p)
        sp, sm = KrakeNN.ad_ve_trt_rates(c.nu_s)
        old = KrakeNN.ADVECoupledParams(c.Nx, c.Ny, c.Wi, 0.05, 4, c.nu_p / c.Wi,
                                        c.nu_s, c.Fx_body, sp, sm)
        @test !KrakeNN.ad_ve_production_reachable(old)
        @test_throws ArgumentError KrakeNN.ad_ve_coupled_params(; kw..., n_substeps=0)
    end
end

# Steady fixed point. (a) copy vs production after the same k steps from the same
# start: bit-exact per step above, so round-off accumulated over k steps; gate
# 1e-10 (not 1e-13: k is several thousand steps). (b) two starts, same fixed
# point: the gap is bounded by the convergence tolerance divided by the
# contraction gap of the map, well below 1e-10 for FWD_TOL = 2e-15 (reasoned,
# not measured).
const VE_STEADY_GATE = 1e-10

@testset "LBM VE copy steady state vs production" begin
    elapsed = @elapsed s = VEStepParity.run_steady()
    for (name, f) in (("driver start", s.fwd_drv), ("fd_check start", s.fwd_fd))
        @info "VE steady forward ($(name))" n_iter=f.n_iter residual=f.residual reached_tol=f.reached_tol
    end
    @info "VE steady copy vs production after k steps" k=s.k rel=s.rels_prod
    @info "VE steady fd_check start vs driver start" rel=s.rels_start J_drv=s.J_drv J_fd=s.J_fd J_rel=s.J_rel seconds=s.seconds total=elapsed

    @test s.fwd_drv.reached_tol
    @test s.fwd_fd.reached_tol
    @test s.nsnap == 1
    @test s.last_step == s.k
    @testset "steady field $(f), copy vs production" for f in VEStepParity.FIELDS
        @test getproperty(s.rels_prod, f) <= VE_STEADY_GATE
    end
    @testset "steady field $(f), two starts" for f in VEStepParity.FIELDS
        @test getproperty(s.rels_start, f) <= VE_STEADY_GATE
    end
    @test s.J_rel <= VE_STEADY_GATE
end
