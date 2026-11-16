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
#   uses nu_lbm = nu_s), polymer_substeps = 4, embedded_geometry = :circle with
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
# The FAST case of the adjoint tests uses dt = 0.05 with 4 substeps, i.e. 0.2
# of polymer time per LBM step, which no production setting reproduces. The
# matched comparison therefore uses the copy with dt = 1/4 and 4 substeps. A
# second testset runs the copy with the FAST setting (dt = 0.05) against the
# same production run and records the psi gap as @test_broken.
#
# Comparison: fluid cells only (the copy writes rest populations and zero psi
# in solid cells, which nothing downstream reads). Per field, the max absolute
# difference divided by the max absolute production value. Gate 1e-13
# (round-off). No Enzyme: plain CPU tier.
#
# A last testset converges the copy at dt = 0.05 and at dt = 1/4 and compares
# the two steady fixed points (the state the adjoint differentiates).

using Test, KrakeNN

module VEStepParity

using KrakeNN
const K = KrakeNN

# FAST 24x24 cut cylinder of test_ad_ve_fd_check.jl / test_ad_ve_sensitivity.jl.
const FAST = (; Nx=24, Ny=24, cx=12.35, cy=11.65, R=5.13,
              Wi=0.5, nu_p=0.02, nu_s=0.08, Fx_body=2e-4, samples=16)
const NSTEPS = 10
const NSUB = 4

function _copy_params(c, dt)
    lambda = c.Wi
    pref = c.nu_p / lambda
    s_plus, s_minus = K.ad_ve_trt_rates(c.nu_s)
    return K.ADVECoupledParams(c.Nx, c.Ny, lambda, dt, NSUB, pref, c.nu_s,
                               c.Fx_body, s_plus, s_minus)
end

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

function _run_production(c, geom_h)
    snaps = NamedTuple[]
    cb = function (step, fl)
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
        polymer_substeps=NSUB, max_steps=NSTEPS,
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

    # matched copy: dt = 1/NSUB, the production substep
    p = _copy_params(c, 1.0 / NSUB)
    w = copy(w0)
    rels = NamedTuple[]
    for k in 1:NSTEPS
        w_out = zeros(Float64, length(w))
        K.ad_ve_coupled_step!(w_out, w, geom.g, geom.q_wall, p, geom.u_profile, 1.0, nothing)
        w = w_out
        push!(rels, _rels(_copy_fields(w, p), snaps[k], fluid))
    end

    # FAST-case copy setting: dt = 0.05 with 4 substeps, one step
    p_fast = _copy_params(c, 0.05)
    w_fast = zeros(Float64, length(w0))
    K.ad_ve_coupled_step!(w_fast, copy(w0), geom.g, geom.q_wall, p_fast,
                          geom.u_profile, 1.0, nothing)
    rels_fast = _rels(_copy_fields(w_fast, p_fast), snaps[1], fluid)

    return (; res, snaps, rels, rels_fast, geo_diff, cut_equal, profile_equal,
            nsnap=length(snaps))
end


# --- steady fixed point: FAST dt = 0.05 versus the production substep 1/4 -------
# The adjoint differentiates the fixed point w* = G(w*), so only the converged
# state matters. Both copies start from ad_ve_initial_state and reconverge to the
# fd_check forward tolerance (2e-15 one-step change). The QoI is ad_ve_J_fx, the
# objective of test_ad_ve_fd_check.jl.
const FWD_TOL = 2e-15

function run_steady()
    c = FAST
    geom = K.ad_ve_build_geom(c.Nx, c.Ny, c.cx, c.cy, c.R;
                              samples=c.samples, u_mean=c.Fx_body)
    fluid = .!geom.g.is_solid
    w0 = K.ad_ve_initial_state(geom.g, c.Nx, c.Ny, 0.05)
    out = map((0.05, 1.0 / NSUB)) do dt
        p = _copy_params(c, dt)
        t = @elapsed fwd = K.ad_ve_forward_solve(w0, geom, p; fwd_tol=FWD_TOL)
        J = K.ad_ve_J_fx(fwd.w_star, geom.pts, geom.g, p)
        (; dt, p, fwd, J, fields=_copy_fields(fwd.w_star, p), seconds=t)
    end
    fast, prod = out
    rels = _rels(fast.fields, prod.fields, fluid)
    J_rel = abs(fast.J - prod.J) / max(abs(prod.J), eps(Float64))
    return (; fast, prod, rels, J_rel)
end
end # module

const VE_STEP_PARITY_GATE = 1e-13

@testset "LBM VE CPU copy vs production step" begin
    elapsed = @elapsed r = VEStepParity.run()
    @info "VE step parity geometry" geo_max_abs_diff=r.geo_diff cut_equal=r.cut_equal profile_equal=r.profile_equal seconds=elapsed
    for (k, rel) in enumerate(r.rels)
        @info "VE step parity step $(k)" rel=rel
    end
    @info "VE step parity FAST dt=0.05 (step 1)" rel=r.rels_fast

    @testset "same inputs" begin
        @test r.nsnap == VEStepParity.NSTEPS
        @test r.res.polymer_substeps == VEStepParity.NSUB
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

    # Known by construction: production integrates one unit of polymer time
    # per LBM step (dt_poly = 1/substeps); the FAST adjoint case integrates 0.2
    # (dt = 0.05, 4 substeps). The log-conformation after one step must differ;
    # the size of the gap is unknown until run (printed above).
    @testset "FAST dt = 0.05 setting is not a production setting" begin
        @test_broken max(r.rels_fast.psixx, r.rels_fast.psixy, r.rels_fast.psiyy) <=
                     VE_STEP_PARITY_GATE
    end
end

# Steady fixed point of the FAST adjoint setting (dt = 0.05, 4 substeps) versus
# the production-equivalent setting (dt = 1/4, 4 substeps, bit-identical per step
# to the production driver, see above). Expected to differ, so @test_broken:
# the constitutive substep is a Lie splitting (deform C -> A C A^T with
# A = exp(dt grad u), then relax C - I by exp(-dt/lambda)), and the fixed point of
# that split map depends on dt. For simple shear, c_xy* = dt gdot e/(1 - e) with
# e = exp(-dt/lambda), i.e. lambda gdot (1 - dt/(2 lambda) + O(dt^2)): about 5 %
# below the exact Oldroyd-B value at dt = 0.05 and 25 % at dt = 1/4 (lambda = 0.5).
# On top of that the copy's polymer time per LBM step (n dt) scales deformation
# and relaxation against the unit-time psi advection (small here: lambda U / R
# ~ 2e-5). The stress, hence the polymer force, ux/uy, rho, f and the drag Fx
# all move with it. Measured gap unknown until run (printed below).
const VE_STEADY_GATE = 1e-10

@testset "LBM VE copy steady state: FAST dt vs production dt" begin
    elapsed = @elapsed s = VEStepParity.run_steady()
    for r in (s.fast, s.prod)
        @info "VE steady forward dt=$(r.dt)" n_iter=r.fwd.n_iter residual=r.fwd.residual reached_tol=r.fwd.reached_tol J=r.J seconds=r.seconds
    end
    @info "VE steady FAST vs production dt" rel=s.rels J_rel=s.J_rel seconds=elapsed

    @test s.fast.fwd.reached_tol
    @test s.prod.fwd.reached_tol
    @testset "steady field $(f)" for f in VEStepParity.FIELDS
        @test_broken getproperty(s.rels, f) <= VE_STEADY_GATE
    end
    @test_broken s.J_rel <= VE_STEADY_GATE
end
