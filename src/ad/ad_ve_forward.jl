# Forward time-march + Cd_polymer QoI for the VE steady shape-adjoint AD track.
# Companion to ad_ve_step.jl. Plain-Julia, Enzyme-tapeable (the QoI ve_ve_J_fx is
# differentiated wrt the state w in the reverse seam); NO `using Enzyme` here.
#
# Ported verbatim (preserving every formula) from the validated scratch chain
# (bench/scratch/ve_ad_c3_matched.jl + ve_ad_c1.jl + ve_ad_c0.jl), namespaced
# ad_ve_*. The forward reconverges the coupled fixed-point map to a TIGHT floor
# (fwd_tol=1e-13 default): the net d(Cd_polymer)/dR is a ~20x catastrophic
# cancellation (explicit + state-response), so a 1e-11 patience floor poisons the
# FD state-response (see ve_ad_c3_discriminator.jl STEP-1B tol-sweep).

using LinearAlgebra: norm

# Default forward reconvergence controls (mirror scratch VE_C3M_* constants).
const AD_VE_FWD_TOL       = 1e-13      # forward reconvergence floor (the default)
const AD_VE_FWD_MAX_STEPS = 60_000     # tight forwards need many iters (~10k at Wi=0.5)
const AD_VE_FWD_PATIENCE  = 4_000

# D2Q9 lattice velocity tuples for the wall-point geometry (mirror scratch _CXV/_CYV).
const AD_VE_CXV = (0, 1, 0, -1, 0, 1, -1, -1, 1)
const AD_VE_CYV = (0, 0, 1, 0, -1, 1, 1, -1, -1)

# ----------------------------------------------------------------------------
# Wall-quadrature point: theta-ordering + arc-length weight ds depend on geometry
# only (not tau), so the QoI is a clean differentiable weighted sum. (scratch WallPoint)
struct ADVEWallPoint
    i::Int
    j::Int
    q::Int
    q_w::Float64
    nx::Float64
    ny::Float64
    ds::Float64
end

# ----------------------------------------------------------------------------
# Matched geometry bundle at radius R: embedded g (LBM is_solid mask), f-side
# cut-link q_wall, wall-quadrature pts, inlet u_profile, radius. (scratch VEGeomM)
struct ADVEGeom
    g::ADVEEmbeddedGeom
    q_wall::Array{Float64,3}
    pts::Vector{ADVEWallPoint}
    u_profile::Vector{Float64}
    radius::Float64
end

"""
    ad_ve_build_wall_points(q_wall, Nx, Ny; cx, cy, radius) -> Vector{ADVEWallPoint}

Precompute the ordered cut-point quadrature (theta-sorted, arc-length weights)
for the polymeric-drag surface integral. Mirror of scratch `build_wall_points`.
"""
function ad_ve_build_wall_points(q_wall, Nx, Ny; cx, cy, radius)
    raw = Vector{NTuple{6,Float64}}()   # (theta, q_w, nx, ny, i, j)
    meta = Vector{NTuple{3,Int}}()      # (i, j, q)
    @inbounds for j in 1:Ny, i in 1:Nx, q in 2:9
        q_w = Float64(q_wall[i, j, q])
        q_w > 0 || continue
        xw = Float64(i - 1) + q_w * Float64(AD_VE_CXV[q])
        yw = Float64(j - 1) + q_w * Float64(AD_VE_CYV[q])
        rx = xw - Float64(cx); ry = yw - Float64(cy)
        r = hypot(rx, ry); r > 0 || continue
        nx = rx / r; ny = ry / r
        theta = atan(ry, rx)
        push!(raw, (theta, q_w, nx, ny, Float64(i), Float64(j)))
        push!(meta, (i, j, q))
    end
    npts = length(raw)
    npts == 0 && return ADVEWallPoint[]
    perm = sortperm(raw; by=first)
    raw_s = raw[perm]; meta_s = meta[perm]
    R = Float64(radius)
    pts = Vector{ADVEWallPoint}(undef, npts)
    @inbounds for k in 1:npts
        theta_prev = k == 1 ? raw_s[end][1] - 2pi : raw_s[k - 1][1]
        theta_next = k == npts ? raw_s[1][1] + 2pi : raw_s[k + 1][1]
        ds = R * 0.5 * (theta_next - theta_prev)
        i, j, q = meta_s[k]
        pts[k] = ADVEWallPoint(i, j, q, raw_s[k][2], raw_s[k][3], raw_s[k][4], ds)
    end
    return pts
end

"""
    ad_ve_poiseuille_profile(Ny, u_mean) -> Vector{Float64}

Parabolic west-face profile over the channel, zero on the half-way walls at
y = -1/2 and Ny - 1/2 (node j at y = j - 1), with row sum exactly `Ny * u_mean`.
Same construction as `parabolic_face_profile_2d` on a channel open over all
rows (the production viscoelastic cylinder).
"""
ad_ve_poiseuille_profile(Ny, u_mean) = _halfway_poiseuille_profile(Float64, Ny, u_mean)

"""
    ad_ve_build_geom(Nx, Ny, cx, cy, R; samples=16, u_mean=2e-4) -> ADVEGeom

Build the matched geometry bundle at radius R: FVFD circle g with the LBM
is_solid mask (`ad_ve_build_matched_geom`), f-side cut-link q_wall, wall-
quadrature pts, inlet u_profile. cx/cy are the control-volume frame for g; the
node frame `(cx-0.5, cy-0.5)` is used for q_wall + drag. Mirror of scratch
`build_c3m_geom`. Reuses the production `precompute_q_wall_cylinder`.
"""
function ad_ve_build_geom(Nx, Ny, cx, cy, R; samples=16, u_mean=2e-4)
    g_fvfd = ad_ve_build_circle_geom(Nx, Ny, cx, cy, R; samples=samples)
    q_wall, lbm_solid = precompute_q_wall_cylinder(Nx, Ny, cx - 0.5, cy - 0.5, R; FT=Float64)
    g = ad_ve_build_matched_geom(g_fvfd, lbm_solid)
    pts = ad_ve_build_wall_points(q_wall, Nx, Ny; cx=cx - 0.5, cy=cy - 0.5, radius=R)
    u_profile = ad_ve_poiseuille_profile(Ny, u_mean)
    return ADVEGeom(g, q_wall, pts, u_profile, Float64(R))
end

# ----------------------------------------------------------------------------
"""
    ad_ve_initial_state(g, Nx, Ny, u0) -> Vector{Float64}

Warm-start stacked state w0 (length 12n): f at equilibrium with a frozen swirl
(rest rho=1 inside solid), psi a small near-identity log-conformation seeded with
a deterministic smooth field (zero in solid). The warm start only affects the
iteration count — the coupled fixed point is independent of it. Plain-Julia, no
RNG dependency (mirror of scratch `c0_initial_state` with the random psi seed
replaced by a deterministic smooth field).
"""
function ad_ve_initial_state(g::ADVEEmbeddedGeom, Nx, Ny, u0)
    n = Nx * Ny
    w = zeros(Float64, 12n)
    @inbounds for j in 1:Ny, i in 1:Nx
        if g.is_solid[i, j]
            for q in 1:9
                w[ad_ve_fpop(i, j, q, Nx, Ny)] = AD_VE_W[q]   # rho=1, u=0 rest
            end
            continue
        end
        x = (i - 1) / Nx; y = (j - 1) / Ny
        ux0 = u0 * (1.0 + 0.3 * sin(2pi * y) + 0.2 * cos(2pi * x))
        uy0 = 0.25 * u0 * sin(2pi * x) * cos(2pi * y)
        usq = ux0 * ux0 + uy0 * uy0
        for q in 1:9
            w[ad_ve_fpop(i, j, q, Nx, Ny)] = ad_ve_feq(q, 1.0, ux0, uy0, usq)
        end
    end
    poff = 9n
    @inbounds for j in 1:Ny, i in 1:Nx
        g.is_solid[i, j] && continue
        k = ad_ve_lin(i, j, Nx)
        x = (i - 1) / Nx; y = (j - 1) / Ny
        w[poff + k]      = 0.05 * sin(3pi * x) * cos(2pi * y)
        w[poff + n + k]  = 0.03 * sin(2pi * x) * sin(3pi * y)
        w[poff + 2n + k] = 0.05 * cos(2pi * x) * sin(2pi * y)
    end
    return w
end

"""
    ad_ve_dJ_dR_geom_explicit(w_star, Nx, Ny, cx, cy, R, p, epsR; samples, u_mean)
        -> Float64

The EXPLICIT geometry partial ∂J/∂R|geom: central-FD of the QoI `ad_ve_J_fx` over
the geometry (pts + g) at a FROZEN state `w_star`. This is the only finite
difference in the production gradient; the state-response is FD-free (the analytic
adjoint). Mirror of scratch `c3m_dJ_dR_geom_explicit`.
"""
function ad_ve_dJ_dR_geom_explicit(w_star, Nx, Ny, cx, cy, R, p::ADVECoupledParams, epsR;
                                   samples=16, u_mean=2e-4)
    gp = ad_ve_build_geom(Nx, Ny, cx, cy, R + epsR; samples=samples, u_mean=u_mean)
    gm = ad_ve_build_geom(Nx, Ny, cx, cy, R - epsR; samples=samples, u_mean=u_mean)
    Jp = ad_ve_J_fx(w_star, gp.pts, gp.g, p)
    Jm = ad_ve_J_fx(w_star, gm.pts, gm.g, p)
    return (Jp - Jm) / (2.0 * epsR)
end

"""
    ad_ve_pack_state(f_h, psixx_h, psixy_h, psiyy_h, Nx, Ny) -> Vector{Float64}

Pack the (f, psi) fields into the stacked flat state w (length 12n). f uses the
popidx layout; psi uses the [psixx; psixy; psiyy] 3n-block layout. Mirror of
scratch `pack_state`.
"""
function ad_ve_pack_state(f_h, psixx_h, psixy_h, psiyy_h, Nx, Ny)
    n = Nx * Ny
    w = zeros(Float64, 12n)
    @inbounds for j in 1:Ny, i in 1:Nx
        for q in 1:9
            w[ad_ve_fpop(i, j, q, Nx, Ny)] = Float64(f_h[i, j, q])
        end
        k = ad_ve_lin(i, j, Nx)
        w[9n + k]      = Float64(psixx_h[i, j])
        w[9n + n + k]  = Float64(psixy_h[i, j])
        w[9n + 2n + k] = Float64(psiyy_h[i, j])
    end
    return w
end

"""
    ad_ve_extract_tau(w, p) -> (txx, txy, tyy)

Extract the polymer stress tau_p = prefactor*(C - I) (3 fields) from the psi'
block of a stacked state. Mirror of scratch `c0_extract_tau`.
"""
function ad_ve_extract_tau(w, p::ADVECoupledParams)
    Nx, Ny = p.Nx, p.Ny
    n = Nx * Ny
    poff = 9n
    txx = zeros(Float64, Nx, Ny); txy = zeros(Float64, Nx, Ny); tyy = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        k = ad_ve_lin(i, j, Nx)
        pxx = w[poff + k]
        pxy = w[poff + n + k]
        pyy = w[poff + 2n + k]
        a, b, d = ad_ve_stress_from_log_2d(pxx, pxy, pyy, p.prefactor)
        txx[i, j] = a; txy[i, j] = b; tyy[i, j] = d
    end
    return txx, txy, tyy
end

# ----------------------------------------------------------------------------
"""
    ad_ve_J_fx(w, pts, g, p) -> Fx

The QoI: Fx of the polymeric drag (the Cd_polymer x-component), a clean arc-
weighted surface quadrature. Differentiable wrt w (psi block) via Enzyme: unpack
psi -> ad_ve_stress_from_log_2d -> tau matrices -> reconstruct at the precomputed
cut points (production `reconstruct_wall_link_value_2d`, a pure polynomial in the
field) -> arc-weighted sum with FROZEN ds/nx/ny. Bit-mirror of the production
`compute_polymeric_drag_2d` (q_wall method, extrapolate=true, order=2) Fx.
Mirror of scratch `ve_J_fx`.

Solid cells are skipped: psi=0 there makes exp_sym2_2d's delta=0 and its 0/0
ifelse branch has a NaN reverse-mode derivative; solid cells carry zero stress
and are never wall points, so skipping is exact.
"""
function ad_ve_J_fx(w, pts::Vector{ADVEWallPoint}, g::ADVEEmbeddedGeom, p::ADVECoupledParams)
    Nx, Ny = p.Nx, p.Ny
    n = Nx * Ny
    poff = 9n
    is_solid = g.is_solid
    txx = zeros(Float64, Nx, Ny)
    txy = zeros(Float64, Nx, Ny)
    tyy = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        k = ad_ve_lin(i, j, Nx)
        pxx = w[poff + k]
        pxy = w[poff + n + k]
        pyy = w[poff + 2n + k]
        a, b, d = ad_ve_stress_from_log_2d(pxx, pxy, pyy, p.prefactor)
        txx[i, j] = a; txy[i, j] = b; tyy[i, j] = d
    end
    Fx = 0.0
    @inbounds for k in eachindex(pts)
        pt = pts[k]
        txx_w = reconstruct_wall_link_value_2d(txx, pt.i, pt.j, pt.q, pt.q_w;
                                               location=:cut, order=2)
        txy_w = reconstruct_wall_link_value_2d(txy, pt.i, pt.j, pt.q, pt.q_w;
                                               location=:cut, order=2)
        Fx += (txx_w * pt.nx + txy_w * pt.ny) * pt.ds
    end
    return Fx
end

# ----------------------------------------------------------------------------
"""
    ad_ve_forward_solve(w0, geom, p; fwd_tol=1e-13, max_steps, patience) -> NamedTuple

Iterate the coupled VE one-step operator `ad_ve_coupled_step!` (with u_profile +
ZouHe rho_out=1) toward the basin floor, returning the deepest-residual state.
The default `fwd_tol=1e-13` is required for the net d(Cd_polymer)/dR (a ~20x
catastrophic cancellation): a looser floor poisons the FD state-response.

Returns `(; w_star, n_iter, residual, converged, reached_tol, last_res, nan_at)`.
Mirror of scratch `ve_forward_matched_tight`.
"""
function ad_ve_forward_solve(w0, geom::ADVEGeom, p::ADVECoupledParams;
                             fwd_tol::Real=AD_VE_FWD_TOL,
                             max_steps::Int=AD_VE_FWD_MAX_STEPS,
                             patience::Int=AD_VE_FWD_PATIENCE)
    n = ad_ve_n(p); out_len = 12n
    w_in = copy(w0); w_out = zeros(Float64, out_len)
    best_res = Inf; best_w = copy(w0); best_iter = 0; since_best = 0
    reached_tol = false; last_res = NaN; nan_at = 0
    for step in 1:max_steps
        ad_ve_coupled_step!(w_out, w_in, geom.g, geom.q_wall, p, geom.u_profile, 1.0, nothing)
        if !all(isfinite, w_out)
            nan_at = step; break
        end
        denom = max(norm(w_in), eps(Float64))
        residual = norm(w_out .- w_in) / denom
        last_res = residual
        if residual < best_res
            best_res = residual; copyto!(best_w, w_out); best_iter = step; since_best = 0
        else
            since_best += 1
        end
        if residual < Float64(fwd_tol)
            reached_tol = true; best_res = residual; copyto!(best_w, w_out); best_iter = step
            break
        end
        since_best >= patience && break
        w_in, w_out = w_out, w_in
    end
    converged = reached_tol || isfinite(best_res)
    return (; w_star=copy(best_w), n_iter=best_iter, residual=Float64(best_res),
            converged, reached_tol, last_res, nan_at)
end

# ----------------------------------------------------------------------------
"""
    ad_ve_antidrift_delta(w_star, geom, p; cx, cy, radius) -> NamedTuple

Anti-drift reference: compares the inline QoI `ad_ve_J_fx` (Fx-only) against the
production `compute_polymeric_drag_2d` on the SAME (tau, q_wall). Both consume an
identical reconstruction, so |Delta| must be machine zero (<= 1e-12). Mirror of
scratch `c3m_antidrift_delta`. (cx, cy are the control-volume frame; the node
frame `(cx-0.5, cy-0.5)` is used for the drag/q_wall quadrature.)
"""
function ad_ve_antidrift_delta(w_star, geom::ADVEGeom, p::ADVECoupledParams;
                               cx, cy, radius)
    Nx, Ny = p.Nx, p.Ny
    txx, txy, tyy = ad_ve_extract_tau(w_star, p)
    prod = compute_polymeric_drag_2d(txx, txy, tyy, geom.q_wall, Nx, Ny;
                                     cx=cx - 0.5, cy=cy - 0.5, radius=radius,
                                     extrapolate=true, reconstruction_order=2)
    inline_Fx = ad_ve_J_fx(w_star, geom.pts, geom.g, p)
    return (; delta=abs(prod.Fx - inline_Fx), prod_Fx=prod.Fx, inline_Fx)
end

# ----------------------------------------------------------------------------
"""
    ad_ve_fd_dCdpoly_dR(w_base, Nx, Ny, cx, cy, R, p, h; samples, u_mean,
                        fwd_tol=1e-13, max_steps, patience) -> NamedTuple

Anti-drift FD reference for d(Cd_polymer)/dR: rebuild the matched geometry at
R +/- h, reconverge the forward tight (warm-started from `w_base`), and central-
difference the inline Cd_polymer at the perturbed converged states. Returns
`(; value, Jp, Jm, fp, fm, cutp, cutm, solp, solm)`. The cut/solid counts let the
caller assert the cut-link topology is fixed across +/-h (off-lattice + small h).
Mirror of scratch `c3m_fd_dJ_dR_converged`.
"""
function ad_ve_fd_dCdpoly_dR(w_base, Nx, Ny, cx, cy, R, p::ADVECoupledParams, h;
                             samples=16, u_mean=2e-4,
                             fwd_tol::Real=AD_VE_FWD_TOL,
                             max_steps::Int=AD_VE_FWD_MAX_STEPS,
                             patience::Int=AD_VE_FWD_PATIENCE)
    gp = ad_ve_build_geom(Nx, Ny, cx, cy, R + h; samples=samples, u_mean=u_mean)
    gm = ad_ve_build_geom(Nx, Ny, cx, cy, R - h; samples=samples, u_mean=u_mean)
    fp = ad_ve_forward_solve(w_base, gp, p; fwd_tol, max_steps, patience)
    fm = ad_ve_forward_solve(w_base, gm, p; fwd_tol, max_steps, patience)
    Jp = ad_ve_J_fx(fp.w_star, gp.pts, gp.g, p)
    Jm = ad_ve_J_fx(fm.w_star, gm.pts, gm.g, p)
    cutp = count(>(0.0), gp.q_wall); cutm = count(>(0.0), gm.q_wall)
    solp = count(gp.g.is_solid); solm = count(gm.g.is_solid)
    return (; value=(Jp - Jm) / (2.0 * h), Jp, Jm, fp, fm, cutp, cutm, solp, solm)
end
