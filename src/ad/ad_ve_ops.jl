# Shared FVFD embedded operators + log-conformation constitutive math + D2Q9 LBM
# helpers + the cut-cell circle geometry for the VE shape-adjoint AD track.
# Companion to ad_ve_step.jl (the tapeable coupled operator includes this first).
#
# Ported verbatim (preserving every formula / ordering / float op) from the
# validated scratch chain (bench/scratch/ve_c0_matched.jl + ve_ad_c0.jl +
# ve_ad_bc_attribution.jl + ve_ad_spike{,_embedded}.jl), namespaced `ad_ve_*`.
# Plain-Julia, Enzyme-tapeable: NO @kernel, NO GPU, NO `using Enzyme`.
# BIT-MIRROR — if the production LBM/LI-BB/TRT/ZouHe/FVFD-advection/constitutive
# algebra changes, update this file too.

# ----------------------------------------------------------------------------
# FVFD domain BC codes (mirror src/fvfd/specs.jl FVFD_BC_*). Used only by the
# embedded scalar accessors; for the cylinder (open-x / wall-y) the open and
# wall branches both reduce to the local-cell value, so the codes are inert but
# kept for a faithful mirror.
const AD_VE_BC_PERIODIC = 1
const AD_VE_BC_OPEN     = 2
const AD_VE_BC_WALL     = 3

const AD_VE_WB = AD_VE_BC_OPEN    # west open
const AD_VE_EB = AD_VE_BC_OPEN    # east open
const AD_VE_SB = AD_VE_BC_WALL    # south wall
const AD_VE_NB = AD_VE_BC_WALL    # north wall

# ----------------------------------------------------------------------------
# f-side D2Q9 constants + helpers (mirror src/ad/ad_step.jl / scratch c0_*).
const AD_VE_CX = (0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0)
const AD_VE_CY = (0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0)
const AD_VE_W  = (4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
                  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0)

@inline ad_ve_lin(i, j, Nx) = i + (j - 1) * Nx
@inline ad_ve_fpop(i, j, q, Nx, Ny) = i + (j - 1) * Nx + (q - 1) * Nx * Ny

@inline function ad_ve_feq(q::Int, rho, ux, uy, usq)
    cu = AD_VE_CX[q] * ux + AD_VE_CY[q] * uy
    return AD_VE_W[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
end

# LI-BB cut-link branch (mirror ad_libb_branch / scratch c0_libb_branch).
@inline function ad_ve_libb_branch(q_w, f_post_here, f_post_back, f_bar_post_here)
    if q_w <= 0.5
        return 2.0 * q_w * f_post_here + (1.0 - 2.0 * q_w) * f_post_back
    else
        inv_two_q = 1.0 / (2.0 * q_w)
        return inv_two_q * f_post_here + (1.0 - inv_two_q) * f_bar_post_here
    end
end

@inline function ad_ve_trt_rates(nu)
    lambda_magic = 3.0 / 16.0
    s_plus = 1.0 / (3.0 * nu + 0.5)
    s_minus = 1.0 / (lambda_magic / (3.0 * nu) + 0.5)
    return s_plus, s_minus
end

# Regularized TRT collide (EXACT mirror of src/bc/specs.jl _trt_collide_local /
# scratch attr_trt_collide_local). Used by the fused ZouHe rebuild.
@inline function ad_ve_trt_collide_local(f1, f2, f3, f4, f5, f6, f7, f8, f9, s_p, s_m)
    rho = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
    ux = (f2 - f4 + f6 - f8 + f9 - f7) / rho
    uy = (f3 - f5 + f6 - f8 + f7 - f9) / rho
    usq = ux * ux + uy * uy
    fe1 = ad_ve_feq(1, rho, ux, uy, usq); fe2 = ad_ve_feq(2, rho, ux, uy, usq)
    fe3 = ad_ve_feq(3, rho, ux, uy, usq); fe4 = ad_ve_feq(4, rho, ux, uy, usq)
    fe5 = ad_ve_feq(5, rho, ux, uy, usq); fe6 = ad_ve_feq(6, rho, ux, uy, usq)
    fe7 = ad_ve_feq(7, rho, ux, uy, usq); fe8 = ad_ve_feq(8, rho, ux, uy, usq)
    fe9 = ad_ve_feq(9, rho, ux, uy, usq)
    Pxx = (f2 - fe2) + (f4 - fe4) + (f6 - fe6) + (f7 - fe7) + (f8 - fe8) + (f9 - fe9)
    Pyy = (f3 - fe3) + (f5 - fe5) + (f6 - fe6) + (f7 - fe7) + (f8 - fe8) + (f9 - fe9)
    Pxy = (f6 - fe6) - (f7 - fe7) + (f8 - fe8) - (f9 - fe9)
    # f_neq_i = w_i / (2 c_s⁴) (c_iα c_iβ - c_s² δ_αβ) Π_αβ
    h = 0.5
    fn1 = -(2 / 3) * (Pxx + Pyy)
    fn2 = (1 / 6) * (2Pxx - Pyy); fn3 = (1 / 6) * (-Pxx + 2Pyy)
    fn4 = fn2; fn5 = fn3
    fn6 = (1 / 12) * (Pxx + Pyy) + 0.25 * Pxy
    fn7 = (1 / 12) * (Pxx + Pyy) - 0.25 * Pxy
    fn8 = fn6; fn9 = fn7
    a = (s_p + s_m) * h; b = (s_p - s_m) * h
    return (fe1 + (1 - s_p) * fn1,
            fe2 + (1 - a) * fn2 - b * fn4, fe3 + (1 - a) * fn3 - b * fn5,
            fe4 + (1 - a) * fn4 - b * fn2, fe5 + (1 - a) * fn5 - b * fn3,
            fe6 + (1 - a) * fn6 - b * fn8, fe7 + (1 - a) * fn7 - b * fn9,
            fe8 + (1 - a) * fn8 - b * fn6, fe9 + (1 - a) * fn9 - b * fn7)
end

# East pressure outlet (mirror of `_bc_east_neq_extrap_2d!`,
# `NeqExtrapolationPressure`): row j of column Nx is rebuilt, without collision,
# from the post-collision populations of column Nx-1 already in `w_out`:
#   f_q(Nx, j) = f_q(Nx-1, j) + f_q^eq(rho_out - rho, u)
# with (rho, u) the raw moments of f(Nx-1, j). Moments in the order of
# `moments_2d`, so the result matches the production kernel bit for bit.
@inline function ad_ve_outlet_neq_extrap_row!(w_out, foff, rho_out, j, Nx, Ny)
    @inbounds begin
        g1 = w_out[foff + ad_ve_fpop(Nx - 1, j, 1, Nx, Ny)]
        g2 = w_out[foff + ad_ve_fpop(Nx - 1, j, 2, Nx, Ny)]
        g3 = w_out[foff + ad_ve_fpop(Nx - 1, j, 3, Nx, Ny)]
        g4 = w_out[foff + ad_ve_fpop(Nx - 1, j, 4, Nx, Ny)]
        g5 = w_out[foff + ad_ve_fpop(Nx - 1, j, 5, Nx, Ny)]
        g6 = w_out[foff + ad_ve_fpop(Nx - 1, j, 6, Nx, Ny)]
        g7 = w_out[foff + ad_ve_fpop(Nx - 1, j, 7, Nx, Ny)]
        g8 = w_out[foff + ad_ve_fpop(Nx - 1, j, 8, Nx, Ny)]
        g9 = w_out[foff + ad_ve_fpop(Nx - 1, j, 9, Nx, Ny)]
        rho = g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 + g9
        inv_rho = 1.0 / rho
        ux = (g2 - g4 + g6 - g7 - g8 + g9) * inv_rho
        uy = (g3 - g5 + g6 + g7 - g8 - g9) * inv_rho
        usq = ux * ux + uy * uy
        drho = rho_out - rho
        w_out[foff + ad_ve_fpop(Nx, j, 1, Nx, Ny)] = g1 + ad_ve_feq(1, drho, ux, uy, usq)
        w_out[foff + ad_ve_fpop(Nx, j, 2, Nx, Ny)] = g2 + ad_ve_feq(2, drho, ux, uy, usq)
        w_out[foff + ad_ve_fpop(Nx, j, 3, Nx, Ny)] = g3 + ad_ve_feq(3, drho, ux, uy, usq)
        w_out[foff + ad_ve_fpop(Nx, j, 4, Nx, Ny)] = g4 + ad_ve_feq(4, drho, ux, uy, usq)
        w_out[foff + ad_ve_fpop(Nx, j, 5, Nx, Ny)] = g5 + ad_ve_feq(5, drho, ux, uy, usq)
        w_out[foff + ad_ve_fpop(Nx, j, 6, Nx, Ny)] = g6 + ad_ve_feq(6, drho, ux, uy, usq)
        w_out[foff + ad_ve_fpop(Nx, j, 7, Nx, Ny)] = g7 + ad_ve_feq(7, drho, ux, uy, usq)
        w_out[foff + ad_ve_fpop(Nx, j, 8, Nx, Ny)] = g8 + ad_ve_feq(8, drho, ux, uy, usq)
        w_out[foff + ad_ve_fpop(Nx, j, 9, Nx, Ny)] = g9 + ad_ve_feq(9, drho, ux, uy, usq)
    end
    return nothing
end

# ----------------------------------------------------------------------------
# log-conformation scalar helpers (mirror logconformation_fv_2d.jl /
# scratch ve_ad_spike.jl). Ported under ad_ve_* names to guarantee the operator
# is bit-exact to the validated scratch chain (the production logfv_* reduce to
# the same float ops for Oldroyd-B, but we keep the ported copies for fidelity).

@inline function ad_ve_exp_sym2_2d(a, b, d)
    T = typeof(a + b + d)
    m = (a + d) / T(2)
    h = (a - d) / T(2)
    delta = hypot(h, b)
    em = exp(m)
    delta2 = delta * delta
    scale = ifelse(delta < sqrt(eps(T)), one(T) + delta2 / T(6), sinh(delta) / delta)
    ch = cosh(delta)
    return (
        em * (ch + scale * h),
        em * scale * b,
        em * (ch - scale * h),
    )
end

@inline function ad_ve_exp_mat2_2d(a, b, c, d)
    T = typeof(a + b + c + d)
    m = (a + d) / T(2)
    h = (a - d) / T(2)
    disc = h * h + b * c
    em = exp(m)
    small = abs(disc) < eps(T)

    ch = if small
        one(T) + disc / T(2)
    elseif disc > zero(T)
        delta = sqrt(disc)
        cosh(delta)
    else
        theta = sqrt(-disc)
        cos(theta)
    end

    scale = if small
        one(T) + disc / T(6)
    elseif disc > zero(T)
        delta = sqrt(disc)
        sinh(delta) / delta
    else
        theta = sqrt(-disc)
        sin(theta) / theta
    end

    return (
        em * (ch + scale * h),
        em * scale * b,
        em * scale * c,
        em * (ch - scale * h),
    )
end

@inline function ad_ve_log_spd_sym2_2d(a, b, d)
    T = typeof(a + b + d)
    m = (a + d) / T(2)
    h = (a - d) / T(2)
    delta = hypot(h, b)
    lp = log(m + delta)
    lm = log(m - delta)
    alpha = (lp + lm) / T(2)
    beta = ifelse(
        delta < sqrt(eps(T)) * max(one(T), abs(m)),
        inv(m) + delta * delta / (T(3) * m * m * m),
        (lp - lm) / (T(2) * delta),
    )
    return (
        alpha + beta * h,
        beta * b,
        alpha - beta * h,
    )
end

# Oldroyd-B relax of C (model_code == OLDROYDB path).
@inline function ad_ve_oldroydb_relax_c_2d(cxx, cxy, cyy, lambda, dt)
    decay = exp(-dt / lambda)
    return (
        one(cxx) + (cxx - one(cxx)) * decay,
        cxy * decay,
        one(cyy) + (cyy - one(cyy)) * decay,
    )
end

# Full constitutive substep in log space: psi -> C, deform, relax, -> psi.
@inline function ad_ve_constitutive_step_log_2d(
    psixx, psixy, psiyy, dudx, dudy, dvdx, dvdy, lambda, dt,
)
    cxx, cxy, cyy = ad_ve_exp_sym2_2d(psixx, psixy, psiyy)
    a, b, c, d = ad_ve_exp_mat2_2d(dt * dudx, dt * dudy, dt * dvdx, dt * dvdy)

    ac_xx = a * cxx + b * cxy
    ac_xy = a * cxy + b * cyy
    ac_yx = c * cxx + d * cxy
    ac_yy = c * cxy + d * cyy

    dxx = ac_xx * a + ac_xy * b
    dxy = ac_xx * c + ac_xy * d
    dyy = ac_yx * c + ac_yy * d
    rxx, rxy, ryy = ad_ve_oldroydb_relax_c_2d(dxx, dxy, dyy, lambda, dt)
    return ad_ve_log_spd_sym2_2d(rxx, rxy, ryy)
end

# Oldroyd-B polymer stress from log-conf: prefactor*(C - I).
@inline function ad_ve_stress_from_log_2d(psixx, psixy, psiyy, prefactor)
    cxx, cxy, cyy = ad_ve_exp_sym2_2d(psixx, psixy, psiyy)
    return (
        prefactor * (cxx - one(cxx)),
        prefactor * cxy,
        prefactor * (cyy - one(cyy)),
    )
end

# ----------------------------------------------------------------------------
# 1-D BC index helper (mirror _fvfd_bc_index_1d / scratch bc_index_1d).
@inline function ad_ve_bc_index_1d(idx, n, lower_bc, upper_bc)
    if 1 <= idx <= n
        return idx
    elseif idx < 1 && lower_bc == AD_VE_BC_PERIODIC
        return idx + n
    elseif idx > n && upper_bc == AD_VE_BC_PERIODIC
        return idx - n
    else
        return 0
    end
end

# Central / one-sided derivative with solid+wall handling (quadratic extrap),
# mirror _fvfd_solid_bc_derivative_x_2d / _y (polymer_wall_extrap = :quadratic).
@inline function ad_ve_deriv_x_2d(field, is_solid, i, j, Nx, inv_dx, inv_2dx, west_bc, east_bc)
    T = eltype(field)
    li = ad_ve_bc_index_1d(i - 1, Nx, west_bc, east_bc)
    ri = ad_ve_bc_index_1d(i + 1, Nx, west_bc, east_bc)
    left = li != 0 && !is_solid[li, j]
    right = ri != 0 && !is_solid[ri, j]
    if left && right
        return (field[ri, j] - field[li, j]) * inv_2dx
    elseif right
        r2i = ad_ve_bc_index_1d(i + 2, Nx, west_bc, east_bc)
        return (r2i != 0 && !is_solid[r2i, j]) ?
               (-T(3) * field[i, j] + T(4) * field[ri, j] - field[r2i, j]) * inv_2dx :
               (field[ri, j] - field[i, j]) * inv_dx
    elseif left
        l2i = ad_ve_bc_index_1d(i - 2, Nx, west_bc, east_bc)
        return (l2i != 0 && !is_solid[l2i, j]) ?
               (T(3) * field[i, j] - T(4) * field[li, j] + field[l2i, j]) * inv_2dx :
               (field[i, j] - field[li, j]) * inv_dx
    else
        return zero(T)
    end
end

@inline function ad_ve_deriv_y_2d(field, is_solid, i, j, Ny, inv_dy, inv_2dy, south_bc, north_bc)
    T = eltype(field)
    dj = ad_ve_bc_index_1d(j - 1, Ny, south_bc, north_bc)
    uj = ad_ve_bc_index_1d(j + 1, Ny, south_bc, north_bc)
    down = dj != 0 && !is_solid[i, dj]
    up = uj != 0 && !is_solid[i, uj]
    if down && up
        return (field[i, uj] - field[i, dj]) * inv_2dy
    elseif up
        u2j = ad_ve_bc_index_1d(j + 2, Ny, south_bc, north_bc)
        return (u2j != 0 && !is_solid[i, u2j]) ?
               (-T(3) * field[i, j] + T(4) * field[i, uj] - field[i, u2j]) * inv_2dy :
               (field[i, uj] - field[i, j]) * inv_dy
    elseif down
        d2j = ad_ve_bc_index_1d(j - 2, Ny, south_bc, north_bc)
        return (d2j != 0 && !is_solid[i, d2j]) ?
               (T(3) * field[i, j] - T(4) * field[i, dj] + field[i, d2j]) * inv_2dy :
               (field[i, j] - field[i, dj]) * inv_dy
    else
        return zero(T)
    end
end

# ----------------------------------------------------------------------------
# MUSCL-Superbee scalar advection (mirror operators_2d_advection.jl).
@inline function ad_ve_superbee_limiter(r)
    one_r = one(r)
    two_r = one_r + one_r
    return max(zero(r), max(min(two_r * r, one_r), min(r, two_r)))
end

@inline function ad_ve_muscl_superbee_face_value(far_upwind, upwind, downwind)
    d_up = upwind - far_upwind
    d_down = downwind - upwind
    r = ifelse(d_down == zero(d_down), zero(d_down), d_up / d_down)
    return upwind + (one(r) / (one(r) + one(r))) * ad_ve_superbee_limiter(r) * d_down
end

# ----------------------------------------------------------------------------
# Embedded geometry container (mirror the subset of FVFDEmbeddedBoundary2D we
# use; scratch struct `EmbeddedGeom`).
struct ADVEEmbeddedGeom
    cell_fraction::Matrix{Float64}
    west_fraction::Matrix{Float64}
    east_fraction::Matrix{Float64}
    south_fraction::Matrix{Float64}
    north_fraction::Matrix{Float64}
    wall_fraction::Matrix{Float64}
    wall_nx::Matrix{Float64}
    wall_ny::Matrix{Float64}
    wall_inv_distance_to_center::Matrix{Float64}
    is_solid::BitMatrix
    cut_count::Matrix{UInt8}
end

# ---- circle cut-cell builder (mirror fvfd_embedded_boundary_from_circle_2d) ----
@inline function ad_ve_interval_overlap_1d(a0, a1, b0, b1)
    return max(zero(a0), min(a1, b1) - max(a0, b0))
end

@inline function ad_ve_circle_vface_frac(x_face, y0, y1, cx, cy, radius)
    dx = x_face - cx
    abs(dx) >= radius && return 1.0
    half_inside = sqrt(max(0.0, radius * radius - dx * dx))
    inside = ad_ve_interval_overlap_1d(y0, y1, cy - half_inside, cy + half_inside)
    return min(1.0, max(0.0, 1.0 - inside / (y1 - y0)))
end

@inline function ad_ve_circle_hface_frac(y_face, x0, x1, cx, cy, radius)
    dy = y_face - cy
    abs(dy) >= radius && return 1.0
    half_inside = sqrt(max(0.0, radius * radius - dy * dy))
    inside = ad_ve_interval_overlap_1d(x0, x1, cx - half_inside, cx + half_inside)
    return min(1.0, max(0.0, 1.0 - inside / (x1 - x0)))
end

function ad_ve_circle_cell_moments(x0, y0, cx, cy, radius, samples)
    inv_s = 1.0 / samples
    r2 = radius * radius
    outside = 0
    xs = 0.0; ys = 0.0
    @inbounds for sj in 1:samples, si in 1:samples
        x = x0 + (si - 0.5) * inv_s
        y = y0 + (sj - 0.5) * inv_s
        dx = x - cx; dy = y - cy
        if dx * dx + dy * dy >= r2
            outside += 1; xs += x; ys += y
        end
    end
    cell = outside / (samples * samples)
    if outside == 0
        return (cell_fraction=cell, centroid_x=x0 + 0.5, centroid_y=y0 + 0.5)
    end
    return (cell_fraction=cell, centroid_x=xs / outside, centroid_y=ys / outside)
end

"""
    ad_ve_build_circle_geom(Nx, Ny, cx, cy, radius; samples=16) -> ADVEEmbeddedGeom

Build the FVFD circle cut-cell geometry (control-volume frame, cells centered at
`(i-0.5, j-0.5)`). Mirror of `fvfd_embedded_boundary_from_circle_2d`. The
returned `is_solid` is the FVFD cell-fraction mask; the matched operator replaces
it with the LBM node mask via `ad_ve_build_matched_geom`.
"""
function ad_ve_build_circle_geom(Nx, Ny, cx, cy, radius; samples=16)
    cf = ones(Float64, Nx, Ny)
    wf = ones(Float64, Nx, Ny); ef = ones(Float64, Nx, Ny)
    sf = ones(Float64, Nx, Ny); nf = ones(Float64, Nx, Ny)
    wall = zeros(Float64, Nx, Ny)
    wnx = zeros(Float64, Nx, Ny); wny = zeros(Float64, Nx, Ny)
    widc = zeros(Float64, Nx, Ny)
    cut = zeros(UInt8, Nx, Ny)
    tol = sqrt(eps(Float64))
    @inbounds for j in 1:Ny, i in 1:Nx
        x0 = Float64(i - 1); x1 = Float64(i)
        y0 = Float64(j - 1); y1 = Float64(j)
        west = ad_ve_circle_vface_frac(x0, y0, y1, cx, cy, radius)
        east = ad_ve_circle_vface_frac(x1, y0, y1, cx, cy, radius)
        south = ad_ve_circle_hface_frac(y0, x0, x1, cx, cy, radius)
        north = ad_ve_circle_hface_frac(y1, x0, x1, cx, cy, radius)
        m = ad_ve_circle_cell_moments(x0, y0, cx, cy, radius, samples)
        cell = m.cell_fraction
        wf[i, j] = west; ef[i, j] = east; sf[i, j] = south; nf[i, j] = north
        cf[i, j] = cell
        area_x = west - east
        area_y = south - north
        len = hypot(area_x, area_y)
        wall[i, j] = len
        if len > tol && tol < cell < 1.0 - tol
            wnx[i, j] = -area_x / len
            wny[i, j] = -area_y / len
            xc = (x0 + x1) / 2; yc = (y0 + y1) / 2
            scd = hypot(xc - cx, yc - cy) - radius
            dist = scd + wnx[i, j] * (m.centroid_x - xc) + wny[i, j] * (m.centroid_y - yc)
            dist = max(dist, eps(Float64))
            widc[i, j] = 1.0 / max(dist, eps(Float64))
            cut[i, j] = UInt8(1)
        end
    end
    is_solid = cf .<= tol
    return ADVEEmbeddedGeom(cf, wf, ef, sf, nf, wall, wnx, wny, widc, is_solid, cut)
end

"""
    ad_ve_build_matched_geom(g, lbm_solid) -> ADVEEmbeddedGeom

Return a copy of `g` whose `is_solid` field is the production LBM node mask
`lbm_solid`, keeping all FVFD fractions / wall geometry / cut_count untouched
(fix #1). `g.is_solid` (FVFD cell-fraction mask) and the LBM mask differ at a
handful of wall cells; the production VE step advects psi on the LBM mask.
"""
function ad_ve_build_matched_geom(g::ADVEEmbeddedGeom, lbm_solid)
    return ADVEEmbeddedGeom(
        g.cell_fraction, g.west_fraction, g.east_fraction, g.south_fraction,
        g.north_fraction, g.wall_fraction, g.wall_nx, g.wall_ny,
        g.wall_inv_distance_to_center, BitMatrix(lbm_solid), g.cut_count)
end

# ----------------------------------------------------------------------------
# Embedded face fractions (mirror _fvfd_xface_fraction_2d / _yface_fraction_2d).
@inline function ad_ve_xface_frac(is_solid, west_fraction, east_fraction, il, ir, j)
    return (is_solid[il, j] || is_solid[ir, j]) ? 0.0 :
           min(east_fraction[il, j], west_fraction[ir, j])
end
@inline function ad_ve_yface_frac(is_solid, south_fraction, north_fraction, i, jd, ju)
    return (is_solid[i, jd] || is_solid[i, ju]) ? 0.0 :
           min(north_fraction[i, jd], south_fraction[i, ju])
end
@inline function ad_ve_xface_avg0(ux, is_solid, il, ir, j)
    return (is_solid[il, j] || is_solid[ir, j]) ? 0.0 : (ux[il, j] + ux[ir, j]) / 2
end
@inline function ad_ve_yface_avg0(uy, is_solid, i, jd, ju)
    return (is_solid[i, jd] || is_solid[i, ju]) ? 0.0 : (uy[i, jd] + uy[i, ju]) / 2
end
@inline function ad_ve_xface_scalar0(field, is_solid, il, ir, j)
    return (is_solid[il, j] || is_solid[ir, j]) ? 0.0 : (field[il, j] + field[ir, j]) / 2
end
@inline function ad_ve_yface_scalar0(field, is_solid, i, jd, ju)
    return (is_solid[i, jd] || is_solid[i, ju]) ? 0.0 : (field[i, jd] + field[i, ju]) / 2
end

# Embedded cell -> face velocity, with the WEST face = inlet profile (fix #2b).
# Mirror fvfd_cell_velocity_to_faces_embedded_2d (west/east OPEN, south/north
# WALL) + the west-profile override (operators_2d.jl L262, scratch
# cell_velocity_to_faces_embedded_westprofile!).
function ad_ve_cell_velocity_to_faces_westprofile!(ux_face, uy_face, ux, uy,
                                                   g::ADVEEmbeddedGeom, Nx, Ny, u_profile)
    is_solid = g.is_solid
    @inbounds for J in 1:Ny, I in 1:(Nx + 1)
        if I == 1
            ux_face[I, J] = is_solid[1, J] ? 0.0 : g.west_fraction[1, J] * u_profile[J]
        elseif I == Nx + 1
            ux_face[I, J] = is_solid[Nx, J] ? 0.0 : g.east_fraction[Nx, J] * ux[Nx, J]
        else
            frac = ad_ve_xface_frac(is_solid, g.west_fraction, g.east_fraction, I - 1, I, J)
            ux_face[I, J] = frac * ad_ve_xface_avg0(ux, is_solid, I - 1, I, J)
        end
    end
    @inbounds for J in 1:(Ny + 1), I in 1:Nx
        if J == 1
            uy_face[I, J] = 0.0            # WALL south
        elseif J == Ny + 1
            uy_face[I, J] = 0.0            # WALL north
        else
            frac = ad_ve_yface_frac(is_solid, g.south_fraction, g.north_fraction, I, J - 1, J)
            uy_face[I, J] = frac * ad_ve_yface_avg0(uy, is_solid, I, J - 1, J)
        end
    end
    return nothing
end

# M42 cylinder-band test (mirror _is_cylinder_band_2d).
@inline function ad_ve_is_cylinder_band(is_solid, i, j, Nx, Ny)
    if i <= 2 || i >= Nx - 1 || j <= 2 || j >= Ny - 1
        return false
    end
    return is_solid[i - 2, j] | is_solid[i - 1, j] |
           is_solid[i + 1, j] | is_solid[i + 2, j] |
           is_solid[i, j - 2] | is_solid[i, j - 1] |
           is_solid[i, j + 1] | is_solid[i, j + 2]
end

# pass-2 cylinder-band one-sided MUSCL relax rhs (mirror _fvfd_muscl_relax_rhs_2d).
@inline function ad_ve_muscl_relax_rhs(phi, ux_face, uy_face, is_solid, i, j, Nx, Ny,
                                       inv_dx, inv_dy)
    ue = ux_face[i + 1, j]; uw = ux_face[i, j]
    vn = uy_face[i, j + 1]; vs = uy_face[i, j]
    phie = if ue >= 0.0
        (i > 1 && !is_solid[i - 1, j] && !is_solid[i + 1, j]) ?
            ad_ve_muscl_superbee_face_value(phi[i - 1, j], phi[i, j], phi[i + 1, j]) : phi[i, j]
    else
        (i + 2 <= Nx && !is_solid[i + 2, j] && !is_solid[i + 1, j]) ?
            ad_ve_muscl_superbee_face_value(phi[i + 2, j], phi[i + 1, j], phi[i, j]) : phi[i + 1, j]
    end
    phiw = if uw >= 0.0
        (i - 2 >= 1 && !is_solid[i - 2, j] && !is_solid[i - 1, j]) ?
            ad_ve_muscl_superbee_face_value(phi[i - 2, j], phi[i - 1, j], phi[i, j]) : phi[i - 1, j]
    else
        (i + 1 <= Nx && !is_solid[i + 1, j] && !is_solid[i - 1, j]) ?
            ad_ve_muscl_superbee_face_value(phi[i + 1, j], phi[i, j], phi[i - 1, j]) : phi[i, j]
    end
    phin = if vn >= 0.0
        (j > 1 && !is_solid[i, j - 1] && !is_solid[i, j + 1]) ?
            ad_ve_muscl_superbee_face_value(phi[i, j - 1], phi[i, j], phi[i, j + 1]) : phi[i, j]
    else
        (j + 2 <= Ny && !is_solid[i, j + 2] && !is_solid[i, j + 1]) ?
            ad_ve_muscl_superbee_face_value(phi[i, j + 2], phi[i, j + 1], phi[i, j]) : phi[i, j + 1]
    end
    phis = if vs >= 0.0
        (j - 2 >= 1 && !is_solid[i, j - 2] && !is_solid[i, j - 1]) ?
            ad_ve_muscl_superbee_face_value(phi[i, j - 2], phi[i, j - 1], phi[i, j]) : phi[i, j - 1]
    else
        (j + 1 <= Ny && !is_solid[i, j + 1] && !is_solid[i, j - 1]) ?
            ad_ve_muscl_superbee_face_value(phi[i, j + 1], phi[i, j], phi[i, j - 1]) : phi[i, j]
    end
    flux_div = (ue * phie - uw * phiw) * inv_dx + (vn * phin - vs * phis) * inv_dy
    divu = (ue - uw) * inv_dx + (vn - vs) * inv_dy
    return -(flux_div - phi[i, j] * divu)
end

# 2-pass MUSCL-relax advection with production Dirichlet edge BC (west=0 at i=1,
# east=east_phi[j] at i=Nx, south/north WALL mirror). dt=1. Mirror of scratch
# `advect_prodbc` (fix #2a). Pass-1 uses the inline `rus` rusanov at the edge/
# solid band; pass-2 overwrites cylinder-band cells.
function ad_ve_advect_prodbc(phi, ux_face, uy_face, is_solid, Nx, Ny, east_phi)
    adv = zeros(Nx, Ny)
    wbc(i, j) = i > 1 ? phi[i - 1, j] : 0.0
    ebc(i, j) = i < Nx ? phi[i + 1, j] : east_phi[j]
    sbc(i, j) = j > 1 ? phi[i, j - 1] : phi[i, j]
    nbc(i, j) = j < Ny ? phi[i, j + 1] : phi[i, j]
    rus(i, j) = begin
        ue = ux_face[i + 1, j]; uw = ux_face[i, j]; vn = uy_face[i, j + 1]; vs = uy_face[i, j]
        phie = ue >= 0 ? phi[i, j] : ebc(i, j)
        phiw = uw >= 0 ? wbc(i, j) : phi[i, j]
        phin = vn >= 0 ? phi[i, j] : nbc(i, j)
        phis = vs >= 0 ? sbc(i, j) : phi[i, j]
        fl = (ue * phie - uw * phiw) + (vn * phin - vs * phis); du = (ue - uw) + (vn - vs)
        -(fl - phi[i, j] * du)
    end
    @inbounds for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]
            adv[i, j] = 0.0; continue
        end
        if i <= 2 || i >= Nx - 1 || j <= 2 || j >= Ny - 1 ||
           is_solid[i - 2, j] || is_solid[i - 1, j] || is_solid[i + 1, j] || is_solid[i + 2, j] ||
           is_solid[i, j - 2] || is_solid[i, j - 1] || is_solid[i, j + 1] || is_solid[i, j + 2]
            adv[i, j] = phi[i, j] + rus(i, j)
        else
            adv[i, j] = phi[i, j] + ad_ve_muscl_relax_rhs(phi, ux_face, uy_face, is_solid, i, j, Nx, Ny, 1.0, 1.0)
        end
    end
    @inbounds for j in 1:Ny, i in 1:Nx
        if !is_solid[i, j] && ad_ve_is_cylinder_band(is_solid, i, j, Nx, Ny)
            adv[i, j] = phi[i, j] + ad_ve_muscl_relax_rhs(phi, ux_face, uy_face, is_solid, i, j, Nx, Ny, 1.0, 1.0)
        end
    end
    return adv
end

# Embedded velocity gradient with wall-gradient correction on a FROZEN field
# (mirror _fvfd_apply_embedded_wall_gradient_2d).
@inline function ad_ve_apply_embedded_wall_gradient(gx, gy, phi, wnx, wny, widc, i, j)
    inv_d = widc[i, j]
    if inv_d > 0.0
        nx = wnx[i, j]; ny = wny[i, j]
        target = phi[i, j] * inv_d
        current = gx * nx + gy * ny
        corr = target - current
        return gx + corr * nx, gy + corr * ny
    end
    return gx, gy
end

# embedded ∇·tau (mirror fvfd_tensor_divergence_embedded_2d_kernel!).
# BC: OPEN-x / WALL-y. At OPEN/WALL edges the kernel uses the local-cell value.
function ad_ve_tensor_divergence_embedded!(fx, fy, tauxx, tauxy, tauyy, g::ADVEEmbeddedGeom,
                                           Nx, Ny, inv_dx, inv_dy)
    is_solid = g.is_solid
    @inbounds for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]
            fx[i, j] = 0.0; fy[i, j] = 0.0
            continue
        end
        # east face
        if i < Nx
            e_frac = ad_ve_xface_frac(is_solid, g.west_fraction, g.east_fraction, i, i + 1, j)
            e_xx = ad_ve_xface_scalar0(tauxx, is_solid, i, i + 1, j)
            e_xy = ad_ve_xface_scalar0(tauxy, is_solid, i, i + 1, j)
        else
            e_frac = g.east_fraction[i, j]; e_xx = tauxx[i, j]; e_xy = tauxy[i, j]
        end
        # west face
        if i > 1
            w_frac = ad_ve_xface_frac(is_solid, g.west_fraction, g.east_fraction, i - 1, i, j)
            w_xx = ad_ve_xface_scalar0(tauxx, is_solid, i - 1, i, j)
            w_xy = ad_ve_xface_scalar0(tauxy, is_solid, i - 1, i, j)
        else
            w_frac = g.west_fraction[i, j]; w_xx = tauxx[i, j]; w_xy = tauxy[i, j]
        end
        # north face
        if j < Ny
            n_frac = ad_ve_yface_frac(is_solid, g.south_fraction, g.north_fraction, i, j, j + 1)
            n_xy = ad_ve_yface_scalar0(tauxy, is_solid, i, j, j + 1)
            n_yy = ad_ve_yface_scalar0(tauyy, is_solid, i, j, j + 1)
        else
            n_frac = g.north_fraction[i, j]; n_xy = tauxy[i, j]; n_yy = tauyy[i, j]
        end
        # south face
        if j > 1
            s_frac = ad_ve_yface_frac(is_solid, g.south_fraction, g.north_fraction, i, j - 1, j)
            s_xy = ad_ve_yface_scalar0(tauxy, is_solid, i, j - 1, j)
            s_yy = ad_ve_yface_scalar0(tauyy, is_solid, i, j - 1, j)
        else
            s_frac = g.south_fraction[i, j]; s_xy = tauxy[i, j]; s_yy = tauyy[i, j]
        end
        volume_fraction = max(g.cell_fraction[i, j], eps(Float64))
        wall_x_length = g.west_fraction[i, j] - g.east_fraction[i, j]
        wall_y_length = g.south_fraction[i, j] - g.north_fraction[i, j]
        fx[i, j] = (
            (e_frac * e_xx - w_frac * w_xx + wall_x_length * tauxx[i, j]) * inv_dx +
            (n_frac * n_xy - s_frac * s_xy + wall_y_length * tauxy[i, j]) * inv_dy
        ) / volume_fraction
        fy[i, j] = (
            (e_frac * e_xy - w_frac * w_xy + wall_x_length * tauxy[i, j]) * inv_dx +
            (n_frac * n_yy - s_frac * s_yy + wall_y_length * tauyy[i, j]) * inv_dy
        ) / volume_fraction
    end
    return nothing
end
