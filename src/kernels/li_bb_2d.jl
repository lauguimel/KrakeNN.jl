using KernelAbstractions

# =====================================================================
# Interpolated bounce-back (LI-BB / Bouzidi) + TRT collision, fused.
#
# For each fluid cell (i, j) and D2Q9 direction q pointing INTO a wall
# (i.e. the neighbour at (i+cx[q], j+cy[q]) is solid), we know the
# wall-cut fraction q_w ∈ (0, 1]:
#
#     q_w = 0  ⇒ wall exactly at the fluid node (ill-posed; we treat
#                 the cell as solid instead)
#     q_w = 1  ⇒ wall exactly at the solid node (same as halfway BB)
#     q_w = ½  ⇒ halfway bounce-back (recovered exactly)
#
# Precomputation from STL / analytic geometry writes q_w into an array
# `q_wall[Nx, Ny, 9]` with value 0 (a sentinel) when the link is fully
# in the fluid. Any q_wall[i,j,q] ∈ (0, 1] flags a cut link.
#
# Post-streaming formula (Bouzidi-Firdaouss-Lallemand 2001, Phys.
# Fluids 13, 3452; stabilised by TRT magic Λ = 3/16, Ginzburg &
# d'Humières 2003, Phys. Rev. E 68, 066614):
#
#     q_w ≤ 1/2 :  f_q̄(x_f) = 2 q_w f̃_q(x_f)
#                              + (1 − 2 q_w) f̃_q̄(x_f)         (linear)
#     q_w > 1/2 :  f_q̄(x_f) = (1/(2 q_w)) f̃_q(x_f)
#                              + (1 − 1/(2 q_w)) f_q̄(x_f, t)
#
# with q̄ the opposite direction and f̃ the post-collision pre-stream
# value at x_f. Adding Ladd's (1994) momentum correction gives a moving
# wall: append  ± (2 w_q ρ_w c_q · u_w)/c_s²  on the two branches.
#
# For the implementation we fuse:
#   (a) pull-stream with the standard halfway BB fallback at domain
#       edges (so cells not flagged by q_wall keep the fused_trt
#       behaviour);
#   (b) TRT collision;
#   (c) LI-BB overwrite on the populations whose incoming link is
#       flagged by q_wall.
#
# A single kernel launch per step.
# =====================================================================

# Bouzidi linear interpolated bounce-back (Bouzidi-Firdaouss-Lallemand
# 2001; consolidated notation Krüger et al. 2017, §5.3.4). For link q
# pointing INTO a wall at fluid node x_f, with wall-cut fraction q_w:
#
#   q_w ≤ 1/2 :  f_{q̄}(x_f, t+Δt) =
#                    2·q_w · f̃_q(x_f, t)
#                  + (1 − 2·q_w) · f̃_q(x_f − c_q, t)
#                  + δ
#
#   q_w > 1/2 :  f_{q̄}(x_f, t+Δt) =
#                    (1/(2 q_w)) · f̃_q(x_f, t)
#                  + (1 − 1/(2 q_w)) · f̃_{q̄}(x_f, t)
#                  + δ / q_w
#
# where f̃ denotes post-collision values, c_q points into the wall, and
# δ = −2 w_q (c_q · u_w) / c_s² is the moving-wall momentum correction
# (Ladd 1994). For stationary walls δ = 0 and these reduce to the
# classical Bouzidi linear bounce-back.
#
# Arguments:
#   q_w         — wall-cut fraction in (0, 1]
#   f_post_here — f̃_q at x_f (post-collision at this cell)
#   f_post_back — f̃_q at x_f − c_q (post-collision at the fluid
#                   neighbour OPPOSITE the wall; accessed here by
#                   reusing the pulled value fp_q of the current step)
#   f_bar_post_here — f̃_{q̄} at x_f (post-collision at this cell)
#   δ          — −(2 w_q / c_s²) · (c_q · u_w)
@inline function _libb_branch(q_w::T, f_post_here::T,
                               f_post_back::T, f_bar_post_here::T,
                               δ::T) where {T}
    half = T(0.5)
    if q_w ≤ half
        return T(2) * q_w * f_post_here +
               (one(T) - T(2) * q_w) * f_post_back + δ
    else
        inv_two_q = one(T) / (T(2) * q_w)
        return inv_two_q * f_post_here +
               (one(T) - inv_two_q) * f_bar_post_here +
               δ * inv_two_q
    end
end

@kernel function fused_trt_libb_step_kernel!(f_out, @Const(f_in),
                                              ρ_out, ux_out, uy_out,
                                              @Const(is_solid),
                                              @Const(q_wall),
                                              @Const(uw_link_x),
                                              @Const(uw_link_y),
                                              Nx, Ny,
                                              s_plus, s_minus)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f_out)
        # 1. Pull-stream (same halfway BB at domain boundary as fused_trt).
        # Neighbor indices are clamped in-range so the unselected ifelse
        # branch never indexes at 0 / Nx+1 / Ny+1 (Julia 1.12 evaluates
        # both ifelse branches).
        im1 = ifelse(i > 1,  i - 1, i)
        ip1 = ifelse(i < Nx, i + 1, i)
        jm1 = ifelse(j > 1,  j - 1, j)
        jp1 = ifelse(j < Ny, j + 1, j)
        fp1 = f_in[i, j, 1]
        fp2 = ifelse(i > 1,             f_in[im1, j,   2], f_in[i, j, 4])
        fp3 = ifelse(j > 1,             f_in[i,   jm1, 3], f_in[i, j, 5])
        fp4 = ifelse(i < Nx,            f_in[ip1, j,   4], f_in[i, j, 2])
        fp5 = ifelse(j < Ny,            f_in[i,   jp1, 5], f_in[i, j, 3])
        fp6 = ifelse(i > 1  && j > 1,   f_in[im1, jm1, 6], f_in[i, j, 8])
        fp7 = ifelse(i < Nx && j > 1,   f_in[ip1, jm1, 7], f_in[i, j, 9])
        fp8 = ifelse(i < Nx && j < Ny,  f_in[ip1, jp1, 8], f_in[i, j, 6])
        fp9 = ifelse(i > 1  && j < Ny,  f_in[im1, jp1, 9], f_in[i, j, 7])

        if is_solid[i, j]
            # Interior solid cell: plain bounce-back (matches fused_trt)
            f_out[i, j, 1] = fp1
            f_out[i, j, 2] = fp4; f_out[i, j, 4] = fp2
            f_out[i, j, 3] = fp5; f_out[i, j, 5] = fp3
            f_out[i, j, 6] = fp8; f_out[i, j, 8] = fp6
            f_out[i, j, 7] = fp9; f_out[i, j, 9] = fp7
            ρ_out[i, j] = one(T); ux_out[i, j] = zero(T); uy_out[i, j] = zero(T)
        else
            # 2. TRT collision (compact form f* = f − a(f − feq) − b(f_bar − feq_bar))
            ρ, ux, uy = moments_2d(fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9)
            usq = ux * ux + uy * uy
            feq1 = feq_2d(Val(1), ρ, ux, uy, usq)
            feq2 = feq_2d(Val(2), ρ, ux, uy, usq)
            feq3 = feq_2d(Val(3), ρ, ux, uy, usq)
            feq4 = feq_2d(Val(4), ρ, ux, uy, usq)
            feq5 = feq_2d(Val(5), ρ, ux, uy, usq)
            feq6 = feq_2d(Val(6), ρ, ux, uy, usq)
            feq7 = feq_2d(Val(7), ρ, ux, uy, usq)
            feq8 = feq_2d(Val(8), ρ, ux, uy, usq)
            feq9 = feq_2d(Val(9), ρ, ux, uy, usq)
            a = (s_plus + s_minus) * T(0.5)
            b = (s_plus - s_minus) * T(0.5)
            # Post-collision populations (pre-LI-BB overwrite)
            fp1c = fp1 - s_plus * (fp1 - feq1)
            fp2c = fp2 - a * (fp2 - feq2) - b * (fp4 - feq4)
            fp4c = fp4 - a * (fp4 - feq4) - b * (fp2 - feq2)
            fp3c = fp3 - a * (fp3 - feq3) - b * (fp5 - feq5)
            fp5c = fp5 - a * (fp5 - feq5) - b * (fp3 - feq3)
            fp6c = fp6 - a * (fp6 - feq6) - b * (fp8 - feq8)
            fp8c = fp8 - a * (fp8 - feq8) - b * (fp6 - feq6)
            fp7c = fp7 - a * (fp7 - feq7) - b * (fp9 - feq9)
            fp9c = fp9 - a * (fp9 - feq9) - b * (fp7 - feq7)

            # 3. LI-BB overwrite on cut links.  For each direction q that
            # is FLAGGED by q_wall[i,j,q] > 0, the incoming population
            # f_{opposite(q)} at this fluid node must be replaced by the
            # Bouzidi-interpolated value. q in KrakeNN's convention:
            #   (2,4) east-west, (3,5) north-south, (6,8) NE-SW, (7,9) NW-SE.
            # Lattice weights: w_2=w_3=w_4=w_5=1/9, w_6..9=1/36. c_s²=1/3.
            # Moving-wall correction δ_q = 2 w_q ρ (c_q · u_w) / c_s²
            #                            = 6 w_q (c_q · u_w)   (ρ = 1 lattice)
            # LI-BB overwrite. For each cut link we read the per-link
            # wall velocity (evaluated at the wall-intersection point)
            # from uw_link_{x,y}[i, j, q], build the moving-wall
            # correction δ = 6 w_{q̄} (c_{q̄} · u_wall), and apply the
            # Bouzidi branch. If the link is not flagged, fall through
            # to the plain TRT-collided value.

            # _libb_branch arg order:
            #   q_w, f̃_q(x_f), f̃_q(x_f − c_q), f̃_{q̄}(x_f), δ_{q̄}
            #
            # "f̃_q(x_f − c_q)" is the post-collision value of direction
            # q at the fluid neighbour opposite the wall. In a fused
            # pull-stream-collide kernel this equals the current-step
            # pulled value fp_q (which came from f_in at the opposite
            # neighbour, itself post-collision from the previous step).
            # δ_{q̄} = −(2 w_{q̄}/c_s²) · (c_{q̄} · u_wall).

            # Pair (2, 4) east / west
            qw2 = q_wall[i, j, 2]
            if qw2 > zero(T)
                δ4 = -T(2/3) * uw_link_x[i, j, 2]
                fp4_new = _libb_branch(qw2, fp2c, fp2, fp4c, δ4)
            else
                fp4_new = fp4c
            end
            qw4 = q_wall[i, j, 4]
            if qw4 > zero(T)
                δ2 =  T(2/3) * uw_link_x[i, j, 4]
                fp2_new = _libb_branch(qw4, fp4c, fp4, fp2c, δ2)
            else
                fp2_new = fp2c
            end
            # Pair (3, 5) north / south
            qw3 = q_wall[i, j, 3]
            if qw3 > zero(T)
                δ5 = -T(2/3) * uw_link_y[i, j, 3]
                fp5_new = _libb_branch(qw3, fp3c, fp3, fp5c, δ5)
            else
                fp5_new = fp5c
            end
            qw5 = q_wall[i, j, 5]
            if qw5 > zero(T)
                δ3 =  T(2/3) * uw_link_y[i, j, 5]
                fp3_new = _libb_branch(qw5, fp5c, fp5, fp3c, δ3)
            else
                fp3_new = fp3c
            end
            # Pair (6, 8) NE / SW
            qw6 = q_wall[i, j, 6]
            if qw6 > zero(T)
                uxw6 = uw_link_x[i, j, 6]; uyw6 = uw_link_y[i, j, 6]
                δ8 = -T(1/6) * (uxw6 + uyw6)
                fp8_new = _libb_branch(qw6, fp6c, fp6, fp8c, δ8)
            else
                fp8_new = fp8c
            end
            qw8 = q_wall[i, j, 8]
            if qw8 > zero(T)
                uxw8 = uw_link_x[i, j, 8]; uyw8 = uw_link_y[i, j, 8]
                δ6 =  T(1/6) * (uxw8 + uyw8)
                fp6_new = _libb_branch(qw8, fp8c, fp8, fp6c, δ6)
            else
                fp6_new = fp6c
            end
            # Pair (7, 9) NW / SE
            qw7 = q_wall[i, j, 7]
            if qw7 > zero(T)
                uxw7 = uw_link_x[i, j, 7]; uyw7 = uw_link_y[i, j, 7]
                δ9 = -T(1/6) * (-uxw7 + uyw7)
                fp9_new = _libb_branch(qw7, fp7c, fp7, fp9c, δ9)
            else
                fp9_new = fp9c
            end
            qw9 = q_wall[i, j, 9]
            if qw9 > zero(T)
                uxw9 = uw_link_x[i, j, 9]; uyw9 = uw_link_y[i, j, 9]
                δ7 =  T(1/6) * (-uxw9 + uyw9)
                fp7_new = _libb_branch(qw9, fp9c, fp9, fp7c, δ7)
            else
                fp7_new = fp7c
            end

            f_out[i, j, 1] = fp1c
            f_out[i, j, 2] = fp2_new; f_out[i, j, 4] = fp4_new
            f_out[i, j, 3] = fp3_new; f_out[i, j, 5] = fp5_new
            f_out[i, j, 6] = fp6_new; f_out[i, j, 8] = fp8_new
            f_out[i, j, 7] = fp7_new; f_out[i, j, 9] = fp9_new

            ρ_out[i, j] = ρ
            ux_out[i, j] = ux
            uy_out[i, j] = uy
        end
    end
end

"""
    precompute_q_wall_cylinder(Nx, Ny, cx, cy, R; FT=Float64)
                               -> (q_wall, is_solid)

Analytically precompute the wall-cut fraction `q_wall[i, j, q]` for a
cylinder centred at `(cx, cy)` with radius `R`, embedded in an `Nx×Ny`
lattice whose node `(i, j)` sits at physical coordinates `(i−1, j−1)`.

The returned `q_wall` is a `Float` array of size `(Nx, Ny, 9)` with the
convention:

  * `q_wall[i, j, q] = 0` → link `q` out of node `(i, j)` stays in the
    fluid (no wall crossing).
  * `q_wall[i, j, q] ∈ (0, 1]` → link `q` crosses the wall at that
    fraction of the link length from the fluid node.

The companion `is_solid[i, j]` boolean mask flags fluid nodes that are
actually inside the cylinder (and therefore treated with plain
bounce-back in the fused kernel).
"""
function precompute_q_wall_cylinder(Nx::Int, Ny::Int,
                                     cx::Real, cy::Real, R::Real;
                                     FT::Type{<:AbstractFloat}=Float64)
    cxT, cyT, RT = FT(cx), FT(cy), FT(R)
    R² = RT * RT
    is_solid = zeros(Bool, Nx, Ny)
    q_wall = zeros(FT, Nx, Ny, 9)
    cxs = velocities_x(D2Q9())
    cys = velocities_y(D2Q9())

    @inbounds for j in 1:Ny, i in 1:Nx
        xf = FT(i - 1); yf = FT(j - 1)
        dx_f = xf - cxT; dy_f = yf - cyT
        if dx_f * dx_f + dy_f * dy_f ≤ R²
            is_solid[i, j] = true
            continue
        end
        for q in 2:9
            cqx = FT(cxs[q]); cqy = FT(cys[q])
            # Neighbour along link q — only consider if it lies inside
            # the cylinder (otherwise the link is not cut)
            xn = xf + cqx; yn = yf + cqy
            dx_n = xn - cxT; dy_n = yn - cyT
            if dx_n * dx_n + dy_n * dy_n > R²
                continue
            end
            # Solve |x_f + t c_q − c_wall|² = R² for t ∈ (0, 1]
            a = cqx * cqx + cqy * cqy
            b = FT(2) * (dx_f * cqx + dy_f * cqy)
            c = dx_f * dx_f + dy_f * dy_f - R²
            disc = b * b - FT(4) * a * c
            if disc < zero(FT)
                continue
            end
            sd = sqrt(disc)
            t1 = (-b - sd) / (FT(2) * a)
            t2 = (-b + sd) / (FT(2) * a)
            t = t1 > zero(FT) ? t1 : t2
            if t > zero(FT) && t ≤ one(FT)
                q_wall[i, j, q] = t
            end
        end
    end
    return q_wall, is_solid
end

"""
    fused_trt_libb_step!(f_out, f_in, ρ, ux, uy, is_solid,
                          q_wall, uw_x, uw_y, Nx, Ny, ν; Λ=3/16)

Fused GPU step: pull-stream + halfway BB on domain edges + TRT
collision + interpolated bounce-back (Bouzidi) on flagged cut links.
`q_wall[i,j,q]` holds the wall-cut fraction per link (0 if no
crossing, (0,1] otherwise). `uw_x`, `uw_y` are per-cell prescribed
wall velocities; pass zero arrays for stationary walls.

Sets the symmetric rate via TRT magic `Λ = 3/16` by default, making
the bounce-back error viscosity-independent (Ginzburg 2003). Pass a
different `Λ` for other choices.
"""
function fused_trt_libb_step!(f_out, f_in, ρ, ux, uy, is_solid,
                               q_wall, uw_link_x, uw_link_y,
                               Nx, Ny, ν; Λ::Real=3/16)
    backend = KernelAbstractions.get_backend(f_in)
    ET = eltype(f_in)
    s_plus, s_minus = trt_rates(ν; Λ=Λ)
    kernel! = fused_trt_libb_step_kernel!(backend)
    kernel!(f_out, f_in, ρ, ux, uy, is_solid,
            q_wall, uw_link_x, uw_link_y,
            Nx, Ny, ET(s_plus), ET(s_minus);
            ndrange=(Nx, Ny))
end

"""
    precompute_q_wall_annulus(Nx, Ny, cx, cy, R_inner, R_outer; FT=Float64)
                              -> (q_wall, is_solid, is_inner_wall)

Precompute cut fractions for a Taylor-Couette-style annular fluid
domain: fluid in `R_inner < r < R_outer`, solid elsewhere. Each cut
link `q_wall[i, j, q] ∈ (0, 1]` records the fraction to the first wall
the link crosses. `is_inner_wall[i, j, q]` is `true` if that wall is
the inner cylinder (so a moving-wall correction applies there), else
`false` for the outer cylinder.
"""
function precompute_q_wall_annulus(Nx::Int, Ny::Int,
                                    cx::Real, cy::Real,
                                    R_inner::Real, R_outer::Real;
                                    FT::Type{<:AbstractFloat}=Float64)
    R_inner < R_outer || error("precompute_q_wall_annulus: R_inner < R_outer required")
    cxT, cyT = FT(cx), FT(cy)
    Ri², Ro² = FT(R_inner)^2, FT(R_outer)^2
    is_solid = zeros(Bool, Nx, Ny)
    q_wall = zeros(FT, Nx, Ny, 9)
    is_inner_wall = zeros(Bool, Nx, Ny, 9)
    cxs = velocities_x(D2Q9())
    cys = velocities_y(D2Q9())

    @inbounds for j in 1:Ny, i in 1:Nx
        xf = FT(i - 1); yf = FT(j - 1)
        dx_f = xf - cxT; dy_f = yf - cyT
        r² = dx_f * dx_f + dy_f * dy_f
        if r² ≤ Ri² || r² ≥ Ro²
            is_solid[i, j] = true
            continue
        end
        for q in 2:9
            cqx = FT(cxs[q]); cqy = FT(cys[q])
            xn = xf + cqx; yn = yf + cqy
            dx_n = xn - cxT; dy_n = yn - cyT
            rn² = dx_n * dx_n + dy_n * dy_n
            # Only flag if the neighbour is inside a solid region
            if rn² > Ri² && rn² < Ro²
                continue
            end
            # Solve quadratic |x_f + t c_q − c|² = R² for both radii,
            # keep the smallest positive root in (0, 1].
            a = cqx * cqx + cqy * cqy
            b = FT(2) * (dx_f * cqx + dy_f * cqy)
            t_best = FT(Inf)
            hit_inner = false
            for (R², is_inner) in ((Ri², true), (Ro², false))
                c = r² - R²
                disc = b * b - FT(4) * a * c
                disc < zero(FT) && continue
                sd = sqrt(disc)
                t1 = (-b - sd) / (FT(2) * a)
                t2 = (-b + sd) / (FT(2) * a)
                for tc in (t1, t2)
                    if tc > zero(FT) && tc ≤ one(FT) && tc < t_best
                        t_best = tc
                        hit_inner = is_inner
                    end
                end
            end
            if t_best ≤ one(FT)
                q_wall[i, j, q] = t_best
                is_inner_wall[i, j, q] = hit_inner
            end
        end
    end
    return q_wall, is_solid, is_inner_wall
end

"""
    wall_velocity_rotating_inner(q_wall, is_inner_wall, cx, cy, Ω)
                                 -> (uw_link_x, uw_link_y)

For an annulus with rotating inner cylinder + stationary outer one,
set the per-link wall velocity to Ω × (x_w − c) on the inner wall and
zero on the outer wall.
"""
function wall_velocity_rotating_inner(q_wall::AbstractArray{T, 3},
                                       is_inner_wall::AbstractArray{Bool, 3},
                                       cx::Real, cy::Real, Ω::Real) where {T}
    Nx, Ny, _ = size(q_wall)
    cxT, cyT, ΩT = T(cx), T(cy), T(Ω)
    uw_link_x = zeros(T, Nx, Ny, 9)
    uw_link_y = zeros(T, Nx, Ny, 9)
    cxs = velocities_x(D2Q9())
    cys = velocities_y(D2Q9())
    @inbounds for q in 2:9, j in 1:Ny, i in 1:Nx
        qw = q_wall[i, j, q]
        (qw == zero(T) || !is_inner_wall[i, j, q]) && continue
        xw = T(i - 1) + qw * T(cxs[q])
        yw = T(j - 1) + qw * T(cys[q])
        uw_link_x[i, j, q] = -ΩT * (yw - cyT)
        uw_link_y[i, j, q] =  ΩT * (xw - cxT)
    end
    return uw_link_x, uw_link_y
end

"""
    wall_velocity_rotating_cylinder(q_wall, cx, cy, Ω; FT=Float64)
                                    -> (uw_link_x, uw_link_y)

For each cut link flagged by `q_wall`, evaluate the rigid-body
rotation velocity `u_w(x) = Ω × (x − c)` at the wall-intersection point
`x_w = x_f + q_w · c_q` (positive Ω is CCW).

Returns two `(Nx, Ny, 9)` arrays ready for `fused_trt_libb_step!`.
For stationary walls, pass `Ω = 0` — the result is zero everywhere.
"""
function wall_velocity_rotating_cylinder(q_wall::AbstractArray{T, 3},
                                          cx::Real, cy::Real, Ω::Real) where {T}
    Nx, Ny, _ = size(q_wall)
    cxT, cyT, ΩT = T(cx), T(cy), T(Ω)
    uw_link_x = zeros(T, Nx, Ny, 9)
    uw_link_y = zeros(T, Nx, Ny, 9)
    cxs = velocities_x(D2Q9())
    cys = velocities_y(D2Q9())
    @inbounds for q in 2:9, j in 1:Ny, i in 1:Nx
        qw = q_wall[i, j, q]
        qw == zero(T) && continue
        xw = T(i - 1) + qw * T(cxs[q])
        yw = T(j - 1) + qw * T(cys[q])
        uw_link_x[i, j, q] = -ΩT * (yw - cyT)
        uw_link_y[i, j, q] =  ΩT * (xw - cxT)
    end
    return uw_link_x, uw_link_y
end
