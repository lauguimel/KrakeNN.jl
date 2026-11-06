# BIT-MIRROR of production fused_trt_libb_v2_step! plus the inlet/outlet
# rebuild of `rebuild_inlet_outlet_libb_2d!` (Zou-He velocity west,
# NeqExtrapolationPressure east). If the production LI-BB/TRT/rebuild algebra
# changes, update this file too.

const AD_CXV = (0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0)
const AD_CYV = (0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0)
const AD_OPP = (1, 4, 5, 2, 3, 8, 9, 6, 7)

@inline ad_qcx(::Val{1}) = 0.0
@inline ad_qcx(::Val{2}) = 1.0
@inline ad_qcx(::Val{3}) = 0.0
@inline ad_qcx(::Val{4}) = -1.0
@inline ad_qcx(::Val{5}) = 0.0
@inline ad_qcx(::Val{6}) = 1.0
@inline ad_qcx(::Val{7}) = -1.0
@inline ad_qcx(::Val{8}) = -1.0
@inline ad_qcx(::Val{9}) = 1.0

@inline ad_qcy(::Val{1}) = 0.0
@inline ad_qcy(::Val{2}) = 0.0
@inline ad_qcy(::Val{3}) = 1.0
@inline ad_qcy(::Val{4}) = 0.0
@inline ad_qcy(::Val{5}) = -1.0
@inline ad_qcy(::Val{6}) = 1.0
@inline ad_qcy(::Val{7}) = 1.0
@inline ad_qcy(::Val{8}) = -1.0
@inline ad_qcy(::Val{9}) = -1.0

@inline ad_qwgt(::Val{1}) = 4.0 / 9.0
@inline ad_qwgt(::Val{2}) = 1.0 / 9.0
@inline ad_qwgt(::Val{3}) = 1.0 / 9.0
@inline ad_qwgt(::Val{4}) = 1.0 / 9.0
@inline ad_qwgt(::Val{5}) = 1.0 / 9.0
@inline ad_qwgt(::Val{6}) = 1.0 / 36.0
@inline ad_qwgt(::Val{7}) = 1.0 / 36.0
@inline ad_qwgt(::Val{8}) = 1.0 / 36.0
@inline ad_qwgt(::Val{9}) = 1.0 / 36.0

@inline function ad_moments_raw(f1::T, f2::T, f3::T, f4::T, f5::T,
                                f6::T, f7::T, f8::T, f9::T) where {T}
    rho = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
    inv_rho = one(T) / rho
    ux = (f2 - f4 + f6 - f7 - f8 + f9) * inv_rho
    uy = (f3 - f5 + f6 + f7 - f8 - f9) * inv_rho
    return rho, ux, uy
end

@inline function ad_feq(::Val{Q}, rho::T, ux::T, uy::T, usq::T) where {Q,T}
    cx = T(ad_qcx(Val(Q)))
    cy = T(ad_qcy(Val(Q)))
    cu = cx * ux + cy * uy
    return T(ad_qwgt(Val(Q))) * rho *
           (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end

@inline function ad_libb_branch(q_w::T, f_post_here::T,
                                f_post_back::T, f_bar_post_here::T) where {T}
    if q_w <= T(0.5)
        return T(2) * q_w * f_post_here +
               (one(T) - T(2) * q_w) * f_post_back
    else
        inv_two_q = one(T) / (T(2) * q_w)
        return inv_two_q * f_post_here +
               (one(T) - inv_two_q) * f_bar_post_here
    end
end

@inline function ad_trt_rates_inline(nu::T) where {T}
    lambda_magic = T(3) / T(16)
    s_plus = one(T) / (T(3) * nu + T(0.5))
    s_minus = one(T) / (lambda_magic / (T(3) * nu) + T(0.5))
    return s_plus, s_minus
end

@inline function ad_trt_regularized_local(f1::T, f2::T, f3::T, f4::T, f5::T,
                                          f6::T, f7::T, f8::T, f9::T,
                                          s_p::T, s_m::T) where {T}
    rho, ux, uy = ad_moments_raw(f1, f2, f3, f4, f5, f6, f7, f8, f9)
    usq = ux * ux + uy * uy
    fe1 = ad_feq(Val(1), rho, ux, uy, usq)
    fe2 = ad_feq(Val(2), rho, ux, uy, usq)
    fe3 = ad_feq(Val(3), rho, ux, uy, usq)
    fe4 = ad_feq(Val(4), rho, ux, uy, usq)
    fe5 = ad_feq(Val(5), rho, ux, uy, usq)
    fe6 = ad_feq(Val(6), rho, ux, uy, usq)
    fe7 = ad_feq(Val(7), rho, ux, uy, usq)
    fe8 = ad_feq(Val(8), rho, ux, uy, usq)
    fe9 = ad_feq(Val(9), rho, ux, uy, usq)

    pxx = (f2 - fe2) + (f4 - fe4) + (f6 - fe6) +
          (f7 - fe7) + (f8 - fe8) + (f9 - fe9)
    pyy = (f3 - fe3) + (f5 - fe5) + (f6 - fe6) +
          (f7 - fe7) + (f8 - fe8) + (f9 - fe9)
    pxy = (f6 - fe6) - (f7 - fe7) + (f8 - fe8) - (f9 - fe9)

    # f_neq_i = w_i / (2 c_s⁴) (c_iα c_iβ - c_s² δ_αβ) Π_αβ, mirror of
    # `_trt_collide_local` in src/bc/specs.jl.
    h = T(0.5)
    fn1 = -T(2 / 3) * (pxx + pyy)
    fn2 =  T(1 / 6) * (T(2) * pxx - pyy)
    fn3 =  T(1 / 6) * (-pxx + T(2) * pyy)
    fn4 =  fn2
    fn5 =  fn3
    fn6 =  T(1 / 12) * (pxx + pyy) + T(1 / 4) * pxy
    fn7 =  T(1 / 12) * (pxx + pyy) - T(1 / 4) * pxy
    fn8 =  fn6
    fn9 =  fn7

    a = (s_p + s_m) * h
    b = (s_p - s_m) * h
    return (
        fe1 + (one(T) - s_p) * fn1,
        fe2 + (one(T) - a) * fn2 - b * fn4,
        fe3 + (one(T) - a) * fn3 - b * fn5,
        fe4 + (one(T) - a) * fn4 - b * fn2,
        fe5 + (one(T) - a) * fn5 - b * fn3,
        fe6 + (one(T) - a) * fn6 - b * fn8,
        fe7 + (one(T) - a) * fn7 - b * fn9,
        fe8 + (one(T) - a) * fn8 - b * fn6,
        fe9 + (one(T) - a) * fn9 - b * fn7,
    )
end

function ad_bulk_v2_step!(out, f, q_wall, is_solid, s_plus, s_minus, Nx::Int, Ny::Int)
    @inbounds for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]
            out[i, j, 1] = 4.0 / 9.0
            out[i, j, 2] = 1.0 / 9.0
            out[i, j, 3] = 1.0 / 9.0
            out[i, j, 4] = 1.0 / 9.0
            out[i, j, 5] = 1.0 / 9.0
            out[i, j, 6] = 1.0 / 36.0
            out[i, j, 7] = 1.0 / 36.0
            out[i, j, 8] = 1.0 / 36.0
            out[i, j, 9] = 1.0 / 36.0
            continue
        end

        fp1 = f[i, j, 1]
        fp2 = i > 1 ? f[i - 1, j, 2] : f[i, j, 4]
        fp3 = j > 1 ? f[i, j - 1, 3] : f[i, j, 5]
        fp4 = i < Nx ? f[i + 1, j, 4] : f[i, j, 2]
        fp5 = j < Ny ? f[i, j + 1, 5] : f[i, j, 3]
        fp6 = (i > 1 && j > 1) ? f[i - 1, j - 1, 6] : f[i, j, 8]
        fp7 = (i < Nx && j > 1) ? f[i + 1, j - 1, 7] : f[i, j, 9]
        fp8 = (i < Nx && j < Ny) ? f[i + 1, j + 1, 8] : f[i, j, 6]
        fp9 = (i > 1 && j < Ny) ? f[i - 1, j + 1, 9] : f[i, j, 7]

        qw2 = q_wall[i, j, 2]
        if qw2 > 0.0
            fp4 = ad_libb_branch(qw2, f[i, j, 2], fp2, f[i, j, 4])
        end
        qw4 = q_wall[i, j, 4]
        if qw4 > 0.0
            fp2 = ad_libb_branch(qw4, f[i, j, 4], fp4, f[i, j, 2])
        end
        qw3 = q_wall[i, j, 3]
        if qw3 > 0.0
            fp5 = ad_libb_branch(qw3, f[i, j, 3], fp3, f[i, j, 5])
        end
        qw5 = q_wall[i, j, 5]
        if qw5 > 0.0
            fp3 = ad_libb_branch(qw5, f[i, j, 5], fp5, f[i, j, 3])
        end
        qw6 = q_wall[i, j, 6]
        if qw6 > 0.0
            fp8 = ad_libb_branch(qw6, f[i, j, 6], fp6, f[i, j, 8])
        end
        qw8 = q_wall[i, j, 8]
        if qw8 > 0.0
            fp6 = ad_libb_branch(qw8, f[i, j, 8], fp8, f[i, j, 6])
        end
        qw7 = q_wall[i, j, 7]
        if qw7 > 0.0
            fp9 = ad_libb_branch(qw7, f[i, j, 7], fp7, f[i, j, 9])
        end
        qw9 = q_wall[i, j, 9]
        if qw9 > 0.0
            fp7 = ad_libb_branch(qw9, f[i, j, 9], fp9, f[i, j, 7])
        end

        rho, ux, uy = ad_moments_raw(fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9)
        usq = ux * ux + uy * uy
        feq1 = ad_feq(Val(1), rho, ux, uy, usq)
        feq2 = ad_feq(Val(2), rho, ux, uy, usq)
        feq3 = ad_feq(Val(3), rho, ux, uy, usq)
        feq4 = ad_feq(Val(4), rho, ux, uy, usq)
        feq5 = ad_feq(Val(5), rho, ux, uy, usq)
        feq6 = ad_feq(Val(6), rho, ux, uy, usq)
        feq7 = ad_feq(Val(7), rho, ux, uy, usq)
        feq8 = ad_feq(Val(8), rho, ux, uy, usq)
        feq9 = ad_feq(Val(9), rho, ux, uy, usq)
        a = 0.5 * (s_plus + s_minus)
        b = 0.5 * (s_plus - s_minus)

        out[i, j, 1] = fp1 - s_plus * (fp1 - feq1)
        out[i, j, 2] = fp2 - a * (fp2 - feq2) - b * (fp4 - feq4)
        out[i, j, 4] = fp4 - a * (fp4 - feq4) - b * (fp2 - feq2)
        out[i, j, 3] = fp3 - a * (fp3 - feq3) - b * (fp5 - feq5)
        out[i, j, 5] = fp5 - a * (fp5 - feq5) - b * (fp3 - feq3)
        out[i, j, 6] = fp6 - a * (fp6 - feq6) - b * (fp8 - feq8)
        out[i, j, 8] = fp8 - a * (fp8 - feq8) - b * (fp6 - feq6)
        out[i, j, 7] = fp7 - a * (fp7 - feq7) - b * (fp9 - feq9)
        out[i, j, 9] = fp9 - a * (fp9 - feq9) - b * (fp7 - feq7)
    end
    return nothing
end

# Mirror of `_bc_east_neq_extrap_2d!` (east `NeqExtrapolationPressure`): row j
# of the outlet column is rebuilt, without collision, from the post-collision
# populations of column Nx-1, which the bulk step has already written to `out`:
#   out[Nx, j, q] = out[Nx-1, j, q] + f_q^eq(rho_out - rho, u)
# with (rho, u) the raw moments of out[Nx-1, j, :]. Same operation order as the
# production kernel, so the two agree bit for bit.
@inline function ad_outlet_neq_extrap_row!(out, rho_out, Nx::Int, j::Int)
    @inbounds begin
        g1 = out[Nx - 1, j, 1]; g2 = out[Nx - 1, j, 2]; g3 = out[Nx - 1, j, 3]
        g4 = out[Nx - 1, j, 4]; g5 = out[Nx - 1, j, 5]; g6 = out[Nx - 1, j, 6]
        g7 = out[Nx - 1, j, 7]; g8 = out[Nx - 1, j, 8]; g9 = out[Nx - 1, j, 9]
        rho, ux, uy = ad_moments_raw(g1, g2, g3, g4, g5, g6, g7, g8, g9)
        usq = ux * ux + uy * uy
        drho = convert(typeof(rho), rho_out) - rho
        out[Nx, j, 1] = g1 + ad_feq(Val(1), drho, ux, uy, usq)
        out[Nx, j, 2] = g2 + ad_feq(Val(2), drho, ux, uy, usq)
        out[Nx, j, 3] = g3 + ad_feq(Val(3), drho, ux, uy, usq)
        out[Nx, j, 4] = g4 + ad_feq(Val(4), drho, ux, uy, usq)
        out[Nx, j, 5] = g5 + ad_feq(Val(5), drho, ux, uy, usq)
        out[Nx, j, 6] = g6 + ad_feq(Val(6), drho, ux, uy, usq)
        out[Nx, j, 7] = g7 + ad_feq(Val(7), drho, ux, uy, usq)
        out[Nx, j, 8] = g8 + ad_feq(Val(8), drho, ux, uy, usq)
        out[Nx, j, 9] = g9 + ad_feq(Val(9), drho, ux, uy, usq)
    end
    return nothing
end

# Mirror of the west Zou-He velocity kernel in `src/bc/rebuild_2d.jl` with
# half-way walls south and north (`_xface_rows_2d`): all rows 1:Ny are rebuilt,
# and on rows 1 and Ny the populations pulled across the wall are the half-way
# bounce-back of the same node, as in `ad_bulk_v2_step!`. The east outlet is
# the extrapolation outlet on rows 2:Ny-1.
function ad_apply_zou_he_rebuild!(out, f, u_profile, rho_out, s_plus, s_minus,
                                  Nx::Int, Ny::Int)
    @inbounds for j in 1:Ny
        bs = j == 1   # south wall row: links from j - 1 bounce back (half-way)
        bn = j == Ny  # north wall row: links from j + 1 bounce back
        fp1 = f[1, j, 1]
        fp3 = bs ? f[1, j, 5] : f[1, j - 1, 3]
        fp4 = f[2, j, 4]
        fp5 = bn ? f[1, j, 3] : f[1, j + 1, 5]
        fp7 = bs ? f[1, j, 9] : f[2, j - 1, 7]
        fp8 = bn ? f[1, j, 6] : f[2, j + 1, 8]
        u_in = u_profile[j]
        rho_w = (fp1 + fp3 + fp5 + 2.0 * (fp4 + fp7 + fp8)) / (1.0 - u_in)
        fp2 = fp4 + (2.0 / 3.0) * rho_w * u_in
        fp6 = fp8 - 0.5 * (fp3 - fp5) + (1.0 / 6.0) * rho_w * u_in
        fp9 = fp7 + 0.5 * (fp3 - fp5) + (1.0 / 6.0) * rho_w * u_in
        F1, F2, F3, F4, F5, F6, F7, F8, F9 =
            ad_trt_regularized_local(fp1, fp2, fp3, fp4, fp5, fp6, fp7,
                                     fp8, fp9, s_plus, s_minus)
        out[1, j, 1] = F1
        out[1, j, 2] = F2
        out[1, j, 3] = F3
        out[1, j, 4] = F4
        out[1, j, 5] = F5
        out[1, j, 6] = F6
        out[1, j, 7] = F7
        out[1, j, 8] = F8
        out[1, j, 9] = F9

        # East outlet (NeqExtrapolationPressure), rows 2:Ny-1 as in
        # `_apply_bc_2d_east_neq_extrap!`: from the post-collision column
        # Nx-1 already written to `out` by the bulk step. The corner rows
        # keep the bulk-step values.
        if 1 < j < Ny
            ad_outlet_neq_extrap_row!(out, rho_out, Nx, j)
        end
    end
    return nothing
end

function ad_step!(out, f, q_wall, is_solid, u_profile, rho_out,
                  s_plus, s_minus, Nx::Int, Ny::Int)
    ad_bulk_v2_step!(out, f, q_wall, is_solid, s_plus, s_minus, Nx, Ny)
    ad_apply_zou_he_rebuild!(out, f, u_profile, rho_out, s_plus, s_minus, Nx, Ny)
    return nothing
end

"""
    ad_step_nu!(out, f, q_wall, is_solid, u_profile, rho_out, ν::Float64, Nx::Int, Ny::Int)

Enzyme-diffable wrapper: computes (s_plus, s_minus) from ν via `ad_trt_rates_inline`,
then calls `ad_step!`. Use this as the target for `Enzyme.autodiff(Reverse, ad_step_nu!, ..., Active(ν), ...)`.
`ad_step!` itself is unchanged.
"""
function ad_step_nu!(out, f, q_wall, is_solid, u_profile, rho_out,
                     ν::Float64, Nx::Int, Ny::Int)
    s_plus, s_minus = ad_trt_rates_inline(ν)
    ad_step!(out, f, q_wall, is_solid, u_profile, rho_out, s_plus, s_minus, Nx, Ny)
    return nothing
end

"""
    ad_bulk_nufield!(out, f, q_wall, is_solid, nu_field::Vector{Float64}, Nx::Int, Ny::Int)

Per-row TRT bulk step: row j uses ν=nu_field[j] to compute (s_plus_j, s_minus_j)
via `ad_trt_rates_inline`. Same loop body as `ad_bulk_v2_step!` but rates vary per row.
Enzyme-diffable wrt `nu_field` (Duplicated array); `ad_bulk_v2_step!` is UNCHANGED.
"""
function ad_bulk_nufield!(out, f, q_wall, is_solid, nu_field::Vector{Float64}, Nx::Int, Ny::Int)
    @inbounds for j in 1:Ny
        s_plus_j, s_minus_j = ad_trt_rates_inline(nu_field[j])
        for i in 1:Nx
            if is_solid[i, j]
                out[i, j, 1] = 4.0 / 9.0
                out[i, j, 2] = 1.0 / 9.0
                out[i, j, 3] = 1.0 / 9.0
                out[i, j, 4] = 1.0 / 9.0
                out[i, j, 5] = 1.0 / 9.0
                out[i, j, 6] = 1.0 / 36.0
                out[i, j, 7] = 1.0 / 36.0
                out[i, j, 8] = 1.0 / 36.0
                out[i, j, 9] = 1.0 / 36.0
                continue
            end

            fp1 = f[i, j, 1]
            fp2 = i > 1 ? f[i - 1, j, 2] : f[i, j, 4]
            fp3 = j > 1 ? f[i, j - 1, 3] : f[i, j, 5]
            fp4 = i < Nx ? f[i + 1, j, 4] : f[i, j, 2]
            fp5 = j < Ny ? f[i, j + 1, 5] : f[i, j, 3]
            fp6 = (i > 1 && j > 1) ? f[i - 1, j - 1, 6] : f[i, j, 8]
            fp7 = (i < Nx && j > 1) ? f[i + 1, j - 1, 7] : f[i, j, 9]
            fp8 = (i < Nx && j < Ny) ? f[i + 1, j + 1, 8] : f[i, j, 6]
            fp9 = (i > 1 && j < Ny) ? f[i - 1, j + 1, 9] : f[i, j, 7]

            qw2 = q_wall[i, j, 2]
            if qw2 > 0.0
                fp4 = ad_libb_branch(qw2, f[i, j, 2], fp2, f[i, j, 4])
            end
            qw4 = q_wall[i, j, 4]
            if qw4 > 0.0
                fp2 = ad_libb_branch(qw4, f[i, j, 4], fp4, f[i, j, 2])
            end
            qw3 = q_wall[i, j, 3]
            if qw3 > 0.0
                fp5 = ad_libb_branch(qw3, f[i, j, 3], fp3, f[i, j, 5])
            end
            qw5 = q_wall[i, j, 5]
            if qw5 > 0.0
                fp3 = ad_libb_branch(qw5, f[i, j, 5], fp5, f[i, j, 3])
            end
            qw6 = q_wall[i, j, 6]
            if qw6 > 0.0
                fp8 = ad_libb_branch(qw6, f[i, j, 6], fp6, f[i, j, 8])
            end
            qw8 = q_wall[i, j, 8]
            if qw8 > 0.0
                fp6 = ad_libb_branch(qw8, f[i, j, 8], fp8, f[i, j, 6])
            end
            qw7 = q_wall[i, j, 7]
            if qw7 > 0.0
                fp9 = ad_libb_branch(qw7, f[i, j, 7], fp7, f[i, j, 9])
            end
            qw9 = q_wall[i, j, 9]
            if qw9 > 0.0
                fp7 = ad_libb_branch(qw9, f[i, j, 9], fp9, f[i, j, 7])
            end

            rho, ux, uy = ad_moments_raw(fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9)
            usq = ux * ux + uy * uy
            feq1 = ad_feq(Val(1), rho, ux, uy, usq)
            feq2 = ad_feq(Val(2), rho, ux, uy, usq)
            feq3 = ad_feq(Val(3), rho, ux, uy, usq)
            feq4 = ad_feq(Val(4), rho, ux, uy, usq)
            feq5 = ad_feq(Val(5), rho, ux, uy, usq)
            feq6 = ad_feq(Val(6), rho, ux, uy, usq)
            feq7 = ad_feq(Val(7), rho, ux, uy, usq)
            feq8 = ad_feq(Val(8), rho, ux, uy, usq)
            feq9 = ad_feq(Val(9), rho, ux, uy, usq)
            a = 0.5 * (s_plus_j + s_minus_j)
            b = 0.5 * (s_plus_j - s_minus_j)

            out[i, j, 1] = fp1 - s_plus_j * (fp1 - feq1)
            out[i, j, 2] = fp2 - a * (fp2 - feq2) - b * (fp4 - feq4)
            out[i, j, 4] = fp4 - a * (fp4 - feq4) - b * (fp2 - feq2)
            out[i, j, 3] = fp3 - a * (fp3 - feq3) - b * (fp5 - feq5)
            out[i, j, 5] = fp5 - a * (fp5 - feq5) - b * (fp3 - feq3)
            out[i, j, 6] = fp6 - a * (fp6 - feq6) - b * (fp8 - feq8)
            out[i, j, 8] = fp8 - a * (fp8 - feq8) - b * (fp6 - feq6)
            out[i, j, 7] = fp7 - a * (fp7 - feq7) - b * (fp9 - feq9)
            out[i, j, 9] = fp9 - a * (fp9 - feq9) - b * (fp7 - feq7)
        end
    end
    return nothing
end

"""
    ad_apply_zou_he_rebuild_nufield!(out, f, u_profile, rho_out, nu_field::Vector{Float64}, Nx::Int, Ny::Int)

Per-row inlet/outlet rebuild: the Zou-He inlet of row j uses ν=nu_field[j]
for its TRT rates; the extrapolation outlet uses no rate. Same body as `ad_apply_zou_he_rebuild!` but rates computed per-row.
`ad_apply_zou_he_rebuild!` is UNCHANGED.
"""
function ad_apply_zou_he_rebuild_nufield!(out, f, u_profile, rho_out,
                                          nu_field::Vector{Float64}, Nx::Int, Ny::Int)
    @inbounds for j in 1:Ny
        s_plus_j, s_minus_j = ad_trt_rates_inline(nu_field[j])

        bs = j == 1   # south wall row: links from j - 1 bounce back (half-way)
        bn = j == Ny  # north wall row: links from j + 1 bounce back
        fp1 = f[1, j, 1]
        fp3 = bs ? f[1, j, 5] : f[1, j - 1, 3]
        fp4 = f[2, j, 4]
        fp5 = bn ? f[1, j, 3] : f[1, j + 1, 5]
        fp7 = bs ? f[1, j, 9] : f[2, j - 1, 7]
        fp8 = bn ? f[1, j, 6] : f[2, j + 1, 8]
        u_in = u_profile[j]
        rho_w = (fp1 + fp3 + fp5 + 2.0 * (fp4 + fp7 + fp8)) / (1.0 - u_in)
        fp2 = fp4 + (2.0 / 3.0) * rho_w * u_in
        fp6 = fp8 - 0.5 * (fp3 - fp5) + (1.0 / 6.0) * rho_w * u_in
        fp9 = fp7 + 0.5 * (fp3 - fp5) + (1.0 / 6.0) * rho_w * u_in
        F1, F2, F3, F4, F5, F6, F7, F8, F9 =
            ad_trt_regularized_local(fp1, fp2, fp3, fp4, fp5, fp6, fp7,
                                     fp8, fp9, s_plus_j, s_minus_j)
        out[1, j, 1] = F1
        out[1, j, 2] = F2
        out[1, j, 3] = F3
        out[1, j, 4] = F4
        out[1, j, 5] = F5
        out[1, j, 6] = F6
        out[1, j, 7] = F7
        out[1, j, 8] = F8
        out[1, j, 9] = F9

        # East outlet (NeqExtrapolationPressure), rows 2:Ny-1 as in
        # `_apply_bc_2d_east_neq_extrap!`: from the post-collision column
        # Nx-1 already written to `out` by the bulk step. The corner rows
        # keep the bulk-step values.
        if 1 < j < Ny
            ad_outlet_neq_extrap_row!(out, rho_out, Nx, j)
        end
    end
    return nothing
end

"""
    ad_step_nufield!(out, f, q_wall, is_solid, u_profile, rho_out,
                     nu_field::Vector{Float64}, Nx::Int, Ny::Int)

Per-row TRT step: row j uses ν=nu_field[j], computes (s_plus_j, s_minus_j)
via `ad_trt_rates_inline`. Enzyme-diffable wrt `nu_field` (Duplicated array).
`ad_step!` and `ad_bulk_v2_step!` are UNCHANGED.
"""
function ad_step_nufield!(out, f, q_wall, is_solid, u_profile, rho_out,
                          nu_field::Vector{Float64}, Nx::Int, Ny::Int)
    ad_bulk_nufield!(out, f, q_wall, is_solid, nu_field, Nx, Ny)
    ad_apply_zou_he_rebuild_nufield!(out, f, u_profile, rho_out, nu_field, Nx, Ny)
    return nothing
end
