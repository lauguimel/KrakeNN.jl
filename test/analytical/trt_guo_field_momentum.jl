# Bug report: the TRT Guo-field collision bricks inject g*F instead of F per
# step, with g = 1 + (s_minus - s_plus)/2 (issue #67).
#
# Case: one collision of a uniform equilibrium state (rho = 1, u0 != 0) under a
# uniform body-force field F, read at an interior cell (no wall link, no cut
# link). The state is uniform, so the populations the cell pulls are the ones it
# was initialised with. With Guo forcing (Guo, Zheng & Shi 2002,
# doi:10.1103/PhysRevE.65.046308) the first moment must change by exactly F in
# one collision, whatever the relaxation rates:
#
#     sum_q c_q (f_out[q] - f_in[q]) = F.
#
# Frozen gate: |dj_d / F_d - 1| <= 1e-9 for every component d.
# Origin: the analytical identity above. Round-off floor measured on the BGK
# control path collide_guo_field_3d! and on the TRT brick at s_plus = s_minus
# (CPU, Float64, 2026-09-28, Apple M3 Max, Julia 1.12.5): <= 1e-12. The gate sits
# three orders above that floor. Measured with the pre-fix bricks at the
# driver-default viscosities: dj/F = 0.516484 (3D, nu = 0.05),
# 0.435973 (3D, nu = 0.04), 0.819444 (2D and AD mirror, nu = 0.10). With the
# fix (CPU, Float64, 2026-10-08, Julia 1.11.9): |dj/F - 1| <= 7e-13 in all four.
#
# Second-moment control (plain @test, passes before and after a correct fix).
# The first-moment gate alone cannot tell the correct fix from a wrong one:
# scaling the WHOLE source by (1 - s_minus/2) also gives dj = F exactly, but it
# changes the even part of the source, which carries the stress. For rho = 1,
# one TRT collision must satisfy
#
#     Pi_out = Pi_in - s_plus (Pi_in - Pi_eq(u*)) + (1 - s_plus/2) (u* F + F u*),
#
# with Pi = sum_q c_q c_q f_q, Pi_eq(u) = I/3 + u u and u* = u0 + F/2 (the
# velocity the brick uses for the equilibrium). The odd part of the source has
# no second moment, so this holds with the current bricks and with the correct
# fix. Frozen gate: max_ab |residual_ab| <= 1e-8 * max_ab |u*_a F_b + F_a u*_b|.
# Origin: the identity above. Measured (same machine and date) with the 3D brick:
# 4.2e-11 of that scale with the current code and 1.4e-11 with the correct fix
# built from the brick output; 0.48 (nu = 0.05) and 0.56 (nu = 0.04) with the
# wrong fix.
#
# The TRT cases were @test_broken until the odd (antisymmetric) part of the
# Guo source was scaled by (1 - s_minus/2), with the even part kept at
# (1 - s_plus/2), in
#   src/kernels/dsl/bricks_3d.jl  (CollideTRTDirectGuoField_3D)
#   src/kernels/dsl/bricks.jl     (CollideTRTDirectGuoField)
#   src/ad/ad_ve_step.jl          (bit mirror of the 2D brick)
# The controls (@test) must keep passing before and after the fix.
#
# CPU only: the defect is algebraic (a prefactor in the collision) and does not
# depend on backend; the GPU runs the same DSL kernel. The testsets in this
# first block are Float64; the analytic split checks further down also run in
# Float32, with gates scaled by eps(Float32)/eps(Float64).

using Test
using Kraken

const _GUO_GATE = 1e-9
const _GUO_PI_GATE = 1e-8

# Second-moment residual of one TRT Guo collision (rho = 1), relative to
# max|u* F + F u*|. c = (cx, cy[, cz]); fpre and fpost are the populations of
# one cell before and after the collision.
function _guo_pi_residual(fpre, fpost, c, us, F, s_plus)
    D = length(us)
    R = 0.0; scale = 0.0
    for a in 1:D, b in 1:D
        Pin = sum(c[a][q] * c[b][q] * fpre[q] for q in eachindex(fpre))
        Pout = sum(c[a][q] * c[b][q] * fpost[q] for q in eachindex(fpost))
        Peq = (a == b ? 1 / 3 : 0.0) + us[a] * us[b]
        src = us[a] * F[b] + F[a] * us[b]
        R = max(R, abs(Pout - (Pin - s_plus * (Pin - Peq) + (1 - s_plus / 2) * src)))
        scale = max(scale, abs(src))
    end
    return R / scale
end

# Λ = nothing leaves the step on its own default Λ, which is what the drivers use.
function _guo_ratio_trt_3d(nu; Λ=nothing, u0=(0.02, -0.01, 0.005), F=(1e-4, -5e-5, 2e-5))
    N = 5
    f_in = zeros(Float64, N, N, N, 19)
    for k in 1:N, j in 1:N, i in 1:N, q in 1:19
        f_in[i, j, k, q] = Kraken.equilibrium(D3Q19(), 1.0, u0[1], u0[2], u0[3], q)
    end
    f_out = zeros(Float64, N, N, N, 19)
    rho = ones(N, N, N); ux = zeros(N, N, N); uy = zeros(N, N, N); uz = zeros(N, N, N)
    is_solid = zeros(Bool, N, N, N)
    q_wall = zeros(N, N, N, 19)
    uwx = zeros(N, N, N, 19); uwy = zeros(N, N, N, 19); uwz = zeros(N, N, N, 19)
    Fx = fill(F[1], N, N, N); Fy = fill(F[2], N, N, N); Fz = fill(F[3], N, N, N)
    kw = Λ === nothing ? (;) : (; Λ)
    Kraken.fused_trt_libb_v2_guo_field_step_3d!(f_out, f_in, rho, ux, uy, uz, is_solid,
        q_wall, uwx, uwy, uwz, Fx, Fy, Fz, N, N, N, nu; periodic_z=true, kw...)
    cx = Kraken.velocities_x(D3Q19()); cy = Kraken.velocities_y(D3Q19())
    cz = Kraken.velocities_z(D3Q19())
    c = 3
    fpre = f_in[c, c, c, :]; fpost = f_out[c, c, c, :]
    dj = (sum(cx[q] * (fpost[q] - fpre[q]) for q in 1:19),
          sum(cy[q] * (fpost[q] - fpre[q]) for q in 1:19),
          sum(cz[q] * (fpost[q] - fpre[q]) for q in 1:19))
    us = u0 .+ F ./ 2
    pi_res = _guo_pi_residual(fpre, fpost, (cx, cy, cz), us, F, 1 / (3nu + 0.5))
    return dj ./ F, pi_res
end

function _guo_ratio_bgk_3d(nu; u0=(0.02, -0.01, 0.005), F=(1e-4, -5e-5, 2e-5))
    N = 5
    f = zeros(Float64, N, N, N, 19)
    for k in 1:N, j in 1:N, i in 1:N, q in 1:19
        f[i, j, k, q] = Kraken.equilibrium(D3Q19(), 1.0, u0[1], u0[2], u0[3], q)
    end
    f0 = copy(f)
    is_solid = zeros(Bool, N, N, N)
    Fx = fill(F[1], N, N, N); Fy = fill(F[2], N, N, N); Fz = fill(F[3], N, N, N)
    Kraken.collide_guo_field_3d!(f, is_solid, Fx, Fy, Fz, 1 / (3nu + 0.5))
    cx = Kraken.velocities_x(D3Q19()); cy = Kraken.velocities_y(D3Q19())
    cz = Kraken.velocities_z(D3Q19())
    c = 3
    dj = (sum(cx[q] * (f[c, c, c, q] - f0[c, c, c, q]) for q in 1:19),
          sum(cy[q] * (f[c, c, c, q] - f0[c, c, c, q]) for q in 1:19),
          sum(cz[q] * (f[c, c, c, q] - f0[c, c, c, q]) for q in 1:19))
    return dj ./ F
end

function _guo_ratio_trt_2d(nu; u0=(0.02, -0.01), F=(1e-4, -5e-5))
    N = 5
    f_in = zeros(Float64, N, N, 9)
    for j in 1:N, i in 1:N, q in 1:9
        f_in[i, j, q] = Kraken.equilibrium(D2Q9(), 1.0, u0[1], u0[2], q)
    end
    f_out = zeros(Float64, N, N, 9)
    rho = ones(N, N); ux = zeros(N, N); uy = zeros(N, N)
    is_solid = zeros(Bool, N, N)
    q_wall = zeros(N, N, 9); uwx = zeros(N, N, 9); uwy = zeros(N, N, 9)
    Fx = fill(F[1], N, N); Fy = fill(F[2], N, N)
    Kraken.fused_trt_libb_v2_guo_field_step!(f_out, f_in, rho, ux, uy, is_solid,
        q_wall, uwx, uwy, Fx, Fy, N, N, nu)
    cx = Kraken.velocities_x(D2Q9()); cy = Kraken.velocities_y(D2Q9())
    c = 3
    fpre = f_in[c, c, :]; fpost = f_out[c, c, :]
    dj = (sum(cx[q] * (fpost[q] - fpre[q]) for q in 1:9),
          sum(cy[q] * (fpost[q] - fpre[q]) for q in 1:9))
    us = u0 .+ F ./ 2
    pi_res = _guo_pi_residual(fpre, fpost, (cx, cy), us, F, 1 / (3nu + 0.5))
    return dj ./ F, pi_res
end

# AD mirror (ad_ve_coupled_step!): fluid at rest, psi = 0 so tau_p = 0, only
# the frozen body force Fx_body acts; cell (18, 8) is far from the cylinder
# (centre (5.35, 8.15), radius 1.6) and from the inlet/outlet columns. The
# fluid is at rest, so u* = (Fb/2, 0) and the second-moment scale is Fb^2.
# Fb = 1e-2 keeps that scale far above round-off (at Fb = 1e-3 the residual
# floor was 3.3e-10 of the scale, only 30 times below the gate). One collision
# only, so the size of Fb does not matter for stability, and dj/F does not
# depend on it.
function _guo_ratio_ad_2d(nu; Fb=1e-2)
    Nx, Ny = 24, 16
    sp, sm = Kraken.ad_ve_trt_rates(nu)
    p = Kraken.ADVECoupledParams(Nx, Ny, 0.5, 0.05, 4, 0.04, nu, Fb, sp, sm)
    geom = Kraken.ad_ve_build_geom(Nx, Ny, 5.35, 8.15, 1.6; samples=8, u_mean=0.0)
    n = Nx * Ny
    w_in = zeros(Float64, 12n)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        w_in[Kraken.ad_ve_fpop(i, j, q, Nx, Ny)] = Kraken.AD_VE_W[q]
    end
    w_out = zeros(Float64, 12n)
    Kraken.ad_ve_coupled_step!(w_out, w_in, geom.g, geom.q_wall, p, geom.u_profile)
    cx = (0, 1, 0, -1, 0, 1, -1, -1, 1)
    cy = (0, 0, 1, 0, -1, 1, 1, -1, -1)
    fpre = [w_in[Kraken.ad_ve_fpop(18, 8, q, Nx, Ny)] for q in 1:9]
    fpost = [w_out[Kraken.ad_ve_fpop(18, 8, q, Nx, Ny)] for q in 1:9]
    jx = sum(cx[q] * (fpost[q] - fpre[q]) for q in 1:9)
    pi_res = _guo_pi_residual(fpre, fpost, (cx, cy), (Fb / 2, 0.0), (Fb, 0.0), sp)
    return jx / Fb, pi_res
end

@testset "TRT Guo-field brick injects exactly F per collision" begin
    # Controls: must pass before and after the fix.
    r_bgk = _guo_ratio_bgk_3d(0.05)
    @test all(abs.(r_bgk .- 1) .<= _GUO_GATE)
    # s_plus == s_minus exactly when nu = sqrt(Λ)/3. Λ is passed explicitly so
    # that this control does not depend on the step's default Λ.
    Λ_magic = 3 / 16
    r_magic, _ = _guo_ratio_trt_3d(sqrt(Λ_magic) / 3; Λ=Λ_magic)
    @test all(abs.(r_magic .- 1) .<= _GUO_GATE)

    # Defect cases: s_plus != s_minus, step default Λ as the drivers call it.
    r3_ext, p3_ext = _guo_ratio_trt_3d(0.05)      # extensional driver default nu_s
    r3_sph, p3_sph = _guo_ratio_trt_3d(0.04)      # VE sphere driver default nu_s
    r2_cyl, p2_cyl = _guo_ratio_trt_2d(0.10)      # 2D logfv cylinder default nu_s + bsd*nu_p
    r_ad, p_ad = _guo_ratio_ad_2d(0.10)           # AD bit mirror of the 2D brick
    println("TRT Guo-field dj/F: 3D nu=0.05 ", r3_ext, "  3D nu=0.04 ", r3_sph,
            "  2D nu=0.10 ", r2_cyl, "  AD nu=0.10 ", r_ad,
            "  controls: BGK ", r_bgk, "  TRT nu=sqrt(3)/12 ", r_magic)
    println("TRT Guo-field second-moment residual / max|u*F+Fu*|: 3D nu=0.05 ", p3_ext,
            "  3D nu=0.04 ", p3_sph, "  2D nu=0.10 ", p2_cyl, "  AD nu=0.10 ", p_ad)

    # Control: the even (stress-carrying) part of the source keeps (1 - s_plus/2).
    @test p3_ext <= _GUO_PI_GATE
    @test p3_sph <= _GUO_PI_GATE
    @test p2_cyl <= _GUO_PI_GATE
    @test p_ad <= _GUO_PI_GATE

    # Former defect, fixed by the even/odd split of the Guo source.
    @test all(abs.(r3_ext .- 1) .<= _GUO_GATE)
    @test all(abs.(r3_sph .- 1) .<= _GUO_GATE)
    @test all(abs.(r2_cyl .- 1) .<= _GUO_GATE)
    @test abs(r_ad - 1) <= _GUO_GATE
end

# ---------------------------------------------------------------------------
# Analytic checks of the even and odd parts of the TRT Guo source, required by
# the design review of the fix (issue #67).
#
# Notation. One collision at the centre of a uniform patch, interior cell, no
# cut link, so the populations the cell pulls are its own (f_pre). With
# rho = sum f_pre, j = sum c f_pre and u* = (j + F/2)/rho (the velocity the
# brick puts in the equilibrium), the reference TRT relaxation is
#     R_q = f_pre_q - a (f_pre_q - feq_q(u*)) - b (f_pre_qbar - feq_qbar(u*)),
#     a = (s+ + s-)/2, b = (s+ - s-)/2,
# with feq written out below (not taken from Kraken). The source the brick added
# is src_q = f_post_q - R_q, split into S+_q = (src_q + src_qbar)/2 (even) and
# S-_q = (src_q - src_qbar)/2 (odd). The fix requires, for ANY (s+, s-):
#     S+_q = (1 - s+/2) w_q [9 (c_q.u*)(c_q.F) - 3 u*.F],
#     S-_q = (1 - s-/2) 3 w_q c_q.F,
# hence, by the 4th-order isotropy of D2Q9 and D3Q19
# (sum_q w_q c_a c_b c_c c_d = (d_ab d_cd + d_ac d_bd + d_ad d_bc)/9):
#     sum_q S+_q = 0,
#     sum_q c_a c_b S+_q = (1 - s+/2) (u*_a F_b + F_a u*_b)   (momentum flux),
#     sum_q c_a S-_q = (1 - s-/2) F_a,
# and the full collision changes the momentum by exactly F:
#     sum_q c_q (f_post_q - f_pre_q) = -s- (j - rho u*) + (1 - s-/2) F = F.
# The pre-fix bricks used (1 - s+/2) on S-, which fails the odd checks and the
# momentum check by the factor (1 - s+/2)/(1 - s-/2) unless s+ = s-.
#
# (s+, s-) pairs, from trt_rates(nu; Λ):
#   nu = 0.05, Λ = 3/16 -> (1.538, 0.800)   3D extensional driver default
#   nu = 0.10, Λ = 3/16 -> (1.250, 0.889)   2D log-FV cylinder default
#   nu = 1/6,  Λ = 3/16 -> (1.000, 1.143)   s- > s+
#   nu = 0.10, Λ = 1/4  -> (1.250, 0.750)
#   nu = 0.04, Λ = 1/12 -> (1.613, 0.837)
#
# Gates, all relative to the size of the expected quantity, set from the
# precision and not fitted. Each check sums Q <= 19 terms c_a c_b f_q with
# |c_a c_b| <= 1 and |f_q| < 0.5, each carrying a rounding of at most
# eps(T)/2 * 0.5 from the collision, so the absolute round-off is below
# Q eps(T)/4 ~ 5 eps(T). With F ~ 3e-2 and u* ~ 0.1 the smallest expected
# quantity is the momentum flux, ~ (1 - s+/2) 2 |u*| |F| >= 1.2e-3, so the
# relative floor is below 5 eps(T) / 1.2e-3 ~ 4e3 eps(T): 1e-12 in Float64,
# 5e-4 in Float32. The gate is 1e-12 * eps(T)/eps(Float64), i.e. 1e-12 in
# Float64 and 5.4e-4 in Float32. The defect these checks guard against is a
# relative error of |s+ - s-| / (2 - s+) >= 0.17 on the odd part for the
# s+ != s- pairs above, more than 300 times the Float32 gate.
# ---------------------------------------------------------------------------

const _GUO_PAIRS = ((0.05, 3 / 16), (0.10, 3 / 16), (1 / 6, 3 / 16), (0.10, 1 / 4), (0.04, 1 / 12))
_guo_split_gate(::Type{T}) where {T} = 1e-12 * eps(T) / eps(Float64)

# Lattice tables as Float64 (C is D x Q).
function _guo_lattice_tables(lat)
    rows = lat isa D3Q19 ?
        (Kraken.velocities_x(lat), Kraken.velocities_y(lat), Kraken.velocities_z(lat)) :
        (Kraken.velocities_x(lat), Kraken.velocities_y(lat))
    C = Float64[r[q] for r in rows, q in 1:length(rows[1])]
    return C, Float64.(collect(Kraken.weights(lat))), Int.(collect(Kraken.opposite(lat)))
end

# Second-order equilibrium written out (the reference does not call Kraken's).
function _guo_feq_ref(C, w, rho, u)
    D, Q = size(C)
    usq = sum(u[a]^2 for a in 1:D)
    return [w[q] * rho * (1 + 3 * sum(C[a, q] * u[a] for a in 1:D) +
            4.5 * sum(C[a, q] * u[a] for a in 1:D)^2 - 1.5 * usq) for q in 1:Q]
end

# Uniform state: equilibrium at (rho0, u0) plus a fixed non-equilibrium part of
# relative size neq (zero for the "from rest" cases).
function _guo_uniform_state(C, w, rho0, u0, neq)
    feq = _guo_feq_ref(C, w, rho0, u0)
    return [feq[q] + neq * w[q] * cos(1.7 * q) for q in eachindex(feq)]
end

# One collision of the 2D production brick (fused_trt_libb_v2_guo_field_step!,
# CollideTRTDirectGuoField) on a 3x3 uniform patch, in precision T.
function _guo_brick_2d(::Type{T}, f0, F, nu, Λ) where {T}
    N = 3
    f_in = Array{T}(undef, N, N, 9)
    for j in 1:N, i in 1:N, q in 1:9
        f_in[i, j, q] = T(f0[q])
    end
    f_out = zeros(T, N, N, 9)
    rho = ones(T, N, N); ux = zeros(T, N, N); uy = zeros(T, N, N)
    is_solid = zeros(Bool, N, N)
    q_wall = zeros(T, N, N, 9); uwx = zeros(T, N, N, 9); uwy = zeros(T, N, N, 9)
    Fx = fill(T(F[1]), N, N); Fy = fill(T(F[2]), N, N)
    Kraken.fused_trt_libb_v2_guo_field_step!(f_out, f_in, rho, ux, uy, is_solid,
        q_wall, uwx, uwy, Fx, Fy, N, N, nu; Λ=Λ)
    return Float64.(f_in[2, 2, :]), Float64.(f_out[2, 2, :])
end

# Same with the 3D brick (fused_trt_libb_v2_guo_field_step_3d!,
# CollideTRTDirectGuoField_3D) on a 3x3x3 patch.
function _guo_brick_3d(::Type{T}, f0, F, nu, Λ) where {T}
    N = 3
    f_in = Array{T}(undef, N, N, N, 19)
    for k in 1:N, j in 1:N, i in 1:N, q in 1:19
        f_in[i, j, k, q] = T(f0[q])
    end
    f_out = zeros(T, N, N, N, 19)
    rho = ones(T, N, N, N); ux = zeros(T, N, N, N); uy = zeros(T, N, N, N)
    uz = zeros(T, N, N, N)
    is_solid = zeros(Bool, N, N, N)
    q_wall = zeros(T, N, N, N, 19)
    uwx = zeros(T, N, N, N, 19); uwy = zeros(T, N, N, N, 19); uwz = zeros(T, N, N, N, 19)
    Fx = fill(T(F[1]), N, N, N); Fy = fill(T(F[2]), N, N, N); Fz = fill(T(F[3]), N, N, N)
    Kraken.fused_trt_libb_v2_guo_field_step_3d!(f_out, f_in, rho, ux, uy, uz, is_solid,
        q_wall, uwx, uwy, uwz, Fx, Fy, Fz, N, N, N, nu; Λ=Λ)
    return Float64.(f_in[2, 2, 2, :]), Float64.(f_out[2, 2, 2, :])
end

# Relative residuals of the source split identities above. sp, sm are the rates
# as the kernel sees them (rounded to T).
function _guo_split_residuals(C, w, opp, fpre, fpost, F, sp, sm)
    D, Q = size(C)
    rho = sum(fpre)
    us = [(sum(C[a, q] * fpre[q] for q in 1:Q) + F[a] / 2) / rho for a in 1:D]
    feq = _guo_feq_ref(C, w, rho, us)
    a = (sp + sm) / 2; b = (sp - sm) / 2
    src = [fpost[q] - (fpre[q] - a * (fpre[q] - feq[q]) - b * (fpre[opp[q]] - feq[opp[q]]))
           for q in 1:Q]
    Sp = [(src[q] + src[opp[q]]) / 2 for q in 1:Q]
    Sm = [(src[q] - src[opp[q]]) / 2 for q in 1:Q]
    ge = 1 - sp / 2; go = 1 - sm / 2
    cu(q) = sum(C[a, q] * us[a] for a in 1:D)
    cF(q) = sum(C[a, q] * F[a] for a in 1:D)
    uF = sum(us[a] * F[a] for a in 1:D)
    Sp_ex = [ge * w[q] * (9 * cu(q) * cF(q) - 3 * uF) for q in 1:Q]
    Sm_ex = [go * 3 * w[q] * cF(q) for q in 1:Q]
    Pi_ex = [ge * (us[x] * F[y] + F[x] * us[y]) for x in 1:D, y in 1:D]
    Pi = [sum(C[x, q] * C[y, q] * Sp[q] for q in 1:Q) for x in 1:D, y in 1:D]
    J = [sum(C[x, q] * Sm[q] for q in 1:Q) for x in 1:D]
    dj = [sum(C[x, q] * (fpost[q] - fpre[q]) for q in 1:Q) for x in 1:D]
    sF = maximum(abs, F); sPi = maximum(abs, Pi_ex)
    return (
        momentum   = maximum(abs, dj .- F) / sF,
        odd_moment = maximum(abs, J .- go .* F) / (go * sF),
        even_flux  = maximum(abs, Pi .- Pi_ex) / sPi,
        even_mass  = abs(sum(Sp)) / sPi,
        even_pop   = maximum(abs, Sp .- Sp_ex) / maximum(abs, Sp_ex),
        odd_pop    = maximum(abs, Sm .- Sm_ex) / maximum(abs, Sm_ex),
    )
end

function _guo_split_case(lat, ::Type{T}, nu, Λ; rho0, u0, F, neq) where {T}
    C, w, opp = _guo_lattice_tables(lat)
    f0 = _guo_uniform_state(C, w, rho0, collect(u0), neq)
    fpre, fpost = lat isa D3Q19 ? _guo_brick_3d(T, f0, F, nu, Λ) : _guo_brick_2d(T, f0, F, nu, Λ)
    sp, sm = Kraken.trt_rates(nu; Λ=Λ)
    return _guo_split_residuals(C, w, opp, fpre, fpost, collect(Float64.(T.(F))),
                                Float64(T(sp)), Float64(T(sm)))
end

@testset "TRT Guo source: even and odd parts, (s+, s-) pairs, 2D and 3D" begin
    for lat in (D2Q9(), D3Q19()), T in (Float64, Float32), (nu, Λ) in _GUO_PAIRS
        D = lat isa D3Q19 ? 3 : 2
        gate = _guo_split_gate(T)
        F = D == 3 ? (0.03, -0.02, 0.01) : (0.03, -0.02)
        # (a) one collision from rest under a uniform F: momentum gain = F.
        r0 = _guo_split_case(lat, T, nu, Λ; rho0=1.0, u0=zeros(D), F, neq=0.0)
        # (b) moving, non-equilibrium state, rho != 1: the full split.
        u0 = D == 3 ? (0.08, -0.05, 0.03) : (0.08, -0.05)
        r = _guo_split_case(lat, T, nu, Λ; rho0=1.03, u0, F, neq=0.02)
        println("TRT Guo split D=$D $T nu=$(round(nu; digits=4)) Λ=$(round(Λ; digits=4)): ",
                "from rest dj ", r0.momentum, "; moving dj ", r.momentum,
                " odd ", r.odd_moment, " flux ", r.even_flux, " mass ", r.even_mass,
                " S+ ", r.even_pop, " S- ", r.odd_pop, " (gate ", gate, ")")
        @test r0.momentum <= gate
        @test r.momentum <= gate
        @test r.odd_moment <= gate
        @test r.even_flux <= gate
        @test r.even_mass <= gate
        @test r.even_pop <= gate
        @test r.odd_pop <= gate
    end
end

# ---------------------------------------------------------------------------
# (c) SRT limit. When s+ = s- = ω the TRT brick must reproduce the BGK Guo
# collision of collide_guo_field_2d! / collide_guo_field_3d! (single
# (1 - ω/2) prefactor on the whole source), population by population.
# Pairs with s+ = s- (to the last bit or within one ulp), from trt_rates:
# Λ = (3 nu)^2 gives s- = s+; Λ = 3/16 at nu = sqrt(3)/12 is the magic case.
# The two kernels evaluate the same algebra in a different order (the TRT brick
# forms u* as (rho (j/rho) + F/2)/rho and sums two split source terms), so they
# differ by a few roundings of populations < 0.5: gate 16 eps(T) absolute,
# i.e. 3.6e-15 (Float64) and 1.9e-6 (Float32). The source itself is
# ~ 3 w |F| ~ 3e-3 here, so the gate is 1e-12 (Float64) of it.
# ---------------------------------------------------------------------------
const _GUO_SRT_PAIRS = ((1 / 6, 1 / 4), (0.05, 0.15^2), (0.10, 0.30^2), (sqrt(3) / 12, 3 / 16))

function _guo_bgk_2d(::Type{T}, f0, F, ω) where {T}
    N = 3
    f = Array{T}(undef, N, N, 9)
    for j in 1:N, i in 1:N, q in 1:9
        f[i, j, q] = T(f0[q])
    end
    Kraken.collide_guo_field_2d!(f, zeros(Bool, N, N), fill(T(F[1]), N, N),
                                 fill(T(F[2]), N, N), T(ω))
    return Float64.(f[2, 2, :])
end

function _guo_bgk_3d(::Type{T}, f0, F, ω) where {T}
    N = 3
    f = Array{T}(undef, N, N, N, 19)
    for k in 1:N, j in 1:N, i in 1:N, q in 1:19
        f[i, j, k, q] = T(f0[q])
    end
    Kraken.collide_guo_field_3d!(f, zeros(Bool, N, N, N), fill(T(F[1]), N, N, N),
                                 fill(T(F[2]), N, N, N), fill(T(F[3]), N, N, N), T(ω))
    return Float64.(f[2, 2, 2, :])
end

@testset "TRT Guo source: SRT limit s+ = s- reproduces the BGK Guo collision" begin
    for lat in (D2Q9(), D3Q19()), T in (Float64, Float32), (nu, Λ) in _GUO_SRT_PAIRS
        D = lat isa D3Q19 ? 3 : 2
        C, w, _ = _guo_lattice_tables(lat)
        F = D == 3 ? (0.03, -0.02, 0.01) : (0.03, -0.02)
        u0 = D == 3 ? [0.08, -0.05, 0.03] : [0.08, -0.05]
        f0 = _guo_uniform_state(C, w, 1.03, u0, 0.02)
        sp, sm = Kraken.trt_rates(nu; Λ=Λ)
        @assert abs(sp - sm) <= 2 * eps(sp) "pair ($nu, $Λ) is not an SRT pair: $sp vs $sm"
        _, f_trt = D == 3 ? _guo_brick_3d(T, f0, F, nu, Λ) : _guo_brick_2d(T, f0, F, nu, Λ)
        f_bgk = D == 3 ? _guo_bgk_3d(T, f0, F, sp) : _guo_bgk_2d(T, f0, F, sp)
        d = maximum(abs, f_trt .- f_bgk)
        println("TRT Guo SRT limit D=$D $T s=$(sp): max|f_TRT - f_BGK| = ", d,
                " (gate ", 16 * eps(T), ")")
        @test d <= 16 * eps(T)
    end
end

# ---------------------------------------------------------------------------
# (d) AD mirror. ad_ve_coupled_step! (src/ad/ad_ve_step.jl) carries its own
# copy of the TRT Guo collision. It must give the same post-collision
# populations as the 2D brick for the same pulled populations, force and rates.
# Setup: random moving non-equilibrium state, random log-conformation psi with a
# nonzero polymer prefactor (so the force has both x and y components that vary
# from cell to cell), a body force Fx_body. The force the AD step applies is not
# exposed, so it is read back from a run at s+ = s- = 1, where the momentum gain
# is F for the old and the new code alike (no even/odd ambiguity); the force
# pipeline (psi advection, constitutive update, divergence) does not depend on
# (s+, s-). The 2D brick is then run on the 3x3 neighbourhood of each probe cell
# with that force. Probe cells are fluid, have fluid neighbours and no cut link.
# Gate 64 eps(Float64) absolute on populations < 0.5: the read-back force
# carries the round-off of a 9-term moment (a few eps), which enters the
# populations through u* and the source with weights <= 3 w_q < 1.5.
# Float64 only: the AD operator is Float64 by construction.
# ---------------------------------------------------------------------------
using Random

function _guo_ad_state(Nx, Ny, rng)
    n = Nx * Ny
    w_in = zeros(Float64, 12n)
    C, wq, _ = _guo_lattice_tables(D2Q9())
    for j in 1:Ny, i in 1:Nx
        rho = 1 + 0.03 * (2rand(rng) - 1)
        u = 0.06 .* (2 .* rand(rng, 2) .- 1)
        feq = _guo_feq_ref(C, wq, rho, u)
        for q in 1:9
            w_in[Kraken.ad_ve_fpop(i, j, q, Nx, Ny)] = feq[q] * (1 + 0.02 * (2rand(rng) - 1))
        end
    end
    w_in[9n+1:12n] .= 0.2 .* (2 .* rand(rng, 3n) .- 1)
    return w_in
end

@testset "TRT Guo source: AD mirror equals the 2D brick" begin
    Nx, Ny = 24, 16
    geom = Kraken.ad_ve_build_geom(Nx, Ny, 5.35, 8.15, 1.6; samples=8, u_mean=0.0)
    w_in = _guo_ad_state(Nx, Ny, MersenneTwister(20261008))
    Fb = 4e-3
    adp(sp, sm) = Kraken.ADVECoupledParams(Nx, Ny, 0.5, 0.05, 4, 0.08, 0.1, Fb, sp, sm)
    probes = ((12, 4), (15, 12), (19, 7), (10, 13))
    for (i, j) in probes, di in -1:1, dj in -1:1
        @assert !geom.g.is_solid[i+di, j+dj]
    end
    for (i, j) in probes
        @assert all(iszero, geom.q_wall[i, j, :])
    end
    cx = Kraken.velocities_x(D2Q9()); cy = Kraken.velocities_y(D2Q9())
    pulled(i, j) = [w_in[Kraken.ad_ve_fpop(i - cx[q], j - cy[q], q, Nx, Ny)] for q in 1:9]
    out_at(w, i, j) = [w[Kraken.ad_ve_fpop(i, j, q, Nx, Ny)] for q in 1:9]

    # Force read-back at s+ = s- = 1.
    w1 = zeros(Float64, length(w_in))
    Kraken.ad_ve_coupled_step!(w1, w_in, geom.g, geom.q_wall, adp(1.0, 1.0), geom.u_profile)
    Fcell = Dict(p => [sum(c[q] * (out_at(w1, p...)[q] - pulled(p...)[q]) for q in 1:9)
                       for c in (cx, cy)] for p in probes)
    println("AD mirror probe forces: ", Fcell)
    @test all(p -> abs(Fcell[p][2]) > 1e-4 && abs(Fcell[p][1] - Fb) > 1e-5, probes)

    # The brick takes (nu, Λ); the AD step takes (s+, s-). Use the same pairs.
    function brick_at(i, j, F, nu, Λ)
        f_in = zeros(Float64, 3, 3, 9)
        for jj in 1:3, ii in 1:3, q in 1:9
            f_in[ii, jj, q] = w_in[Kraken.ad_ve_fpop(i + ii - 2, j + jj - 2, q, Nx, Ny)]
        end
        f_out = zeros(Float64, 3, 3, 9)
        Kraken.fused_trt_libb_v2_guo_field_step!(f_out, f_in,
            ones(3, 3), zeros(3, 3), zeros(3, 3), zeros(Bool, 3, 3),
            zeros(3, 3, 9), zeros(3, 3, 9), zeros(3, 3, 9),
            fill(F[1], 3, 3), fill(F[2], 3, 3), 3, 3, nu; Λ=Λ)
        return f_out[2, 2, :]
    end

    gate = 64 * eps(Float64)
    for (nu, Λ) in ((1 / 6, 1 / 4), (0.10, 3 / 16), (1 / 6, 3 / 16), (0.04, 1 / 12))
        sp, sm = Kraken.trt_rates(nu; Λ=Λ)
        w_out = zeros(Float64, length(w_in))
        Kraken.ad_ve_coupled_step!(w_out, w_in, geom.g, geom.q_wall, adp(sp, sm), geom.u_profile)
        for p in probes
            d = maximum(abs, out_at(w_out, p...) .- brick_at(p..., Fcell[p], nu, Λ))
            println("AD mirror vs 2D brick: s+=$(sp) s-=$(sm) cell=$(p) max|df| = ", d,
                    " (gate ", gate, ")")
            @test d <= gate
        end
    end
end
