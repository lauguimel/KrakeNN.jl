# =====================================================================
# Kernel DSL — brick library.
#
# Each brick is a singleton type + `required_args` + `emit_code`.
# Emitted `Expr` blocks assume the shared local-variable vocabulary:
#   fp1..fp9          pulled populations (post-stream, pre-collision)
#   ρ, ux, uy, usq    moments (pre-collision)
#   feq1..feq9        equilibrium populations
#   fp1c..fp9c        post-collision populations (intermediate)
#   fp2_new..fp9_new  post-BC-overwrite populations (LI-BB)
#   T                 eltype(f_out), set once at kernel entry
#
# Code fragments are COPIED VERBATIM from existing hand-written fused
# kernels so the generated code is bit-exact. Do not refactor for
# "clarity" — downstream tests assert equality under `.==`.
# =====================================================================

# ------------------------------------------------------------------
# Streaming
# ------------------------------------------------------------------

"Pull-stream D2Q9 with halfway-BB fallback at domain edges."
struct PullHalfwayBB <: LBMBrick end
required_args(::PullHalfwayBB) = (:f_in, :Nx, :Ny)
phase(::PullHalfwayBB) = :pre_solid
emit_code(::PullHalfwayBB) = quote
    fp1 = f_in[i, j, 1]
    # Clamped indices keep ifelse branchless (GPU-safe) while avoiding
    # out-of-bounds reads that cause segfaults on CPU at page boundaries.
    fp2 = ifelse(i > 1,             f_in[max(i - 1, 1),  j,              2], f_in[i, j, 4])
    fp3 = ifelse(j > 1,             f_in[i,              max(j - 1, 1),  3], f_in[i, j, 5])
    fp4 = ifelse(i < Nx,            f_in[min(i + 1, Nx), j,              4], f_in[i, j, 2])
    fp5 = ifelse(j < Ny,            f_in[i,              min(j + 1, Ny), 5], f_in[i, j, 3])
    fp6 = ifelse(i > 1  && j > 1,   f_in[max(i - 1, 1),  max(j - 1, 1),  6], f_in[i, j, 8])
    fp7 = ifelse(i < Nx && j > 1,   f_in[min(i + 1, Nx), max(j - 1, 1),  7], f_in[i, j, 9])
    fp8 = ifelse(i < Nx && j < Ny,  f_in[min(i + 1, Nx), min(j + 1, Ny), 8], f_in[i, j, 6])
    fp9 = ifelse(i > 1  && j < Ny,  f_in[max(i - 1, 1),  min(j + 1, Ny), 9], f_in[i, j, 7])
end

"Semi-Lagrangian pull D2Q9: interpolate f_in at precomputed departure points."
struct PullSLBM <: LBMBrick end
required_args(::PullSLBM) = (:f_in, :i_dep, :j_dep, :Nx, :Ny, :periodic_ξ, :periodic_η)
phase(::PullSLBM) = :pre_solid
emit_code(::PullSLBM) = quote
    fp1 = bilinear_f(f_in, i_dep[i, j, 1], j_dep[i, j, 1], 1, Nx, Ny, periodic_ξ, periodic_η)
    fp2 = bilinear_f(f_in, i_dep[i, j, 2], j_dep[i, j, 2], 2, Nx, Ny, periodic_ξ, periodic_η)
    fp3 = bilinear_f(f_in, i_dep[i, j, 3], j_dep[i, j, 3], 3, Nx, Ny, periodic_ξ, periodic_η)
    fp4 = bilinear_f(f_in, i_dep[i, j, 4], j_dep[i, j, 4], 4, Nx, Ny, periodic_ξ, periodic_η)
    fp5 = bilinear_f(f_in, i_dep[i, j, 5], j_dep[i, j, 5], 5, Nx, Ny, periodic_ξ, periodic_η)
    fp6 = bilinear_f(f_in, i_dep[i, j, 6], j_dep[i, j, 6], 6, Nx, Ny, periodic_ξ, periodic_η)
    fp7 = bilinear_f(f_in, i_dep[i, j, 7], j_dep[i, j, 7], 7, Nx, Ny, periodic_ξ, periodic_η)
    fp8 = bilinear_f(f_in, i_dep[i, j, 8], j_dep[i, j, 8], 8, Nx, Ny, periodic_ξ, periodic_η)
    fp9 = bilinear_f(f_in, i_dep[i, j, 9], j_dep[i, j, 9], 9, Nx, Ny, periodic_ξ, periodic_η)
end

"Semi-Lagrangian pull D2Q9 with biquadratic (3×3 Lagrange) interpolation. O(Δx³) instead of O(Δx²) for bilinear — critical on stretched meshes where the interpolation error dominates the numerical diffusion."
struct PullSLBMBiquad <: LBMBrick end
required_args(::PullSLBMBiquad) = (:f_in, :is_solid, :i_dep, :j_dep, :Nx, :Ny, :periodic_ξ, :periodic_η)
phase(::PullSLBMBiquad) = :pre_solid
emit_code(::PullSLBMBiquad) = quote
    fp1 = biquadratic_f(f_in, is_solid, i_dep[i, j, 1], j_dep[i, j, 1], 1, Nx, Ny, periodic_ξ, periodic_η)
    fp2 = biquadratic_f(f_in, is_solid, i_dep[i, j, 2], j_dep[i, j, 2], 2, Nx, Ny, periodic_ξ, periodic_η)
    fp3 = biquadratic_f(f_in, is_solid, i_dep[i, j, 3], j_dep[i, j, 3], 3, Nx, Ny, periodic_ξ, periodic_η)
    fp4 = biquadratic_f(f_in, is_solid, i_dep[i, j, 4], j_dep[i, j, 4], 4, Nx, Ny, periodic_ξ, periodic_η)
    fp5 = biquadratic_f(f_in, is_solid, i_dep[i, j, 5], j_dep[i, j, 5], 5, Nx, Ny, periodic_ξ, periodic_η)
    fp6 = biquadratic_f(f_in, is_solid, i_dep[i, j, 6], j_dep[i, j, 6], 6, Nx, Ny, periodic_ξ, periodic_η)
    fp7 = biquadratic_f(f_in, is_solid, i_dep[i, j, 7], j_dep[i, j, 7], 7, Nx, Ny, periodic_ξ, periodic_η)
    fp8 = biquadratic_f(f_in, is_solid, i_dep[i, j, 8], j_dep[i, j, 8], 8, Nx, Ny, periodic_ξ, periodic_η)
    fp9 = biquadratic_f(f_in, is_solid, i_dep[i, j, 9], j_dep[i, j, 9], 9, Nx, Ny, periodic_ξ, periodic_η)
end

"Non-equilibrium rescaling for SLBM on non-uniform meshes. After interpolation, the f_neq part is scaled for the departure cell tau. This brick corrects it to the arrival cell tau: f = f_eq + (τ_arr-0.5)/(τ_dep-0.5) * f_neq."
struct RescaleNonEq <: LBMBrick end
required_args(::RescaleNonEq) = (:s_plus, :i_dep, :j_dep, :Nx, :Ny, :periodic_ξ, :periodic_η)
phase(::RescaleNonEq) = :pre_solid
emit_code(::RescaleNonEq) = quote
    ρ_d = fp1 + fp2 + fp3 + fp4 + fp5 + fp6 + fp7 + fp8 + fp9
    inv_ρ_d = one(T) / ρ_d
    ux_d = (fp2 - fp4 + fp6 - fp7 - fp8 + fp9) * inv_ρ_d
    uy_d = (fp3 - fp5 + fp6 + fp7 - fp8 - fp9) * inv_ρ_d
    usq_d = ux_d * ux_d + uy_d * uy_d

    τ_arr = one(T) / s_plus[i, j]

    # Departure tau: nearest node to the departure point of direction 2
    i0_d = max(1, min(Nx, unsafe_trunc(Int, floor(i_dep[i, j, 2] + T(0.5)))))
    j0_d = max(1, min(Ny, unsafe_trunc(Int, floor(j_dep[i, j, 2] + T(0.5)))))
    τ_dep = one(T) / s_plus[i0_d, j0_d]

    r_neq = (τ_arr - T(0.5)) / max(τ_dep - T(0.5), T(1e-6))

    feq1_d = feq_2d(Val(1), ρ_d, ux_d, uy_d, usq_d)
    feq2_d = feq_2d(Val(2), ρ_d, ux_d, uy_d, usq_d)
    feq3_d = feq_2d(Val(3), ρ_d, ux_d, uy_d, usq_d)
    feq4_d = feq_2d(Val(4), ρ_d, ux_d, uy_d, usq_d)
    feq5_d = feq_2d(Val(5), ρ_d, ux_d, uy_d, usq_d)
    feq6_d = feq_2d(Val(6), ρ_d, ux_d, uy_d, usq_d)
    feq7_d = feq_2d(Val(7), ρ_d, ux_d, uy_d, usq_d)
    feq8_d = feq_2d(Val(8), ρ_d, ux_d, uy_d, usq_d)
    feq9_d = feq_2d(Val(9), ρ_d, ux_d, uy_d, usq_d)

    fp1 = feq1_d + r_neq * (fp1 - feq1_d)
    fp2 = feq2_d + r_neq * (fp2 - feq2_d)
    fp3 = feq3_d + r_neq * (fp3 - feq3_d)
    fp4 = feq4_d + r_neq * (fp4 - feq4_d)
    fp5 = feq5_d + r_neq * (fp5 - feq5_d)
    fp6 = feq6_d + r_neq * (fp6 - feq6_d)
    fp7 = feq7_d + r_neq * (fp7 - feq7_d)
    fp8 = feq8_d + r_neq * (fp8 - feq8_d)
    fp9 = feq9_d + r_neq * (fp9 - feq9_d)
end

# ------------------------------------------------------------------
# Solid-cell handling
# ------------------------------------------------------------------

"Legacy: solid cells do bounce-back swap on pulled pops. Bug cause for LI-BB."
struct SolidSwapBB <: LBMBrick end
required_args(::SolidSwapBB) = (:is_solid, :f_out, :ρ_out, :ux_out, :uy_out)
phase(::SolidSwapBB) = :solid
emit_code(::SolidSwapBB) = quote
    f_out[i, j, 1] = fp1
    f_out[i, j, 2] = fp4; f_out[i, j, 4] = fp2
    f_out[i, j, 3] = fp5; f_out[i, j, 5] = fp3
    f_out[i, j, 6] = fp8; f_out[i, j, 8] = fp6
    f_out[i, j, 7] = fp9; f_out[i, j, 9] = fp7
    ρ_out[i, j] = one(T)
    ux_out[i, j] = zero(T)
    uy_out[i, j] = zero(T)
end

"Fix: solid cells carry REST-EQUILIBRIUM populations (ρ=1, u=0 → f_q = w_q). Paired with ApplyLiBB for cut-link BCs. Unlike bare-zero inertia, this keeps the mass sourced to fluid neighbours physically sensible (any fluid cell pulling a population from a solid cell reads a w_q, not 0, so the intermediate post-collision fp_qc used by ApplyLiBB's fallback branches stays well-scaled)."
struct SolidInert <: LBMBrick end
required_args(::SolidInert) = (:is_solid, :f_out, :ρ_out, :ux_out, :uy_out)
phase(::SolidInert) = :solid
emit_code(::SolidInert) = quote
    f_out[i, j, 1] = feq_2d(Val(1), one(T), zero(T), zero(T), zero(T))
    f_out[i, j, 2] = feq_2d(Val(2), one(T), zero(T), zero(T), zero(T))
    f_out[i, j, 3] = feq_2d(Val(3), one(T), zero(T), zero(T), zero(T))
    f_out[i, j, 4] = feq_2d(Val(4), one(T), zero(T), zero(T), zero(T))
    f_out[i, j, 5] = feq_2d(Val(5), one(T), zero(T), zero(T), zero(T))
    f_out[i, j, 6] = feq_2d(Val(6), one(T), zero(T), zero(T), zero(T))
    f_out[i, j, 7] = feq_2d(Val(7), one(T), zero(T), zero(T), zero(T))
    f_out[i, j, 8] = feq_2d(Val(8), one(T), zero(T), zero(T), zero(T))
    f_out[i, j, 9] = feq_2d(Val(9), one(T), zero(T), zero(T), zero(T))
    ρ_out[i, j] = one(T)
    ux_out[i, j] = zero(T)
    uy_out[i, j] = zero(T)
end

# ------------------------------------------------------------------
# Moments
# ------------------------------------------------------------------

"Compute ρ, ux, uy, usq from fp1..fp9."
struct Moments <: LBMBrick end
required_args(::Moments) = ()
emit_code(::Moments) = quote
    ρ, ux, uy = moments_2d(fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9)
    usq = ux * ux + uy * uy
end

"Recompute ρ, ux, uy from post-BC populations (fp1c + fp*_new for q=2..9). Used after ApplyLiBB."
struct RecomputeMoments <: LBMBrick end
required_args(::RecomputeMoments) = ()
emit_code(::RecomputeMoments) = quote
    ρ = fp1c + fp2_new + fp3_new + fp4_new + fp5_new +
        fp6_new + fp7_new + fp8_new + fp9_new
    ux = (fp2_new - fp4_new + fp6_new - fp8_new + fp9_new - fp7_new) / ρ
    uy = (fp3_new - fp5_new + fp6_new - fp8_new - fp9_new + fp7_new) / ρ
end

# ------------------------------------------------------------------
# Collisions — direct write (for fused_bgk / fused_trt oracles)
# ------------------------------------------------------------------

"BGK collision, written directly to f_out[i, j, :]. Matches fused_bgk_step_kernel!."
struct CollideBGKDirect <: LBMBrick end
required_args(::CollideBGKDirect) = (:f_out, :ω)
emit_code(::CollideBGKDirect) = quote
    f_out[i, j, 1] = fp1 - ω * (fp1 - feq_2d(Val(1), ρ, ux, uy, usq))
    f_out[i, j, 2] = fp2 - ω * (fp2 - feq_2d(Val(2), ρ, ux, uy, usq))
    f_out[i, j, 3] = fp3 - ω * (fp3 - feq_2d(Val(3), ρ, ux, uy, usq))
    f_out[i, j, 4] = fp4 - ω * (fp4 - feq_2d(Val(4), ρ, ux, uy, usq))
    f_out[i, j, 5] = fp5 - ω * (fp5 - feq_2d(Val(5), ρ, ux, uy, usq))
    f_out[i, j, 6] = fp6 - ω * (fp6 - feq_2d(Val(6), ρ, ux, uy, usq))
    f_out[i, j, 7] = fp7 - ω * (fp7 - feq_2d(Val(7), ρ, ux, uy, usq))
    f_out[i, j, 8] = fp8 - ω * (fp8 - feq_2d(Val(8), ρ, ux, uy, usq))
    f_out[i, j, 9] = fp9 - ω * (fp9 - feq_2d(Val(9), ρ, ux, uy, usq))
end

"TRT collision with per-cell relaxation rates from 2D arrays s_plus[i,j], s_minus[i,j]."
struct CollideTRTLocalDirect <: LBMBrick end
required_args(::CollideTRTLocalDirect) = (:f_out, :s_plus, :s_minus)
emit_code(::CollideTRTLocalDirect) = quote
    feq1 = feq_2d(Val(1), ρ, ux, uy, usq)
    feq2 = feq_2d(Val(2), ρ, ux, uy, usq)
    feq3 = feq_2d(Val(3), ρ, ux, uy, usq)
    feq4 = feq_2d(Val(4), ρ, ux, uy, usq)
    feq5 = feq_2d(Val(5), ρ, ux, uy, usq)
    feq6 = feq_2d(Val(6), ρ, ux, uy, usq)
    feq7 = feq_2d(Val(7), ρ, ux, uy, usq)
    feq8 = feq_2d(Val(8), ρ, ux, uy, usq)
    feq9 = feq_2d(Val(9), ρ, ux, uy, usq)
    sp_local = s_plus[i, j]
    sm_local = s_minus[i, j]
    a = (sp_local + sm_local) * T(0.5)
    b = (sp_local - sm_local) * T(0.5)
    f_out[i, j, 1] = fp1 - sp_local * (fp1 - feq1)
    f_out[i, j, 2] = fp2 - a * (fp2 - feq2) - b * (fp4 - feq4)
    f_out[i, j, 4] = fp4 - a * (fp4 - feq4) - b * (fp2 - feq2)
    f_out[i, j, 3] = fp3 - a * (fp3 - feq3) - b * (fp5 - feq5)
    f_out[i, j, 5] = fp5 - a * (fp5 - feq5) - b * (fp3 - feq3)
    f_out[i, j, 6] = fp6 - a * (fp6 - feq6) - b * (fp8 - feq8)
    f_out[i, j, 8] = fp8 - a * (fp8 - feq8) - b * (fp6 - feq6)
    f_out[i, j, 7] = fp7 - a * (fp7 - feq7) - b * (fp9 - feq9)
    f_out[i, j, 9] = fp9 - a * (fp9 - feq9) - b * (fp7 - feq7)
end

"""Regularized TRT collision with per-cell rates (Latt & Chopard 2006).
Reconstructs f_neq from the physical stress tensor Π⁽¹⁾ only, filtering
ghost modes that cause instability at τ→0.5. The TRT magic parameter
is embedded in s_minus. Stable down to τ ≈ 0.5001."""
struct CollideRegularizedTRTLocal <: LBMBrick end
required_args(::CollideRegularizedTRTLocal) = (:f_out, :s_plus, :s_minus)
emit_code(::CollideRegularizedTRTLocal) = quote
    feq1 = feq_2d(Val(1), ρ, ux, uy, usq)
    feq2 = feq_2d(Val(2), ρ, ux, uy, usq)
    feq3 = feq_2d(Val(3), ρ, ux, uy, usq)
    feq4 = feq_2d(Val(4), ρ, ux, uy, usq)
    feq5 = feq_2d(Val(5), ρ, ux, uy, usq)
    feq6 = feq_2d(Val(6), ρ, ux, uy, usq)
    feq7 = feq_2d(Val(7), ρ, ux, uy, usq)
    feq8 = feq_2d(Val(8), ρ, ux, uy, usq)
    feq9 = feq_2d(Val(9), ρ, ux, uy, usq)
    sp_local = s_plus[i, j]
    sm_local = s_minus[i, j]
    # Stress tensor Π⁽¹⁾ = Σ_q c_qi c_qj (f_q - f_q^eq)
    # D2Q9 velocities: q1=(0,0), q2=(1,0), q3=(0,1), q4=(-1,0), q5=(0,-1),
    #                  q6=(1,1), q7=(-1,1), q8=(-1,-1), q9=(1,-1)
    Pxx = (fp2-feq2) + (fp4-feq4) + (fp6-feq6) + (fp7-feq7) + (fp8-feq8) + (fp9-feq9)
    Pyy = (fp3-feq3) + (fp5-feq5) + (fp6-feq6) + (fp7-feq7) + (fp8-feq8) + (fp9-feq9)
    Pxy = (fp6-feq6) - (fp7-feq7) + (fp8-feq8) - (fp9-feq9)
    # Reconstruct regularized f_neq from Π only (Latt 2006, Eq. 17)
    # f_neq_reg[q] = w[q]/(2 cs⁴) × (c_qi c_qj - cs² δ_ij) Π_ij
    # cs² = 1/3, cs⁴ = 1/9
    # For q=1 (0,0): (-1/3 Pxx - 1/3 Pyy) × 4/9 / (2/9) = ...
    # Simplified per-direction formulas:
    inv2 = T(0.5)
    fneq1 = -inv2 * T(2/9) * (Pxx + Pyy)
    fneq2 =  inv2 * T(1/9) * (T(2)*Pxx - Pyy)
    fneq3 =  inv2 * T(1/9) * (-Pxx + T(2)*Pyy)
    fneq4 =  inv2 * T(1/9) * (T(2)*Pxx - Pyy)
    fneq5 =  inv2 * T(1/9) * (-Pxx + T(2)*Pyy)
    fneq6 =  inv2 * T(1/36) * (Pxx + Pyy) + T(1/4) * Pxy
    fneq7 =  inv2 * T(1/36) * (Pxx + Pyy) - T(1/4) * Pxy
    fneq8 =  inv2 * T(1/36) * (Pxx + Pyy) + T(1/4) * Pxy
    fneq9 =  inv2 * T(1/36) * (Pxx + Pyy) - T(1/4) * Pxy
    # TRT relaxation on regularized f_neq (symmetric/antisymmetric split)
    a = (sp_local + sm_local) * T(0.5)
    b = (sp_local - sm_local) * T(0.5)
    f_out[i, j, 1] = feq1 + (one(T) - sp_local) * fneq1
    f_out[i, j, 2] = feq2 + (one(T) - a) * fneq2 - b * fneq4
    f_out[i, j, 4] = feq4 + (one(T) - a) * fneq4 - b * fneq2
    f_out[i, j, 3] = feq3 + (one(T) - a) * fneq3 - b * fneq5
    f_out[i, j, 5] = feq5 + (one(T) - a) * fneq5 - b * fneq3
    f_out[i, j, 6] = feq6 + (one(T) - a) * fneq6 - b * fneq8
    f_out[i, j, 8] = feq8 + (one(T) - a) * fneq8 - b * fneq6
    f_out[i, j, 7] = feq7 + (one(T) - a) * fneq7 - b * fneq9
    f_out[i, j, 9] = feq9 + (one(T) - a) * fneq9 - b * fneq7
end

"TRT collision, written directly to f_out[i, j, :]. Matches fused_trt_step_kernel!."
struct CollideTRTDirect <: LBMBrick end
required_args(::CollideTRTDirect) = (:f_out, :s_plus, :s_minus)
emit_code(::CollideTRTDirect) = quote
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
    f_out[i, j, 1] = fp1 - s_plus * (fp1 - feq1)
    f_out[i, j, 2] = fp2 - a * (fp2 - feq2) - b * (fp4 - feq4)
    f_out[i, j, 4] = fp4 - a * (fp4 - feq4) - b * (fp2 - feq2)
    f_out[i, j, 3] = fp3 - a * (fp3 - feq3) - b * (fp5 - feq5)
    f_out[i, j, 5] = fp5 - a * (fp5 - feq5) - b * (fp3 - feq3)
    f_out[i, j, 6] = fp6 - a * (fp6 - feq6) - b * (fp8 - feq8)
    f_out[i, j, 8] = fp8 - a * (fp8 - feq8) - b * (fp6 - feq6)
    f_out[i, j, 7] = fp7 - a * (fp7 - feq7) - b * (fp9 - feq9)
    f_out[i, j, 9] = fp9 - a * (fp9 - feq9) - b * (fp7 - feq7)
end

"TRT collision with per-cell Guo body force, written directly to f_out."
struct CollideTRTDirectGuoField <: LBMBrick end
required_args(::CollideTRTDirectGuoField) =
    (:f_out, :s_plus, :s_minus, :Fx_field, :Fy_field)
emit_code(::CollideTRTDirectGuoField) = quote
    fx = Fx_field[i, j]
    fy = Fy_field[i, j]
    if fx != zero(T) || fy != zero(T)
        inv_ρ = one(T) / ρ
        ux = (ρ * ux + fx / T(2)) * inv_ρ
        uy = (ρ * uy + fy / T(2)) * inv_ρ
        usq = ux * ux + uy * uy
    end

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

    # Guo source S_q = w_q [3 (c_q - u)·F + 9 (c_q·u)(c_q·F)], split into its
    # even part S⁺_q = w_q [9 (c_q·u)(c_q·F) - 3 u·F] (second moment u F + F u)
    # and its odd part S⁻_q = 3 w_q c_q·F (first moment F). Each part is scaled
    # by the Crank–Nicolson factor of the rate that relaxes it,
    #   (1 - s_plus/2) S⁺_q + (1 - s_minus/2) S⁻_q,
    # which is the TRT form of the MRT source (I - D/2) M S, D the diagonal
    # matrix of relaxation rates (Silva 2020, Comput. Fluids 203, 104537,
    # doi:10.1016/j.compfluid.2020.104537; Premnath, Pattison & Banerjee 2009,
    # Phys. Rev. E 79, 026703, doi:10.1103/PhysRevE.79.026703). With the F/2
    # shift of the equilibrium velocity above, the first moment gains exactly F
    # per collision for any (s_plus, s_minus). A single (1 - s_plus/2) factor
    # on the whole S_q injects (1 + (s_minus - s_plus)/2) F instead.
    guo_even = one(T) - s_plus / T(2)
    guo_odd = one(T) - s_minus / T(2)
    uF = ux * fx + uy * fy

    Sp1 = T(4.0 / 9.0) * (-uF) * T(3)
    Sp24 = T(1.0 / 9.0) * (ux * fx * T(9) - uF * T(3))
    Sm24 = T(1.0 / 9.0) * fx * T(3)
    Sp35 = T(1.0 / 9.0) * (uy * fy * T(9) - uF * T(3))
    Sm35 = T(1.0 / 9.0) * fy * T(3)
    Sp68 = T(1.0 / 36.0) * ((ux + uy) * (fx + fy) * T(9) - uF * T(3))
    Sm68 = T(1.0 / 36.0) * (fx + fy) * T(3)
    Sp79 = T(1.0 / 36.0) * ((-ux + uy) * (-fx + fy) * T(9) - uF * T(3))
    Sm79 = T(1.0 / 36.0) * (-fx + fy) * T(3)

    f_out[i, j, 1] = fp1 - s_plus * (fp1 - feq1) + guo_even * Sp1
    f_out[i, j, 2] = fp2 - a * (fp2 - feq2) - b * (fp4 - feq4) + guo_even * Sp24 + guo_odd * Sm24
    f_out[i, j, 4] = fp4 - a * (fp4 - feq4) - b * (fp2 - feq2) + guo_even * Sp24 - guo_odd * Sm24
    f_out[i, j, 3] = fp3 - a * (fp3 - feq3) - b * (fp5 - feq5) + guo_even * Sp35 + guo_odd * Sm35
    f_out[i, j, 5] = fp5 - a * (fp5 - feq5) - b * (fp3 - feq3) + guo_even * Sp35 - guo_odd * Sm35
    f_out[i, j, 6] = fp6 - a * (fp6 - feq6) - b * (fp8 - feq8) + guo_even * Sp68 + guo_odd * Sm68
    f_out[i, j, 8] = fp8 - a * (fp8 - feq8) - b * (fp6 - feq6) + guo_even * Sp68 - guo_odd * Sm68
    f_out[i, j, 7] = fp7 - a * (fp7 - feq7) - b * (fp9 - feq9) + guo_even * Sp79 + guo_odd * Sm79
    f_out[i, j, 9] = fp9 - a * (fp9 - feq9) - b * (fp7 - feq7) + guo_even * Sp79 - guo_odd * Sm79
end

"TRT collision with Liu/Yu Hermite stress source written directly to f_out."
struct CollideTRTDirectHermite <: LBMBrick end
required_args(::CollideTRTDirectHermite) =
    (:f_out, :s_plus, :s_minus, :q_wall,
     :tau_p_xx, :tau_p_xy, :tau_p_yy, :source_scale, :source_on_cutlinks)
emit_code(::CollideTRTDirectHermite) = quote
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

    cut_link = false
    if !source_on_cutlinks
        for qsrc in 2:9
            cut_link |= q_wall[i, j, qsrc] > zero(T)
        end
    end
    txx = tau_p_xx[i, j]
    txy = tau_p_xy[i, j]
    tyy = tau_p_yy[i, j]
    local_source_scale = cut_link ? zero(T) : source_scale
    pre = -s_plus * T(9.0 / 2.0) * local_source_scale
    cs2 = T(1 / 3)
    wr = T(4 / 9)
    wa = T(1 / 9)
    we = T(1 / 36)
    T1 = pre * wr * (-cs2 * (txx + tyy))
    T2 = pre * wa * ((one(T) - cs2) * txx - cs2 * tyy)
    T3 = pre * wa * (-cs2 * txx + (one(T) - cs2) * tyy)
    T6 = pre * we * ((one(T) - cs2) * txx + (one(T) - cs2) * tyy + T(2) * txy)
    T7 = pre * we * ((one(T) - cs2) * txx + (one(T) - cs2) * tyy - T(2) * txy)

    f_out[i, j, 1] = fp1 - s_plus * (fp1 - feq1) + T1
    f_out[i, j, 2] = fp2 - a * (fp2 - feq2) - b * (fp4 - feq4) + T2
    f_out[i, j, 4] = fp4 - a * (fp4 - feq4) - b * (fp2 - feq2) + T2
    f_out[i, j, 3] = fp3 - a * (fp3 - feq3) - b * (fp5 - feq5) + T3
    f_out[i, j, 5] = fp5 - a * (fp5 - feq5) - b * (fp3 - feq3) + T3
    f_out[i, j, 6] = fp6 - a * (fp6 - feq6) - b * (fp8 - feq8) + T6
    f_out[i, j, 8] = fp8 - a * (fp8 - feq8) - b * (fp6 - feq6) + T6
    f_out[i, j, 7] = fp7 - a * (fp7 - feq7) - b * (fp9 - feq9) + T7
    f_out[i, j, 9] = fp9 - a * (fp9 - feq9) - b * (fp7 - feq7) + T7
end

# ------------------------------------------------------------------
# Collisions — intermediate write (for fused_trt_libb)
# ------------------------------------------------------------------

"TRT collision writing to intermediate fp1c..fp9c. Used before ApplyLiBB overwrite."
struct CollideTRT <: LBMBrick end
required_args(::CollideTRT) = (:s_plus, :s_minus)
emit_code(::CollideTRT) = quote
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
    fp1c = fp1 - s_plus * (fp1 - feq1)
    fp2c = fp2 - a * (fp2 - feq2) - b * (fp4 - feq4)
    fp4c = fp4 - a * (fp4 - feq4) - b * (fp2 - feq2)
    fp3c = fp3 - a * (fp3 - feq3) - b * (fp5 - feq5)
    fp5c = fp5 - a * (fp5 - feq5) - b * (fp3 - feq3)
    fp6c = fp6 - a * (fp6 - feq6) - b * (fp8 - feq8)
    fp8c = fp8 - a * (fp8 - feq8) - b * (fp6 - feq6)
    fp7c = fp7 - a * (fp7 - feq7) - b * (fp9 - feq9)
    fp9c = fp9 - a * (fp9 - feq9) - b * (fp7 - feq7)
end

# ------------------------------------------------------------------
# Boundary conditions
# ------------------------------------------------------------------

"Pre-collision halfway-BB substitution on flagged cut links. Replaces the pulled pop fp_{q̄} (which came from a solid cell and is junk) with `f_in[i,j,q] + δ_{q̄}` — a lag-1 halfway-BB estimate, consistent with the classical halfway-BB storage at solids (which my DSL replaces with SolidInert equilibrium). At q_w=0.5 this is Bouzidi for pre-collision. Used alongside ApplyLiBB (post-collision) to close the moment-consistency loop."
struct ApplyHalfwayBBPrePhase <: LBMBrick end
required_args(::ApplyHalfwayBBPrePhase) = (:f_in, :q_wall, :uw_link_x, :uw_link_y)
emit_code(::ApplyHalfwayBBPrePhase) = quote
    # Pair (2, 4): link q=2 flagged → corrupted pop is fp4.
    if q_wall[i, j, 2] > zero(T)
        δ4 = -T(2/3) * uw_link_x[i, j, 2]
        fp4 = f_in[i, j, 2] + δ4
    end
    if q_wall[i, j, 4] > zero(T)
        δ2 =  T(2/3) * uw_link_x[i, j, 4]
        fp2 = f_in[i, j, 4] + δ2
    end
    # Pair (3, 5)
    if q_wall[i, j, 3] > zero(T)
        δ5 = -T(2/3) * uw_link_y[i, j, 3]
        fp5 = f_in[i, j, 3] + δ5
    end
    if q_wall[i, j, 5] > zero(T)
        δ3 =  T(2/3) * uw_link_y[i, j, 5]
        fp3 = f_in[i, j, 5] + δ3
    end
    # Pair (6, 8)
    if q_wall[i, j, 6] > zero(T)
        δ8 = -T(1/6) * (uw_link_x[i, j, 6] + uw_link_y[i, j, 6])
        fp8 = f_in[i, j, 6] + δ8
    end
    if q_wall[i, j, 8] > zero(T)
        δ6 =  T(1/6) * (uw_link_x[i, j, 8] + uw_link_y[i, j, 8])
        fp6 = f_in[i, j, 8] + δ6
    end
    # Pair (7, 9)
    if q_wall[i, j, 7] > zero(T)
        δ9 = -T(1/6) * (-uw_link_x[i, j, 7] + uw_link_y[i, j, 7])
        fp9 = f_in[i, j, 7] + δ9
    end
    if q_wall[i, j, 9] > zero(T)
        δ7 =  T(1/6) * (-uw_link_x[i, j, 9] + uw_link_y[i, j, 9])
        fp7 = f_in[i, j, 9] + δ7
    end
end

"Pre-collision axes-only halfway-BB substitution. Like ApplyHalfwayBBPrePhase but skips diagonal pops (q=6..9) so that diagonals remain as rest-equilibrium from SolidInert. Hypothesis-testing brick for investigating the near-wall residual of ApplyHalfwayBBPrePhase on diagonal links."
struct ApplyHalfwayBBPrePhaseAxes <: LBMBrick end
required_args(::ApplyHalfwayBBPrePhaseAxes) = (:f_in, :q_wall, :uw_link_x, :uw_link_y)
emit_code(::ApplyHalfwayBBPrePhaseAxes) = quote
    if q_wall[i, j, 2] > zero(T)
        δ4 = -T(2/3) * uw_link_x[i, j, 2]
        fp4 = f_in[i, j, 2] + δ4
    end
    if q_wall[i, j, 4] > zero(T)
        δ2 =  T(2/3) * uw_link_x[i, j, 4]
        fp2 = f_in[i, j, 4] + δ2
    end
    if q_wall[i, j, 3] > zero(T)
        δ5 = -T(2/3) * uw_link_y[i, j, 3]
        fp5 = f_in[i, j, 3] + δ5
    end
    if q_wall[i, j, 5] > zero(T)
        δ3 =  T(2/3) * uw_link_y[i, j, 5]
        fp3 = f_in[i, j, 5] + δ3
    end
end

"Full Bouzidi interpolated bounce-back at the PRE-COLLISION phase. For arbitrary q_w ∈ (0, 1]. Substitutes each corrupted pulled pop fp_{q̄} using `_libb_branch(q_w, fp_post_here, fp_post_back, fp_bar_post_here, δ)` with all three values being lag-1 (post-collision from the previous step). This generalizes ApplyHalfwayBBPrePhase (which assumes q_w=0.5) to STL-style boundaries. At q_w=0.5 it reduces to ApplyHalfwayBBPrePhase exactly.

Mapping of the three f̃ arguments in a pull-stream-collide fused kernel:
 - `f_post_here` = f̃_q(x_f, t) ≈ f_in[i,j,q]  (pop q at current cell, post-coll-prev)
 - `f_post_back` = f̃_q(x_f − c_q, t) ≈ fp_q   (pulled pop q = post-coll-prev at opposite-wall neighbour)
 - `f_bar_post_here` = f̃_{q̄}(x_f, t) ≈ f_in[i,j,q̄]  (pop q̄ at current cell, post-coll-prev)"
struct ApplyLiBBPrePhase <: LBMBrick end
required_args(::ApplyLiBBPrePhase) = (:f_in, :q_wall, :uw_link_x, :uw_link_y)
emit_code(::ApplyLiBBPrePhase) = quote
    # Pair (2, 4): link q=2 flagged → corrupted pop is fp4 (=q̄ of q=2).
    qw2 = q_wall[i, j, 2]
    if qw2 > zero(T)
        δ4 = -T(2/3) * uw_link_x[i, j, 2]
        fp4 = _libb_branch(qw2, f_in[i, j, 2], fp2, f_in[i, j, 4], δ4)
    end
    qw4 = q_wall[i, j, 4]
    if qw4 > zero(T)
        δ2 =  T(2/3) * uw_link_x[i, j, 4]
        fp2 = _libb_branch(qw4, f_in[i, j, 4], fp4, f_in[i, j, 2], δ2)
    end
    # Pair (3, 5)
    qw3 = q_wall[i, j, 3]
    if qw3 > zero(T)
        δ5 = -T(2/3) * uw_link_y[i, j, 3]
        fp5 = _libb_branch(qw3, f_in[i, j, 3], fp3, f_in[i, j, 5], δ5)
    end
    qw5 = q_wall[i, j, 5]
    if qw5 > zero(T)
        δ3 =  T(2/3) * uw_link_y[i, j, 5]
        fp3 = _libb_branch(qw5, f_in[i, j, 5], fp5, f_in[i, j, 3], δ3)
    end
    # Pair (6, 8)
    qw6 = q_wall[i, j, 6]
    if qw6 > zero(T)
        δ8 = -T(1/6) * (uw_link_x[i, j, 6] + uw_link_y[i, j, 6])
        fp8 = _libb_branch(qw6, f_in[i, j, 6], fp6, f_in[i, j, 8], δ8)
    end
    qw8 = q_wall[i, j, 8]
    if qw8 > zero(T)
        δ6 =  T(1/6) * (uw_link_x[i, j, 8] + uw_link_y[i, j, 8])
        fp6 = _libb_branch(qw8, f_in[i, j, 8], fp8, f_in[i, j, 6], δ6)
    end
    # Pair (7, 9)
    qw7 = q_wall[i, j, 7]
    if qw7 > zero(T)
        δ9 = -T(1/6) * (-uw_link_x[i, j, 7] + uw_link_y[i, j, 7])
        fp9 = _libb_branch(qw7, f_in[i, j, 7], fp7, f_in[i, j, 9], δ9)
    end
    qw9 = q_wall[i, j, 9]
    if qw9 > zero(T)
        δ7 =  T(1/6) * (-uw_link_x[i, j, 9] + uw_link_y[i, j, 9])
        fp7 = _libb_branch(qw9, f_in[i, j, 9], fp9, f_in[i, j, 7], δ7)
    end
end

@inline function _bouzidi_fl_post_value(qw::T, f_q_here::T, f_q_ff::T,
                                        f_qbar_here::T, delta::T,
                                        has_ff::Bool) where {T}
    if qw <= T(0.5)
        if has_ff
            two_qw = T(2) * qw
            return two_qw * f_q_here + (one(T) - two_qw) * f_q_ff + delta
        else
            return f_q_here + delta
        end
    else
        inv_two_qw = one(T) / (T(2) * qw)
        return inv_two_qw * f_q_here + (one(T) - inv_two_qw) * f_qbar_here +
               delta * inv_two_qw
    end
end

"Post-collision Bouzidi-FL interpolated bounce-back. Runs after collision has written f_out, then overwrites f_out[i,j,qbar] on flagged cut links. The wall-cell terms use current-step post-collision values from f_out; the q <= 0.5 far-fluid term uses lag-1 f_in at x_f - c_q, with halfway-BB fallback when that neighbour is unavailable. The moving-wall correction follows the existing _libb_branch scaling convention."
struct ApplyBouzidiFLPostCollide <: LBMBrick end
required_args(::ApplyBouzidiFLPostCollide) =
    (:f_out, :f_in, :q_wall, :uw_link_x, :uw_link_y, :is_solid, :ρ_out, :Nx, :Ny)
phase(::ApplyBouzidiFLPostCollide) = :fluid
emit_code(::ApplyBouzidiFLPostCollide) = quote
    rho_w = ρ_out[i, j]
    half = T(0.5)

    f2_here = f_out[i, j, 2]
    f3_here = f_out[i, j, 3]
    f4_here = f_out[i, j, 4]
    f5_here = f_out[i, j, 5]
    f6_here = f_out[i, j, 6]
    f7_here = f_out[i, j, 7]
    f8_here = f_out[i, j, 8]
    f9_here = f_out[i, j, 9]

    qw2 = q_wall[i, j, 2]
    if qw2 > zero(T)
        delta4 = -(T(2) / T(3)) * rho_w * uw_link_x[i, j, 2]
        has_ff2 = false
        f2_ff = f2_here
        if qw2 <= half
            i2_ff = i - 1
            j2_ff = j
            has_ff2 = i2_ff >= 1 && i2_ff <= Nx && j2_ff >= 1 && j2_ff <= Ny && !is_solid[i2_ff, j2_ff]
            f2_ff = has_ff2 ? f_in[i2_ff, j2_ff, 2] : f2_here
        end
        f_out[i, j, 4] = _bouzidi_fl_post_value(qw2, f2_here, f2_ff, f4_here, delta4, qw2 <= half && has_ff2)
    end

    qw4 = q_wall[i, j, 4]
    if qw4 > zero(T)
        delta2 = (T(2) / T(3)) * rho_w * uw_link_x[i, j, 4]
        has_ff4 = false
        f4_ff = f4_here
        if qw4 <= half
            i4_ff = i + 1
            j4_ff = j
            has_ff4 = i4_ff >= 1 && i4_ff <= Nx && j4_ff >= 1 && j4_ff <= Ny && !is_solid[i4_ff, j4_ff]
            f4_ff = has_ff4 ? f_in[i4_ff, j4_ff, 4] : f4_here
        end
        f_out[i, j, 2] = _bouzidi_fl_post_value(qw4, f4_here, f4_ff, f2_here, delta2, qw4 <= half && has_ff4)
    end

    qw3 = q_wall[i, j, 3]
    if qw3 > zero(T)
        delta5 = -(T(2) / T(3)) * rho_w * uw_link_y[i, j, 3]
        has_ff3 = false
        f3_ff = f3_here
        if qw3 <= half
            i3_ff = i
            j3_ff = j - 1
            has_ff3 = i3_ff >= 1 && i3_ff <= Nx && j3_ff >= 1 && j3_ff <= Ny && !is_solid[i3_ff, j3_ff]
            f3_ff = has_ff3 ? f_in[i3_ff, j3_ff, 3] : f3_here
        end
        f_out[i, j, 5] = _bouzidi_fl_post_value(qw3, f3_here, f3_ff, f5_here, delta5, qw3 <= half && has_ff3)
    end

    qw5 = q_wall[i, j, 5]
    if qw5 > zero(T)
        delta3 = (T(2) / T(3)) * rho_w * uw_link_y[i, j, 5]
        has_ff5 = false
        f5_ff = f5_here
        if qw5 <= half
            i5_ff = i
            j5_ff = j + 1
            has_ff5 = i5_ff >= 1 && i5_ff <= Nx && j5_ff >= 1 && j5_ff <= Ny && !is_solid[i5_ff, j5_ff]
            f5_ff = has_ff5 ? f_in[i5_ff, j5_ff, 5] : f5_here
        end
        f_out[i, j, 3] = _bouzidi_fl_post_value(qw5, f5_here, f5_ff, f3_here, delta3, qw5 <= half && has_ff5)
    end

    qw6 = q_wall[i, j, 6]
    if qw6 > zero(T)
        delta8 = -(T(1) / T(6)) * rho_w * (uw_link_x[i, j, 6] + uw_link_y[i, j, 6])
        has_ff6 = false
        f6_ff = f6_here
        if qw6 <= half
            i6_ff = i - 1
            j6_ff = j - 1
            has_ff6 = i6_ff >= 1 && i6_ff <= Nx && j6_ff >= 1 && j6_ff <= Ny && !is_solid[i6_ff, j6_ff]
            f6_ff = has_ff6 ? f_in[i6_ff, j6_ff, 6] : f6_here
        end
        f_out[i, j, 8] = _bouzidi_fl_post_value(qw6, f6_here, f6_ff, f8_here, delta8, qw6 <= half && has_ff6)
    end

    qw8 = q_wall[i, j, 8]
    if qw8 > zero(T)
        delta6 = (T(1) / T(6)) * rho_w * (uw_link_x[i, j, 8] + uw_link_y[i, j, 8])
        has_ff8 = false
        f8_ff = f8_here
        if qw8 <= half
            i8_ff = i + 1
            j8_ff = j + 1
            has_ff8 = i8_ff >= 1 && i8_ff <= Nx && j8_ff >= 1 && j8_ff <= Ny && !is_solid[i8_ff, j8_ff]
            f8_ff = has_ff8 ? f_in[i8_ff, j8_ff, 8] : f8_here
        end
        f_out[i, j, 6] = _bouzidi_fl_post_value(qw8, f8_here, f8_ff, f6_here, delta6, qw8 <= half && has_ff8)
    end

    qw7 = q_wall[i, j, 7]
    if qw7 > zero(T)
        delta9 = -(T(1) / T(6)) * rho_w * (-uw_link_x[i, j, 7] + uw_link_y[i, j, 7])
        has_ff7 = false
        f7_ff = f7_here
        if qw7 <= half
            i7_ff = i + 1
            j7_ff = j - 1
            has_ff7 = i7_ff >= 1 && i7_ff <= Nx && j7_ff >= 1 && j7_ff <= Ny && !is_solid[i7_ff, j7_ff]
            f7_ff = has_ff7 ? f_in[i7_ff, j7_ff, 7] : f7_here
        end
        f_out[i, j, 9] = _bouzidi_fl_post_value(qw7, f7_here, f7_ff, f9_here, delta9, qw7 <= half && has_ff7)
    end

    qw9 = q_wall[i, j, 9]
    if qw9 > zero(T)
        delta7 = (T(1) / T(6)) * rho_w * (-uw_link_x[i, j, 9] + uw_link_y[i, j, 9])
        has_ff9 = false
        f9_ff = f9_here
        if qw9 <= half
            i9_ff = i - 1
            j9_ff = j + 1
            has_ff9 = i9_ff >= 1 && i9_ff <= Nx && j9_ff >= 1 && j9_ff <= Ny && !is_solid[i9_ff, j9_ff]
            f9_ff = has_ff9 ? f_in[i9_ff, j9_ff, 9] : f9_here
        end
        f_out[i, j, 7] = _bouzidi_fl_post_value(qw9, f9_here, f9_ff, f7_here, delta7, qw9 <= half && has_ff9)
    end
end

"""
Two-pass Bouzidi-FL interpolated bounce-back (pass-2 brick).

Runs after pass-1 (`_TRT_LIBB_V2_GUO_FIELD_SPEC`) has globally synchronised:
`f_out` and `ρ_out` now hold the *current step's* post-collision values
everywhere (no cross-thread race, no lag-1). Reads `f_q_here = f_out[i, j, q]`,
`f_q_ff = f_out[i_ff, j_ff, q]`, and `rho_w = ρ_out[i, j]` — all lag-0. Writes
`f_out[i, j, qbar]` on flagged cut links. Eliminates the q ≤ 0.5 lag mismatch
and the secondary lag-1 ρ_w issue documented in
`bench/viscoelastic_audit/M30_PHASE2B_AUDIT_VERDICT.md`.
"""
struct ApplyBouzidiFLPostCollideTwoPass <: LBMBrick end
required_args(::ApplyBouzidiFLPostCollideTwoPass) =
    (:f_out, :q_wall, :uw_link_x, :uw_link_y, :is_solid, :ρ_out, :Nx, :Ny)
phase(::ApplyBouzidiFLPostCollideTwoPass) = :fluid
emit_code(::ApplyBouzidiFLPostCollideTwoPass) = quote
    rho_w = ρ_out[i, j]
    half = T(0.5)

    f2_here = f_out[i, j, 2]
    f3_here = f_out[i, j, 3]
    f4_here = f_out[i, j, 4]
    f5_here = f_out[i, j, 5]
    f6_here = f_out[i, j, 6]
    f7_here = f_out[i, j, 7]
    f8_here = f_out[i, j, 8]
    f9_here = f_out[i, j, 9]

    qw2 = q_wall[i, j, 2]
    if qw2 > zero(T)
        delta4 = -(T(2) / T(3)) * rho_w * uw_link_x[i, j, 2]
        has_ff2 = false
        f2_ff = f2_here
        if qw2 <= half
            i2_ff = i - 1
            j2_ff = j
            has_ff2 = i2_ff >= 1 && i2_ff <= Nx && j2_ff >= 1 && j2_ff <= Ny && !is_solid[i2_ff, j2_ff]
            f2_ff = has_ff2 ? f_out[i2_ff, j2_ff, 2] : f2_here
        end
        f_out[i, j, 4] = _bouzidi_fl_post_value(qw2, f2_here, f2_ff, f4_here, delta4, qw2 <= half && has_ff2)
    end

    qw4 = q_wall[i, j, 4]
    if qw4 > zero(T)
        delta2 = (T(2) / T(3)) * rho_w * uw_link_x[i, j, 4]
        has_ff4 = false
        f4_ff = f4_here
        if qw4 <= half
            i4_ff = i + 1
            j4_ff = j
            has_ff4 = i4_ff >= 1 && i4_ff <= Nx && j4_ff >= 1 && j4_ff <= Ny && !is_solid[i4_ff, j4_ff]
            f4_ff = has_ff4 ? f_out[i4_ff, j4_ff, 4] : f4_here
        end
        f_out[i, j, 2] = _bouzidi_fl_post_value(qw4, f4_here, f4_ff, f2_here, delta2, qw4 <= half && has_ff4)
    end

    qw3 = q_wall[i, j, 3]
    if qw3 > zero(T)
        delta5 = -(T(2) / T(3)) * rho_w * uw_link_y[i, j, 3]
        has_ff3 = false
        f3_ff = f3_here
        if qw3 <= half
            i3_ff = i
            j3_ff = j - 1
            has_ff3 = i3_ff >= 1 && i3_ff <= Nx && j3_ff >= 1 && j3_ff <= Ny && !is_solid[i3_ff, j3_ff]
            f3_ff = has_ff3 ? f_out[i3_ff, j3_ff, 3] : f3_here
        end
        f_out[i, j, 5] = _bouzidi_fl_post_value(qw3, f3_here, f3_ff, f5_here, delta5, qw3 <= half && has_ff3)
    end

    qw5 = q_wall[i, j, 5]
    if qw5 > zero(T)
        delta3 = (T(2) / T(3)) * rho_w * uw_link_y[i, j, 5]
        has_ff5 = false
        f5_ff = f5_here
        if qw5 <= half
            i5_ff = i
            j5_ff = j + 1
            has_ff5 = i5_ff >= 1 && i5_ff <= Nx && j5_ff >= 1 && j5_ff <= Ny && !is_solid[i5_ff, j5_ff]
            f5_ff = has_ff5 ? f_out[i5_ff, j5_ff, 5] : f5_here
        end
        f_out[i, j, 3] = _bouzidi_fl_post_value(qw5, f5_here, f5_ff, f3_here, delta3, qw5 <= half && has_ff5)
    end

    qw6 = q_wall[i, j, 6]
    if qw6 > zero(T)
        delta8 = -(T(1) / T(6)) * rho_w * (uw_link_x[i, j, 6] + uw_link_y[i, j, 6])
        has_ff6 = false
        f6_ff = f6_here
        if qw6 <= half
            i6_ff = i - 1
            j6_ff = j - 1
            has_ff6 = i6_ff >= 1 && i6_ff <= Nx && j6_ff >= 1 && j6_ff <= Ny && !is_solid[i6_ff, j6_ff]
            f6_ff = has_ff6 ? f_out[i6_ff, j6_ff, 6] : f6_here
        end
        f_out[i, j, 8] = _bouzidi_fl_post_value(qw6, f6_here, f6_ff, f8_here, delta8, qw6 <= half && has_ff6)
    end

    qw8 = q_wall[i, j, 8]
    if qw8 > zero(T)
        delta6 = (T(1) / T(6)) * rho_w * (uw_link_x[i, j, 8] + uw_link_y[i, j, 8])
        has_ff8 = false
        f8_ff = f8_here
        if qw8 <= half
            i8_ff = i + 1
            j8_ff = j + 1
            has_ff8 = i8_ff >= 1 && i8_ff <= Nx && j8_ff >= 1 && j8_ff <= Ny && !is_solid[i8_ff, j8_ff]
            f8_ff = has_ff8 ? f_out[i8_ff, j8_ff, 8] : f8_here
        end
        f_out[i, j, 6] = _bouzidi_fl_post_value(qw8, f8_here, f8_ff, f6_here, delta6, qw8 <= half && has_ff8)
    end

    qw7 = q_wall[i, j, 7]
    if qw7 > zero(T)
        delta9 = -(T(1) / T(6)) * rho_w * (-uw_link_x[i, j, 7] + uw_link_y[i, j, 7])
        has_ff7 = false
        f7_ff = f7_here
        if qw7 <= half
            i7_ff = i + 1
            j7_ff = j - 1
            has_ff7 = i7_ff >= 1 && i7_ff <= Nx && j7_ff >= 1 && j7_ff <= Ny && !is_solid[i7_ff, j7_ff]
            f7_ff = has_ff7 ? f_out[i7_ff, j7_ff, 7] : f7_here
        end
        f_out[i, j, 9] = _bouzidi_fl_post_value(qw7, f7_here, f7_ff, f9_here, delta9, qw7 <= half && has_ff7)
    end

    qw9 = q_wall[i, j, 9]
    if qw9 > zero(T)
        delta7 = (T(1) / T(6)) * rho_w * (-uw_link_x[i, j, 9] + uw_link_y[i, j, 9])
        has_ff9 = false
        f9_ff = f9_here
        if qw9 <= half
            i9_ff = i - 1
            j9_ff = j + 1
            has_ff9 = i9_ff >= 1 && i9_ff <= Nx && j9_ff >= 1 && j9_ff <= Ny && !is_solid[i9_ff, j9_ff]
            f9_ff = has_ff9 ? f_out[i9_ff, j9_ff, 9] : f9_here
        end
        f_out[i, j, 7] = _bouzidi_fl_post_value(qw9, f9_here, f9_ff, f7_here, delta7, qw9 <= half && has_ff9)
    end
end

"""
Cut-link-only ρ recompute (pass-3 brick for `:bouzidi_fl_twopass`).

After pass-2 has overwritten `f_out[i, j, qbar]` on flagged cut links, the
stale `ρ_out` written by pass-1 (which summed the pre-Bouzidi-FL pops) is no
longer consistent with `f_out` at those cells. Downstream readers — the
log-FV polymer pipeline + the next step's Guo body force `f → ρ` chain — see
a `rho_w` that contradicts the cut-link f-set, the inconsistency identified
in `bench/viscoelastic_audit/M34_FIX_DIAG_VERDICT.md` as the residual cause
of the +1.6 % Cd bias at R=30 Wi=0.1 and the divergence at R=60 Wi=0.1 / R=30
Wi=1.0. This brick re-sums `f_out[i, j, 1..9]` and overwrites `ρ_out[i, j]`
ONLY on cells with at least one cut link (`q_wall[i, j, q] > 0` for some
q ∈ 2..9). Non-cut-link cells (including solids and pure-fluid bulk) keep
the pass-1 `ρ_out` bit-exact. M34v3, 2026-05-22.
"""
struct ApplyCutLinkRhoRecompute <: LBMBrick end
required_args(::ApplyCutLinkRhoRecompute) =
    (:f_out, :ρ_out, :q_wall, :is_solid, :Nx, :Ny)
phase(::ApplyCutLinkRhoRecompute) = :fluid
emit_code(::ApplyCutLinkRhoRecompute) = quote
    if !is_solid[i, j]
        any_cut = false
        @inbounds for qsrc in 2:9
            if q_wall[i, j, qsrc] > zero(T)
                any_cut = true
                break
            end
        end
        if any_cut
            ρ_out[i, j] = f_out[i, j, 1] + f_out[i, j, 2] + f_out[i, j, 3] +
                          f_out[i, j, 4] + f_out[i, j, 5] + f_out[i, j, 6] +
                          f_out[i, j, 7] + f_out[i, j, 8] + f_out[i, j, 9]
        end
    end
end

"Bouzidi interpolated bounce-back (LI-BB) overwrite on flagged cut links. Reads fp*c + fp*, writes fp*_new for q=2..9."
struct ApplyLiBB <: LBMBrick end
required_args(::ApplyLiBB) = (:q_wall, :uw_link_x, :uw_link_y)
emit_code(::ApplyLiBB) = quote
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
end

# ------------------------------------------------------------------
# Write-back
# ------------------------------------------------------------------

"Write raw fp1..fp9 to f_out[i, j, :]. Used for pull-only debug kernels."
struct WriteF <: LBMBrick end
required_args(::WriteF) = (:f_out,)
emit_code(::WriteF) = quote
    f_out[i, j, 1] = fp1
    f_out[i, j, 2] = fp2
    f_out[i, j, 3] = fp3
    f_out[i, j, 4] = fp4
    f_out[i, j, 5] = fp5
    f_out[i, j, 6] = fp6
    f_out[i, j, 7] = fp7
    f_out[i, j, 8] = fp8
    f_out[i, j, 9] = fp9
end

"Write f_out composed of fp1c (rest) + fp*_new for q=2..9. Used after ApplyLiBB."
struct WriteFLiBB <: LBMBrick end
required_args(::WriteFLiBB) = (:f_out,)
emit_code(::WriteFLiBB) = quote
    f_out[i, j, 1] = fp1c
    f_out[i, j, 2] = fp2_new; f_out[i, j, 4] = fp4_new
    f_out[i, j, 3] = fp3_new; f_out[i, j, 5] = fp5_new
    f_out[i, j, 6] = fp6_new; f_out[i, j, 8] = fp8_new
    f_out[i, j, 7] = fp7_new; f_out[i, j, 9] = fp9_new
end

"Write ρ, ux, uy to ρ_out, ux_out, uy_out at (i, j)."
struct WriteMoments <: LBMBrick end
required_args(::WriteMoments) = (:ρ_out, :ux_out, :uy_out)
emit_code(::WriteMoments) = quote
    ρ_out[i, j] = ρ
    ux_out[i, j] = ux
    uy_out[i, j] = uy
end
