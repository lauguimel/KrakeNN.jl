using KernelAbstractions

# --- 3D log-conformation primitives ---
#
# This file intentionally starts with the local building blocks needed by a
# 3D log-conf CDE: Psi <-> C, stress reconstruction, and the local
# Oldroyd-B log source. The population collision/driver integration should
# only be enabled after these low-level canaries are green.

@inline function _logconf_loewner_exp_inv(ψi, ψj, ci, cj)
    T = typeof(ψi)
    dψ = ψi - ψj
    dc = ci - cj
    return ifelse(abs(dc) > T(1e-12), dψ / dc, inv(ci))
end

@inline function _project_grad_sym3(
    duxdx, duxdy, duxdz,
    duydx, duydy, duydz,
    duzdx, duzdy, duzdz,
    v11, v21, v31,
    v12, v22, v32,
    v13, v23, v33,
)
    # L_ij = d u_i / d x_j. Return V' L V.
    l11 = v11 * (duxdx * v11 + duxdy * v21 + duxdz * v31) +
          v21 * (duydx * v11 + duydy * v21 + duydz * v31) +
          v31 * (duzdx * v11 + duzdy * v21 + duzdz * v31)
    l12 = v11 * (duxdx * v12 + duxdy * v22 + duxdz * v32) +
          v21 * (duydx * v12 + duydy * v22 + duydz * v32) +
          v31 * (duzdx * v12 + duzdy * v22 + duzdz * v32)
    l13 = v11 * (duxdx * v13 + duxdy * v23 + duxdz * v33) +
          v21 * (duydx * v13 + duydy * v23 + duydz * v33) +
          v31 * (duzdx * v13 + duzdy * v23 + duzdz * v33)

    l21 = v12 * (duxdx * v11 + duxdy * v21 + duxdz * v31) +
          v22 * (duydx * v11 + duydy * v21 + duydz * v31) +
          v32 * (duzdx * v11 + duzdy * v21 + duzdz * v31)
    l22 = v12 * (duxdx * v12 + duxdy * v22 + duxdz * v32) +
          v22 * (duydx * v12 + duydy * v22 + duydz * v32) +
          v32 * (duzdx * v12 + duzdy * v22 + duzdz * v32)
    l23 = v12 * (duxdx * v13 + duxdy * v23 + duxdz * v33) +
          v22 * (duydx * v13 + duydy * v23 + duydz * v33) +
          v32 * (duzdx * v13 + duzdy * v23 + duzdz * v33)

    l31 = v13 * (duxdx * v11 + duxdy * v21 + duxdz * v31) +
          v23 * (duydx * v11 + duydy * v21 + duydz * v31) +
          v33 * (duzdx * v11 + duzdy * v21 + duzdz * v31)
    l32 = v13 * (duxdx * v12 + duxdy * v22 + duxdz * v32) +
          v23 * (duydx * v12 + duydy * v22 + duydz * v32) +
          v33 * (duzdx * v12 + duzdy * v22 + duzdz * v32)
    l33 = v13 * (duxdx * v13 + duxdy * v23 + duxdz * v33) +
          v23 * (duydx * v13 + duydy * v23 + duydz * v33) +
          v33 * (duzdx * v13 + duzdy * v23 + duzdz * v33)

    return l11, l12, l13, l21, l22, l23, l31, l32, l33
end

@inline function logconf_source_3d(
    ψxx::T, ψxy::T, ψxz::T, ψyy::T, ψyz::T, ψzz::T,
    duxdx::T, duxdy::T, duxdz::T,
    duydx::T, duydy::T, duydz::T,
    duzdx::T, duzdy::T, duzdz::T,
    λ::T, component::Int,
) where {T<:AbstractFloat}
    return logconf_source_with_divergence_3d(
        ψxx, ψxy, ψxz, ψyy, ψyz, ψzz,
        duxdx, duxdy, duxdz,
        duydx, duydy, duydz,
        duzdx, duzdy, duzdz,
        duxdx + duydy + duzdz,
        λ, component,
    )
end

@inline function logconf_source_with_divergence_3d(
    ψxx::T, ψxy::T, ψxz::T, ψyy::T, ψyz::T, ψzz::T,
    duxdx::T, duxdy::T, duxdz::T,
    duydx::T, duydy::T, duydz::T,
    duzdx::T, duzdy::T, duzdz::T,
    advective_divu::T, λ::T, component::Int,
) where {T<:AbstractFloat}
    ψ1, ψ2, ψ3,
    v11, v21, v31,
    v12, v22, v32,
    v13, v23, v33 = eigen_sym3x3(ψxx, ψxy, ψxz, ψyy, ψyz, ψzz)

    c1 = exp(ψ1)
    c2 = exp(ψ2)
    c3 = exp(ψ3)
    inv_λ = inv(λ)

    l11, l12, l13,
    l21, l22, l23,
    l31, l32, l33 = _project_grad_sym3(
        duxdx, duxdy, duxdz,
        duydx, duydy, duydz,
        duzdx, duzdy, duzdz,
        v11, v21, v31, v12, v22, v32, v13, v23, v33,
    )

    s11p = T(2) * l11 - (one(T) - inv(c1)) * inv_λ
    s22p = T(2) * l22 - (one(T) - inv(c2)) * inv_λ
    s33p = T(2) * l33 - (one(T) - inv(c3)) * inv_λ
    s12p = _logconf_loewner_exp_inv(ψ1, ψ2, c1, c2) * (l12 * c2 + c1 * l21)
    s13p = _logconf_loewner_exp_inv(ψ1, ψ3, c1, c3) * (l13 * c3 + c1 * l31)
    s23p = _logconf_loewner_exp_inv(ψ2, ψ3, c2, c3) * (l23 * c3 + c2 * l32)

    sxx, sxy, sxz, syy, syz, szz = _sym3_from_eigenvalues(
        s11p, s22p, s33p,
        v11, v21, v31, v12, v22, v32, v13, v23, v33,
    )
    # Add off-diagonal eigenframe source before rotating. _sym3_from_eigenvalues
    # handles only diagonal spectra, so add R*Soff*R' explicitly.
    sxx += T(2) * (s12p * v11 * v12 + s13p * v11 * v13 + s23p * v12 * v13)
    sxy += s12p * (v11 * v22 + v21 * v12) +
           s13p * (v11 * v23 + v21 * v13) +
           s23p * (v12 * v23 + v22 * v13)
    sxz += s12p * (v11 * v32 + v31 * v12) +
           s13p * (v11 * v33 + v31 * v13) +
           s23p * (v12 * v33 + v32 * v13)
    syy += T(2) * (s12p * v21 * v22 + s13p * v21 * v23 + s23p * v22 * v23)
    syz += s12p * (v21 * v32 + v31 * v22) +
           s13p * (v21 * v33 + v31 * v23) +
           s23p * (v22 * v33 + v32 * v23)
    szz += T(2) * (s12p * v31 * v32 + s13p * v31 * v33 + s23p * v32 * v33)

    if component == 1
        return sxx + ψxx * advective_divu
    elseif component == 2
        return sxy + ψxy * advective_divu
    elseif component == 3
        return sxz + ψxz * advective_divu
    elseif component == 4
        return syy + ψyy * advective_divu
    elseif component == 5
        return syz + ψyz * advective_divu
    else
        return szz + ψzz * advective_divu
    end
end

# ---------------------------------------------------------------------
# FENE-P (Peterlin) log-conformation source.
#
# Identical eigenframe machinery to `logconf_source_with_divergence_3d`,
# but the upper-convected relaxation is multiplied by the Peterlin factor
#
#     f = (L² − 3) / (L² − tr C),    tr C = Σ exp(ψ_i),
#
# i.e. in the diagonal eigenframe the relaxation term becomes
# −(f − 1/c_i)·inv_λ instead of the Oldroyd-B −(1 − 1/c_i)·inv_λ.
#
# At equilibrium (C=I, tr C=3) → f=1, and `L2_fene ≤ 0` (or `L²→∞`)
# returns f=1 → the Oldroyd-B source is recovered BIT-IDENTICALLY (the
# expression collapses to exactly the OB diagonal term). The off-diagonal
# eigenframe couplings (`s12p, s13p, s23p`) are unchanged: the FENE-P
# spring force is isotropic in the relaxation and only rescales the
# diagonal restoring term, exactly as for Oldroyd-B.
# ---------------------------------------------------------------------
@inline function logconf_source_with_divergence_fenep_3d(
    ψxx::T, ψxy::T, ψxz::T, ψyy::T, ψyz::T, ψzz::T,
    duxdx::T, duxdy::T, duxdz::T,
    duydx::T, duydy::T, duydz::T,
    duzdx::T, duzdy::T, duzdz::T,
    advective_divu::T, λ::T, L2_fene::T, component::Int,
) where {T<:AbstractFloat}
    ψ1, ψ2, ψ3,
    v11, v21, v31,
    v12, v22, v32,
    v13, v23, v33 = eigen_sym3x3(ψxx, ψxy, ψxz, ψyy, ψyz, ψzz)

    c1 = exp(ψ1)
    c2 = exp(ψ2)
    c3 = exp(ψ3)
    inv_λ = inv(λ)

    # Peterlin factor f = (L²−3)/(L²−trC). f=1 when L2_fene<=0 (OB limit)
    # or as L²→∞. The denominator is clamped strictly positive so that the
    # finite-extensibility wall (trC→L²) never produces a NaN/Inf source.
    trC = c1 + c2 + c3
    fene = ifelse(
        L2_fene > zero(T),
        (L2_fene - T(3)) / max(L2_fene - trC, T(1e-6) * L2_fene),
        one(T),
    )

    l11, l12, l13,
    l21, l22, l23,
    l31, l32, l33 = _project_grad_sym3(
        duxdx, duxdy, duxdz,
        duydx, duydy, duydz,
        duzdx, duzdy, duzdz,
        v11, v21, v31, v12, v22, v32, v13, v23, v33,
    )

    s11p = T(2) * l11 - (fene - inv(c1)) * inv_λ
    s22p = T(2) * l22 - (fene - inv(c2)) * inv_λ
    s33p = T(2) * l33 - (fene - inv(c3)) * inv_λ
    s12p = _logconf_loewner_exp_inv(ψ1, ψ2, c1, c2) * (l12 * c2 + c1 * l21)
    s13p = _logconf_loewner_exp_inv(ψ1, ψ3, c1, c3) * (l13 * c3 + c1 * l31)
    s23p = _logconf_loewner_exp_inv(ψ2, ψ3, c2, c3) * (l23 * c3 + c2 * l32)

    sxx, sxy, sxz, syy, syz, szz = _sym3_from_eigenvalues(
        s11p, s22p, s33p,
        v11, v21, v31, v12, v22, v32, v13, v23, v33,
    )
    sxx += T(2) * (s12p * v11 * v12 + s13p * v11 * v13 + s23p * v12 * v13)
    sxy += s12p * (v11 * v22 + v21 * v12) +
           s13p * (v11 * v23 + v21 * v13) +
           s23p * (v12 * v23 + v22 * v13)
    sxz += s12p * (v11 * v32 + v31 * v12) +
           s13p * (v11 * v33 + v31 * v13) +
           s23p * (v12 * v33 + v32 * v13)
    syy += T(2) * (s12p * v21 * v22 + s13p * v21 * v23 + s23p * v22 * v23)
    syz += s12p * (v21 * v32 + v31 * v22) +
           s13p * (v21 * v33 + v31 * v23) +
           s23p * (v22 * v33 + v32 * v23)
    szz += T(2) * (s12p * v31 * v32 + s13p * v31 * v33 + s23p * v32 * v33)

    if component == 1
        return sxx + ψxx * advective_divu
    elseif component == 2
        return sxy + ψxy * advective_divu
    elseif component == 3
        return sxz + ψxz * advective_divu
    elseif component == 4
        return syy + ψyy * advective_divu
    elseif component == 5
        return syz + ψyz * advective_divu
    else
        return szz + ψzz * advective_divu
    end
end

# ---------------------------------------------------------------------
# Giesekus log-conformation source.
#
# Identical eigenframe machinery to `logconf_source_with_divergence_3d`,
# but the upper-convected relaxation gains the Giesekus anisotropic
# (quadratic) mobility term. The Giesekus conformation relaxation is
#
#     −(1/λ)[ (C − I) + α·(C − I)² ],
#
# which in the diagonal eigenframe is, per eigenvalue c_i,
#
#     (c_i − 1)·(1 + α·(c_i − 1)).
#
# Mapped to the log variable (chain rule dΨ_i/dt = (1/c_i)·dc_i/dt) the
# eigenframe diagonal restoring term becomes
#
#     −(1 − 1/c_i)·(1 + α·(c_i − 1))·inv_λ
#
# instead of the Oldroyd-B −(1 − 1/c_i)·inv_λ. At α = 0 the factor
# (1 + α(c_i−1)) is exactly 1, so the expression collapses to the OB
# diagonal term BIT-IDENTICALLY and the Oldroyd-B source is recovered.
# The off-diagonal eigenframe couplings (`s12p, s13p, s23p`) are the
# deformation/advection contribution and are unchanged: the Giesekus
# mobility only rescales the diagonal restoring term.
# ---------------------------------------------------------------------
@inline function logconf_source_with_divergence_giesekus_3d(
    ψxx::T, ψxy::T, ψxz::T, ψyy::T, ψyz::T, ψzz::T,
    duxdx::T, duxdy::T, duxdz::T,
    duydx::T, duydy::T, duydz::T,
    duzdx::T, duzdy::T, duzdz::T,
    advective_divu::T, λ::T, α::T, component::Int,
) where {T<:AbstractFloat}
    ψ1, ψ2, ψ3,
    v11, v21, v31,
    v12, v22, v32,
    v13, v23, v33 = eigen_sym3x3(ψxx, ψxy, ψxz, ψyy, ψyz, ψzz)

    c1 = exp(ψ1)
    c2 = exp(ψ2)
    c3 = exp(ψ3)
    inv_λ = inv(λ)

    l11, l12, l13,
    l21, l22, l23,
    l31, l32, l33 = _project_grad_sym3(
        duxdx, duxdy, duxdz,
        duydx, duydy, duydz,
        duzdx, duzdy, duzdz,
        v11, v21, v31, v12, v22, v32, v13, v23, v33,
    )

    # Giesekus quadratic mobility factor per eigenvalue. At α=0 this is
    # exactly 1 → the OB diagonal term −(1 − 1/c_i)·inv_λ is recovered
    # bit-identically.
    g1 = one(T) + α * (c1 - one(T))
    g2 = one(T) + α * (c2 - one(T))
    g3 = one(T) + α * (c3 - one(T))

    s11p = T(2) * l11 - (one(T) - inv(c1)) * g1 * inv_λ
    s22p = T(2) * l22 - (one(T) - inv(c2)) * g2 * inv_λ
    s33p = T(2) * l33 - (one(T) - inv(c3)) * g3 * inv_λ
    s12p = _logconf_loewner_exp_inv(ψ1, ψ2, c1, c2) * (l12 * c2 + c1 * l21)
    s13p = _logconf_loewner_exp_inv(ψ1, ψ3, c1, c3) * (l13 * c3 + c1 * l31)
    s23p = _logconf_loewner_exp_inv(ψ2, ψ3, c2, c3) * (l23 * c3 + c2 * l32)

    sxx, sxy, sxz, syy, syz, szz = _sym3_from_eigenvalues(
        s11p, s22p, s33p,
        v11, v21, v31, v12, v22, v32, v13, v23, v33,
    )
    sxx += T(2) * (s12p * v11 * v12 + s13p * v11 * v13 + s23p * v12 * v13)
    sxy += s12p * (v11 * v22 + v21 * v12) +
           s13p * (v11 * v23 + v21 * v13) +
           s23p * (v12 * v23 + v22 * v13)
    sxz += s12p * (v11 * v32 + v31 * v12) +
           s13p * (v11 * v33 + v31 * v13) +
           s23p * (v12 * v33 + v32 * v13)
    syy += T(2) * (s12p * v21 * v22 + s13p * v21 * v23 + s23p * v22 * v23)
    syz += s12p * (v21 * v32 + v31 * v22) +
           s13p * (v21 * v33 + v31 * v23) +
           s23p * (v22 * v33 + v32 * v23)
    szz += T(2) * (s12p * v31 * v32 + s13p * v31 * v33 + s23p * v32 * v33)

    if component == 1
        return sxx + ψxx * advective_divu
    elseif component == 2
        return sxy + ψxy * advective_divu
    elseif component == 3
        return sxz + ψxz * advective_divu
    elseif component == 4
        return syy + ψyy * advective_divu
    elseif component == 5
        return syz + ψyz * advective_divu
    else
        return szz + ψzz * advective_divu
    end
end

# ---------------------------------------------------------------------
# Phan-Thien–Tanner (PTT) log-conformation source.
#
# Identical eigenframe machinery to `logconf_source_with_divergence_3d`,
# but the upper-convected relaxation is multiplied by a SCALAR function
# `Y(trC)` of the trace `trC = Σ exp(ψ_i)` (non-affine slip ξ = 0, i.e.
# upper-convected PTT). The PTT conformation relaxation is
#
#     −(1/λ)·Y(trC)·(C − I),
#
# which in the diagonal eigenframe is, per eigenvalue c_i,
#
#     Y(trC)·(c_i − 1),
#
# with the SAME scalar `Y(trC)` shared by every eigenvalue (unlike the
# Giesekus per-eigenvalue factor `1 + α(c_i−1)`). Mapped to the log
# variable (chain rule dΨ_i/dt = (1/c_i)·dc_i/dt) the eigenframe diagonal
# restoring term becomes
#
#     −Y(trC)·(1 − 1/c_i)·inv_λ
#
# instead of the Oldroyd-B −(1 − 1/c_i)·inv_λ. With
#
#     Y(trC) = 1 + ε·(trC − 3)          (variant == 1, linear PTT)
#     Y(trC) = exp( ε·(trC − 3) )       (variant == 2, exponential PTT)
#
# At ε = 0 the multiplier is exactly 1, so the expression collapses to the
# OB diagonal term BIT-IDENTICALLY and the Oldroyd-B source is recovered.
# The off-diagonal eigenframe couplings (`s12p, s13p, s23p`) are the
# deformation/advection contribution and are unchanged: the PTT trace
# multiplier only rescales the diagonal restoring term.
# ---------------------------------------------------------------------
@inline function logconf_source_with_divergence_ptt_3d(
    ψxx::T, ψxy::T, ψxz::T, ψyy::T, ψyz::T, ψzz::T,
    duxdx::T, duxdy::T, duxdz::T,
    duydx::T, duydy::T, duydz::T,
    duzdx::T, duzdy::T, duzdz::T,
    advective_divu::T, λ::T, ε::T, variant::Int, component::Int,
) where {T<:AbstractFloat}
    ψ1, ψ2, ψ3,
    v11, v21, v31,
    v12, v22, v32,
    v13, v23, v33 = eigen_sym3x3(ψxx, ψxy, ψxz, ψyy, ψyz, ψzz)

    c1 = exp(ψ1)
    c2 = exp(ψ2)
    c3 = exp(ψ3)
    inv_λ = inv(λ)

    l11, l12, l13,
    l21, l22, l23,
    l31, l32, l33 = _project_grad_sym3(
        duxdx, duxdy, duxdz,
        duydx, duydy, duydz,
        duzdx, duzdy, duzdz,
        v11, v21, v31, v12, v22, v32, v13, v23, v33,
    )

    # PTT scalar trace multiplier Y(trC), shared by all eigenvalues.
    # At ε=0 → Y=1 exactly (both variants), so the diagonal restoring term
    # −Y·(1 − 1/c_i)·inv_λ collapses to the OB term bit-identically.
    trC = c1 + c2 + c3
    Y_trC = ifelse(
        variant == 2,
        exp(ε * (trC - T(3))),
        one(T) + ε * (trC - T(3)),
    )

    s11p = T(2) * l11 - Y_trC * (one(T) - inv(c1)) * inv_λ
    s22p = T(2) * l22 - Y_trC * (one(T) - inv(c2)) * inv_λ
    s33p = T(2) * l33 - Y_trC * (one(T) - inv(c3)) * inv_λ
    s12p = _logconf_loewner_exp_inv(ψ1, ψ2, c1, c2) * (l12 * c2 + c1 * l21)
    s13p = _logconf_loewner_exp_inv(ψ1, ψ3, c1, c3) * (l13 * c3 + c1 * l31)
    s23p = _logconf_loewner_exp_inv(ψ2, ψ3, c2, c3) * (l23 * c3 + c2 * l32)

    sxx, sxy, sxz, syy, syz, szz = _sym3_from_eigenvalues(
        s11p, s22p, s33p,
        v11, v21, v31, v12, v22, v32, v13, v23, v33,
    )
    sxx += T(2) * (s12p * v11 * v12 + s13p * v11 * v13 + s23p * v12 * v13)
    sxy += s12p * (v11 * v22 + v21 * v12) +
           s13p * (v11 * v23 + v21 * v13) +
           s23p * (v12 * v23 + v22 * v13)
    sxz += s12p * (v11 * v32 + v31 * v12) +
           s13p * (v11 * v33 + v31 * v13) +
           s23p * (v12 * v33 + v32 * v13)
    syy += T(2) * (s12p * v21 * v22 + s13p * v21 * v23 + s23p * v22 * v23)
    syz += s12p * (v21 * v32 + v31 * v22) +
           s13p * (v21 * v33 + v31 * v23) +
           s23p * (v22 * v33 + v32 * v23)
    szz += T(2) * (s12p * v31 * v32 + s13p * v31 * v33 + s23p * v32 * v33)

    if component == 1
        return sxx + ψxx * advective_divu
    elseif component == 2
        return sxy + ψxy * advective_divu
    elseif component == 3
        return sxz + ψxz * advective_divu
    elseif component == 4
        return syy + ψyy * advective_divu
    elseif component == 5
        return syz + ψyz * advective_divu
    else
        return szz + ψzz * advective_divu
    end
end

@kernel function collide_logconf_3d_kernel!(
    g, @Const(Ψ_field),
    @Const(ux), @Const(uy), @Const(uz),
    @Const(Ψ_xx), @Const(Ψ_xy), @Const(Ψ_xz),
    @Const(Ψ_yy), @Const(Ψ_yz), @Const(Ψ_zz),
    @Const(is_solid),
    tau_plus, tau_minus, lambda, component, Nx, Ny, Nz,
)
    i, j, k = @index(Global, NTuple)

    @inbounds if !is_solid[i, j, k]
        T = eltype(g)
        φ = Ψ_field[i, j, k]
        u = ux[i, j, k]
        v = uy[i, j, k]
        w = uz[i, j, k]
        usq = u * u + v * v + w * w

        ip = ifelse(i < Nx, i + 1, 1)
        im = ifelse(i > 1, i - 1, Nx)
        jp = min(j + 1, Ny)
        jm = max(j - 1, 1)
        kp = min(k + 1, Nz)
        km = max(k - 1, 1)

        duxdx = (ux[ip, j, k] - ux[im, j, k]) / T(2)
        duxdy = (ux[i, jp, k] - ux[i, jm, k]) / T(2)
        duxdz = (ux[i, j, kp] - ux[i, j, km]) / T(2)
        duydx = (uy[ip, j, k] - uy[im, j, k]) / T(2)
        duydy = (uy[i, jp, k] - uy[i, jm, k]) / T(2)
        duydz = (uy[i, j, kp] - uy[i, j, km]) / T(2)
        duzdx = (uz[ip, j, k] - uz[im, j, k]) / T(2)
        duzdy = (uz[i, jp, k] - uz[i, jm, k]) / T(2)
        duzdz = (uz[i, j, kp] - uz[i, j, km]) / T(2)

        S = logconf_source_3d(
            Ψ_xx[i, j, k], Ψ_xy[i, j, k], Ψ_xz[i, j, k],
            Ψ_yy[i, j, k], Ψ_yz[i, j, k], Ψ_zz[i, j, k],
            duxdx, duxdy, duxdz,
            duydx, duydy, duydz,
            duzdx, duzdy, duzdz,
            T(lambda), component,
        )

        g1  = g[i, j, k, 1];  g2  = g[i, j, k, 2];  g3  = g[i, j, k, 3]
        g4  = g[i, j, k, 4];  g5  = g[i, j, k, 5];  g6  = g[i, j, k, 6]
        g7  = g[i, j, k, 7];  g8  = g[i, j, k, 8];  g9  = g[i, j, k, 9]
        g10 = g[i, j, k, 10]; g11 = g[i, j, k, 11]; g12 = g[i, j, k, 12]
        g13 = g[i, j, k, 13]; g14 = g[i, j, k, 14]; g15 = g[i, j, k, 15]
        g16 = g[i, j, k, 16]; g17 = g[i, j, k, 17]; g18 = g[i, j, k, 18]
        g19 = g[i, j, k, 19]

        ge1  = feq_3d(Val(1),  φ, u, v, w, usq)
        ge2  = feq_3d(Val(2),  φ, u, v, w, usq)
        ge3  = feq_3d(Val(3),  φ, u, v, w, usq)
        ge4  = feq_3d(Val(4),  φ, u, v, w, usq)
        ge5  = feq_3d(Val(5),  φ, u, v, w, usq)
        ge6  = feq_3d(Val(6),  φ, u, v, w, usq)
        ge7  = feq_3d(Val(7),  φ, u, v, w, usq)
        ge8  = feq_3d(Val(8),  φ, u, v, w, usq)
        ge9  = feq_3d(Val(9),  φ, u, v, w, usq)
        ge10 = feq_3d(Val(10), φ, u, v, w, usq)
        ge11 = feq_3d(Val(11), φ, u, v, w, usq)
        ge12 = feq_3d(Val(12), φ, u, v, w, usq)
        ge13 = feq_3d(Val(13), φ, u, v, w, usq)
        ge14 = feq_3d(Val(14), φ, u, v, w, usq)
        ge15 = feq_3d(Val(15), φ, u, v, w, usq)
        ge16 = feq_3d(Val(16), φ, u, v, w, usq)
        ge17 = feq_3d(Val(17), φ, u, v, w, usq)
        ge18 = feq_3d(Val(18), φ, u, v, w, usq)
        ge19 = feq_3d(Val(19), φ, u, v, w, usq)

        ωp = one(T) / T(tau_plus)
        ωm = one(T) / T(tau_minus)
        half = T(0.5)
        wr = T(1 / 3)
        wa = T(1 / 18)
        we = T(1 / 36)

        g[i, j, k, 1] = g1 - ωp * (g1 - ge1) + wr * S

        gp23 = (g2 + g3) * half;  gm23 = (g2 - g3) * half
        ep23 = (ge2 + ge3) * half; em23 = (ge2 - ge3) * half
        g[i, j, k, 2] = g2 - ωp * (gp23 - ep23) - ωm * (gm23 - em23) + wa * S
        g[i, j, k, 3] = g3 - ωp * (gp23 - ep23) + ωm * (gm23 - em23) + wa * S

        gp45 = (g4 + g5) * half;  gm45 = (g4 - g5) * half
        ep45 = (ge4 + ge5) * half; em45 = (ge4 - ge5) * half
        g[i, j, k, 4] = g4 - ωp * (gp45 - ep45) - ωm * (gm45 - em45) + wa * S
        g[i, j, k, 5] = g5 - ωp * (gp45 - ep45) + ωm * (gm45 - em45) + wa * S

        gp67 = (g6 + g7) * half;  gm67 = (g6 - g7) * half
        ep67 = (ge6 + ge7) * half; em67 = (ge6 - ge7) * half
        g[i, j, k, 6] = g6 - ωp * (gp67 - ep67) - ωm * (gm67 - em67) + wa * S
        g[i, j, k, 7] = g7 - ωp * (gp67 - ep67) + ωm * (gm67 - em67) + wa * S

        gp_a = (g8 + g11) * half;  gm_a = (g8 - g11) * half
        ep_a = (ge8 + ge11) * half; em_a = (ge8 - ge11) * half
        g[i, j, k, 8]  = g8  - ωp * (gp_a - ep_a) - ωm * (gm_a - em_a) + we * S
        g[i, j, k, 11] = g11 - ωp * (gp_a - ep_a) + ωm * (gm_a - em_a) + we * S

        gp_b = (g9 + g10) * half;  gm_b = (g9 - g10) * half
        ep_b = (ge9 + ge10) * half; em_b = (ge9 - ge10) * half
        g[i, j, k, 9]  = g9  - ωp * (gp_b - ep_b) - ωm * (gm_b - em_b) + we * S
        g[i, j, k, 10] = g10 - ωp * (gp_b - ep_b) + ωm * (gm_b - em_b) + we * S

        gp_c = (g12 + g15) * half;  gm_c = (g12 - g15) * half
        ep_c = (ge12 + ge15) * half; em_c = (ge12 - ge15) * half
        g[i, j, k, 12] = g12 - ωp * (gp_c - ep_c) - ωm * (gm_c - em_c) + we * S
        g[i, j, k, 15] = g15 - ωp * (gp_c - ep_c) + ωm * (gm_c - em_c) + we * S

        gp_d = (g13 + g14) * half;  gm_d = (g13 - g14) * half
        ep_d = (ge13 + ge14) * half; em_d = (ge13 - ge14) * half
        g[i, j, k, 13] = g13 - ωp * (gp_d - ep_d) - ωm * (gm_d - em_d) + we * S
        g[i, j, k, 14] = g14 - ωp * (gp_d - ep_d) + ωm * (gm_d - em_d) + we * S

        gp_e = (g16 + g19) * half;  gm_e = (g16 - g19) * half
        ep_e = (ge16 + ge19) * half; em_e = (ge16 - ge19) * half
        g[i, j, k, 16] = g16 - ωp * (gp_e - ep_e) - ωm * (gm_e - em_e) + we * S
        g[i, j, k, 19] = g19 - ωp * (gp_e - ep_e) + ωm * (gm_e - em_e) + we * S

        gp_f = (g17 + g18) * half;  gm_f = (g17 - g18) * half
        ep_f = (ge17 + ge18) * half; em_f = (ge17 - ge18) * half
        g[i, j, k, 17] = g17 - ωp * (gp_f - ep_f) - ωm * (gm_f - em_f) + we * S
        g[i, j, k, 18] = g18 - ωp * (gp_f - ep_f) + ωm * (gm_f - em_f) + we * S
    end
end

function collide_logconf_3d!(
    g, Ψ_field, ux, uy, uz,
    Ψ_xx, Ψ_xy, Ψ_xz, Ψ_yy, Ψ_yz, Ψ_zz,
    is_solid, tau_plus, lambda;
    magic=0.25, component=1,
)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny, Nz = size(Ψ_field)
    T = eltype(g)
    tau_minus = magic / (tau_plus - 0.5) + 0.5
    kernel! = collide_logconf_3d_kernel!(backend)
    kernel!(
        g, Ψ_field, ux, uy, uz,
        Ψ_xx, Ψ_xy, Ψ_xz, Ψ_yy, Ψ_yz, Ψ_zz,
        is_solid, T(tau_plus), T(tau_minus), T(lambda),
        Int(component), Nx, Ny, Nz; ndrange=(Nx, Ny, Nz),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function psi_to_C_3d_kernel!(
    C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
    @Const(Ψ_xx), @Const(Ψ_xy), @Const(Ψ_xz),
    @Const(Ψ_yy), @Const(Ψ_yz), @Const(Ψ_zz),
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        cxx, cxy, cxz, cyy, cyz, czz = mat_exp_sym3x3(
            Ψ_xx[i, j, k], Ψ_xy[i, j, k], Ψ_xz[i, j, k],
            Ψ_yy[i, j, k], Ψ_yz[i, j, k], Ψ_zz[i, j, k],
        )
        C_xx[i, j, k] = cxx
        C_xy[i, j, k] = cxy
        C_xz[i, j, k] = cxz
        C_yy[i, j, k] = cyy
        C_yz[i, j, k] = cyz
        C_zz[i, j, k] = czz
    end
end

function psi_to_C_3d!(
    C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
    Ψ_xx, Ψ_xy, Ψ_xz, Ψ_yy, Ψ_yz, Ψ_zz,
)
    backend = KernelAbstractions.get_backend(C_xx)
    Nx, Ny, Nz = size(C_xx)
    kernel! = psi_to_C_3d_kernel!(backend)
    kernel!(
        C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
        Ψ_xx, Ψ_xy, Ψ_xz, Ψ_yy, Ψ_yz, Ψ_zz;
        ndrange=(Nx, Ny, Nz),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function C_to_psi_3d_kernel!(
    Ψ_xx, Ψ_xy, Ψ_xz, Ψ_yy, Ψ_yz, Ψ_zz,
    @Const(C_xx), @Const(C_xy), @Const(C_xz),
    @Const(C_yy), @Const(C_yz), @Const(C_zz),
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ψxx, ψxy, ψxz, ψyy, ψyz, ψzz = mat_log_spd_sym3x3(
            C_xx[i, j, k], C_xy[i, j, k], C_xz[i, j, k],
            C_yy[i, j, k], C_yz[i, j, k], C_zz[i, j, k],
        )
        Ψ_xx[i, j, k] = ψxx
        Ψ_xy[i, j, k] = ψxy
        Ψ_xz[i, j, k] = ψxz
        Ψ_yy[i, j, k] = ψyy
        Ψ_yz[i, j, k] = ψyz
        Ψ_zz[i, j, k] = ψzz
    end
end

function C_to_psi_3d!(
    Ψ_xx, Ψ_xy, Ψ_xz, Ψ_yy, Ψ_yz, Ψ_zz,
    C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
)
    backend = KernelAbstractions.get_backend(Ψ_xx)
    Nx, Ny, Nz = size(Ψ_xx)
    kernel! = C_to_psi_3d_kernel!(backend)
    kernel!(
        Ψ_xx, Ψ_xy, Ψ_xz, Ψ_yy, Ψ_yz, Ψ_zz,
        C_xx, C_xy, C_xz, C_yy, C_yz, C_zz;
        ndrange=(Nx, Ny, Nz),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function stress_from_logconf_3d_kernel!(
    tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz,
    @Const(Ψ_xx), @Const(Ψ_xy), @Const(Ψ_xz),
    @Const(Ψ_yy), @Const(Ψ_yz), @Const(Ψ_zz),
    G, L2_fene,
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        T = eltype(tau_xx)
        cxx, cxy, cxz, cyy, cyz, czz = mat_exp_sym3x3(
            Ψ_xx[i, j, k], Ψ_xy[i, j, k], Ψ_xz[i, j, k],
            Ψ_yy[i, j, k], Ψ_yz[i, j, k], Ψ_zz[i, j, k],
        )
        trC = cxx + cyy + czz
        fene = ifelse(L2_fene > zero(T),
                      L2_fene / max(L2_fene - trC, T(0.01)),
                      one(T))
        tau_xx[i, j, k] = G * fene * (cxx - one(T))
        tau_xy[i, j, k] = G * fene * cxy
        tau_xz[i, j, k] = G * fene * cxz
        tau_yy[i, j, k] = G * fene * (cyy - one(T))
        tau_yz[i, j, k] = G * fene * cyz
        tau_zz[i, j, k] = G * fene * (czz - one(T))
    end
end

function compute_stress_from_logconf_3d!(
    tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz,
    Ψ_xx, Ψ_xy, Ψ_xz, Ψ_yy, Ψ_yz, Ψ_zz;
    G=1.0, L_max=0.0,
)
    backend = KernelAbstractions.get_backend(tau_xx)
    Nx, Ny, Nz = size(tau_xx)
    FT = eltype(tau_xx)
    L2 = L_max > 0 ? FT(L_max * L_max) : FT(0)
    kernel! = stress_from_logconf_3d_kernel!(backend)
    kernel!(
        tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz,
        Ψ_xx, Ψ_xy, Ψ_xz, Ψ_yy, Ψ_yz, Ψ_zz,
        FT(G), L2; ndrange=(Nx, Ny, Nz),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end
