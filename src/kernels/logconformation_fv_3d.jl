"""
    logfv_oldroydb_subcycle_estimate_3d(max_grad_norm, lambda, dt=1; kwargs...)

Estimate a single global subcycle count for the 3D local Oldroyd-B
log-conformation constitutive step. The returned named tuple mirrors the 2D
FVFD estimator and is intended to be evaluated on the host before launching
the fixed-loop RK2 constitutive kernel.
"""
function logfv_oldroydb_subcycle_estimate_3d(
    max_grad_norm::Real,
    lambda::Real,
    dt::Real=1;
    relative_tolerance::Real=0.01,
    max_deformation_increment::Real=0.05,
    max_memory_deformation_increment::Real=Inf,
    min_substeps::Integer=1,
    max_substeps::Integer=64,
)
    max_grad_norm >= 0 || throw(ArgumentError("max_grad_norm must be non-negative"))
    lambda > 0 || throw(ArgumentError("lambda must be positive"))
    dt > 0 || throw(ArgumentError("dt must be positive"))
    max_deformation_increment > 0 ||
        throw(ArgumentError("max_deformation_increment must be positive"))
    max_memory_deformation_increment > 0 ||
        throw(ArgumentError("max_memory_deformation_increment must be positive"))
    min_substeps >= 1 || throw(ArgumentError("min_substeps must be >= 1"))
    max_substeps >= min_substeps ||
        throw(ArgumentError("max_substeps must be >= min_substeps"))

    relax_increment = dt / lambda
    deformation_increment = dt * max_grad_norm
    memory_deformation_increment = lambda * max_grad_norm
    max_relax_increment = logfv_oldroydb_split_relax_increment(relative_tolerance)
    relax_substeps = max(min_substeps, ceil(Int, relax_increment / max_relax_increment))
    deformation_substeps = max(min_substeps, ceil(Int, deformation_increment / max_deformation_increment))
    memory_deformation_substeps = max(
        min_substeps,
        ceil(Int, memory_deformation_increment / max_memory_deformation_increment),
    )
    raw_substeps = max(relax_substeps, deformation_substeps, memory_deformation_substeps)
    recommended = min(raw_substeps, max_substeps)

    return (;
        recommended,
        raw_substeps,
        relax_substeps,
        deformation_substeps,
        memory_deformation_substeps,
        clamped=raw_substeps > max_substeps,
        relax_increment,
        deformation_increment,
        memory_deformation_increment,
        max_relax_increment,
        max_deformation_increment,
        max_memory_deformation_increment,
        relative_tolerance,
    )
end

function logfv_recommended_oldroydb_substeps_3d(args...; kwargs...)
    return logfv_oldroydb_subcycle_estimate_3d(args...; kwargs...).recommended
end

@inline _logfv_grad_norm2_3d(a, b, c, d, e, f, g, h, p) =
    a * a + b * b + c * c + d * d + e * e + f * f + g * g + h * h + p * p

"""
    logfv_max_grad_norm_3d(duxdx, duxdy, duxdz, duydx, duydy, duydz, duzdx, duzdy, duzdz)

Maximum over the cells of the Frobenius norm of the velocity gradient, as a
`Float64`. The reduction runs on the arrays' own device and only the scalar
comes back, so a GPU run does not copy nine fields to the host every step.
The squares are summed in the arrays' element type.
"""
function logfv_max_grad_norm_3d(
    duxdx, duxdy, duxdz,
    duydx, duydy, duydz,
    duzdx, duzdy, duzdz,
)
    max_norm2 = mapreduce(
        _logfv_grad_norm2_3d, max,
        duxdx, duxdy, duxdz, duydx, duydy, duydz, duzdx, duzdy, duzdz;
        init=zero(eltype(duxdx)),
    )
    return sqrt(Float64(max_norm2))
end

@kernel function logfv_constitutive_step_log_3d_kernel!(
    psixx_out, psixy_out, psixz_out, psiyy_out, psiyz_out, psizz_out,
    @Const(psixx), @Const(psixy), @Const(psixz),
    @Const(psiyy), @Const(psiyz), @Const(psizz),
    @Const(duxdx), @Const(duxdy), @Const(duxdz),
    @Const(duydx), @Const(duydy), @Const(duydz),
    @Const(duzdx), @Const(duzdy), @Const(duzdz),
    lambda, dt, n_sub, Nx, Ny, Nz,
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny && k <= Nz
            T = eltype(psixx_out)
            pxx = psixx[i, j, k]
            pxy = psixy[i, j, k]
            pxz = psixz[i, j, k]
            pyy = psiyy[i, j, k]
            pyz = psiyz[i, j, k]
            pzz = psizz[i, j, k]
            l11 = duxdx[i, j, k]
            l12 = duxdy[i, j, k]
            l13 = duxdz[i, j, k]
            l21 = duydx[i, j, k]
            l22 = duydy[i, j, k]
            l23 = duydz[i, j, k]
            l31 = duzdx[i, j, k]
            l32 = duzdy[i, j, k]
            l33 = duzdz[i, j, k]
            h = T(dt) / T(n_sub)
            half_h = T(0.5) * h
            lambda_t = T(lambda)
            divu_transport = zero(T)

            for _ in 1:n_sub
                k1xx = logconf_source_with_divergence_3d(
                    pxx, pxy, pxz, pyy, pyz, pzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, 1,
                )
                k1xy = logconf_source_with_divergence_3d(
                    pxx, pxy, pxz, pyy, pyz, pzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, 2,
                )
                k1xz = logconf_source_with_divergence_3d(
                    pxx, pxy, pxz, pyy, pyz, pzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, 3,
                )
                k1yy = logconf_source_with_divergence_3d(
                    pxx, pxy, pxz, pyy, pyz, pzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, 4,
                )
                k1yz = logconf_source_with_divergence_3d(
                    pxx, pxy, pxz, pyy, pyz, pzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, 5,
                )
                k1zz = logconf_source_with_divergence_3d(
                    pxx, pxy, pxz, pyy, pyz, pzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, 6,
                )

                qxx = pxx + h * k1xx
                qxy = pxy + h * k1xy
                qxz = pxz + h * k1xz
                qyy = pyy + h * k1yy
                qyz = pyz + h * k1yz
                qzz = pzz + h * k1zz

                k2xx = logconf_source_with_divergence_3d(
                    qxx, qxy, qxz, qyy, qyz, qzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, 1,
                )
                k2xy = logconf_source_with_divergence_3d(
                    qxx, qxy, qxz, qyy, qyz, qzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, 2,
                )
                k2xz = logconf_source_with_divergence_3d(
                    qxx, qxy, qxz, qyy, qyz, qzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, 3,
                )
                k2yy = logconf_source_with_divergence_3d(
                    qxx, qxy, qxz, qyy, qyz, qzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, 4,
                )
                k2yz = logconf_source_with_divergence_3d(
                    qxx, qxy, qxz, qyy, qyz, qzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, 5,
                )
                k2zz = logconf_source_with_divergence_3d(
                    qxx, qxy, qxz, qyy, qyz, qzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, 6,
                )

                pxx += half_h * (k1xx + k2xx)
                pxy += half_h * (k1xy + k2xy)
                pxz += half_h * (k1xz + k2xz)
                pyy += half_h * (k1yy + k2yy)
                pyz += half_h * (k1yz + k2yz)
                pzz += half_h * (k1zz + k2zz)
            end

            psixx_out[i, j, k] = pxx
            psixy_out[i, j, k] = pxy
            psixz_out[i, j, k] = pxz
            psiyy_out[i, j, k] = pyy
            psiyz_out[i, j, k] = pyz
            psizz_out[i, j, k] = pzz
        end
    end
end

# ---------------------------------------------------------------------
# FENE-P (Peterlin) variant of the unsplit RK2 constitutive step.
#
# Structurally identical to `logfv_constitutive_step_log_3d_kernel!`, but
# integrates `logconf_source_with_divergence_fenep_3d` (extra `L2_fene`
# scalar). With `L2_fene <= 0` the Peterlin factor is exactly 1 and this
# kernel reproduces the Oldroyd-B trajectory bit-for-bit; the dedicated OB
# kernel above is left untouched so the OB path never changes.
# ---------------------------------------------------------------------
@kernel function logfv_constitutive_step_log_fenep_3d_kernel!(
    psixx_out, psixy_out, psixz_out, psiyy_out, psiyz_out, psizz_out,
    @Const(psixx), @Const(psixy), @Const(psixz),
    @Const(psiyy), @Const(psiyz), @Const(psizz),
    @Const(duxdx), @Const(duxdy), @Const(duxdz),
    @Const(duydx), @Const(duydy), @Const(duydz),
    @Const(duzdx), @Const(duzdy), @Const(duzdz),
    lambda, dt, L2_fene, n_sub, Nx, Ny, Nz,
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny && k <= Nz
            T = eltype(psixx_out)
            pxx = psixx[i, j, k]
            pxy = psixy[i, j, k]
            pxz = psixz[i, j, k]
            pyy = psiyy[i, j, k]
            pyz = psiyz[i, j, k]
            pzz = psizz[i, j, k]
            l11 = duxdx[i, j, k]
            l12 = duxdy[i, j, k]
            l13 = duxdz[i, j, k]
            l21 = duydx[i, j, k]
            l22 = duydy[i, j, k]
            l23 = duydz[i, j, k]
            l31 = duzdx[i, j, k]
            l32 = duzdy[i, j, k]
            l33 = duzdz[i, j, k]
            h = T(dt) / T(n_sub)
            half_h = T(0.5) * h
            lambda_t = T(lambda)
            L2_t = T(L2_fene)
            divu_transport = zero(T)

            for _ in 1:n_sub
                k1xx = logconf_source_with_divergence_fenep_3d(
                    pxx, pxy, pxz, pyy, pyz, pzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, L2_t, 1,
                )
                k1xy = logconf_source_with_divergence_fenep_3d(
                    pxx, pxy, pxz, pyy, pyz, pzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, L2_t, 2,
                )
                k1xz = logconf_source_with_divergence_fenep_3d(
                    pxx, pxy, pxz, pyy, pyz, pzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, L2_t, 3,
                )
                k1yy = logconf_source_with_divergence_fenep_3d(
                    pxx, pxy, pxz, pyy, pyz, pzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, L2_t, 4,
                )
                k1yz = logconf_source_with_divergence_fenep_3d(
                    pxx, pxy, pxz, pyy, pyz, pzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, L2_t, 5,
                )
                k1zz = logconf_source_with_divergence_fenep_3d(
                    pxx, pxy, pxz, pyy, pyz, pzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, L2_t, 6,
                )

                qxx = pxx + h * k1xx
                qxy = pxy + h * k1xy
                qxz = pxz + h * k1xz
                qyy = pyy + h * k1yy
                qyz = pyz + h * k1yz
                qzz = pzz + h * k1zz

                k2xx = logconf_source_with_divergence_fenep_3d(
                    qxx, qxy, qxz, qyy, qyz, qzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, L2_t, 1,
                )
                k2xy = logconf_source_with_divergence_fenep_3d(
                    qxx, qxy, qxz, qyy, qyz, qzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, L2_t, 2,
                )
                k2xz = logconf_source_with_divergence_fenep_3d(
                    qxx, qxy, qxz, qyy, qyz, qzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, L2_t, 3,
                )
                k2yy = logconf_source_with_divergence_fenep_3d(
                    qxx, qxy, qxz, qyy, qyz, qzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, L2_t, 4,
                )
                k2yz = logconf_source_with_divergence_fenep_3d(
                    qxx, qxy, qxz, qyy, qyz, qzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, L2_t, 5,
                )
                k2zz = logconf_source_with_divergence_fenep_3d(
                    qxx, qxy, qxz, qyy, qyz, qzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, L2_t, 6,
                )

                pxx += half_h * (k1xx + k2xx)
                pxy += half_h * (k1xy + k2xy)
                pxz += half_h * (k1xz + k2xz)
                pyy += half_h * (k1yy + k2yy)
                pyz += half_h * (k1yz + k2yz)
                pzz += half_h * (k1zz + k2zz)
            end

            psixx_out[i, j, k] = pxx
            psixy_out[i, j, k] = pxy
            psixz_out[i, j, k] = pxz
            psiyy_out[i, j, k] = pyy
            psiyz_out[i, j, k] = pyz
            psizz_out[i, j, k] = pzz
        end
    end
end

function logfv_constitutive_step_log_fenep_3d!(
    psixx_out, psixy_out, psixz_out, psiyy_out, psiyz_out, psizz_out,
    psixx, psixy, psixz, psiyy, psiyz, psizz,
    duxdx, duxdy, duxdz,
    duydx, duydy, duydz,
    duzdx, duzdy, duzdz,
    lambda, dt, L2_fene, n_sub;
    sync::Bool=true,
)
    n_sub >= 1 || throw(ArgumentError("n_sub must be >= 1"))
    backend = KernelAbstractions.get_backend(psixx_out)
    Nx, Ny, Nz = size(psixx_out)
    kernel! = logfv_constitutive_step_log_fenep_3d_kernel!(backend)
    kernel!(
        psixx_out, psixy_out, psixz_out, psiyy_out, psiyz_out, psizz_out,
        psixx, psixy, psixz, psiyy, psiyz, psizz,
        duxdx, duxdy, duxdz,
        duydx, duydy, duydz,
        duzdx, duzdy, duzdz,
        lambda, dt, L2_fene, Int(n_sub), Nx, Ny, Nz;
        ndrange=(Nx, Ny, Nz),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function logfv_constitutive_step_log_fenep_3d!(
    psixx, psixy, psixz, psiyy, psiyz, psizz,
    duxdx, duxdy, duxdz,
    duydx, duydy, duydz,
    duzdx, duzdy, duzdz,
    lambda, dt, L2_fene, n_sub;
    sync::Bool=true,
)
    return logfv_constitutive_step_log_fenep_3d!(
        psixx, psixy, psixz, psiyy, psiyz, psizz,
        psixx, psixy, psixz, psiyy, psiyz, psizz,
        duxdx, duxdy, duxdz,
        duydx, duydy, duydz,
        duzdx, duzdy, duzdz,
        lambda, dt, L2_fene, n_sub;
        sync,
    )
end

# ---------------------------------------------------------------------
# Giesekus variant of the unsplit RK2 constitutive step.
#
# Structurally identical to `logfv_constitutive_step_log_fenep_3d_kernel!`,
# but integrates `logconf_source_with_divergence_giesekus_3d` (mobility
# scalar `alpha` in place of the Peterlin `L2_fene`). With `alpha == 0`
# the Giesekus quadratic factor is exactly 1, so this kernel reproduces
# the Oldroyd-B trajectory bit-for-bit; the dedicated OB kernel above is
# left untouched so the OB path never changes.
# ---------------------------------------------------------------------
@kernel function logfv_constitutive_step_log_giesekus_3d_kernel!(
    psixx_out, psixy_out, psixz_out, psiyy_out, psiyz_out, psizz_out,
    @Const(psixx), @Const(psixy), @Const(psixz),
    @Const(psiyy), @Const(psiyz), @Const(psizz),
    @Const(duxdx), @Const(duxdy), @Const(duxdz),
    @Const(duydx), @Const(duydy), @Const(duydz),
    @Const(duzdx), @Const(duzdy), @Const(duzdz),
    lambda, dt, alpha, n_sub, Nx, Ny, Nz,
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny && k <= Nz
            T = eltype(psixx_out)
            pxx = psixx[i, j, k]
            pxy = psixy[i, j, k]
            pxz = psixz[i, j, k]
            pyy = psiyy[i, j, k]
            pyz = psiyz[i, j, k]
            pzz = psizz[i, j, k]
            l11 = duxdx[i, j, k]
            l12 = duxdy[i, j, k]
            l13 = duxdz[i, j, k]
            l21 = duydx[i, j, k]
            l22 = duydy[i, j, k]
            l23 = duydz[i, j, k]
            l31 = duzdx[i, j, k]
            l32 = duzdy[i, j, k]
            l33 = duzdz[i, j, k]
            h = T(dt) / T(n_sub)
            half_h = T(0.5) * h
            lambda_t = T(lambda)
            alpha_t = T(alpha)
            divu_transport = zero(T)

            for _ in 1:n_sub
                k1xx = logconf_source_with_divergence_giesekus_3d(
                    pxx, pxy, pxz, pyy, pyz, pzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, alpha_t, 1,
                )
                k1xy = logconf_source_with_divergence_giesekus_3d(
                    pxx, pxy, pxz, pyy, pyz, pzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, alpha_t, 2,
                )
                k1xz = logconf_source_with_divergence_giesekus_3d(
                    pxx, pxy, pxz, pyy, pyz, pzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, alpha_t, 3,
                )
                k1yy = logconf_source_with_divergence_giesekus_3d(
                    pxx, pxy, pxz, pyy, pyz, pzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, alpha_t, 4,
                )
                k1yz = logconf_source_with_divergence_giesekus_3d(
                    pxx, pxy, pxz, pyy, pyz, pzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, alpha_t, 5,
                )
                k1zz = logconf_source_with_divergence_giesekus_3d(
                    pxx, pxy, pxz, pyy, pyz, pzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, alpha_t, 6,
                )

                qxx = pxx + h * k1xx
                qxy = pxy + h * k1xy
                qxz = pxz + h * k1xz
                qyy = pyy + h * k1yy
                qyz = pyz + h * k1yz
                qzz = pzz + h * k1zz

                k2xx = logconf_source_with_divergence_giesekus_3d(
                    qxx, qxy, qxz, qyy, qyz, qzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, alpha_t, 1,
                )
                k2xy = logconf_source_with_divergence_giesekus_3d(
                    qxx, qxy, qxz, qyy, qyz, qzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, alpha_t, 2,
                )
                k2xz = logconf_source_with_divergence_giesekus_3d(
                    qxx, qxy, qxz, qyy, qyz, qzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, alpha_t, 3,
                )
                k2yy = logconf_source_with_divergence_giesekus_3d(
                    qxx, qxy, qxz, qyy, qyz, qzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, alpha_t, 4,
                )
                k2yz = logconf_source_with_divergence_giesekus_3d(
                    qxx, qxy, qxz, qyy, qyz, qzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, alpha_t, 5,
                )
                k2zz = logconf_source_with_divergence_giesekus_3d(
                    qxx, qxy, qxz, qyy, qyz, qzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, alpha_t, 6,
                )

                pxx += half_h * (k1xx + k2xx)
                pxy += half_h * (k1xy + k2xy)
                pxz += half_h * (k1xz + k2xz)
                pyy += half_h * (k1yy + k2yy)
                pyz += half_h * (k1yz + k2yz)
                pzz += half_h * (k1zz + k2zz)
            end

            psixx_out[i, j, k] = pxx
            psixy_out[i, j, k] = pxy
            psixz_out[i, j, k] = pxz
            psiyy_out[i, j, k] = pyy
            psiyz_out[i, j, k] = pyz
            psizz_out[i, j, k] = pzz
        end
    end
end

function logfv_constitutive_step_log_giesekus_3d!(
    psixx_out, psixy_out, psixz_out, psiyy_out, psiyz_out, psizz_out,
    psixx, psixy, psixz, psiyy, psiyz, psizz,
    duxdx, duxdy, duxdz,
    duydx, duydy, duydz,
    duzdx, duzdy, duzdz,
    lambda, dt, alpha, n_sub;
    sync::Bool=true,
)
    n_sub >= 1 || throw(ArgumentError("n_sub must be >= 1"))
    backend = KernelAbstractions.get_backend(psixx_out)
    Nx, Ny, Nz = size(psixx_out)
    kernel! = logfv_constitutive_step_log_giesekus_3d_kernel!(backend)
    kernel!(
        psixx_out, psixy_out, psixz_out, psiyy_out, psiyz_out, psizz_out,
        psixx, psixy, psixz, psiyy, psiyz, psizz,
        duxdx, duxdy, duxdz,
        duydx, duydy, duydz,
        duzdx, duzdy, duzdz,
        lambda, dt, alpha, Int(n_sub), Nx, Ny, Nz;
        ndrange=(Nx, Ny, Nz),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function logfv_constitutive_step_log_giesekus_3d!(
    psixx, psixy, psixz, psiyy, psiyz, psizz,
    duxdx, duxdy, duxdz,
    duydx, duydy, duydz,
    duzdx, duzdy, duzdz,
    lambda, dt, alpha, n_sub;
    sync::Bool=true,
)
    return logfv_constitutive_step_log_giesekus_3d!(
        psixx, psixy, psixz, psiyy, psiyz, psizz,
        psixx, psixy, psixz, psiyy, psiyz, psizz,
        duxdx, duxdy, duxdz,
        duydx, duydy, duydz,
        duzdx, duzdy, duzdz,
        lambda, dt, alpha, n_sub;
        sync,
    )
end

# ---------------------------------------------------------------------
# Phan-Thien–Tanner (PTT) variant of the unsplit RK2 constitutive step.
#
# Structurally identical to `logfv_constitutive_step_log_giesekus_3d_kernel!`,
# but integrates `logconf_source_with_divergence_ptt_3d` (extensibility
# scalar `epsilon` + integer `variant` selecting the linear vs exponential
# trace multiplier in place of the Giesekus mobility `alpha`). With
# `epsilon == 0` the PTT trace multiplier Y(trC) is exactly 1, so this
# kernel reproduces the Oldroyd-B trajectory bit-for-bit; the dedicated OB
# kernel above is left untouched so the OB path never changes.
# ---------------------------------------------------------------------
@kernel function logfv_constitutive_step_log_ptt_3d_kernel!(
    psixx_out, psixy_out, psixz_out, psiyy_out, psiyz_out, psizz_out,
    @Const(psixx), @Const(psixy), @Const(psixz),
    @Const(psiyy), @Const(psiyz), @Const(psizz),
    @Const(duxdx), @Const(duxdy), @Const(duxdz),
    @Const(duydx), @Const(duydy), @Const(duydz),
    @Const(duzdx), @Const(duzdy), @Const(duzdz),
    lambda, dt, epsilon, variant, n_sub, Nx, Ny, Nz,
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny && k <= Nz
            T = eltype(psixx_out)
            pxx = psixx[i, j, k]
            pxy = psixy[i, j, k]
            pxz = psixz[i, j, k]
            pyy = psiyy[i, j, k]
            pyz = psiyz[i, j, k]
            pzz = psizz[i, j, k]
            l11 = duxdx[i, j, k]
            l12 = duxdy[i, j, k]
            l13 = duxdz[i, j, k]
            l21 = duydx[i, j, k]
            l22 = duydy[i, j, k]
            l23 = duydz[i, j, k]
            l31 = duzdx[i, j, k]
            l32 = duzdy[i, j, k]
            l33 = duzdz[i, j, k]
            h = T(dt) / T(n_sub)
            half_h = T(0.5) * h
            lambda_t = T(lambda)
            eps_t = T(epsilon)
            variant_i = Int(variant)
            divu_transport = zero(T)

            for _ in 1:n_sub
                k1xx = logconf_source_with_divergence_ptt_3d(
                    pxx, pxy, pxz, pyy, pyz, pzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, eps_t, variant_i, 1,
                )
                k1xy = logconf_source_with_divergence_ptt_3d(
                    pxx, pxy, pxz, pyy, pyz, pzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, eps_t, variant_i, 2,
                )
                k1xz = logconf_source_with_divergence_ptt_3d(
                    pxx, pxy, pxz, pyy, pyz, pzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, eps_t, variant_i, 3,
                )
                k1yy = logconf_source_with_divergence_ptt_3d(
                    pxx, pxy, pxz, pyy, pyz, pzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, eps_t, variant_i, 4,
                )
                k1yz = logconf_source_with_divergence_ptt_3d(
                    pxx, pxy, pxz, pyy, pyz, pzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, eps_t, variant_i, 5,
                )
                k1zz = logconf_source_with_divergence_ptt_3d(
                    pxx, pxy, pxz, pyy, pyz, pzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, eps_t, variant_i, 6,
                )

                qxx = pxx + h * k1xx
                qxy = pxy + h * k1xy
                qxz = pxz + h * k1xz
                qyy = pyy + h * k1yy
                qyz = pyz + h * k1yz
                qzz = pzz + h * k1zz

                k2xx = logconf_source_with_divergence_ptt_3d(
                    qxx, qxy, qxz, qyy, qyz, qzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, eps_t, variant_i, 1,
                )
                k2xy = logconf_source_with_divergence_ptt_3d(
                    qxx, qxy, qxz, qyy, qyz, qzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, eps_t, variant_i, 2,
                )
                k2xz = logconf_source_with_divergence_ptt_3d(
                    qxx, qxy, qxz, qyy, qyz, qzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, eps_t, variant_i, 3,
                )
                k2yy = logconf_source_with_divergence_ptt_3d(
                    qxx, qxy, qxz, qyy, qyz, qzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, eps_t, variant_i, 4,
                )
                k2yz = logconf_source_with_divergence_ptt_3d(
                    qxx, qxy, qxz, qyy, qyz, qzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, eps_t, variant_i, 5,
                )
                k2zz = logconf_source_with_divergence_ptt_3d(
                    qxx, qxy, qxz, qyy, qyz, qzz,
                    l11, l12, l13, l21, l22, l23, l31, l32, l33,
                    divu_transport, lambda_t, eps_t, variant_i, 6,
                )

                pxx += half_h * (k1xx + k2xx)
                pxy += half_h * (k1xy + k2xy)
                pxz += half_h * (k1xz + k2xz)
                pyy += half_h * (k1yy + k2yy)
                pyz += half_h * (k1yz + k2yz)
                pzz += half_h * (k1zz + k2zz)
            end

            psixx_out[i, j, k] = pxx
            psixy_out[i, j, k] = pxy
            psixz_out[i, j, k] = pxz
            psiyy_out[i, j, k] = pyy
            psiyz_out[i, j, k] = pyz
            psizz_out[i, j, k] = pzz
        end
    end
end

function logfv_constitutive_step_log_ptt_3d!(
    psixx_out, psixy_out, psixz_out, psiyy_out, psiyz_out, psizz_out,
    psixx, psixy, psixz, psiyy, psiyz, psizz,
    duxdx, duxdy, duxdz,
    duydx, duydy, duydz,
    duzdx, duzdy, duzdz,
    lambda, dt, epsilon, n_sub;
    variant::Symbol=:linear,
    sync::Bool=true,
)
    n_sub >= 1 || throw(ArgumentError("n_sub must be >= 1"))
    (variant === :linear || variant === :exponential) ||
        throw(ArgumentError("PTT variant must be :linear or :exponential"))
    variant_code = variant === :exponential ? 2 : 1
    backend = KernelAbstractions.get_backend(psixx_out)
    Nx, Ny, Nz = size(psixx_out)
    kernel! = logfv_constitutive_step_log_ptt_3d_kernel!(backend)
    kernel!(
        psixx_out, psixy_out, psixz_out, psiyy_out, psiyz_out, psizz_out,
        psixx, psixy, psixz, psiyy, psiyz, psizz,
        duxdx, duxdy, duxdz,
        duydx, duydy, duydz,
        duzdx, duzdy, duzdz,
        lambda, dt, epsilon, variant_code, Int(n_sub), Nx, Ny, Nz;
        ndrange=(Nx, Ny, Nz),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function logfv_constitutive_step_log_ptt_3d!(
    psixx, psixy, psixz, psiyy, psiyz, psizz,
    duxdx, duxdy, duxdz,
    duydx, duydy, duydz,
    duzdx, duzdy, duzdz,
    lambda, dt, epsilon, n_sub;
    variant::Symbol=:linear,
    sync::Bool=true,
)
    return logfv_constitutive_step_log_ptt_3d!(
        psixx, psixy, psixz, psiyy, psiyz, psizz,
        psixx, psixy, psixz, psiyy, psiyz, psizz,
        duxdx, duxdy, duxdz,
        duydx, duydy, duydz,
        duzdx, duzdy, duzdz,
        lambda, dt, epsilon, n_sub;
        variant, sync,
    )
end

function logfv_constitutive_step_log_3d!(
    psixx_out, psixy_out, psixz_out, psiyy_out, psiyz_out, psizz_out,
    psixx, psixy, psixz, psiyy, psiyz, psizz,
    duxdx, duxdy, duxdz,
    duydx, duydy, duydz,
    duzdx, duzdy, duzdz,
    lambda, dt, n_sub;
    sync::Bool=true,
)
    n_sub >= 1 || throw(ArgumentError("n_sub must be >= 1"))
    backend = KernelAbstractions.get_backend(psixx_out)
    Nx, Ny, Nz = size(psixx_out)
    kernel! = logfv_constitutive_step_log_3d_kernel!(backend)
    kernel!(
        psixx_out, psixy_out, psixz_out, psiyy_out, psiyz_out, psizz_out,
        psixx, psixy, psixz, psiyy, psiyz, psizz,
        duxdx, duxdy, duxdz,
        duydx, duydy, duydz,
        duzdx, duzdy, duzdz,
        lambda, dt, Int(n_sub), Nx, Ny, Nz;
        ndrange=(Nx, Ny, Nz),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function logfv_constitutive_step_log_3d!(
    psixx, psixy, psixz, psiyy, psiyz, psizz,
    duxdx, duxdy, duxdz,
    duydx, duydy, duydz,
    duzdx, duzdy, duzdz,
    lambda, dt, n_sub;
    sync::Bool=true,
)
    return logfv_constitutive_step_log_3d!(
        psixx, psixy, psixz, psiyy, psiyz, psizz,
        psixx, psixy, psixz, psiyy, psiyz, psizz,
        duxdx, duxdy, duxdz,
        duydx, duydy, duydz,
        duzdx, duzdy, duzdz,
        lambda, dt, n_sub;
        sync,
    )
end

# ---------------------------------------------------------------------
# Shared constitutive-step dispatch (DRY): pick the per-model RK2 step
# from the polymer `model`, forwarding the same ψ_out/ψ_in/∇u arg list to
# every variant. This is the single point both FVFD drivers (Poiseuille +
# extensional) call, so adding a constitutive model means a new branch
# HERE only — the drivers never duplicate the if/elseif.
#
# Bit-identity guarantees:
#   * LogConfOldroydB         → the dedicated OB kernel (untouched).
#   * LogConfFENEP            → the FENE-P kernel with L²=polymer_max_extensibility
#                               (identical to the prior `isfinite(L2)` branch).
#   * LogConfGiesekus(α=0)    → its kernel reproduces the OB trajectory.
#   * LogConfPTT(ε=0)         → its kernel reproduces the OB trajectory.
# ---------------------------------------------------------------------
function logfv_constitutive_step_dispatch_3d!(
    model::AbstractPolymerModel,
    psixx_out, psixy_out, psixz_out, psiyy_out, psiyz_out, psizz_out,
    psixx, psixy, psixz, psiyy, psiyz, psizz,
    duxdx, duxdy, duxdz,
    duydx, duydy, duydz,
    duzdx, duzdy, duzdz,
    lambda, dt, n_sub;
    sync::Bool=true,
)
    if model isa LogConfFENEP
        logfv_constitutive_step_log_fenep_3d!(
            psixx_out, psixy_out, psixz_out, psiyy_out, psiyz_out, psizz_out,
            psixx, psixy, psixz, psiyy, psiyz, psizz,
            duxdx, duxdy, duxdz,
            duydx, duydy, duydz,
            duzdx, duzdy, duzdz,
            lambda, dt, oftype(lambda, polymer_max_extensibility(model)), n_sub;
            sync,
        )
    elseif model isa LogConfGiesekus
        logfv_constitutive_step_log_giesekus_3d!(
            psixx_out, psixy_out, psixz_out, psiyy_out, psiyz_out, psizz_out,
            psixx, psixy, psixz, psiyy, psiyz, psizz,
            duxdx, duxdy, duxdz,
            duydx, duydy, duydz,
            duzdx, duzdy, duzdz,
            lambda, dt, oftype(lambda, polymer_mobility(model)), n_sub;
            sync,
        )
    elseif model isa LogConfPTT
        logfv_constitutive_step_log_ptt_3d!(
            psixx_out, psixy_out, psixz_out, psiyy_out, psiyz_out, psizz_out,
            psixx, psixy, psixz, psiyy, psiyz, psizz,
            duxdx, duxdy, duxdz,
            duydx, duydy, duydz,
            duzdx, duzdy, duzdz,
            lambda, dt, oftype(lambda, polymer_ptt_epsilon(model)), n_sub;
            variant=polymer_ptt_variant(model), sync,
        )
    else
        logfv_constitutive_step_log_3d!(
            psixx_out, psixy_out, psixz_out, psiyy_out, psiyz_out, psizz_out,
            psixx, psixy, psixz, psiyy, psiyz, psizz,
            duxdx, duxdy, duxdz,
            duydx, duydy, duydz,
            duzdx, duzdy, duzdz,
            lambda, dt, n_sub;
            sync,
        )
    end
    return nothing
end
