using KernelAbstractions

# --- 3D Refinement exchange kernels (D3Q19, trilinear + Filippova-Hanel) ---

# Inline D3Q19 equilibrium for GPU kernels (Metal-compatible, no Float64 constants)
@inline function _feq_3d(q::Int, ρ, ux, uy, uz)
    TT = typeof(ρ)
    # D3Q19 weights: rest=1/3, axis(2-7)=1/18, edge(8-19)=1/36
    w = ifelse(q == 1, one(TT) / TT(3),
        ifelse(q <= 7, one(TT) / TT(18), one(TT) / TT(36)))
    # D3Q19 velocities (cx, cy, cz) — avoid Float64 lookup
    cx = TT(ifelse(q==2, 1, ifelse(q==3, -1,
         ifelse(q==8, 1, ifelse(q==9, -1, ifelse(q==10, 1, ifelse(q==11, -1,
         ifelse(q==12, 1, ifelse(q==13, -1, ifelse(q==14, 1, ifelse(q==15, -1, 0)))))))))))
    cy = TT(ifelse(q==4, 1, ifelse(q==5, -1,
         ifelse(q==8, 1, ifelse(q==9, 1, ifelse(q==10, -1, ifelse(q==11, -1,
         ifelse(q==16, 1, ifelse(q==17, -1, ifelse(q==18, 1, ifelse(q==19, -1, 0)))))))))))
    cz = TT(ifelse(q==6, 1, ifelse(q==7, -1,
         ifelse(q==12, 1, ifelse(q==13, 1, ifelse(q==14, -1, ifelse(q==15, -1,
         ifelse(q==16, 1, ifelse(q==17, 1, ifelse(q==18, -1, ifelse(q==19, -1, 0)))))))))))
    cu = cx * ux + cy * uy + cz * uz
    usq = ux * ux + uy * uy + uz * uz
    return w * ρ * (one(TT) + TT(3) * cu + TT(4.5) * cu * cu - TT(1.5) * usq)
end

# =====================================================================
# Prolongation: coarse → fine ghost fill with Filippova-Hanel (D3Q19)
# =====================================================================

@kernel function prolongate_f_rescaled_3d_kernel!(
        f_fine, @Const(f_coarse), @Const(rho_c), @Const(ux_c), @Const(uy_c), @Const(uz_c),
        ratio, Nx_inner, Ny_inner, Nz_inner, n_ghost,
        i_c_start, j_c_start, k_c_start,
        Nx_c, Ny_c, Nz_c, omega_c, omega_f)

    i_f, j_f, k_f = @index(Global, NTuple)

    @inbounds begin
        is_ghost = (i_f <= n_ghost) || (i_f > n_ghost + Nx_inner) ||
                   (j_f <= n_ghost) || (j_f > n_ghost + Ny_inner) ||
                   (k_f <= n_ghost) || (k_f > n_ghost + Nz_inner)
        if is_ghost
            T = eltype(f_fine)
            alpha = (T(omega_f) - T(0.5)) * (one(T) / (T(omega_c) - T(0.5)))

            # Fine cell center → coarse continuous coords
            xc = T(i_f - n_ghost - 1) / T(ratio) + T(i_c_start) + T(0.5) / T(ratio)
            yc = T(j_f - n_ghost - 1) / T(ratio) + T(j_c_start) + T(0.5) / T(ratio)
            zc = T(k_f - n_ghost - 1) / T(ratio) + T(k_c_start) + T(0.5) / T(ratio)

            # Trilinear stencil
            i0r = unsafe_trunc(Int, xc); j0r = unsafe_trunc(Int, yc); k0r = unsafe_trunc(Int, zc)
            tx = xc - T(i0r); ty = yc - T(j0r); tz = zc - T(k0r)
            i0 = clamp(i0r, 1, Nx_c); i1 = clamp(i0r+1, 1, Nx_c)
            j0 = clamp(j0r, 1, Ny_c); j1 = clamp(j0r+1, 1, Ny_c)
            k0 = clamp(k0r, 1, Nz_c); k1 = clamp(k0r+1, 1, Nz_c)
            tx = clamp(tx, zero(T), one(T))
            ty = clamp(ty, zero(T), one(T))
            tz = clamp(tz, zero(T), one(T))

            w000=(one(T)-tx)*(one(T)-ty)*(one(T)-tz)
            w100=tx*(one(T)-ty)*(one(T)-tz)
            w010=(one(T)-tx)*ty*(one(T)-tz)
            w110=tx*ty*(one(T)-tz)
            w001=(one(T)-tx)*(one(T)-ty)*tz
            w101=tx*(one(T)-ty)*tz
            w011=(one(T)-tx)*ty*tz
            w111=tx*ty*tz

            # Interpolate macroscopic
            rho_f = w000*rho_c[i0,j0,k0]+w100*rho_c[i1,j0,k0]+w010*rho_c[i0,j1,k0]+w110*rho_c[i1,j1,k0]+
                    w001*rho_c[i0,j0,k1]+w101*rho_c[i1,j0,k1]+w011*rho_c[i0,j1,k1]+w111*rho_c[i1,j1,k1]
            ux_f = w000*ux_c[i0,j0,k0]+w100*ux_c[i1,j0,k0]+w010*ux_c[i0,j1,k0]+w110*ux_c[i1,j1,k0]+
                   w001*ux_c[i0,j0,k1]+w101*ux_c[i1,j0,k1]+w011*ux_c[i0,j1,k1]+w111*ux_c[i1,j1,k1]
            uy_f = w000*uy_c[i0,j0,k0]+w100*uy_c[i1,j0,k0]+w010*uy_c[i0,j1,k0]+w110*uy_c[i1,j1,k0]+
                   w001*uy_c[i0,j0,k1]+w101*uy_c[i1,j0,k1]+w011*uy_c[i0,j1,k1]+w111*uy_c[i1,j1,k1]
            uz_f = w000*uz_c[i0,j0,k0]+w100*uz_c[i1,j0,k0]+w010*uz_c[i0,j1,k0]+w110*uz_c[i1,j1,k0]+
                   w001*uz_c[i0,j0,k1]+w101*uz_c[i1,j0,k1]+w011*uz_c[i0,j1,k1]+w111*uz_c[i1,j1,k1]

            for q in 1:19
                feq_fine = _feq_3d(q, rho_f, ux_f, uy_f, uz_f)

                fneq = w000*(f_coarse[i0,j0,k0,q]-_feq_3d(q,rho_c[i0,j0,k0],ux_c[i0,j0,k0],uy_c[i0,j0,k0],uz_c[i0,j0,k0]))+
                       w100*(f_coarse[i1,j0,k0,q]-_feq_3d(q,rho_c[i1,j0,k0],ux_c[i1,j0,k0],uy_c[i1,j0,k0],uz_c[i1,j0,k0]))+
                       w010*(f_coarse[i0,j1,k0,q]-_feq_3d(q,rho_c[i0,j1,k0],ux_c[i0,j1,k0],uy_c[i0,j1,k0],uz_c[i0,j1,k0]))+
                       w110*(f_coarse[i1,j1,k0,q]-_feq_3d(q,rho_c[i1,j1,k0],ux_c[i1,j1,k0],uy_c[i1,j1,k0],uz_c[i1,j1,k0]))+
                       w001*(f_coarse[i0,j0,k1,q]-_feq_3d(q,rho_c[i0,j0,k1],ux_c[i0,j0,k1],uy_c[i0,j0,k1],uz_c[i0,j0,k1]))+
                       w101*(f_coarse[i1,j0,k1,q]-_feq_3d(q,rho_c[i1,j0,k1],ux_c[i1,j0,k1],uy_c[i1,j0,k1],uz_c[i1,j0,k1]))+
                       w011*(f_coarse[i0,j1,k1,q]-_feq_3d(q,rho_c[i0,j1,k1],ux_c[i0,j1,k1],uy_c[i0,j1,k1],uz_c[i0,j1,k1]))+
                       w111*(f_coarse[i1,j1,k1,q]-_feq_3d(q,rho_c[i1,j1,k1],ux_c[i1,j1,k1],uy_c[i1,j1,k1],uz_c[i1,j1,k1]))

                f_fine[i_f, j_f, k_f, q] = feq_fine + alpha * fneq
            end
        end
    end
end

"""
    prolongate_f_rescaled_3d!(f_fine, f_coarse, rho_c, ux_c, uy_c, uz_c,

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.prolongate_f_rescaled_3d!)
```
"""
function prolongate_f_rescaled_3d!(f_fine, f_coarse, rho_c, ux_c, uy_c, uz_c,
                                    ratio, Nx_inner, Ny_inner, Nz_inner, n_ghost,
                                    i_c_start, j_c_start, k_c_start,
                                    Nx_c, Ny_c, Nz_c, omega_c, omega_f)
    backend = KernelAbstractions.get_backend(f_fine)
    Nx_f = Nx_inner + 2 * n_ghost
    Ny_f = Ny_inner + 2 * n_ghost
    Nz_f = Nz_inner + 2 * n_ghost
    kernel! = prolongate_f_rescaled_3d_kernel!(backend)
    FT = eltype(f_fine)
    kernel!(f_fine, f_coarse, rho_c, ux_c, uy_c, uz_c,
            ratio, Nx_inner, Ny_inner, Nz_inner, n_ghost,
            i_c_start, j_c_start, k_c_start,
            Nx_c, Ny_c, Nz_c, FT(omega_c), FT(omega_f);
            ndrange=(Nx_f, Ny_f, Nz_f))
end

# =====================================================================
# Full prolongation (init time — operates on ALL cells, not just ghosts)
# =====================================================================

@kernel function prolongate_f_rescaled_full_3d_kernel!(
        f_fine, @Const(f_coarse), @Const(rho_c), @Const(ux_c), @Const(uy_c), @Const(uz_c),
        ratio, Nx_inner, Ny_inner, Nz_inner, n_ghost,
        i_c_start, j_c_start, k_c_start,
        Nx_c, Ny_c, Nz_c, omega_c, omega_f)

    i_f, j_f, k_f = @index(Global, NTuple)

    @inbounds begin
        T = eltype(f_fine)
        alpha = (T(omega_f) - T(0.5)) * (one(T) / (T(omega_c) - T(0.5)))

        xc = T(i_f - n_ghost - 1) / T(ratio) + T(i_c_start) + T(0.5) / T(ratio)
        yc = T(j_f - n_ghost - 1) / T(ratio) + T(j_c_start) + T(0.5) / T(ratio)
        zc = T(k_f - n_ghost - 1) / T(ratio) + T(k_c_start) + T(0.5) / T(ratio)

        i0r = unsafe_trunc(Int, xc); j0r = unsafe_trunc(Int, yc); k0r = unsafe_trunc(Int, zc)
        tx = clamp(xc - T(i0r), zero(T), one(T))
        ty = clamp(yc - T(j0r), zero(T), one(T))
        tz = clamp(zc - T(k0r), zero(T), one(T))
        i0 = clamp(i0r, 1, Nx_c); i1 = clamp(i0r+1, 1, Nx_c)
        j0 = clamp(j0r, 1, Ny_c); j1 = clamp(j0r+1, 1, Ny_c)
        k0 = clamp(k0r, 1, Nz_c); k1 = clamp(k0r+1, 1, Nz_c)

        w000=(one(T)-tx)*(one(T)-ty)*(one(T)-tz); w100=tx*(one(T)-ty)*(one(T)-tz)
        w010=(one(T)-tx)*ty*(one(T)-tz);           w110=tx*ty*(one(T)-tz)
        w001=(one(T)-tx)*(one(T)-ty)*tz;           w101=tx*(one(T)-ty)*tz
        w011=(one(T)-tx)*ty*tz;                     w111=tx*ty*tz

        rho_f = w000*rho_c[i0,j0,k0]+w100*rho_c[i1,j0,k0]+w010*rho_c[i0,j1,k0]+w110*rho_c[i1,j1,k0]+
                w001*rho_c[i0,j0,k1]+w101*rho_c[i1,j0,k1]+w011*rho_c[i0,j1,k1]+w111*rho_c[i1,j1,k1]
        ux_f = w000*ux_c[i0,j0,k0]+w100*ux_c[i1,j0,k0]+w010*ux_c[i0,j1,k0]+w110*ux_c[i1,j1,k0]+
               w001*ux_c[i0,j0,k1]+w101*ux_c[i1,j0,k1]+w011*ux_c[i0,j1,k1]+w111*ux_c[i1,j1,k1]
        uy_f = w000*uy_c[i0,j0,k0]+w100*uy_c[i1,j0,k0]+w010*uy_c[i0,j1,k0]+w110*uy_c[i1,j1,k0]+
               w001*uy_c[i0,j0,k1]+w101*uy_c[i1,j0,k1]+w011*uy_c[i0,j1,k1]+w111*uy_c[i1,j1,k1]
        uz_f = w000*uz_c[i0,j0,k0]+w100*uz_c[i1,j0,k0]+w010*uz_c[i0,j1,k0]+w110*uz_c[i1,j1,k0]+
               w001*uz_c[i0,j0,k1]+w101*uz_c[i1,j0,k1]+w011*uz_c[i0,j1,k1]+w111*uz_c[i1,j1,k1]

        for q in 1:19
            feq_fine = _feq_3d(q, rho_f, ux_f, uy_f, uz_f)
            fneq = w000*(f_coarse[i0,j0,k0,q]-_feq_3d(q,rho_c[i0,j0,k0],ux_c[i0,j0,k0],uy_c[i0,j0,k0],uz_c[i0,j0,k0]))+
                   w100*(f_coarse[i1,j0,k0,q]-_feq_3d(q,rho_c[i1,j0,k0],ux_c[i1,j0,k0],uy_c[i1,j0,k0],uz_c[i1,j0,k0]))+
                   w010*(f_coarse[i0,j1,k0,q]-_feq_3d(q,rho_c[i0,j1,k0],ux_c[i0,j1,k0],uy_c[i0,j1,k0],uz_c[i0,j1,k0]))+
                   w110*(f_coarse[i1,j1,k0,q]-_feq_3d(q,rho_c[i1,j1,k0],ux_c[i1,j1,k0],uy_c[i1,j1,k0],uz_c[i1,j1,k0]))+
                   w001*(f_coarse[i0,j0,k1,q]-_feq_3d(q,rho_c[i0,j0,k1],ux_c[i0,j0,k1],uy_c[i0,j0,k1],uz_c[i0,j0,k1]))+
                   w101*(f_coarse[i1,j0,k1,q]-_feq_3d(q,rho_c[i1,j0,k1],ux_c[i1,j0,k1],uy_c[i1,j0,k1],uz_c[i1,j0,k1]))+
                   w011*(f_coarse[i0,j1,k1,q]-_feq_3d(q,rho_c[i0,j1,k1],ux_c[i0,j1,k1],uy_c[i0,j1,k1],uz_c[i0,j1,k1]))+
                   w111*(f_coarse[i1,j1,k1,q]-_feq_3d(q,rho_c[i1,j1,k1],ux_c[i1,j1,k1],uy_c[i1,j1,k1],uz_c[i1,j1,k1]))
            f_fine[i_f, j_f, k_f, q] = feq_fine + alpha * fneq
        end
    end
end

"""
    prolongate_f_rescaled_full_3d!(f_fine, f_coarse, rho_c, ux_c, uy_c, uz_c,

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.prolongate_f_rescaled_full_3d!)
```
"""
function prolongate_f_rescaled_full_3d!(f_fine, f_coarse, rho_c, ux_c, uy_c, uz_c,
                                         ratio, Nx_inner, Ny_inner, Nz_inner, n_ghost,
                                         i_c_start, j_c_start, k_c_start,
                                         Nx_c, Ny_c, Nz_c, omega_c, omega_f)
    backend = KernelAbstractions.get_backend(f_fine)
    Nx_f = Nx_inner + 2 * n_ghost
    Ny_f = Ny_inner + 2 * n_ghost
    Nz_f = Nz_inner + 2 * n_ghost
    kernel! = prolongate_f_rescaled_full_3d_kernel!(backend)
    FT = eltype(f_fine)
    kernel!(f_fine, f_coarse, rho_c, ux_c, uy_c, uz_c,
            ratio, Nx_inner, Ny_inner, Nz_inner, n_ghost,
            i_c_start, j_c_start, k_c_start,
            Nx_c, Ny_c, Nz_c, FT(omega_c), FT(omega_f);
            ndrange=(Nx_f, Ny_f, Nz_f))
end

# =====================================================================
# Temporal ghost fill (sub-cycling interpolation between time n and n+1)
# =====================================================================

@kernel function prolongate_f_rescaled_temporal_3d_kernel!(
        f_fine,
        @Const(f_curr), @Const(rho_curr), @Const(ux_curr), @Const(uy_curr), @Const(uz_curr),
        @Const(f_prev), @Const(rho_prev), @Const(ux_prev), @Const(uy_prev), @Const(uz_prev),
        ratio, Nx_inner, Ny_inner, Nz_inner, n_ghost,
        i_c_start, j_c_start, k_c_start,
        Nx_c, Ny_c, Nz_c, omega_c, omega_f, t_frac,
        Ni_prev, Nj_prev, Nk_prev)

    i_f, j_f, k_f = @index(Global, NTuple)

    @inbounds begin
        is_ghost = (i_f <= n_ghost) || (i_f > n_ghost + Nx_inner) ||
                   (j_f <= n_ghost) || (j_f > n_ghost + Ny_inner) ||
                   (k_f <= n_ghost) || (k_f > n_ghost + Nz_inner)
        if is_ghost
            T = eltype(f_fine)
            alpha = (T(omega_f) - T(0.5)) * (one(T) / (T(omega_c) - T(0.5)))
            tf = T(t_frac)
            omtf = one(T) - tf

            xc = T(i_f - n_ghost - 1) / T(ratio) + T(i_c_start) + T(0.5) / T(ratio)
            yc = T(j_f - n_ghost - 1) / T(ratio) + T(j_c_start) + T(0.5) / T(ratio)
            zc = T(k_f - n_ghost - 1) / T(ratio) + T(k_c_start) + T(0.5) / T(ratio)

            i0r = unsafe_trunc(Int, xc); j0r = unsafe_trunc(Int, yc); k0r = unsafe_trunc(Int, zc)
            tx = clamp(xc - T(i0r), zero(T), one(T))
            ty = clamp(yc - T(j0r), zero(T), one(T))
            tz = clamp(zc - T(k0r), zero(T), one(T))

            # Coarse grid indices (current state at n+1)
            ic0 = clamp(i0r, 1, Nx_c); ic1 = clamp(i0r+1, 1, Nx_c)
            jc0 = clamp(j0r, 1, Ny_c); jc1 = clamp(j0r+1, 1, Ny_c)
            kc0 = clamp(k0r, 1, Nz_c); kc1 = clamp(k0r+1, 1, Nz_c)

            # Prev buffer indices (local, offset from i_lo = max(i_c_start-1, 1))
            # Clamp to Ni_prev (actual saved data size), NOT buffer size
            i_lo = max(i_c_start - 1, 1)
            j_lo = max(j_c_start - 1, 1)
            k_lo = max(k_c_start - 1, 1)
            ip0 = clamp(i0r - i_lo + 1, 1, Ni_prev); ip1 = clamp(i0r+1 - i_lo + 1, 1, Ni_prev)
            jp0 = clamp(j0r - j_lo + 1, 1, Nj_prev); jp1 = clamp(j0r+1 - j_lo + 1, 1, Nj_prev)
            kp0 = clamp(k0r - k_lo + 1, 1, Nk_prev); kp1 = clamp(k0r+1 - k_lo + 1, 1, Nk_prev)

            w000=(one(T)-tx)*(one(T)-ty)*(one(T)-tz); w100=tx*(one(T)-ty)*(one(T)-tz)
            w010=(one(T)-tx)*ty*(one(T)-tz);           w110=tx*ty*(one(T)-tz)
            w001=(one(T)-tx)*(one(T)-ty)*tz;           w101=tx*(one(T)-ty)*tz
            w011=(one(T)-tx)*ty*tz;                     w111=tx*ty*tz

            # Temporally interpolated macroscopic at each stencil node, then spatial interp
            rho_f = zero(T); uxf = zero(T); uyf = zero(T); uzf = zero(T)
            for (ws, ic, jc, kc, ipp, jpp, kpp) in (
                (w000, ic0, jc0, kc0, ip0, jp0, kp0), (w100, ic1, jc0, kc0, ip1, jp0, kp0),
                (w010, ic0, jc1, kc0, ip0, jp1, kp0), (w110, ic1, jc1, kc0, ip1, jp1, kp0),
                (w001, ic0, jc0, kc1, ip0, jp0, kp1), (w101, ic1, jc0, kc1, ip1, jp0, kp1),
                (w011, ic0, jc1, kc1, ip0, jp1, kp1), (w111, ic1, jc1, kc1, ip1, jp1, kp1))
                r = omtf * rho_prev[ipp, jpp, kpp] + tf * rho_curr[ic, jc, kc]
                u = omtf * ux_prev[ipp, jpp, kpp]  + tf * ux_curr[ic, jc, kc]
                v = omtf * uy_prev[ipp, jpp, kpp]  + tf * uy_curr[ic, jc, kc]
                w = omtf * uz_prev[ipp, jpp, kpp]  + tf * uz_curr[ic, jc, kc]
                rho_f += ws * r; uxf += ws * u; uyf += ws * v; uzf += ws * w
            end

            for q in 1:19
                feq_fine = _feq_3d(q, rho_f, uxf, uyf, uzf)

                fneq = zero(T)
                for (ws, ic, jc, kc, ipp, jpp, kpp) in (
                    (w000, ic0, jc0, kc0, ip0, jp0, kp0), (w100, ic1, jc0, kc0, ip1, jp0, kp0),
                    (w010, ic0, jc1, kc0, ip0, jp1, kp0), (w110, ic1, jc1, kc0, ip1, jp1, kp0),
                    (w001, ic0, jc0, kc1, ip0, jp0, kp1), (w101, ic1, jc0, kc1, ip1, jp0, kp1),
                    (w011, ic0, jc1, kc1, ip0, jp1, kp1), (w111, ic1, jc1, kc1, ip1, jp1, kp1))
                    r = omtf * rho_prev[ipp, jpp, kpp] + tf * rho_curr[ic, jc, kc]
                    u = omtf * ux_prev[ipp, jpp, kpp]  + tf * ux_curr[ic, jc, kc]
                    v = omtf * uy_prev[ipp, jpp, kpp]  + tf * uy_curr[ic, jc, kc]
                    ww = omtf * uz_prev[ipp, jpp, kpp] + tf * uz_curr[ic, jc, kc]
                    f_interp = omtf * f_prev[ipp, jpp, kpp, q] + tf * f_curr[ic, jc, kc, q]
                    fneq += ws * (f_interp - _feq_3d(q, r, u, v, ww))
                end

                f_fine[i_f, j_f, k_f, q] = feq_fine + alpha * fneq
            end
        end
    end
end

"""
    prolongate_f_rescaled_temporal_3d!(f_fine,

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.prolongate_f_rescaled_temporal_3d!)
```
"""
function prolongate_f_rescaled_temporal_3d!(f_fine,
        f_curr, rho_curr, ux_curr, uy_curr, uz_curr,
        f_prev, rho_prev, ux_prev, uy_prev, uz_prev,
        ratio, Nx_inner, Ny_inner, Nz_inner, n_ghost,
        i_c_start, j_c_start, k_c_start,
        Nx_c, Ny_c, Nz_c, omega_c, omega_f, t_frac)
    backend = KernelAbstractions.get_backend(f_fine)
    Nx_f = Nx_inner + 2 * n_ghost
    Ny_f = Ny_inner + 2 * n_ghost
    Nz_f = Nz_inner + 2 * n_ghost
    # Compute actual saved data size in prev buffers (matches save_coarse_state_3d!)
    i_lo = max(i_c_start - 1, 1)
    j_lo = max(j_c_start - 1, 1)
    k_lo = max(k_c_start - 1, 1)
    Ni_prev = min(i_c_start + Nx_inner ÷ ratio, Nx_c) - i_lo + 1
    Nj_prev = min(j_c_start + Ny_inner ÷ ratio, Ny_c) - j_lo + 1
    Nk_prev = min(k_c_start + Nz_inner ÷ ratio, Nz_c) - k_lo + 1
    kernel! = prolongate_f_rescaled_temporal_3d_kernel!(backend)
    kernel!(f_fine,
            f_curr, rho_curr, ux_curr, uy_curr, uz_curr,
            f_prev, rho_prev, ux_prev, uy_prev, uz_prev,
            ratio, Nx_inner, Ny_inner, Nz_inner, n_ghost,
            i_c_start, j_c_start, k_c_start,
            Nx_c, Ny_c, Nz_c, eltype(f_fine)(omega_c), eltype(f_fine)(omega_f),
            eltype(f_fine)(t_frac), Ni_prev, Nj_prev, Nk_prev;
            ndrange=(Nx_f, Ny_f, Nz_f))
end

# =====================================================================
# Restriction: fine → coarse (block-average + inverse Filippova-Hanel)
# =====================================================================

@kernel function restrict_f_rescaled_3d_kernel!(
        f_coarse, rho_c, ux_c, uy_c, uz_c,
        @Const(f_fine), @Const(rho_f), @Const(ux_f), @Const(uy_f), @Const(uz_f),
        ratio, n_ghost, i_c_start, j_c_start, k_c_start,
        omega_c, omega_f)

    ic, jc, kc = @index(Global, NTuple)

    @inbounds begin
        T = eltype(f_coarse)
        alpha_f2c = (T(omega_c) - T(0.5)) * (one(T) / (T(omega_f) - T(0.5)))
        inv_r3 = one(T) / T(ratio * ratio * ratio)

        i_c = ic + i_c_start - 1
        j_c = jc + j_c_start - 1
        k_c = kc + k_c_start - 1

        # Block-average macroscopic
        rho_avg = zero(T); ux_avg = zero(T); uy_avg = zero(T); uz_avg = zero(T)
        for dk in 0:ratio-1, dj in 0:ratio-1, di in 0:ratio-1
            i_ff = n_ghost + (ic - 1) * ratio + di + 1
            j_ff = n_ghost + (jc - 1) * ratio + dj + 1
            k_ff = n_ghost + (kc - 1) * ratio + dk + 1
            rho_avg += rho_f[i_ff, j_ff, k_ff]
            ux_avg  += ux_f[i_ff, j_ff, k_ff]
            uy_avg  += uy_f[i_ff, j_ff, k_ff]
            uz_avg  += uz_f[i_ff, j_ff, k_ff]
        end
        rho_avg *= inv_r3; ux_avg *= inv_r3; uy_avg *= inv_r3; uz_avg *= inv_r3

        rho_c[i_c, j_c, k_c] = rho_avg
        ux_c[i_c, j_c, k_c] = ux_avg
        uy_c[i_c, j_c, k_c] = uy_avg
        uz_c[i_c, j_c, k_c] = uz_avg

        # Block-average f_neq + inverse rescale
        for q in 1:19
            feq_c = _feq_3d(q, rho_avg, ux_avg, uy_avg, uz_avg)
            fneq_avg = zero(T)
            for dk in 0:ratio-1, dj in 0:ratio-1, di in 0:ratio-1
                i_ff = n_ghost + (ic - 1) * ratio + di + 1
                j_ff = n_ghost + (jc - 1) * ratio + dj + 1
                k_ff = n_ghost + (kc - 1) * ratio + dk + 1
                fneq_avg += f_fine[i_ff, j_ff, k_ff, q] -
                    _feq_3d(q, rho_f[i_ff,j_ff,k_ff], ux_f[i_ff,j_ff,k_ff],
                            uy_f[i_ff,j_ff,k_ff], uz_f[i_ff,j_ff,k_ff])
            end
            fneq_avg *= inv_r3
            f_coarse[i_c, j_c, k_c, q] = feq_c + alpha_f2c * fneq_avg
        end
    end
end

"""
    restrict_f_rescaled_3d!(f_coarse, rho_c, ux_c, uy_c, uz_c,

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.restrict_f_rescaled_3d!)
```
"""
function restrict_f_rescaled_3d!(f_coarse, rho_c, ux_c, uy_c, uz_c,
                                  f_fine, rho_f, ux_f, uy_f, uz_f,
                                  ratio, n_ghost, i_c_start, j_c_start, k_c_start,
                                  Nx_overlap, Ny_overlap, Nz_overlap,
                                  omega_c, omega_f)
    backend = KernelAbstractions.get_backend(f_coarse)
    kernel! = restrict_f_rescaled_3d_kernel!(backend)
    FT = eltype(f_coarse)
    kernel!(f_coarse, rho_c, ux_c, uy_c, uz_c,
            f_fine, rho_f, ux_f, uy_f, uz_f,
            ratio, n_ghost, i_c_start, j_c_start, k_c_start,
            FT(omega_c), FT(omega_f);
            ndrange=(Nx_overlap, Ny_overlap, Nz_overlap))
end

# =====================================================================
# Thermal prolongation: trilinear interpolation WITHOUT Filippova-Hanel
# (g_eq is linear in T, so simple interpolation suffices)
# =====================================================================

"""Trilinear + temporal interpolation of coarse g into fine ghost cells (3D)."""
@kernel function _fill_thermal_ghost_temporal_3d_kernel!(
        g_fine, @Const(g_coarse), @Const(g_prev),
        ratio, Nx_inner, Ny_inner, Nz_inner, n_ghost,
        i_offset, j_offset, k_offset,
        Nx_c, Ny_c, Nz_c, t_frac,
        i_lo, j_lo, k_lo, Ni_prev, Nj_prev, Nk_prev)

    i_f, j_f, k_f = @index(Global, NTuple)

    @inbounds begin
        is_ghost = (i_f <= n_ghost) || (i_f > n_ghost + Nx_inner) ||
                   (j_f <= n_ghost) || (j_f > n_ghost + Ny_inner) ||
                   (k_f <= n_ghost) || (k_f > n_ghost + Nz_inner)
        if is_ghost
            T = eltype(g_fine)
            t = T(t_frac)
            omt = one(T) - t

            xc = T(i_f - n_ghost - 1) / T(ratio) + T(i_offset) + T(0.5) / T(ratio)
            yc = T(j_f - n_ghost - 1) / T(ratio) + T(j_offset) + T(0.5) / T(ratio)
            zc = T(k_f - n_ghost - 1) / T(ratio) + T(k_offset) + T(0.5) / T(ratio)

            i0r = unsafe_trunc(Int, xc); j0r = unsafe_trunc(Int, yc); k0r = unsafe_trunc(Int, zc)
            tx = clamp(xc - T(i0r), zero(T), one(T))
            ty = clamp(yc - T(j0r), zero(T), one(T))
            tz = clamp(zc - T(k0r), zero(T), one(T))

            i0 = clamp(i0r, 1, Nx_c); i1 = clamp(i0r+1, 1, Nx_c)
            j0 = clamp(j0r, 1, Ny_c); j1 = clamp(j0r+1, 1, Ny_c)
            k0 = clamp(k0r, 1, Nz_c); k1 = clamp(k0r+1, 1, Nz_c)

            w000=(one(T)-tx)*(one(T)-ty)*(one(T)-tz); w100=tx*(one(T)-ty)*(one(T)-tz)
            w010=(one(T)-tx)*ty*(one(T)-tz);           w110=tx*ty*(one(T)-tz)
            w001=(one(T)-tx)*(one(T)-ty)*tz;           w101=tx*(one(T)-ty)*tz
            w011=(one(T)-tx)*ty*tz;                     w111=tx*ty*tz

            # Local indices into g_prev buffer
            ip0 = clamp(i0 - i_lo + 1, 1, Ni_prev)
            ip1 = clamp(i1 - i_lo + 1, 1, Ni_prev)
            jp0 = clamp(j0 - j_lo + 1, 1, Nj_prev)
            jp1 = clamp(j1 - j_lo + 1, 1, Nj_prev)
            kp0 = clamp(k0 - k_lo + 1, 1, Nk_prev)
            kp1 = clamp(k1 - k_lo + 1, 1, Nk_prev)

            for q in 1:19
                # Temporal blend at each stencil point
                g000 = omt * g_prev[ip0, jp0, kp0, q] + t * g_coarse[i0, j0, k0, q]
                g100 = omt * g_prev[ip1, jp0, kp0, q] + t * g_coarse[i1, j0, k0, q]
                g010 = omt * g_prev[ip0, jp1, kp0, q] + t * g_coarse[i0, j1, k0, q]
                g110 = omt * g_prev[ip1, jp1, kp0, q] + t * g_coarse[i1, j1, k0, q]
                g001 = omt * g_prev[ip0, jp0, kp1, q] + t * g_coarse[i0, j0, k1, q]
                g101 = omt * g_prev[ip1, jp0, kp1, q] + t * g_coarse[i1, j0, k1, q]
                g011 = omt * g_prev[ip0, jp1, kp1, q] + t * g_coarse[i0, j1, k1, q]
                g111 = omt * g_prev[ip1, jp1, kp1, q] + t * g_coarse[i1, j1, k1, q]

                # Spatial trilinear interpolation
                g_fine[i_f, j_f, k_f, q] = w000*g000 + w100*g100 + w010*g010 + w110*g110 +
                                             w001*g001 + w101*g101 + w011*g011 + w111*g111
            end
        end
    end
end

function _fill_thermal_ghost_temporal_3d!(g_fine, g_coarse, g_prev,
                                           ratio, Nx_inner, Ny_inner, Nz_inner, n_ghost,
                                           i_offset, j_offset, k_offset,
                                           Nx_c, Ny_c, Nz_c, t_frac)
    backend = KernelAbstractions.get_backend(g_fine)
    Nx_f = Nx_inner + 2 * n_ghost
    Ny_f = Ny_inner + 2 * n_ghost
    Nz_f = Nz_inner + 2 * n_ghost
    i_lo = max(i_offset - 1, 1)
    j_lo = max(j_offset - 1, 1)
    k_lo = max(k_offset - 1, 1)
    i_c_end = i_offset + Nx_inner ÷ ratio - 1
    j_c_end = j_offset + Ny_inner ÷ ratio - 1
    k_c_end = k_offset + Nz_inner ÷ ratio - 1
    Ni_prev = min(i_c_end + 1, Nx_c) - i_lo + 1
    Nj_prev = min(j_c_end + 1, Ny_c) - j_lo + 1
    Nk_prev = min(k_c_end + 1, Nz_c) - k_lo + 1
    kernel! = _fill_thermal_ghost_temporal_3d_kernel!(backend)
    kernel!(g_fine, g_coarse, g_prev, ratio, Nx_inner, Ny_inner, Nz_inner, n_ghost,
            i_offset, j_offset, k_offset, Nx_c, Ny_c, Nz_c,
            eltype(g_fine)(t_frac),
            i_lo, j_lo, k_lo, Ni_prev, Nj_prev, Nk_prev;
            ndrange=(Nx_f, Ny_f, Nz_f))
end

# =====================================================================
# Full thermal prolongation (init time — all cells, no FH)
# =====================================================================

@kernel function _fill_thermal_full_3d_kernel!(g_fine, @Const(g_coarse),
                                                ratio, n_ghost,
                                                i_offset, j_offset, k_offset,
                                                Nx_c, Ny_c, Nz_c)
    i_f, j_f, k_f = @index(Global, NTuple)

    @inbounds begin
        T = eltype(g_fine)
        xc = T(i_f - n_ghost - 1) / T(ratio) + T(i_offset) + T(0.5) / T(ratio)
        yc = T(j_f - n_ghost - 1) / T(ratio) + T(j_offset) + T(0.5) / T(ratio)
        zc = T(k_f - n_ghost - 1) / T(ratio) + T(k_offset) + T(0.5) / T(ratio)

        i0r = unsafe_trunc(Int, xc); j0r = unsafe_trunc(Int, yc); k0r = unsafe_trunc(Int, zc)
        tx = clamp(xc - T(i0r), zero(T), one(T))
        ty = clamp(yc - T(j0r), zero(T), one(T))
        tz = clamp(zc - T(k0r), zero(T), one(T))
        i0 = clamp(i0r, 1, Nx_c); i1 = clamp(i0r+1, 1, Nx_c)
        j0 = clamp(j0r, 1, Ny_c); j1 = clamp(j0r+1, 1, Ny_c)
        k0 = clamp(k0r, 1, Nz_c); k1 = clamp(k0r+1, 1, Nz_c)

        w000=(one(T)-tx)*(one(T)-ty)*(one(T)-tz); w100=tx*(one(T)-ty)*(one(T)-tz)
        w010=(one(T)-tx)*ty*(one(T)-tz);           w110=tx*ty*(one(T)-tz)
        w001=(one(T)-tx)*(one(T)-ty)*tz;           w101=tx*(one(T)-ty)*tz
        w011=(one(T)-tx)*ty*tz;                     w111=tx*ty*tz

        for q in 1:19
            g_fine[i_f, j_f, k_f, q] = w000 * g_coarse[i0,j0,k0,q] +
                                         w100 * g_coarse[i1,j0,k0,q] +
                                         w010 * g_coarse[i0,j1,k0,q] +
                                         w110 * g_coarse[i1,j1,k0,q] +
                                         w001 * g_coarse[i0,j0,k1,q] +
                                         w101 * g_coarse[i1,j0,k1,q] +
                                         w011 * g_coarse[i0,j1,k1,q] +
                                         w111 * g_coarse[i1,j1,k1,q]
        end
    end
end

# =====================================================================
# Thermal restriction: simple block-average over ratio³ cells (no FH)
# =====================================================================

@kernel function _restrict_thermal_simple_3d_kernel!(g_coarse, Temp_c,
                                                      @Const(g_fine), @Const(Temp_f),
                                                      ratio, n_ghost,
                                                      i_offset, j_offset, k_offset)
    ic, jc, kc = @index(Global, NTuple)

    @inbounds begin
        T = eltype(g_coarse)
        i_c = ic + i_offset - 1
        j_c = jc + j_offset - 1
        k_c = kc + k_offset - 1
        inv_r3 = one(T) / T(ratio * ratio * ratio)

        # Block-average temperature
        Temp_avg = zero(T)
        for dk in 0:ratio-1, dj in 0:ratio-1, di in 0:ratio-1
            i_f = n_ghost + (ic - 1) * ratio + di + 1
            j_f = n_ghost + (jc - 1) * ratio + dj + 1
            k_f = n_ghost + (kc - 1) * ratio + dk + 1
            Temp_avg += Temp_f[i_f, j_f, k_f]
        end
        Temp_c[i_c, j_c, k_c] = Temp_avg * inv_r3

        # Block-average g populations
        for q in 1:19
            g_avg = zero(T)
            for dk in 0:ratio-1, dj in 0:ratio-1, di in 0:ratio-1
                i_f = n_ghost + (ic - 1) * ratio + di + 1
                j_f = n_ghost + (jc - 1) * ratio + dj + 1
                k_f = n_ghost + (kc - 1) * ratio + dk + 1
                g_avg += g_fine[i_f, j_f, k_f, q]
            end
            g_coarse[i_c, j_c, k_c, q] = g_avg * inv_r3
        end
    end
end
