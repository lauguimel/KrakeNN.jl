using KernelAbstractions

# --- Viscoelastic evolution kernels (2D) ---
#
# Two formulations:
#   StressFormulation:   evolve τ_p directly
#   LogConfFormulation:  evolve Θ = log(C) (Fattal & Kupferman 2004)
#
# Both are implemented as fused advection+source+relaxation kernels
# (single launch per time step for the 3 tensor components).

# ============================================================
# Polymeric stress divergence → force for Guo scheme
# ============================================================

@kernel function compute_polymeric_force_2d_kernel!(Fx_p, Fy_p,
                                                      @Const(tau_xx), @Const(tau_xy), @Const(tau_yy),
                                                      Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(Fx_p)

        # Central differences for divergence: F_i = ∂τ_ij/∂x_j
        # Periodic/clamped indices
        ip = ifelse(i < Nx, i + 1, 1)
        im = ifelse(i > 1,  i - 1, Nx)
        jp = min(j + 1, Ny)
        jm = max(j - 1, 1)

        # Fx = ∂τ_xx/∂x + ∂τ_xy/∂y
        Fx_p[i,j] = (tau_xx[ip,j] - tau_xx[im,j]) / T(2) +
                     (tau_xy[i,jp] - tau_xy[i,jm]) / T(2)

        # Fy = ∂τ_xy/∂x + ∂τ_yy/∂y
        Fy_p[i,j] = (tau_xy[ip,j] - tau_xy[im,j]) / T(2) +
                     (tau_yy[i,jp] - tau_yy[i,jm]) / T(2)
    end
end

"""
    compute_polymeric_force_2d!(Fx_p, Fy_p, tau_xx, tau_xy, tau_yy)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.compute_polymeric_force_2d!)
```
"""
function compute_polymeric_force_2d!(Fx_p, Fy_p, tau_xx, tau_xy, tau_yy)
    backend = KernelAbstractions.get_backend(Fx_p)
    Nx, Ny = size(Fx_p)
    kernel! = compute_polymeric_force_2d_kernel!(backend)
    kernel!(Fx_p, Fy_p, tau_xx, tau_xy, tau_yy, Nx, Ny; ndrange=(Nx, Ny))
end

# ============================================================
# Stress formulation: evolve τ_p directly
# ============================================================
#
# Upper-convected Maxwell:
#   ∂τ_p/∂t + u·∇τ_p = τ_p·∇u + (∇u)ᵀ·τ_p - (1/λ)·τ_p + (ν_p/λ)·(∇u + (∇u)ᵀ)
#
# Upwind advection + source + relaxation in a single kernel.

@kernel function evolve_stress_2d_kernel!(tau_xx_new, tau_xy_new, tau_yy_new,
                                            @Const(tau_xx), @Const(tau_xy), @Const(tau_yy),
                                            @Const(ux), @Const(uy),
                                            nu_p, lambda, Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(tau_xx)
        inv_lambda = one(T) / lambda

        # Velocity at this node
        u = ux[i,j]
        v = uy[i,j]

        # --- Upwind advection of τ_p ---
        # Neighbor indices (periodic in x, clamped in y)
        ip = ifelse(i < Nx, i + 1, 1)
        im = ifelse(i > 1,  i - 1, Nx)
        jp = min(j + 1, Ny)
        jm = max(j - 1, 1)

        # Upwind in x
        dtxx_dx = ifelse(u > zero(T),
                         tau_xx[i,j] - tau_xx[im,j],
                         tau_xx[ip,j] - tau_xx[i,j])
        dtxy_dx = ifelse(u > zero(T),
                         tau_xy[i,j] - tau_xy[im,j],
                         tau_xy[ip,j] - tau_xy[i,j])
        dtyy_dx = ifelse(u > zero(T),
                         tau_yy[i,j] - tau_yy[im,j],
                         tau_yy[ip,j] - tau_yy[i,j])

        # Upwind in y
        dtxx_dy = ifelse(v > zero(T),
                         tau_xx[i,j] - tau_xx[i,jm],
                         tau_xx[i,jp] - tau_xx[i,j])
        dtxy_dy = ifelse(v > zero(T),
                         tau_xy[i,j] - tau_xy[i,jm],
                         tau_xy[i,jp] - tau_xy[i,j])
        dtyy_dy = ifelse(v > zero(T),
                         tau_yy[i,j] - tau_yy[i,jm],
                         tau_yy[i,jp] - tau_yy[i,j])

        # Advection: -u·∇τ
        adv_xx = -(u * dtxx_dx + v * dtxx_dy)
        adv_xy = -(u * dtxy_dx + v * dtxy_dy)
        adv_yy = -(u * dtyy_dx + v * dtyy_dy)

        # --- Velocity gradient (central differences) ---
        dudx = (ux[ip,j] - ux[im,j]) / T(2)
        dudy = (ux[i,jp] - ux[i,jm]) / T(2)
        dvdx = (uy[ip,j] - uy[im,j]) / T(2)
        dvdy = (uy[i,jp] - uy[i,jm]) / T(2)

        # --- Source: upper-convected derivative L·τ + τ·Lᵀ ---
        # L_ij = ∂u_i/∂x_j (transpose convention).
        # (L·τ + τ·Lᵀ)_xx = 2·(L11·τxx + L12·τxy) = 2·(dudx·τxx + dudy·τxy)
        # (L·τ + τ·Lᵀ)_yy = 2·(L21·τxy + L22·τyy) = 2·(dvdx·τxy + dvdy·τyy)
        # (L·τ + τ·Lᵀ)_xy = (dudx+dvdy)·τxy + dudy·τyy + dvdx·τxx
        txx = tau_xx[i,j]; txy = tau_xy[i,j]; tyy = tau_yy[i,j]
        src_xx = T(2) * (txx * dudx + txy * dudy)
        src_xy = txx * dvdx + tyy * dudy + txy * (dudx + dvdy)
        src_yy = T(2) * (txy * dvdx + tyy * dvdy)

        # Newtonian contribution: (ν_p/λ)·(∇u + (∇u)ᵀ)
        newt_xx = nu_p * inv_lambda * T(2) * dudx
        newt_xy = nu_p * inv_lambda * (dudy + dvdx)
        newt_yy = nu_p * inv_lambda * T(2) * dvdy

        # --- Relaxation: -(1/λ)·τ_p ---
        relax_xx = -inv_lambda * txx
        relax_xy = -inv_lambda * txy
        relax_yy = -inv_lambda * tyy

        # --- Update (explicit Euler) ---
        tau_xx_new[i,j] = txx + adv_xx + src_xx + newt_xx + relax_xx
        tau_xy_new[i,j] = txy + adv_xy + src_xy + newt_xy + relax_xy
        tau_yy_new[i,j] = tyy + adv_yy + src_yy + newt_yy + relax_yy
    end
end

"""
    evolve_stress_2d!(tau_xx_new, tau_xy_new, tau_yy_new,

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.evolve_stress_2d!)
```
"""
function evolve_stress_2d!(tau_xx_new, tau_xy_new, tau_yy_new,
                            tau_xx, tau_xy, tau_yy,
                            ux, uy, nu_p, lambda)
    backend = KernelAbstractions.get_backend(tau_xx)
    Nx, Ny = size(tau_xx)
    T = eltype(tau_xx)
    kernel! = evolve_stress_2d_kernel!(backend)
    kernel!(tau_xx_new, tau_xy_new, tau_yy_new,
            tau_xx, tau_xy, tau_yy,
            ux, uy, T(nu_p), T(lambda), Nx, Ny; ndrange=(Nx, Ny))
end

# ============================================================
# Log-conformation formulation: evolve Θ = log(C)
# ============================================================
#
# Fattal & Kupferman (2004):
#   ∂Θ/∂t + u·∇Θ = Ω·Θ - Θ·Ω + 2B + (1/λ)·(e^{-Θ} - I)
#
# where Ω (rotation) and B (extension) come from decomposing
# ∇u in the eigenbasis of C = exp(Θ).
#
# For FENE-P: relaxation becomes (1/λ)·(f(tr(C))·e^{-Θ} - I)
# where f = L²/(L² - tr(C)) is the Peterlin function.

@kernel function evolve_logconf_2d_kernel!(Theta_xx_new, Theta_xy_new, Theta_yy_new,
                                             @Const(Theta_xx), @Const(Theta_xy), @Const(Theta_yy),
                                             @Const(ux), @Const(uy),
                                             inv_lambda, L2_fene, Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(Theta_xx)

        u = ux[i,j]
        v = uy[i,j]

        # --- Upwind advection of Θ ---
        ip = ifelse(i < Nx, i + 1, 1)
        im = ifelse(i > 1,  i - 1, Nx)
        jp = min(j + 1, Ny)
        jm = max(j - 1, 1)

        # Current values
        θ_xx = Theta_xx[i,j]; θ_xy = Theta_xy[i,j]; θ_yy = Theta_yy[i,j]

        # Upwind x
        dθxx_dx = ifelse(u > zero(T), θ_xx - Theta_xx[im,j], Theta_xx[ip,j] - θ_xx)
        dθxy_dx = ifelse(u > zero(T), θ_xy - Theta_xy[im,j], Theta_xy[ip,j] - θ_xy)
        dθyy_dx = ifelse(u > zero(T), θ_yy - Theta_yy[im,j], Theta_yy[ip,j] - θ_yy)

        # Upwind y
        dθxx_dy = ifelse(v > zero(T), θ_xx - Theta_xx[i,jm], Theta_xx[i,jp] - θ_xx)
        dθxy_dy = ifelse(v > zero(T), θ_xy - Theta_xy[i,jm], Theta_xy[i,jp] - θ_xy)
        dθyy_dy = ifelse(v > zero(T), θ_yy - Theta_yy[i,jm], Theta_yy[i,jp] - θ_yy)

        adv_xx = -(u * dθxx_dx + v * dθxx_dy)
        adv_xy = -(u * dθxy_dx + v * dθxy_dy)
        adv_yy = -(u * dθyy_dx + v * dθyy_dy)

        # --- Velocity gradient ---
        dudx = (ux[ip,j] - ux[im,j]) / T(2)
        dudy = (ux[i,jp] - ux[i,jm]) / T(2)
        dvdx = (uy[ip,j] - uy[im,j]) / T(2)
        dvdy = (uy[i,jp] - uy[i,jm]) / T(2)

        # --- Eigendecomposition of C = exp(Θ) ---
        # First get eigenvalues/vectors of Θ
        λ1, λ2, e1x, e1y, e2x, e2y = eigen_sym2x2(θ_xx, θ_xy, θ_yy)

        # C eigenvalues = exp(Θ eigenvalues)
        exp_λ1 = exp(λ1)
        exp_λ2 = exp(λ2)

        # --- Decompose ∇u in eigenbasis (Fattal & Kupferman 2004) ---
        # NOTE: pass eigenvalues of Θ (not C); the function uses exp(λ) internally.
        Omega12, B11, B22 = decompose_velocity_gradient(dudx, dudy, dvdx, dvdy,
                                                         e1x, e1y, e2x, e2y, λ1, λ2)

        # --- Source terms in Θ equation ---
        # ΩΘ - ΘΩ + 2B  (in eigenbasis, then transform back)
        # In eigenbasis: Θ_eig = diag(λ1, λ2)
        # [ΩΘ - ΘΩ]_12 = Ω12 · (λ1 - λ2)
        # [2B]_11 = 2B11, [2B]_22 = 2B22

        # Source in eigenbasis
        src_eig_11 = T(2) * B11
        src_eig_22 = T(2) * B22
        src_eig_12 = Omega12 * (λ1 - λ2)

        # --- Relaxation: (1/λ)·(e^{-Θ} - I) ---
        # In eigenbasis: diag(1/λ · (exp(-λ_i) - 1))
        # With FENE-P Peterlin function: f = L²/(L² - tr(C))
        trC = exp_λ1 + exp_λ2
        fene_factor = ifelse(L2_fene > zero(T),
                             L2_fene / max(L2_fene - trC, T(0.01)),
                             one(T))  # L2=0 means Oldroyd-B (no FENE)

        relax_eig_11 = inv_lambda * (fene_factor * exp(-λ1) - one(T))
        relax_eig_22 = inv_lambda * (fene_factor * exp(-λ2) - one(T))
        relax_eig_12 = zero(T)

        # --- Total source in eigenbasis ---
        tot_11 = src_eig_11 + relax_eig_11
        tot_22 = src_eig_22 + relax_eig_22
        tot_12 = src_eig_12 + relax_eig_12

        # --- Transform back to lab frame: S_lab = R · S_eig · Rᵀ ---
        S_xx = e1x * e1x * tot_11 + e2x * e2x * tot_22 + T(2) * e1x * e2x * tot_12
        S_xy = e1x * e1y * tot_11 + e2x * e2y * tot_22 + (e1x * e2y + e1y * e2x) * tot_12
        S_yy = e1y * e1y * tot_11 + e2y * e2y * tot_22 + T(2) * e1y * e2y * tot_12

        # --- Update Θ (explicit Euler) ---
        Theta_xx_new[i,j] = θ_xx + adv_xx + S_xx
        Theta_xy_new[i,j] = θ_xy + adv_xy + S_xy
        Theta_yy_new[i,j] = θ_yy + adv_yy + S_yy
    end
end

"""
    evolve_logconf_2d!(Theta_xx_new, Theta_xy_new, Theta_yy_new,

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.evolve_logconf_2d!)
```
"""
function evolve_logconf_2d!(Theta_xx_new, Theta_xy_new, Theta_yy_new,
                             Theta_xx, Theta_xy, Theta_yy,
                             ux, uy; lambda=1.0, L_max=0.0)
    backend = KernelAbstractions.get_backend(Theta_xx)
    Nx, Ny = size(Theta_xx)
    FT = eltype(Theta_xx)
    inv_lambda = FT(1.0 / lambda)
    L2 = L_max > 0 ? FT(L_max * L_max) : FT(0)
    kernel! = evolve_logconf_2d_kernel!(backend)
    kernel!(Theta_xx_new, Theta_xy_new, Theta_yy_new,
            Theta_xx, Theta_xy, Theta_yy,
            ux, uy, inv_lambda, L2, Nx, Ny; ndrange=(Nx, Ny))
end

# ============================================================
# Compute polymeric stress from conformation tensor
# ============================================================

"""
    compute_stress_from_conf_2d!(tau_xx, tau_xy, tau_yy, Cxx, Cxy, Cyy, G)

Compute τ_p = G · (C - I) where G = ν_p/λ is the elastic modulus.
For FENE-P: τ_p = G · f(tr(C)) · (C - I).
"""
@kernel function stress_from_conf_2d_kernel!(tau_xx, tau_xy, tau_yy,
                                               @Const(Cxx), @Const(Cxy), @Const(Cyy),
                                               G, L2_fene)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(tau_xx)
        cxx = Cxx[i,j]; cxy = Cxy[i,j]; cyy = Cyy[i,j]
        trC = cxx + cyy
        fene = ifelse(L2_fene > zero(T),
                      L2_fene / max(L2_fene - trC, T(0.01)),
                      one(T))
        tau_xx[i,j] = G * fene * (cxx - one(T))
        tau_xy[i,j] = G * fene * cxy
        tau_yy[i,j] = G * fene * (cyy - one(T))
    end
end

"""
    compute_stress_from_conf_2d!(tau_xx, tau_xy, tau_yy, Cxx, Cxy, Cyy;

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.compute_stress_from_conf_2d!)
```
"""
function compute_stress_from_conf_2d!(tau_xx, tau_xy, tau_yy, Cxx, Cxy, Cyy;
                                       G=1.0, L_max=0.0)
    backend = KernelAbstractions.get_backend(tau_xx)
    Nx, Ny = size(tau_xx)
    FT = eltype(tau_xx)
    L2 = L_max > 0 ? FT(L_max * L_max) : FT(0)
    kernel! = stress_from_conf_2d_kernel!(backend)
    kernel!(tau_xx, tau_xy, tau_yy, Cxx, Cxy, Cyy, FT(G), L2; ndrange=(Nx, Ny))
end

"""
    compute_stress_from_logconf_2d!(tau_xx, tau_xy, tau_yy, Θ_xx, Θ_xy, Θ_yy, G; L_max=0.0)

Compute τ_p from log-conformation Θ: C = exp(Θ), then τ_p = G · f(tr(C)) · (C - I).
"""
@kernel function stress_from_logconf_2d_kernel!(tau_xx, tau_xy, tau_yy,
                                                  @Const(Theta_xx), @Const(Theta_xy), @Const(Theta_yy),
                                                  G, L2_fene)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(tau_xx)
        # C = exp(Θ)
        cxx, cxy, cyy = mat_exp_sym2x2(Theta_xx[i,j], Theta_xy[i,j], Theta_yy[i,j])
        trC = cxx + cyy
        fene = ifelse(L2_fene > zero(T),
                      L2_fene / max(L2_fene - trC, T(0.01)),
                      one(T))
        tau_xx[i,j] = G * fene * (cxx - one(T))
        tau_xy[i,j] = G * fene * cxy
        tau_yy[i,j] = G * fene * (cyy - one(T))
    end
end

"""
    compute_stress_from_logconf_2d!(tau_xx, tau_xy, tau_yy,

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.compute_stress_from_logconf_2d!)
```
"""
function compute_stress_from_logconf_2d!(tau_xx, tau_xy, tau_yy,
                                          Theta_xx, Theta_xy, Theta_yy;
                                          G=1.0, L_max=0.0)
    backend = KernelAbstractions.get_backend(tau_xx)
    Nx, Ny = size(tau_xx)
    FT = eltype(tau_xx)
    L2 = L_max > 0 ? FT(L_max * L_max) : FT(0)
    kernel! = stress_from_logconf_2d_kernel!(backend)
    kernel!(tau_xx, tau_xy, tau_yy, Theta_xx, Theta_xy, Theta_yy,
            FT(G), L2; ndrange=(Nx, Ny))
end
