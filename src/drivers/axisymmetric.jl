# --- Axisymmetric D2Q9 (z-r coordinates) ---

"""
    collide_axisymmetric_2d!(f, is_solid, ω, Nx, Ny)

BGK collision with axisymmetric source terms for D2Q9 in (z,r) coordinates.
x-direction = z (axial), y-direction = r (radial).

Source term from Peng et al. (2003) / Zhou (2011):
Accounts for 1/r geometric terms in cylindrical Navier-Stokes.
r = j - 0.5 (half-way BB, r=0 axis at j=0.5).
"""
@kernel function collide_axisymmetric_2d_kernel!(f, @Const(is_solid), ω, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j]
            tmp2 = f[i,j,2]; f[i,j,2] = f[i,j,4]; f[i,j,4] = tmp2
            tmp3 = f[i,j,3]; f[i,j,3] = f[i,j,5]; f[i,j,5] = tmp3
            tmp6 = f[i,j,6]; f[i,j,6] = f[i,j,8]; f[i,j,8] = tmp6
            tmp7 = f[i,j,7]; f[i,j,7] = f[i,j,9]; f[i,j,9] = tmp7
        else
            T = eltype(f)
            f1=f[i,j,1]; f2=f[i,j,2]; f3=f[i,j,3]; f4=f[i,j,4]
            f5=f[i,j,5]; f6=f[i,j,6]; f7=f[i,j,7]; f8=f[i,j,8]; f9=f[i,j,9]

            ρ = f1+f2+f3+f4+f5+f6+f7+f8+f9
            inv_ρ = one(T) / ρ
            uz = (f2 - f4 + f6 - f7 - f8 + f9) * inv_ρ  # axial (x=z)
            ur = (f3 - f5 + f6 + f7 - f8 - f9) * inv_ρ  # radial (y=r)
            usq = uz * uz + ur * ur

            # Radial position: r = j - 0.5 (half-way BB, axis at r=0)
            r = T(j) - T(0.5)
            inv_r = one(T) / r

            # Standard BGK collision
            cu = zero(T)
            feq = T(4.0/9.0) * ρ * (one(T) - T(1.5)*usq)
            f[i,j,1] = f1 - ω*(f1-feq)

            cu = uz
            feq = T(1.0/9.0) * ρ * (one(T) + T(3)*cu + T(4.5)*cu*cu - T(1.5)*usq)
            f[i,j,2] = f2 - ω*(f2-feq)

            cu = ur
            feq = T(1.0/9.0) * ρ * (one(T) + T(3)*cu + T(4.5)*cu*cu - T(1.5)*usq)
            f[i,j,3] = f3 - ω*(f3-feq)

            cu = -uz
            feq = T(1.0/9.0) * ρ * (one(T) + T(3)*cu + T(4.5)*cu*cu - T(1.5)*usq)
            f[i,j,4] = f4 - ω*(f4-feq)

            cu = -ur
            feq = T(1.0/9.0) * ρ * (one(T) + T(3)*cu + T(4.5)*cu*cu - T(1.5)*usq)
            f[i,j,5] = f5 - ω*(f5-feq)

            cu = uz + ur
            feq = T(1.0/36.0) * ρ * (one(T) + T(3)*cu + T(4.5)*cu*cu - T(1.5)*usq)
            f[i,j,6] = f6 - ω*(f6-feq)

            cu = -uz + ur
            feq = T(1.0/36.0) * ρ * (one(T) + T(3)*cu + T(4.5)*cu*cu - T(1.5)*usq)
            f[i,j,7] = f7 - ω*(f7-feq)

            cu = -uz - ur
            feq = T(1.0/36.0) * ρ * (one(T) + T(3)*cu + T(4.5)*cu*cu - T(1.5)*usq)
            f[i,j,8] = f8 - ω*(f8-feq)

            cu = uz - ur
            feq = T(1.0/36.0) * ρ * (one(T) + T(3)*cu + T(4.5)*cu*cu - T(1.5)*usq)
            f[i,j,9] = f9 - ω*(f9-feq)

            # Axisymmetric source: S_q = -w_q * ur/r * (1 + (c_qr*ur)*3 ... )
            # Simplified form (Zhou 2011): S_q = -f_eq_q * ur / r
            # This accounts for the ∂(r·ur)/r∂r - ur/r mass/momentum correction
            pref = -ur * inv_r

            f[i,j,1] = f[i,j,1] + pref * T(4.0/9.0) * ρ * (one(T) - T(1.5)*usq)
            f[i,j,2] = f[i,j,2] + pref * T(1.0/9.0) * ρ * (one(T) + T(3)*uz + T(4.5)*uz*uz - T(1.5)*usq)
            f[i,j,3] = f[i,j,3] + pref * T(1.0/9.0) * ρ * (one(T) + T(3)*ur + T(4.5)*ur*ur - T(1.5)*usq)
            f[i,j,4] = f[i,j,4] + pref * T(1.0/9.0) * ρ * (one(T) - T(3)*uz + T(4.5)*uz*uz - T(1.5)*usq)
            f[i,j,5] = f[i,j,5] + pref * T(1.0/9.0) * ρ * (one(T) - T(3)*ur + T(4.5)*ur*ur - T(1.5)*usq)
            f[i,j,6] = f[i,j,6] + pref * T(1.0/36.0) * ρ * (one(T) + T(3)*(uz+ur) + T(4.5)*(uz+ur)*(uz+ur) - T(1.5)*usq)
            f[i,j,7] = f[i,j,7] + pref * T(1.0/36.0) * ρ * (one(T) + T(3)*(-uz+ur) + T(4.5)*(-uz+ur)*(-uz+ur) - T(1.5)*usq)
            f[i,j,8] = f[i,j,8] + pref * T(1.0/36.0) * ρ * (one(T) + T(3)*(-uz-ur) + T(4.5)*(-uz-ur)*(-uz-ur) - T(1.5)*usq)
            f[i,j,9] = f[i,j,9] + pref * T(1.0/36.0) * ρ * (one(T) + T(3)*(uz-ur) + T(4.5)*(uz-ur)*(uz-ur) - T(1.5)*usq)
        end
    end
end

"""
    collide_axisymmetric_2d!(f, is_solid, ω, Nx, Ny)

Public function in the high-level simulation driver API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.collide_axisymmetric_2d!)
```
"""
function collide_axisymmetric_2d!(f, is_solid, ω, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = collide_axisymmetric_2d_kernel!(backend)
    kernel!(f, is_solid, ω, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# --- Li et al. (2010) axisymmetric LBM scheme ---
#
# From: "Improved axisymmetric lattice Boltzmann scheme", PRE 81, 056707
#
# Key formulas (Eqs. 12-13, 16-17):
# Source:    S_α = [(e_αi - u_i)·F_i / (ρ·cs²) - u_r/r] · f_α^eq
#            where F_r = -2μ·u_r/r²  (only radial component)
# Collision: f̂(x+e, t+1) = f̂(x,t) - ω_f·[f̂ - f^eq] + (1-0.5·ω_f)·S
#            where ω_f = [1 + τ·e_αr/r] / (τ + 0.5)  — direction-dependent!
# Velocity:  u_i = Σ e_αi·f̂_α / [Σ f̂_α + μ/r²·δ_ir]
# Density:   ρ = Σ f̂_α / [1 + 0.5·u_r/r]

# The driver uses a combined stream → collision approach.
# Since ω_f depends on direction, we implement the full collision inline.

@kernel function collide_li_axisym_2d_kernel!(f, @Const(is_solid), τ, Ny, Fz_body)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j]
            tmp = f[i,j,2]; f[i,j,2] = f[i,j,4]; f[i,j,4] = tmp
            tmp = f[i,j,3]; f[i,j,3] = f[i,j,5]; f[i,j,5] = tmp
            tmp = f[i,j,6]; f[i,j,6] = f[i,j,8]; f[i,j,8] = tmp
            tmp = f[i,j,7]; f[i,j,7] = f[i,j,9]; f[i,j,9] = tmp
        else
            T = eltype(f)
            f1=f[i,j,1]; f2=f[i,j,2]; f3=f[i,j,3]; f4=f[i,j,4]
            f5=f[i,j,5]; f6=f[i,j,6]; f7=f[i,j,7]; f8=f[i,j,8]; f9=f[i,j,9]

            r = T(j) - T(0.5)
            inv_r = one(T) / r
            μ = τ * T(1.0/3.0)  # dynamic viscosity = τ·cs²

            # Macroscopic: Eq. (16) — u_i = Σ e_αi·f̂ / [Σ f̂ + μ/r²·δ_ir]
            sum_f = f1+f2+f3+f4+f5+f6+f7+f8+f9
            mom_z = f2-f4+f6-f7-f8+f9
            mom_r = f3-f5+f6+f7-f8-f9

            denom_z = sum_f  # no correction for z
            denom_r = sum_f + μ * inv_r * inv_r  # Eq. (16) with δ_ir

            uz = mom_z / denom_z
            ur = mom_r / denom_r

            # Density: Eq. (17) — ρ = Σ f̂ / [1 + 0.5·u_r/r]
            ρ = sum_f / (one(T) + T(0.5) * ur * inv_r)

            usq = uz*uz + ur*ur

            # Force: F_r = -2μ·u_r/r², F_z = Fz_body (external body force)
            Fr = -T(2) * μ * ur * inv_r * inv_r
            Fz = Fz_body

            # D2Q9 velocity components: e_αr for each direction
            # q: 1  2  3  4  5  6  7  8  9
            # cr: 0  0  1  0 -1  1  1 -1 -1
            # cz: 0  1  0 -1  0  1 -1 -1  1

            # For each direction α:
            # ω_f = [1 + τ·e_αr/r] / (τ + 0.5)          — Eq. (13)
            # S_α = [(e_αi - u_i)·F_geom/(ρcs²) - ur/r] · f_eq  — Eq. (12)
            # Guo_α = (1-0.5ω_f)·w_α·[(c-u)·F_ext/cs² + (c·u)(c·F_ext)/cs⁴]
            # f = f - ω_f·(f-feq) + (1-0.5ω_f)·S_α + Guo_α

            inv_cs2 = T(3)
            ur_over_r = ur * inv_r
            ω_std = one(T) / (τ + T(0.5))  # standard ω for Guo external force
            guo_ext_pref = one(T) - T(0.5) * ω_std  # (1-ω/2) for external force

            # Helper: compute collision for one direction
            # cz_q, cr_q = lattice velocity components
            # w_q = weight
            # For HP with ur=0: S_α=0, only ω_f asymmetry + Guo external force matter

            # Macro for each direction: BGK(ω_f) + axisym source + Guo external force
            # Inlined for all 9 directions:

            # q=1: rest (cz=0, cr=0), w=4/9
            feq = T(4.0/9.0)*ρ*(one(T) - T(1.5)*usq)
            ω_f = one(T) / (τ + T(0.5))
            # Geometric source: S = [(0-uz)*0 + (0-ur)*Fr]/(ρ·cs²) - ur/r] * feq
            Sq = ((-ur)*Fr * inv_cs2 / ρ - ur_over_r) * feq
            # Guo external: w*(c-u)·Fext*3 + w*(c·u)(c·Fext)*9
            Gq = T(4.0/9.0)*((-uz)*Fz*T(3))
            f[i,j,1] = f1 - ω_f*(f1-feq) + (one(T)-T(0.5)*ω_f)*Sq + guo_ext_pref*Gq

            # q=2: E (cz=1, cr=0), w=1/9
            cu=uz; feq=T(1.0/9.0)*ρ*(one(T)+T(3)*cu+T(4.5)*cu*cu-T(1.5)*usq)
            ω_f = one(T) / (τ + T(0.5))
            Sq = (((-ur)*Fr * inv_cs2 / ρ) - ur_over_r) * feq
            Gq = T(1.0/9.0)*((one(T)-uz)*Fz*T(3) + uz*Fz*T(9))
            f[i,j,2] = f2 - ω_f*(f2-feq) + (one(T)-T(0.5)*ω_f)*Sq + guo_ext_pref*Gq

            # q=3: N (cz=0, cr=+1), w=1/9
            cu=ur; feq=T(1.0/9.0)*ρ*(one(T)+T(3)*cu+T(4.5)*cu*cu-T(1.5)*usq)
            ω_f = (one(T) + τ*inv_r) / (τ + T(0.5))
            Sq = (((one(T)-ur)*Fr * inv_cs2 / ρ) - ur_over_r) * feq
            Gq = T(1.0/9.0)*((-uz)*Fz*T(3))
            f[i,j,3] = f3 - ω_f*(f3-feq) + (one(T)-T(0.5)*ω_f)*Sq + guo_ext_pref*Gq

            # q=4: W (cz=-1, cr=0), w=1/9
            cu=-uz; feq=T(1.0/9.0)*ρ*(one(T)+T(3)*cu+T(4.5)*cu*cu-T(1.5)*usq)
            ω_f = one(T) / (τ + T(0.5))
            Sq = (((-ur)*Fr * inv_cs2 / ρ) - ur_over_r) * feq
            Gq = T(1.0/9.0)*((-one(T)-uz)*Fz*T(3) + uz*Fz*T(9))
            f[i,j,4] = f4 - ω_f*(f4-feq) + (one(T)-T(0.5)*ω_f)*Sq + guo_ext_pref*Gq

            # q=5: S (cz=0, cr=-1), w=1/9
            cu=-ur; feq=T(1.0/9.0)*ρ*(one(T)+T(3)*cu+T(4.5)*cu*cu-T(1.5)*usq)
            ω_f = (one(T) - τ*inv_r) / (τ + T(0.5))
            Sq = (((-one(T)-ur)*Fr * inv_cs2 / ρ) - ur_over_r) * feq
            Gq = T(1.0/9.0)*((-uz)*Fz*T(3))
            f[i,j,5] = f5 - ω_f*(f5-feq) + (one(T)-T(0.5)*ω_f)*Sq + guo_ext_pref*Gq

            # q=6: NE (cz=1, cr=+1), w=1/36
            cu=uz+ur; feq=T(1.0/36.0)*ρ*(one(T)+T(3)*cu+T(4.5)*cu*cu-T(1.5)*usq)
            ω_f = (one(T) + τ*inv_r) / (τ + T(0.5))
            Sq = (((one(T)-ur)*Fr * inv_cs2 / ρ) - ur_over_r) * feq
            Gq = T(1.0/36.0)*((one(T)-uz)*Fz*T(3) + cu*Fz*T(9))
            f[i,j,6] = f6 - ω_f*(f6-feq) + (one(T)-T(0.5)*ω_f)*Sq + guo_ext_pref*Gq

            # q=7: NW (cz=-1, cr=+1), w=1/36
            cu=-uz+ur; feq=T(1.0/36.0)*ρ*(one(T)+T(3)*cu+T(4.5)*cu*cu-T(1.5)*usq)
            ω_f = (one(T) + τ*inv_r) / (τ + T(0.5))
            Sq = (((one(T)-ur)*Fr * inv_cs2 / ρ) - ur_over_r) * feq
            Gq = T(1.0/36.0)*((-one(T)-uz)*Fz*T(3) + cu*(-Fz)*T(9))
            f[i,j,7] = f7 - ω_f*(f7-feq) + (one(T)-T(0.5)*ω_f)*Sq + guo_ext_pref*Gq

            # q=8: SW (cz=-1, cr=-1), w=1/36
            cu=-uz-ur; feq=T(1.0/36.0)*ρ*(one(T)+T(3)*cu+T(4.5)*cu*cu-T(1.5)*usq)
            ω_f = (one(T) - τ*inv_r) / (τ + T(0.5))
            Sq = (((-one(T)-ur)*Fr * inv_cs2 / ρ) - ur_over_r) * feq
            Gq = T(1.0/36.0)*((-one(T)-uz)*Fz*T(3) + cu*(-Fz)*T(9))
            f[i,j,8] = f8 - ω_f*(f8-feq) + (one(T)-T(0.5)*ω_f)*Sq + guo_ext_pref*Gq

            # q=9: SE (cz=1, cr=-1), w=1/36
            cu=uz-ur; feq=T(1.0/36.0)*ρ*(one(T)+T(3)*cu+T(4.5)*cu*cu-T(1.5)*usq)
            ω_f = (one(T) - τ*inv_r) / (τ + T(0.5))
            Sq = (((-one(T)-ur)*Fr * inv_cs2 / ρ) - ur_over_r) * feq
            Gq = T(1.0/36.0)*((one(T)-uz)*Fz*T(3) + cu*Fz*T(9))
            f[i,j,9] = f9 - ω_f*(f9-feq) + (one(T)-T(0.5)*ω_f)*Sq + guo_ext_pref*Gq
        end
    end
end

"""
    collide_li_axisym_2d!(f, is_solid, τ, Nx, Ny; Fz_body=0.0)

Public function in the high-level simulation driver API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.collide_li_axisym_2d!)
```
"""
function collide_li_axisym_2d!(f, is_solid, τ, Nx, Ny; Fz_body=0.0)
    backend = KernelAbstractions.get_backend(f)
    T = eltype(f)
    kernel! = collide_li_axisym_2d_kernel!(backend)
    kernel!(f, is_solid, T(τ), Ny, T(Fz_body); ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

"""
    run_hagen_poiseuille_2d(; Nz=4, Nr=32, ν=0.1, Fz=1e-5, max_steps=10000, backend, T)

Hagen-Poiseuille pipe flow (axisymmetric). Validates axisymmetric LBM.
Analytical: u_z(r) = Fz/(4ν) * (R² - r²) where R = Nr (the halfway bounce-back
wall at j=Nr sits half a cell beyond the last fluid node r=Nr-0.5).
"""
function run_hagen_poiseuille_2d(; Nz=4, Nr=32, nu=nothing, ν=0.1, Fz=1e-5, max_steps=10000,
                                  backend=KernelAbstractions.CPU(), FT=Float64)
    !isnothing(nu) && (ν = nu)
    Nx, Ny = Nz, Nr
    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=ν, u_lid=0.0, max_steps=max_steps)
    state = initialize_2d(config, FT; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid
    ω = FT(omega(config))

    # Axisymmetric approach: standard BGK + Guo forcing with TWO forces:
    # 1. Body force Fz (drives the flow)
    # 2. Axisym viscous correction Fz_axi = ν/r · ∂uz/∂r (from macroscopic field)
    # Plus mass source -ρ·ur/r (negligible for HP since ur≈0)
    #
    # The correction is computed from the macroscopic velocity field at each step
    # using central FD on the (converging) uz field.

    # Pre-allocate force arrays
    Fz_total = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fr_total = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    for step in 1:max_steps
        stream_periodic_x_axisym_2d!(f_out, f_in, Nx, Ny)

        # Compute axisymmetric correction from current macroscopic field
        # (uses ux from PREVIOUS step — lagged but stable)
        if step > 1
            uz_cpu = Array(ux)  # ux = uz in axisym convention
            Fz_cpu = zeros(FT, Nx, Ny)
            for j in 2:Ny-1, i in 1:Nx
                r = FT(j) - FT(0.5)
                duz_dr = (uz_cpu[i, j+1] - uz_cpu[i, j-1]) / FT(2)
                Fz_cpu[i, j] = FT(Fz) + FT(ν) / r * duz_dr
            end
            # j=1 (near axis, r=0.5): L'Hôpital — lim(r→0) ν/r·∂u/∂r = ν·∂²u/∂r²
            # By symmetry: u(r=-Δr) = u(r=+Δr) → u(j=0) = u(j=2)
            # ∂²u/∂r² ≈ (u[j+1] - 2u[j] + u[j-1]) = 2*(u[2] - u[1])
            for i in 1:Nx
                d2uz_dr2 = FT(2) * (uz_cpu[i, 2] - uz_cpu[i, 1])
                Fz_cpu[i, 1] = FT(Fz) + FT(ν) * d2uz_dr2
            end
            # j=Ny (wall): just body force
            for i in 1:Nx
                Fz_cpu[i, Ny] = FT(Fz)
            end
            copyto!(Fz_total, Fz_cpu)
        else
            # First step: just body force
            fill!(Array(Fz_total), FT(Fz))
            Fz_cpu = fill(FT(Fz), Nx, Ny)
            copyto!(Fz_total, Fz_cpu)
        end

        # Collision with per-node Guo force field
        collide_guo_field_2d!(f_out, is_solid, Fz_total, Fr_total, ω)
        compute_macroscopic_2d!(ρ, ux, uy, f_out)
        f_in, f_out = f_out, f_in
    end

    return (ρ=Array(ρ), uz=Array(ux), ur=Array(uy), config=config)
end

# --- Rayleigh-Plateau pinch-off (axisymmetric VOF-LBM) ---

"""
    run_rp_axisym_2d(; Nz=256, Nr=40, R0=15, λ_ratio=7.0, ε=0.05,
                      σ=0.01, ν=0.05, ρ_l=1.0, ρ_g=0.01,
                      max_steps=10000, output_interval=500, backend, T)

Rayleigh-Plateau capillary instability in axisymmetric geometry.

Coordinates: x=z (axial, periodic), y=r (radial, axis at j=1, wall at j=Nr).
A liquid jet of radius R0 with sinusoidal perturbation R(z) = R0(1-ε·cos(2πz/λ)).
λ > 2πR0 → unstable → pinch-off.

The azimuthal curvature κ₂ = n_r/r drives the instability (absent in 2D planar).
"""
function run_rp_axisym_2d(; Nz=256, Nr=40, R0=15,
                           lambda_ratio=nothing, λ_ratio=7.0,
                           epsilon=nothing, ε=0.05,
                           sigma=nothing, σ=0.01, nu=nothing, ν=0.05,
                           rho_l=nothing, ρ_l=1.0, rho_g=nothing, ρ_g=0.01,
                           max_steps=10000, output_interval=500,
                           backend=KernelAbstractions.CPU(), FT=Float64)
    !isnothing(lambda_ratio) && (λ_ratio = lambda_ratio)
    !isnothing(epsilon) && (ε = epsilon)
    !isnothing(sigma)   && (σ = sigma)
    !isnothing(nu)      && (ν = nu)
    !isnothing(rho_l)   && (ρ_l = rho_l)
    !isnothing(rho_g)   && (ρ_g = rho_g)
    Nx, Ny = Nz, Nr
    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=ν, u_lid=0.0, max_steps=max_steps)
    state = initialize_2d(config, FT; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid
    ω = FT(omega(config))

    # VOF arrays
    C     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    C_new = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    nx_n  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    ny_n  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    κ     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fx_st = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fy_st = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    # Axisymmetric force arrays
    Fz_field = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fr_field = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    λ = FT(λ_ratio * R0)

    # Initialize: axisymmetric jet with perturbation
    C_cpu = zeros(FT, Nx, Ny)
    w = weights(D2Q9())
    f_cpu = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        r = FT(j) - FT(0.5)  # radial position
        R_local = FT(R0) * (one(FT) - FT(ε) * cos(FT(2π) * FT(i - 1) / λ))
        C_cpu[i, j] = FT(0.5) * (one(FT) - tanh((r - R_local) / FT(1.5)))
        ρ_init = C_cpu[i,j] * FT(ρ_l) + (one(FT) - C_cpu[i,j]) * FT(ρ_g)
        for q in 1:9
            f_cpu[i, j, q] = FT(w[q]) * ρ_init
        end
    end
    copyto!(C, C_cpu)
    copyto!(f_in, f_cpu)
    copyto!(f_out, f_cpu)

    r_min_history = FT[]
    times = Int[]

    for step in 1:max_steps
        # 1. Stream (axisym: periodic z, specular axis, wall at Nr)
        stream_periodic_x_axisym_2d!(f_out, f_in, Nx, Ny)

        # 2. Macroscopic
        compute_macroscopic_2d!(ρ, ux, uy, f_out)

        # 3. VOF advection (same as Cartesian for incompressible)
        advect_vof_step!(C, C_new, ux, uy, Nx, Ny)
        copyto!(C, C_new)

        # 4. Curvature: meridional (κ₁) + azimuthal (κ₂ = n_r/r)
        compute_vof_normal_2d!(nx_n, ny_n, C, Nx, Ny)
        compute_hf_curvature_2d!(κ, C, nx_n, ny_n, Nx, Ny)
        add_azimuthal_curvature_2d!(κ, C, ny_n, Ny)

        # 5. Surface tension force
        compute_surface_tension_2d!(Fx_st, Fy_st, κ, C, σ, Nx, Ny)

        # 6. Axisymmetric viscous correction + surface tension
        uz_cpu = Array(ux)
        Fz_cpu = zeros(FT, Nx, Ny)
        Fr_cpu = Array(Fy_st)
        Fx_st_cpu = Array(Fx_st)
        for j in 1:Ny, i in 1:Nx
            r = FT(j) - FT(0.5)
            if j > 1 && j < Ny
                duz_dr = (uz_cpu[i,j+1] - uz_cpu[i,j-1]) / FT(2)
            elseif j == 1
                duz_dr = FT(2) * (uz_cpu[i,2] - uz_cpu[i,1])
                duz_dr = duz_dr  # L'Hôpital: ν/r·∂u/∂r → ν·∂²u/∂r²
            else
                duz_dr = zero(FT)
            end
            axisym_corr = j == 1 ? FT(ν) * duz_dr : FT(ν) / r * duz_dr
            Fz_cpu[i,j] = Fx_st_cpu[i,j] + axisym_corr
            Fr_cpu[i,j] = Fr_cpu[i,j]  # surface tension in r already there
        end
        copyto!(Fz_field, Fz_cpu)
        copyto!(Fr_field, Fr_cpu)

        # 7. Two-phase collision with per-node force
        collide_twophase_2d!(f_out, C, Fz_field, Fr_field, is_solid;
                             ρ_l=ρ_l, ρ_g=ρ_g, ν_l=ν, ν_g=ν)

        f_in, f_out = f_out, f_in

        # Track minimum jet radius
        if step % output_interval == 0 || step == 1
            C_cpu = Array(C)
            r_min_val = FT(Inf)
            for i in 1:Nx
                # Find interface position (where C ≈ 0.5)
                for j in 1:Ny-1
                    if C_cpu[i,j] > 0.5 && C_cpu[i,j+1] <= 0.5
                        # Linear interpolation
                        r_interf = (j - 0.5) + (C_cpu[i,j] - 0.5) / (C_cpu[i,j] - C_cpu[i,j+1])
                        r_min_val = min(r_min_val, r_interf)
                        break
                    end
                end
            end
            push!(r_min_history, r_min_val)
            push!(times, step)
        end
    end

    compute_macroscopic_2d!(ρ, ux, uy, f_in)

    return (ρ=Array(ρ), uz=Array(ux), ur=Array(uy), C=Array(C),
            r_min=r_min_history, times=times, config=config,
            σ=σ, R0=R0, λ=λ, ε=ε)
end

# --- CIJ jet: axisymmetric two-phase with pulsed inlet ---

"""
    run_cij_jet_axisym_2d(; Re=200, We=600, δ=0.02, R0=40, u_lb=0.04, ...)

Continuous InkJet (CIJ) axisymmetric jet simulation: Rayleigh-Plateau breakup
of a liquid jet stimulated by a pulsed velocity inlet.

Coordinates: x=z (axial, inlet at i=1), y=r (radial, axis at j=1).

Reproduces the Basilisk setup from Roche et al.: axisymmetric NS-VOF with
pulsed inlet u(t) = u_lb·(1 + δ·sin(2π·f·t)), where f = u_lb/(7·R0)
corresponds to a perturbation wavelength λ = 7·R0 (near the Rayleigh-Plateau
optimal wavenumber k·R0 ≈ 0.9).

Uses MRT collision for stability at high Re (τ close to 0.5).

Physical parameters mapped through dimensionless numbers:
- Re = u_lb·R0/ν_l  (Reynolds)
- We = ρ_l·u_lb²·R0/σ (Weber)
- ρ_l/ρ_g = ρ_ratio, μ_l/μ_g = μ_ratio

Returns interface snapshots at each output step for comparison with reference data.
"""
function run_cij_jet_axisym_2d(;
        Re=200, We=600, δ=0.02,
        R0=40, u_lb=0.04,
        domain_ratio=80,    # domain length = domain_ratio × R0
        nr_ratio=3,         # radial extent = nr_ratio × R0
        ρ_ratio=10.0,       # ρ_l / ρ_g
        μ_ratio=10.0,       # μ_l / μ_g
        init_length=4,      # initial jet length in R0 units
        max_steps=200_000,
        output_interval=2000,
        output_dir="cij_jet",
        backend=KernelAbstractions.CPU(),
        FT=Float64)

    # --- Derive LBM parameters from dimensionless numbers ---
    ρ_l = FT(1.0)
    ρ_g = ρ_l / FT(ρ_ratio)
    ν_l = FT(u_lb) * FT(R0) / FT(Re)
    # μ_l/μ_g = μ_ratio → ρ_l·ν_l / (ρ_g·ν_g) = μ_ratio → ν_g = ρ_l·ν_l / (ρ_g·μ_ratio)
    ν_g = ρ_l * ν_l / (ρ_g * FT(μ_ratio))
    σ_lb = ρ_l * FT(u_lb)^2 * FT(R0) / FT(We)

    τ_l = FT(3) * ν_l + FT(0.5)
    τ_g = FT(3) * ν_g + FT(0.5)

    @info "CIJ jet LBM parameters" Re We δ R0 u_lb ν_l ν_g σ_lb τ_l τ_g ρ_l ρ_g

    if τ_l < FT(0.505) || τ_g < FT(0.505)
        @warn "Relaxation time close to 0.5 — simulation may be unstable" τ_l τ_g
    end

    # --- Domain ---
    Nz = Int(domain_ratio * R0)   # axial
    Nr = Int(nr_ratio * R0)       # radial
    Nx, Ny = Nz, Nr

    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=Float64(ν_l), u_lid=0.0, max_steps=max_steps)
    state = initialize_2d(config, FT; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid

    # VOF arrays
    C      = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    C_new  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    nx_n   = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    ny_n   = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    cc_field = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    κ      = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fx_st  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fy_st  = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    # Perturbation frequency: λ = 7·R0, period T = 7·R0/u_lb
    T_period = FT(7) * FT(R0) / FT(u_lb)
    f_stim = one(FT) / T_period

    # Pre-compute inlet profiles (constant spatial shape, only amplitude varies)
    ux_inlet = KernelAbstractions.zeros(backend, FT, Ny)
    uy_inlet = KernelAbstractions.zeros(backend, FT, Ny)
    # Normalized radial profile: Hagen-Poiseuille (parabolic) with smooth cutoff
    # u(r) = u_max * max(0, 1 - (r/R0)²) — smooth at r=0, zero gradient at r=R0
    # Blended with a tanh to avoid discontinuity in first derivative
    inlet_profile_cpu = zeros(FT, Ny)
    inlet_vof_cpu = zeros(FT, Ny)
    W_smooth = FT(3)  # interface smoothing width (lattice units)
    for j in 1:Ny
        r = FT(j) - FT(0.5)
        # Smooth envelope (tanh cutoff at R0)
        envelope = FT(0.5) * (one(FT) - tanh((r - FT(R0)) / W_smooth))
        # Flat profile (uniform velocity across jet, like Basilisk flat.c)
        inlet_profile_cpu[j] = envelope
        inlet_vof_cpu[j] = FT(0.5) * (one(FT) - tanh((r - FT(R0)) / FT(1.5)))
    end
    inlet_profile = KernelAbstractions.zeros(backend, FT, Ny)
    copyto!(inlet_profile, inlet_profile_cpu)
    inlet_vof_gpu = KernelAbstractions.zeros(backend, FT, Ny)
    copyto!(inlet_vof_gpu, inlet_vof_cpu)

    # --- Initialize: flat jet of length init_length·R0 ---
    C_cpu = zeros(FT, Nx, Ny)
    f_cpu = zeros(FT, Nx, Ny, 9)
    w = weights(D2Q9())
    L_init = FT(init_length * R0)

    for j in 1:Ny, i in 1:Nx
        r = FT(j) - FT(0.5)
        z = FT(i) - FT(0.5)
        c_r = FT(0.5) * (one(FT) - tanh((r - FT(R0)) / FT(1.5)))
        c_z = FT(0.5) * (one(FT) + tanh((L_init - z) / FT(1.5)))
        C_cpu[i,j] = c_r * c_z
        # Uniform LBM density (density ratio captured via body forces, not distributions)
        ρ_init = ρ_l
        ux_init = C_cpu[i,j] * FT(u_lb)
        for q in 1:9
            cx = FT(velocities_x(D2Q9())[q])
            cu = cx * ux_init
            usq = ux_init^2
            f_cpu[i,j,q] = FT(w[q]) * ρ_init * (one(FT) + FT(3)*cu + FT(4.5)*cu^2 - FT(1.5)*usq)
        end
    end
    copyto!(C, C_cpu)
    copyto!(f_in, f_cpu)
    copyto!(f_out, f_cpu)

    # --- Output setup ---
    mkpath(output_dir)
    pvd = create_pvd(joinpath(output_dir, "cij_jet"))

    # Storage for interface snapshots and breakup tracking
    interface_snapshots = Dict{Int, Vector{NTuple{2,FT}}}()
    breakup_detected = false
    breakup_step = 0

    @info "CIJ jet simulation" Nz Nr T_period max_steps output_dir

    # --- Main loop ---
    for step in 1:max_steps

        # 1. Stream (axisym: non-periodic x, specular j=1, wall j=Ny)
        stream_axisym_inlet_2d!(f_out, f_in, Nx, Ny)

        # 2. Inlet BC: pulsed Zou-He velocity (inside jet only)
        u_t = FT(u_lb) * (one(FT) + FT(δ) * sin(FT(2π) * f_stim * FT(step)))
        # Scale pre-computed profile by time-varying amplitude
        ux_inlet .= inlet_profile .* u_t
        apply_zou_he_west_spatial_2d!(f_out, ux_inlet, uy_inlet, Nx, Ny)

        # 3. Outlet BC: Zou-He pressure (ρ=1 reference, works with MRT stability)
        apply_zou_he_pressure_east_2d!(f_out, Nx, Ny; ρ_out=1.0)

        # 4. Macroscopic
        compute_macroscopic_2d!(ρ, ux, uy, f_out)

        # 5. Impose VOF at inlet (first column, GPU kernel)
        set_vof_west_2d!(C, inlet_vof_gpu)

        # 6. VOF advection (PLIC with MYC normals, CFL sub-stepping)
        advect_vof_plic_step!(C, C_new, nx_n, ny_n, cc_field, ux, uy, Nx, Ny;
                              step=step)
        copyto!(C, C_new)

        # 7. Curvature: meridional (κ₁) + azimuthal (κ₂ = n_r/r)
        compute_vof_normal_2d!(nx_n, ny_n, C, Nx, Ny)
        compute_hf_curvature_2d!(κ, C, nx_n, ny_n, Nx, Ny)
        add_azimuthal_curvature_2d!(κ, C, ny_n, Ny)

        # 8. Surface tension force (CSF: F = σ·κ·∇C)
        compute_surface_tension_2d!(Fx_st, Fy_st, κ, C, σ_lb, Nx, Ny)

        # 9. Axisymmetric viscous correction: ν/r · ∂u_z/∂r
        add_axisym_viscous_correction_2d!(Fx_st, ux, C, ν_l, ν_g, Ny)

        # 10. Two-phase MRT collision (stable at high Re / low τ)
        collide_twophase_mrt_2d!(f_out, C, Fx_st, Fy_st, is_solid;
                                  ρ_l=ρ_l, ρ_g=ρ_g, ν_l=ν_l, ν_g=ν_g)

        # 11. Swap
        f_in, f_out = f_out, f_in

        # --- Output and diagnostics ---
        if step % output_interval == 0
            C_out = Array(C)
            ux_out = Array(ux)

            # Extract interface contour (C = 0.5)
            interface_pts = NTuple{2,FT}[]
            for i in 1:Nx, j in 1:Ny-1
                if (C_out[i,j] - FT(0.5)) * (C_out[i,j+1] - FT(0.5)) < 0
                    frac = (FT(0.5) - C_out[i,j]) / (C_out[i,j+1] - C_out[i,j])
                    push!(interface_pts, (FT(i) - FT(0.5), (FT(j) - FT(0.5)) + frac))
                end
            end
            interface_snapshots[step] = interface_pts

            # Breakup detection: only after jet has had time to grow (> 5 periods)
            if !breakup_detected && step > 5 * Int(round(T_period))
                # Scan behind the jet tip for a pinch-off (gap in liquid column)
                for i in Int(round(5*R0)):Nx-1
                    # Check if a previously filled region has broken
                    max_c = maximum(C_out[i, 1:min(2*R0, Ny)])
                    if max_c < FT(0.01)
                        # Verify it's a real pinch-off: liquid exists both upstream and downstream
                        has_upstream = any(C_out[max(1,i-5*R0):i-1, 1] .> FT(0.5))
                        has_downstream = i + 5 <= Nx && any(C_out[i+1:min(i+5*R0,Nx), 1] .> FT(0.5))
                        if has_upstream && has_downstream
                            breakup_detected = true
                            breakup_step = step
                            @info "Breakup detected" step z=FT(i)-FT(0.5) t_phys=step*u_lb/R0
                            break
                        end
                    end
                end
            end

            # Write VTK (2D: dx = 1.0 in lattice units)
            t_phys = FT(step) * FT(u_lb) / FT(R0)
            write_vtk_to_pvd(pvd,
                joinpath(output_dir, "cij_jet_$(lpad(step, 7, '0'))"),
                Nx, Ny, 1.0,
                Dict("C" => C_out, "rho" => Array(ρ),
                     "ux" => ux_out, "uy" => Array(uy),
                     "kappa" => Array(κ)),
                Float64(t_phys))

            ρ_out_arr = Array(ρ)
            sum_C = sum(C_out)
            @info "Step $step / $max_steps" t_phys n_interface=length(interface_pts) sum_C ρ_range=extrema(ρ_out_arr) ux_range=extrema(ux_out) breakup=breakup_detected
        end

        # Stop 3 periods after breakup (like Basilisk)
        if breakup_detected && step > breakup_step + 3 * Int(round(T_period))
            @info "Stopping: 3 periods after breakup" step
            break
        end
    end

    # Finalize PVD
    vtk_save(pvd)
    compute_macroscopic_2d!(ρ, ux, uy, f_in)

    return (ρ=Array(ρ), uz=Array(ux), ur=Array(uy), C=Array(C),
            interfaces=interface_snapshots,
            breakup_detected=breakup_detected, breakup_step=breakup_step,
            params=(; Re, We, δ, R0, u_lb, ν_l, ν_g, σ_lb, ρ_l, ρ_g, Nz, Nr, T_period))
end

# --- Rayleigh-Plateau with VOF-PLIC + pressure MRT + density-weighted CSF ---

"""
    run_rp_pressure_vof_2d(; Nz, Nr, R0, λ_ratio, ε, σ, ν, ρ_l, ρ_g, ...)

Rayleigh-Plateau instability using VOF-PLIC (sharp, mass-conserving interface)
with pressure-based MRT collision (ρ_lbm ≈ 1, supports high density ratios).

Uses density-weighted CSF (Tryggvason 2011) so that F/ρ is bounded for any
density ratio → stable at ρ_l/ρ_g up to 1000.

Coordinates: x=z (axial, periodic), y=r (radial, axis at j=1, wall at j=Nr).
"""
function run_rp_pressure_vof_2d(; Nz=256, Nr=40, R0=15,
                                  lambda_ratio=nothing, λ_ratio=7.0,
                                  epsilon=nothing, ε=0.05,
                                  sigma=nothing, σ=0.01, nu=nothing, ν=0.05,
                                  rho_l=nothing, ρ_l=1.0, rho_g=nothing, ρ_g=0.001,
                                  max_steps=10000, output_interval=500,
                                  backend=KernelAbstractions.CPU(), FT=Float64)
    !isnothing(lambda_ratio) && (λ_ratio = lambda_ratio)
    !isnothing(epsilon) && (ε = epsilon)
    !isnothing(sigma)   && (σ = sigma)
    !isnothing(nu)      && (ν = nu)
    !isnothing(rho_l)   && (ρ_l = rho_l)
    !isnothing(rho_g)   && (ρ_g = rho_g)
    Nx, Ny = Nz, Nr
    λ = FT(λ_ratio * R0)
    ν_l = FT(ν); ν_g = FT(ν)

    @info "RP pressure-VOF" Nz Nr R0 λ_ratio ε σ ν ρ_l ρ_g

    # Arrays
    C      = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    C_new  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    nx_n   = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    ny_n   = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    cc_field = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    κ      = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fx_st  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fy_st  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    f_in   = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    f_out  = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    p      = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    ux     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    uy     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny)

    # Initialize perturbed cylinder
    C_cpu = zeros(FT, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        r = FT(j) - FT(0.5)
        R_local = FT(R0) * (one(FT) - FT(ε) * cos(FT(2π) * FT(i - 1) / λ))
        C_cpu[i,j] = FT(0.5) * (one(FT) - tanh((r - R_local) / FT(1.5)))
    end
    copyto!(C, C_cpu)

    f_cpu = init_pressure_vof_equilibrium(C_cpu, zeros(FT, Nx, Ny), zeros(FT, Nx, Ny),
                                           ρ_l, ρ_g, FT)
    copyto!(f_in, f_cpu); copyto!(f_out, f_cpu)

    r_min_history = FT[]
    times = Int[]

    for step in 1:max_steps
        # 1. Stream (periodic z, specular axis, wall at Nr)
        stream_periodic_x_axisym_2d!(f_out, f_in, Nx, Ny)

        # 2. Macroscopic (pressure-based: u = (j+F/2)/ρ(C))
        compute_macroscopic_pressure_2d!(p, ux, uy, f_out, C, Fx_st, Fy_st;
                                          ρ_l=ρ_l, ρ_g=ρ_g)

        # 3. VOF-PLIC advection (sharp, mass-conserving)
        advect_vof_plic_step!(C, C_new, nx_n, ny_n, cc_field, ux, uy, Nx, Ny;
                              step=step)
        copyto!(C, C_new)

        # 4. Curvature: meridional (HF) + azimuthal (n_r/r)
        compute_vof_normal_2d!(nx_n, ny_n, C, Nx, Ny)
        compute_hf_curvature_2d!(κ, C, nx_n, ny_n, Nx, Ny)
        add_azimuthal_curvature_2d!(κ, C, ny_n, Ny)

        # 5. Density-weighted CSF (F/ρ bounded → stable at any ρ_ratio)
        ramp = min(FT(step) / FT(500), one(FT))
        compute_surface_tension_weighted_2d!(Fx_st, Fy_st, κ, C, σ * ramp, Nx, Ny;
                                              ρ_l=ρ_l, ρ_g=ρ_g)

        # 6. Axisymmetric viscous correction (skip at very high ρ_ratio for stability)
        if ρ_g > ρ_l * FT(0.01)
            add_axisym_viscous_correction_2d!(Fx_st, ux, C, ν_l, ν_g, Ny)
        end

        # 7. Pressure-based MRT collision
        collide_pressure_vof_mrt_2d!(f_out, C, Fx_st, Fy_st, is_solid;
                                      ρ_l=ρ_l, ρ_g=ρ_g, ν_l=ν_l, ν_g=ν_g)

        f_in, f_out = f_out, f_in

        # Track r_min
        if step % output_interval == 0 || step == 1
            C_cpu = Array(C)
            r_min_val = FT(Inf)
            for i in 1:Nx
                for j in 1:Ny-1
                    if C_cpu[i,j] > 0.5 && C_cpu[i,j+1] <= 0.5
                        r_interf = (FT(j) - FT(0.5)) +
                                   (C_cpu[i,j] - FT(0.5)) / (C_cpu[i,j] - C_cpu[i,j+1])
                        r_min_val = min(r_min_val, r_interf)
                        break
                    end
                end
            end
            push!(r_min_history, r_min_val)
            push!(times, step)
        end
    end

    compute_macroscopic_pressure_2d!(p, ux, uy, f_in, C, Fx_st, Fy_st;
                                      ρ_l=ρ_l, ρ_g=ρ_g)

    return (p=Array(p), uz=Array(ux), ur=Array(uy), C=Array(C),
            r_min=r_min_history, times=times,
            σ=FT(σ), R0=FT(R0), λ=λ, ε=FT(ε), ρ_l=FT(ρ_l), ρ_g=FT(ρ_g))
end

# --- Hybrid: VOF-PLIC (mass) + smoothed C_s (LBM stability) + pressure MRT ---

"""
    run_rp_hybrid_2d(; Nz, Nr, R0, λ_ratio, ε, σ, ν, ρ_l, ρ_g, n_smooth, ...)

Rayleigh-Plateau instability using the hybrid PLIC/smooth approach:
- **C** (sharp, PLIC-advected): source of truth for mass, geometry, κ
- **C_s** (smoothed from C each step): ρ(C_s) for collision/streaming stability

Single distribution f_q, no Allen-Cahn, no phase-field.
Stable at ρ_ratio up to 1000 with exact mass conservation.
"""
function run_rp_hybrid_2d(; Nz=256, Nr=40, R0=15,
                            lambda_ratio=nothing, λ_ratio=7.0,
                            epsilon=nothing, ε=0.05,
                            sigma=nothing, σ=0.01, nu=nothing, ν=0.05,
                            rho_l=nothing, ρ_l=1.0, rho_g=nothing, ρ_g=0.001,
                            gas_model::Symbol=:smooth,
                            n_smooth=3,
                            n_ghost_layers=3, C_ghost_threshold=0.5,
                            ghost_reset_feq::Bool=true, ghost_extrap::Bool=true,
                            mass_correction::Bool=true,
                            W_pf=4.0, tau_g=nothing, τ_g=0.7,
                            max_steps=10000, output_interval=500,
                            backend=KernelAbstractions.CPU(), FT=Float64)
    !isnothing(lambda_ratio) && (λ_ratio = lambda_ratio)
    !isnothing(epsilon) && (ε = epsilon)
    !isnothing(sigma)   && (σ = sigma)
    !isnothing(nu)      && (ν = nu)
    !isnothing(rho_l)   && (ρ_l = rho_l)
    !isnothing(rho_g)   && (ρ_g = rho_g)
    !isnothing(tau_g)   && (τ_g = tau_g)
    gas_model in (:smooth, :ghost, :phasefield) ||
        error("gas_model must be :smooth, :ghost, or :phasefield (got :$gas_model)")

    Nx, Ny = Nz, Nr
    λ = FT(λ_ratio * R0)
    ν_l = FT(ν); ν_g = FT(ν)

    # Effective LBM densities: ghost runs at ρ=1 everywhere
    if gas_model == :ghost
        ρ_eff_l = one(FT); ρ_eff_g = one(FT)
    else
        ρ_eff_l = FT(ρ_l); ρ_eff_g = FT(ρ_g)
    end

    @info "RP hybrid" gas_model Nz Nr R0 λ_ratio ε σ ν ρ_eff_l ρ_eff_g

    # --- Common arrays ---
    C      = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fx_st  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fy_st  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    κ      = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    f_in   = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    f_out  = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    p      = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    ux     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    uy     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny)

    # VOF geometry (smooth/ghost use PLIC)
    C_new    = gas_model != :phasefield ? KernelAbstractions.zeros(backend, FT, Nx, Ny) : nothing
    nx_n     = gas_model != :phasefield ? KernelAbstractions.zeros(backend, FT, Nx, Ny) : nothing
    ny_n     = KernelAbstractions.zeros(backend, FT, Nx, Ny)  # also used for azimuthal κ
    cc_field = gas_model != :phasefield ? KernelAbstractions.zeros(backend, FT, Nx, Ny) : nothing

    # --- Model-specific arrays ---
    C_s = C_tmp = nothing
    ux_tmp = uy_tmp = is_valid = is_valid_new = nothing
    φ = μ_pf = g_in = g_out = Ax = Ay = nothing
    β_pf = κ_pf = zero(FT)

    if gas_model == :smooth
        C_s  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        C_tmp = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    elseif gas_model == :ghost
        ux_tmp     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        uy_tmp     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        is_valid     = KernelAbstractions.zeros(backend, Bool, Nx, Ny)
        is_valid_new = KernelAbstractions.zeros(backend, Bool, Nx, Ny)
        # Smoothed C for CSF gradient only (collision uses sharp C at ρ=1)
        C_s  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        C_tmp = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    elseif gas_model == :phasefield
        φ    = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        μ_pf = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        g_in  = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
        g_out = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
        Ax   = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        Ay   = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        pf = phasefield_params(FT(σ), FT(W_pf))
        β_pf = FT(pf.β); κ_pf = FT(pf.κ)
    end

    # --- Initialize perturbed cylinder ---
    C_cpu = zeros(FT, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        r = FT(j) - FT(0.5)
        R_local = FT(R0) * (one(FT) - FT(ε) * cos(FT(2π) * FT(i - 1) / λ))
        C_cpu[i,j] = FT(0.5) * (one(FT) - tanh((r - R_local) / FT(1.5)))
    end
    copyto!(C, C_cpu)

    # --- Equilibrium initialization ---
    if gas_model == :phasefield
        φ_cpu = FT(2) .* C_cpu .- one(FT)  # C ∈ [0,1] → φ ∈ [-1,1]
        copyto!(φ, φ_cpu)
        g_cpu = init_phasefield_equilibrium(φ_cpu, zeros(FT, Nx, Ny), zeros(FT, Nx, Ny), FT)
        copyto!(g_in, g_cpu); copyto!(g_out, g_cpu)
        f_cpu = init_pressure_equilibrium(φ_cpu, zeros(FT, Nx, Ny), zeros(FT, Nx, Ny),
                                           ρ_eff_l, ρ_eff_g, FT)
        copyto!(f_in, f_cpu); copyto!(f_out, f_cpu)
    elseif gas_model == :smooth
        smooth_vof_2d!(C_s, C, C_tmp; n_passes=n_smooth)
        C_s_cpu = Array(C_s)
        f_cpu = init_pressure_vof_equilibrium(C_s_cpu, zeros(FT, Nx, Ny), zeros(FT, Nx, Ny),
                                               ρ_eff_l, ρ_eff_g, FT)
        copyto!(f_in, f_cpu); copyto!(f_out, f_cpu)
    else  # :ghost
        f_cpu = init_pressure_vof_equilibrium(C_cpu, zeros(FT, Nx, Ny), zeros(FT, Nx, Ny),
                                               ρ_eff_l, ρ_eff_g, FT)
        copyto!(f_in, f_cpu); copyto!(f_out, f_cpu)
    end

    mass_ref = sum(C_cpu)
    r_min_history = FT[]
    times = Int[]

    for step in 1:max_steps
        # 1. Stream
        stream_periodic_x_axisym_2d!(f_out, f_in, Nx, Ny)
        if gas_model == :phasefield
            stream_periodic_x_axisym_2d!(g_out, g_in, Nx, Ny)
        end

        # 2. Interface update + macroscopic
        if gas_model == :smooth
            smooth_vof_2d!(C_s, C, C_tmp; n_passes=n_smooth)
            compute_macroscopic_pressure_2d!(p, ux, uy, f_out, C_s, Fx_st, Fy_st;
                                              ρ_l=ρ_eff_l, ρ_g=ρ_eff_g)
        elseif gas_model == :ghost
            # Smoothed C for CSF gradient (sharp C is too narrow for HF curvature)
            smooth_vof_2d!(C_s, C, C_tmp; n_passes=n_smooth)
            compute_macroscopic_pressure_2d!(p, ux, uy, f_out, C, Fx_st, Fy_st;
                                              ρ_l=one(FT), ρ_g=one(FT))
            if ghost_extrap
                extrapolate_velocity_ghost_2d!(ux, uy, C, ux_tmp, uy_tmp,
                                                is_valid, is_valid_new;
                                                C_threshold=FT(C_ghost_threshold),
                                                n_layers=n_ghost_layers)
            end
        else  # :phasefield
            compute_phi_2d!(φ, g_out)
            compute_vof_from_phi_2d!(C, φ)
            compute_macroscopic_phasefield_2d!(p, ux, uy, f_out, φ, Fx_st, Fy_st;
                                                ρ_l=ρ_eff_l, ρ_g=ρ_eff_g)
        end

        # 3. VOF advection (PLIC, sharp — skip for phasefield)
        if gas_model != :phasefield
            advect_vof_plic_step!(C, C_new, nx_n, ny_n, cc_field, ux, uy, Nx, Ny;
                                  step=step)
            copyto!(C, C_new)
            if mass_correction
                correct_mass_2d!(C, mass_ref)
            end
        end

        # 4. Surface tension
        ramp = min(FT(step) / FT(500), one(FT))
        if gas_model == :phasefield
            compute_chemical_potential_2d!(μ_pf, φ, β_pf, κ_pf)
            add_azimuthal_chemical_potential_2d!(μ_pf, φ, κ_pf, Ny)
            compute_phasefield_force_2d!(Fx_st, Fy_st, μ_pf, φ)
            if ramp < one(FT)
                Fx_st .*= ramp; Fy_st .*= ramp
            end
        else
            compute_vof_normal_2d!(nx_n, ny_n, C, Nx, Ny)
            compute_hf_curvature_2d!(κ, C, nx_n, ny_n, Nx, Ny)
            add_azimuthal_curvature_2d!(κ, C, ny_n, Ny)
            C_grad = (gas_model == :smooth || gas_model == :ghost) ? C_s : C
            compute_surface_tension_weighted_2d!(Fx_st, Fy_st, κ, C_grad, σ * ramp, Nx, Ny;
                                                  ρ_l=ρ_eff_l, ρ_g=ρ_eff_g)
        end

        # 5. Axisymmetric viscous correction
        C_ref = gas_model == :smooth ? C_s : C
        add_axisym_viscous_weighted_2d!(Fx_st, ux, C_ref, ν_l, ν_g, ρ_eff_l, ρ_eff_g, Ny)

        # 6. Collision
        if gas_model == :phasefield
            compute_antidiffusion_flux_2d!(Ax, Ay, φ)
            collide_allen_cahn_2d!(g_out, ux, uy, Ax, Ay; τ_g=FT(τ_g), W=FT(W_pf))
            add_azimuthal_allen_cahn_source_2d!(g_out, φ; τ_g=FT(τ_g))
            collide_pressure_phasefield_mrt_2d!(f_out, φ, Fx_st, Fy_st, is_solid;
                                                  ρ_l=ρ_eff_l, ρ_g=ρ_eff_g,
                                                  ν_l=ν_l, ν_g=ν_g)
        else
            C_coll = gas_model == :smooth ? C_s : C
            collide_pressure_vof_mrt_2d!(f_out, C_coll, Fx_st, Fy_st, is_solid;
                                          ρ_l=ρ_eff_l, ρ_g=ρ_eff_g,
                                          ν_l=ν_l, ν_g=ν_g)
        end

        # Ghost: reset gas f to equilibrium with extrapolated velocity
        if gas_model == :ghost && ghost_reset_feq
            reset_feq_ghost_2d!(f_out, ux, uy, C; C_threshold=FT(C_ghost_threshold))
        end

        f_in, f_out = f_out, f_in
        if gas_model == :phasefield
            g_in, g_out = g_out, g_in
        end

        # Track r_min from C
        if step % output_interval == 0 || step == 1
            C_cpu = Array(C)
            r_min_val = FT(Inf)
            for i in 1:Nx
                for j in 1:Ny-1
                    if C_cpu[i,j] > 0.5 && C_cpu[i,j+1] <= 0.5
                        r_interf = (FT(j) - FT(0.5)) +
                                   (C_cpu[i,j] - FT(0.5)) / (C_cpu[i,j] - C_cpu[i,j+1])
                        r_min_val = min(r_min_val, r_interf)
                        break
                    end
                end
            end
            push!(r_min_history, r_min_val)
            push!(times, step)
        end
    end

    # Final macroscopic
    if gas_model == :phasefield
        compute_macroscopic_phasefield_2d!(p, ux, uy, f_in, φ, Fx_st, Fy_st;
                                            ρ_l=ρ_eff_l, ρ_g=ρ_eff_g)
    elseif gas_model == :smooth
        compute_macroscopic_pressure_2d!(p, ux, uy, f_in, C_s, Fx_st, Fy_st;
                                          ρ_l=ρ_eff_l, ρ_g=ρ_eff_g)
    else  # :ghost
        compute_macroscopic_pressure_2d!(p, ux, uy, f_in, C, Fx_st, Fy_st;
                                          ρ_l=one(FT), ρ_g=one(FT))
    end

    return (p=Array(p), uz=Array(ux), ur=Array(uy),
            C=Array(C), C_s=isnothing(C_s) ? nothing : Array(C_s),
            r_min=r_min_history, times=times,
            gas_model=gas_model,
            σ=FT(σ), R0=FT(R0), λ=λ, ε=FT(ε), ρ_l=FT(ρ_eff_l), ρ_g=FT(ρ_eff_g))
end

# --- CIJ jet: hybrid PLIC/smooth + pressure MRT ---

"""
    run_cij_jet_hybrid_2d(; gas_model=:smooth, Re, We, δ, R0, u_lb, ρ_ratio, μ_ratio, ...)

CIJ jet breakup with selectable gas model:
- `:smooth` — PLIC + smoothed VOF for collision/forces (default)
- `:ghost`  — PLIC + ghost fluid (ρ_lbm=1 everywhere, gas velocity extrapolated)
- `:phasefield` — Allen-Cahn diffuse interface (second LBM distribution)

All models use pressure-based MRT collision and axisymmetric geometry.
"""
function run_cij_jet_hybrid_2d(;
        Re=200, We=600, delta=nothing, δ=0.02,
        R0=40, u_lb=0.04,
        domain_ratio=80, nr_ratio=3,
        rho_ratio=nothing, ρ_ratio=1000.0,
        mu_ratio=nothing, μ_ratio=10.0,
        gas_model::Symbol=:smooth,
        n_smooth=3,
        n_ghost_layers=5, C_ghost_threshold=0.5,
        W_pf=4.0, tau_ac=nothing, τ_ac=0.7,
        init_length=4, max_steps=200_000,
        output_interval=2000,
        output_dir="cij_jet_hybrid",
        backend=KernelAbstractions.CPU(), FT=Float64)
    !isnothing(delta)     && (δ = delta)
    !isnothing(rho_ratio) && (ρ_ratio = rho_ratio)
    !isnothing(mu_ratio)  && (μ_ratio = mu_ratio)
    !isnothing(tau_ac)    && (τ_ac = tau_ac)

    gas_model in (:smooth, :ghost, :phasefield) ||
        error("gas_model must be :smooth, :ghost, or :phasefield (got :$gas_model)")

    # --- Derive LBM parameters ---
    ρ_l = FT(1.0)
    σ_lb = ρ_l * FT(u_lb)^2 * FT(R0) / FT(We)
    ν_l = FT(u_lb) * FT(R0) / FT(Re)

    if gas_model == :ghost
        ρ_eff_l = one(FT); ρ_eff_g = one(FT)
        ν_g = ν_l / FT(μ_ratio)  # μ_g/ρ_lbm = (μ_l/μ_ratio)/1
    else
        ρ_eff_l = ρ_l
        ρ_eff_g = ρ_l / FT(ρ_ratio)
        ν_g = ρ_l * ν_l / (ρ_eff_g * FT(μ_ratio))
    end

    τ_l = FT(3) * ν_l + FT(0.5)
    τ_g_visc = FT(3) * ν_g + FT(0.5)

    @info "CIJ jet" gas_model Re We δ R0 u_lb ν_l ν_g σ_lb τ_l τ_g_visc ρ_eff_l ρ_eff_g

    if τ_l < FT(0.505) || τ_g_visc < FT(0.505)
        @warn "Relaxation time close to 0.5" τ_l τ_g_visc
    end

    # --- Domain ---
    Nz = Int(domain_ratio * R0)
    Nr = Int(nr_ratio * R0)
    Nx, Ny = Nz, Nr

    # --- Common arrays ---
    C      = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fx_st  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Fy_st  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    κ      = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    f_in   = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    f_out  = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    p      = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    ux     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    uy     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny)
    ny_n   = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    # VOF geometry (smooth/ghost use PLIC)
    C_new    = gas_model != :phasefield ? KernelAbstractions.zeros(backend, FT, Nx, Ny) : nothing
    nx_n     = gas_model != :phasefield ? KernelAbstractions.zeros(backend, FT, Nx, Ny) : nothing
    cc_field = gas_model != :phasefield ? KernelAbstractions.zeros(backend, FT, Nx, Ny) : nothing

    # --- Model-specific arrays ---
    C_s = C_tmp = nothing
    ux_tmp = uy_tmp = is_valid = is_valid_new = nothing
    φ = μ_pf = g_in = g_out = Ax = Ay = nothing
    β_pf = κ_pf = zero(FT)

    if gas_model == :smooth
        C_s  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        C_tmp = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    elseif gas_model == :ghost
        ux_tmp     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        uy_tmp     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        is_valid     = KernelAbstractions.zeros(backend, Bool, Nx, Ny)
        is_valid_new = KernelAbstractions.zeros(backend, Bool, Nx, Ny)
        # Smoothed C for CSF gradient only (collision uses sharp C at ρ=1)
        C_s  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        C_tmp = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    elseif gas_model == :phasefield
        φ    = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        μ_pf = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        g_in  = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
        g_out = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
        Ax   = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        Ay   = KernelAbstractions.zeros(backend, FT, Nx, Ny)
        pf = phasefield_params(σ_lb, FT(W_pf))
        β_pf = FT(pf.β); κ_pf = FT(pf.κ)
    end

    # Perturbation frequency
    T_period = FT(7) * FT(R0) / FT(u_lb)
    f_stim = one(FT) / T_period

    # --- Inlet profiles ---
    W_smooth = FT(3)
    inlet_profile_cpu = zeros(FT, Ny)
    for j in 1:Ny
        r = FT(j) - FT(0.5)
        inlet_profile_cpu[j] = FT(0.5) * (one(FT) - tanh((r - FT(R0)) / W_smooth))
    end
    inlet_profile = KernelAbstractions.zeros(backend, FT, Ny)
    copyto!(inlet_profile, inlet_profile_cpu)
    ux_inlet = KernelAbstractions.zeros(backend, FT, Ny)
    uy_inlet = KernelAbstractions.zeros(backend, FT, Ny)

    # VOF inlet profile (smooth/ghost)
    inlet_vof_gpu = nothing
    if gas_model != :phasefield
        inlet_vof_cpu = zeros(FT, Ny)
        for j in 1:Ny
            r = FT(j) - FT(0.5)
            inlet_vof_cpu[j] = FT(0.5) * (one(FT) - tanh((r - FT(R0)) / FT(1.5)))
        end
        inlet_vof_gpu = KernelAbstractions.zeros(backend, FT, Ny)
        copyto!(inlet_vof_gpu, inlet_vof_cpu)
    end

    # Phasefield inlet profile
    inlet_phi_gpu = nothing
    if gas_model == :phasefield
        inlet_phi_cpu = zeros(FT, Ny)
        for j in 1:Ny
            r = FT(j) - FT(0.5)
            inlet_phi_cpu[j] = -tanh((r - FT(R0)) / FT(W_pf))
        end
        inlet_phi_gpu = KernelAbstractions.zeros(backend, FT, Ny)
        copyto!(inlet_phi_gpu, inlet_phi_cpu)
    end

    # --- Initialize: flat jet ---
    L_init = FT(init_length * R0)

    if gas_model == :phasefield
        φ_cpu = zeros(FT, Nx, Ny)
        ux_cpu = zeros(FT, Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            r = FT(j) - FT(0.5); z = FT(i) - FT(0.5)
            φ_r = -tanh((r - FT(R0)) / FT(W_pf))
            φ_z = tanh((L_init - z) / FT(W_pf))
            φ_cpu[i,j] = min(φ_r, φ_z)
            ux_cpu[i,j] = max(zero(FT), (one(FT) + φ_cpu[i,j]) / FT(2)) * FT(u_lb)
        end
        copyto!(φ, φ_cpu)
        C_cpu = (φ_cpu .+ one(FT)) ./ FT(2)
        copyto!(C, C_cpu)
        f_cpu = init_pressure_equilibrium(φ_cpu, ux_cpu, zeros(FT, Nx, Ny),
                                           ρ_eff_l, ρ_eff_g, FT)
        g_cpu = init_phasefield_equilibrium(φ_cpu, ux_cpu, zeros(FT, Nx, Ny), FT)
        copyto!(f_in, f_cpu); copyto!(f_out, f_cpu)
        copyto!(g_in, g_cpu); copyto!(g_out, g_cpu)
    else
        C_cpu = zeros(FT, Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            r = FT(j) - FT(0.5); z = FT(i) - FT(0.5)
            c_r = FT(0.5) * (one(FT) - tanh((r - FT(R0)) / FT(1.5)))
            c_z = FT(0.5) * (one(FT) + tanh((L_init - z) / FT(1.5)))
            C_cpu[i,j] = c_r * c_z
        end
        copyto!(C, C_cpu)

        ux_cpu = zeros(FT, Nx, Ny)
        if gas_model == :smooth
            smooth_vof_2d!(C_s, C, C_tmp; n_passes=n_smooth)
            C_init_cpu = Array(C_s)
        else  # :ghost — use sharp C
            C_init_cpu = copy(C_cpu)
        end
        for j in 1:Ny, i in 1:Nx
            ux_cpu[i,j] = C_init_cpu[i,j] * FT(u_lb)
        end
        f_cpu = init_pressure_vof_equilibrium(C_init_cpu, ux_cpu, zeros(FT, Nx, Ny),
                                               ρ_eff_l, ρ_eff_g, FT)
        copyto!(f_in, f_cpu); copyto!(f_out, f_cpu)
    end

    # --- Output & tracking ---
    mkpath(output_dir)
    pvd = create_pvd(joinpath(output_dir, "cij_$(gas_model)"))
    interface_snapshots = Dict{Int, Vector{NTuple{2,FT}}}()
    breakup_detected = false
    breakup_step = 0

    @info "CIJ simulation" gas_model Nz Nr T_period max_steps output_dir

    # --- Main loop ---
    for step in 1:max_steps

        # 1. Stream
        stream_axisym_inlet_2d!(f_out, f_in, Nx, Ny)
        if gas_model == :phasefield
            stream_axisym_inlet_2d!(g_out, g_in, Nx, Ny)
        end

        # 2. Inlet BC: pulsed Zou-He velocity
        u_t = FT(u_lb) * (one(FT) + FT(δ) * sin(FT(2π) * f_stim * FT(step)))
        ux_inlet .= inlet_profile .* u_t
        apply_zou_he_west_spatial_2d!(f_out, ux_inlet, uy_inlet, Nx, Ny)

        # 3. Phasefield inlet BC
        if gas_model == :phasefield
            set_phasefield_west_2d!(g_out, inlet_phi_gpu, ux_inlet)
        end

        # 4. Outlet BC
        apply_zou_he_pressure_east_2d!(f_out, Nx, Ny; ρ_out=1.0)
        if gas_model == :phasefield
            extrapolate_phasefield_east_2d!(g_out, Nx, Ny)
        end

        # 5. Interface update + macroscopic
        if gas_model == :smooth
            smooth_vof_2d!(C_s, C, C_tmp; n_passes=n_smooth)
            compute_macroscopic_pressure_2d!(p, ux, uy, f_out, C_s, Fx_st, Fy_st;
                                              ρ_l=ρ_eff_l, ρ_g=ρ_eff_g)
        elseif gas_model == :ghost
            smooth_vof_2d!(C_s, C, C_tmp; n_passes=n_smooth)
            compute_macroscopic_pressure_2d!(p, ux, uy, f_out, C, Fx_st, Fy_st;
                                              ρ_l=one(FT), ρ_g=one(FT))
            extrapolate_velocity_ghost_2d!(ux, uy, C, ux_tmp, uy_tmp,
                                            is_valid, is_valid_new;
                                            C_threshold=FT(C_ghost_threshold),
                                            n_layers=n_ghost_layers)
        else  # :phasefield
            compute_phi_2d!(φ, g_out)
            compute_vof_from_phi_2d!(C, φ)
            compute_macroscopic_phasefield_2d!(p, ux, uy, f_out, φ, Fx_st, Fy_st;
                                                ρ_l=ρ_eff_l, ρ_g=ρ_eff_g)
        end

        # 6. VOF advection (smooth/ghost only)
        if gas_model != :phasefield
            set_vof_west_2d!(C, inlet_vof_gpu)
            advect_vof_plic_step!(C, C_new, nx_n, ny_n, cc_field, ux, uy, Nx, Ny;
                                  step=step)
            copyto!(C, C_new)
        end

        # 7. Surface tension
        ramp = min(FT(step) / FT(500), one(FT))
        if gas_model == :phasefield
            compute_chemical_potential_2d!(μ_pf, φ, β_pf, κ_pf)
            add_azimuthal_chemical_potential_2d!(μ_pf, φ, κ_pf, Ny)
            compute_phasefield_force_2d!(Fx_st, Fy_st, μ_pf, φ)
            if ramp < one(FT)
                Fx_st .*= ramp; Fy_st .*= ramp
            end
        else
            compute_vof_normal_2d!(nx_n, ny_n, C, Nx, Ny)
            compute_hf_curvature_2d!(κ, C, nx_n, ny_n, Nx, Ny)
            add_azimuthal_curvature_2d!(κ, C, ny_n, Ny)
            C_grad = (gas_model == :smooth || gas_model == :ghost) ? C_s : C
            compute_surface_tension_weighted_2d!(Fx_st, Fy_st, κ, C_grad, σ_lb * ramp, Nx, Ny;
                                                  ρ_l=ρ_eff_l, ρ_g=ρ_eff_g)
        end

        # 8. Axisymmetric viscous correction
        C_ref = gas_model == :smooth ? C_s : C
        add_axisym_viscous_weighted_2d!(Fx_st, ux, C_ref, ν_l, ν_g, ρ_eff_l, ρ_eff_g, Ny)

        # 9. Collision
        if gas_model == :phasefield
            compute_antidiffusion_flux_2d!(Ax, Ay, φ)
            collide_allen_cahn_2d!(g_out, ux, uy, Ax, Ay; τ_g=FT(τ_ac), W=FT(W_pf))
            add_azimuthal_allen_cahn_source_2d!(g_out, φ; τ_g=FT(τ_ac))
            collide_pressure_phasefield_mrt_2d!(f_out, φ, Fx_st, Fy_st, is_solid;
                                                  ρ_l=ρ_eff_l, ρ_g=ρ_eff_g,
                                                  ν_l=ν_l, ν_g=ν_g)
        else
            C_coll = gas_model == :smooth ? C_s : C
            collide_pressure_vof_mrt_2d!(f_out, C_coll, Fx_st, Fy_st, is_solid;
                                          ρ_l=ρ_eff_l, ρ_g=ρ_eff_g,
                                          ν_l=ν_l, ν_g=ν_g)
        end

        # Ghost: reset gas f to equilibrium
        if gas_model == :ghost
            reset_feq_ghost_2d!(f_out, ux, uy, C; C_threshold=FT(C_ghost_threshold))
        end

        # 10. Swap
        f_in, f_out = f_out, f_in
        if gas_model == :phasefield
            g_in, g_out = g_out, g_in
        end

        # --- Output ---
        if step % output_interval == 0
            C_out = Array(C)
            ux_out = Array(ux)

            # Interface contour (C = 0.5)
            interface_pts = NTuple{2,FT}[]
            for i in 1:Nx, j in 1:Ny-1
                if (C_out[i,j] - FT(0.5)) * (C_out[i,j+1] - FT(0.5)) < 0
                    frac = (FT(0.5) - C_out[i,j]) / (C_out[i,j+1] - C_out[i,j])
                    push!(interface_pts, (FT(i) - FT(0.5), (FT(j) - FT(0.5)) + frac))
                end
            end
            interface_snapshots[step] = interface_pts

            # Breakup detection
            if !breakup_detected && step > 5 * Int(round(T_period))
                for i in Int(round(5*R0)):Nx-1
                    max_c = maximum(C_out[i, 1:min(2*R0, Ny)])
                    if max_c < FT(0.01)
                        has_upstream = any(C_out[max(1,i-5*R0):i-1, 1] .> FT(0.5))
                        has_downstream = i + 5 <= Nx && any(C_out[i+1:min(i+5*R0,Nx), 1] .> FT(0.5))
                        if has_upstream && has_downstream
                            breakup_detected = true
                            breakup_step = step
                            @info "Breakup detected" step z=FT(i)-FT(0.5) t_phys=step*u_lb/R0
                            break
                        end
                    end
                end
            end

            t_phys = FT(step) * FT(u_lb) / FT(R0)
            write_vtk_to_pvd(pvd,
                joinpath(output_dir, "cij_$(gas_model)_$(lpad(step, 7, '0'))"),
                Nx, Ny, 1.0,
                Dict("C" => C_out, "p" => Array(p),
                     "ux" => ux_out, "uy" => Array(uy),
                     "kappa" => Array(κ)),
                Float64(t_phys))

            sum_C = sum(C_out)
            @info "Step $step / $max_steps" t_phys n_interface=length(interface_pts) sum_C ux_range=extrema(ux_out) breakup=breakup_detected
        end

        if breakup_detected && step > breakup_step + 3 * Int(round(T_period))
            @info "Stopping: 3 periods after breakup" step
            break
        end
    end

    vtk_save(pvd)

    # Final macroscopic
    if gas_model == :phasefield
        compute_phi_2d!(φ, g_in)
        compute_chemical_potential_2d!(μ_pf, φ, β_pf, κ_pf)
        compute_phasefield_force_2d!(Fx_st, Fy_st, μ_pf, φ)
        compute_macroscopic_phasefield_2d!(p, ux, uy, f_in, φ, Fx_st, Fy_st;
                                            ρ_l=ρ_eff_l, ρ_g=ρ_eff_g)
    elseif gas_model == :smooth
        smooth_vof_2d!(C_s, C, C_tmp; n_passes=n_smooth)
        compute_macroscopic_pressure_2d!(p, ux, uy, f_in, C_s, Fx_st, Fy_st;
                                          ρ_l=ρ_eff_l, ρ_g=ρ_eff_g)
    else  # :ghost
        compute_macroscopic_pressure_2d!(p, ux, uy, f_in, C, Fx_st, Fy_st;
                                          ρ_l=one(FT), ρ_g=one(FT))
    end

    return (p=Array(p), uz=Array(ux), ur=Array(uy),
            C=Array(C), C_s=isnothing(C_s) ? nothing : Array(C_s),
            interfaces=interface_snapshots,
            breakup_detected=breakup_detected, breakup_step=breakup_step,
            params=(; Re, We, δ, R0, u_lb, ν_l, ν_g, σ_lb, gas_model,
                    ρ_l=ρ_eff_l, ρ_g=ρ_eff_g, Nz, Nr, T_period))
end
