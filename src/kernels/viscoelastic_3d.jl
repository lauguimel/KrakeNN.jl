using KernelAbstractions

# ============================================================
# Polymeric stress divergence → Guo body force (3D)
# ============================================================
#
# 3D analogue of `compute_polymeric_force_2d!` (viscoelastic_2d.jl:16-46).
# Builds the first-moment body force F_poly = ∇·τ_p over the 6-component
# symmetric polymer stress (τ_xx, τ_xy, τ_xz, τ_yy, τ_yz, τ_zz):
#
#   Fx = ∂τ_xx/∂x + ∂τ_xy/∂y + ∂τ_xz/∂z
#   Fy = ∂τ_xy/∂x + ∂τ_yy/∂y + ∂τ_yz/∂z
#   Fz = ∂τ_xz/∂x + ∂τ_yz/∂y + ∂τ_zz/∂z
#
# Central differences. Per-axis x / z treatment of the boundary cells:
# `periodic_*` wraps (channel / duct topology); `wall_*` uses the same
# second-order one-sided difference as the y walls (y is never periodic);
# otherwise the stencil clamps, which returns (τ₂ − τ₁)/2 at the first cell.
# The clamp is what an open face wants when the conformation there is
# reset or copied (inflow C = I, zero-gradient outflow): a one-sided
# difference would amplify that jump. This is the VALIDATED 2D production
# coupling (cylinder cut-link, <1% vs RheoTool) ported to 3D: the polymer
# enters the momentum equation EXACTLY ONCE as a Guo body force at the
# solvent rate ω_s, with NO (1±ω/2) denominator and lattice viscosity = ν_s
# (bsd = 0). It replaces the standalone re-relaxed `apply_hermite_source_3d!`.

# ∂τ/∂x at (i, j, k): wrapped central difference when periodic; at a wall,
# second-order one-sided at i = 1 and i = Nx; otherwise clamped.
@inline function _polymer_force_dx_3d(tau, i, j, k, Nx, periodic, wall, half, T)
    if periodic
        ip = i < Nx ? i + 1 : 1
        im = i > 1 ? i - 1 : Nx
        return (tau[ip, j, k] - tau[im, j, k]) * half
    elseif wall && i == 1
        return (-T(3) * tau[1, j, k] + T(4) * tau[2, j, k] - tau[3, j, k]) * half
    elseif wall && i == Nx
        return (T(3) * tau[Nx, j, k] - T(4) * tau[Nx - 1, j, k] + tau[Nx - 2, j, k]) * half
    else
        ip = i < Nx ? i + 1 : Nx
        im = i > 1 ? i - 1 : 1
        return (tau[ip, j, k] - tau[im, j, k]) * half
    end
end

# ∂τ/∂z, same rule as `_polymer_force_dx_3d`.
@inline function _polymer_force_dz_3d(tau, i, j, k, Nz, periodic, wall, half, T)
    if periodic
        kp = k < Nz ? k + 1 : 1
        km = k > 1 ? k - 1 : Nz
        return (tau[i, j, kp] - tau[i, j, km]) * half
    elseif wall && k == 1
        return (-T(3) * tau[i, j, 1] + T(4) * tau[i, j, 2] - tau[i, j, 3]) * half
    elseif wall && k == Nz
        return (T(3) * tau[i, j, Nz] - T(4) * tau[i, j, Nz - 1] + tau[i, j, Nz - 2]) * half
    else
        kp = k < Nz ? k + 1 : Nz
        km = k > 1 ? k - 1 : 1
        return (tau[i, j, kp] - tau[i, j, km]) * half
    end
end

@kernel function compute_polymeric_force_3d_kernel!(Fx_p, Fy_p, Fz_p,
                                                      @Const(tau_xx), @Const(tau_xy),
                                                      @Const(tau_xz), @Const(tau_yy),
                                                      @Const(tau_yz), @Const(tau_zz),
                                                      Nx, Ny, Nz,
                                                      periodic_x, periodic_z,
                                                      wall_x, wall_z)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        T = eltype(Fx_p)

        # x/z: wrap when periodic, 2nd-order ONE-SIDED at a wall, else clamp
        # (`_polymer_force_dx_3d` / `_dz_3d`). y wall rows use the same
        # one-sided difference instead of a degenerate clamped central
        # difference (the wall-aware ∂/∂y the forensic recipe requires —
        # a raw central diff at j=1/Ny injects a spurious near-wall force that
        # biases the constitutive self-consistency, since the bulk ∇·τ_p ≈ 0).

        # ∂τ/∂y with wall-aware one-sided 2nd-order at the no-slip y-faces.
        # interior: (τ[j+1]-τ[j-1])/2 ; j=1: (-3τ₁+4τ₂-τ₃)/2 ; j=Ny: (3τ_Ny-4τ_{Ny-1}+τ_{Ny-2})/2
        half = T(0.5)
        dy_xy = j == 1  ? (-T(3)*tau_xy[i,1,k] + T(4)*tau_xy[i,2,k] - tau_xy[i,3,k]) * half :
                j == Ny ? ( T(3)*tau_xy[i,Ny,k] - T(4)*tau_xy[i,Ny-1,k] + tau_xy[i,Ny-2,k]) * half :
                          (tau_xy[i,j+1,k] - tau_xy[i,j-1,k]) * half
        dy_yy = j == 1  ? (-T(3)*tau_yy[i,1,k] + T(4)*tau_yy[i,2,k] - tau_yy[i,3,k]) * half :
                j == Ny ? ( T(3)*tau_yy[i,Ny,k] - T(4)*tau_yy[i,Ny-1,k] + tau_yy[i,Ny-2,k]) * half :
                          (tau_yy[i,j+1,k] - tau_yy[i,j-1,k]) * half
        dy_yz = j == 1  ? (-T(3)*tau_yz[i,1,k] + T(4)*tau_yz[i,2,k] - tau_yz[i,3,k]) * half :
                j == Ny ? ( T(3)*tau_yz[i,Ny,k] - T(4)*tau_yz[i,Ny-1,k] + tau_yz[i,Ny-2,k]) * half :
                          (tau_yz[i,j+1,k] - tau_yz[i,j-1,k]) * half

        # Fx = ∂τ_xx/∂x + ∂τ_xy/∂y + ∂τ_xz/∂z
        Fx_p[i,j,k] = _polymer_force_dx_3d(tau_xx, i, j, k, Nx, periodic_x, wall_x, half, T) + dy_xy +
                      _polymer_force_dz_3d(tau_xz, i, j, k, Nz, periodic_z, wall_z, half, T)

        # Fy = ∂τ_xy/∂x + ∂τ_yy/∂y + ∂τ_yz/∂z
        Fy_p[i,j,k] = _polymer_force_dx_3d(tau_xy, i, j, k, Nx, periodic_x, wall_x, half, T) + dy_yy +
                      _polymer_force_dz_3d(tau_yz, i, j, k, Nz, periodic_z, wall_z, half, T)

        # Fz = ∂τ_xz/∂x + ∂τ_yz/∂y + ∂τ_zz/∂z
        Fz_p[i,j,k] = _polymer_force_dx_3d(tau_xz, i, j, k, Nx, periodic_x, wall_x, half, T) + dy_yz +
                      _polymer_force_dz_3d(tau_zz, i, j, k, Nz, periodic_z, wall_z, half, T)
    end
end

"""
    compute_polymeric_force_3d!(Fx_p, Fy_p, Fz_p,
                                 tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz;
                                 periodic_x=true, periodic_z=true,
                                 wall_x=false, wall_z=false)

Compute the 3D polymeric body force `F_poly = ∇·τ_p` (first moment) from the
6-component symmetric polymer stress, for the Guo coupling. The y faces are
walls: their boundary cells use a second-order one-sided difference. In x and
z, `periodic_*` wraps the neighbour stencil (channel / duct); `wall_*` selects
the same one-sided difference as the y walls, for a no-slip face in that
direction; a face that is neither (an inflow or outflow) keeps a clamped
central difference, which does not amplify the jump left by a conformation
reset or copy there. 3D counterpart of `compute_polymeric_force_2d!`.
"""
function compute_polymeric_force_3d!(Fx_p, Fy_p, Fz_p,
                                      tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz;
                                      periodic_x::Bool=true, periodic_z::Bool=true,
                                      wall_x::Bool=false, wall_z::Bool=false)
    backend = KernelAbstractions.get_backend(Fx_p)
    Nx, Ny, Nz = size(Fx_p)
    (periodic_x && wall_x) && throw(ArgumentError("x cannot be both periodic and a wall"))
    (periodic_z && wall_z) && throw(ArgumentError("z cannot be both periodic and a wall"))
    Ny >= 3 || throw(ArgumentError("the one-sided y-wall stencil needs Ny ≥ 3 (Ny = $Ny)"))
    (wall_x && Nx < 3) &&
        throw(ArgumentError("a wall in x needs Nx ≥ 3 for the one-sided stencil (Nx = $Nx)"))
    (wall_z && Nz < 3) &&
        throw(ArgumentError("a wall in z needs Nz ≥ 3 for the one-sided stencil (Nz = $Nz)"))
    kernel! = compute_polymeric_force_3d_kernel!(backend)
    kernel!(Fx_p, Fy_p, Fz_p,
            tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz,
            Nx, Ny, Nz, periodic_x, periodic_z, wall_x, wall_z; ndrange=(Nx, Ny, Nz))
    KernelAbstractions.synchronize(backend)
end

# --- 3D Hermite stress source (D3Q19) for viscoelastic post-collision ---
#
# 3D port of `apply_hermite_source_2d!` (Liu et al. 2025, Eq. 25 with the
# post-collision CE half-step correction).
#
# T_q = -s_plus · (9/2) / (1 − s_plus/2) · w_q ·
#         [(c_qx² − cs²)·τxx + (c_qy² − cs²)·τyy + (c_qz² − cs²)·τzz +
#          2·c_qx·c_qy·τxy + 2·c_qx·c_qz·τxz + 2·c_qy·c_qz·τyz]
#
# with cs² = 1/3 and the standard D3Q19 weights wr = 1/3, wa = 1/18, we = 1/36.

@kernel function apply_hermite_source_3d_kernel!(f, @Const(is_solid), s_plus,
                                                   @Const(tau_p_xx),
                                                   @Const(tau_p_xy),
                                                   @Const(tau_p_xz),
                                                   @Const(tau_p_yy),
                                                   @Const(tau_p_yz),
                                                   @Const(tau_p_zz))
    i, j, k = @index(Global, NTuple)
    @inbounds if !is_solid[i, j, k]
        T = eltype(f)
        txx = tau_p_xx[i,j,k]; txy = tau_p_xy[i,j,k]; txz = tau_p_xz[i,j,k]
        tyy = tau_p_yy[i,j,k]; tyz = tau_p_yz[i,j,k]; tzz = tau_p_zz[i,j,k]
        pre = -s_plus * T(9.0/2.0) / (one(T) - s_plus / T(2))
        cs2 = T(1/3)
        wr = T(1/3); wa = T(1/18); we = T(1/36)
        a = one(T) - cs2          # = 2/3, common 2nd-Hermite axial weight

        # Common combinations used by multiple directions
        s_xyz = txx + tyy + tzz                       # diagonal trace
        diag_x = a*txx - cs2*tyy - cs2*tzz            # +x or −x: q=2/3
        diag_y = -cs2*txx + a*tyy - cs2*tzz           # +y or −y: q=4/5
        diag_z = -cs2*txx - cs2*tyy + a*tzz           # +z or −z: q=6/7
        # Edges share a 3-component diagonal (axial-pair sum) plus a sign
        # on the off-diagonal coupling.
        edge_xy_diag = a*(txx + tyy) - cs2*tzz
        edge_xz_diag = a*(txx + tzz) - cs2*tyy
        edge_yz_diag = a*(tyy + tzz) - cs2*txx

        # Rest population (q=1)
        T1  = pre * wr * (-cs2 * s_xyz)
        # Axial pairs share Hτ within a pair
        T_x = pre * wa * diag_x          # q=2 and q=3
        T_y = pre * wa * diag_y          # q=4, q=5
        T_z = pre * wa * diag_z          # q=6, q=7
        # xy edge group (q=8..11): pair (8, 11) has +2·τxy, pair (9, 10) has −2·τxy
        Txy_p = pre * we * (edge_xy_diag + T(2)*txy)   # q=8, q=11
        Txy_m = pre * we * (edge_xy_diag - T(2)*txy)   # q=9, q=10
        # xz edge group (q=12..15): pair (12, 15) +2·τxz, pair (13, 14) −2·τxz
        Txz_p = pre * we * (edge_xz_diag + T(2)*txz)   # q=12, q=15
        Txz_m = pre * we * (edge_xz_diag - T(2)*txz)   # q=13, q=14
        # yz edge group (q=16..19): pair (16, 19) +2·τyz, pair (17, 18) −2·τyz
        Tyz_p = pre * we * (edge_yz_diag + T(2)*tyz)   # q=16, q=19
        Tyz_m = pre * we * (edge_yz_diag - T(2)*tyz)   # q=17, q=18

        f[i,j,k,1]  += T1
        f[i,j,k,2]  += T_x;  f[i,j,k,3]  += T_x
        f[i,j,k,4]  += T_y;  f[i,j,k,5]  += T_y
        f[i,j,k,6]  += T_z;  f[i,j,k,7]  += T_z
        f[i,j,k,8]  += Txy_p; f[i,j,k,11] += Txy_p
        f[i,j,k,9]  += Txy_m; f[i,j,k,10] += Txy_m
        f[i,j,k,12] += Txz_p; f[i,j,k,15] += Txz_p
        f[i,j,k,13] += Txz_m; f[i,j,k,14] += Txz_m
        f[i,j,k,16] += Tyz_p; f[i,j,k,19] += Tyz_p
        f[i,j,k,17] += Tyz_m; f[i,j,k,18] += Tyz_m
    end
end

"""
    apply_hermite_source_3d!(f, is_solid, s_plus,
                              tau_p_xx, tau_p_xy, tau_p_xz,
                              tau_p_yy, tau_p_yz, tau_p_zz)

Post-collision injection of the 3D Hermite viscoelastic stress source on
D3Q19 populations. 3D port of `apply_hermite_source_2d!`.

Pass `s_plus = 1/(3ν+0.5)` for TRT, or `s_plus = ω` for BGK.
"""
function apply_hermite_source_3d!(f, is_solid, s_plus,
                                    tau_p_xx, tau_p_xy, tau_p_xz,
                                    tau_p_yy, tau_p_yz, tau_p_zz)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny, Nz = size(f, 1), size(f, 2), size(f, 3)
    T = eltype(f)
    kernel! = apply_hermite_source_3d_kernel!(backend)
    kernel!(f, is_solid, T(s_plus),
            tau_p_xx, tau_p_xy, tau_p_xz,
            tau_p_yy, tau_p_yz, tau_p_zz; ndrange=(Nx, Ny, Nz))
    KernelAbstractions.synchronize(backend)
end

# ============================================================
# Polymeric stress update — 3D τ_p = G·(C - I) for Oldroyd-B
# ============================================================
#
# For Oldroyd-B and FENE-P (Peterlin: f(trC) = L²/(L²-trC)) only.
# Other models can dispatch their own `update_polymer_stress_3d!`.

@kernel function _update_polymer_stress_3d_oldroyd_kernel!(tau_xx, tau_xy, tau_xz,
                                                              tau_yy, tau_yz, tau_zz,
                                                              @Const(C_xx), @Const(C_xy),
                                                              @Const(C_xz), @Const(C_yy),
                                                              @Const(C_yz), @Const(C_zz),
                                                              G, L2_fene)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        T = eltype(tau_xx)
        cxx = C_xx[i,j,k]; cyy = C_yy[i,j,k]; czz = C_zz[i,j,k]
        cxy = C_xy[i,j,k]; cxz = C_xz[i,j,k]; cyz = C_yz[i,j,k]
        trC = cxx + cyy + czz
        fene = ifelse(L2_fene > zero(T),
                      L2_fene / max(L2_fene - trC, T(0.01)),
                      one(T))
        tau_xx[i,j,k] = G * fene * (cxx - one(T))
        tau_yy[i,j,k] = G * fene * (cyy - one(T))
        tau_zz[i,j,k] = G * fene * (czz - one(T))
        tau_xy[i,j,k] = G * fene * cxy
        tau_xz[i,j,k] = G * fene * cxz
        tau_yz[i,j,k] = G * fene * cyz
    end
end

"""
    update_polymer_stress_3d!(tau_p_xx, tau_p_xy, tau_p_xz,
                                tau_p_yy, tau_p_yz, tau_p_zz,
                                C_xx, C_xy, C_xz, C_yy, C_yz, C_zz, model)

Compute the 3D polymeric stress from the conformation tensor and an
`AbstractPolymerModel`. Currently implemented for `OldroydBSpec` (and
`LogConfOldroydB` after `psi_to_C` reconstruction). Mirrors the 2D
dispatch in `viscoelastic_spec.jl::update_polymer_stress!`.
"""
function update_polymer_stress_3d!(tau_xx, tau_xy, tau_xz,
                                     tau_yy, tau_yz, tau_zz,
                                     C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
                                     model::OldroydBSpec)
    backend = KernelAbstractions.get_backend(tau_xx)
    Nx, Ny, Nz = size(tau_xx)
    FT = eltype(tau_xx)
    kernel! = _update_polymer_stress_3d_oldroyd_kernel!(backend)
    kernel!(tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz,
            C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
            FT(polymer_modulus(model)), FT(0.0); ndrange=(Nx, Ny, Nz))
    KernelAbstractions.synchronize(backend)
end

function update_polymer_stress_3d!(tau_xx, tau_xy, tau_xz,
                                     tau_yy, tau_yz, tau_zz,
                                     C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
                                     model::LogConfOldroydB)
    backend = KernelAbstractions.get_backend(tau_xx)
    Nx, Ny, Nz = size(tau_xx)
    FT = eltype(tau_xx)
    kernel! = _update_polymer_stress_3d_oldroyd_kernel!(backend)
    kernel!(tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz,
            C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
            FT(polymer_modulus(model)), FT(0.0); ndrange=(Nx, Ny, Nz))
    KernelAbstractions.synchronize(backend)
end

# ============================================================
# FENE-P (Peterlin) polymeric stress  τ_p = G·(f·C − I)
# ============================================================
#
# Peterlin factor f = (L²−3)/(L²−trC), the SAME factor used by the FENE-P
# constitutive source `logconf_source_with_divergence_fenep_3d` so that the
# stress and the relaxation closure are consistent. As L²→∞, f→1 and this
# recovers the Oldroyd-B stress τ_p = G·(C−I). A small floor on (L²−trC)
# keeps the factor finite at the finite-extensibility limit (trC→L²).
@kernel function _update_polymer_stress_3d_fenep_kernel!(tau_xx, tau_xy, tau_xz,
                                                            tau_yy, tau_yz, tau_zz,
                                                            @Const(C_xx), @Const(C_xy),
                                                            @Const(C_xz), @Const(C_yy),
                                                            @Const(C_yz), @Const(C_zz),
                                                            G, L2_fene)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        T = eltype(tau_xx)
        cxx = C_xx[i,j,k]; cyy = C_yy[i,j,k]; czz = C_zz[i,j,k]
        cxy = C_xy[i,j,k]; cxz = C_xz[i,j,k]; cyz = C_yz[i,j,k]
        trC = cxx + cyy + czz
        # Identical f to logconf_source_with_divergence_fenep_3d.
        fene = (L2_fene - T(3)) / max(L2_fene - trC, T(1e-6) * L2_fene)
        tau_xx[i,j,k] = G * (fene * cxx - one(T))
        tau_yy[i,j,k] = G * (fene * cyy - one(T))
        tau_zz[i,j,k] = G * (fene * czz - one(T))
        tau_xy[i,j,k] = G * fene * cxy
        tau_xz[i,j,k] = G * fene * cxz
        tau_yz[i,j,k] = G * fene * cyz
    end
end

function update_polymer_stress_3d!(tau_xx, tau_xy, tau_xz,
                                     tau_yy, tau_yz, tau_zz,
                                     C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
                                     model::LogConfFENEP)
    backend = KernelAbstractions.get_backend(tau_xx)
    Nx, Ny, Nz = size(tau_xx)
    FT = eltype(tau_xx)
    kernel! = _update_polymer_stress_3d_fenep_kernel!(backend)
    kernel!(tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz,
            C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
            FT(polymer_modulus(model)),
            FT(polymer_max_extensibility(model)); ndrange=(Nx, Ny, Nz))
    KernelAbstractions.synchronize(backend)
end

# Giesekus and PTT both reconstruct stress as τ_p = G·(C − I), identical to
# Oldroyd-B — the model nonlinearity lives entirely in the constitutive
# evolution (mobility α / trace multiplier Y(trC)), not in the stress
# closure. Reuse the OB kernel (L2_fene = 0 ⇒ Peterlin factor = 1).
function update_polymer_stress_3d!(tau_xx, tau_xy, tau_xz,
                                     tau_yy, tau_yz, tau_zz,
                                     C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
                                     model::LogConfGiesekus)
    backend = KernelAbstractions.get_backend(tau_xx)
    Nx, Ny, Nz = size(tau_xx)
    FT = eltype(tau_xx)
    kernel! = _update_polymer_stress_3d_oldroyd_kernel!(backend)
    kernel!(tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz,
            C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
            FT(polymer_modulus(model)), FT(0.0); ndrange=(Nx, Ny, Nz))
    KernelAbstractions.synchronize(backend)
end

function update_polymer_stress_3d!(tau_xx, tau_xy, tau_xz,
                                     tau_yy, tau_yz, tau_zz,
                                     C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
                                     model::LogConfPTT)
    backend = KernelAbstractions.get_backend(tau_xx)
    Nx, Ny, Nz = size(tau_xx)
    FT = eltype(tau_xx)
    kernel! = _update_polymer_stress_3d_oldroyd_kernel!(backend)
    kernel!(tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz,
            C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
            FT(polymer_modulus(model)), FT(0.0); ndrange=(Nx, Ny, Nz))
    KernelAbstractions.synchronize(backend)
end
