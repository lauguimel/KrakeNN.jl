using KernelAbstractions

# --- Conformation tensor LBM (TRT, D3Q19) for 3D viscoelastic flows ---
#
# 3D port of `conformation_lbm_2d.jl`. Six independent scalar D3Q19
# advection-diffusion-reaction LBMs evolve C_xx, C_xy, C_xz, C_yy, C_yz,
# C_zz. Source S = Φ_αβ contains the upper-convected derivative + relaxation:
#
#   ∂C/∂t + u·∇C = C·∇u + (∇u)ᵀ·C - (1/λ)(C - I)
#
# Component dispatch via integer kwarg `component`:
#   1=xx, 2=xy, 3=xz, 4=yy, 5=yz, 6=zz
#
# D3Q19 indexing (Kraken convention from src/lattice/d3q19.jl, 1-indexed):
#   1 : rest          (0, 0, 0)        opp = 1
#   2 : +x            (+1, 0, 0)       opp = 3
#   3 : −x            (−1, 0, 0)       opp = 2
#   4 : +y            ( 0,+1, 0)       opp = 5
#   5 : −y            ( 0,−1, 0)       opp = 4
#   6 : +z            ( 0, 0,+1)       opp = 7
#   7 : −z            ( 0, 0,−1)       opp = 6
#   8 : (+x,+y)                        opp = 11
#   9 : (−x,+y)                        opp = 10
#  10 : (+x,−y)                        opp = 9
#  11 : (−x,−y)                        opp = 8
#  12 : (+x,+z)                        opp = 15
#  13 : (−x,+z)                        opp = 14
#  14 : (+x,−z)                        opp = 13
#  15 : (−x,−z)                        opp = 12
#  16 : (+y,+z)                        opp = 19
#  17 : (−y,+z)                        opp = 18
#  18 : (+y,−z)                        opp = 17
#  19 : (−y,−z)                        opp = 16

# ------------------------------------------------------------------
# Wall-aware first derivatives along y and z (3D port of the 2D
# `_wall_aware_dy_2d`/`_wall_aware_dx_2d` helpers in conformation_lbm_2d.jl).
#
# A naive clamped central difference `(a[j+1]-a[j-1])/2` with
# `jp=min(j+1,Ny); jm=max(j-1,1)` returns HALF the true gradient at the wall
# rows (j=1, j=Ny) for a cell-centred linear profile, biasing the Oldroyd-B
# production term there; the conformation LBM's artificial diffusion then
# bleeds that wall deficit into the bulk (defect #2). These helpers use a
# second-order one-sided stencil at the boundary rows (and a first-order
# one-sided fallback), matching the validated 2D kernel. GPU-safe (no
# allocation, branch-minimal, mirrors the 2D idiom verbatim).
# ------------------------------------------------------------------
@inline function _wall_aware_dy_3d(a, is_solid, i, j, k, Ny, ::Type{T}) where {T}
    plus_ok = j < Ny && !is_solid[i, j + 1, k]
    minus_ok = j > 1 && !is_solid[i, j - 1, k]
    plus2_ok = j < Ny - 1 && plus_ok && !is_solid[i, j + 2, k]
    minus2_ok = j > 2 && minus_ok && !is_solid[i, j - 2, k]
    if plus_ok && minus_ok
        return (a[i, j + 1, k] - a[i, j - 1, k]) / T(2)
    elseif plus2_ok
        return (-T(3) * a[i, j, k] + T(4) * a[i, j + 1, k] - a[i, j + 2, k]) / T(2)
    elseif minus2_ok
        return (T(3) * a[i, j, k] - T(4) * a[i, j - 1, k] + a[i, j - 2, k]) / T(2)
    elseif plus_ok
        return a[i, j + 1, k] - a[i, j, k]
    elseif minus_ok
        return a[i, j, k] - a[i, j - 1, k]
    else
        return zero(T)
    end
end

@inline function _wall_aware_dz_3d(a, is_solid, i, j, k, Nz, ::Type{T}) where {T}
    plus_ok = k < Nz && !is_solid[i, j, k + 1]
    minus_ok = k > 1 && !is_solid[i, j, k - 1]
    plus2_ok = k < Nz - 1 && plus_ok && !is_solid[i, j, k + 2]
    minus2_ok = k > 2 && minus_ok && !is_solid[i, j, k - 2]
    if plus_ok && minus_ok
        return (a[i, j, k + 1] - a[i, j, k - 1]) / T(2)
    elseif plus2_ok
        return (-T(3) * a[i, j, k] + T(4) * a[i, j, k + 1] - a[i, j, k + 2]) / T(2)
    elseif minus2_ok
        return (T(3) * a[i, j, k] - T(4) * a[i, j, k - 1] + a[i, j, k - 2]) / T(2)
    elseif plus_ok
        return a[i, j, k + 1] - a[i, j, k]
    elseif minus_ok
        return a[i, j, k] - a[i, j, k - 1]
    else
        return zero(T)
    end
end

# ------------------------------------------------------------------
# Source term for component α∈{1..6} using all 6 macroscopic fields.
# Returns the scalar S to be distributed as w_q · S.
# ------------------------------------------------------------------
@inline function _conf_source_3d(component::Int, inv_λ::T,
                                  cxx::T, cxy::T, cxz::T,
                                  cyy::T, cyz::T, czz::T,
                                  duxdx::T, duxdy::T, duxdz::T,
                                  duydx::T, duydy::T, duydz::T,
                                  duzdx::T, duzdy::T, duzdz::T) where {T}
    divu = duxdx + duydy + duzdz
    if component == 1        # xx
        return -inv_λ * (cxx - one(T)) +
               T(2) * (cxx * duxdx + cxy * duxdy + cxz * duxdz) + cxx * divu
    elseif component == 2    # xy
        return -inv_λ * cxy +
               (cxx * duydx + cyy * duxdy +
                cxy * (duxdx + duydy) + cxz * duydz + cyz * duxdz) + cxy * divu
    elseif component == 3    # xz
        return -inv_λ * cxz +
               (cxx * duzdx + czz * duxdz +
                cxz * (duxdx + duzdz) + cxy * duzdy + cyz * duxdy) + cxz * divu
    elseif component == 4    # yy
        return -inv_λ * (cyy - one(T)) +
               T(2) * (cxy * duydx + cyy * duydy + cyz * duydz) + cyy * divu
    elseif component == 5    # yz
        return -inv_λ * cyz +
               (cyy * duzdy + czz * duydz +
                cyz * (duydy + duzdz) + cxy * duzdx + cxz * duydx) + cyz * divu
    else                     # zz (component == 6)
        return -inv_λ * (czz - one(T)) +
               T(2) * (cxz * duzdx + cyz * duzdy + czz * duzdz) + czz * divu
    end
end

@kernel function collide_conformation_3d_kernel!(g, @Const(C_field),
                                                   @Const(ux), @Const(uy), @Const(uz),
                                                   @Const(C_xx), @Const(C_xy), @Const(C_xz),
                                                   @Const(C_yy), @Const(C_yz), @Const(C_zz),
                                                   @Const(is_solid),
                                                   tau_plus, tau_minus, lambda,
                                                   component, Nx, Ny, Nz)
    i, j, k = @index(Global, NTuple)

    @inbounds if !is_solid[i, j, k]
        T = eltype(g)
        φ = C_field[i, j, k]
        u = ux[i, j, k]; v = uy[i, j, k]; w = uz[i, j, k]
        usq = u*u + v*v + w*w

        # Velocity gradient: periodic-x central diff; wall-aware (one-sided
        # 2nd-order at the y/z boundary rows) for ∂/∂y and ∂/∂z so the
        # Oldroyd-B production term sees the FULL shear rate at the walls
        # (defect #2 fix — mirrors the validated 2D `_wall_aware_dy_2d`).
        ip = ifelse(i < Nx, i + 1, 1)
        im = ifelse(i > 1,  i - 1, Nx)

        duxdx = (ux[ip,j,k] - ux[im,j,k]) / T(2)
        duxdy = _wall_aware_dy_3d(ux, is_solid, i, j, k, Ny, T)
        duxdz = _wall_aware_dz_3d(ux, is_solid, i, j, k, Nz, T)
        duydx = (uy[ip,j,k] - uy[im,j,k]) / T(2)
        duydy = _wall_aware_dy_3d(uy, is_solid, i, j, k, Ny, T)
        duydz = _wall_aware_dz_3d(uy, is_solid, i, j, k, Nz, T)
        duzdx = (uz[ip,j,k] - uz[im,j,k]) / T(2)
        duzdy = _wall_aware_dy_3d(uz, is_solid, i, j, k, Ny, T)
        duzdz = _wall_aware_dz_3d(uz, is_solid, i, j, k, Nz, T)

        cxx = C_xx[i,j,k]; cxy = C_xy[i,j,k]; cxz = C_xz[i,j,k]
        cyy = C_yy[i,j,k]; cyz = C_yz[i,j,k]; czz = C_zz[i,j,k]
        inv_λ = one(T) / T(lambda)
        S = _conf_source_3d(component, inv_λ, cxx, cxy, cxz, cyy, cyz, czz,
                              duxdx, duxdy, duxdz,
                              duydx, duydy, duydz,
                              duzdx, duzdy, duzdz)

        # Pre-load all 19 populations
        g1  = g[i,j,k,1];  g2  = g[i,j,k,2];  g3  = g[i,j,k,3]
        g4  = g[i,j,k,4];  g5  = g[i,j,k,5];  g6  = g[i,j,k,6]
        g7  = g[i,j,k,7];  g8  = g[i,j,k,8];  g9  = g[i,j,k,9]
        g10 = g[i,j,k,10]; g11 = g[i,j,k,11]; g12 = g[i,j,k,12]
        g13 = g[i,j,k,13]; g14 = g[i,j,k,14]; g15 = g[i,j,k,15]
        g16 = g[i,j,k,16]; g17 = g[i,j,k,17]; g18 = g[i,j,k,18]
        g19 = g[i,j,k,19]

        # Equilibria with φ instead of ρ (same Mach expansion is valid
        # for an advected scalar at low velocity)
        ge1  = feq_3d(Val(1),  φ, u, v, w, usq); ge2  = feq_3d(Val(2),  φ, u, v, w, usq)
        ge3  = feq_3d(Val(3),  φ, u, v, w, usq); ge4  = feq_3d(Val(4),  φ, u, v, w, usq)
        ge5  = feq_3d(Val(5),  φ, u, v, w, usq); ge6  = feq_3d(Val(6),  φ, u, v, w, usq)
        ge7  = feq_3d(Val(7),  φ, u, v, w, usq); ge8  = feq_3d(Val(8),  φ, u, v, w, usq)
        ge9  = feq_3d(Val(9),  φ, u, v, w, usq); ge10 = feq_3d(Val(10), φ, u, v, w, usq)
        ge11 = feq_3d(Val(11), φ, u, v, w, usq); ge12 = feq_3d(Val(12), φ, u, v, w, usq)
        ge13 = feq_3d(Val(13), φ, u, v, w, usq); ge14 = feq_3d(Val(14), φ, u, v, w, usq)
        ge15 = feq_3d(Val(15), φ, u, v, w, usq); ge16 = feq_3d(Val(16), φ, u, v, w, usq)
        ge17 = feq_3d(Val(17), φ, u, v, w, usq); ge18 = feq_3d(Val(18), φ, u, v, w, usq)
        ge19 = feq_3d(Val(19), φ, u, v, w, usq)

        ωp = one(T) / T(tau_plus)
        ωm = one(T) / T(tau_minus)
        half = T(0.5)
        wr = T(1/3); wa = T(1/18); we = T(1/36)

        # Self-paired rest population (q=1)
        g[i,j,k,1] = g1 - ωp * (g1 - ge1) + wr * S

        # TRT pair (q, qopp) inlined: post_q = g_q − ωp·(gp − ep) − ωm·(gm − em)
        # with gp/gm = (g_q ± g_qopp)/2 and ep/em = (eq_q ± eq_qopp)/2. The
        # opposite member uses the same gp/ep but the sign on (gm − em) flips.

        # Axial x: pair (2, 3)
        gp23 = (g2 + g3)*half;  gm23 = (g2 - g3)*half
        ep23 = (ge2 + ge3)*half; em23 = (ge2 - ge3)*half
        g[i,j,k,2] = g2 - ωp*(gp23 - ep23) - ωm*(gm23 - em23) + wa*S
        g[i,j,k,3] = g3 - ωp*(gp23 - ep23) + ωm*(gm23 - em23) + wa*S
        # Axial y: pair (4, 5)
        gp45 = (g4 + g5)*half;  gm45 = (g4 - g5)*half
        ep45 = (ge4 + ge5)*half; em45 = (ge4 - ge5)*half
        g[i,j,k,4] = g4 - ωp*(gp45 - ep45) - ωm*(gm45 - em45) + wa*S
        g[i,j,k,5] = g5 - ωp*(gp45 - ep45) + ωm*(gm45 - em45) + wa*S
        # Axial z: pair (6, 7)
        gp67 = (g6 + g7)*half;  gm67 = (g6 - g7)*half
        ep67 = (ge6 + ge7)*half; em67 = (ge6 - ge7)*half
        g[i,j,k,6] = g6 - ωp*(gp67 - ep67) - ωm*(gm67 - em67) + wa*S
        g[i,j,k,7] = g7 - ωp*(gp67 - ep67) + ωm*(gm67 - em67) + wa*S
        # xy edges: (8, 11)
        gp_a = (g8 + g11)*half;  gm_a = (g8 - g11)*half
        ep_a = (ge8 + ge11)*half; em_a = (ge8 - ge11)*half
        g[i,j,k,8]  = g8  - ωp*(gp_a - ep_a) - ωm*(gm_a - em_a) + we*S
        g[i,j,k,11] = g11 - ωp*(gp_a - ep_a) + ωm*(gm_a - em_a) + we*S
        # xy edges: (9, 10)
        gp_b = (g9 + g10)*half;  gm_b = (g9 - g10)*half
        ep_b = (ge9 + ge10)*half; em_b = (ge9 - ge10)*half
        g[i,j,k,9]  = g9  - ωp*(gp_b - ep_b) - ωm*(gm_b - em_b) + we*S
        g[i,j,k,10] = g10 - ωp*(gp_b - ep_b) + ωm*(gm_b - em_b) + we*S
        # xz edges: (12, 15)
        gp_c = (g12 + g15)*half;  gm_c = (g12 - g15)*half
        ep_c = (ge12 + ge15)*half; em_c = (ge12 - ge15)*half
        g[i,j,k,12] = g12 - ωp*(gp_c - ep_c) - ωm*(gm_c - em_c) + we*S
        g[i,j,k,15] = g15 - ωp*(gp_c - ep_c) + ωm*(gm_c - em_c) + we*S
        # xz edges: (13, 14)
        gp_d = (g13 + g14)*half;  gm_d = (g13 - g14)*half
        ep_d = (ge13 + ge14)*half; em_d = (ge13 - ge14)*half
        g[i,j,k,13] = g13 - ωp*(gp_d - ep_d) - ωm*(gm_d - em_d) + we*S
        g[i,j,k,14] = g14 - ωp*(gp_d - ep_d) + ωm*(gm_d - em_d) + we*S
        # yz edges: (16, 19)
        gp_e = (g16 + g19)*half;  gm_e = (g16 - g19)*half
        ep_e = (ge16 + ge19)*half; em_e = (ge16 - ge19)*half
        g[i,j,k,16] = g16 - ωp*(gp_e - ep_e) - ωm*(gm_e - em_e) + we*S
        g[i,j,k,19] = g19 - ωp*(gp_e - ep_e) + ωm*(gm_e - em_e) + we*S
        # yz edges: (17, 18)
        gp_f = (g17 + g18)*half;  gm_f = (g17 - g18)*half
        ep_f = (ge17 + ge18)*half; em_f = (ge17 - ge18)*half
        g[i,j,k,17] = g17 - ωp*(gp_f - ep_f) - ωm*(gm_f - em_f) + we*S
        g[i,j,k,18] = g18 - ωp*(gp_f - ep_f) + ωm*(gm_f - em_f) + we*S
    end
end

"""
    collide_conformation_3d!(g, C_field, ux, uy, uz,
                              C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
                              is_solid, tau_plus, lambda;
                              magic=0.25, component=1)

TRT collision + source for one scalar component of the conformation
tensor in 3D D3Q19. `component ∈ {1=xx, 2=xy, 3=xz, 4=yy, 5=yz, 6=zz}`.
"""
function collide_conformation_3d!(g, C_field, ux, uy, uz,
                                    C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
                                    is_solid, tau_plus, lambda;
                                    magic=0.25, component=1)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny, Nz = size(C_field)
    T = eltype(g)
    tau_minus = magic / (tau_plus - 0.5) + 0.5
    kernel! = collide_conformation_3d_kernel!(backend)
    kernel!(g, C_field, ux, uy, uz,
            C_xx, C_xy, C_xz, C_yy, C_yz, C_zz, is_solid,
            T(tau_plus), T(tau_minus), T(lambda),
            Int(component), Nx, Ny, Nz; ndrange=(Nx, Ny, Nz))
    KernelAbstractions.synchronize(backend)
end

# ============================================================
# Initialization & macroscopic recovery
# ============================================================

@kernel function init_conformation_field_3d_kernel!(g, @Const(C_field),
                                                      @Const(ux), @Const(uy), @Const(uz))
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        T = eltype(g)
        φ = C_field[i, j, k]
        u = ux[i, j, k]; v = uy[i, j, k]; w = uz[i, j, k]
        usq = u*u + v*v + w*w
        for q in 1:19
            g[i,j,k,q] = q == 1  ? feq_3d(Val(1),  φ, u, v, w, usq) :
                          q == 2  ? feq_3d(Val(2),  φ, u, v, w, usq) :
                          q == 3  ? feq_3d(Val(3),  φ, u, v, w, usq) :
                          q == 4  ? feq_3d(Val(4),  φ, u, v, w, usq) :
                          q == 5  ? feq_3d(Val(5),  φ, u, v, w, usq) :
                          q == 6  ? feq_3d(Val(6),  φ, u, v, w, usq) :
                          q == 7  ? feq_3d(Val(7),  φ, u, v, w, usq) :
                          q == 8  ? feq_3d(Val(8),  φ, u, v, w, usq) :
                          q == 9  ? feq_3d(Val(9),  φ, u, v, w, usq) :
                          q == 10 ? feq_3d(Val(10), φ, u, v, w, usq) :
                          q == 11 ? feq_3d(Val(11), φ, u, v, w, usq) :
                          q == 12 ? feq_3d(Val(12), φ, u, v, w, usq) :
                          q == 13 ? feq_3d(Val(13), φ, u, v, w, usq) :
                          q == 14 ? feq_3d(Val(14), φ, u, v, w, usq) :
                          q == 15 ? feq_3d(Val(15), φ, u, v, w, usq) :
                          q == 16 ? feq_3d(Val(16), φ, u, v, w, usq) :
                          q == 17 ? feq_3d(Val(17), φ, u, v, w, usq) :
                          q == 18 ? feq_3d(Val(18), φ, u, v, w, usq) :
                                    feq_3d(Val(19), φ, u, v, w, usq)
        end
    end
end

function init_conformation_field_3d!(g, C_field, ux, uy, uz)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny, Nz = size(C_field)
    kernel! = init_conformation_field_3d_kernel!(backend)
    kernel!(g, C_field, ux, uy, uz; ndrange=(Nx, Ny, Nz))
    KernelAbstractions.synchronize(backend)
end

@kernel function compute_conformation_macro_3d_kernel!(C_field, @Const(g))
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        s = zero(eltype(g))
        for q in 1:19
            s += g[i,j,k,q]
        end
        C_field[i,j,k] = s
    end
end

function compute_conformation_macro_3d!(C_field, g)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny, Nz = size(C_field)
    kernel! = compute_conformation_macro_3d_kernel!(backend)
    kernel!(C_field, g; ndrange=(Nx, Ny, Nz))
    KernelAbstractions.synchronize(backend)
end

# ============================================================
# CNEBB (Conservative Non-Equilibrium Bounce-Back) for 3D
# Liu et al. 2025 Eqs (38-39), 3D port.
# ============================================================

# D3Q19 lookup helpers — must match d3q19.jl
@inline _opp_q3(q) = (1, 3, 2, 5, 4, 7, 6, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16)[q]
@inline _cx_q3(q)  = (0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0)[q]
@inline _cy_q3(q)  = (0,  0,  0,  1, -1,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1)[q]
@inline _cz_q3(q)  = (0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1)[q]

@kernel function apply_cnebb_3d_kernel!(g_post, @Const(g_pre), @Const(is_solid),
                                          C_field, Nx, Ny, Nz)
    i, j, k = @index(Global, NTuple)
    @inbounds if !is_solid[i, j, k]
        T = eltype(g_post)

        # Detect any solid or out-of-domain neighbour. Domain boundaries are
        # physical walls for the conformation populations as well.
        any_solid = false
        for q in 2:19
            ni = i + _cx_q3(q); nj = j + _cy_q3(q); nk = k + _cz_q3(q)
            if !(1 ≤ ni ≤ Nx && 1 ≤ nj ≤ Ny && 1 ≤ nk ≤ Nz) || is_solid[ni, nj, nk]
                any_solid = true
            end
        end

        if any_solid
            # Conservative φ from Γ ∪ H sums
            φ = g_post[i,j,k,1]
            for q in 2:19
                si = i - _cx_q3(q); sj = j - _cy_q3(q); sk = k - _cz_q3(q)
                src_solid = !(1 ≤ si ≤ Nx && 1 ≤ sj ≤ Ny && 1 ≤ sk ≤ Nz) ||
                            is_solid[si, sj, sk]
                φ += src_solid ? g_pre[i,j,k,_opp_q3(q)] : g_post[i,j,k,q]
            end

            # Equilibrium at u_wall = 0 (no-slip)
            usq = zero(T); zT = zero(T)
            ge_q = (
                feq_3d(Val(1),  φ, zT, zT, zT, usq), feq_3d(Val(2),  φ, zT, zT, zT, usq),
                feq_3d(Val(3),  φ, zT, zT, zT, usq), feq_3d(Val(4),  φ, zT, zT, zT, usq),
                feq_3d(Val(5),  φ, zT, zT, zT, usq), feq_3d(Val(6),  φ, zT, zT, zT, usq),
                feq_3d(Val(7),  φ, zT, zT, zT, usq), feq_3d(Val(8),  φ, zT, zT, zT, usq),
                feq_3d(Val(9),  φ, zT, zT, zT, usq), feq_3d(Val(10), φ, zT, zT, zT, usq),
                feq_3d(Val(11), φ, zT, zT, zT, usq), feq_3d(Val(12), φ, zT, zT, zT, usq),
                feq_3d(Val(13), φ, zT, zT, zT, usq), feq_3d(Val(14), φ, zT, zT, zT, usq),
                feq_3d(Val(15), φ, zT, zT, zT, usq), feq_3d(Val(16), φ, zT, zT, zT, usq),
                feq_3d(Val(17), φ, zT, zT, zT, usq), feq_3d(Val(18), φ, zT, zT, zT, usq),
                feq_3d(Val(19), φ, zT, zT, zT, usq))

            # Reconstruct unknowns via NEBB: g_post[q] = ge_q + (g_post[opp] − ge_opp)
            for q in 2:19
                si = i - _cx_q3(q); sj = j - _cy_q3(q); sk = k - _cz_q3(q)
                src_solid = !(1 ≤ si ≤ Nx && 1 ≤ sj ≤ Ny && 1 ≤ sk ≤ Nz) ||
                            is_solid[si, sj, sk]
                if src_solid
                    oq = _opp_q3(q)
                    g_post[i,j,k,q] = ge_q[q] + (g_post[i,j,k,oq] - ge_q[oq])
                end
            end

            nonrest = zero(T)
            for q in 2:19
                nonrest += g_post[i,j,k,q]
            end
            g_post[i,j,k,1] = φ - nonrest

            C_field[i,j,k] = φ
        end
    end
end

"""
    apply_cnebb_conformation_3d!(g_post, g_pre, is_solid, C_field)

3D port of `apply_cnebb_conformation_2d!` — Liu et al. 2025 Eqs (38-39)
on D3Q19 conformation populations. Conservatively reconstructs the 6
solid-sourced populations at any fluid cell with at least one solid
D3Q19 neighbour, and updates the macroscopic component `C_field` to the
exact φ moment.
"""
function apply_cnebb_conformation_3d!(g_post, g_pre, is_solid, C_field)
    backend = KernelAbstractions.get_backend(g_post)
    Nx, Ny, Nz = size(C_field)
    kernel! = apply_cnebb_3d_kernel!(backend)
    kernel!(g_post, g_pre, is_solid, C_field, Nx, Ny, Nz; ndrange=(Nx, Ny, Nz))
    KernelAbstractions.synchronize(backend)
end

# ============================================================
# CNEBB on the y-faces only (for periodic-xz channel / Couette)
# ============================================================
#
# The general `apply_cnebb_conformation_3d!` treats ALL six domain edges as
# walls. For a planar Couette / channel box that is periodic in x and z, only
# the y-faces (j = 1 and j = Ny) are physical walls; the x/z edges must stay
# periodic (the gradient stencil already wraps x and the streamer wraps x,z).
# This variant applies the conservative NEBB reconstruction at j = 1 and
# j = Ny only, leaving the periodic faces untouched.

@kernel function apply_cnebb_y_walls_3d_kernel!(g_post, @Const(g_pre),
                                                  C_field, Nx, Ny, Nz)
    i, j, k = @index(Global, NTuple)
    @inbounds if j == 1 || j == Ny
        T = eltype(g_post)
        # Wall normal sign: +y neighbour missing at j=1, -y neighbour at j=Ny.
        # A source population q is "from the wall" iff stepping back by c_q
        # crosses the y-face (j - cy_q outside [1,Ny]).
        φ = g_post[i,j,k,1]
        for q in 2:19
            sj = j - _cy_q3(q)
            src_solid = !(1 ≤ sj ≤ Ny)
            φ += src_solid ? g_pre[i,j,k,_opp_q3(q)] : g_post[i,j,k,q]
        end

        usq = zero(T); zT = zero(T)
        ge_q = (
            feq_3d(Val(1),  φ, zT, zT, zT, usq), feq_3d(Val(2),  φ, zT, zT, zT, usq),
            feq_3d(Val(3),  φ, zT, zT, zT, usq), feq_3d(Val(4),  φ, zT, zT, zT, usq),
            feq_3d(Val(5),  φ, zT, zT, zT, usq), feq_3d(Val(6),  φ, zT, zT, zT, usq),
            feq_3d(Val(7),  φ, zT, zT, zT, usq), feq_3d(Val(8),  φ, zT, zT, zT, usq),
            feq_3d(Val(9),  φ, zT, zT, zT, usq), feq_3d(Val(10), φ, zT, zT, zT, usq),
            feq_3d(Val(11), φ, zT, zT, zT, usq), feq_3d(Val(12), φ, zT, zT, zT, usq),
            feq_3d(Val(13), φ, zT, zT, zT, usq), feq_3d(Val(14), φ, zT, zT, zT, usq),
            feq_3d(Val(15), φ, zT, zT, zT, usq), feq_3d(Val(16), φ, zT, zT, zT, usq),
            feq_3d(Val(17), φ, zT, zT, zT, usq), feq_3d(Val(18), φ, zT, zT, zT, usq),
            feq_3d(Val(19), φ, zT, zT, zT, usq))

        for q in 2:19
            sj = j - _cy_q3(q)
            if !(1 ≤ sj ≤ Ny)
                oq = _opp_q3(q)
                g_post[i,j,k,q] = ge_q[q] + (g_post[i,j,k,oq] - ge_q[oq])
            end
        end

        nonrest = zero(T)
        for q in 2:19
            nonrest += g_post[i,j,k,q]
        end
        g_post[i,j,k,1] = φ - nonrest
        C_field[i,j,k] = φ
    end
end

"""
    apply_cnebb_conformation_y_walls_3d!(g_post, g_pre, C_field)

CNEBB conformation wall BC on the y-faces (j = 1 and j = Ny) only — the
periodic-xz analogue of `apply_cnebb_conformation_3d!`. Reconstructs the
solid-sourced populations at the two y-walls and leaves the periodic x/z faces
untouched. Use for planar Couette / Poiseuille channels.
"""
function apply_cnebb_conformation_y_walls_3d!(g_post, g_pre, C_field)
    backend = KernelAbstractions.get_backend(g_post)
    Nx, Ny, Nz = size(C_field)
    kernel! = apply_cnebb_y_walls_3d_kernel!(backend)
    kernel!(g_post, g_pre, C_field, Nx, Ny, Nz; ndrange=(Nx, Ny, Nz))
    KernelAbstractions.synchronize(backend)
end

# ============================================================
# Inlet / outlet conformation reset (3D)
# ============================================================

@kernel function _reset_conf_inlet_3d_kernel!(g, @Const(C_inlet), @Const(u_profile))
    j, k = @index(Global, NTuple)
    @inbounds begin
        T = eltype(g)
        φ = C_inlet[j, k]
        u = u_profile[j, k]
        usq = u * u
        zT = zero(T)
        for q in 1:19
            g[1,j,k,q] = q == 1  ? feq_3d(Val(1),  φ, u, zT, zT, usq) :
                          q == 2  ? feq_3d(Val(2),  φ, u, zT, zT, usq) :
                          q == 3  ? feq_3d(Val(3),  φ, u, zT, zT, usq) :
                          q == 4  ? feq_3d(Val(4),  φ, u, zT, zT, usq) :
                          q == 5  ? feq_3d(Val(5),  φ, u, zT, zT, usq) :
                          q == 6  ? feq_3d(Val(6),  φ, u, zT, zT, usq) :
                          q == 7  ? feq_3d(Val(7),  φ, u, zT, zT, usq) :
                          q == 8  ? feq_3d(Val(8),  φ, u, zT, zT, usq) :
                          q == 9  ? feq_3d(Val(9),  φ, u, zT, zT, usq) :
                          q == 10 ? feq_3d(Val(10), φ, u, zT, zT, usq) :
                          q == 11 ? feq_3d(Val(11), φ, u, zT, zT, usq) :
                          q == 12 ? feq_3d(Val(12), φ, u, zT, zT, usq) :
                          q == 13 ? feq_3d(Val(13), φ, u, zT, zT, usq) :
                          q == 14 ? feq_3d(Val(14), φ, u, zT, zT, usq) :
                          q == 15 ? feq_3d(Val(15), φ, u, zT, zT, usq) :
                          q == 16 ? feq_3d(Val(16), φ, u, zT, zT, usq) :
                          q == 17 ? feq_3d(Val(17), φ, u, zT, zT, usq) :
                          q == 18 ? feq_3d(Val(18), φ, u, zT, zT, usq) :
                                    feq_3d(Val(19), φ, u, zT, zT, usq)
        end
    end
end

"""
    reset_conformation_inlet_3d!(g, C_inlet, u_profile, Ny, Nz)

Force the inlet plane (i=1) populations to equilibrium at the prescribed
analytical conformation `C_inlet[j, k]` and velocity `u_profile[j, k]`.
Both arrays are 2D (Ny × Nz) device arrays.
"""
function reset_conformation_inlet_3d!(g, C_inlet, u_profile, Ny, Nz)
    backend = KernelAbstractions.get_backend(g)
    kernel! = _reset_conf_inlet_3d_kernel!(backend)
    kernel!(g, C_inlet, u_profile; ndrange=(Ny, Nz))
    KernelAbstractions.synchronize(backend)
end

@kernel function _reset_conf_outlet_3d_kernel!(g, Nx)
    j, k = @index(Global, NTuple)
    @inbounds begin
        for q in 1:19
            g[Nx, j, k, q] = g[Nx-1, j, k, q]
        end
    end
end

"""
    reset_conformation_outlet_3d!(g, Nx, Ny, Nz)

Zero-gradient extrapolation of g at the outlet plane (i=Nx) — the 3D
analog of `reset_conformation_outlet_2d!`.
"""
function reset_conformation_outlet_3d!(g, Nx, Ny, Nz)
    backend = KernelAbstractions.get_backend(g)
    kernel! = _reset_conf_outlet_3d_kernel!(backend)
    kernel!(g, Nx; ndrange=(Ny, Nz))
    KernelAbstractions.synchronize(backend)
end
