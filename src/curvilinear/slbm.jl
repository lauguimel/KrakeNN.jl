using KernelAbstractions

# ===========================================================================
# Semi-Lagrangian LBM (SLBM) kernels for the curvilinear v0.2 path.
#
# Formulation (Krämer et al. 2017, Phys. Rev. E 95, 023305):
#   After collision, f_q(P, t+Δt) ← f_q(P − c_q · dx_ref, t)
# Since P − c_q · dx_ref is not a mesh node on a warped grid, we
# interpolate f_q at the departure point from the surrounding nodes.
#
# Departure grid indices (i_dep, j_dep) are precomputed per (i, j, q) at
# simulation start. The streaming step then consists of an interpolation
# plus a standard collision, all fused in one GPU kernel.
#
# Bilinear interpolation is O(Δx²) on smooth fields. Biquadratic
# (3×3 Lagrange) is O(Δx³) and recovers second-order accuracy on
# distorted meshes (Wilde et al. 2020, Comput. Fluids 204, 104519).
# ===========================================================================

"""
    SLBMGeometry{T, AT3, AT2}

Precomputed departure grid indices for semi-Lagrangian streaming on a
curvilinear mesh. Built once from a `CurvilinearMesh`; passed to the
fused kernel at every timestep.

# Fields
- `i_dep, j_dep`: floating-point departure indices `[Nξ, Nη, 9]`. For
  direction `q` at node `(i, j)`, the departure is
  `(i_dep[i,j,q], j_dep[i,j,q])` in grid-index space.
- `periodic_ξ, periodic_η`: mirror the parent mesh so the kernel can
  wrap or clamp.
- `Nξ, Nη`: logical grid extents.
- `dx_ref`: physical distance per lattice unit (copied from mesh).

For a uniform Cartesian mesh with isotropic spacing, `i_dep[i,j,q] =
i − c_qx` and `j_dep[i,j,q] = j − c_qy` exactly — SLBM reduces to
standard stream-collide.
"""
struct SLBMGeometry{T<:AbstractFloat, AT3<:AbstractArray{T, 3}}
    i_dep::AT3
    j_dep::AT3
    Nξ::Int
    Nη::Int
    periodic_ξ::Bool
    periodic_η::Bool
    dx_ref::T
end

"""
    build_slbm_geometry(mesh::CurvilinearMesh) -> SLBMGeometry

Precompute the grid-index departure points for every node and every
D2Q9 direction. Linearises the mapping locally via the stored metric:

    Δi = −dx_ref · denom_ξ · (dξ/dX · c_qx + dξ/dY · c_qy)
    Δj = −dx_ref · denom_η · (dη/dX · c_qx + dη/dY · c_qy)

with the inverse Jacobian `dξ/dX = dY/dη / J`, etc.
"""
function build_slbm_geometry(mesh::CurvilinearMesh{T};
                              local_cfl::Bool=false,
                              dx_ref::Real=0) where {T}
    Nξ, Nη = mesh.Nξ, mesh.Nη
    cx = velocities_x(D2Q9())
    cy = velocities_y(D2Q9())

    denom_ξ = T(mesh.periodic_ξ ? Nξ : Nξ - 1)
    denom_η = T(mesh.periodic_η ? Nη : Nη - 1)
    Δξ = one(T) / denom_ξ
    Δη = one(T) / denom_η

    i_dep = zeros(T, Nξ, Nη, 9)
    j_dep = zeros(T, Nξ, Nη, 9)
    dxr = dx_ref > 0 ? T(dx_ref) : mesh.dx_ref

    @inbounds for j in 1:Nη, i in 1:Nξ
        J = mesh.J[i, j]
        dXdξ = mesh.dXdξ[i, j]
        dXdη = mesh.dXdη[i, j]
        dYdξ = mesh.dYdξ[i, j]
        dYdη = mesh.dYdη[i, j]
        ξx =  dYdη / J
        ξy = -dXdη / J
        ηx = -dYdξ / J
        ηy =  dXdξ / J

        if local_cfl
            # Local cell size → CFL ≈ 1 per cell
            lξ = sqrt(dXdξ^2 + dYdξ^2) * Δξ
            lη = sqrt(dXdη^2 + dYdη^2) * Δη
            dxr_local = sqrt(lξ * lη)
        else
            dxr_local = dxr
        end

        for q in 1:9
            cqx = T(cx[q])
            cqy = T(cy[q])
            Δi = -dxr_local * denom_ξ * (ξx * cqx + ξy * cqy)
            Δj = -dxr_local * denom_η * (ηx * cqx + ηy * cqy)
            i_dep[i, j, q] = T(i) + Δi
            j_dep[i, j, q] = T(j) + Δj
        end
    end

    return SLBMGeometry{T, Array{T, 3}}(i_dep, j_dep, Nξ, Nη,
                                         mesh.periodic_ξ, mesh.periodic_η, dxr)
end

"""
    transfer_slbm_geometry(geom::SLBMGeometry, backend) -> SLBMGeometry

Move the precomputed departure arrays to the device backend
(`KernelAbstractions.CPU()`, `CUDABackend()`, `MetalBackend()`).
"""
function transfer_slbm_geometry(geom::SLBMGeometry{T}, backend) where {T}
    i_dep_dev = KernelAbstractions.allocate(backend, T, size(geom.i_dep))
    j_dep_dev = KernelAbstractions.allocate(backend, T, size(geom.j_dep))
    copyto!(i_dep_dev, geom.i_dep)
    copyto!(j_dep_dev, geom.j_dep)
    return SLBMGeometry{T, typeof(i_dep_dev)}(i_dep_dev, j_dep_dev,
                                                geom.Nξ, geom.Nη,
                                                geom.periodic_ξ, geom.periodic_η,
                                                geom.dx_ref)
end

# ---------------------------------------------------------------------------
# Interpolation helpers. All marked @inline so the kernel fuses the
# index math without function-call overhead on GPU. No allocations.
# ---------------------------------------------------------------------------

@inline function _wrap_or_clamp(i::Int, N::Int, periodic::Bool)
    if periodic
        # mod1(i, N) returns a value in 1..N, wrapping both sides.
        return ((i - 1) % N + N) % N + 1
    else
        return clamp(i, 1, N)
    end
end

@inline function bilinear_f(f_in, ifloat::T, jfloat::T, q::Int,
                             Nξ::Int, Nη::Int,
                             periodic_ξ::Bool, periodic_η::Bool) where {T}
    i0 = unsafe_trunc(Int, floor(ifloat))
    j0 = unsafe_trunc(Int, floor(jfloat))
    α = ifloat - T(i0)
    β = jfloat - T(j0)
    i0w = _wrap_or_clamp(i0,     Nξ, periodic_ξ)
    i1w = _wrap_or_clamp(i0 + 1, Nξ, periodic_ξ)
    j0w = _wrap_or_clamp(j0,     Nη, periodic_η)
    j1w = _wrap_or_clamp(j0 + 1, Nη, periodic_η)
    f00 = f_in[i0w, j0w, q]
    f10 = f_in[i1w, j0w, q]
    f01 = f_in[i0w, j1w, q]
    f11 = f_in[i1w, j1w, q]
    return (one(T) - α) * (one(T) - β) * f00 +
           α           * (one(T) - β) * f10 +
           (one(T) - α) * β           * f01 +
           α           * β           * f11
end

# Biquadratic (3×3 Lagrange) interpolation. Centered stencil: pick the
# NEAREST node (i_c, j_c) to (ifloat, jfloat) and use the 3×3 block
# around it. Local 1D weights for shift α ∈ [-0.5, 0.5]:
#
#     w_{-1}(α) = α(α − 1) / 2
#     w_{0}(α)  = 1 − α²
#     w_{+1}(α) = α(α + 1) / 2
#
# At integer departure (α = β = 0) this returns f[i_c, j_c] exactly —
# so the kernel still reduces to pull-stream on a Cartesian mesh.
# O(Δx³) on smooth fields (Wilde et al. 2020, Comput. Fluids 204).
@inline function biquadratic_f(f_in, is_solid, ifloat::T, jfloat::T, q::Int,
                                Nξ::Int, Nη::Int,
                                periodic_ξ::Bool, periodic_η::Bool) where {T}
    ic = unsafe_trunc(Int, floor(ifloat + T(0.5)))
    jc = unsafe_trunc(Int, floor(jfloat + T(0.5)))
    α = ifloat - T(ic)
    β = jfloat - T(jc)
    im = _wrap_or_clamp(ic - 1, Nξ, periodic_ξ)
    i0 = _wrap_or_clamp(ic,     Nξ, periodic_ξ)
    ip = _wrap_or_clamp(ic + 1, Nξ, periodic_ξ)
    jm = _wrap_or_clamp(jc - 1, Nη, periodic_η)
    j0 = _wrap_or_clamp(jc,     Nη, periodic_η)
    jp = _wrap_or_clamp(jc + 1, Nη, periodic_η)
    # If any of the 9 stencil cells is solid (bounced populations) the
    # biquadratic Lagrange weights would amplify the near-wall
    # discontinuity and destabilise the step. Fall back to bilinear,
    # which has only non-negative weights.
    near_wall = is_solid[im, jm] | is_solid[im, j0] | is_solid[im, jp] |
                is_solid[i0, jm] | is_solid[i0, j0] | is_solid[i0, jp] |
                is_solid[ip, jm] | is_solid[ip, j0] | is_solid[ip, jp]
    if near_wall
        return bilinear_f(f_in, ifloat, jfloat, q, Nξ, Nη, periodic_ξ, periodic_η)
    end
    half = T(0.5)
    wα_m = α * (α - one(T)) * half
    wα_0 = one(T) - α * α
    wα_p = α * (α + one(T)) * half
    wβ_m = β * (β - one(T)) * half
    wβ_0 = one(T) - β * β
    wβ_p = β * (β + one(T)) * half
    return wα_m * (wβ_m * f_in[im, jm, q] + wβ_0 * f_in[im, j0, q] + wβ_p * f_in[im, jp, q]) +
           wα_0 * (wβ_m * f_in[i0, jm, q] + wβ_0 * f_in[i0, j0, q] + wβ_p * f_in[i0, jp, q]) +
           wα_p * (wβ_m * f_in[ip, jm, q] + wβ_0 * f_in[ip, j0, q] + wβ_p * f_in[ip, jp, q])
end

# ---------------------------------------------------------------------------
# Fused SLBM + BGK kernel. Single GPU dispatch per timestep.
#
# Steps per thread:
#   1. For each of the 9 D2Q9 directions, interpolate f_in at the
#      precomputed departure point.
#   2. If solid, bounce-back (swap opposite pairs).
#   3. Otherwise: compute moments (ρ, ux, uy), apply BGK relaxation,
#      write both f_out and macroscopic fields.
# ---------------------------------------------------------------------------

@kernel function slbm_bgk_step_kernel!(f_out, @Const(f_in),
                                        ρ_out, ux_out, uy_out,
                                        @Const(is_solid),
                                        @Const(i_dep), @Const(j_dep),
                                        Nξ, Nη, ω,
                                        periodic_ξ, periodic_η)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f_out)
        # 1. Semi-Lagrangian pull for each direction
        fp1 = bilinear_f(f_in, i_dep[i, j, 1], j_dep[i, j, 1], 1, Nξ, Nη, periodic_ξ, periodic_η)
        fp2 = bilinear_f(f_in, i_dep[i, j, 2], j_dep[i, j, 2], 2, Nξ, Nη, periodic_ξ, periodic_η)
        fp3 = bilinear_f(f_in, i_dep[i, j, 3], j_dep[i, j, 3], 3, Nξ, Nη, periodic_ξ, periodic_η)
        fp4 = bilinear_f(f_in, i_dep[i, j, 4], j_dep[i, j, 4], 4, Nξ, Nη, periodic_ξ, periodic_η)
        fp5 = bilinear_f(f_in, i_dep[i, j, 5], j_dep[i, j, 5], 5, Nξ, Nη, periodic_ξ, periodic_η)
        fp6 = bilinear_f(f_in, i_dep[i, j, 6], j_dep[i, j, 6], 6, Nξ, Nη, periodic_ξ, periodic_η)
        fp7 = bilinear_f(f_in, i_dep[i, j, 7], j_dep[i, j, 7], 7, Nξ, Nη, periodic_ξ, periodic_η)
        fp8 = bilinear_f(f_in, i_dep[i, j, 8], j_dep[i, j, 8], 8, Nξ, Nη, periodic_ξ, periodic_η)
        fp9 = bilinear_f(f_in, i_dep[i, j, 9], j_dep[i, j, 9], 9, Nξ, Nη, periodic_ξ, periodic_η)

        if is_solid[i, j]
            # Halfway bounce-back
            f_out[i, j, 1] = fp1
            f_out[i, j, 2] = fp4; f_out[i, j, 4] = fp2
            f_out[i, j, 3] = fp5; f_out[i, j, 5] = fp3
            f_out[i, j, 6] = fp8; f_out[i, j, 8] = fp6
            f_out[i, j, 7] = fp9; f_out[i, j, 9] = fp7
            ρ_out[i, j] = one(T)
            ux_out[i, j] = zero(T)
            uy_out[i, j] = zero(T)
        else
            ρ, ux, uy = moments_2d(fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9)
            usq = ux * ux + uy * uy
            f_out[i, j, 1] = fp1 - ω * (fp1 - feq_2d(Val(1), ρ, ux, uy, usq))
            f_out[i, j, 2] = fp2 - ω * (fp2 - feq_2d(Val(2), ρ, ux, uy, usq))
            f_out[i, j, 3] = fp3 - ω * (fp3 - feq_2d(Val(3), ρ, ux, uy, usq))
            f_out[i, j, 4] = fp4 - ω * (fp4 - feq_2d(Val(4), ρ, ux, uy, usq))
            f_out[i, j, 5] = fp5 - ω * (fp5 - feq_2d(Val(5), ρ, ux, uy, usq))
            f_out[i, j, 6] = fp6 - ω * (fp6 - feq_2d(Val(6), ρ, ux, uy, usq))
            f_out[i, j, 7] = fp7 - ω * (fp7 - feq_2d(Val(7), ρ, ux, uy, usq))
            f_out[i, j, 8] = fp8 - ω * (fp8 - feq_2d(Val(8), ρ, ux, uy, usq))
            f_out[i, j, 9] = fp9 - ω * (fp9 - feq_2d(Val(9), ρ, ux, uy, usq))
            ρ_out[i, j] = ρ
            ux_out[i, j] = ux
            uy_out[i, j] = uy
        end
    end
end

@kernel function slbm_bgk_biquad_step_kernel!(f_out, @Const(f_in),
                                                ρ_out, ux_out, uy_out,
                                                @Const(is_solid),
                                                @Const(i_dep), @Const(j_dep),
                                                Nξ, Nη, ω,
                                                periodic_ξ, periodic_η)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f_out)
        fp1 = biquadratic_f(f_in, is_solid, i_dep[i, j, 1], j_dep[i, j, 1], 1, Nξ, Nη, periodic_ξ, periodic_η)
        fp2 = biquadratic_f(f_in, is_solid, i_dep[i, j, 2], j_dep[i, j, 2], 2, Nξ, Nη, periodic_ξ, periodic_η)
        fp3 = biquadratic_f(f_in, is_solid, i_dep[i, j, 3], j_dep[i, j, 3], 3, Nξ, Nη, periodic_ξ, periodic_η)
        fp4 = biquadratic_f(f_in, is_solid, i_dep[i, j, 4], j_dep[i, j, 4], 4, Nξ, Nη, periodic_ξ, periodic_η)
        fp5 = biquadratic_f(f_in, is_solid, i_dep[i, j, 5], j_dep[i, j, 5], 5, Nξ, Nη, periodic_ξ, periodic_η)
        fp6 = biquadratic_f(f_in, is_solid, i_dep[i, j, 6], j_dep[i, j, 6], 6, Nξ, Nη, periodic_ξ, periodic_η)
        fp7 = biquadratic_f(f_in, is_solid, i_dep[i, j, 7], j_dep[i, j, 7], 7, Nξ, Nη, periodic_ξ, periodic_η)
        fp8 = biquadratic_f(f_in, is_solid, i_dep[i, j, 8], j_dep[i, j, 8], 8, Nξ, Nη, periodic_ξ, periodic_η)
        fp9 = biquadratic_f(f_in, is_solid, i_dep[i, j, 9], j_dep[i, j, 9], 9, Nξ, Nη, periodic_ξ, periodic_η)
        if is_solid[i, j]
            f_out[i, j, 1] = fp1
            f_out[i, j, 2] = fp4; f_out[i, j, 4] = fp2
            f_out[i, j, 3] = fp5; f_out[i, j, 5] = fp3
            f_out[i, j, 6] = fp8; f_out[i, j, 8] = fp6
            f_out[i, j, 7] = fp9; f_out[i, j, 9] = fp7
            ρ_out[i, j] = one(T); ux_out[i, j] = zero(T); uy_out[i, j] = zero(T)
        else
            ρ, ux, uy = moments_2d(fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9)
            usq = ux * ux + uy * uy
            f_out[i, j, 1] = fp1 - ω * (fp1 - feq_2d(Val(1), ρ, ux, uy, usq))
            f_out[i, j, 2] = fp2 - ω * (fp2 - feq_2d(Val(2), ρ, ux, uy, usq))
            f_out[i, j, 3] = fp3 - ω * (fp3 - feq_2d(Val(3), ρ, ux, uy, usq))
            f_out[i, j, 4] = fp4 - ω * (fp4 - feq_2d(Val(4), ρ, ux, uy, usq))
            f_out[i, j, 5] = fp5 - ω * (fp5 - feq_2d(Val(5), ρ, ux, uy, usq))
            f_out[i, j, 6] = fp6 - ω * (fp6 - feq_2d(Val(6), ρ, ux, uy, usq))
            f_out[i, j, 7] = fp7 - ω * (fp7 - feq_2d(Val(7), ρ, ux, uy, usq))
            f_out[i, j, 8] = fp8 - ω * (fp8 - feq_2d(Val(8), ρ, ux, uy, usq))
            f_out[i, j, 9] = fp9 - ω * (fp9 - feq_2d(Val(9), ρ, ux, uy, usq))
            ρ_out[i, j] = ρ; ux_out[i, j] = ux; uy_out[i, j] = uy
        end
    end
end

"""
    slbm_bgk_step!(f_out, f_in, ρ, ux, uy, is_solid, geom, ω;
                   interp=:bilinear)

Single fused kernel for semi-Lagrangian BGK on a curvilinear mesh:
interpolated streaming + bounce-back + collision + macroscopic moments.

`interp ∈ (:bilinear, :biquadratic)`. Bilinear is O(Δx²), biquadratic
is O(Δx³) on smooth fields — recommended on distorted meshes (Wilde
et al. 2020). Both reduce to pull-stream exactly on an isotropic
Cartesian mesh.

Arguments match `fused_bgk_step!` plus an `SLBMGeometry` carrying the
precomputed departure indices. `geom.i_dep`, `geom.j_dep` must live on
the same backend as `f_in` (use `transfer_slbm_geometry` if needed).
"""
function slbm_bgk_step!(f_out, f_in, ρ, ux, uy, is_solid,
                        geom::SLBMGeometry, ω;
                        interp::Symbol=:bilinear)
    backend = KernelAbstractions.get_backend(f_in)
    ET = eltype(f_in)
    kernel! = interp === :biquadratic ?
              slbm_bgk_biquad_step_kernel!(backend) :
              slbm_bgk_step_kernel!(backend)
    kernel!(f_out, f_in, ρ, ux, uy, is_solid,
            geom.i_dep, geom.j_dep,
            geom.Nξ, geom.Nη, ET(ω),
            geom.periodic_ξ, geom.periodic_η;
            ndrange=(geom.Nξ, geom.Nη))
end

# ---------------------------------------------------------------------------
# Fused SLBM + MRT kernel. Same SL streaming as the BGK variant, but
# the collision relaxes in moment space with separate rates for
# (e, ε, q, pxx, pxy) — more stable on distorted curvilinear meshes
# (Lallemand & Luo 2000; Budinski 2014 for curvilinear applications).
#
# Conserved moments (ρ, jx, jy) pass through untouched. Kinematic
# viscosity sets s_ν = 1/(3ν + 0.5). Other rates follow KrakeNN's
# existing collide_mrt_2d! defaults (s_e = s_ε = 1.4, s_q = 1.2).
# ---------------------------------------------------------------------------

@kernel function slbm_mrt_step_kernel!(f_out, @Const(f_in),
                                        ρ_out, ux_out, uy_out,
                                        @Const(is_solid),
                                        @Const(i_dep), @Const(j_dep),
                                        Nξ, Nη,
                                        s_e, s_eps, s_q, s_nu,
                                        periodic_ξ, periodic_η)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f_out)
        # 1. Semi-Lagrangian pull for each direction
        f1 = bilinear_f(f_in, i_dep[i, j, 1], j_dep[i, j, 1], 1, Nξ, Nη, periodic_ξ, periodic_η)
        f2 = bilinear_f(f_in, i_dep[i, j, 2], j_dep[i, j, 2], 2, Nξ, Nη, periodic_ξ, periodic_η)
        f3 = bilinear_f(f_in, i_dep[i, j, 3], j_dep[i, j, 3], 3, Nξ, Nη, periodic_ξ, periodic_η)
        f4 = bilinear_f(f_in, i_dep[i, j, 4], j_dep[i, j, 4], 4, Nξ, Nη, periodic_ξ, periodic_η)
        f5 = bilinear_f(f_in, i_dep[i, j, 5], j_dep[i, j, 5], 5, Nξ, Nη, periodic_ξ, periodic_η)
        f6 = bilinear_f(f_in, i_dep[i, j, 6], j_dep[i, j, 6], 6, Nξ, Nη, periodic_ξ, periodic_η)
        f7 = bilinear_f(f_in, i_dep[i, j, 7], j_dep[i, j, 7], 7, Nξ, Nη, periodic_ξ, periodic_η)
        f8 = bilinear_f(f_in, i_dep[i, j, 8], j_dep[i, j, 8], 8, Nξ, Nη, periodic_ξ, periodic_η)
        f9 = bilinear_f(f_in, i_dep[i, j, 9], j_dep[i, j, 9], 9, Nξ, Nη, periodic_ξ, periodic_η)

        if is_solid[i, j]
            f_out[i, j, 1] = f1
            f_out[i, j, 2] = f4; f_out[i, j, 4] = f2
            f_out[i, j, 3] = f5; f_out[i, j, 5] = f3
            f_out[i, j, 6] = f8; f_out[i, j, 8] = f6
            f_out[i, j, 7] = f9; f_out[i, j, 9] = f7
            ρ_out[i, j] = one(T)
            ux_out[i, j] = zero(T)
            uy_out[i, j] = zero(T)
        else
            # Transform to moment space (M · f)
            ρ   = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
            e   = -T(4)*f1 - f2 - f3 - f4 - f5 + T(2)*(f6 + f7 + f8 + f9)
            eps =  T(4)*f1 - T(2)*(f2 + f3 + f4 + f5) + f6 + f7 + f8 + f9
            jx  = f2 - f4 + f6 - f7 - f8 + f9
            qx  = -T(2)*f2 + T(2)*f4 + f6 - f7 - f8 + f9
            jy  = f3 - f5 + f6 + f7 - f8 - f9
            qy  = -T(2)*f3 + T(2)*f5 + f6 + f7 - f8 - f9
            pxx = f2 - f3 + f4 - f5
            pxy = f6 - f7 + f8 - f9

            inv_ρ = one(T) / ρ
            ux = jx * inv_ρ
            uy = jy * inv_ρ
            usq = ux * ux + uy * uy

            # Equilibrium moments
            e_eq   = -T(2)*ρ + T(3)*ρ*usq
            eps_eq = ρ - T(3)*ρ*usq
            qx_eq  = -ρ*ux
            qy_eq  = -ρ*uy
            pxx_eq = ρ*(ux*ux - uy*uy)
            pxy_eq = ρ*ux*uy

            # Relax moments (ρ, jx, jy conserved)
            es  = e   - s_e   * (e   - e_eq)
            ep  = eps - s_eps * (eps - eps_eq)
            qxs = qx  - s_q   * (qx  - qx_eq)
            qys = qy  - s_q   * (qy  - qy_eq)
            ps  = pxx - s_nu  * (pxx - pxx_eq)
            pxys = pxy - s_nu * (pxy - pxy_eq)

            # Transform back (M⁻¹ · m*)
            r = ρ
            f_out[i,j,1] = T(1/9)*r  - T(1/9)*es  + T(1/9)*ep
            f_out[i,j,2] = T(1/9)*r  - T(1/36)*es - T(1/18)*ep + T(1/6)*jx  - T(1/6)*qxs + T(1/4)*ps
            f_out[i,j,3] = T(1/9)*r  - T(1/36)*es - T(1/18)*ep + T(1/6)*jy  - T(1/6)*qys - T(1/4)*ps
            f_out[i,j,4] = T(1/9)*r  - T(1/36)*es - T(1/18)*ep - T(1/6)*jx  + T(1/6)*qxs + T(1/4)*ps
            f_out[i,j,5] = T(1/9)*r  - T(1/36)*es - T(1/18)*ep - T(1/6)*jy  + T(1/6)*qys - T(1/4)*ps
            f_out[i,j,6] = T(1/9)*r  + T(1/18)*es + T(1/36)*ep + T(1/6)*jx  + T(1/12)*qxs + T(1/6)*jy  + T(1/12)*qys + T(1/4)*pxys
            f_out[i,j,7] = T(1/9)*r  + T(1/18)*es + T(1/36)*ep - T(1/6)*jx  - T(1/12)*qxs + T(1/6)*jy  + T(1/12)*qys - T(1/4)*pxys
            f_out[i,j,8] = T(1/9)*r  + T(1/18)*es + T(1/36)*ep - T(1/6)*jx  - T(1/12)*qxs - T(1/6)*jy  - T(1/12)*qys + T(1/4)*pxys
            f_out[i,j,9] = T(1/9)*r  + T(1/18)*es + T(1/36)*ep + T(1/6)*jx  + T(1/12)*qxs - T(1/6)*jy  - T(1/12)*qys - T(1/4)*pxys

            ρ_out[i, j] = ρ
            ux_out[i, j] = ux
            uy_out[i, j] = uy
        end
    end
end

# ---------------------------------------------------------------------------
# SLBM + BGK with moving-wall bounce-back.
#
# When `is_solid[i,j]` is true, populations are reflected and a
# momentum correction is added so that the wall advects the fluid at
# prescribed velocity `(uw_x[i,j], uw_y[i,j])` (Ladd 1994):
#
#   f_q*(wall) = f_{q'}(wall) + 6 · w_q · ρ_w · (c_q · u_wall)
#
# For stationary walls set `uw_x = uw_y = 0` and this reduces exactly
# to `slbm_bgk_step!`. ρ_w is approximated as 1 (low-Mach limit); a
# fluid-side extrapolation can be substituted later if needed.
# ---------------------------------------------------------------------------

@kernel function slbm_bgk_moving_step_kernel!(f_out, @Const(f_in),
                                               ρ_out, ux_out, uy_out,
                                               @Const(is_solid),
                                               @Const(uw_x), @Const(uw_y),
                                               @Const(i_dep), @Const(j_dep),
                                               Nξ, Nη, ω,
                                               periodic_ξ, periodic_η)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f_out)
        fp1 = bilinear_f(f_in, i_dep[i, j, 1], j_dep[i, j, 1], 1, Nξ, Nη, periodic_ξ, periodic_η)
        fp2 = bilinear_f(f_in, i_dep[i, j, 2], j_dep[i, j, 2], 2, Nξ, Nη, periodic_ξ, periodic_η)
        fp3 = bilinear_f(f_in, i_dep[i, j, 3], j_dep[i, j, 3], 3, Nξ, Nη, periodic_ξ, periodic_η)
        fp4 = bilinear_f(f_in, i_dep[i, j, 4], j_dep[i, j, 4], 4, Nξ, Nη, periodic_ξ, periodic_η)
        fp5 = bilinear_f(f_in, i_dep[i, j, 5], j_dep[i, j, 5], 5, Nξ, Nη, periodic_ξ, periodic_η)
        fp6 = bilinear_f(f_in, i_dep[i, j, 6], j_dep[i, j, 6], 6, Nξ, Nη, periodic_ξ, periodic_η)
        fp7 = bilinear_f(f_in, i_dep[i, j, 7], j_dep[i, j, 7], 7, Nξ, Nη, periodic_ξ, periodic_η)
        fp8 = bilinear_f(f_in, i_dep[i, j, 8], j_dep[i, j, 8], 8, Nξ, Nη, periodic_ξ, periodic_η)
        fp9 = bilinear_f(f_in, i_dep[i, j, 9], j_dep[i, j, 9], 9, Nξ, Nη, periodic_ξ, periodic_η)

        if is_solid[i, j]
            ρ_w = one(T)
            uxw = uw_x[i, j]
            uyw = uw_y[i, j]
            # 6 · w_q · ρ · (c_q · u_w). Weights: axis 1/9, diagonals 1/36.
            # So axial δ = (2/3)·ρ·(c_q · u_w), diagonal δ = (1/6)·ρ·(c_q · u_w).
            δx = T(2/3) * ρ_w * uxw         # east-west axis
            δy = T(2/3) * ρ_w * uyw         # north-south axis
            δne = T(1/6) * ρ_w * ( uxw + uyw)
            δnw = T(1/6) * ρ_w * (-uxw + uyw)
            f_out[i, j, 1] = fp1
            f_out[i, j, 2] = fp4 + δx
            f_out[i, j, 4] = fp2 - δx
            f_out[i, j, 3] = fp5 + δy
            f_out[i, j, 5] = fp3 - δy
            f_out[i, j, 6] = fp8 + δne
            f_out[i, j, 8] = fp6 - δne
            f_out[i, j, 7] = fp9 + δnw
            f_out[i, j, 9] = fp7 - δnw
            ρ_out[i, j] = ρ_w
            ux_out[i, j] = uxw
            uy_out[i, j] = uyw
        else
            ρ, ux, uy = moments_2d(fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9)
            usq = ux * ux + uy * uy
            f_out[i, j, 1] = fp1 - ω * (fp1 - feq_2d(Val(1), ρ, ux, uy, usq))
            f_out[i, j, 2] = fp2 - ω * (fp2 - feq_2d(Val(2), ρ, ux, uy, usq))
            f_out[i, j, 3] = fp3 - ω * (fp3 - feq_2d(Val(3), ρ, ux, uy, usq))
            f_out[i, j, 4] = fp4 - ω * (fp4 - feq_2d(Val(4), ρ, ux, uy, usq))
            f_out[i, j, 5] = fp5 - ω * (fp5 - feq_2d(Val(5), ρ, ux, uy, usq))
            f_out[i, j, 6] = fp6 - ω * (fp6 - feq_2d(Val(6), ρ, ux, uy, usq))
            f_out[i, j, 7] = fp7 - ω * (fp7 - feq_2d(Val(7), ρ, ux, uy, usq))
            f_out[i, j, 8] = fp8 - ω * (fp8 - feq_2d(Val(8), ρ, ux, uy, usq))
            f_out[i, j, 9] = fp9 - ω * (fp9 - feq_2d(Val(9), ρ, ux, uy, usq))
            ρ_out[i, j] = ρ
            ux_out[i, j] = ux
            uy_out[i, j] = uy
        end
    end
end

@kernel function slbm_bgk_moving_biquad_step_kernel!(f_out, @Const(f_in),
                                                      ρ_out, ux_out, uy_out,
                                                      @Const(is_solid),
                                                      @Const(uw_x), @Const(uw_y),
                                                      @Const(i_dep), @Const(j_dep),
                                                      Nξ, Nη, ω,
                                                      periodic_ξ, periodic_η)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f_out)
        fp1 = biquadratic_f(f_in, is_solid, i_dep[i, j, 1], j_dep[i, j, 1], 1, Nξ, Nη, periodic_ξ, periodic_η)
        fp2 = biquadratic_f(f_in, is_solid, i_dep[i, j, 2], j_dep[i, j, 2], 2, Nξ, Nη, periodic_ξ, periodic_η)
        fp3 = biquadratic_f(f_in, is_solid, i_dep[i, j, 3], j_dep[i, j, 3], 3, Nξ, Nη, periodic_ξ, periodic_η)
        fp4 = biquadratic_f(f_in, is_solid, i_dep[i, j, 4], j_dep[i, j, 4], 4, Nξ, Nη, periodic_ξ, periodic_η)
        fp5 = biquadratic_f(f_in, is_solid, i_dep[i, j, 5], j_dep[i, j, 5], 5, Nξ, Nη, periodic_ξ, periodic_η)
        fp6 = biquadratic_f(f_in, is_solid, i_dep[i, j, 6], j_dep[i, j, 6], 6, Nξ, Nη, periodic_ξ, periodic_η)
        fp7 = biquadratic_f(f_in, is_solid, i_dep[i, j, 7], j_dep[i, j, 7], 7, Nξ, Nη, periodic_ξ, periodic_η)
        fp8 = biquadratic_f(f_in, is_solid, i_dep[i, j, 8], j_dep[i, j, 8], 8, Nξ, Nη, periodic_ξ, periodic_η)
        fp9 = biquadratic_f(f_in, is_solid, i_dep[i, j, 9], j_dep[i, j, 9], 9, Nξ, Nη, periodic_ξ, periodic_η)
        if is_solid[i, j]
            ρ_w = one(T); uxw = uw_x[i, j]; uyw = uw_y[i, j]
            δx  = T(2/3) * ρ_w * uxw
            δy  = T(2/3) * ρ_w * uyw
            δne = T(1/6) * ρ_w * ( uxw + uyw)
            δnw = T(1/6) * ρ_w * (-uxw + uyw)
            f_out[i, j, 1] = fp1
            f_out[i, j, 2] = fp4 + δx; f_out[i, j, 4] = fp2 - δx
            f_out[i, j, 3] = fp5 + δy; f_out[i, j, 5] = fp3 - δy
            f_out[i, j, 6] = fp8 + δne; f_out[i, j, 8] = fp6 - δne
            f_out[i, j, 7] = fp9 + δnw; f_out[i, j, 9] = fp7 - δnw
            ρ_out[i, j] = ρ_w; ux_out[i, j] = uxw; uy_out[i, j] = uyw
        else
            ρ, ux, uy = moments_2d(fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9)
            usq = ux * ux + uy * uy
            f_out[i, j, 1] = fp1 - ω * (fp1 - feq_2d(Val(1), ρ, ux, uy, usq))
            f_out[i, j, 2] = fp2 - ω * (fp2 - feq_2d(Val(2), ρ, ux, uy, usq))
            f_out[i, j, 3] = fp3 - ω * (fp3 - feq_2d(Val(3), ρ, ux, uy, usq))
            f_out[i, j, 4] = fp4 - ω * (fp4 - feq_2d(Val(4), ρ, ux, uy, usq))
            f_out[i, j, 5] = fp5 - ω * (fp5 - feq_2d(Val(5), ρ, ux, uy, usq))
            f_out[i, j, 6] = fp6 - ω * (fp6 - feq_2d(Val(6), ρ, ux, uy, usq))
            f_out[i, j, 7] = fp7 - ω * (fp7 - feq_2d(Val(7), ρ, ux, uy, usq))
            f_out[i, j, 8] = fp8 - ω * (fp8 - feq_2d(Val(8), ρ, ux, uy, usq))
            f_out[i, j, 9] = fp9 - ω * (fp9 - feq_2d(Val(9), ρ, ux, uy, usq))
            ρ_out[i, j] = ρ; ux_out[i, j] = ux; uy_out[i, j] = uy
        end
    end
end

"""
    slbm_bgk_moving_step!(f_out, f_in, ρ, ux, uy, is_solid,
                          uw_x, uw_y, geom, ω; interp=:bilinear)

Fused SLBM + BGK step with Ladd (1994) moving-wall bounce-back.
`uw_x`, `uw_y` are per-cell prescribed wall velocities (nonzero only on
solid cells; ignored on fluid cells). `interp ∈ (:bilinear, :biquadratic)`.
Reduces to `slbm_bgk_step!` when both wall velocities are zero.
"""
function slbm_bgk_moving_step!(f_out, f_in, ρ, ux, uy, is_solid,
                                uw_x, uw_y, geom::SLBMGeometry, ω;
                                interp::Symbol=:bilinear)
    backend = KernelAbstractions.get_backend(f_in)
    ET = eltype(f_in)
    kernel! = interp === :biquadratic ?
              slbm_bgk_moving_biquad_step_kernel!(backend) :
              slbm_bgk_moving_step_kernel!(backend)
    kernel!(f_out, f_in, ρ, ux, uy, is_solid,
            uw_x, uw_y,
            geom.i_dep, geom.j_dep,
            geom.Nξ, geom.Nη, ET(ω),
            geom.periodic_ξ, geom.periodic_η;
            ndrange=(geom.Nξ, geom.Nη))
end

"""
    slbm_mrt_step!(f_out, f_in, ρ, ux, uy, is_solid, geom, ν;
                   s_e=1.4, s_eps=1.4, s_q=1.2)

Fused SLBM + MRT collision in one GPU dispatch. `ν` is the kinematic
viscosity (lattice units); the stress rate is computed as
`s_ν = 1/(3ν + 0.5)`.
"""
function slbm_mrt_step!(f_out, f_in, ρ, ux, uy, is_solid,
                        geom::SLBMGeometry, ν;
                        s_e=1.4, s_eps=1.4, s_q=1.2)
    backend = KernelAbstractions.get_backend(f_in)
    ET = eltype(f_in)
    s_nu = ET(1.0 / (3.0 * ν + 0.5))
    kernel! = slbm_mrt_step_kernel!(backend)
    kernel!(f_out, f_in, ρ, ux, uy, is_solid,
            geom.i_dep, geom.j_dep,
            geom.Nξ, geom.Nη,
            ET(s_e), ET(s_eps), ET(s_q), s_nu,
            geom.periodic_ξ, geom.periodic_η;
            ndrange=(geom.Nξ, geom.Nη))
end

# ===========================================================================
# Local relaxation rate for SLBM on non-uniform meshes.
#
# On a stretched mesh, the physical cell size varies. The collision τ must
# be adjusted per cell to maintain the target physical viscosity ν:
#
#   τ_local = r * (τ_ref − 0.5) + 0.5
#
# where r = Δx_local / dx_ref is the local-to-reference cell size ratio.
# This is the same Filippova-Hänel rescaling used in grid refinement.
# ===========================================================================

"""
    compute_local_omega_2d(mesh; ν, Λ=3/16, scaling=:quadratic, τ_floor=0.5)
        -> (s_plus_field, s_minus_field)

Precompute per-cell TRT relaxation rates on a curvilinear mesh, accounting
for the local cell size variation. Returns 2D arrays `s_plus[Nξ, Nη]` and
`s_minus[Nξ, Nη]`.

On meshes with aggressive stretching (cell-size ratio ≳ 10×), quadratic
SLBM rescaling drives `τ → 0.5` on the largest cells, which destabilises
TRT. The `τ_floor` kwarg clamps `τ_local ≥ τ_floor`, trading a small
excess numerical viscosity on coarse cells (where resolution is already
poor so it matters little) for stability. Set `τ_floor=0.5` (default) to
disable the clamp — identical to pre-clamp behaviour.

The local cell size is estimated as
`√(dXdξ² + dYdξ²) · Δξ · √(dXdη² + dYdη²) · Δη`.
"""
function compute_local_omega_2d(mesh::CurvilinearMesh{T}; ν::Real, Λ::Real=3/16,
                                 scaling::Symbol=:quadratic,
                                 τ_floor::Real=0.5) where {T}
    Nξ, Nη = mesh.Nξ, mesh.Nη
    denom_ξ = T(mesh.periodic_ξ ? Nξ : Nξ - 1)
    denom_η = T(mesh.periodic_η ? Nη : Nη - 1)
    Δξ = one(T) / denom_ξ
    Δη = one(T) / denom_η

    s_plus_ref, s_minus_ref = trt_rates(ν; Λ=Λ)
    τ_plus_ref  = one(T) / T(s_plus_ref)
    τ_minus_ref = one(T) / T(s_minus_ref)
    τ_lo = T(τ_floor)

    sp = zeros(T, Nξ, Nη)
    sm = zeros(T, Nξ, Nη)

    @inbounds for j in 1:Nη, i in 1:Nξ
        lξ = sqrt(mesh.dXdξ[i, j]^2 + mesh.dYdξ[i, j]^2) * Δξ
        lη = sqrt(mesh.dXdη[i, j]^2 + mesh.dYdη[i, j]^2) * Δη
        dx_local = sqrt(lξ * lη)
        r = dx_local / mesh.dx_ref

        if scaling === :quadratic
            # SLBM with global dt: ν = (τ-0.5)/3 * dx²/dt. For uniform ν
            # with varying dx: τ_local - 0.5 = (dx_ref/dx_local)² * (τ_ref - 0.5)
            r_inv2 = (mesh.dx_ref / dx_local)^2
            τ_plus_local  = r_inv2 * (τ_plus_ref  - T(0.5)) + T(0.5)
            τ_minus_local = r_inv2 * (τ_minus_ref - T(0.5)) + T(0.5)
        elseif scaling === :none
            # Uniform τ (no local rescaling). Use this for diagnostic
            # comparison or when local-CFL is disabled.
            τ_plus_local  = τ_plus_ref
            τ_minus_local = τ_minus_ref
        else  # :linear (Filippova-Hänel, for refinement with Δt ∝ Δx)
            τ_plus_local  = r * (τ_plus_ref  - T(0.5)) + T(0.5)
            τ_minus_local = r * (τ_minus_ref - T(0.5)) + T(0.5)
        end
        sp[i, j] = one(T) / max(τ_plus_local,  τ_lo)
        sm[i, j] = one(T) / max(τ_minus_local, τ_lo)
    end

    return sp, sm
end

# ===========================================================================
# q_wall precomputation for SLBM + LI-BB on curvilinear meshes.
#
# On a body-fitted mesh (e.g. O-grid), solid nodes sit on the mesh
# boundary (j=1 inner wall, j=Nη outer wall). For each fluid node and
# each D2Q9 direction q, we check whether the computational neighbor
# (i+cx_q, j+cy_q) is solid. If so, the link is cut and we compute
# q_w = |P_fluid → P_wall| / |P_fluid → P_neighbor| in physical space,
# where P_wall is the intersection of the segment with the body surface.
# ===========================================================================

"""
    precompute_q_wall_slbm_cylinder_2d(mesh, is_solid, cx, cy, R;
        omega_inner=0, FT=Float64) -> (q_wall, uw_link_x, uw_link_y)

Precompute LI-BB cut fractions on a curvilinear mesh for a cylindrical
body centered at `(cx, cy)` with radius `R` in physical space.

For each fluid node `(i,j)` and direction `q`, if the computational
neighbor `(i+cxq, j+cyq)` is solid, we ray-intersect the physical-space
segment from the fluid node to the neighbor with the circle. The wall
velocity for a rotating inner cylinder is `u_wall = omega_inner × r`.
"""
function precompute_q_wall_slbm_cylinder_2d(
        mesh::CurvilinearMesh{T},
        is_solid::AbstractMatrix{Bool},
        cx_body::Real, cy_body::Real, R_body::Real;
        omega_inner::Real=0,
        FT::Type{<:AbstractFloat}=T) where {T}

    Nξ, Nη = mesh.Nξ, mesh.Nη
    cxs = velocities_x(D2Q9())
    cys = velocities_y(D2Q9())

    q_wall    = zeros(FT, Nξ, Nη, 9)
    uw_link_x = zeros(FT, Nξ, Nη, 9)
    uw_link_y = zeros(FT, Nξ, Nη, 9)

    cx_b, cy_b, R_b = FT(cx_body), FT(cy_body), FT(R_body)
    R² = R_b * R_b
    Ω = FT(omega_inner)

    @inbounds for j in 1:Nη, i in 1:Nξ
        is_solid[i, j] && continue

        Xf = mesh.X[i, j]
        Yf = mesh.Y[i, j]

        for q in 2:9
            diq = cxs[q]
            djq = cys[q]
            in_ = i + diq
            jn_ = j + djq

            if mesh.periodic_ξ
                in_ = ((in_ - 1) % Nξ + Nξ) % Nξ + 1
            end

            (in_ < 1 || in_ > Nξ || jn_ < 1 || jn_ > Nη) && continue
            is_solid[in_, jn_] || continue

            Xn = mesh.X[in_, jn_]
            Yn = mesh.Y[in_, jn_]

            # Ray-circle intersection: P(t) = P_fluid + t*(P_neighbor - P_fluid)
            dx = Xn - Xf
            dy = Yn - Yf
            fx = Xf - cx_b
            fy = Yf - cy_b

            a = dx * dx + dy * dy
            b_coeff = FT(2) * (fx * dx + fy * dy)
            c_coeff = fx * fx + fy * fy - R²
            disc = b_coeff * b_coeff - FT(4) * a * c_coeff

            if disc < zero(FT)
                q_wall[i, j, q] = FT(0.5)
            else
                sd = sqrt(disc)
                t1 = (-b_coeff - sd) / (FT(2) * a)
                t2 = (-b_coeff + sd) / (FT(2) * a)
                t = (t1 > zero(FT) && t1 ≤ one(FT)) ? t1 :
                    (t2 > zero(FT) && t2 ≤ one(FT)) ? t2 : FT(0.5)
                q_wall[i, j, q] = t
            end

            if !iszero(Ω)
                tw = q_wall[i, j, q]
                Xw = Xf + tw * dx
                Yw = Yf + tw * dy
                uw_link_x[i, j, q] = -Ω * (Yw - cy_b)
                uw_link_y[i, j, q] =  Ω * (Xw - cx_b)
            end
        end
    end

    return q_wall, uw_link_x, uw_link_y
end

# ---------------------------------------------------------------------------
# Fused SLBM + TRT + LI-BB kernel.
# ---------------------------------------------------------------------------

@kernel function slbm_trt_libb_step_kernel!(f_out, @Const(f_in),
                                              ρ_out, ux_out, uy_out,
                                              @Const(is_solid),
                                              @Const(q_wall),
                                              @Const(uw_link_x), @Const(uw_link_y),
                                              @Const(i_dep), @Const(j_dep),
                                              Nξ, Nη, s_plus, s_minus,
                                              periodic_ξ, periodic_η)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f_out)

        fp1 = bilinear_f(f_in, i_dep[i, j, 1], j_dep[i, j, 1], 1, Nξ, Nη, periodic_ξ, periodic_η)
        fp2 = bilinear_f(f_in, i_dep[i, j, 2], j_dep[i, j, 2], 2, Nξ, Nη, periodic_ξ, periodic_η)
        fp3 = bilinear_f(f_in, i_dep[i, j, 3], j_dep[i, j, 3], 3, Nξ, Nη, periodic_ξ, periodic_η)
        fp4 = bilinear_f(f_in, i_dep[i, j, 4], j_dep[i, j, 4], 4, Nξ, Nη, periodic_ξ, periodic_η)
        fp5 = bilinear_f(f_in, i_dep[i, j, 5], j_dep[i, j, 5], 5, Nξ, Nη, periodic_ξ, periodic_η)
        fp6 = bilinear_f(f_in, i_dep[i, j, 6], j_dep[i, j, 6], 6, Nξ, Nη, periodic_ξ, periodic_η)
        fp7 = bilinear_f(f_in, i_dep[i, j, 7], j_dep[i, j, 7], 7, Nξ, Nη, periodic_ξ, periodic_η)
        fp8 = bilinear_f(f_in, i_dep[i, j, 8], j_dep[i, j, 8], 8, Nξ, Nη, periodic_ξ, periodic_η)
        fp9 = bilinear_f(f_in, i_dep[i, j, 9], j_dep[i, j, 9], 9, Nξ, Nη, periodic_ξ, periodic_η)

        if is_solid[i, j]
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
        else
            # LI-BB pre-phase: Bouzidi interpolation on cut links.
            # Uses _libb_branch(q_w, f_post_here, f_post_back, f_bar_post_here, δ)
            # which does linear interpolation with the true q_w fraction.
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

            # Moments
            ρ = fp1 + fp2 + fp3 + fp4 + fp5 + fp6 + fp7 + fp8 + fp9
            inv_ρ = one(T) / ρ
            ux = (fp2 - fp4 + fp6 - fp7 - fp8 + fp9) * inv_ρ
            uy = (fp3 - fp5 + fp6 + fp7 - fp8 - fp9) * inv_ρ
            usq = ux * ux + uy * uy

            # TRT collision
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

            ρ_out[i, j] = ρ
            ux_out[i, j] = ux
            uy_out[i, j] = uy
        end
    end
end

"""
    slbm_trt_libb_step!(f_out, f_in, ρ, ux, uy, is_solid,
                         q_wall, uw_link_x, uw_link_y,
                         geom, ν; Λ=3/16)

Fused SLBM + TRT + LI-BB step on a curvilinear mesh. Combines semi-
Lagrangian streaming, LI-BB pre-phase substitution on cut links, and
TRT collision with magic parameter Λ.
"""
function slbm_trt_libb_step!(f_out, f_in, ρ, ux, uy, is_solid,
                              q_wall, uw_link_x, uw_link_y,
                              geom::SLBMGeometry, ν; Λ::Real=3/16)
    backend = KernelAbstractions.get_backend(f_in)
    ET = eltype(f_in)
    s_plus, s_minus = trt_rates(ν; Λ=Λ)
    kernel! = slbm_trt_libb_step_kernel!(backend)
    kernel!(f_out, f_in, ρ, ux, uy, is_solid,
            q_wall, uw_link_x, uw_link_y,
            geom.i_dep, geom.j_dep,
            geom.Nξ, geom.Nη, ET(s_plus), ET(s_minus),
            geom.periodic_ξ, geom.periodic_η;
            ndrange=(geom.Nξ, geom.Nη))
end

# ---------------------------------------------------------------------------
# Local-τ variant (2D, D2Q9): per-cell s_plus[i,j], s_minus[i,j] for SLBM on
# stretched curvilinear meshes. Pure DSL-brick assembly — mirror of the 3D
# `slbm_trt_libb_step_local_3d!` (src/curvilinear/slbm_3d.jl). Required when
# the physical cell size varies across the mesh (e.g. gmsh Bump, O-grid),
# where a single global τ breaks down: small cells → τ_eff → 0.5 (unstable),
# large cells → ν_eff too high (biased drag).
#
#   PullSLBM → SolidInert → ApplyLiBBPrePhase →
#   Moments → CollideTRTLocalDirect → WriteMoments
# ---------------------------------------------------------------------------

const _SLBM_TRT_LIBB_LOCAL_SPEC_2D = LBMSpec(
    PullSLBM(), SolidInert(),
    ApplyLiBBPrePhase(),
    Moments(), CollideTRTLocalDirect(),
    WriteMoments();
    stencil = :D2Q9,
)

const _SLBM_TRT_LIBB_LOCAL_BIQUAD_SPEC_2D = LBMSpec(
    PullSLBMBiquad(), SolidInert(),
    ApplyLiBBPrePhase(),
    Moments(), CollideTRTLocalDirect(),
    WriteMoments();
    stencil = :D2Q9,
)

const _SLBM_REG_LIBB_LOCAL_SPEC_2D = LBMSpec(
    PullSLBM(), SolidInert(),
    ApplyLiBBPrePhase(),
    Moments(), CollideRegularizedTRTLocal(),
    WriteMoments();
    stencil = :D2Q9,
)

"""
    slbm_trt_libb_step_local_2d!(f_out, f_in, ρ, ux, uy, is_solid,
                                  q_wall, uw_link_x, uw_link_y,
                                  geom, sp_field, sm_field)

SLBM + TRT + LI-BB step (D2Q9) with PER-CELL relaxation rates from the
device-side 2D arrays `sp_field[Nξ, Nη]` and `sm_field[Nξ, Nη]` (e.g.
produced by `compute_local_omega_2d` with `scaling=:quadratic` and
copied to the backend). Required on stretched curvilinear meshes where
the local cell size — and therefore τ — varies per cell.
"""
function slbm_trt_libb_step_local_2d!(f_out, f_in, ρ, ux, uy, is_solid,
                                        q_wall, uw_link_x, uw_link_y,
                                        geom::SLBMGeometry,
                                        sp_field, sm_field)
    backend = KernelAbstractions.get_backend(f_in)
    kernel! = build_lbm_kernel(backend, _SLBM_TRT_LIBB_LOCAL_SPEC_2D)
    kernel!(f_out, ρ, ux, uy, f_in, is_solid,
            q_wall, uw_link_x, uw_link_y,
            geom.i_dep, geom.j_dep,
            geom.Nξ, geom.Nη,
            sp_field, sm_field,
            geom.periodic_ξ, geom.periodic_η;
            ndrange=(geom.Nξ, geom.Nη))
end

"""
    slbm_reg_libb_step_local_2d!(...)

SLBM + Regularized TRT + LI-BB step (D2Q9). Uses Latt & Chopard (2006)
regularization: reconstructs f_neq from the stress tensor Π⁽¹⁾ only,
filtering ghost modes. Stable at τ→0.5 on stretched meshes where the
standard TRT/BGK collision diverges.
"""
function slbm_reg_libb_step_local_2d!(f_out, f_in, ρ, ux, uy, is_solid,
                                        q_wall, uw_link_x, uw_link_y,
                                        geom::SLBMGeometry,
                                        sp_field, sm_field)
    backend = KernelAbstractions.get_backend(f_in)
    kernel! = build_lbm_kernel(backend, _SLBM_REG_LIBB_LOCAL_SPEC_2D)
    kernel!(f_out, ρ, ux, uy, f_in, is_solid,
            q_wall, uw_link_x, uw_link_y,
            geom.i_dep, geom.j_dep,
            geom.Nξ, geom.Nη,
            sp_field, sm_field,
            geom.periodic_ξ, geom.periodic_η;
            ndrange=(geom.Nξ, geom.Nη))
end

"""
    slbm_trt_libb_step_local_biquad_2d!(f_out, f_in, ρ, ux, uy, is_solid,

Public function in the curvilinear mesh and SLBM API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.slbm_trt_libb_step_local_biquad_2d!)
```
"""
function slbm_trt_libb_step_local_biquad_2d!(f_out, f_in, ρ, ux, uy, is_solid,
                                               q_wall, uw_link_x, uw_link_y,
                                               geom::SLBMGeometry,
                                               sp_field, sm_field)
    backend = KernelAbstractions.get_backend(f_in)
    kernel! = build_lbm_kernel(backend, _SLBM_TRT_LIBB_LOCAL_BIQUAD_SPEC_2D)
    kernel!(f_out, ρ, ux, uy, f_in, is_solid,
            q_wall, uw_link_x, uw_link_y,
            geom.i_dep, geom.j_dep,
            geom.Nξ, geom.Nη,
            sp_field, sm_field,
            geom.periodic_ξ, geom.periodic_η;
            ndrange=(geom.Nξ, geom.Nη))
end
