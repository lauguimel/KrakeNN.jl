# --- Steady adjoint of the 3D D3Q19 GNF step (part 1/2): forward map and linearisation ---
#
# 3D counterpart of src/ad/gnf_adjoint.jl (which holds the method overview).
# Scope v1: halfway bounce-back walls (q_w = 1/2, static) and voxel obstacles,
# body-force driven, every axis periodic; no Zou-He, no curved LI-BB.
#
# Lattice. The adjoint works on the reduced periodic lattice (nx, ny, nz) with a
# voxel mask `is_solid`; walls are solid planes. The production step
# `fused_trt_gnf_libb_guo_field_step_3d!` runs on the padded lattice
# (nx+2, ny+2, nz+2), the ghosts refilled by `_wrap_gnf_periodic_3d!` on every
# axis after each step and q_wall = 1/2 on every link from a fluid node to a
# solid one: on the interior this is exactly a periodic step on the reduced
# lattice (`run_gnf3d_periodic_steady`).
#
# State: w = (f, τ⁺) post-collision, flat vector of length n = 20 N
# (N = nx ny nz): component (i, j, k, q) at i + nx (j−1) + nx ny (k−1) + N (q−1),
# q = 1..19 the populations, q = 20 the even relaxation time τ⁺ = 1/s⁺ (the
# kernel stores s⁺; τ⁺ is the same state in another variable). One step:
#
#   arrivals fa = S f: fa_q(x) = f_q(x − c_q) if x − c_q is fluid, else
#                      f_q̄(x) (halfway bounce-back, `_libb_branch` at q_w = 1/2)
#   fluid node x: (f'_x, τ⁺'_x) = H(fa_x, τ⁺_x), H = TRT + Guo at ν(I1(fa_x, τ⁺_x))
#   solid node:   f' = rest equilibrium (constant), τ⁺ untouched
#
# so S restricted to fluid populations is a permutation, solid components are
# constants and never read by a fluid node. Active space = the 20 components of
# every fluid node; A = ∂G/∂w = Bdiag · S maps it into itself.
#
# Local blocks: as in 2D, ν = ν_c + κ_c (I1 − I1_c) + δν with I1 = S:S
# (γ̇ = √(2 I1) in the kernel), κ_c = dν/dI1 by ForwardDiff of
# `gnf_viscosity(closure, I1)`, and the 20 × 21 ForwardDiff Jacobian of the cell
# map in (fa_1..fa_19, τ⁺, δν). s⁻ follows from ν (TRT magic parameter Λ).
#
# Gauge: mass is conserved (TRT and the Guo source conserve Σ_q f for every ν),
# c = 1 on fluid populations, 0 on τ⁺, satisfies Aᵀ c = c; the adjoint system
# is bordered with ĉ = c/‖c‖ as in 2D.

const _G3_CX = (0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0)
const _G3_CY = (0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1)
const _G3_CZ = (0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1)
const _G3_OPP = (1, 3, 2, 5, 4, 7, 6, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16)

# ============================================================
# Indexing and streaming permutation of the reduced periodic lattice
# ============================================================

@inline _gnf3d_lin(i, j, k, q, nx, ny, nz) = i + nx * ((j - 1) + ny * ((k - 1) + nz * (q - 1)))

# Linear index of the stored component read by arrival slot q at node (i, j, k)
# (pull from x − c_q with periodic wrap; halfway bounce-back f_q̄(x) when x − c_q is solid).
@inline function _gnf3d_src(i, j, k, q, nx, ny, nz, solid)
    q == 1 && return _gnf3d_lin(i, j, k, 1, nx, ny, nz)
    @inbounds begin
        im = mod1(i - _G3_CX[q], nx)
        jm = mod1(j - _G3_CY[q], ny)
        km = mod1(k - _G3_CZ[q], nz)
        s = solid[im, jm, km]
        return s ? _gnf3d_lin(i, j, k, _G3_OPP[q], nx, ny, nz) :
                   _gnf3d_lin(im, jm, km, q, nx, ny, nz)
    end
end

"""
    gnf3d_pack_state(f, tau) -> w

Flat state vector `[vec(f); vec(tau)]` (length 20 nx ny nz) of the 3D step on
the reduced lattice: `f` (nx, ny, nz, 19) post-collision populations, `tau`
(nx, ny, nz) the even relaxation time τ⁺ = 1/s⁺.
"""
gnf3d_pack_state(f, tau) = vcat(vec(f), vec(tau))

"""
    gnf3d_unpack_state(w, nx, ny, nz) -> (f, tau)

Copies of the populations (nx, ny, nz, 19) and τ⁺ (nx, ny, nz) held in a flat
3D state vector.
"""
function gnf3d_unpack_state(w, nx::Int, ny::Int, nz::Int)
    N = nx * ny * nz
    return reshape(w[1:19N], nx, ny, nz, 19), reshape(w[19N+1:20N], nx, ny, nz)
end

# ============================================================
# Cell map (host, generic number type: Float64 or ForwardDiff duals)
# ============================================================

# Moments, half-force-shifted velocity, equilibria and Guo-corrected stress
# moment of the arrivals, as `CollideTRTGNFGuoField_3D`.
@inline function _gnf3d_moments_pi(fa::SVector{19,T}, Fx, Fy, Fz) where {T}
    ρ, ux0, uy0, uz0 = moments_3d(fa[1], fa[2], fa[3], fa[4], fa[5], fa[6], fa[7], fa[8],
                                  fa[9], fa[10], fa[11], fa[12], fa[13], fa[14], fa[15],
                                  fa[16], fa[17], fa[18], fa[19])
    fx = T(Fx); fy = T(Fy); fz = T(Fz)
    ux = (ρ * ux0 + fx / 2) / ρ
    uy = (ρ * uy0 + fy / 2) / ρ
    uz = (ρ * uz0 + fz / 2) / ρ
    usq = ux * ux + uy * uy + uz * uz
    feq = ntuple(q -> feq_3d(Val(q), ρ, ux, uy, uz, usq), Val(19))
    n = ntuple(q -> fa[q] - feq[q], Val(19))
    Pxx = n[2] + n[3] + n[8] + n[9] + n[10] + n[11] + n[12] + n[13] + n[14] + n[15]
    Pyy = n[4] + n[5] + n[8] + n[9] + n[10] + n[11] + n[16] + n[17] + n[18] + n[19]
    Pzz = n[6] + n[7] + n[12] + n[13] + n[14] + n[15] + n[16] + n[17] + n[18] + n[19]
    Pxy = n[8] - n[9] - n[10] + n[11]
    Pxz = n[12] - n[13] - n[14] + n[15]
    Pyz = n[16] - n[17] - n[18] + n[19]
    Pxx += fx * ux; Pyy += fy * uy; Pzz += fz * uz
    Pxy += (fx * uy + ux * fy) / 2
    Pxz += (fx * uz + ux * fz) / 2
    Pyz += (fy * uz + uy * fz) / 2
    return ρ, (ux, uy, uz), feq, (Pxx, Pyy, Pzz, Pxy, Pxz, Pyz), (fx, fy, fz)
end

# I1 = S:S with S = −3 Π / (2 ρ τ⁺_prev) (`strain_rate_from_pi_3d` gives √(2 I1)).
@inline function _gnf3d_invariant(P, ρ, τ)
    c = -3 / (2 * ρ * τ)
    Sxx = P[1] * c; Syy = P[2] * c; Szz = P[3] * c
    Sxy = P[4] * c; Sxz = P[5] * c; Syz = P[6] * c
    return Sxx * Sxx + Syy * Syy + Szz * Szz + 2 * (Sxy * Sxy + Sxz * Sxz + Syz * Syz)
end

@inline function _g3pair(fp, fq, ep, eq, a, b, gs, gm)
    return (fp - a * (fp - ep) - b * (fq - eq) + gs + gm,
            fq - a * (fq - eq) - b * (fp - ep) + gs - gm)
end

# TRT + Guo collision of `CollideTRTGNFGuoField_3D` at viscosity ν; returns the
# 19 post-collision populations and τ⁺' = 3ν + 1/2.
@inline function _gnf3d_collide(fa::SVector{19}, u, feq, F, ν, Λ)
    ux, uy, uz = u; fx, fy, fz = F
    sp = 1 / (3 * ν + 0.5)
    sm = 1 / (Λ / (3 * ν) + 0.5)
    a = (sp + sm) / 2; b = (sp - sm) / 2
    ge = 1 - sp / 2; go = 1 - sm / 2
    wr = 1 / 3; wa = 1 / 18; we = 1 / 36
    uF = ux * fx + uy * fy + uz * fz
    Sp1 = wr * (-uF) * 3
    f1 = fa[1] - sp * (fa[1] - feq[1]) + ge * Sp1
    f2, f3 = _g3pair(fa[2], fa[3], feq[2], feq[3], a, b,
                     ge * wa * (ux * fx * 9 - uF * 3), go * wa * fx * 3)
    f4, f5 = _g3pair(fa[4], fa[5], feq[4], feq[5], a, b,
                     ge * wa * (uy * fy * 9 - uF * 3), go * wa * fy * 3)
    f6, f7 = _g3pair(fa[6], fa[7], feq[6], feq[7], a, b,
                     ge * wa * (uz * fz * 9 - uF * 3), go * wa * fz * 3)
    f8, f11 = _g3pair(fa[8], fa[11], feq[8], feq[11], a, b,
                      ge * we * ((ux + uy) * (fx + fy) * 9 - uF * 3), go * we * (fx + fy) * 3)
    f9, f10 = _g3pair(fa[9], fa[10], feq[9], feq[10], a, b,
                      ge * we * ((-ux + uy) * (-fx + fy) * 9 - uF * 3), go * we * (-fx + fy) * 3)
    f12, f15 = _g3pair(fa[12], fa[15], feq[12], feq[15], a, b,
                       ge * we * ((ux + uz) * (fx + fz) * 9 - uF * 3), go * we * (fx + fz) * 3)
    f13, f14 = _g3pair(fa[13], fa[14], feq[13], feq[14], a, b,
                       ge * we * ((-ux + uz) * (-fx + fz) * 9 - uF * 3), go * we * (-fx + fz) * 3)
    f16, f19 = _g3pair(fa[16], fa[19], feq[16], feq[19], a, b,
                       ge * we * ((uy + uz) * (fy + fz) * 9 - uF * 3), go * we * (fy + fz) * 3)
    f17, f18 = _g3pair(fa[17], fa[18], feq[17], feq[18], a, b,
                       ge * we * ((-uy + uz) * (-fy + fz) * 9 - uF * 3), go * we * (-fy + fz) * 3)
    fo = SVector{19}(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
                     f16, f17, f18, f19)
    return fo, 3 * ν + 0.5
end

"""
    gnf3d_cell_update(fa, tau_prev, F, closure, Λ=3/16) -> (f_out, tau_new, I1, nu)

Host reference of one fluid-node update of the 3D GNF step
(`CollideTRTGNFGuoField_3D`): arrivals `fa::SVector{19}`, previous even
relaxation time τ⁺ `tau_prev`, body force `F = (Fx, Fy, Fz)` (lattice units),
viscosity ν = `gnf_viscosity(closure, I1)` at I1 = S:S. Returns the 19
post-collision populations, τ⁺' = 3ν + 1/2, I1 and ν.
"""
@inline function gnf3d_cell_update(fa::SVector{19}, tau_prev, F, closure, Λ=3 / 16)
    ρ, u, feq, P, Fc = _gnf3d_moments_pi(fa, F[1], F[2], F[3])
    I1 = _gnf3d_invariant(P, ρ, tau_prev)
    ν = gnf_viscosity(closure, I1)
    fo, τn = _gnf3d_collide(fa, u, feq, Fc, ν, Λ)
    return fo, τn, I1, ν
end

# Cell map of the linearised step in x = (fa_1..fa_19, τ⁺_prev, δν).
@inline function _gnf3d_lin_cell(x::SVector{21}, F, ν_c, κ_c, I1_c, Λ)
    fa = SVector{19}(ntuple(q -> x[q], Val(19)))
    ρ, u, feq, P, Fc = _gnf3d_moments_pi(fa, F[1], F[2], F[3])
    I1 = _gnf3d_invariant(P, ρ, x[20])
    ν = ν_c + κ_c * (I1 - I1_c) + x[21]
    fo, τn = _gnf3d_collide(fa, u, feq, Fc, ν, Λ)
    return vcat(fo, SVector(τn))
end

# ============================================================
# Forward: production kernel on the padded periodic lattice
# ============================================================

_gnf3d_force3(F::Real) = (Float64(F), 0.0, 0.0)
_gnf3d_force3(F::Tuple) = (length(F) == 3 || throw(ArgumentError("F must be (Fx, Fy, Fz)"));
                           (Float64(F[1]), Float64(F[2]), Float64(F[3])))

function _gnf3d_periodic_setup(is_solid::AbstractArray{Bool,3}, closure, F, nu_init, Λ,
                               f_init, tau_init, backend, FT)
    nx, ny, nz = size(is_solid)
    dims = (nx + 2, ny + 2, nz + 2)
    Fx, Fy, Fz = _gnf3d_force3(F)
    model = _gnf3d_rheology(FT, closure)
    solid_r = Array{Bool}(is_solid)
    solid_p = [solid_r[mod1(i - 1, nx), mod1(j - 1, ny), mod1(k - 1, nz)]
               for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3]]
    q_wall = zeros(FT, dims..., 19)
    for k in 2:nz+1, j in 2:ny+1, i in 2:nx+1
        solid_p[i, j, k] && continue
        for q in 2:19
            solid_p[i + _G3_CX[q], j + _G3_CY[q], k + _G3_CZ[q]] && (q_wall[i, j, k, q] = FT(0.5))
        end
    end
    sp0 = inv(FT(3) * FT(nu_init) + FT(0.5))
    sm0 = inv(FT(Λ) / (FT(3) * FT(nu_init)) + FT(0.5))
    sp = fill(sp0, dims); sm = fill(sm0, dims)
    f = zeros(FT, dims..., 19)
    fh = f_init === nothing ? nothing : Array(f_init)
    th = tau_init === nothing ? nothing : Array(tau_init)
    for k in 2:nz+1, j in 2:ny+1, i in 2:nx+1
        ir, jr, kr = i - 1, j - 1, k - 1
        solid = solid_p[i, j, k]
        for q in 1:19
            f[i, j, k, q] = fh !== nothing ? FT(fh[ir, jr, kr, q]) :
                solid ? equilibrium(D3Q19(), one(FT), zero(FT), zero(FT), zero(FT), q) :
                        equilibrium(D3Q19(), one(FT), FT(-Fx / 2), FT(-Fy / 2), FT(-Fz / 2), q)
        end
        if !solid && th !== nothing
            τ = FT(th[ir, jr, kr])
            sp[i, j, k] = inv(τ)
            sm[i, j, k] = inv(FT(Λ) / (τ - FT(0.5)) + FT(0.5))
        end
    end
    axes3 = (1, 2, 3)
    for a in (f, sp, sm)
        _wrap_gnf_periodic_3d!(a, axes3)
    end
    force = ntuple(d -> FT.((d == 1 ? Fx : d == 2 ? Fy : Fz) .* .!solid_p), 3)
    dev(a) = _gnf_backend_copy_3d(backend, a)
    f_in = dev(f)
    return (; f_in, f_out=similar(f_in), rho=dev(ones(FT, dims)), ux=dev(zeros(FT, dims)),
            uy=dev(zeros(FT, dims)), uz=dev(zeros(FT, dims)), is_solid=dev(solid_p),
            q_wall=dev(q_wall), uw=dev(zeros(FT, dims..., 19)), Fx=dev(force[1]),
            Fy=dev(force[2]), Fz=dev(force[3]), sp=dev(sp), sm=dev(sm), dims, nx, ny, nz,
            model, Λ=FT(Λ), F=(Fx, Fy, Fz), axes3)
end

function _gnf3d_periodic_step!(s, f_out, f_in)
    fused_trt_gnf_libb_guo_field_step_3d!(f_out, f_in, s.rho, s.ux, s.uy, s.uz, s.is_solid,
        s.q_wall, s.uw, s.uw, s.uw, s.Fx, s.Fy, s.Fz, s.sp, s.sm,
        s.dims[1], s.dims[2], s.dims[3], s.model; Λ=s.Λ)
    for a in (f_out, s.sp, s.sm)
        _wrap_gnf_periodic_3d!(a, s.axes3)
    end
    return nothing
end

"""
    run_gnf3d_periodic_steady(; is_solid, closure, F=(0, 0, 0), nu_init=0.1, Λ=3/16,
                              tol=1e-10, check_every=100, max_steps=1_000_000,
                              f_init=nothing, tau_init=nothing,
                              backend=CPU(), FT=Float64) -> NamedTuple

Steady 3D GNF flow on a fully periodic lattice of size `size(is_solid)` with
voxel solids (`is_solid`, walls are solid planes) and halfway bounce-back
(q_w = 1/2) on every fluid–solid link, driven by the body force `F`
(`(Fx, Fy, Fz)` or a scalar Fx, lattice units). Each step is the production
`fused_trt_gnf_libb_guo_field_step_3d!` on the lattice padded with one
periodic ghost layer per face (TRT, magic parameter Λ), so this is the forward
map linearised by [`gnf3d_linearise`](@ref). `closure`: any model accepted by
the 3D step, including [`LBMLawViscosity`](@ref). Start: `f_init`
(nx, ny, nz, 19) and `tau_init` (τ⁺ at fluid nodes), else rest equilibrium
shifted by −F/2 at viscosity `nu_init`. Stops when the largest change of the
velocity over `check_every` steps is ≤ `tol` times its largest magnitude
(`tol = 0`: run `max_steps`).

Returns `(; f, tau, rho, ux, uy, uz, steps, converged)` on the reduced
lattice, host arrays: post-collision `f`, τ⁺ = 1/s⁺ (fluid nodes), and the
kernel velocity u = (Σ c fa + F/2)/ρ of the last step, equal to the
post-collision moment (Σ c f − F/2)/ρ of `f` (`gnf3d_post_moments`).
"""
function run_gnf3d_periodic_steady(; is_solid, closure, F=(0.0, 0.0, 0.0), nu_init::Real=0.1,
                                   Λ::Real=3 / 16, tol::Real=1e-10, check_every::Int=100,
                                   max_steps::Int=1_000_000, f_init=nothing, tau_init=nothing,
                                   backend=KernelAbstractions.CPU(), FT::Type=Float64)
    s = _gnf3d_periodic_setup(is_solid, closure, F, nu_init, Λ, f_init, tau_init, backend, FT)
    nx, ny, nz = s.nx, s.ny, s.nz
    I = (2:nx+1, 2:ny+1, 2:nz+1)
    vel = (view(s.ux, I...), view(s.uy, I...), view(s.uz, I...))
    prev = map(v -> (p = similar(v, size(v)); p .= v; p), vel)
    f_in, f_out = s.f_in, s.f_out
    steps = 0; converged = false
    for step in 1:max_steps
        _gnf3d_periodic_step!(s, f_out, f_in)
        f_in, f_out = f_out, f_in
        steps = step
        if step % check_every == 0 || step == max_steps
            change = maximum(maximum(abs.(vel[d] .- prev[d])) for d in 1:3)
            scale = maximum(maximum(abs, vel[d]) for d in 1:3)
            converged = change <= FT(tol) * scale
            (converged && tol > 0) && break
            for d in 1:3
                prev[d] .= vel[d]
            end
        end
    end
    KernelAbstractions.synchronize(backend)
    f = Array(f_in)[I..., :]
    tau = inv.(Array(s.sp)[I...])
    return (; f, tau, rho=Array(s.rho)[I...], ux=Array(s.ux)[I...], uy=Array(s.uy)[I...],
            uz=Array(s.uz)[I...], steps, converged)
end

"""
    gnf3d_step_state(w, is_solid, closure, F; Λ=3/16, nu_solid=0.1, backend=CPU(), FT=Float64) -> w'

One production 3D GNF step (`run_gnf3d_periodic_steady` with one step) applied
to a flat state `w` of the reduced lattice (`gnf3d_pack_state`). Solid
components of w' are the rest equilibrium and τ⁺ = 3 `nu_solid` + 1/2 (inactive).
"""
function gnf3d_step_state(w, is_solid, closure, F; Λ::Real=3 / 16, nu_solid::Real=0.1,
                          backend=KernelAbstractions.CPU(), FT::Type=Float64)
    nx, ny, nz = size(is_solid)
    f, tau = gnf3d_unpack_state(Array(w), nx, ny, nz)
    r = run_gnf3d_periodic_steady(; is_solid, closure, F, nu_init=nu_solid, Λ, tol=0.0,
                                  max_steps=1, f_init=f, tau_init=tau, backend, FT)
    return gnf3d_pack_state(r.f, r.tau)
end

"""
    gnf3d_host_step(w, is_solid, closure, F; Λ=3/16) -> w'

Host reference of the same step written with the per-node functions of the
adjoint (`_gnf3d_src` streaming, [`gnf3d_cell_update`](@ref)); fluid components
only (solid components of w' are copied from w). Used to check that the
linearised model is the production step.
"""
function gnf3d_host_step(w, is_solid, closure, F; Λ::Real=3 / 16)
    wh = Float64.(Array(w)); solid = Array{Bool}(is_solid)
    nx, ny, nz = size(solid)
    Fc = _gnf3d_force3(F)
    cl = _gnf_host_closure(gnf_convert_closure(Float64, closure))
    out = copy(wh)
    for k in 1:nz, j in 1:ny, i in 1:nx
        solid[i, j, k] && continue
        fa = SVector{19,Float64}(ntuple(q -> wh[_gnf3d_src(i, j, k, q, nx, ny, nz, solid)], Val(19)))
        fo, τn, _, _ = gnf3d_cell_update(fa, wh[_gnf3d_lin(i, j, k, 20, nx, ny, nz)], Fc, cl,
                                         Float64(Λ))
        for q in 1:19
            out[_gnf3d_lin(i, j, k, q, nx, ny, nz)] = fo[q]
        end
        out[_gnf3d_lin(i, j, k, 20, nx, ny, nz)] = τn
    end
    return out
end

# ============================================================
# Linearisation (host)
# ============================================================

"""
    gnf3d_active_mask(is_solid) -> Vector{Bool}

Active components of the flat 3D state (length 20 nx ny nz): the 20 components
of every fluid node (solid components are constants of the step and never read
by a fluid node under halfway bounce-back).
"""
function gnf3d_active_mask(is_solid::AbstractArray{Bool,3})
    fl = vec(.!Array{Bool}(is_solid))
    return repeat(fl, 20)
end

"""
    GNF3DLinearisation

Host data of the 3D step linearised at a state w = (f, τ⁺) (`gnf3d_linearise`):
blocks `B` (nx, ny, nz, 20, 21) (zero at solid nodes; column 21 = ∂(f', τ⁺')/∂ν),
the strain invariant `I1` = S:S, viscosity `nu` and closure slope `dnu_dI1` used
by the step from w, the post-collision moments `rho`, `ux`, `uy`, `uz` of w
(velocity 0 at solid nodes), the force `F`, Λ, the active mask and counts.
"""
struct GNF3DLinearisation
    nx::Int
    ny::Int
    nz::Int
    is_solid::Array{Bool,3}
    B::Array{Float64,5}
    I1::Array{Float64,3}
    nu::Array{Float64,3}
    dnu_dI1::Array{Float64,3}
    rho::Array{Float64,3}
    ux::Array{Float64,3}
    uy::Array{Float64,3}
    uz::Array{Float64,3}
    F::NTuple{3,Float64}
    Λ::Float64
    active::Vector{Bool}
    n_fluid::Int
    n_active::Int
    n_small_I1::Int
    I1_min::Float64
    time::Float64
end

"""
    gnf3d_post_moments(f, is_solid, F) -> (rho, ux, uy, uz)

Host density and velocity of post-collision 3D populations,
u = (Σ c f − F/2)/ρ, velocity 0 at solid nodes.
"""
function gnf3d_post_moments(f, is_solid::AbstractArray{Bool,3}, F)
    fh = Float64.(Array(f)); Fx, Fy, Fz = _gnf3d_force3(F)
    nx, ny, nz = size(is_solid)
    rho = zeros(nx, ny, nz); ux = zeros(nx, ny, nz); uy = zeros(nx, ny, nz); uz = zeros(nx, ny, nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        ρ = 0.0; mx = 0.0; my = 0.0; mz = 0.0
        for q in 1:19
            v = fh[i, j, k, q]
            ρ += v; mx += _G3_CX[q] * v; my += _G3_CY[q] * v; mz += _G3_CZ[q] * v
        end
        rho[i, j, k] = ρ
        if !is_solid[i, j, k]
            ux[i, j, k] = (mx - Fx / 2) / ρ
            uy[i, j, k] = (my - Fy / 2) / ρ
            uz[i, j, k] = (mz - Fz / 2) / ρ
        end
    end
    return rho, ux, uy, uz
end

"""
    gnf3d_linearise(f, tau, is_solid, closure, F; Λ=3/16) -> GNF3DLinearisation

Local blocks of the 3D GNF step at the post-collision state (f, τ⁺) of the
reduced periodic lattice (any backend; computed on the host in Float64). For
every fluid node: arrivals fa = S f, (I1_c, ν_c) of the step from w
([`gnf3d_cell_update`](@ref)), closure slope κ_c = dν/dI1 (ForwardDiff in I1),
and the 20 × 21 Jacobian of the linearised cell map in (fa_c, τ⁺_c, δν).
"""
function gnf3d_linearise(f, tau, is_solid::AbstractArray{Bool,3}, closure, F; Λ::Real=3 / 16)
    t0 = time()
    solid = Array{Bool}(is_solid)
    nx, ny, nz = size(solid)
    size(f) == (nx, ny, nz, 19) || throw(DimensionMismatch("f must be $(nx)×$(ny)×$(nz)×19"))
    size(tau) == (nx, ny, nz) || throw(DimensionMismatch("tau must be $(nx)×$(ny)×$(nz)"))
    w = gnf3d_pack_state(Float64.(Array(f)), Float64.(Array(tau)))
    Fc = _gnf3d_force3(F); L = Float64(Λ)
    cl = _gnf_host_closure(gnf_convert_closure(Float64, closure))
    B = zeros(nx, ny, nz, 20, 21)
    I1 = zeros(nx, ny, nz); nu = zeros(nx, ny, nz); dnu = zeros(nx, ny, nz)
    Threads.@threads for k in 1:nz
        for j in 1:ny, i in 1:nx
            solid[i, j, k] && continue
            fa = SVector{19,Float64}(ntuple(q -> w[_gnf3d_src(i, j, k, q, nx, ny, nz, solid)], Val(19)))
            τp = w[_gnf3d_lin(i, j, k, 20, nx, ny, nz)]
            _, _, s, ν = gnf3d_cell_update(fa, τp, Fc, cl, L)
            κ = ForwardDiff.derivative(t -> gnf_viscosity(cl, t), s)
            x0 = SVector{21,Float64}(ntuple(q -> q <= 19 ? fa[q] : q == 20 ? τp : 0.0, Val(21)))
            Jc = ForwardDiff.jacobian(x -> _gnf3d_lin_cell(x, Fc, ν, κ, s, L), x0)
            for col in 1:21, r in 1:20
                B[i, j, k, r, col] = Jc[r, col]
            end
            I1[i, j, k] = s; nu[i, j, k] = ν; dnu[i, j, k] = κ
        end
    end
    rho, ux, uy, uz = gnf3d_post_moments(f, solid, Fc)
    active = gnf3d_active_mask(solid)
    fl = .!solid
    n_small = count(fl .& (I1 .< 1e-12))
    I1min = any(fl) ? minimum(I1[fl]) : NaN
    return GNF3DLinearisation(nx, ny, nz, solid, B, I1, nu, dnu, rho, ux, uy, uz, Fc, L,
                              active, count(fl), count(active), n_small, I1min, time() - t0)
end
