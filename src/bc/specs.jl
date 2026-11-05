using KernelAbstractions

# =====================================================================
# Local TRT collide helpers reused by the face BC kernels below.
# =====================================================================

@inline function _trt_collide_local(f1::T, f2::T, f3::T, f4::T, f5::T,
                                     f6::T, f7::T, f8::T, f9::T,
                                     s_p::T, s_m::T) where {T}
    ρ  = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
    ux = (f2 - f4 + f6 - f8 + f9 - f7) / ρ
    uy = (f3 - f5 + f6 - f8 + f7 - f9) / ρ
    usq = ux * ux + uy * uy
    fe1 = feq_2d(Val(1), ρ, ux, uy, usq)
    fe2 = feq_2d(Val(2), ρ, ux, uy, usq)
    fe3 = feq_2d(Val(3), ρ, ux, uy, usq)
    fe4 = feq_2d(Val(4), ρ, ux, uy, usq)
    fe5 = feq_2d(Val(5), ρ, ux, uy, usq)
    fe6 = feq_2d(Val(6), ρ, ux, uy, usq)
    fe7 = feq_2d(Val(7), ρ, ux, uy, usq)
    fe8 = feq_2d(Val(8), ρ, ux, uy, usq)
    fe9 = feq_2d(Val(9), ρ, ux, uy, usq)
    # Regularized TRT: reconstruct f_neq from stress tensor Π only,
    # f_neq_i = w_i / (2 c_s⁴) (c_iα c_iβ - c_s² δ_αβ) Π_αβ with c_s² = 1/3,
    # so that Σ f_neq = 0, Σ c f_neq = 0 and Σ c c f_neq = Π exactly.
    Pxx = (f2-fe2) + (f4-fe4) + (f6-fe6) + (f7-fe7) + (f8-fe8) + (f9-fe9)
    Pyy = (f3-fe3) + (f5-fe5) + (f6-fe6) + (f7-fe7) + (f8-fe8) + (f9-fe9)
    Pxy = (f6-fe6) - (f7-fe7) + (f8-fe8) - (f9-fe9)
    h = T(0.5)
    fn1 = -T(2/3) * (Pxx + Pyy)
    fn2 =  T(1/6) * (T(2)*Pxx - Pyy)
    fn3 =  T(1/6) * (-Pxx + T(2)*Pyy)
    fn4 =  fn2
    fn5 =  fn3
    fn6 =  T(1/12) * (Pxx + Pyy) + T(1/4) * Pxy
    fn7 =  T(1/12) * (Pxx + Pyy) - T(1/4) * Pxy
    fn8 =  fn6
    fn9 =  fn7
    a = (s_p + s_m) * h
    b = (s_p - s_m) * h
    return (
        fe1 + (one(T) - s_p) * fn1,
        fe2 + (one(T) - a) * fn2 - b * fn4,
        fe3 + (one(T) - a) * fn3 - b * fn5,
        fe4 + (one(T) - a) * fn4 - b * fn2,
        fe5 + (one(T) - a) * fn5 - b * fn3,
        fe6 + (one(T) - a) * fn6 - b * fn8,
        fe7 + (one(T) - a) * fn7 - b * fn9,
        fe8 + (one(T) - a) * fn8 - b * fn6,
        fe9 + (one(T) - a) * fn9 - b * fn7,
    )
end

@inline function _trt_collide_local_3d(f1::T, f2::T, f3::T, f4::T, f5::T,
                                        f6::T, f7::T, f8::T, f9::T,
                                        f10::T, f11::T, f12::T, f13::T, f14::T,
                                        f15::T, f16::T, f17::T, f18::T, f19::T,
                                        s_p::T, s_m::T) where {T}
    ρ  = f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14+f15+f16+f17+f18+f19
    ux = ((f2 - f3) + (f8 - f11) + (f10 - f9) + (f12 - f15) + (f14 - f13)) / ρ
    uy = ((f4 - f5) + (f8 - f11) + (f9 - f10) + (f16 - f19) + (f18 - f17)) / ρ
    uz = ((f6 - f7) + (f12 - f15) + (f13 - f14) + (f16 - f19) + (f17 - f18)) / ρ
    a = (s_p + s_m) * T(0.5)
    b = (s_p - s_m) * T(0.5)
    usq = ux * ux + uy * uy + uz * uz
    fe1  = feq_3d(Val(1),  ρ, ux, uy, uz, usq)
    fe2  = feq_3d(Val(2),  ρ, ux, uy, uz, usq)
    fe3  = feq_3d(Val(3),  ρ, ux, uy, uz, usq)
    fe4  = feq_3d(Val(4),  ρ, ux, uy, uz, usq)
    fe5  = feq_3d(Val(5),  ρ, ux, uy, uz, usq)
    fe6  = feq_3d(Val(6),  ρ, ux, uy, uz, usq)
    fe7  = feq_3d(Val(7),  ρ, ux, uy, uz, usq)
    fe8  = feq_3d(Val(8),  ρ, ux, uy, uz, usq)
    fe9  = feq_3d(Val(9),  ρ, ux, uy, uz, usq)
    fe10 = feq_3d(Val(10), ρ, ux, uy, uz, usq)
    fe11 = feq_3d(Val(11), ρ, ux, uy, uz, usq)
    fe12 = feq_3d(Val(12), ρ, ux, uy, uz, usq)
    fe13 = feq_3d(Val(13), ρ, ux, uy, uz, usq)
    fe14 = feq_3d(Val(14), ρ, ux, uy, uz, usq)
    fe15 = feq_3d(Val(15), ρ, ux, uy, uz, usq)
    fe16 = feq_3d(Val(16), ρ, ux, uy, uz, usq)
    fe17 = feq_3d(Val(17), ρ, ux, uy, uz, usq)
    fe18 = feq_3d(Val(18), ρ, ux, uy, uz, usq)
    fe19 = feq_3d(Val(19), ρ, ux, uy, uz, usq)
    F1  = f1  - s_p * (f1  - fe1)
    F2  = f2  - a*(f2  - fe2)  - b*(f3  - fe3)
    F3  = f3  - a*(f3  - fe3)  - b*(f2  - fe2)
    F4  = f4  - a*(f4  - fe4)  - b*(f5  - fe5)
    F5  = f5  - a*(f5  - fe5)  - b*(f4  - fe4)
    F6  = f6  - a*(f6  - fe6)  - b*(f7  - fe7)
    F7  = f7  - a*(f7  - fe7)  - b*(f6  - fe6)
    F8  = f8  - a*(f8  - fe8)  - b*(f11 - fe11)
    F11 = f11 - a*(f11 - fe11) - b*(f8  - fe8)
    F9  = f9  - a*(f9  - fe9)  - b*(f10 - fe10)
    F10 = f10 - a*(f10 - fe10) - b*(f9  - fe9)
    F12 = f12 - a*(f12 - fe12) - b*(f15 - fe15)
    F15 = f15 - a*(f15 - fe15) - b*(f12 - fe12)
    F13 = f13 - a*(f13 - fe13) - b*(f14 - fe14)
    F14 = f14 - a*(f14 - fe14) - b*(f13 - fe13)
    F16 = f16 - a*(f16 - fe16) - b*(f19 - fe19)
    F19 = f19 - a*(f19 - fe19) - b*(f16 - fe16)
    F17 = f17 - a*(f17 - fe17) - b*(f18 - fe18)
    F18 = f18 - a*(f18 - fe18) - b*(f17 - fe17)
    return F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17,F18,F19
end

# =====================================================================
# Modular boundary conditions for 2D/3D LI-BB V2 drivers.
#
# The old `rebuild_inlet_outlet_libb_{2d,3d}!` hard-coded one BC combo
# (Zou-He velocity west + Zou-He pressure east). Drivers now accept a
# `BCSpec{2,3}D` that names the BC per face; `apply_bc_rebuild_{2d,3d}!`
# dispatches on the per-face type and the backend, launching the right
# kernel for each active face. Face kernels are normal `@kernel`
# functions specialised per BC type — dispatch happens at Julia's method
# level (compiled on first call by the JIT), no `eval` needed.
#
# Design notes:
#
# 1. Each BC reconstructs `f_out[face, :, ...]` from the PRE-step `f_in`
#    values streamed from the interior, applies the Zou-He closure, and
#    does a local TRT collide. This bypasses the fused kernel's
#    halfway-BB fallback corruption at non-wall boundaries — same logic
#    as the old hardcoded rebuild functions.
#
# 2. `HalfwayBB` is a no-op: the fused kernel's PullHalfwayBB brick
#    already handles it at domain edges. Used for stationary channel
#    walls (j=1, j=Ny in 2D; (j,k) ∈ {1,Ny}×{1,Nz} faces in 3D).
#
# 3. Corners / face edges (where two BC faces meet) are left to the
#    kernel's halfway-BB fallback. For the typical Schäfer-Turek setup
#    (parabolic inlet with u=0 at walls + channel walls halfway-BB),
#    the BC values at corners are consistent (u=0).
#
# 4. `ZouHeVelocity` takes a profile (function or precomputed device
#    array). A pure uniform BC is just `ZouHeVelocity(u_uniform)` which
#    builds a constant profile on the backend.
# =====================================================================

"""
    AbstractBC

Public type or module in the kernel-level LBM operation.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using Kraken

Kraken.AbstractBC
```
"""
abstract type AbstractBC end

"Kernel-fallback halfway bounce-back. No-op (the fused LBM kernel
already applies halfway-BB at domain edges via PullHalfwayBB)."
struct HalfwayBB <: AbstractBC end

"Multi-block interface edge — no BC applied (ghost exchange handles it)."
struct InterfaceBC <: AbstractBC end

"""
    ZouHeVelocity(profile)

Zou-He velocity boundary condition. `profile` is a device array of
length N (N = Ny for x-faces, Nx for y-faces, Nx·Nz or similar for
z-faces in 3D) giving the normal-into-domain velocity at each cell on
the face. Tangential velocity is assumed zero.

For a 2D x-face with a parabolic channel profile, pass the discrete
`u(y)` array. For uniform inflow, `fill(T(u_in), Ny)`.
"""
struct ZouHeVelocity{A<:AbstractArray} <: AbstractBC
    profile::A
    physical_dir::Symbol
end
ZouHeVelocity(profile::AbstractArray) = ZouHeVelocity(profile, :auto)

"""
    ZouHePressure(ρ_out)

Zou-He pressure boundary condition. `ρ_out` is a scalar target density;
normal velocity is computed from the known streamed-in populations.
Tangential velocity is zero.

On the east face of the 2D rebuild path (`apply_bc_rebuild_2d!`), the
closed node is collided with the regularized TRT of `_trt_collide_local`.
There the outlet carries a sign-flipping step-to-step mode that does not
decay, and grows when the outlet corner rows keep the half-way bounce-back
of the bulk kernel; the steady residual then stalls instead of reaching
round-off. For a steady outlet on that path use `NeqExtrapolationPressure`.
"""
struct ZouHePressure{T<:Real} <: AbstractBC
    ρ_out::T
    physical_dir::Symbol
end
ZouHePressure(ρ_out::Real) = ZouHePressure(ρ_out, :auto)

"""
    NeqExtrapolationPressure(ρ_out)

Pressure outlet for the east face of the 2D rebuild path
(`apply_bc_rebuild_2d!`), by non-equilibrium extrapolation. After the bulk
step has written the post-collision populations, each outlet node
`(Nx, j)` is overwritten with

    f_q(Nx, j) = f_q^eq(ρ_out, u) + [f_q(Nx-1, j) - f_q^eq(ρ, u)],

where `ρ` and `u` are the raw moments `Σ f` and `Σ c f / Σ f` of the
post-collision populations of the neighbour `(Nx-1, j)`. No collision is
applied at `Nx`: the copied non-equilibrium part is already post-collision,
so the rule does not depend on the relaxation rates.

Density is imposed and velocity has zero normal gradient in the raw-moment
sense. The log-FV drivers use TRT Guo forcing and read velocity as `Σ c f / ρ`,
without a `+F/(2ρ)` correction; this outlet copies the neighbour's post-collision
raw velocity and applies no additional force at `Nx`. Only rows `2:Ny-1` are
rebuilt; the two corners retain the bulk-step state (half-way bounce-back at
wall corners). The measured maximum Poiseuille velocity error is first order
at the outlet corners. Place the outlet several obstacle diameters downstream.
Backflow (`u_x < 0`) is not supported.

Only the east face of `BCSpec2D` accepts it; requires `Nx ≥ 3`.
"""
struct NeqExtrapolationPressure{T<:Real} <: AbstractBC
    ρ_out::T
end

"""
    BCSpec2D(; west, east, south, north)

Per-face BC specification for a 2D rectangular domain. Defaults are
`HalfwayBB` (kernel-fallback) so only set the faces that differ.
"""
struct BCSpec2D{W<:AbstractBC, E<:AbstractBC, S<:AbstractBC, N<:AbstractBC}
    west::W
    east::E
    south::S
    north::N
end
BCSpec2D(; west::AbstractBC=HalfwayBB(), east::AbstractBC=HalfwayBB(),
           south::AbstractBC=HalfwayBB(), north::AbstractBC=HalfwayBB()) =
    BCSpec2D(west, east, south, north)

"""
    BCSpec3D(; west, east, south, north, bottom, top)

Per-face BC specification for a 3D rectangular box domain.
"""
struct BCSpec3D{W<:AbstractBC, E<:AbstractBC, S<:AbstractBC,
                 N<:AbstractBC, B<:AbstractBC, T<:AbstractBC}
    west::W; east::E; south::S; north::N; bottom::B; top::T
end
BCSpec3D(; west::AbstractBC=HalfwayBB(), east::AbstractBC=HalfwayBB(),
           south::AbstractBC=HalfwayBB(), north::AbstractBC=HalfwayBB(),
           bottom::AbstractBC=HalfwayBB(), top::AbstractBC=HalfwayBB()) =
    BCSpec3D(west, east, south, north, bottom, top)

# ----------------------------------------------------------------------
# 2D face kernels — Zou-He velocity, Zou-He pressure
# ----------------------------------------------------------------------

