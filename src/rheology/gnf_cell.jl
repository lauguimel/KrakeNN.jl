# --- Per-cell pure functions for the GNF BGK + Guo collision (D2Q9) ---
#
# The production kernel `collide_rheology_guo_2d_kernel!` does, at a fluid
# node: moments with the Guo half-force, equilibrium, strain rate from the
# non-equilibrium stress with the relaxation time of the PREVIOUS step,
# ν = ν(γ̇), BGK relaxation plus the Guo source, and it stores τ = 1/ω.
#
# The functions below split that sequence into pure functions of the nine
# arrived (post-streaming) populations, held in an `SVector{9}`, and scalars.
# They are used by the KernelAbstractions kernels of
# `kernels/collide_gnf_cell_2d.jl` on CPU and GPU alike, and they are generic
# in the number type so that ForwardDiff can differentiate one cell (the 10×10
# local blocks of the steady adjoint). Arithmetic and operation
# order are copied from the production kernel, so the split path reproduces it
# bit for bit (test T1 in `test/rheology/gnf_cell_test.jl`).
#
# The closure sees the strain-rate invariant I1 = S:S, not γ̇ = √(2 I1): the
# square root has an infinite derivative at S = 0 and would put NaN in the
# ForwardDiff blocks there; √(2 I1) stays bit-identical to the production γ̇.
#
# State convention: the augmented state of one step is (f, τ), with τ lagged
# by one step: I1 at step k uses τ from step k-1. At a fixed point τ_prev = τ_new.

"""
    gnf_moments_cell(fp, Fx, Fy) -> (ρ, ux, uy)

Density and Guo-corrected velocity of one cell, `u = (Σ c f + F/2)/ρ`, from the
nine arrived populations `fp` (D2Q9 order of `feq_2d`) and the body force.
"""
@inline function gnf_moments_cell(fp::SVector{9}, Fx, Fy)
    T = promote_type(eltype(fp), typeof(Fx), typeof(Fy))
    f1 = T(fp[1]); f2 = T(fp[2]); f3 = T(fp[3]); f4 = T(fp[4]); f5 = T(fp[5])
    f6 = T(fp[6]); f7 = T(fp[7]); f8 = T(fp[8]); f9 = T(fp[9])
    fx = T(Fx); fy = T(Fy)
    ρ = f1+f2+f3+f4+f5+f6+f7+f8+f9
    inv_ρ = one(T) / ρ
    ux = ((f2-f4+f6-f7-f8+f9) + fx/T(2)) * inv_ρ
    uy = ((f3-f5+f6+f7-f8-f9) + fy/T(2)) * inv_ρ
    return ρ, ux, uy
end

"""
    gnf_feq_cell(ρ, ux, uy) -> SVector{9}

D2Q9 second-order equilibrium of one cell (same expressions as `feq_2d`).
"""
@inline function gnf_feq_cell(ρ, ux, uy)
    T = promote_type(typeof(ρ), typeof(ux), typeof(uy))
    r = T(ρ); vx = T(ux); vy = T(uy)
    usq = vx*vx + vy*vy
    return SVector{9,T}(feq_2d(Val(1), r, vx, vy, usq), feq_2d(Val(2), r, vx, vy, usq),
                        feq_2d(Val(3), r, vx, vy, usq), feq_2d(Val(4), r, vx, vy, usq),
                        feq_2d(Val(5), r, vx, vy, usq), feq_2d(Val(6), r, vx, vy, usq),
                        feq_2d(Val(7), r, vx, vy, usq), feq_2d(Val(8), r, vx, vy, usq),
                        feq_2d(Val(9), r, vx, vy, usq))
end

"""
    gnf_strain_invariant_cell(fp, ρ, ux, uy, τ_prev) -> I1 = S:S

Strain-rate invariant of one cell from the non-equilibrium stress of the
arrived populations, `S = -Π^neq/(2ρ c_s² τ_prev)`, `I1 = S:S`
(`strain_rate_invariant_2d`). No forcing correction, as in the production
kernel: the Guo term `½(F u + u F)` is not added to Π^neq; its size is
measured by test T2c. `(ρ, ux, uy)` must be the Guo-corrected moments of `fp`.

The closures take `I1`, not `γ̇ = √(2 I1)`: `I1` is smooth in `(fp, τ_prev)`
everywhere, including at `S = 0` where `√` has an infinite derivative.
"""
@inline function gnf_strain_invariant_cell(fp::SVector{9}, ρ, ux, uy, τ_prev)
    T = promote_type(eltype(fp), typeof(ρ), typeof(ux), typeof(uy), typeof(τ_prev))
    r = T(ρ)
    fe = gnf_feq_cell(r, T(ux), T(uy))
    return strain_rate_invariant_2d(
        T(fp[1]), T(fp[2]), T(fp[3]), T(fp[4]), T(fp[5]),
        T(fp[6]), T(fp[7]), T(fp[8]), T(fp[9]),
        fe[1], fe[2], fe[3], fe[4], fe[5], fe[6], fe[7], fe[8], fe[9],
        r, T(τ_prev))
end

"""
    gnf_shear_rate(I1) -> γ̇ = √(2 I1)

Shear-rate magnitude from the invariant. `sqrt(2 * I1)` performs the last two
operations of `strain_rate_magnitude_2d`, so the result is bit-identical to it.
"""
@inline gnf_shear_rate(I1) = sqrt(2 * I1)

"""
    gnf_strain_rate_cell(fp, ρ, ux, uy, τ_prev) -> γ̇

Shear-rate magnitude of one cell, `γ̇ = √(2 S:S)`, equal bit for bit to the
production `strain_rate_magnitude_2d`. Diagnostic only: the step and its
derivatives go through `gnf_strain_invariant_cell`.
"""
@inline gnf_strain_rate_cell(fp::SVector{9}, ρ, ux, uy, τ_prev) =
    gnf_shear_rate(gnf_strain_invariant_cell(fp, ρ, ux, uy, τ_prev))

"""
    gnf_collide_cell(fp, Fx, Fy, ν) -> (fo::SVector{9}, τ_new)

BGK relaxation with kinematic viscosity `ν` plus the Guo source, for one fluid
cell, from its arrived populations `fp`. Returns the post-collision populations
and the relaxation time stored for the next step, `τ_new = 1/ω`,
`ω = 1/(3ν + 1/2)` (same rounding as the production kernel).
"""
@inline function gnf_collide_cell(fp::SVector{9}, Fx, Fy, ν)
    T = promote_type(eltype(fp), typeof(Fx), typeof(Fy), typeof(ν))
    f1 = T(fp[1]); f2 = T(fp[2]); f3 = T(fp[3]); f4 = T(fp[4]); f5 = T(fp[5])
    f6 = T(fp[6]); f7 = T(fp[7]); f8 = T(fp[8]); f9 = T(fp[9])
    fx = T(Fx); fy = T(Fy)

    ρ, ux, uy = gnf_moments_cell(SVector{9,T}(f1, f2, f3, f4, f5, f6, f7, f8, f9), fx, fy)
    fe = gnf_feq_cell(ρ, ux, uy)

    ω = one(T) / (T(3) * T(ν) + T(0.5))
    τ_new = one(T) / ω
    guo_pref = one(T) - ω / T(2)

    Sq1 = T(4.0/9.0)*((-ux)*fx+(-uy)*fy)*T(3)
    Sq2 = T(1.0/9.0)*((one(T)-ux)*fx+(-uy)*fy)*T(3)+T(1.0/9.0)*ux*fx*T(9)
    Sq3 = T(1.0/9.0)*((-ux)*fx+(one(T)-uy)*fy)*T(3)+T(1.0/9.0)*uy*fy*T(9)
    Sq4 = T(1.0/9.0)*((-one(T)-ux)*fx+(-uy)*fy)*T(3)+T(1.0/9.0)*ux*fx*T(9)
    Sq5 = T(1.0/9.0)*((-ux)*fx+(-one(T)-uy)*fy)*T(3)+T(1.0/9.0)*uy*fy*T(9)
    Sq6 = T(1.0/36.0)*((one(T)-ux)*fx+(one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(ux+uy)*(fx+fy)*T(9)
    Sq7 = T(1.0/36.0)*((-one(T)-ux)*fx+(one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(-ux+uy)*(-fx+fy)*T(9)
    Sq8 = T(1.0/36.0)*((-one(T)-ux)*fx+(-one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(-ux-uy)*(-fx-fy)*T(9)
    Sq9 = T(1.0/36.0)*((one(T)-ux)*fx+(-one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(ux-uy)*(fx-fy)*T(9)

    fo = SVector{9,T}(f1-ω*(f1-fe[1])+guo_pref*Sq1,
                      f2-ω*(f2-fe[2])+guo_pref*Sq2,
                      f3-ω*(f3-fe[3])+guo_pref*Sq3,
                      f4-ω*(f4-fe[4])+guo_pref*Sq4,
                      f5-ω*(f5-fe[5])+guo_pref*Sq5,
                      f6-ω*(f6-fe[6])+guo_pref*Sq6,
                      f7-ω*(f7-fe[7])+guo_pref*Sq7,
                      f8-ω*(f8-fe[8])+guo_pref*Sq8,
                      f9-ω*(f9-fe[9])+guo_pref*Sq9)
    return fo, τ_new
end

"""
    gnf_viscosity(closure, I1) -> ν

Kinematic viscosity (lattice units) given by a closure at the strain-rate
invariant `I1 = S:S` (shear rate `γ̇ = √(2 I1)`). Other closures, including
the local linearisation used by the steady adjoint, add methods.

For a `GeneralizedNewtonian` model this is `effective_viscosity(closure, √(2 I1))`,
whose derivative in `I1` is NaN at `I1 = 0` under ForwardDiff (`√` there).
`CarreauYasuda` has a method that is regular at `I1 = 0` (below).
"""
@inline gnf_viscosity(closure::GeneralizedNewtonian, I1) =
    effective_viscosity(closure, gnf_shear_rate(I1))

# Carreau-Yasuda in I1: (λγ̇)^a = (2λ²I1)^(a/2). For I1 > 0 the production
# expression is used on γ̇ = √(2 I1), so the value is bit-identical to
# `effective_viscosity` (test T1). At I1 = 0 both forms give the same value
# (factor exactly 1), and the I1 form has a finite derivative when a ≥ 2: dν/dI1 = (η0 − η∞)(n − 1)λ² for
# a = 2 (Carreau, the obstacle-channel case). For a < 2 the model itself has an infinite
# derivative in I1 at 0; no rewriting removes it.
@inline function gnf_viscosity(m::CarreauYasuda, I1)
    I1 > zero(I1) && return effective_viscosity(m, gnf_shear_rate(I1))
    T = typeof(m.eta_0)
    lga = (2 * m.lambda * m.lambda * I1)^(m.a / 2)
    factor = (one(T) + lga)^((m.n - one(T)) / m.a)
    return m.eta_inf + (m.eta_0 - m.eta_inf) * factor
end

"""
    gnf_convert_closure(T, closure) -> closure in number type T

Closure with its parameters converted to the number type `T` of the solver
arrays. Without it, a `Float64` closure in a `Float32` run makes the viscosity
`Float64`: the split path rounds it to `Float32` when it stores ν, the fused path
does not, and the two paths drift apart. Closures without a method are returned
unchanged and then set the precision of ν themselves.
"""
gnf_convert_closure(::Type, closure) = closure
gnf_convert_closure(::Type{T}, m::CarreauYasuda) where {T<:AbstractFloat} =
    CarreauYasuda(T(m.eta_0), T(m.eta_inf), T(m.lambda), T(m.a), T(m.n), m.thermal)

"""
    gnf_cell_update(fp, τ_prev, Fx, Fy, closure) -> (fo, τ_new, I1, ν)

One full fluid-cell update, the composition used by the fused kernel and the
function a ForwardDiff local Jacobian differentiates with respect to
`(fp, τ_prev)`: moments → I1(fp, τ_prev) → ν = closure(I1) → BGK + Guo.
Every output, `I1` included, has finite derivatives at `S = 0` for a
Carreau-Yasuda closure with `a ≥ 2`.
"""
@inline function gnf_cell_update(fp::SVector{9}, τ_prev, Fx, Fy, closure)
    ρ, ux, uy = gnf_moments_cell(fp, Fx, Fy)
    I1 = gnf_strain_invariant_cell(fp, ρ, ux, uy, τ_prev)
    ν = gnf_viscosity(closure, I1)
    fo, τ_new = gnf_collide_cell(fp, Fx, Fy, ν)
    return fo, τ_new, I1, ν
end
