# =====================================================================
# Modular viscoelastic specs — decouple polymer model and wall BC
# from the driver (same spirit as BCSpec2D for hydrodynamics).
#
# Dispatch compiles the right update/BC per spec type at first call.
# Adding a new polymer model or polymer wall BC means:
#   1. struct MyModel <: AbstractPolymerModel + fields
#   2. update_polymer_stress!(τ_p_*, C_*, model::MyModel)
#   3. (optional) custom C source override via collide_conformation_2d!
# No driver changes required.
# =====================================================================

"""
    AbstractPolymerModel

Constitutive closure mapping conformation tensor C to polymer stress τ_p.
Concrete subtypes: `OldroydBSpec`, `FENEP` (planned), `Giesekus` (planned).
"""
abstract type AbstractPolymerModel end

"""
    OldroydBSpec(; G, λ)

Linear Oldroyd-B: τ_p = G · (C − I) with G = ν_p / λ.
- `G`: polymer shear modulus
- `λ`: polymer relaxation time
"""
struct OldroydBSpec{T<:AbstractFloat} <: AbstractPolymerModel
    G::T
    λ::T
end
OldroydBSpec(; G, λ) = OldroydBSpec(promote(float(G), float(λ))...)

polymer_modulus(m::OldroydBSpec) = m.G
polymer_relaxation_time(m::OldroydBSpec) = m.λ

"""
    LogConfOldroydB(; G, λ)

Oldroyd-B evolved in the log-conformation variable Ψ = log(C) (Fattal
& Kupferman 2004). Preserves positive-definiteness of C exactly at the
discrete level — required for stability at high Weissenberg numbers
where direct-C schemes blow up in extensional regions.

Stress reconstruction: C = exp(Ψ), τ_p = G · (C − I).

Use `collide_logconf_2d!` in place of `collide_conformation_2d!` in
the driver time loop when this model is active.
"""
struct LogConfOldroydB{T<:AbstractFloat} <: AbstractPolymerModel
    G::T
    λ::T
end
LogConfOldroydB(; G, λ) = LogConfOldroydB(promote(float(G), float(λ))...)

polymer_modulus(m::LogConfOldroydB) = m.G
polymer_relaxation_time(m::LogConfOldroydB) = m.λ

"""
    LogConfFENEP(; G, λ, L²)
    LogConfFENEP(; G, λ, Lmax2)

FENE-P (finitely-extensible nonlinear elastic, Peterlin closure) evolved
in the log-conformation variable Ψ = log(C). Identical log-conf machinery
to `LogConfOldroydB` (Fattal & Kupferman 2004) but the upper-convected
relaxation is multiplied by the Peterlin factor

    f = (L² − 3) / (L² − tr C),

so that `dC/dt = ∇uᵀ·C + C·∇u − (1/λ)(f·C − I)`. The factor caps the
trace below `L²` (finite extensibility) and reduces the shear/extensional
response relative to Oldroyd-B. At equilibrium (C=I, tr C=3) → f=1, and as
`L² → ∞` → f→1, so FENE-P reduces to Oldroyd-B exactly.

- `G`:  polymer shear modulus (G = ν_p / λ)
- `λ`:  polymer relaxation time
- `L²`: maximum extensibility (`Lmax²`), must satisfy `L² > 3`.

Stress reconstruction (Peterlin-consistent): τ_p = G · (f·C − I).
"""
struct LogConfFENEP{T<:AbstractFloat} <: AbstractPolymerModel
    G::T
    λ::T
    L²::T
end
function LogConfFENEP(; G, λ, L²=nothing, Lmax2=nothing)
    L2 = L² === nothing ? Lmax2 : L²
    L2 === nothing && throw(ArgumentError("LogConfFENEP requires L² (alias Lmax2)"))
    float(L2) > 3 || throw(ArgumentError("LogConfFENEP requires L² > 3"))
    return LogConfFENEP(promote(float(G), float(λ), float(L2))...)
end

polymer_modulus(m::LogConfFENEP) = m.G
polymer_relaxation_time(m::LogConfFENEP) = m.λ

"""
    LogConfGiesekus(; G, λ, α)

Giesekus model evolved in the log-conformation variable Ψ = log(C)
(Fattal & Kupferman 2004). Identical log-conf machinery to
`LogConfOldroydB`, but the upper-convected relaxation gains an
anisotropic (quadratic) mobility term controlled by `α ∈ [0, 0.5]`:

    dC/dt = ∇uᵀ·C + C·∇u − (1/λ)[ (C − I) + α·(C − I)² ].

In the eigenframe of C (eigenvalues c_i) the relaxation is diagonal,
with per-eigenvalue restoring term

    (c_i − 1) + α·(c_i − 1)² = (c_i − 1)·(1 + α·(c_i − 1)),

so the mobility quadratically enhances relaxation of stretched modes
(`c_i > 1`), producing shear-thinning and a bounded steady response. At
`α = 0` the term collapses to `(c_i − 1)`, recovering Oldroyd-B exactly
(bit-identically).

- `G`: polymer shear modulus (G = ν_p / λ)
- `λ`: polymer relaxation time
- `α`: mobility / anisotropy parameter, `0 ≤ α ≤ 0.5`.

Stress reconstruction (same as Oldroyd-B): τ_p = G · (C − I).
"""
struct LogConfGiesekus{T<:AbstractFloat} <: AbstractPolymerModel
    G::T
    λ::T
    α::T
end
function LogConfGiesekus(; G, λ, α)
    αf = float(α)
    (αf >= 0 && αf <= 0.5) ||
        throw(ArgumentError("LogConfGiesekus requires 0 ≤ α ≤ 0.5"))
    return LogConfGiesekus(promote(float(G), float(λ), αf)...)
end

polymer_modulus(m::LogConfGiesekus) = m.G
polymer_relaxation_time(m::LogConfGiesekus) = m.λ

"""
    LogConfPTT(; G, λ, ε, variant=:linear)

Phan-Thien–Tanner (PTT) model evolved in the log-conformation variable
Ψ = log(C) (Fattal & Kupferman 2004). Identical log-conf machinery to
`LogConfOldroydB`, but the upper-convected relaxation is multiplied by a
SCALAR function `Y(trC)` of the trace `trC = tr C = Σ exp(ψ_i)` (the
non-affine slip parameter ξ is set to 0, i.e. upper-convected PTT):

    dC/dt = ∇uᵀ·C + C·∇u − (1/λ)·Y(trC)·(C − I),

with, for `variant=:linear` (linear PTT, Phan-Thien & Tanner 1977),

    Y(trC) = 1 + ε·(trC − 3),

or, for `variant=:exponential` (exponential PTT, Phan-Thien 1978),

    Y(trC) = exp( ε·(trC − 3) ).

Unlike Giesekus, the PTT multiplier is the SAME scalar `Y(trC)` for every
eigenvalue (it depends only on the trace), so in the eigenframe of C the
per-eigenvalue restoring term is `Y(trC)·(c_i − 1)`, i.e. in the log
variable `−Y(trC)·(1 − 1/c_i)·inv_λ`. At equilibrium (C=I, trC=3) → Y=1,
and at `ε = 0` → Y=1 exactly, so the term collapses to the Oldroyd-B
diagonal restoring term BIT-IDENTICALLY and Oldroyd-B is recovered. A
positive `ε` enhances relaxation of stretched modes (large trC),
producing shear-thinning and a bounded steady response.

- `G`: polymer shear modulus (G = ν_p / λ)
- `λ`: polymer relaxation time
- `ε`: extensibility / elongational parameter, `ε ≥ 0`.
- `variant`: `:linear` (default) or `:exponential`.

Stress reconstruction (same as Oldroyd-B): τ_p = G · (C − I).
"""
struct LogConfPTT{T<:AbstractFloat} <: AbstractPolymerModel
    G::T
    λ::T
    ε::T
    variant::Symbol
end
function LogConfPTT(; G, λ, ε, variant::Symbol=:linear)
    εf = float(ε)
    εf >= 0 || throw(ArgumentError("LogConfPTT requires ε ≥ 0"))
    (variant === :linear || variant === :exponential) ||
        throw(ArgumentError("LogConfPTT variant must be :linear or :exponential"))
    G2, λ2, ε2 = promote(float(G), float(λ), εf)
    return LogConfPTT(G2, λ2, ε2, variant)
end

polymer_modulus(m::LogConfPTT) = m.G
polymer_relaxation_time(m::LogConfPTT) = m.λ

"""
    polymer_max_extensibility(model) -> L²

Maximum extensibility `L²` (`Lmax²`) for finitely-extensible models.
Returns `Inf` for unbounded models (Oldroyd-B), so callers can branch on
`isfinite(...)` to recover the OB path.
"""
polymer_max_extensibility(::AbstractPolymerModel) = Inf
polymer_max_extensibility(m::LogConfFENEP) = m.L²

"""
    polymer_mobility(model) -> α

Giesekus mobility / anisotropy parameter `α`. Returns `0` for models
without a quadratic mobility term (Oldroyd-B, FENE-P), so callers can
branch on `iszero(...)` to recover the Oldroyd-B path bit-identically.
"""
polymer_mobility(::AbstractPolymerModel) = 0.0
polymer_mobility(m::LogConfGiesekus) = m.α

"""
    polymer_ptt_epsilon(model) -> ε

Phan-Thien–Tanner extensibility parameter `ε`. Returns `0` for models
without a PTT trace multiplier (Oldroyd-B, FENE-P, Giesekus), so callers
can branch on `iszero(...)` to recover the Oldroyd-B path bit-identically.
"""
polymer_ptt_epsilon(::AbstractPolymerModel) = 0.0
polymer_ptt_epsilon(m::LogConfPTT) = m.ε

"""
    polymer_ptt_variant(model) -> Symbol

PTT trace-multiplier variant: `:linear` (`Y = 1 + ε(trC−3)`) or
`:exponential` (`Y = exp(ε(trC−3))`). Defaults to `:linear` for every
non-PTT model (where it is inert because `ε = 0`).
"""
polymer_ptt_variant(::AbstractPolymerModel) = :linear
polymer_ptt_variant(m::LogConfPTT) = m.variant

"""
    uses_log_conformation(model) -> Bool

Dispatch hook: `true` if the driver must evolve Ψ = log(C) instead of
C, and reconstruct C via `psi_to_C_2d!` before computing τ_p.
"""
uses_log_conformation(::AbstractPolymerModel) = false
uses_log_conformation(::LogConfOldroydB) = true
uses_log_conformation(::LogConfFENEP) = true
uses_log_conformation(::LogConfGiesekus) = true
uses_log_conformation(::LogConfPTT) = true

"""
    update_polymer_stress!(τ_p_xx, τ_p_xy, τ_p_yy, C_xx, C_xy, C_yy, model)

In-place update τ_p ← f(C) from the constitutive closure.
"""
function update_polymer_stress!(τ_p_xx, τ_p_xy, τ_p_yy,
                                  C_xx, C_xy, C_yy,
                                  m::OldroydBSpec)
    G = eltype(τ_p_xx)(m.G)
    if iszero(G)
        fill!(τ_p_xx, zero(eltype(τ_p_xx)))
        fill!(τ_p_xy, zero(eltype(τ_p_xy)))
        fill!(τ_p_yy, zero(eltype(τ_p_yy)))
        return nothing
    end
    @. τ_p_xx = G * (C_xx - 1)
    @. τ_p_xy = G * C_xy
    @. τ_p_yy = G * (C_yy - 1)
    return nothing
end

# For LogConfOldroydB, callers pass C arrays that have already been
# reconstructed from Ψ via `psi_to_C_2d!`. The stress formula is then
# identical to the direct Oldroyd-B case.
function update_polymer_stress!(τ_p_xx, τ_p_xy, τ_p_yy,
                                  C_xx, C_xy, C_yy,
                                  m::LogConfOldroydB)
    G = eltype(τ_p_xx)(m.G)
    if iszero(G)
        fill!(τ_p_xx, zero(eltype(τ_p_xx)))
        fill!(τ_p_xy, zero(eltype(τ_p_xy)))
        fill!(τ_p_yy, zero(eltype(τ_p_yy)))
        return nothing
    end
    @. τ_p_xx = G * (C_xx - 1)
    @. τ_p_xy = G * C_xy
    @. τ_p_yy = G * (C_yy - 1)
    return nothing
end

# Giesekus uses the same linear stress reconstruction τ_p = G·(C − I) as
# Oldroyd-B; the quadratic mobility lives only in the conformation
# evolution, not in the stress closure.
function update_polymer_stress!(τ_p_xx, τ_p_xy, τ_p_yy,
                                  C_xx, C_xy, C_yy,
                                  m::LogConfGiesekus)
    G = eltype(τ_p_xx)(m.G)
    if iszero(G)
        fill!(τ_p_xx, zero(eltype(τ_p_xx)))
        fill!(τ_p_xy, zero(eltype(τ_p_xy)))
        fill!(τ_p_yy, zero(eltype(τ_p_yy)))
        return nothing
    end
    @. τ_p_xx = G * (C_xx - 1)
    @. τ_p_xy = G * C_xy
    @. τ_p_yy = G * (C_yy - 1)
    return nothing
end

# PTT uses the same linear stress reconstruction τ_p = G·(C − I) as
# Oldroyd-B; the trace multiplier Y(trC) lives only in the conformation
# evolution, not in the stress closure.
function update_polymer_stress!(τ_p_xx, τ_p_xy, τ_p_yy,
                                  C_xx, C_xy, C_yy,
                                  m::LogConfPTT)
    G = eltype(τ_p_xx)(m.G)
    if iszero(G)
        fill!(τ_p_xx, zero(eltype(τ_p_xx)))
        fill!(τ_p_xy, zero(eltype(τ_p_xy)))
        fill!(τ_p_yy, zero(eltype(τ_p_yy)))
        return nothing
    end
    @. τ_p_xx = G * (C_xx - 1)
    @. τ_p_xy = G * C_xy
    @. τ_p_yy = G * (C_yy - 1)
    return nothing
end

# ---------------------------------------------------------------------
# Polymer wall BC
# ---------------------------------------------------------------------

"""
    AbstractPolymerWallBC

Boundary treatment for the conformation LBM populations g_* at fluid
cells adjacent to solid walls. Concrete types: `CNEBB`, `CNEBBQAware`,
`CNEBBField`, `CNEBBFieldEquilibrium`, `CNEBBEqGradient`,
`CNEBBCutLinkEqGradient`, `ExtrapEqWallBC`, `LogFieldWallBC`,
`YLW_A`, `YLW_B`, `YLWBalanceOnly`,
`NoPolymerWallBC`.
"""
abstract type AbstractPolymerWallBC end

"""
    CNEBB()

Conservative non-equilibrium bounce-back (Yu et al. 2025, Eqs. 38-40).
This is the strict halfway/staircase formulation: `q_wall` is ignored even
when the driver provides cut-link fractions.
"""
struct CNEBB <: AbstractPolymerWallBC end

"""
    CNEBBQAware()

Diagnostic hybrid boundary condition: Yu-style NEBB macro recovery with the
experimental `q_wall` reconstruction branch. This is not the Yu et al. 2025
CNEBB formula; keep it explicit for audit comparisons only.
"""
struct CNEBBQAware <: AbstractPolymerWallBC end

"""
    CNEBBField()

Diagnostic CNEBB variant that pins the recovered wall macro field to the
current node-centered `C_field`, then rebalances the rest population. This is
exact on analytic stationary patch tests when `C_field` is already exact, but
it is not the conservative Yu et al. 2025 Eq. 38 recovery and must remain
explicit until validated on higher-level canaries.
"""
struct CNEBBField <: AbstractPolymerWallBC end

"""
    CNEBBFieldEquilibrium()

Diagnostic CNEBB variant that first applies `CNEBBField()` and then resets
explicit cut-link wall-cell populations to local equilibrium. This isolates
whether residual errors come from the wall-cell macro value or from outgoing
non-equilibrium populations propagated away from that wall cell. It is not a
conservative Yu et al. 2025 boundary condition.
"""
struct CNEBBFieldEquilibrium <: AbstractPolymerWallBC end

"""
    CNEBBEqGradient()

Diagnostic CNEBB variant that recovers the wall macro field by preserving the
local equilibrium gradient and transporting only non-equilibrium residuals.
It is exact for linear equilibrium patches at cut-links; unlike `CNEBB()`,
this is not the strict Yu et al. 2025 Eq. 38 recovery.
"""
struct CNEBBEqGradient <: AbstractPolymerWallBC end

"""
    CNEBBCutLinkEqGradient()

Diagnostic CNEBB variant that applies the `CNEBBEqGradient` equilibrium
correction only on explicit embedded cut-link cells (`q_wall > 0`). Domain
walls keep the strict Yu/CNEBB recovery, avoiding the planar Poiseuille
instability of the full `CNEBBEqGradient` canary.
"""
struct CNEBBCutLinkEqGradient <: AbstractPolymerWallBC end

"""
    ExtrapEqWallBC()

Production 2D cut-link boundary condition for conformation populations. For
each population whose upstream node is solid, reconstruct the virtual upstream
macro state by wall-aware linear extrapolation from the fluid cut cell, write
that population to equilibrium, then rebalance the rest population to preserve
the local conformation field exactly.
"""
struct ExtrapEqWallBC <: AbstractPolymerWallBC end

"""
    LogFieldWallBC()

Wall treatment for `LogConfOldroydB` only. It applies field-pinned scalar
NEBB to the evolved log-conformation components Ψ, preserving the node macro
Ψ exactly at wall-adjacent cells. This is intentionally separate from
`CNEBBField()`: it is a log-space closure, not the conservative C-space
Yu/Liu CNEBB formula.
"""
struct LogFieldWallBC <: AbstractPolymerWallBC end

function _assert_validation_polymer_wall_bc(bc::AbstractPolymerWallBC;
                                            allow_diagnostic::Bool=false)
    if !allow_diagnostic &&
       (bc isa CNEBBField || bc isa CNEBBFieldEquilibrium ||
        bc isa CNEBBEqGradient || bc isa CNEBBCutLinkEqGradient)
        error("$(typeof(bc)) is a diagnostic wall BC and is not allowed in validation flows; use CNEBB() or pass allow_diagnostic_polymer_bc=true in audit code.")
    end
    return nothing
end

function _assert_validation_conformation_collision_window(
        conformation_collision::Symbol,
        tau_plus;
        allow_diagnostic::Bool=false)
    allow_diagnostic && return nothing
    tau_p_f = Float64(tau_plus)
    high_schmidt_regularized =
        conformation_collision in (:regularized, :liu_eq26) &&
        0.5 < tau_p_f <= 0.5002
    validated =
        (conformation_collision === :trt && abs(tau_p_f - 1.0) ≤ 1e-12) ||
        high_schmidt_regularized
    validated || error("conformation_collision=$(conformation_collision) with tau_plus=$(tau_plus) is outside the analytic CDE patch-test validation windows; use (:trt, 1.0), (:regularized/:liu_eq26, 0.5 < tau_plus ≤ 0.5002), or pass allow_diagnostic_conformation_collision=true in audit code.")
    return nothing
end

function _assert_validation_log_wall_bc(model::AbstractPolymerModel,
                                        bc::AbstractPolymerWallBC;
                                        allow_diagnostic::Bool=false)
    allow_diagnostic && return nothing
    if uses_log_conformation(model)
        if !(bc isa NoPolymerWallBC || bc isa LogFieldWallBC)
            error("log-conformation with polymer wall BC $(typeof(bc)) applies a C-space wall reconstruction to Ψ=log(C); use LogFieldWallBC() or pass allow_diagnostic_log_wall_bc=true in audit code.")
        end
    elseif bc isa LogFieldWallBC
        error("LogFieldWallBC is defined only for LogConfOldroydB; use CNEBB() or ExtrapEqWallBC() for direct-C validation.")
    end
    return nothing
end

"""
    YLW_A(; tau_plus=1.0)

Yu-Li-Wen 2020 modified curved-wall scheme A applied to conformation
populations: MLS reconstruction plus rest-population leakage correction.
Requires `q_wall`; currently implemented for 2D D2Q9 only.
"""
struct YLW_A{T} <: AbstractPolymerWallBC
    tau_plus::T
end
YLW_A(; tau_plus=1.0) = YLW_A(tau_plus)

"""
    YLW_B(; tau_plus=1.0)

Yu-Li-Wen 2020 modified curved-wall scheme B applied to conformation
populations: MLS reconstruction with fictitious ghost density from the
incoming/outgoing balance. Requires `q_wall`; currently implemented for 2D
D2Q9 only.
"""
struct YLW_B{T} <: AbstractPolymerWallBC
    tau_plus::T
end
YLW_B(; tau_plus=1.0) = YLW_B(tau_plus)

"""
    YLWBalanceOnly()

Diagnostic Yu-Li-Wen-inspired balance-only variant: reconstruct missing
populations with local NEBB, then apply the YLW scheme-A rest-population
incoming/outgoing balance. No MLS interpolation and no fictitious density.
"""
struct YLWBalanceOnly <: AbstractPolymerWallBC end

"""
    NoPolymerWallBC()

Do nothing — let the underlying `stream_2d!` handle bounce-back as the
default. Use only if there are no solid walls adjacent to fluid cells.
"""
struct NoPolymerWallBC <: AbstractPolymerWallBC end

_polymer_wall_links_present(q_wall) = any(q -> q > zero(q), q_wall)
_solid_cells_present(is_solid) = any(is_solid)

function _assert_no_polymer_wall_links(q_wall, ::NoPolymerWallBC)
    _polymer_wall_links_present(q_wall) &&
        error("NoPolymerWallBC cannot be used when q_wall contains polymer wall links; use CNEBB() or an explicit diagnostic wall BC.")
    return nothing
end

function _assert_no_solid_polymer_walls(is_solid, ::NoPolymerWallBC)
    _solid_cells_present(is_solid) &&
        error("NoPolymerWallBC cannot be used when is_solid contains polymer wall cells; use CNEBB() or an explicit diagnostic wall BC.")
    return nothing
end

"""
    apply_polymer_wall_bc!(g_post, g_pre, is_solid, C, bc::AbstractPolymerWallBC)

Dispatch hook; default falls through for `NoPolymerWallBC`.
"""
function apply_polymer_wall_bc!(g_post, g_pre, is_solid, C, bc::NoPolymerWallBC)
    _assert_no_solid_polymer_walls(is_solid, bc)
    return nothing
end
function apply_polymer_wall_bc!(g_post, g_pre, is_solid, C, ux, uy, bc::NoPolymerWallBC)
    _assert_no_solid_polymer_walls(is_solid, bc)
    return nothing
end
function apply_polymer_wall_bc!(g_post, g_pre, is_solid, q_wall, C, bc::NoPolymerWallBC)
    _assert_no_polymer_wall_links(q_wall, bc)
    return nothing
end
function apply_polymer_wall_bc!(g_post, g_pre, is_solid, q_wall, C, ux, uy, bc::NoPolymerWallBC)
    _assert_no_polymer_wall_links(q_wall, bc)
    return nothing
end
# CNEBB dispatches on the dimensionality of the populations array:
# 3 dims = (Nx, Ny, 9) → 2D D2Q9; 4 dims = (Nx, Ny, Nz, 19) → 3D D3Q19.
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid, C,
                                  ::CNEBB) where {T}
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid,
                                  q_wall, C, ::CNEBB) where {T}
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid, C,
                                  ux, uy, ::CNEBB) where {T}
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C, ux, uy)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid,
                                  q_wall, C, ux, uy, ::CNEBB) where {T}
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C, ux, uy)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid, C,
                                  ::CNEBBQAware) where {T}
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid,
                                  q_wall, C, ::CNEBBQAware) where {T}
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, q_wall, C)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid, C,
                                  ux, uy, ::CNEBBQAware) where {T}
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C, ux, uy)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid,
                                  q_wall, C, ux, uy, ::CNEBBQAware) where {T}
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, q_wall, C, ux, uy)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid, C,
                                  ::CNEBBField) where {T}
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C;
                                 phi_mode=:field)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid,
                                  q_wall, C, ::CNEBBField) where {T}
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, q_wall, C;
                                 phi_mode=:field)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid, C,
                                  ux, uy, ::CNEBBField) where {T}
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C, ux, uy;
                                 phi_mode=:field)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid,
                                  q_wall, C, ux, uy, ::CNEBBField) where {T}
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, q_wall, C, ux, uy;
                                 phi_mode=:field)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid, C,
                                  ::CNEBBFieldEquilibrium) where {T}
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C;
                                 phi_mode=:field)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid,
                                  q_wall, C, ::CNEBBFieldEquilibrium) where {T}
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, q_wall, C;
                                 phi_mode=:field)
    reset_cutlink_conformation_equilibrium_2d!(g_post, C, is_solid, q_wall)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid, C,
                                  ux, uy, ::CNEBBFieldEquilibrium) where {T}
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C, ux, uy;
                                 phi_mode=:field)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid,
                                  q_wall, C, ux, uy,
                                  ::CNEBBFieldEquilibrium) where {T}
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, q_wall, C, ux, uy;
                                 phi_mode=:field)
    reset_cutlink_conformation_equilibrium_2d!(g_post, C, ux, uy, is_solid, q_wall)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid, C,
                                  ::CNEBBEqGradient) where {T}
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C;
                                 phi_mode=:eq_gradient)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid,
                                  q_wall, C, ::CNEBBEqGradient) where {T}
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, q_wall, C;
                                 phi_mode=:eq_gradient)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid, C,
                                  ux, uy, ::CNEBBEqGradient) where {T}
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C, ux, uy;
                                 phi_mode=:eq_gradient)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid,
                                  q_wall, C, ux, uy, ::CNEBBEqGradient) where {T}
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, q_wall, C, ux, uy;
                                 phi_mode=:eq_gradient)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid, C,
                                  ::CNEBBCutLinkEqGradient) where {T}
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid,
                                  q_wall, C, ::CNEBBCutLinkEqGradient) where {T}
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, q_wall, C;
                                 phi_mode=:eq_gradient_cutlink)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid, C,
                                  ux, uy, ::CNEBBCutLinkEqGradient) where {T}
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C, ux, uy)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid,
                                  q_wall, C, ux, uy,
                                  ::CNEBBCutLinkEqGradient) where {T}
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, q_wall, C, ux, uy;
                                 phi_mode=:eq_gradient_cutlink)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid, C,
                                  ::ExtrapEqWallBC) where {T}
    error("ExtrapEqWallBC requires explicit q_wall, C, ux, and uy fields")
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid, C,
                                  ux, uy, ::ExtrapEqWallBC) where {T}
    error("ExtrapEqWallBC requires explicit q_wall cut-link fractions")
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid,
                                  q_wall, C, ::ExtrapEqWallBC) where {T}
    error("ExtrapEqWallBC requires ux and uy fields")
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid,
                                  q_wall, C, ux, uy,
                                  ::ExtrapEqWallBC) where {T}
    apply_extrap_eq_conformation_2d!(g_post, is_solid, q_wall, C, ux, uy)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid, C,
                                  ::LogFieldWallBC) where {T}
    apply_cnebb_field_conformation_2d!(g_post, g_pre, is_solid, C)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid,
                                  q_wall, C, ::LogFieldWallBC) where {T}
    apply_cnebb_field_conformation_2d!(g_post, g_pre, is_solid, q_wall, C)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid, C,
                                  ux, uy, ::LogFieldWallBC) where {T}
    apply_cnebb_field_conformation_2d!(g_post, g_pre, is_solid, C, ux, uy)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid,
                                  q_wall, C, ux, uy, ::LogFieldWallBC) where {T}
    apply_cnebb_field_conformation_2d!(g_post, g_pre, is_solid, q_wall,
                                       C, ux, uy)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,4}, g_pre, is_solid, C,
                                  ::CNEBB) where {T}
    apply_cnebb_conformation_3d!(g_post, g_pre, is_solid, C)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,4}, g_pre, is_solid, C,
                                  ::CNEBBQAware) where {T}
    apply_cnebb_conformation_3d!(g_post, g_pre, is_solid, C)
    return nothing
end

function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid,
                                  q_wall, C, bc::YLW_A) where {T}
    apply_ylw_a_conformation_2d!(g_post, g_pre, is_solid, q_wall, C;
                                 tau_plus=bc.tau_plus)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid,
                                  q_wall, C, ux, uy, bc::YLW_A) where {T}
    apply_ylw_a_conformation_2d!(g_post, g_pre, is_solid, q_wall, C, ux, uy;
                                 tau_plus=bc.tau_plus)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid,
                                  q_wall, C, bc::YLW_B) where {T}
    apply_ylw_b_conformation_2d!(g_post, g_pre, is_solid, q_wall, C;
                                 tau_plus=bc.tau_plus)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid,
                                  q_wall, C, ux, uy, bc::YLW_B) where {T}
    apply_ylw_b_conformation_2d!(g_post, g_pre, is_solid, q_wall, C, ux, uy;
                                 tau_plus=bc.tau_plus)
    return nothing
end

function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid,
                                  q_wall, C, ::YLWBalanceOnly) where {T}
    apply_ylw_balance_conformation_2d!(g_post, g_pre, is_solid, q_wall, C)
    return nothing
end
function apply_polymer_wall_bc!(g_post::AbstractArray{T,3}, g_pre, is_solid,
                                  q_wall, C, ux, uy, ::YLWBalanceOnly) where {T}
    apply_ylw_balance_conformation_2d!(g_post, g_pre, is_solid, q_wall, C, ux, uy)
    return nothing
end
