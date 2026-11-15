# Relaxation-parameter map of the steady 3D FV conformation problem (plan
# "discrete viscoelastic adjoints", step F1) and the admissible-domain checks of
# parameters and conformation states.
#
# FVRelaxationParameters maps theta to the relaxation time lambda, the polymer
# viscosity nu_p, free entries of the law parameters and the uniform solvent
# viscosity nu_s. The adjoint takes the generic parameter-dual path of
# fv_adjoint.jl (ceil(P / chunk) evaluations of the conformation residual kernel
# with dual coefficients): lambda, nu_p and theta live in the isbits kernel
# argument _FVConformationData3D, so they need no device buffer; nu_s fills one
# device viscosity buffer. G = nu_p / lambda is recomputed from the promoted
# values, so lambda reaches both 1/lambda in the relaxation term and G in the
# polymer stress.
#
# Learned laws (relaxation_theta_array) keep theta in an array: the mapped
# entries are written into one parameter buffer (_FVParamBuffers.aux) of the
# problem's backend per evaluation.
#
# Domains are checked, never clipped: a parameter outside its admissible range
# or a state whose conformation is not positive definite (FENE-P: tr c >= L2)
# throws.

const _FV_RELAXATION_NAMES = (:lambda, :nu_p, :theta, :nu_s)

"""
    FVRelaxationParameters(which...; theta_slots = nothing) <: FVParameterMap

Parameter map of a 3D conformation-active problem
([`FVConformation3D`](@ref)): `which` is a subset of `(:lambda, :nu_p, :theta,
:nu_s)` (as symbols or one tuple), and `θ` concatenates, in the order of
`which`,

- `:lambda`: the relaxation time `λ`;
- `:nu_p`: the polymer viscosity `ν_p` (kinematic; `G = ν_p/λ` follows both);
- `:theta`: the free entries `theta[theta_slots]` of the law parameters. Default
  slots: `(2,)` for [`PolynomialRelaxation`](@ref), i.e. the Giesekus mobility
  `α` with `b1 = 1` fixed (`b1` cannot be separated from `λ` in the relaxation
  term); every entry otherwise (`ε` of the PTT laws, `L²` of FENE-P, the `7H`
  weights of [`NeuralRelaxation`](@ref), whose parameters are an array: the
  mapped ones go through one buffer of the problem's backend);
- `:nu_s`: the solvent viscosity, uniform in every cell.

Per law: Oldroyd-B `(:lambda, :nu_p, :nu_s)`; Giesekus, PTT and FENE-P add
`:theta`. Reparametrisations (log λ, log(L² − 3), ...) belong to the optimiser,
not to the map.

Admissible values (checked at every evaluation, `ArgumentError` otherwise, no
clipping): `λ, ν_p, ν_s > 0`; [`PolynomialRelaxation`](@ref) `b1 > 0` and
`0 ≤ α = b2/b1 ≤ 1/2` (the range of this map, not a general bound of the
Giesekus model); otherwise the law's [`relaxation_check`](@ref) (`ε ≥ 0`,
`L² > 3`). States are checked by [`fv_check_conformation_domain`](@ref), which
[`fv_steady_adjoint`](@ref) and [`fv_steady_tangent`](@ref) call on their state.
"""
struct FVRelaxationParameters <: FVParameterMap
    which::Vector{Symbol}
    theta_slots::Union{Nothing,Vector{Int}}
    function FVRelaxationParameters(which::Symbol...; theta_slots = nothing)
        isempty(which) &&
            throw(ArgumentError("FVRelaxationParameters needs at least one of $(_FV_RELAXATION_NAMES)"))
        for w in which
            w in _FV_RELAXATION_NAMES ||
                throw(ArgumentError("unknown relaxation parameter :$w (expected one of $(_FV_RELAXATION_NAMES))"))
        end
        allunique(which) || throw(ArgumentError("repeated relaxation parameter in $(which)"))
        slots = nothing
        if theta_slots !== nothing
            :theta in which || throw(ArgumentError("theta_slots given but :theta is not mapped"))
            slots = Int[q for q in theta_slots]
            !isempty(slots) && allunique(slots) && all(>=(1), slots) ||
                throw(ArgumentError("theta_slots must be distinct positive indices, got $(theta_slots)"))
        end
        return new(collect(which), slots)
    end
end
FVRelaxationParameters(which::Union{Tuple,AbstractVector}; theta_slots = nothing) =
    FVRelaxationParameters(which...; theta_slots)

_fv_relaxation_default_slots(::PolynomialRelaxation) = [2]
_fv_relaxation_default_slots(law::AbstractRelaxationLaw) = collect(1:relaxation_nparams(law))

function _fv_relaxation_conformation(p)
    p isa FVSteadyProblem3D && p.conformation isa _FVConformationData3D ||
        throw(ArgumentError("FVRelaxationParameters requires a 3D conformation-active problem"))
    return p.conformation
end

# positions of each mapped group in θ (0 when absent), the theta slots, length
function _fv_relaxation_layout(m::FVRelaxationParameters, law::AbstractRelaxationLaw)
    slots = m.theta_slots === nothing ? _fv_relaxation_default_slots(law) : m.theta_slots
    np = relaxation_nparams(law)
    all(q -> q <= np, slots) ||
        throw(ArgumentError("theta_slots $(slots) out of range for $(nameof(typeof(law))) ($np parameters)"))
    off = 0
    lambda = 0; nu_p = 0; theta = 0; nu_s = 0
    for w in m.which
        if w === :lambda
            lambda = off + 1; off += 1
        elseif w === :nu_p
            nu_p = off + 1; off += 1
        elseif w === :theta
            theta = off + 1; off += length(slots)
        else
            nu_s = off + 1; off += 1
        end
    end
    return (; lambda, nu_p, theta, nu_s, slots, n = off)
end

fv_nparams(m::FVRelaxationParameters, prob) = _fv_relaxation_layout(m, _fv_relaxation_conformation(prob).law).n

_fv_real(x::ForwardDiff.Dual) = _fv_real(ForwardDiff.value(x))
_fv_real(x::Real) = x

# law parameter tuple with the mapped slots taken from θ, all entries of type S
function _fv_relaxation_theta(base::NTuple{K,Any}, o, θ, ::Type{S}) where {K,S}
    return ntuple(Val(K)) do q
        r = o.theta == 0 ? nothing : findfirst(==(q), o.slots)
        r === nothing ? S(base[q]) : S(θ[o.theta + r - 1])
    end
end

# Admissible law parameters of the map (real parts); see the docstring.
_fv_relaxation_domain(law::AbstractRelaxationLaw, theta) = relaxation_check(law, theta)
function _fv_relaxation_domain(::PolynomialRelaxation, theta)
    b1, b2 = theta
    b1 > 0 || throw(ArgumentError("PolynomialRelaxation: theta[1] = b1 must be positive, got $b1"))
    0 <= b2 <= b1 / 2 ||
        throw(ArgumentError("PolynomialRelaxation: the mobility alpha = b2/b1 must lie in [0, 1/2] " *
                            "(range of FVRelaxationParameters), got $(b2 / b1)"))
    return nothing
end

# host Float64 copy of the law parameters (tuple, or array on any backend)
_fv_relaxation_base(theta::Tuple) = map(x -> Float64(_fv_real(x)), theta)
_fv_relaxation_base(theta::AbstractArray) = Float64[_fv_real(x) for x in Array(theta)]

# array parameters (relaxation_theta_array): host vector with the mapped slots from θ
function _fv_relaxation_theta_vector(base::AbstractVector, o, θ, ::Type{S}) where {S}
    v = S[S(x) for x in base]
    if o.theta != 0
        for (r, q) in enumerate(o.slots)
            v[q] = S(θ[o.theta + r - 1])
        end
    end
    return v
end

function _fv_relaxation_check(c::_FVConformationData3D, o, θ)
    v = Float64[_fv_real(x) for x in θ]
    all(isfinite, v) || throw(ArgumentError("relaxation parameters must be finite, got $v"))
    for (k, name) in ((o.lambda, "lambda"), (o.nu_p, "nu_p"), (o.nu_s, "nu_s"))
        k == 0 || v[k] > 0 || throw(ArgumentError("$name must be positive, got $(v[k])"))
    end
    if o.theta != 0
        base = _fv_relaxation_base(c.theta)
        th = c.theta isa AbstractArray ? _fv_relaxation_theta_vector(base, o, v, Float64) :
             _fv_relaxation_theta(base, o, v, Float64)
        _fv_relaxation_domain(c.law, th)
    end
    return nothing
end

# nu_s: one viscosity buffer; mapped array parameters: one parameter buffer
# (aux) on the problem's backend
function _fv_param_buffers(m::FVRelaxationParameters, prob, ::Type{T}) where {T}
    c = _fv_relaxation_conformation(prob)
    aux = :theta in m.which && c.theta isa AbstractArray ? similar(c.theta, T, length(c.theta)) : nothing
    return _FVParamBuffers(:nu_s in m.which ? similar(prob.nu, T) : nothing, nothing, nothing, aux)
end

function _fv_param_problem!(B, m::FVRelaxationParameters, p, θ)
    c = _fv_relaxation_conformation(p)
    o = _fv_relaxation_layout(m, c.law)
    length(θ) == o.n || throw(DimensionMismatch("θ has length $(length(θ)), the map expects $(o.n)"))
    _fv_relaxation_check(c, o, θ)
    conf = c
    if o.lambda + o.nu_p + o.theta > 0
        # common type of the coefficients: dual as soon as one mapped entry is
        S = promote_type(eltype(θ), typeof(c.lambda))
        lambda = o.lambda == 0 ? S(c.lambda) : S(θ[o.lambda])
        nu_p = o.nu_p == 0 ? S(c.nu_p) : S(θ[o.nu_p])
        theta = if c.theta isa AbstractArray
            # array parameters: unmapped -> the problem's array (Float64), mapped ->
            # host vector of S copied into the buffer
            o.theta == 0 ? c.theta :
            copyto!(B.aux, _fv_relaxation_theta_vector(_fv_relaxation_base(c.theta), o, θ, S))
        else
            _fv_relaxation_theta(c.theta, o, θ, S)
        end
        # same arithmetic as the lowering (_fv3_lower_conformation): G = nu_p / lambda
        conf = _FVConformationData3D(lambda, nu_p / lambda, nu_p, c.law, theta)
    end
    nu = p.nu
    if o.nu_s > 0
        fill!(B.nu, θ[o.nu_s])
        nu = B.nu
    end
    return _fv_with(p; nu, conformation = conf)
end

# ------------------------------------------------------------------ state domain
"""
    fv_conformation_domain(prob::FVSteadyProblem3D, U) -> (; min_eigenvalue, max_trace, fenep_margin)

Smallest eigenvalue and largest trace of the conformation tensor over the fluid
cells of the state `U` (host or device) of a conformation-active problem, and
for [`FENEPRelaxation`](@ref) the margin `L² − max tr c` (`Inf` for the other
laws). A non-finite component gives `min_eigenvalue = -Inf`. Host reduction
(one copy of `U`).
"""
function fv_conformation_domain(prob::FVSteadyProblem3D, U::AbstractVector)
    c = prob.conformation
    c isa _FVConformationData3D || throw(ArgumentError("problem has no conformation unknowns"))
    length(U) == fv_nunknowns(prob) || throw(DimensionMismatch("state length does not match problem"))
    g = prob.grid
    Uh = Array(U)
    tag = Array(prob.tag)
    M = zeros(Float64, 3, 3)
    mineig = Inf
    maxtr = -Inf
    for cell in 1:fv_ncells(prob)
        tag[cell] == FV_CELL_SOLID && continue
        x = ntuple(m -> Float64(_fv_real(Uh[_fv_cell_index(g, cell, _fv3_conf_slot(g.layout, m))])), Val(6))
        if !all(isfinite, x)
            mineig = -Inf; maxtr = Inf
            continue
        end
        for b in 1:3, a in 1:3
            M[a, b] = _fvc_get(x, a, b)
        end
        mineig = min(mineig, eigmin(Symmetric(M)))
        maxtr = max(maxtr, x[1] + x[4] + x[6])
    end
    margin = c.law isa FENEPRelaxation ? Float64(_fv_real(c.theta[1])) - maxtr : Inf
    return (; min_eigenvalue = mineig, max_trace = maxtr, fenep_margin = margin)
end

"""
    fv_check_conformation_domain(prob::FVSteadyProblem3D, U) -> NamedTuple

[`fv_conformation_domain`](@ref), throwing `DomainError` unless `c` is positive
definite in every fluid cell and, for FENE-P, `max tr c < L²`. A state is
accepted (adjoint, tangent, finite differences, optimiser step) only after this
check; a rejected optimiser trial is a rejection, not a clipped state.
"""
function fv_check_conformation_domain(prob::FVSteadyProblem3D, U::AbstractVector)
    d = fv_conformation_domain(prob, U)
    d.min_eigenvalue > 0 ||
        throw(DomainError(d.min_eigenvalue, "the conformation tensor is not positive definite in every " *
                                            "fluid cell (smallest eigenvalue $(d.min_eigenvalue))"))
    d.fenep_margin > 0 ||
        throw(DomainError(d.fenep_margin, "FENE-P: max tr c = $(d.max_trace) reaches L2 " *
                                          "(margin L2 - max tr c = $(d.fenep_margin))"))
    return d
end

_fv_check_state_domain(prob::_FVConfProblem3D, U) = (fv_check_conformation_domain(prob, U); nothing)
