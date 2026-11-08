"""
    TinyMLP{NIN,H,L}(; shift=(0.0,...), scale=(1.0,...))

A scalar-output dense network with `NIN` inputs, `L ≥ 1` tanh hidden layers
of width `H`. Normalize input `i` as `(x[i]-shift[i])/scale[i]`. Fixed tuples
hold finite shifts and finite nonzero scales; dimensions are type parameters.

Flat parameters contain each hidden layer's column-major weight matrix,
followed by its bias vector; then the output's `H` weights and scalar bias.
The first matrix is `H × NIN`, subsequent matrices are `H × H`.
The count is `H*NIN + H + (L-1)*(H*H+H) + H + 1`.
Evaluation and the handwritten reverse pass use only static tuples.
"""
struct TinyMLP{NIN,H,L,T} <: AbstractScalarLaw
    shift::NTuple{NIN,T}
    scale::NTuple{NIN,T}
    function TinyMLP{NIN,H,L}(; shift=ntuple(_ -> 0.0, NIN),
                              scale=ntuple(_ -> 1.0, NIN)) where {NIN,H,L}
        all(d -> d isa Int && d > 0, (NIN, H, L)) ||
            throw(ArgumentError("NIN, H and L must be positive integers"))
        (shift isa Tuple && scale isa Tuple && length(shift) == NIN && length(scale) == NIN) ||
            throw(ArgumentError("shift and scale must be tuples of length NIN"))
        all(isfinite, shift) || throw(ArgumentError("shift must be finite"))
        all(v -> isfinite(v) && !iszero(v), scale) ||
            throw(ArgumentError("scale must be finite and nonzero"))
        values = promote(map(float, shift)..., map(float, scale)...)
        T = typeof(values[1])
        return new{NIN,H,L,T}(ntuple(i -> values[i], NIN),
                              ntuple(i -> values[NIN+i], NIN))
    end
end

@inline law_ninputs(::TinyMLP{NIN}) where {NIN} = NIN
@inline law_nparams(::TinyMLP{NIN,H,L}) where {NIN,H,L} =
    H*NIN + H + (L-1)*(H*H+H) + H + 1
@inline _law_softplus(t) = t > zero(t) ? t + log1p(exp(-t)) : log1p(exp(t))
@inline _law_logistic(t) = t >= zero(t) ? inv(one(t) + exp(-t)) : exp(t)/(one(t) + exp(t))
@inline _law_weight(t, positive::Bool) = positive ? _law_softplus(t) : t
@inline _law_weight_derivative(t, positive::Bool) = positive ? _law_logistic(t) : one(t)
@inline _mlp_offset(::TinyMLP{NIN,H}, ::Val{K}) where {NIN,H,K} =
    K == 1 ? 0 : H*NIN + H + (K-2)*(H*H+H)
@inline _mlp_output_offset(m::TinyMLP{NIN,H,L}) where {NIN,H,L} = law_nparams(m)-H-1
@inline _mlp_width(::TinyMLP{NIN,H}) where {NIN,H} = H
@inline _mlp_normalize(m::TinyMLP{NIN}, x) where {NIN} =
    promote(ntuple(i -> (x[i]-m.shift[i])/m.scale[i], Val(NIN))...)

@inline function _mlp_hidden(m::TinyMLP{NIN,H}, a::NTuple{NI}, theta,
                             k::Val{K}, ::Val{MONO}, base::Int) where {NIN,H,NI,K,MONO}
    offset = base + _mlp_offset(m, k)
    return ntuple(Val(H)) do i
        @inbounds v = theta[offset+H*NI+i] + zero(a[1])
        @inbounds for j in 1:NI
            w = _law_weight(theta[offset+(j-1)*H+i], MONO && (K > 1 || j == 1))
            v += w * a[j]
        end
        tanh(v)
    end
end

@inline function _mlp_cache(m::TinyMLP{NIN,H,L}, a, theta, k::Val{K}, mode, base::Int) where {NIN,H,L,K}
    h = _mlp_hidden(m, a, theta, k, mode, base)
    K == L && return (h,)
    return (h, _mlp_cache(m, h, theta, Val(K+1), mode, base)...)
end

@inline function _mlp_output(m::TinyMLP{NIN,H}, h, theta, ::Val{MONO}, base::Int) where {NIN,H,MONO}
    offset = base + _mlp_output_offset(m)
    @inbounds v = theta[offset+H+1] + zero(h[1])
    @inbounds for i in 1:H
        v += _law_weight(theta[offset+i], MONO) * h[i]
    end
    return v
end

@inline function _mlp_value(m::TinyMLP, x, theta, mode, base::Int=0)
    z = _mlp_normalize(m, x)
    layers = _mlp_cache(m, z, theta, Val(1), mode, base)
    return _mlp_output(m, last(layers), theta, mode, base)
end

@inline law_value(m::TinyMLP, x::Tuple, theta::AbstractVector) =
    _mlp_value(m, x, theta, Val(false))

@inline function _mlp_reverse!(gtheta, m::TinyMLP{NIN,H}, z, layers, theta,
                               delta, k::Val{K}, mode::Val{MONO}, base::Int) where {NIN,H,K,MONO}
    h = layers[K]
    a = K == 1 ? z : layers[K-1]
    ni = K == 1 ? NIN : H
    offset = base + _mlp_offset(m, k)
    d = ntuple(i -> delta[i] * (one(h[i]) - h[i]*h[i]), Val(H))
    @inbounds for i in 1:H
        gtheta[offset+H*ni+i] += d[i]
        for j in 1:ni
            p = offset+(j-1)*H+i
            gtheta[p] += d[i]*a[j]*_law_weight_derivative(theta[p], MONO && (K > 1 || j == 1))
        end
    end
    da = ntuple(Val(ni)) do j
        v = zero(d[1] * theta[1])
        @inbounds for i in 1:H
            v += d[i] * _law_weight(theta[offset+(j-1)*H+i], MONO && (K > 1 || j == 1))
        end
        v
    end
    K == 1 && return da
    return _mlp_reverse!(gtheta, m, z, layers, theta, da, Val(K-1), mode, base)
end

@inline function _mlp_pullback!(gtheta, m::TinyMLP{NIN,H,L}, x, theta,
                                s, mode::Val{MONO}, base::Int=0) where {NIN,H,L,MONO}
    z = _mlp_normalize(m, x)
    layers = _mlp_cache(m, z, theta, Val(1), mode, base)
    h = last(layers)
    offset = base + _mlp_output_offset(m)
    @inbounds for i in 1:H
        gtheta[offset+i] += s*h[i]*_law_weight_derivative(theta[offset+i], MONO)
    end
    @inbounds gtheta[offset+H+1] += s
    delta = ntuple(i -> s*_law_weight(theta[offset+i], MONO), Val(H))
    dz = _mlp_reverse!(gtheta, m, z, layers, theta, delta, Val(L), mode, base)
    return ntuple(i -> dz[i]/m.scale[i], Val(NIN))
end

@inline law_pullback!(gtheta, m::TinyMLP, x::Tuple, theta::AbstractVector, s) =
    _mlp_pullback!(gtheta, m, x, theta, s, Val(false))

function law_init(m::TinyMLP; rng=nothing)
    theta = [0.05 * (rng === nothing ? randn() : randn(rng)) for _ in 1:law_nparams(m)]
    theta[end] = 0.0
    return theta
end

"""
    LearnedViscosity(mlp::TinyMLP; nu_ref=1.0, monotone=true, epsilon=1e-3)

Positive viscosity with inputs `(γ̇, fields...)`, `γ̇ > 0`. Normalize
`(log(γ̇), fields...)` using `mlp.shift` and `mlp.scale`, writing the result
as `(u1, z...)`. `nu_ref > 0` sets the viscosity unit.

With `monotone=true`, log stress is
`F = log(τ/nu_ref) = (epsilon + softplus(raw_slope))*u1 + g(z) + h(u1,z)`
and `ν = nu_ref*exp(F-log(γ̇))`. The network `h` uses `mlp` with softplus
on the first input column, all subsequent hidden weights and output weights;
its biases and field-input columns are unrestricted. The separate network
`g` has the same width and depth, only the field inputs and no constraints.
For `NIN=1`, `g` is absent. A positive first normalization scale and fixed
`epsilon > 0` ensure strictly increasing stress at fixed fields. Viscosity
itself may decrease with shear rate.

Flat parameters are `[h TinyMLP layout, g TinyMLP layout, raw_slope]`, where
`g` uses `NIN-1` inputs and is omitted for `NIN=1`. With `monotone=false`,
parameters retain exactly the original TinyMLP layout and
`ν = nu_ref*exp(MLP)`; stress monotonicity is not imposed.

`law_init` uses hidden raw parameters drawn from `0.05*randn`, zero output
biases, effective `h` output weights `w0=0.01` (raw inverse softplus), and
unrestricted output weights drawn from `0.01*randn` for `g` and free mode.
For `epsilon < 1`, the initial skip slope is approximately one. The bounded
initial residual has magnitude at most `H*w0 + sum(abs, g_output_weights)`.
Thus viscosity starts near `nu_ref*exp(u1-log(γ̇))`, which equals `nu_ref`
only for identity shear normalization. For `epsilon ≥ 1`, the lower bound
precludes a unit initial slope. All configuration values must be finite.
No clipping is applied; extreme parameters can overflow the exponential.
"""
struct LearnedViscosity{M<:TinyMLP,MONO,T,G} <: AbstractScalarLaw
    mlp::M
    field_mlp::G
    nu_ref::T
    monotone::Bool
    epsilon::T
    function LearnedViscosity(mlp::M; nu_ref=1.0, monotone::Bool=true,
                              epsilon=1e-3) where {M<:TinyMLP}
        isfinite(nu_ref) && nu_ref > 0 || throw(ArgumentError("nu_ref must be finite and positive"))
        isfinite(epsilon) && epsilon > 0 || throw(ArgumentError("epsilon must be finite and positive"))
        !monotone || mlp.scale[1] > 0 ||
            throw(ArgumentError("monotone viscosity requires a positive first scale"))
        nr, ep = promote(float(nu_ref), float(epsilon))
        fields = monotone ? _law_field_mlp(mlp) : nothing
        return new{M,monotone,typeof(nr),typeof(fields)}(mlp, fields, nr, monotone, ep)
    end
end

@inline function _law_field_mlp(m::TinyMLP{NIN,H,L}) where {NIN,H,L}
    NIN == 1 && return nothing
    return TinyMLP{NIN-1,H,L}(; shift=Base.tail(m.shift), scale=Base.tail(m.scale))
end
@inline _law_field_nparams(::Nothing) = 0
@inline _law_field_nparams(m::TinyMLP) = law_nparams(m)
@inline law_ninputs(law::LearnedViscosity) = law_ninputs(law.mlp)
@inline law_nparams(law::LearnedViscosity{M,MONO}) where {M,MONO} =
    law_nparams(law.mlp) + _law_field_nparams(law.field_mlp) + MONO
@inline _law_loginputs(x::Tuple) = (log(x[1]), Base.tail(x)...)

@inline function law_value(law::LearnedViscosity{M,MONO}, x::Tuple,
                           theta::AbstractVector) where {M,MONO}
    u = _law_loginputs(x)
    f = _mlp_value(law.mlp, u, theta, Val(MONO))
    if MONO
        @inbounds slope = law.epsilon + _law_softplus(theta[law_nparams(law)])
        u1 = (u[1]-law.mlp.shift[1])/law.mlp.scale[1]
        f += slope*u1 - u[1]
        if law.field_mlp !== nothing
            f += _mlp_value(law.field_mlp, Base.tail(u), theta, Val(false),
                            law_nparams(law.mlp))
        end
    end
    return law.nu_ref * exp(f)
end

@inline function law_pullback!(gtheta, law::LearnedViscosity{M,MONO}, x::Tuple,
                              theta::AbstractVector, s) where {M,MONO}
    u = _law_loginputs(x)
    q = s * law_value(law, x, theta)
    du = _mlp_pullback!(gtheta, law.mlp, u, theta, q, Val(MONO))
    if MONO
        p = law_nparams(law)
        u1 = (u[1]-law.mlp.shift[1])/law.mlp.scale[1]
        @inbounds gtheta[p] += q*u1*_law_logistic(theta[p])
        @inbounds slope = law.epsilon + _law_softplus(theta[p])
        dgamma = (du[1] + q*(slope/law.mlp.scale[1]-one(slope)))/x[1]
        if law.field_mlp !== nothing
            dz = _mlp_pullback!(gtheta, law.field_mlp, Base.tail(u), theta,
                                q, Val(false), law_nparams(law.mlp))
            fields = ntuple(i -> du[i+1]+dz[i], Val(law_ninputs(law)-1))
            return (dgamma, fields...)
        end
        return (dgamma,)
    end
    return (du[1]/x[1], Base.tail(du)...)
end

@inline _law_invsoftplus(x) = x + log(-expm1(-x))

function _law_init_output!(theta, m::TinyMLP, base::Int, monotone::Bool)
    offset = base + _mlp_output_offset(m)
    for i in 1:_mlp_width(m)
        theta[offset+i] = monotone ? _law_invsoftplus(0.01) : 0.2*theta[offset+i]
    end
    return theta
end

function law_init(law::LearnedViscosity{M,MONO}; rng=nothing) where {M,MONO}
    theta = law_init(law.mlp; rng)
    _law_init_output!(theta, law.mlp, 0, MONO)
    if MONO
        if law.field_mlp !== nothing
            append!(theta, law_init(law.field_mlp; rng))
            _law_init_output!(theta, law.field_mlp, law_nparams(law.mlp), false)
        end
        skip = law.epsilon < 1 ? 1.0-law.epsilon : 1e-3
        push!(theta, _law_invsoftplus(skip))
    end
    return theta
end
