"""
    DenseNet(nin, hidden; activation=:tanh, output=:identity,
             shift=zeros, scale=ones, max_hidden_layers=4, max_width=64)

Scalar-output feed-forward network as an in-kernel law. `nin` inputs, hidden
widths `hidden` (tuple, e.g. `(16, 16)`), then a width-1 output layer.
`activation` is one tag or a tuple of tags (one per hidden layer) from
`:tanh, :relu, :sigmoid, :softplus, :identity`; `output` is the output
transform from `:identity, :exp, :softplus` (also accepted: any hidden tag).
Input `i` is normalized as `(x[i]-shift[i])/scale[i]`.

The type is `DenseNet{NIN,Ws,Acts,T}`: `Ws = (h1,…,hL,1)` and
`Acts = (a1,…,aL,output)` are compile-time tuples, so the law is isbits and
the forward/reverse passes unroll to straight-line code (no heap allocation,
generic in the number type, including ForwardDiff duals).

Flat parameters follow the [`TinyMLP`](@ref) layout for every layer: the
column-major `out × in` weight matrix, then the bias vector (the output layer
is a `1 × h_L` matrix and a scalar bias). Total count:
`sum(Ws[k]*Ws[k-1] + Ws[k])` with `Ws[0] = NIN`.

`max_hidden_layers` and `max_width` are load-time guards against networks that
would blow up GPU kernel compile time and register use; raise them explicitly
to override.
"""
struct DenseNet{NIN,Ws,Acts,T} <: AbstractScalarLaw
    shift::NTuple{NIN,T}
    scale::NTuple{NIN,T}
    function DenseNet{NIN,Ws,Acts,T}(shift::NTuple{NIN,T},
                                     scale::NTuple{NIN,T}) where {NIN,Ws,Acts,T}
        (NIN isa Int && NIN > 0) || throw(ArgumentError("NIN must be a positive integer"))
        (Ws isa Tuple && all(w -> w isa Int && w > 0, Ws) && length(Ws) >= 2 && Ws[end] == 1) ||
            throw(ArgumentError("Ws must be a tuple of positive widths (h1,…,hL,1), L ≥ 1"))
        (Acts isa Tuple && length(Acts) == length(Ws) && all(a -> a in _DN_ACTS, Acts)) ||
            throw(ArgumentError("Acts must hold one tag per layer, each in $(_DN_ACTS)"))
        all(isfinite, shift) || throw(ArgumentError("shift must be finite"))
        all(v -> isfinite(v) && !iszero(v), scale) ||
            throw(ArgumentError("scale must be finite and nonzero"))
        return new{NIN,Ws,Acts,T}(shift, scale)
    end
end

const _DN_ACTS = (:tanh, :relu, :sigmoid, :softplus, :identity, :exp)
const _DN_OUTPUTS = (:identity, :exp, :softplus)

function DenseNet(nin::Integer, hidden::Tuple; activation=:tanh, output::Symbol=:identity,
                  shift=ntuple(_ -> 0.0, nin), scale=ntuple(_ -> 1.0, nin),
                  max_hidden_layers::Integer=4, max_width::Integer=64)
    L = length(hidden)
    L >= 1 || throw(ArgumentError("at least one hidden layer is required"))
    all(w -> w isa Integer && w > 0, hidden) ||
        throw(ArgumentError("hidden widths must be positive integers"))
    L <= max_hidden_layers ||
        throw(ArgumentError("$(L) hidden layers exceed max_hidden_layers=$(max_hidden_layers); raise it explicitly to override"))
    maximum(hidden) <= max_width ||
        throw(ArgumentError("hidden width $(maximum(hidden)) exceeds max_width=$(max_width); raise it explicitly to override"))
    acts = activation isa Symbol ? ntuple(_ -> activation, L) : Tuple(activation)
    length(acts) == L || throw(ArgumentError("activation must be one tag or a tuple of $(L) tags"))
    all(a -> a in _DN_ACTS && a != :exp, acts) ||
        throw(ArgumentError("hidden activation tags must be in $(filter(!=(:exp), collect(_DN_ACTS)))"))
    output in _DN_OUTPUTS || throw(ArgumentError("output must be one of $(_DN_OUTPUTS)"))
    (shift isa Tuple && scale isa Tuple && length(shift) == nin && length(scale) == nin) ||
        throw(ArgumentError("shift and scale must be tuples of length nin"))
    vals = promote(map(float, shift)..., map(float, scale)...)
    T = typeof(vals[1])
    return DenseNet{Int(nin),(map(Int, hidden)..., 1),(acts..., output),T}(
        ntuple(i -> vals[i], nin), ntuple(i -> vals[nin+i], nin))
end

@inline law_ninputs(::DenseNet{NIN}) where {NIN} = NIN

# Compile-time layer geometry. Layer k maps _dn_in(k) -> Ws[k] numbers.
@inline _dn_in(::DenseNet{NIN,Ws}, ::Val{K}) where {NIN,Ws,K} = K == 1 ? NIN : Ws[K-1]
@inline _dn_nlayers(::DenseNet{NIN,Ws}) where {NIN,Ws} = length(Ws)
Base.@assume_effects :foldable function _dn_offset(nin::Int, Ws::Tuple, k::Int)
    off = 0
    ni = nin
    for m in 1:k-1
        off += Ws[m]*ni + Ws[m]
        ni = Ws[m]
    end
    return off
end
@inline _dn_offset(m::DenseNet{NIN,Ws}, ::Val{K}) where {NIN,Ws,K} = _dn_offset(NIN, Ws, K)
@inline law_nparams(m::DenseNet{NIN,Ws}) where {NIN,Ws} = _dn_offset(NIN, Ws, length(Ws)+1)

# Activations and derivatives expressed through the activation output h.
@inline _dn_act(::Val{:tanh}, z) = tanh(z)
@inline _dn_act(::Val{:relu}, z) = z > zero(z) ? z : zero(z)
@inline _dn_act(::Val{:sigmoid}, z) = _law_logistic(z)
@inline _dn_act(::Val{:softplus}, z) = _law_softplus(z)
@inline _dn_act(::Val{:identity}, z) = z
@inline _dn_act(::Val{:exp}, z) = exp(z)
@inline _dn_dact(::Val{:tanh}, h) = one(h) - h*h
@inline _dn_dact(::Val{:relu}, h) = h > zero(h) ? one(h) : zero(h)
@inline _dn_dact(::Val{:sigmoid}, h) = h*(one(h) - h)
@inline _dn_dact(::Val{:softplus}, h) = -expm1(-h)
@inline _dn_dact(::Val{:identity}, h) = one(h)
@inline _dn_dact(::Val{:exp}, h) = h

@inline _dn_normalize(m::DenseNet{NIN}, x) where {NIN} =
    promote(ntuple(i -> (x[i]-m.shift[i])/m.scale[i], Val(NIN))...)

@inline function _dn_layer(m::DenseNet{NIN,Ws,Acts}, a::NTuple{NI}, theta,
                           k::Val{K}) where {NIN,Ws,Acts,NI,K}
    no = Ws[K]
    act = Val(Acts[K])
    offset = _dn_offset(m, k)
    return ntuple(Val(no)) do i
        @inbounds v = theta[offset+no*NI+i] + zero(a[1])
        @inbounds for j in 1:NI
            v += theta[offset+(j-1)*no+i] * a[j]
        end
        _dn_act(act, v)
    end
end

# Layer loops are unrolled by @generated functions: plain recursion over Val
# widens in inference once the network has mixed widths/activations.
@generated function _dn_cache(m::DenseNet{NIN,Ws}, z, theta) where {NIN,Ws}
    L = length(Ws)
    names = [Symbol(:h, k) for k in 1:L]
    body = Any[]
    for k in 1:L
        prev = k == 1 ? :z : names[k-1]
        push!(body, :($(names[k]) = _dn_layer(m, $prev, theta, Val($k))))
    end
    return Expr(:block, body..., Expr(:return, Expr(:tuple, names...)))
end

@inline function law_value(m::DenseNet, x::Tuple, theta::AbstractVector)
    z = _dn_normalize(m, x)
    return @inbounds last(_dn_cache(m, z, theta))[1]
end

@inline function _dn_back!(gtheta, m::DenseNet{NIN,Ws,Acts}, z, layers, theta,
                           delta, k::Val{K}) where {NIN,Ws,Acts,K}
    no = Ws[K]
    ni = K == 1 ? NIN : Ws[K-1]
    h = layers[K]
    a = K == 1 ? z : layers[K-1]
    act = Val(Acts[K])
    offset = _dn_offset(m, k)
    d = ntuple(i -> delta[i] * _dn_dact(act, h[i]), Val(no))
    @inbounds for i in 1:no
        gtheta[offset+no*ni+i] += d[i]
        for j in 1:ni
            gtheta[offset+(j-1)*no+i] += d[i]*a[j]
        end
    end
    return ntuple(Val(ni)) do j
        v = zero(d[1] * theta[1])
        @inbounds for i in 1:no
            v += d[i] * theta[offset+(j-1)*no+i]
        end
        v
    end
end

@generated function _dn_reverse!(gtheta, m::DenseNet{NIN,Ws}, z, layers, theta, s) where {NIN,Ws}
    L = length(Ws)
    body = Any[:(delta = (s,))]
    for k in L:-1:1
        push!(body, :(delta = _dn_back!(gtheta, m, z, layers, theta, delta, Val($k))))
    end
    return Expr(:block, body..., :(return delta))
end

@inline function law_pullback!(gtheta, m::DenseNet{NIN}, x::Tuple,
                               theta::AbstractVector, s) where {NIN}
    z = _dn_normalize(m, x)
    layers = _dn_cache(m, z, theta)
    dz = _dn_reverse!(gtheta, m, z, layers, theta, s)
    return ntuple(i -> dz[i]/m.scale[i], Val(NIN))
end

function law_init(m::DenseNet{NIN,Ws}; rng=nothing) where {NIN,Ws}
    theta = zeros(law_nparams(m))
    ni = NIN
    for k in 1:length(Ws)
        off = _dn_offset(NIN, Ws, k)
        no = Ws[k]
        for p in 1:no*ni
            theta[off+p] = 0.05 * (rng === nothing ? randn() : randn(rng))
        end
        ni = no
    end
    return theta
end
