# Plane-PIV observation operator for 3D velocity fields (cell-centred Cartesian grid).
#
# Layout: u[i, j, k] holds the value at x = (i - 1/2) hx, y = (j - 1/2) hy, z = (k - 1/2) hz
# (the layout of the D3Q19 drivers). Interpolation is trilinear between cell centres, with
# constant extrapolation outside the first/last centre (linear exactness holds in the interior).
#
# The operator is linear and separable: with the normal axis n and in-plane axes (a, b) (always
# in ascending axis order),
#     sample[p, q] = sum_{ia, ib, k} Wa[p, ia] * Wb[q, ib] * wn[k] * f(ia, ib, k)
# where wn is the light-sheet weight along the normal, Wa/Wb the box-averaged interpolation
# along the in-plane axes. Only broadcasts, reductions and matrix products touch the field, so
# the same code runs on CPU and CUDA arrays; the weights are built on the host and copied to
# the array type of a template field.

"""
    PIVPlane(; normal, position, xs, ys, window=(0, 0), thickness=0, stereo=false,
               nquad_window=nothing, nquad_thickness=nothing)

Description of a PIV measurement plane (physical units, same units as the grid spacing `h`
given to [`piv_operator`](@ref)).

- `normal`: `:x`, `:y` or `:z`; `position`: physical coordinate of the plane along it (need not
  be a cell centre).
- `xs`, `ys`: centres of the vectors along the two in-plane axes, in ascending axis order
  (`normal = :x` -> (y, z); `:y` -> (x, z); `:z` -> (x, y)). They may differ from the simulation grid.
- `window = (wa, wb)`: interrogation-window size. Each sample is the box average over
  `[c - w/2, c + w/2]` of the trilinear interpolant, discretised by the midpoint rule with
  `nquad_window` sub-points per axis (default: odd count giving spacing <= h/2, 1 if `w = 0`).
- `thickness`: standard deviation `sigma` of the Gaussian light-sheet weight
  `w(s) ~ exp(-s^2 / (2 sigma^2))` of the normal distance `s`, truncated at `3 sigma`,
  discretised on `nquad_thickness` equispaced symmetric points (default: odd count with
  spacing <= h/2) and normalised to sum 1. `thickness = 0` is pure trilinear interpolation.
- `stereo`: also return the out-of-plane component. Samples are the tuple of in-plane
  components in ascending axis order, then the out-of-plane one.
"""
struct PIVPlane{XS,YS}
    normal::Symbol
    position::Float64
    xs::XS
    ys::YS
    window::NTuple{2,Float64}
    thickness::Float64
    stereo::Bool
    nquad_window::Union{Nothing,NTuple{2,Int}}
    nquad_thickness::Union{Nothing,Int}
end

function PIVPlane(; normal::Symbol, position, xs, ys, window=(0, 0), thickness=0,
                  stereo::Bool=false, nquad_window=nothing, nquad_thickness=nothing)
    normal in (:x, :y, :z) || throw(ArgumentError("normal must be :x, :y or :z, got $normal"))
    thickness >= 0 || throw(ArgumentError("thickness must be >= 0"))
    all(>=(0), window) || throw(ArgumentError("window sizes must be >= 0"))
    nqw = nquad_window === nothing ? nothing :
          (nquad_window isa Integer ? (Int(nquad_window), Int(nquad_window)) : (Int(nquad_window[1]), Int(nquad_window[2])))
    PIVPlane(normal, Float64(position), collect(Float64, xs), collect(Float64, ys),
             (Float64(window[1]), Float64(window[2])), Float64(thickness), stereo, nqw,
             nquad_thickness === nothing ? nothing : Int(nquad_thickness))
end

"""
    PIVOperator

Linear map built by [`piv_operator`](@ref); use [`piv_apply`](@ref),
[`piv_apply_adjoint`](@ref) and [`piv_misfit`](@ref).
"""
struct PIVOperator{Ax,S,T,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    Wa::M      # (nsa, Na)
    Wb::M      # (nsb, Nb)
    wn::V      # (Nn,)
end

# 1D trilinear weights: sparse (index, weight) pairs for a physical position, clamped.
@inline function _hat_weights!(row::AbstractVector{Float64}, x::Float64, h::Float64, N::Int, w::Float64)
    t = clamp(x / h + 0.5, 1.0, Float64(N))
    i0 = min(floor(Int, t), N - 1)
    N == 1 && (row[1] += w; return)
    fr = t - i0
    row[i0] += w * (1 - fr)
    row[i0 + 1] += w * fr
    return
end

_oddcount(len, h) = len == 0 ? 1 : max(3, 2 * cld(ceil(Int, 2len / h), 2) + 1)

function _inplane_matrix(centres::Vector{Float64}, width::Float64, h::Float64, N::Int, nq)
    m = nq === nothing ? _oddcount(width, h) : nq
    W = zeros(Float64, length(centres), N)
    for (p, c) in enumerate(centres)
        row = view(W, p, :)
        for q in 1:m
            off = width == 0 ? 0.0 : ((q - 0.5) / m - 0.5) * width
            _hat_weights!(row, c + off, h, N, 1.0 / m)
        end
    end
    return W
end

function _normal_weights(pos::Float64, sigma::Float64, h::Float64, N::Int, nq)
    wn = zeros(Float64, N)
    if sigma == 0
        _hat_weights!(wn, pos, h, N, 1.0)
    else
        m = nq === nothing ? _oddcount(6sigma, h) : (isodd(nq) ? nq : nq + 1)
        s = range(-3sigma, 3sigma; length=m)
        g = exp.(-(s .^ 2) ./ (2sigma^2))
        g ./= sum(g)
        for q in 1:m
            _hat_weights!(wn, pos + s[q], h, N, g[q])
        end
    end
    return wn
end

"""
    piv_operator(plane::PIVPlane, template::AbstractArray{T,3}; h=(1, 1, 1)) -> PIVOperator

Build the operator for fields shaped like `template` (cell-centred, spacing `h = (hx, hy, hz)`).
The weights are stored with the element and array type of `template` (CPU or CUDA).
"""
function piv_operator(plane::PIVPlane, template::AbstractArray{T,3}; h=(1, 1, 1)) where {T}
    n = (:x, :y, :z)
    ax = findfirst(==(plane.normal), n)
    ia, ib = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)
    N = size(template)
    hh = (Float64(h[1]), Float64(h[2]), Float64(h[3]))
    nqw = plane.nquad_window
    Wa = _inplane_matrix(plane.xs, plane.window[1], hh[ia], N[ia], nqw === nothing ? nothing : nqw[1])
    Wb = _inplane_matrix(plane.ys, plane.window[2], hh[ib], N[ib], nqw === nothing ? nothing : nqw[2])
    wn = _normal_weights(plane.position, plane.thickness, hh[ax], N[ax], plane.nquad_thickness)
    dev(A) = copyto!(similar(template, T, size(A)), T.(A))
    return PIVOperator{ax,plane.stereo,T,typeof(dev(Wa)),typeof(dev(wn))}(dev(Wa), dev(Wb), dev(wn))
end

_ncomp(::PIVOperator{Ax,S}) where {Ax,S} = S ? Val(3) : Val(2)
# Velocity component indices (1=x, 2=y, 3=z) measured, in sample order.
_components(::PIVOperator{1}) = (2, 3, 1)
_components(::PIVOperator{2}) = (1, 3, 2)
_components(::PIVOperator{3}) = (1, 2, 3)

_contract(::Val{1}, f, wn) = dropdims(sum(f .* reshape(wn, :, 1, 1); dims=1); dims=1)
_contract(::Val{2}, f, wn) = dropdims(sum(f .* reshape(wn, 1, :, 1); dims=2); dims=2)
_contract(::Val{3}, f, wn) = dropdims(sum(f .* reshape(wn, 1, 1, :); dims=3); dims=3)
_expand(::Val{1}, h, wn) = reshape(h, 1, size(h)...) .* reshape(wn, :, 1, 1)
_expand(::Val{2}, h, wn) = reshape(h, size(h, 1), 1, size(h, 2)) .* reshape(wn, 1, :, 1)
_expand(::Val{3}, h, wn) = reshape(h, size(h)..., 1) .* reshape(wn, 1, 1, :)

"""
    piv_apply(op::PIVOperator{Ax}, u::NTuple{3,AbstractArray{T,3}}) -> NTuple{2 or 3, Matrix}

Samples of the in-plane components (ascending axis order), then the out-of-plane one if the
plane is `stereo`. `u = (ux, uy, uz)`; the unmeasured component may be any array of the right size.
"""
function piv_apply(op::PIVOperator{Ax}, u::NTuple{3,AbstractArray{T,3}}) where {Ax,T}
    comps = _components(op)
    return ntuple(c -> op.Wa * _contract(Val(Ax), u[comps[c]], op.wn) * transpose(op.Wb), _ncomp(op))
end

"""
    piv_apply_adjoint(op, g::Tuple; template) -> NTuple{3,Array}

Exact transpose of [`piv_apply`](@ref): `g` holds one cotangent matrix per measured component;
returns the cotangent `(gx, gy, gz)` of the 3D field (zero for unmeasured components).
`template` is a field array giving the size/array type of the result.
"""
function piv_apply_adjoint(op::PIVOperator{Ax,S,T}, g::Tuple; template::AbstractArray{T,3}) where {Ax,S,T}
    # field cotangent in sample-slot order, zero-padded to three slots, then permuted to (x, y, z)
    slots = ntuple(c -> c <= length(g) ? _expand(Val(Ax), transpose(op.Wa) * g[c] * op.Wb, op.wn) :
                        zero(template), 3)
    inv = _slot_of_component(op)
    return ntuple(d -> slots[inv[d]], 3)
end
_slot_of_component(::PIVOperator{1}) = (3, 1, 2)
_slot_of_component(::PIVOperator{2}) = (1, 3, 2)
_slot_of_component(::PIVOperator{3}) = (1, 2, 3)

"""
    piv_misfit(op, u, data; mask=nothing) -> (L, grad)

`L = 1/2 sum_k m_k (sample_k - data_k)^2` over all measured components, with `data` a tuple of
matrices like [`piv_apply`](@ref)'s output and `mask` (missing vectors: `m = 0`; data there may
be `NaN`) either `nothing`, one matrix shared by all components, or a tuple of matrices.
`grad = (dL/dux, dL/duy, dL/duz)` is the exact gradient with respect to the 3D field.
"""
function piv_misfit(op::PIVOperator, u::NTuple{3,AbstractArray{T,3}}, data::Tuple; mask=nothing) where {T}
    s = piv_apply(op, u)
    nc = length(s)
    m = ntuple(c -> mask === nothing ? one.(s[c]) :
                    (mask isa Tuple ? mask[c] : mask), nc)
    r = ntuple(c -> ifelse.(m[c] .> 0, s[c] .- data[c], zero(T)), nc)
    L = sum(c -> sum(m[c] .* r[c] .^ 2), 1:nc) / 2
    return L, piv_apply_adjoint(op, ntuple(c -> m[c] .* r[c], nc); template=u[1])
end
