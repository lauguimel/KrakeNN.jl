# Steady GNF adjoint (part 2/5): device operator and bordered adjoint.
# Split from src/ad/gnf_adjoint.jl, which holds the overview of the method.


# ============================================================
# Device operator: JVP / VJP kernels
# ============================================================

"""
    GNFStepOperator(lin::GNFLinearisation; backend=CPU(), T=Float64)

Device copy of the 10 × 10 blocks, solid mask, active mask (`active`, in T),
all-ones mask (`ones`, for unprojected products) and gauge vector `chat`
(Float64, ĉ = c/‖c‖). `T` is the arithmetic type of the products;
Float32 products can be used with a Float64 gauge vector.
"""
struct GNFStepOperator{T, TB<:AbstractArray{T,4}, TS<:AbstractMatrix{Bool},
                       TV<:AbstractVector{T}, TC<:AbstractVector{Float64}}
    Nx::Int
    Ny::Int
    n::Int
    B::TB
    solid::TS
    active::TV
    ones::TV
    chat::TC
end

function GNFStepOperator(lin::GNFLinearisation; backend=KernelAbstractions.CPU(), T::Type=Float64)
    Nx, Ny = lin.Nx, lin.Ny
    N = Nx * Ny; n = 10N
    B = KernelAbstractions.allocate(backend, T, Nx, Ny, 10, 10)
    copyto!(B, T.(lin.B[:, :, :, 1:10]))
    solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny)
    copyto!(solid, lin.is_solid)
    act = KernelAbstractions.allocate(backend, T, n)
    copyto!(act, T.(lin.active))
    on = KernelAbstractions.allocate(backend, T, n)
    fill!(on, one(T))
    c = Float64.(lin.active); c[9N+1:10N] .= 0
    c ./= norm(c)
    chat = KernelAbstractions.allocate(backend, Float64, n)
    copyto!(chat, c)
    return GNFStepOperator(Nx, Ny, n, B, solid, act, on, chat)
end

Base.eltype(::GNFStepOperator{T}) where {T} = T

# Per-node pieces of the products. The arithmetic type T is a static parameter
# of these helpers (never a captured local), so the kernels stay type-stable on
# every backend.

# Projected values of the ten components that node (i, j) reads in the JVP:
# the nine arrival slots (pull sources) and its own τ.
@inline function _gnf_gather_arrivals(x, pin, ::Type{T}, i, j, Nx, Ny) where {T}
    return ntuple(Val(10)) do s
        k = s <= 9 ? _spxwy_src(i, j, s, Nx, Ny) : _spxwy_lin(i, j, 10, Nx, Ny)
        @inbounds T(pin[k]) * T(x[k])
    end
end

# Projected values of the ten stored components of node (i, j).
@inline function _gnf_gather_local(x, pin, ::Type{T}, i, j, Nx, Ny) where {T}
    return ntuple(Val(10)) do r
        k = _spxwy_lin(i, j, r, Nx, Ny)
        @inbounds T(pin[k]) * T(x[k])
    end
end

# Block product B_c a (JVP) and its transpose B_cᵀ λ (VJP) at node (i, j).
@inline function _gnf_block_mul(B, a::NTuple{10,T}, i, j) where {T}
    return ntuple(Val(10)) do r
        acc = zero(T)
        for s in 1:10
            @inbounds acc += B[i, j, r, s] * a[s]
        end
        acc
    end
end

@inline function _gnf_block_mul_t(B, λ::NTuple{10,T}, i, j) where {T}
    return ntuple(Val(10)) do s
        acc = zero(T)
        for r in 1:10
            @inbounds acc += B[i, j, r, s] * λ[r]
        end
        acc
    end
end

# Solid node: half-way bounce-back swap of the populations, τ unchanged (the
# swap is its own transpose).
@inline _gnf_swap(a::NTuple{10,T}) where {T} =
    (a[1], a[4], a[5], a[2], a[3], a[8], a[9], a[6], a[7], a[10])

@kernel function gnf_jvp_2d_kernel!(y, @Const(x), @Const(B), @Const(solid),
                                    @Const(pin), @Const(pout), Nx, Ny)
    i, j = @index(Global, NTuple)
    a = _gnf_gather_arrivals(x, pin, eltype(B), i, j, Nx, Ny)
    issolid = @inbounds solid[i, j]
    out = issolid ? _gnf_swap(a) : _gnf_block_mul(B, a, i, j)
    for r in 1:10
        k = _spxwy_lin(i, j, r, Nx, Ny)
        @inbounds y[k] = pout[k] * out[r]
    end
end

@kernel function gnf_vjp_2d_kernel!(y, @Const(x), @Const(B), @Const(solid),
                                    @Const(pin), @Const(pout), Nx, Ny)
    i, j = @index(Global, NTuple)
    λ = _gnf_gather_local(x, pin, eltype(B), i, j, Nx, Ny)
    issolid = @inbounds solid[i, j]
    z = issolid ? _gnf_swap(λ) : _gnf_block_mul_t(B, λ, i, j)
    # transposed stream: arrival slot (i, j, s) goes back to its source component
    for s in 1:9
        k = _spxwy_src(i, j, s, Nx, Ny)
        @inbounds y[k] = pout[k] * z[s]
    end
    k = _spxwy_lin(i, j, 10, Nx, Ny)
    @inbounds y[k] = pout[k] * z[10]
end

"""
    gnf_jvp!(y, op, x; project=true) -> y

Tangent product y = P A P x of the linearised step (A = ∂G/∂w). `x`, `y`: flat
vectors of length 10 Nx Ny on the device of `op`, distinct. `project=false`
gives the unprojected A x.
"""
function gnf_jvp!(y, op::GNFStepOperator, x; project::Bool=true)
    m = project ? op.active : op.ones
    backend = KernelAbstractions.get_backend(op.B)
    gnf_jvp_2d_kernel!(backend)(y, x, op.B, op.solid, m, m, op.Nx, op.Ny; ndrange=(op.Nx, op.Ny))
    return y
end

"""
    gnf_vjp!(y, op, x; project=true) -> y

Adjoint product y = P Aᵀ P x (transposed blocks, then the transposed stream).
`project=false` gives the unprojected Aᵀ x.
"""
function gnf_vjp!(y, op::GNFStepOperator, x; project::Bool=true)
    m = project ? op.active : op.ones
    backend = KernelAbstractions.get_backend(op.B)
    gnf_vjp_2d_kernel!(backend)(y, x, op.B, op.solid, m, m, op.Nx, op.Ny; ndrange=(op.Nx, op.Ny))
    return y
end

"""
    GNFBorderedAdjoint(op)

Callable `A(y, x)` applying the gauge-bordered adjoint operator to vectors of
length n + 1: y[1:n] = x − P Aᵀ P x + ĉ x[n+1] (identity on inactive
components), y[n+1] = ĉᵀ x[1:n].
"""
struct GNFBorderedAdjoint{TO<:GNFStepOperator, TV<:AbstractVector{Float64}}
    op::TO
    tmp::TV
end
GNFBorderedAdjoint(op::GNFStepOperator) = GNFBorderedAdjoint(op, similar(op.chat))

function (A::GNFBorderedAdjoint)(y, x)
    n = A.op.n
    xn = view(x, 1:n); yn = view(y, 1:n); xs = view(x, n+1:n+1)
    gnf_vjp!(A.tmp, A.op, xn)
    yn .= xn .- A.tmp .+ A.op.chat .* xs
    fill!(view(y, n+1:n+1), dot(A.op.chat, xn))
    return y
end
