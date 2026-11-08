using KernelAbstractions

# =====================================================================
# Kernel DSL — runtime builder.
#
# `build_lbm_kernel(backend, spec)` compiles an `LBMSpec` into a
# KernelAbstractions `@kernel`. The body is the ordered concatenation
# of `emit_code(brick)` blocks, wrapped in `@inbounds begin … end`,
# with `i, j = @index(Global, NTuple)` and `T = eltype(f_out)` set up.
#
# Kernel signature = canonical sort of the union of `required_args`
# over all bricks. Unused arrays are not in the signature → no GPU
# register pressure from dummy args.
#
# Cache key = (Stencil, Bricks::Type, typeof(backend)). Every unique
# spec × backend pair compiles once; subsequent calls return the
# cached kernel function.
#
# Generated kernel names are `gensym`'d into the `KrakeNN` module
# (i.e. `@__MODULE__` when this file is included from KrakeNN.jl). They
# pollute the module namespace but are unreachable by name from user
# code. Acceptable cost for robust interop with `@kernel`.
# =====================================================================

const LBM_KERNEL_CACHE = Dict{Any, Any}()

function _instantiate_bricks(::Type{Bricks}) where {Bricks <: Tuple}
    return [T() for T in Bricks.parameters]
end

function _collect_args(bricks::Vector)
    seen = Symbol[]
    for b in bricks
        for a in required_args(b)
            a in seen || push!(seen, a)
        end
    end
    return _canonical_sort(seen)
end

"""
    build_lbm_kernel(backend, spec::LBMSpec) -> kernel function

Compile `spec` into a fused `@kernel` on `backend`. Cached.

The returned object is callable as
`kernel!(args...; ndrange=(Nx, Ny))` where `args` must match the
canonical signature order (see `CANONICAL_ARG_ORDER` in
`lbm_spec.jl`) restricted to the union of `required_args` of the
spec's bricks.
"""
function build_lbm_kernel(backend, ::LBMSpec{S, Bricks}) where {S, Bricks}
    key = (S, Bricks, typeof(backend))
    haskey(LBM_KERNEL_CACHE, key) && return LBM_KERNEL_CACHE[key]

    bricks = _instantiate_bricks(Bricks)
    args = _collect_args(bricks)

    # Partition by phase, preserving spec order within each bucket.
    pre_bricks    = filter(b -> phase(b) === :pre_solid, bricks)
    solid_bricks  = filter(b -> phase(b) === :solid,     bricks)
    fluid_bricks  = filter(b -> phase(b) === :fluid,     bricks)

    pre_body   = Expr(:block, Expr[emit_code(b) for b in pre_bricks]...)
    solid_body = Expr(:block, Expr[emit_code(b) for b in solid_bricks]...)
    fluid_body = Expr(:block, Expr[emit_code(b) for b in fluid_bricks]...)

    # Body assembly. If the spec has no :solid brick, we skip the
    # is_solid branching entirely (useful for pure-fluid kernels such
    # as the pull-only MVP).
    # Stencil → index dimensionality. :D2Q9 uses (i, j); :D3Q19 uses (i, j, k).
    is_3d    = S === :D3Q19
    is_solid_ref = is_3d ? :(is_solid[i, j, k]) : :(is_solid[i, j])
    idx_decl = is_3d ? :((i, j, k) = @index(Global, NTuple)) :
                       :((i, j)    = @index(Global, NTuple))
    inner_body = if isempty(solid_bricks)
        Expr(:block, pre_body, fluid_body)
    else
        Expr(:block, pre_body,
             Expr(:if,
                  is_solid_ref,
                  solid_body,
                  fluid_body))
    end

    kname = gensym(:lbm_gen_kernel)
    # Define the kernel AND return its constructor in a single eval.
    # Julia 1.12 enforces strict world-age for module bindings created
    # by eval; accessing them via getfield from the caller's world
    # triggers a warning and can segfault. Returning the constructor
    # directly from eval avoids the cross-world getfield entirely.
    src = quote
        @kernel function $(kname)($(args...))
            $idx_decl
            T = eltype(f_out)
            @inbounds begin
                $inner_body
            end
        end
        $(kname)
    end
    ctor = Core.eval(@__MODULE__, src)
    raw_kernel! = Base.invokelatest(ctor, backend)
    wrapped = (args...; kwargs...) -> Base.invokelatest(raw_kernel!, args...; kwargs...)
    LBM_KERNEL_CACHE[key] = wrapped
    return wrapped
end

"""
    spec_args(spec::LBMSpec) -> Vector{Symbol}

Return the canonical argument-name vector for a given spec. Useful
for call-site discipline: calls to the compiled kernel must pass
arguments in this exact order.
"""
function spec_args(::LBMSpec{S, Bricks}) where {S, Bricks}
    return _collect_args(_instantiate_bricks(Bricks))
end
