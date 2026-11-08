"""
    FVStateTemperature()

Typed scalar-law input selecting the first scalar unknown (temperature).
Pass it in the `fields` tuple of [`FVLawViscosity`](@ref); it requires an
energy-active problem. Face inputs use the mean of the two fluid temperatures,
and wall/heating inputs use the fluid-cell temperature.
"""
struct FVStateTemperature end

_fv_law_input_backend(f::AbstractMatrix, theta) =
    typeof(KernelAbstractions.get_backend(f)) === typeof(KernelAbstractions.get_backend(theta))
_fv_law_input_backend(::FVStateTemperature, theta) = true
_fv_law_input_shape(f::AbstractMatrix, shape) = size(f) == shape
_fv_law_input_shape(::FVStateTemperature, shape) = true
_fv_adapt_law_input(to, f::AbstractMatrix) = to(f)
_fv_adapt_law_input(to, f::FVStateTemperature) = f
_fv_law_input_eltype(::Type{FVStateTemperature}) = Bool
_fv_law_input_eltype(::Type{F}) where {F<:AbstractMatrix} = eltype(F)

"""
    FVLawViscosity(law::AbstractScalarLaw, theta, fields=(); eps=1e-3)

Scalar constitutive law in the existing steady FV face-viscosity hook. Inputs
are `(gamma, fields...)`, where `gamma = sqrt(2*(S:S + eps*eps/2))` uses the
same invariant and floating-point operation order as [`FVGNFViscosity`](@ref).
The law must be isbits; `theta` is a flat parameter vector and `fields` is a
fixed tuple of prescribed cell matrices on the same backend or typed
[`FVStateTemperature`](@ref) selectors. Prescribed matrices remain fixed;
temperature selectors read the current state, including its dual derivatives.
Pass this model to `fv_steady_problem(...; viscosity=model)`.

Interior and periodic faces take each field's arithmetic mean at P and N.
Domain walls, inlets, immersed nominal faces and reconstructed physical-cut
surfaces take the fluid-cell P value. Prescribed fields are not extrapolated:
the existing extrapolation of given viscosity in the Newtonian path is
unchanged. The velocity-gradient reconstruction is exactly the GNF one.

`eps` is a model regularisation, fixed during parameter differentiation.
It must be positive except for a regular-at-rest `ClassicalLaw` (currently
Carreau--Yasuda with `a >= 2`); that adapter uses the existing invariant form
at rest and retains the GNF rounding everywhere. Non-smooth classical clamps
are preserved. [`FVLawParameters`](@ref) differentiates `theta` by contracting
face-viscosity cotangents with local law pullbacks. The parameter-dual residual
chunks remain available as a reference with `law_gradient = :duals`.
"""
struct FVLawViscosity{L<:AbstractScalarLaw,A,F,T}
    law::L
    theta::A
    fields::F
    eps::T
end

_fv_law_regular_at_rest(::AbstractScalarLaw, theta) = false
_fv_law_regular_at_rest(::ClassicalLaw{<:CarreauYasuda}, theta) =
    _fv_ib_primal(Array(theta)[4]) >= 2
function FVLawViscosity(law::AbstractScalarLaw, theta::AbstractVector,
                        fields::Tuple=(); eps::Real=1e-3)
    isbitstype(typeof(law)) || throw(ArgumentError("FV scalar law must be isbits"))
    length(theta) == law_nparams(law) ||
        throw(DimensionMismatch("theta must have $(law_nparams(law)) parameters"))
    length(fields) == law_ninputs(law)-1 ||
        throw(ArgumentError("law requires $(law_ninputs(law)-1) extra inputs, got $(length(fields))"))
    all(f -> f isa Union{AbstractMatrix,FVStateTemperature}, fields) ||
        throw(ArgumentError("law inputs must be cell matrices or FVStateTemperature selectors"))
    all(f -> _fv_law_input_backend(f, theta), fields) ||
        throw(ArgumentError("theta and prescribed fields must use the same backend"))
    matrices = filter(f -> f isa AbstractMatrix, fields)
    isempty(matrices) || all(f -> size(f) == size(first(matrices)), matrices) ||
        throw(DimensionMismatch("prescribed fields must have the same cell shape"))
    isfinite(eps) && eps >= 0 || throw(ArgumentError("eps must be finite and nonnegative"))
    eps > 0 || _fv_law_regular_at_rest(law, theta) ||
        throw(ArgumentError("this scalar law requires eps > 0 to be differentiable at rest"))
    return FVLawViscosity(law, theta, fields, float(eps))
end

function _fv_validate_viscosity(v::FVLawViscosity, g, nu)
    isbitstype(typeof(v.law)) || throw(ArgumentError("FV scalar law must be isbits"))
    length(v.theta) == law_nparams(v.law) ||
        throw(DimensionMismatch("theta does not match the law parameter count"))
    length(v.fields) == law_ninputs(v.law)-1 ||
        throw(ArgumentError("prescribed field count does not match the law"))
    isfinite(v.eps) && v.eps >= 0 &&
        (v.eps > 0 || _fv_law_regular_at_rest(v.law, v.theta)) ||
        throw(ArgumentError("scalar law must be regular at rest: use eps > 0"))
    any(f -> f isa FVStateTemperature, v.fields) && _fv_nfields(g) < 4 &&
        throw(ArgumentError("FVStateTemperature requires an active energy model"))
    size(nu) == (g.Nx, g.Ny) && all(f -> _fv_law_input_shape(f, size(nu)), v.fields) ||
        throw(DimensionMismatch("prescribed fields must match the $(g.Nx) × $(g.Ny) grid"))
    backend = typeof(KernelAbstractions.get_backend(nu))
    typeof(KernelAbstractions.get_backend(v.theta)) === backend &&
        all(f -> _fv_law_input_backend(f, v.theta), v.fields) ||
        throw(ArgumentError("law parameters and fields must use the problem backend"))
    return nothing
end

_fv_with_fields(v::FVLawViscosity, fields::Tuple) =
    FVLawViscosity(v.law, v.theta, fields; eps=v.eps)
_fv_adapt_viscosity(to, v::FVLawViscosity) =
    FVLawViscosity(v.law, to(v.theta), map(f -> _fv_adapt_law_input(to, f), v.fields), v.eps)

# KernelAbstractions already depends on Adapt. Its device argument conversion
# recursively lowers only the arrays; no validation runs on device-array views.
function KernelAbstractions.Adapt.adapt_structure(to, v::FVLawViscosity)
    adapt = KernelAbstractions.Adapt.adapt
    return FVLawViscosity(v.law, adapt(to, v.theta), adapt(to, v.fields), v.eps)
end

# Include fixed law constants in promotion, e.g. Float64 normalization with
# Float32 parameters. Recursion is over concrete isbits field types only.
@inline _fv_law_number_type(::Type{T}) where {T<:Number} = T
@inline _fv_law_number_type(::Type{<:ClassicalLaw}) = Bool
@inline _fv_law_number_type(::Type{TinyMLP{N,H,L,T}}) where {N,H,L,T} = T
@inline _fv_law_number_type(::Type{LearnedViscosity{M,B,T,G}}) where {M,B,T,G} =
    promote_type(T, _fv_law_number_type(M), _fv_law_number_type(G))
# Explicit: the generic fallback below splats too many field types (e.g. the
# 8-tuple of E5ThermalLaw) for inference, leaving a dynamic call on device.
@inline _fv_law_number_type(::Type{E5ThermalLaw{T}}) where {T} = T
@inline _fv_law_number_type(::Type{NTuple{N,T}}) where {N,T<:Number} = T
@inline _fv_law_number_type(::Type{T}) where {T} =
    promote_type(Bool, ntuple(i -> _fv_law_number_type(fieldtype(T, i)), Val(fieldcount(T)))...)
@inline function _fv_visc_eltype(::FVLawViscosity{L,A,F,T}) where {L,A,F,T}
    FT = ntuple(i -> _fv_law_input_eltype(fieldtype(F, i)), Val(fieldcount(F)))
    return promote_type(T, eltype(A), _fv_law_number_type(L), FT...)
end
@inline _fv_visc_zero(v::FVLawViscosity) = zero(_fv_visc_eltype(v))

@inline _fv_law_invariant(law::AbstractScalarLaw, theta, I1, fields) =
    law_value(law, (gnf_shear_rate(I1), fields...), theta)

# This is the same scalar model as ClassicalLaw, evaluated in I1 so that
# eps=0/rest remains differentiable, as in the existing GNF Carreau adapter.
@inline function _fv_law_invariant(law::ClassicalLaw{<:CarreauYasuda}, theta, I1, fields)
    @inbounds model = CarreauYasuda(theta[1], theta[2], theta[3], theta[4], theta[5];
                                   thermal=law.model.thermal)
    return gnf_viscosity(model, I1)
end
@inline function _fv_gnf_nu(v::FVLawViscosity, ux, uy, vx, vy, fields::Tuple)
    sxy = (uy + vx) / 2
    I1 = ux * ux + vy * vy + 2 * sxy * sxy
    return _fv_law_invariant(v.law, v.theta, I1 + v.eps * v.eps / 2, fields)
end
@inline function _fv_face_nu(v::FVLawViscosity, nuP, nuN, ux, uy, vx, vy, i, j, ii, jj)
    fields = map(v.fields) do f
        @inbounds (f[i, j] + f[ii, jj]) / 2
    end
    return _fv_gnf_nu(v, ux, uy, vx, vy, fields)
end
@inline function _fv_wall_nu(v::FVLawViscosity, nuP, ux, uy, vx, vy, i, j)
    fields = map(f -> @inbounds(f[i, j]), v.fields)
    return _fv_gnf_nu(v, ux, uy, vx, vy, fields)
end

const _FVLawProblem = FVSteadyProblem2D{<:FVGrid2D,<:Any,<:Any,<:Any,<:Any,<:FVLawViscosity}
function fv_jacobian_pattern(p::_FVLawProblem)
    _fv_validate_viscosity(p.visc, p.grid, p.nu)
    return _fv_gnf_jacobian_pattern(p)
end

"""
    FVLawParameters(n) <: FVParameterMap

Map `theta` (length `n = law_nparams(law)`) to an [`FVLawViscosity`](@ref)
problem. The immutable law and prescribed fields stay fixed. The adjoint uses
one residual traversal to collect four oriented face-viscosity cotangents per
cell, then accumulates local [`law_pullback!`](@ref) contributions in bounded
tiles. The state Jacobian already includes the shear-rate dependence on `U`;
this parameter pullback holds `U` fixed. `law_gradient = :duals` selects the
parameter-dual reference using the same workspace.
`fv_set_params!(solver, map, theta; fields=(...))` can also replace prescribed
field values while retaining the Jacobian structure and symbolic analysis.
"""
struct FVLawParameters <: FVParameterMap
    nparams::Int
    function FVLawParameters(n::Integer)
        n >= 0 || throw(ArgumentError("parameter count must be nonnegative"))
        return new(Int(n))
    end
end
fv_nparams(m::FVLawParameters, p) = m.nparams
function _fv_param_buffers(m::FVLawParameters, p::_FVLawProblem, ::Type{T}) where {T}
    m.nparams == law_nparams(p.visc.law) ||
        throw(DimensionMismatch("FVLawParameters count does not match the law"))
    return similar(p.visc.theta, T)
end
function _fv_param_problem!(theta, ::FVLawParameters, p::_FVLawProblem, values)
    copyto!(theta, values)
    v = p.visc
    return _fv_with(p; visc=FVLawViscosity(v.law, theta, v.fields, v.eps))
end

# A residual face evaluates its law once: either the interior/periodic hook or
# _fv_wall_flux_components -> _fv_wall_stress. Physical-cut evaluations belong
# to the objective, not the residual. Slots deliberately retain both oriented
# evaluations of a shared face; each belongs to a different residual row block.
# At fixed U, R_i = r_i(U, nu_i1, ..., nu_i4) gives
# -lambda' * R_theta = -sum_i,d (lambda_i' * dr_i/dnu_id) * dlaw(x_id,theta)/dtheta.
# The returned input cotangent is unused: x_id(U) already belongs to the state
# Jacobian, and differentiating it here would count that dependence twice.
const _FVLawFaceDual = ForwardDiff.Dual{FVParamTag,Float64,4}
struct _FVLawCapture{L,X} <: AbstractScalarLaw
    law::L
    inputs::X
    cell::Int
    face::Int
end
@inline law_ninputs(law::_FVLawCapture) = law_ninputs(law.law)
@inline law_nparams(law::_FVLawCapture) = law_nparams(law.law)
@inline _fv_law_number_type(::Type{<:_FVLawCapture}) = _FVLawFaceDual
@inline function _fv_face_viscosity(v::FVLawViscosity{<:_FVLawCapture}, d)
    c = v.law
    return FVLawViscosity(_FVLawCapture(c.law, c.inputs, c.cell, d),
                          v.theta, v.fields, v.eps)
end
@inline function _fv_law_invariant(c::_FVLawCapture, theta, I1, fields)
    # Preserve the original invariant evaluation, including Carreau at rest.
    nu = _fv_law_invariant(c.law, theta, I1, fields)
    x = (gnf_shear_rate(I1), fields...)
    @inbounds for a in 1:law_ninputs(c)
        c.inputs[c.cell, c.face, a] = x[a]
    end
    return _FVLawFaceDual(Float64(nu),
        ForwardDiff.Partials(ntuple(k -> k == c.face ? 1.0 : 0.0, Val(4))))
end

struct _FVLawCotangentWorkspace{V,X,S,P}
    theta::V
    inputs::X
    sensitivity::S
    partials::P
    gradient::V
    per::Int
end
function _fv_law_workspace(s::FVNewtonSolver, m::FVLawParameters)
    law = s.prob.visc.law
    nc = s.prob.grid.Nx * s.prob.grid.Ny
    per = 64
    theta = similar(s.R, Float64, m.nparams)
    inputs = similar(s.R, Float64, nc, 4, law_ninputs(law))
    sensitivity = similar(s.R, Float64, nc, 4)
    # Tile first makes neighbouring GPU workers update adjacent addresses.
    partials = similar(s.R, Float64, cld(nc, per), m.nparams)
    return _FVLawCotangentWorkspace(theta, inputs, sensitivity, partials,
                                    similar(theta), per)
end

# Construct the capture wrapper inside the kernel from adapted arguments.
# No wrapper containing host arrays crosses the device launch boundary.
@kernel function _fv_law_capture_kernel!(inputs, sensitivity, @Const(U), @Const(lambda),
                                        @Const(tag), @Const(nu), @Const(bvel), @Const(src),
                                        g, visc, qcut, wvel, closure, face_pair)
    i, j = @index(Global, NTuple)
    c = i + (j - 1) * g.Nx
    @inbounds for d in 1:4, a in 1:law_ninputs(visc.law)
        inputs[c, d, a] = 0.0
    end
    capture = _FVLawCapture(visc.law, inputs, c, 0)
    v = FVLawViscosity(capture, visc.theta, visc.fields, visc.eps)
    ru, rv, rc = fv_cell_residual(U, tag, nu, bvel, src, g, i, j, v,
                                  qcut, wvel, closure, face_pair)
    pu, pv, pc = ForwardDiff.partials(ru), ForwardDiff.partials(rv), ForwardDiff.partials(rc)
    b = fv_index(g, i, j, 0)
    @inbounds for d in 1:4
        sensitivity[c, d] = -(lambda[b + _fv_velocity_slot(g.layout, 1)] * pu[d] + lambda[b + _fv_velocity_slot(g.layout, 2)] * pv[d] +
                              lambda[b + _fv_pressure_slot(g.layout)] * pc[d])
    end
end

# Allocation-free AbstractVector interface to one exclusive tile accumulator.
struct _FVLawPartialColumn{T,A} <: AbstractVector{T}
    values::A
    tile::Int
end
Base.size(g::_FVLawPartialColumn) = (size(g.values, 2),)
Base.IndexStyle(::Type{<:_FVLawPartialColumn}) = IndexLinear()
@inline Base.getindex(g::_FVLawPartialColumn, p::Int) = @inbounds g.values[g.tile, p]
@inline Base.setindex!(g::_FVLawPartialColumn, v, p::Int) = (@inbounds g.values[g.tile, p] = v)

@kernel function _fv_law_pullback_kernel!(partials, @Const(inputs), @Const(sensitivity),
                                         law, @Const(theta), nc, per, ::Val{NI}) where {NI}
    tile = @index(Global)
    g = _FVLawPartialColumn{eltype(partials),typeof(partials)}(partials, tile)
    @inbounds for p in 1:law_nparams(law)
        g[p] = 0.0
    end
    @inbounds for c in ((tile - 1) * per + 1):min(tile * per, nc), d in 1:4
        s = sensitivity[c, d]
        if !iszero(s)
            x = ntuple(a -> inputs[c, d, a], Val(NI))
            law_pullback!(g, law, x, theta, s)
        end
    end
end

@kernel function _fv_law_reduce_kernel!(gradient, @Const(partials))
    p = @index(Global)
    acc = 0.0
    @inbounds for tile in axes(partials, 1)
        acc += partials[tile, p]
    end
    @inbounds gradient[p] = acc
end

function _fv_param_cotangent!(ws::FVAdjointWorkspace, m::FVLawParameters,
                              s::FVNewtonSolver, U, theta; law_gradient = :contraction)
    _fv_check_law_gradient(law_gradient)
    law_gradient === :duals && return _fv_param_cotangent_duals!(ws, m, s, U, theta)
    theta === nothing && throw(ArgumentError("theta is required for FVLawParameters"))
    prob = s.prob
    _fv_check_nparams(m, prob, theta)
    w = ws.law
    copyto!(w.theta, theta)
    # Read the current prescribed fields, including replacements by fv_set_params!.
    v = FVLawViscosity(prob.visc.law, w.theta, prob.visc.fields, prob.visc.eps)
    qcut, wvel, closure = _fv_immersed_args(prob.immersed)
    g = prob.grid
    backend = KernelAbstractions.get_backend(U)
    _fv_law_capture_kernel!(backend, 64)(w.inputs, w.sensitivity, U, ws.λ,
        prob.tag, prob.nu, prob.bvel, prob.src, g, v, qcut, wvel, closure,
        _fv_face_pair(prob.immersed); ndrange = (g.Nx, g.Ny))
    _fv_law_pullback_kernel!(backend, 64)(w.partials, w.inputs, w.sensitivity,
        v.law, w.theta, g.Nx * g.Ny, w.per, Val(law_ninputs(v.law));
        ndrange = size(w.partials, 1))
    if m.nparams > 0
        _fv_law_reduce_kernel!(backend, 64)(w.gradient, w.partials; ndrange = m.nparams)
    end
    KernelAbstractions.synchronize(backend)
    # Preserve the established owning host-vector return contract across reuse.
    return w.gradient isa Array ? copy(w.gradient) : Array(w.gradient)
end
