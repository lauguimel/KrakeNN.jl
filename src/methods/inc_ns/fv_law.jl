"""
    FVLawViscosity(law::AbstractScalarLaw, theta, fields=(); eps=1e-3)

Scalar constitutive law in the existing steady FV face-viscosity hook. Inputs
are `(gamma, fields...)`, where `gamma = sqrt(2*(S:S + eps*eps/2))` uses the
same invariant and floating-point operation order as [`FVGNFViscosity`](@ref).
The law must be isbits; `theta` is a flat parameter vector and `fields` is a
fixed tuple of prescribed cell matrices on the same backend. They are never
state unknowns. Pass this model to `fv_steady_problem(...; viscosity=model)`.

Interior and periodic faces take each field's arithmetic mean at P and N.
Domain walls, inlets, immersed nominal faces and reconstructed physical-cut
surfaces take the fluid-cell P value. Prescribed fields are not extrapolated:
the existing extrapolation of given viscosity in the Newtonian path is
unchanged. The velocity-gradient reconstruction is exactly the GNF one.

`eps` is a model regularisation, fixed during parameter differentiation.
It must be positive except for a regular-at-rest `ClassicalLaw` (currently
Carreau--Yasuda with `a >= 2`); that adapter uses the existing invariant form
at rest and retains the GNF rounding everywhere. Non-smooth classical clamps
are preserved. [`FVLawParameters`](@ref) differentiates `theta` through the
existing parameter-dual residual chunks, without a separate face pullback.
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
        throw(ArgumentError("law requires $(law_ninputs(law)-1) prescribed fields, got $(length(fields))"))
    all(f -> f isa AbstractMatrix, fields) ||
        throw(ArgumentError("prescribed fields must be cell matrices"))
    all(f -> typeof(KernelAbstractions.get_backend(f)) ===
             typeof(KernelAbstractions.get_backend(theta)), fields) ||
        throw(ArgumentError("theta and prescribed fields must use the same backend"))
    isempty(fields) || all(f -> size(f) == size(first(fields)), fields) ||
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
    size(nu) == (g.Nx, g.Ny) && all(f -> size(f) == size(nu), v.fields) ||
        throw(DimensionMismatch("prescribed fields must match the $(g.Nx) × $(g.Ny) grid"))
    backend = typeof(KernelAbstractions.get_backend(nu))
    typeof(KernelAbstractions.get_backend(v.theta)) === backend &&
        all(f -> typeof(KernelAbstractions.get_backend(f)) === backend, v.fields) ||
        throw(ArgumentError("law parameters and fields must use the problem backend"))
    return nothing
end

_fv_with_fields(v::FVLawViscosity, fields::Tuple) =
    FVLawViscosity(v.law, v.theta, fields; eps=v.eps)
_fv_adapt_viscosity(to, v::FVLawViscosity) =
    FVLawViscosity(v.law, to(v.theta), map(to, v.fields), v.eps)

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
@inline _fv_law_number_type(::Type{T}) where {T} =
    promote_type(Bool, ntuple(i -> _fv_law_number_type(fieldtype(T, i)), Val(fieldcount(T)))...)
@inline function _fv_visc_eltype(::FVLawViscosity{L,A,F,T}) where {L,A,F,T}
    FT = ntuple(i -> eltype(fieldtype(F, i)), Val(fieldcount(F)))
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
problem. The immutable law and prescribed fields stay fixed. The adjoint
uploads parameter duals into a preallocated vector on the solver backend,
then uses the existing residual chunks and global adjoint contraction.
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
