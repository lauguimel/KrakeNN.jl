# Generalised-Newtonian (GNF) viscosity in the steady colocated FV residual
# and its Newton solver. Roadmap "differentiable FV + cuDSS", capability C3.
#
# The viscosity of a face is the law of src/rheology (the SAME law types and
# the same `gnf_viscosity(law, I1)` evaluation as the LBM GNF drivers and the
# GNF adjoint) at the regularised face strain rate
#     γ̇_ε = √(2 S:S + ε²),   S = (∇u + ∇uᵀ)/2   (full 2D tensor),
# evaluated inside the C1 residual kernel through the hooks _fv_face_nu /
# _fv_wall_nu (fv_steady_residual.jl). The face gradient is the one the viscous
# flux already uses: compact normal difference, mean of the two Green-Gauss
# cell gradients tangentially; a wall face takes the one-sided normal
# difference and the cell gradient. Because the residual kernel evaluates ν
# itself, the coloured dual assembly of C2 differentiates through ν(U): the
# Jacobian includes ∂ν/∂U with no extra code. Only the sparsity pattern
# changes, and it is derived here from the exact stencil of the face strain
# rate (3 × 3 cell block for u and v; no radius-2 diamond).

"""
    FVGNFViscosity(law, eps)

Generalised-Newtonian viscosity model of an [`FVSteadyProblem2D`](@ref): the
kinematic viscosity of a face is `gnf_viscosity(law, I1 + ε²/2)`, i.e.
`effective_viscosity(law, γ̇_ε)` with `γ̇_ε = √(2 S:S + ε²)` from the face
velocity gradient (full strain-rate tensor). `law` is one of the
`GeneralizedNewtonian` laws of `src/rheology/models.jl`, parameters in
kinematic FV units (the same law objects as the LBM GNF drivers; their
`thermal` coupling is not used here).

The regularisation `ε > 0` is a change of model, not a numerical device: it
removes the infinite viscosity (shear-thinning) or the infinite derivative of
`γ̇ = √(2 S:S)` at zero shear, so that the residual is differentiable on
states with `S = 0` (rest, centrelines). The power-law with `ε > 0` is the
regularised power law `K (γ̇² + ε²)^((n-1)/2)`. `ε = 0` is accepted only for
laws regular at `S = 0` (`Newtonian`; `CarreauYasuda` with `a ≥ 2`, whose
`gnf_viscosity` method is written in `I1`).

Non-smooth parts of a law are kept as they are: `PowerLaw`'s clamp to
`[nu_min, nu_max]` (bounds in FV units) has a zero derivative where it is
active, which stalls Newton; pass `nu_min = 0, nu_max = Inf` unless the clamp
is wanted.
"""
struct FVGNFViscosity{L<:GeneralizedNewtonian,T}
    law::L
    eps::T
    function FVGNFViscosity(law::L, eps::T) where {L<:GeneralizedNewtonian,T<:Real}
        eps >= 0 || throw(ArgumentError("gnf_eps must be ≥ 0, got $eps"))
        if iszero(eps) && !_fv_gnf_regular_at_rest(law)
            throw(ArgumentError("$(nameof(L)) has an infinite derivative at zero shear: " *
                                "set the regularisation gnf_eps > 0"))
        end
        return new{L,T}(law, eps)
    end
end

_fv_gnf_regular_at_rest(::GeneralizedNewtonian) = false
_fv_gnf_regular_at_rest(::Newtonian) = true
_fv_gnf_regular_at_rest(m::CarreauYasuda) = m.a >= 2

@inline function _fv_gnf_nu(v::FVGNFViscosity, ux, uy, vx, vy)
    sxy = (uy + vx) / 2
    I1 = ux * ux + vy * vy + 2 * sxy * sxy            # S:S
    return gnf_viscosity(v.law, I1 + v.eps * v.eps / 2)
end
@inline _fv_face_nu(v::FVGNFViscosity, nuP, nuN, ux, uy, vx, vy) = _fv_gnf_nu(v, ux, uy, vx, vy)
@inline _fv_wall_nu(v::FVGNFViscosity, nuP, ux, uy, vx, vy) = _fv_gnf_nu(v, ux, uy, vx, vy)

_fv_law_params(m::GeneralizedNewtonian) =
    Tuple(getfield(m, f) for f in fieldnames(typeof(m)) if f !== :thermal)
_fv_visc_eltype(v::FVGNFViscosity) = promote_type(typeof(v.eps), map(typeof, _fv_law_params(v.law))...)
@inline _fv_visc_zero(v::FVGNFViscosity) = zero(_fv_visc_eltype(v))

# a steady FV problem whose viscosity model is GNF
const _FVGNFProblem = FVSteadyProblem2D{<:FVGrid2D,<:Any,<:Any,<:Any,<:Any,<:FVGNFViscosity}

# ------------------------------------------------------------------ parameters (C4)
"""
    fv_gnf_parameter_names(prob) -> Tuple{Vararg{Symbol}}

Names of the law parameters of a GNF problem, in the order of
[`fv_gnf_parameters`](@ref) (the fields of the law, `thermal` excluded; e.g.
`(:K, :n, :nu_min, :nu_max)` for `PowerLaw`,
`(:eta_0, :eta_inf, :lambda, :a, :n)` for `CarreauYasuda`).
"""
fv_gnf_parameter_names(p::_FVGNFProblem) =
    Tuple(f for f in fieldnames(typeof(p.visc.law)) if f !== :thermal)

"""
    fv_gnf_parameters(prob) -> Vector

Law parameters of a GNF problem as a vector θ (order of
[`fv_gnf_parameter_names`](@ref)).
"""
fv_gnf_parameters(p::_FVGNFProblem) =
    collect(promote(_fv_law_params(p.visc.law)...))

"""
    fv_with_gnf_parameters(prob, θ; eps = prob.visc.eps) -> FVSteadyProblem2D

The same problem (arrays shared, not copied) with the law rebuilt from the
parameter vector θ, of any number type: a θ of `ForwardDiff.Dual` gives
`∂R/∂θ` through the same residual kernel (C4, directional duals), on the CPU
or GPU backend (the law is an isbits kernel argument).
"""
function fv_with_gnf_parameters(p::_FVGNFProblem,
                                θ::AbstractVector; eps = p.visc.eps)
    law = p.visc.law
    np = length(_fv_law_params(law))
    length(θ) == np || throw(DimensionMismatch("$(nameof(typeof(law))) has $np parameters, got $(length(θ))"))
    T = promote_type(eltype(θ), typeof(eps))
    newlaw = typeof(law).name.wrapper(ntuple(k -> T(θ[k]), np)..., law.thermal)
    return FVSteadyProblem2D(p.grid, p.tag, p.nu, p.bvel, p.src, FVGNFViscosity(newlaw, T(eps)))
end

# ------------------------------------------------------------------ pattern
"""
    fv_jacobian_pattern(prob::FVSteadyProblem2D with FVGNFViscosity) -> SparseMatrixCSC{Bool,Int}

Pattern of `∂R/∂U` when the viscosity depends on the state: the Newtonian
pattern ([`fv_jacobian_pattern`](@ref)) plus the velocity unknowns that feed a
face viscosity, taken from the exact stencil of the face strain rate. The
viscosity of the face between `P` and its neighbour `N` across an x-face reads
`u, v` at `P`, `N` and their north/south neighbours (tangential Green-Gauss
gradients); across a y-face, at `P`, `N` and their east/west neighbours. A
wall face reads `P` and its tangential neighbours; an outlet face has no
viscous flux. The momentum rows of `P` read the viscosity of all its
non-outlet faces (viscous flux, and the Rhie-Chow coefficient of the
convective flux); the continuity row the viscosity of its interior faces
(Rhie-Chow coefficient); pinned and solid rows are unchanged. Each neighbour
is reached by the same face classification as the residual (walls, periodic
wrap, solid cells), so the pattern is the 3 × 3 cell block for `u` and `v` in
the interior (27 entries per interior row, 26.9 per row on a large grid),
narrower than the radius-2 diamond of a cell-centred viscosity.
"""
function fv_jacobian_pattern(prob::_FVGNFProblem)
    S0 = invoke(fv_jacobian_pattern, Tuple{FVSteadyProblem2D}, prob)
    g = prob.grid
    tag = Array(prob.tag)
    n = fv_nunknowns(g)
    I = Int[]; J = Int[]
    sizehint!(I, 12 * n); sizehint!(J, 12 * n)
    function nb(i, j, d)
        i == 0 && return (0, 0)
        kind, ii, jj = _fv_face_kind(g, tag, i, j, d)
        return kind == _FV_F_INT ? (ii, jj) : (0, 0)
    end
    tang(c, d) = d <= 2 ? (nb(c..., 3), nb(c..., 4)) : (nb(c..., 1), nb(c..., 2))
    function pushuv(r, c)
        c[1] == 0 && return nothing
        push!(I, r, r); push!(J, fv_index(g, c[1], c[2], 1), fv_index(g, c[1], c[2], 2))
        return nothing
    end
    for j in 1:g.Ny, i in 1:g.Nx
        tag[i, j] == FV_CELL_SOLID && continue
        P = (i, j)
        ru = fv_index(g, i, j, 1); rv = ru + 1; rc = ru + 2
        pinned = tag[i, j] == FV_CELL_PINNED
        for d in 1:4
            kind, ii, jj = _fv_face_kind(g, tag, i, j, d)
            kind == _FV_F_OUTLET && continue
            cells = kind == _FV_F_INT ?
                (P, (ii, jj), tang(P, d)..., tang((ii, jj), d)...) :
                (P, tang(P, d)...)
            for c in cells
                pushuv(ru, c); pushuv(rv, c)
                (kind == _FV_F_INT && !pinned) && pushuv(rc, c)
            end
        end
    end
    S1 = sparse(I, J, trues(length(I)), n, n, |)
    return S0 .| S1
end
