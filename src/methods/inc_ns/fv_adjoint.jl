# Steady discrete adjoint of the colocated FV Newton solver (fv_newton.jl).
# Roadmap "differentiable FV + cuDSS", capability C4.
#
# The adjoint differentiates the ASSEMBLY, not the solver. At a converged state
# U(θ) of R(U, θ) = 0 and for an objective 𝒥(U, θ):
#     Jᵀ λ = ∂𝒥/∂U                      (J = ∂R/∂U, exact, unshifted)
#     d𝒥/dθ = ∂𝒥/∂θ − λᵀ ∂R/∂θ
# - J and its factors are those of the Newton solver: `fv_newton_finalize!`
#   re-assembles J at U without continuation shift and REFACTORIZES it
#   (numeric only, the symbolic analysis of the first Newton step is kept); the
#   transposed solve goes through `lin_solve!(...; transpose = true)` (UMFPACK
#   on the CPU, cuDSS `solve_mode = 'T'` with CUDSS.jl >= 1.0 on the GPU).
# - λᵀ ∂R/∂θ is computed by evaluating the C1 residual kernel itself with
#   ForwardDiff duals in the PARAMETER arrays of the problem (ν field, boundary
#   velocities, sources): chunks of `N` parameters per kernel launch for a few
#   scalar parameters, and a column colouring of ∂R/∂ν for a per-cell
#   viscosity field (one launch per `N` colours, ~5 colours). No CPU mirror,
#   no hand-written derivative: the same kernel gives the solution and the
#   gradient (roadmap V5).
# - 𝒥 is evaluated on the host (objectives are small reductions); its
#   gradients ∂𝒥/∂U and ∂𝒥/∂θ come from ForwardDiff restricted to the unknowns
#   (and cells) the objective reads. Wall forces differentiate each local face
#   stencil once, so their cost does not multiply grid size by wall support.

# ------------------------------------------------------------------ parameters
"""
    FVParamTag

ForwardDiff tag of the parameter duals of the FV adjoint (distinct from
[`FVJacobianTag`](@ref), the tag of the state duals of the Newton assembly).
"""
struct FVParamTag end

"""
    FVParameterMap

How a parameter vector `θ` enters the steady FV problem. A map turns a base
[`FVSteadyProblem2D`](@ref) and `θ` (Float64 or dual numbers) into the problem
at `θ` with [`fv_param_problem`](@ref); the residual kernel then carries the
duals. Built-in maps:

- [`FVUniformViscosity`](@ref)`()`: `θ = [ν]`, uniform cell viscosity;
- [`FVViscosityField`](@ref)`()`: `θ = vec(ν)`, one viscosity per cell
  (coloured assembly of `λᵀ ∂R/∂ν`);
- [`FVSideVelocity`](@ref)`(prob, side)`: `θ = [U]`, the velocity of one
  wall/inlet side is `U × shape`;
- [`FVUniformSource`](@ref)`(comp)`: `θ = [s]`, uniform volumetric source of
  equation `comp` (1 = x momentum, 2 = y momentum, 3 = mass);
- [`FVProblemMap`](@ref)`(f, nparams)`: any `f(prob, θ) -> FVSteadyProblem2D`
  generic in the element type of `θ` (the plug-in point for law parameters).

The built-in maps preserve `prob.visc`. For a GNF problem, the cell field
`nu` is ignored, so viscosity-field maps have zero derivatives; use
`FVProblemMap(fv_with_gnf_parameters, length(fv_gnf_parameters(prob)))` to
differentiate the constitutive-law parameters instead.
"""
abstract type FVParameterMap end

"""
    FVUniformViscosity() <: FVParameterMap

`θ = [ν]`: every cell has viscosity `ν`.
"""
struct FVUniformViscosity <: FVParameterMap end

"""
    FVViscosityField() <: FVParameterMap

`θ = vec(ν)` (length `Nx Ny`, column-major cell order `c = i + (j - 1) Nx`):
one kinematic viscosity per cell. The adjoint returns the per-cell cotangent
`s = −λᵀ ∂R/∂ν` as an `Nx × Ny` array (zero at solid cells), the FV analogue of
the `s` of [`gnf_steady_adjoint`](@ref). The cotangent is assembled with a
column colouring of `∂R/∂ν` (a cell's viscosity enters the rows of the cell and
of its interior face neighbours).
"""
struct FVViscosityField <: FVParameterMap end

"""
    FVSideVelocity(prob, side::Symbol; shape = nothing) <: FVParameterMap

`θ = [U]`: the prescribed velocity of the boundary faces of `side`
(`:E, :W, :N, :S`, a wall or inlet side) is `U × shape(x, y)`, all other sides
keep the velocities of the problem the map is applied to. `shape` is a
`(x, y) -> (u, v)` function at the face centres; default: unit tangential
velocity (`(1, 0)` on `:N`/`:S`, `(0, 1)` on `:E`/`:W`), i.e. a lid speed.
"""
struct FVSideVelocity <: FVParameterMap
    side::Int
    keep::Array{Float64,3}     # 0 on the side, 1 elsewhere (2 × L × 4)
    shape::Array{Float64,3}    # shape on the side, 0 elsewhere
end

function FVSideVelocity(prob::FVSteadyProblem2D, side::Symbol; shape = nothing)
    d = findfirst(==(side), _FV_SIDES)
    d === nothing && throw(ArgumentError("side must be one of $(_FV_SIDES), got :$side"))
    g = prob.grid
    L = max(g.Nx, g.Ny)
    keep = ones(Float64, 2, L, 4); keep[:, :, d] .= 0
    shp = zeros(Float64, 2, L, 4)
    h = Float64(g.h)
    f = shape === nothing ? (d <= 2 ? ((x, y) -> (0.0, 1.0)) : ((x, y) -> (1.0, 0.0))) : shape
    nl = d <= 2 ? g.Ny : g.Nx
    for l in 1:nl
        x = d == 1 ? g.Nx * h : d == 2 ? 0.0 : (l - 0.5) * h
        y = d == 3 ? g.Ny * h : d == 4 ? 0.0 : (l - 0.5) * h
        uv = f(x, y)
        shp[1, l, d] = uv[1]; shp[2, l, d] = uv[2]
    end
    return FVSideVelocity(d, keep, shp)
end

"""
    FVUniformSource(comp::Int) <: FVParameterMap

`θ = [s]`: the volumetric source of equation `comp` (1 = x momentum, i.e. a
body force per unit mass; 2 = y momentum; 3 = mass) is `s` in every cell.
"""
struct FVUniformSource <: FVParameterMap
    comp::Int
    function FVUniformSource(comp::Integer)
        1 <= comp <= 3 || throw(ArgumentError("comp must be 1, 2 or 3"))
        return new(Int(comp))
    end
end

"""
    FVProblemMap(f, nparams) <: FVParameterMap

Generic map: `f(prob, θ) -> FVSteadyProblem2D` returns the problem at `θ`
(length `nparams`) with arrays on the backend of `prob`. `f` must be generic in
the element type of `θ` (it is called with `ForwardDiff.Dual` entries) and
must not depend on the state. Allocates its arrays at every call (one call per
chunk of parameters per gradient).
"""
struct FVProblemMap{F} <: FVParameterMap
    f::F
    nparams::Int
end

"""
    fv_nparams(map, prob) -> Int

Length of the parameter vector `θ` of `map` for the problem `prob`.
"""
fv_nparams(::FVUniformViscosity, prob) = 1
fv_nparams(::FVViscosityField, prob) = prob.grid.Nx * prob.grid.Ny
fv_nparams(::FVSideVelocity, prob) = 1
fv_nparams(::FVUniformSource, prob) = 1
fv_nparams(m::FVProblemMap, prob) = m.nparams

# preallocated parameter arrays of element type T on the backend of prob, plus
# map constants moved to that backend
struct _FVParamBuffers{NU,BV,SR,AX}
    nu::NU
    bvel::BV
    src::SR
    aux::AX
end

_fv_map_aux(::FVParameterMap, prob) = nothing
function _fv_map_aux(m::FVSideVelocity, prob)
    size(m.keep) == size(prob.bvel) ||
        throw(DimensionMismatch("FVSideVelocity built for a different grid"))
    keep = similar(prob.bvel, Float64); copyto!(keep, m.keep)
    shp = similar(prob.bvel, Float64); copyto!(shp, m.shape)
    return (keep, shp)
end

_fv_param_buffers(m::FVParameterMap, prob, ::Type{T}) where {T} =
    _FVParamBuffers(similar(prob.nu, T), similar(prob.bvel, T), similar(prob.src, T),
                    _fv_map_aux(m, prob))
_fv_param_buffers(::Union{FVUniformViscosity,FVViscosityField}, prob, ::Type{T}) where {T} =
    _FVParamBuffers(similar(prob.nu, T), nothing, nothing, nothing)
_fv_param_buffers(m::FVSideVelocity, prob, ::Type{T}) where {T} =
    _FVParamBuffers(nothing, similar(prob.bvel, T), nothing, _fv_map_aux(m, prob))
_fv_param_buffers(::FVUniformSource, prob, ::Type{T}) where {T} =
    _FVParamBuffers(nothing, nothing, similar(prob.src, T), nothing)
_fv_param_buffers(::FVProblemMap, prob, ::Type{T}) where {T} =
    _FVParamBuffers(nothing, nothing, nothing, nothing)

_fv_with(p::FVSteadyProblem2D; nu = p.nu, bvel = p.bvel, src = p.src, visc = p.visc,
          immersed = p.immersed, energy = p.energy) =
    FVSteadyProblem2D(p.grid, p.tag, nu, bvel, src, visc, immersed, energy)

function _fv_check_nparams(m, prob, θ)
    length(θ) == fv_nparams(m, prob) ||
        throw(DimensionMismatch("θ has length $(length(θ)), the map expects $(fv_nparams(m, prob))"))
end

# in-place problem at θ in the buffers B (no device allocation for the
# built-in maps)
function _fv_param_problem!(B, ::FVUniformViscosity, prob, θ)
    fill!(B.nu, θ[1])
    return _fv_with(prob; nu = B.nu)
end
function _fv_param_problem!(B, ::FVViscosityField, prob, θ)
    copyto!(B.nu, reshape(θ, size(B.nu)))
    return _fv_with(prob; nu = B.nu)
end
function _fv_param_problem!(B, ::FVSideVelocity, prob, θ)
    keep, shp = B.aux
    t = θ[1]
    B.bvel .= prob.bvel .* keep .+ t .* shp
    return _fv_with(prob; bvel = B.bvel)
end
function _fv_param_problem!(B, m::FVUniformSource, prob, θ)
    copyto!(B.src, prob.src)
    view(B.src, m.comp:_fv_nfields(prob.grid):length(B.src)) .= θ[1]
    return _fv_with(prob; src = B.src)
end
function _fv_param_problem!(B, m::FVProblemMap, prob, θ)
    mapped = m.f(prob, θ)
    mapped.grid == prob.grid || throw(ArgumentError("parameter map changed grid; build a new solver"))
    (mapped.tag === prob.tag || Array(mapped.tag) == Array(prob.tag)) ||
        throw(ArgumentError("parameter map changed cell tags; build a new solver"))
    if prob.immersed !== nothing || mapped.immersed !== nothing
        (prob.immersed !== nothing && mapped.immersed !== nothing) ||
            throw(ArgumentError("parameter map changed immersed topology; build a new solver"))
        prob.immersed.closure == mapped.immersed.closure ||
            throw(ArgumentError("parameter map changed immersed closure; build a new solver"))
        _fv_check_mass(_fv_mass(prob.immersed), _fv_mass(mapped.immersed))
    end
    return mapped
end

"""
    fv_param_problem(map, prob, θ) -> FVSteadyProblem2D

The problem `prob` with the parameters `θ` of `map` substituted (arrays on the
backend of `prob`, element type promoted with `eltype(θ)`; duals allowed).
Allocating; the adjoint uses preallocated buffers instead.
"""
function fv_param_problem(m::FVParameterMap, prob::FVSteadyProblem2D, θ::AbstractVector)
    _fv_check_nparams(m, prob, θ)
    T = promote_type(eltype(θ), Float64)
    return _fv_param_problem!(_fv_param_buffers(m, prob, T), m, prob, θ)
end

"""
    fv_set_params!(solver, map, θ; fields = nothing) -> solver

Write the Float64 parameters `θ` into the arrays of `solver.prob` in place
(viscosity, boundary velocities, sources), and replace its viscosity model
when the map changes law parameters, so that the next
[`fv_newton!`](@ref) solves the problem at `θ` with the SAME sparsity
pattern, colouring and cuDSS symbolic analysis (an optimisation loop over `θ`
never re-analyses). The map must keep the grid, cell tags, array types and
shapes, wall closure, and the exact viscosity-model type. A change of model
type or immersed topology requires a new solver because the Jacobian pattern
may change. These constraints are checked before any parameter arrays are
overwritten. Cut fractions, wall velocities, surface data and mass coefficients
are copied into the existing device arrays.
For [`FVLawViscosity`](@ref), `fields` may replace prescribed field values;
their count, shapes, array types and backend must remain unchanged.
"""
function fv_set_params!(s::FVNewtonSolver, m::FVParameterMap, θ::AbstractVector; fields = nothing)
    P = fv_param_problem(m, s.prob, Vector{Float64}(Array(θ)))
    fields === nothing || (P = _fv_with(P; visc = _fv_with_fields(P.visc, fields)))
    _fv_validate_viscosity(P.visc, P.grid, P.nu)
    typeof(P.visc) === typeof(s.prob.visc) ||
        throw(ArgumentError("the parameter map must keep the solver's viscosity-model type"))
    typeof(P) === typeof(s.prob) ||
        throw(ArgumentError("the parameter map must keep the solver's problem and array types"))
    P.grid == s.prob.grid || throw(ArgumentError("the parameter map must keep the solver's grid"))
    (P.tag === s.prob.tag || Array(P.tag) == Array(s.prob.tag)) ||
        throw(ArgumentError("the parameter map changed cell tags; build a new solver"))
    size(P.nu) == size(s.prob.nu) && size(P.bvel) == size(s.prob.bvel) &&
        size(P.src) == size(s.prob.src) ||
        throw(DimensionMismatch("the parameter map must keep all parameter array shapes"))
    if P.immersed !== nothing
        old = s.prob.immersed; new = P.immersed
        old.h == new.h && old.closure == new.closure ||
            throw(ArgumentError("the parameter map must keep the immersed grid and closure"))
        Array(old.tag) == Array(new.tag) ||
            throw(ArgumentError("the parameter map changed immersed cell tags; build a new solver"))
        size(old.qcut) == size(new.qcut) && size(old.wvel) == size(new.wvel) ||
            throw(DimensionMismatch("the parameter map must keep immersed array shapes"))
        _fv_check_surface(old.surface, new.surface)
        _fv_check_mass(_fv_mass(old), _fv_mass(new))
    end
    P.nu === s.prob.nu || copyto!(s.prob.nu, P.nu)
    P.bvel === s.prob.bvel || copyto!(s.prob.bvel, P.bvel)
    P.src === s.prob.src || copyto!(s.prob.src, P.src)
    if P.immersed !== nothing
        old = s.prob.immersed; new = P.immersed
        old.qcut === new.qcut || copyto!(old.qcut, new.qcut)
        old.wvel === new.wvel || copyto!(old.wvel, new.wvel)
        _fv_copy_surface!(old.surface, new.surface)
        _fv_copy_mass!(_fv_mass(old), _fv_mass(new))
    end
    s.prob = _fv_with(s.prob; visc = P.visc)
    s.fac_sig = (NaN, NaN, NaN)         # factors no longer match the problem
    _fv_sync(s.R)
    return s
end

# ------------------------------------------------------------------ objectives
"""
    FVObjective

A scalar objective `𝒥(U, θ)` of the converged FV state. An objective type
implements

- `fv_objective(obj, U, prob)`: its value from a HOST state `U` and a host
  problem `prob`, generic in the element types of both (dual numbers);
- `fv_objective_support(obj, prob) -> Vector{Int}`: the state indices it
  reads (its gradient `∂𝒥/∂U` is zero elsewhere);
- `fv_objective_explicit(obj) -> Bool`: whether it reads the parameter arrays
  of `prob` (viscosity, boundary velocities, sources) — then `∂𝒥/∂θ` is
  computed; an explicit dependence on `ν` must be limited to the cells whose
  unknowns are in the support (true of every built-in objective).

Built-in: [`FVStateObs`](@ref) (misfit of weighted state observations, the FV
counterpart of `GNFVelocityObs`), [`FVLinearFunctional`](@ref) (flow rate,
pressure drop: [`fv_flow_rate`](@ref), [`fv_pressure_drop`](@ref)) and
[`FVWallForce`](@ref) (force of the fluid on walls, e.g. drag).
"""
abstract type FVObjective end

fv_objective_explicit(::FVObjective) = false

"""
    FVStateObs(index, weight, target, scale)
    FVStateObs(prob, profile::LineProfile, target; scale = 1.0)

Observations of the FV state: value `m` is `scale · Σ_k weight[k, m] ·
U[index[k, m]]` (`index` = state index from [`fv_index`](@ref), any
component; `0` = unused slot); objective `𝒥 = ½ Σ_m (value_m − target_m)²`.
Same convention as `GNFVelocityObs` of the LBM GNF adjoint. The second form
observes `profile.field` (`:u`/`:ux`, `:v`/`:uy` or `:p`) at the cell indices
`(i, j)` of a platform [`LineProfile`](@ref), one observation per cell.
"""
struct FVStateObs <: FVObjective
    index::Matrix{Int}
    weight::Matrix{Float64}
    target::Vector{Float64}
    scale::Float64
end

function _fv_field_comp(f::Symbol, layout::FVFieldLayout = FVFieldLayout{0}())
    f in (:u, :ux) && return _fv_velocity_slot(layout, 1)
    f in (:v, :uy) && return _fv_velocity_slot(layout, 2)
    f === :p && return _fv_pressure_slot(layout)
    throw(ArgumentError("FV observable field must be :u/:ux, :v/:uy or :p, got :$f"))
end

function FVStateObs(prob::FVSteadyProblem2D, lp::LineProfile, target::AbstractVector;
                    scale::Real = 1.0)
    comp = _fv_field_comp(lp.field, prob.grid.layout)
    idx = [fv_index(prob.grid, ij[1], ij[2], comp) for ij in lp.indices]
    length(target) == length(idx) || throw(DimensionMismatch("one target per profile point"))
    return FVStateObs(reshape(idx, 1, :), ones(1, length(idx)), Vector{Float64}(target), Float64(scale))
end

"""
    fv_obs_values(obs::FVStateObs, U) -> Vector

Observed values of the host state `U` (generic element type).
"""
function fv_obs_values(obs::FVStateObs, U::AbstractVector)
    M = length(obs.target)
    T = eltype(U)
    v = Vector{T}(undef, M)
    for m in 1:M
        acc = zero(T)
        for k in axes(obs.index, 1)
            q = obs.index[k, m]
            q == 0 && continue
            acc += obs.weight[k, m] * U[q]
        end
        v[m] = obs.scale * acc
    end
    return v
end

fv_objective(obs::FVStateObs, U, prob) = sum(abs2, fv_obs_values(obs, U) .- obs.target) / 2
fv_objective_support(obs::FVStateObs, prob) = unique!(filter(!=(0), vec(obs.index)))

"""
    FVLinearFunctional(index, weight)

`𝒥 = Σ_k weight[k] · U[index[k]]`: a linear functional of the state (flow
rate, mean pressure difference, ...). See [`fv_flow_rate`](@ref) and
[`fv_pressure_drop`](@ref).
"""
struct FVLinearFunctional <: FVObjective
    index::Vector{Int}
    weight::Vector{Float64}
end

function fv_objective(f::FVLinearFunctional, U, prob)
    acc = zero(eltype(U))
    for k in eachindex(f.index)
        acc += f.weight[k] * U[f.index[k]]
    end
    return acc
end
fv_objective_support(f::FVLinearFunctional, prob) = unique(f.index)

_fv_is_fluid(tag, i, j) = tag[i, j] != FV_CELL_SOLID

"""
    fv_flow_rate(prob, i) -> FVLinearFunctional

Volume flow rate (per unit depth) through cell column `i`: `h Σ_j u(i, j)`
over its fluid cells.
"""
function fv_flow_rate(prob::FVSteadyProblem2D, i::Integer)
    g = prob.grid; tag = Array(prob.tag)
    js = [j for j in 1:g.Ny if _fv_is_fluid(tag, i, j)]
    return FVLinearFunctional([fv_index(g, i, j, _fv_velocity_slot(g.layout, 1)) for j in js], fill(Float64(g.h), length(js)))
end

"""
    fv_pressure_drop(prob, i1, i2) -> FVLinearFunctional

Mean pressure over the fluid cells of column `i1` minus that of column `i2`.
"""
function fv_pressure_drop(prob::FVSteadyProblem2D, i1::Integer, i2::Integer)
    g = prob.grid; tag = Array(prob.tag)
    idx = Int[]; w = Float64[]
    for (i, sgn) in ((i1, 1.0), (i2, -1.0))
        js = [j for j in 1:g.Ny if _fv_is_fluid(tag, i, j)]
        append!(idx, [fv_index(g, i, j, _fv_pressure_slot(g.layout)) for j in js])
        append!(w, fill(sgn / length(js), length(js)))
    end
    return FVLinearFunctional(idx, w)
end

"""
    FVWallForce(dir; region = :solid, pressure = :cell)

`kind=:residual_flux` explicitly selects the exact complete residual wall
momentum flux, including convection; pressure overrides are rejected.
`kind=:surface` selects physical-cut traction with analytic arc weights,
requiring geometry built by `fv_immersed_boundary`. `wall=1,2,...` selects
one circle (inner/outer for annuli); zero selects all. Surface geometry and
wall data remain differentiable. Irregular midpoint arc quadrature does not
annihilate constant stress exactly. The default `kind=:legacy` preserves
the C4 convention below, excluding advection and permitting its pressure
postprocessing override.

Force (per unit depth) exerted by the fluid on walls, component `dir`
(1 = x, e.g. the drag of an obstacle in an x-directed flow; 2 = y): the sum,
over the wall faces of `region`, of the pressure and viscous momentum
contributions — `h (p_f n − ν_w (∇u + ∇uᵀ)·n)`, using the same wall-stress
helper, viscosity model and immersed closure as the residual. On C1/domain
walls the strain uses one-sided normal differences `(u_w − u_P)·2/h`, oriented
along the coordinate axes, and the cell gradient tangentially; immersed L/Q
walls use the ghost normal slope and tangential gradients extrapolated to the
nominal face. `ν_w` is the cell viscosity (extrapolated for immersed L/Q) for a
Newtonian problem or the GNF law evaluated at that wall strain. The advective
contribution is excluded, preserving the C4 force convention. The objective
depends explicitly on the viscosity law and on the wall velocities.
`region = :solid` takes the faces between fluid cells and solid cells;
`region = :E, :W, :N, :S` the wall/inlet faces of that domain side.

`pressure = :cell` uses the residual's wall pressure (`p_f = p_P` in C1);
`pressure = :linear`
extrapolates linearly from the two cells normal to the wall,
`p_f = p_P + (p_P − p_PP)/2` (falls back to `p_P` when the second cell is
not fluid). The second option is a post-processing choice of the objective
only (the residual is unchanged). Prefer `:linear` for drag when pressure
varies normally to the wall; `:cell` remains the default for compatibility
with the residual's pressure closure. Both choices use the same wall strain
and viscosity model.
"""
abstract type FVWallObjective <: FVObjective end

struct FVWallForce <: FVWallObjective
    dir::Int
    region::Symbol
    pressure::Symbol
    kind::Symbol
    wall::Int
    function FVWallForce(dir::Integer; region::Symbol = :solid, pressure::Symbol = :cell,
                         kind::Symbol = :legacy, wall::Integer = 0)
        dir in (1, 2) || throw(ArgumentError("dir must be 1 (x) or 2 (y)"))
        region in (:solid, :E, :W, :N, :S) || throw(ArgumentError("region must be :solid, :E, :W, :N or :S"))
        pressure in (:cell, :linear) || throw(ArgumentError("pressure must be :cell or :linear"))
        kind in (:legacy, :residual_flux, :surface) || throw(ArgumentError("unknown wall objective kind $kind"))
        kind != :legacy && pressure != :cell && throw(ArgumentError("pressure override is only available for the legacy objective"))
        kind == :surface && region != :solid && throw(ArgumentError("surface objectives require immersed solid walls"))
        wall >= 0 || throw(ArgumentError("wall must be nonnegative (0 selects all walls)"))
        return new(Int(dir), region, pressure, kind, Int(wall))
    end
end

fv_objective_explicit(::FVWallObjective) = true

# is face d of fluid cell (i, j) a wall face of `region`?
function _fv_wall_face(g, tag, i, j, d, region)
    kind, _, _ = _fv_face_kind(g, tag, i, j, d)
    kind == _FV_F_DIRICHLET || return false
    ii = i + (d == 1) - (d == 2)
    jj = j + (d == 3) - (d == 4)
    inside = 1 <= ii <= g.Nx && 1 <= jj <= g.Ny
    if region === :solid
        return inside       # in-domain Dirichlet face = solid neighbour (periodic wrap included)
    else
        return !inside && _FV_SIDES[d] === region
    end
end

# Host-only local viscosity overrides. The shared stress helper reads P and,
# for immersed L/Q faces, O; no full-grid dual viscosity array is needed.
struct _FVWallViscosity{A,T} <: AbstractMatrix{T}
    values::A
    index::NTuple{2,Int}
    local_values::NTuple{2,T}
end
Base.size(nu::_FVWallViscosity) = size(nu.values)
@inline function Base.getindex(nu::_FVWallViscosity{A,T}, i::Int, j::Int) where {A,T}
    k = i + (j - 1) * size(nu, 1)
    return k == nu.index[1] ? nu.local_values[1] :
           k == nu.index[2] ? nu.local_values[2] : T(nu.values[i, j])
end

@inline function _fv_wall_nu_cells(prob, i, j, d, qcut, closure)
    g = prob.grid
    p = i + (j - 1) * g.Nx
    if _fv_immersed_face(g, qcut, i, j, d)
        oi = i - (d == 1) + (d == 2)
        oj = j - (d == 3) + (d == 4)
        return p, oi + (oj - 1) * g.Nx
    end
    return p, 0
end

# One shared face flux for objective values and local derivatives. Preserve
# the local nuP override and allow nuO to carry the second immersed derivative.
@inline function _fv_wall_force_face(f::FVWallForce, U, prob, i, j, d,
                                     qcut, wvel, closure, nuP = prob.nu[i, j], nuO = nothing)
    f.kind == :surface && return _fv_surface_force_face(f, U, prob, i, j, d, nuP, nuO)
    g = prob.grid; tag = prob.tag; bvel = prob.bvel
    h = g.h
    cells = _fv_wall_nu_cells(prob, i, j, d, qcut, closure)
    opposite = nuO === nothing ? (cells[2] > 0 ? prob.nu[cells[2]] : nuP) : nuO
    nu = _FVWallViscosity(prob.nu, cells, promote(nuP, opposite))
    if f.kind == :residual_flux
        gP = _fv_cell_grad(U, tag, bvel, g, i, j, qcut, wvel, closure)
        flux = _fv_wall_momentum_flux(U, tag, nu, bvel, g, i, j, d, prob.visc,
                                      qcut, wvel, closure, gP, _fv_face_pair(prob.immersed))
        T = promote_type(eltype(U), eltype(nu), eltype(bvel),
                         _fv_visc_eltype(prob.visc), _fv_immersed_eltype(prob.immersed))
        return flux[f.dir] + zero(T)
    end
    isx = d <= 2
    sx = (d == 1 || d == 3) ? 1 : -1
    pP = _fv_val(U, g, i, j, _fv_pressure_slot(g.layout))
    gP = _fv_cell_grad(U, tag, bvel, g, i, j, qcut, wvel, closure)
    _, _, _, _, pf, _, _, dun, dvn =
        _fv_wall_data(U, bvel, g, i, j, d, qcut, wvel, closure)
    gF = _fv_wall_gradient(U, tag, bvel, g, i, j, d, gP, qcut, wvel, closure)
    tx, ty = _fv_wall_stress(nu, prob.visc, g, i, j, d, gF, dun, dvn, qcut, closure)
    if f.pressure === :linear
        # Second cell away from the wall; retain the C1 fallback at corners.
        dop = d == 1 ? 2 : d == 2 ? 1 : d == 3 ? 4 : 3
        kind, ii, jj = _fv_face_kind(g, tag, i, j, dop)
        if kind == _FV_F_INT
            pf = pP + (pP - _fv_val(U, g, ii, jj, _fv_pressure_slot(g.layout))) / 2
        end
    end
    force = f.dir == 1 ? -h * tx + (isx ? h * sx * pf : zero(pf)) :
                         -h * ty + (isx ? zero(pf) : h * sx * pf)
    # GNF ignores the cell viscosity, but its zero derivative must retain
    # the local dual type when only viscosity was seeded. Conversion preserves
    # the signed zero of the Newtonian force.
    T = promote_type(eltype(U), eltype(nu), eltype(bvel),
                     _fv_visc_eltype(prob.visc), _fv_immersed_eltype(prob.immersed))
    return convert(T, force)
end

function _fv_wall_faces(f::FVWallObjective, prob)
    g = prob.grid; tag = prob.tag
    (f.kind == :surface || f.wall > 0) && _fv_surface_data(prob)
    faces = NTuple{3,Int}[]
    # A domain-side objective only reads that side; solid faces can be anywhere.
    ir = f.region === :E ? (g.Nx:g.Nx) : f.region === :W ? (1:1) : (1:g.Nx)
    jr = f.region === :N ? (g.Ny:g.Ny) : f.region === :S ? (1:1) : (1:g.Ny)
    for j in jr, i in ir
        tag[i, j] == FV_CELL_SOLID && continue
        for d in 1:4
            _fv_wall_face(g, tag, i, j, d, f.region) || continue
            if f.kind == :surface || f.wall > 0
                _fv_surface_data(prob).wall[d, i, j] > 0 || continue
                f.wall > 0 && _fv_surface_data(prob).wall[d, i, j] != f.wall && continue
            end
            push!(faces, (i, j, d))
        end
    end
    return faces
end

function _fv_wall_force_sum(f, U, prob, faces)
    T = promote_type(eltype(U), eltype(prob.nu), eltype(prob.bvel),
                     _fv_visc_eltype(prob.visc), _fv_immersed_eltype(prob.immersed))
    F = zero(T)
    qcut, wvel, closure = _fv_immersed_args(prob.immersed)
    for (i, j, d) in faces
        F += _fv_wall_force_face(f, U, prob, i, j, d, qcut, wvel, closure)
    end
    return F
end

fv_objective(f::FVWallObjective, U, prob) = _fv_wall_force_sum(f, U, prob, _fv_wall_faces(f, prob))

function fv_objective_support(f::FVWallObjective, prob)
    g = prob.grid; tag = Array(prob.tag)
    cells = Set{Int}()
    for j in 1:g.Ny, i in 1:g.Nx
        tag[i, j] == FV_CELL_SOLID && continue
        any(_fv_wall_face(g, tag, i, j, d, f.region) for d in 1:4) || continue
        # Share the local differentiation support, including tangential
        # neighbours of O for the immersed nominal-face stress.
        for c in _fv_wall_state_cells(prob, i, j, tag)
            c > 0 && push!(cells, c)
        end
    end
    support = [_fv_cell_index(g, c, comp) for c in cells for comp in 1:_fv_nfields(g)]
    if f.kind == :residual_flux && _fv_mass(prob.immersed) !== nothing
        face_pair = prob.immersed.mass.face_pair
        for (i,j,d) in _fv_wall_faces(f, prob)
            k = face_pair[d,i,j]
            k > 0 && push!(support, fv_nunknowns(g)+k)
        end
    end
    return sort!(unique!(support))
end

# ∂𝒥/∂U on the host: forward duals seeded on the support only, `chunk` at a time
struct _FVObjTag end

# Host-only lazy state duals: one fixed stencil, never a full-grid dual vector.
# Repeated periodic neighbours are removed before seeding and scattering.
struct _FVWallState{A,N} <: AbstractVector{ForwardDiff.Dual{_FVObjTag,Float64,N}}
    values::A
    index::NTuple{N,Int}
end
Base.size(U::_FVWallState) = size(U.values)
@inline function Base.getindex(U::_FVWallState{A,N}, k::Int) where {A,N}
    return ForwardDiff.Dual{_FVObjTag}(U.values[k],
        ForwardDiff.Partials(ntuple(s -> U.index[s] == k ? 1.0 : 0.0, Val(N))))
end

function _fv_wall_state_cells(prob, i, j, tag = prob.tag)
    g = prob.grid
    if prob.immersed !== nothing && prob.immersed.closure.ghost != 0
        # P, its axial neighbours and all four diagonals cover gP and the
        # tangential parts of gO. Keep solid/out-of-domain slots unseeded.
        return ntuple(Val(9)) do k
            ii, jj = i + mod1(k, 3) - 2, j + (k - 1) ÷ 3 - 1
            if ii < 1 || ii > g.Nx
                _fv_side_bc(g, ii < 1 ? 2 : 1) == FV_BC_PERIODIC || return 0
                ii = mod1(ii, g.Nx)
            end
            if jj < 1 || jj > g.Ny
                _fv_side_bc(g, jj < 1 ? 4 : 3) == FV_BC_PERIODIC || return 0
                jj = mod1(jj, g.Ny)
            end
            return tag[ii, jj] == FV_CELL_SOLID ? 0 : ii + (jj - 1) * g.Nx
        end
    end
    return ntuple(Val(5)) do k
        k == 1 && return i + (j - 1) * g.Nx
        kind, ii, jj = _fv_face_kind(g, tag, i, j, k - 1)
        return kind == _FV_F_INT ? ii + (jj - 1) * g.Nx : 0
    end
end

function _fv_wall_state(U, prob, i, j, d=0)
    raw = _fv_wall_state_cells(prob, i, j)
    cells = ntuple(k -> any(l -> raw[l] == raw[k], 1:(k - 1)) ? 0 : raw[k], length(raw))
    nf = _fv_nfields(prob.grid)
    indices = ntuple(k -> _fv_cell_index(prob.grid, cells[(k - 1) ÷ nf + 1], mod1(k, nf)), nf * length(raw))
    face_pair = _fv_face_pair(prob.immersed)
    if face_pair !== nothing && d > 0
        k = face_pair[d,i,j]
        k > 0 && return _FVWallState(U, (indices..., fv_nunknowns(prob.grid)+k))
    end
    return _FVWallState(U, indices)
end

function _fv_objective_grad_u(obj::FVWallObjective, Uh::Vector{Float64}, hprob; chunk::Int = 8)
    gU = zeros(Float64, length(Uh))
    J = 0.0
    qcut, wvel, closure = _fv_immersed_args(hprob.immersed)
    for (i, j, d) in _fv_wall_faces(obj, hprob)
        Ud = _fv_wall_state(Uh, hprob, i, j, d)
        v = _fv_wall_force_face(obj, Ud, hprob, i, j, d, qcut, wvel, closure)
        J += ForwardDiff.value(v)
        pd = ForwardDiff.partials(v)
        for k in eachindex(Ud.index)
            q = Ud.index[k]
            q > 0 && (gU[q] += pd[k])
        end
    end
    return J, gU
end

function _fv_objective_grad_u(obj::FVObjective, Uh::Vector{Float64}, hprob; chunk::Int = 8)
    supp = fv_objective_support(obj, hprob)
    D = ForwardDiff.Dual{_FVObjTag,Float64,chunk}
    zp = ForwardDiff.Partials(ntuple(_ -> 0.0, Val(chunk)))
    Ud = D[D(x, zp) for x in Uh]
    gU = zeros(Float64, length(Uh))
    Jv = fv_objective(obj, Uh, hprob)
    for off in 0:chunk:(length(supp) - 1)
        ks = off+1:min(off + chunk, length(supp))
        for (m, k) in enumerate(ks)
            Ud[supp[k]] = D(Uh[supp[k]], ForwardDiff.Partials(ntuple(c -> c == m ? 1.0 : 0.0, Val(chunk))))
        end
        pd = ForwardDiff.partials(fv_objective(obj, Ud, hprob))
        for (m, k) in enumerate(ks)
            gU[supp[k]] = pd[m]
            Ud[supp[k]] = D(Uh[supp[k]], zp)
        end
    end
    return Float64(Jv), gU
end

# explicit ∂𝒥/∂θ on the host
function _fv_objective_grad_theta(obj::FVObjective, m::FVParameterMap, Uh, hprob, θ;
                                  chunk::Int = 8)
    P = fv_nparams(m, hprob)
    gθ = zeros(Float64, P)
    fv_objective_explicit(obj) || return gθ
    D = ForwardDiff.Dual{_FVObjTag,Float64,chunk}
    for off in 0:chunk:(P - 1)
        θd = [D(θ[k], ForwardDiff.Partials(ntuple(c -> c == k - off ? 1.0 : 0.0, Val(chunk))))
              for k in 1:P]
        pd = ForwardDiff.partials(fv_objective(obj, Uh, fv_param_problem(m, hprob, θd)))
        for k in off+1:min(off + chunk, P)
            gθ[k] = pd[k - off]
        end
    end
    return gθ
end

# viscosity field: only the cells of the support can carry an explicit
# dependence (contract of FVObjective)
function _fv_objective_grad_theta(obj::FVObjective, ::FVViscosityField, Uh, hprob, θ;
                                  chunk::Int = 8)
    g = hprob.grid
    gθ = zeros(Float64, g.Nx * g.Ny)
    fv_objective_explicit(obj) || return gθ
    cells = unique!([_fv_cell_from_index(g, k) for k in fv_objective_support(obj, hprob) if k <= fv_nunknowns(g)])
    D = ForwardDiff.Dual{_FVObjTag,Float64,chunk}
    zp = ForwardDiff.Partials(ntuple(_ -> 0.0, Val(chunk)))
    nud = D[D(x, zp) for x in hprob.nu]
    pd_prob = _fv_with(hprob; nu = nud)
    for off in 0:chunk:(length(cells) - 1)
        ks = off+1:min(off + chunk, length(cells))
        for (m, k) in enumerate(ks)
            c = cells[k]
            nud[c] = D(hprob.nu[c], ForwardDiff.Partials(ntuple(q -> q == m ? 1.0 : 0.0, Val(chunk))))
        end
        pd = ForwardDiff.partials(fv_objective(obj, Uh, pd_prob))
        for (m, k) in enumerate(ks)
            c = cells[k]
            gθ[c] = pd[m]
            nud[c] = D(hprob.nu[c], zp)
        end
    end
    return gθ
end

# Two local duals per face cover P and the opposite fluid cell O of immersed
# L/Q. A uniform parameter seeds both; its derivative is the sum of the two.
function _fv_wall_nu_gradient(obj, m, Uh, hprob, θ)
    gθ = zeros(Float64, fv_nparams(m, hprob))
    qcut, wvel, closure = _fv_immersed_args(hprob.immersed)
    for (i, j, d) in _fv_wall_faces(obj, hprob)
        p, o = _fv_wall_nu_cells(hprob, i, j, d, qcut, closure)
        νP = m isa FVUniformViscosity ? θ[1] : hprob.nu[p]
        νO = m isa FVUniformViscosity ? θ[1] : o > 0 ? hprob.nu[o] : νP
        νPd = ForwardDiff.Dual{_FVObjTag}(νP, 1.0, 0.0)
        νOd = ForwardDiff.Dual{_FVObjTag}(νO, 0.0, 1.0)
        v = _fv_wall_force_face(obj, Uh, hprob, i, j, d, qcut, wvel, closure, νPd, νOd)
        pd = ForwardDiff.partials(v)
        if m isa FVViscosityField
            gθ[p] += pd[1]
            o > 0 && (gθ[o] += pd[2])
        else
            gθ[1] += pd[1] + pd[2]
        end
    end
    return gθ
end

_fv_objective_grad_theta(obj::FVWallObjective, m::FVViscosityField, Uh, hprob, θ; chunk::Int = 8) =
    _fv_wall_nu_gradient(obj, m, Uh, hprob, θ)
_fv_objective_grad_theta(obj::FVWallObjective, m::FVUniformViscosity, Uh, hprob, θ; chunk::Int = 8) =
    _fv_wall_nu_gradient(obj, m, Uh, hprob, θ)
_fv_objective_grad_theta(::FVWallObjective, ::FVUniformSource, Uh, hprob, θ; chunk::Int = 8) = zeros(1)

# ------------------------------------------------------------------ workspace
"""
    FVAdjointWorkspace

Preallocated device buffers of the FV adjoint (build with
[`fv_adjoint_workspace`](@ref)): adjoint state `λ`, right-hand side `g`,
tangent buffers, the dual residual `Rd` and dual parameter arrays
(`Dual{FVParamTag,Float64,N}`), partial sums of `λᵀ ∂R/∂θ`, and for
[`FVViscosityField`](@ref) the colouring of `∂R/∂ν` and the cotangent field.
[`FVLawParameters`](@ref) additionally stores real parameters, face inputs and
cotangents, tile partial sums and a parameter reduction buffer. Both scalar-law
gradient modes reuse these buffers without allocating device arrays per call.
"""
struct FVAdjointWorkspace{N,M,V,VD,B,VI,S,PT,LC}
    map::M
    λ::V
    g::V
    r::V
    Rd::VD
    buf::B
    nucolors::VI
    nncolors::Int
    s::S
    Pt::PT
    Ph::Matrix{Float64}
    per::Int
    law::LC
end

# Scalar-law maps specialize this after their constitutive adapter is loaded.
_fv_law_workspace(s, m) = nothing

"""
    fv_nu_coloring(prob) -> (colors, ncolors)

Column colouring of `∂R/∂ν` (rows grouped by cell): the viscosity of fluid
cell `c` enters the three rows of `c` and of its interior face neighbours;
two cells of one colour share no such row. Immersed extrapolation reads the
opposite fluid cell, already among these neighbours, so the colouring and
the gather below retain the same support. CPU, once per geometry.
"""
function fv_nu_coloring(prob::FVSteadyProblem2D)
    g = prob.grid; tag = Array(prob.tag)
    nc = g.Nx * g.Ny
    I = Int[]; J = Int[]
    for j in 1:g.Ny, i in 1:g.Nx
        c = i + (j - 1) * g.Nx
        push!(I, c); push!(J, c)
        tag[i, j] == FV_CELL_SOLID && continue
        for d in 1:4
            kind, ii, jj = _fv_face_kind(g, tag, i, j, d)
            kind == _FV_F_INT && (push!(I, c); push!(J, ii + (jj - 1) * g.Nx))
        end
    end
    return fv_jacobian_coloring(sparse(I, J, trues(length(I)), nc, nc, |))
end

function _fv_require_isothermal_adjoint(prob)
    prob.energy === nothing || throw(ArgumentError(
        "thermal sensitivities are not supported yet: heating-source parameter derivatives are not implemented"))
    return nothing
end

"""
    fv_adjoint_workspace(solver, map; chunk = 4) -> FVAdjointWorkspace

Allocate the adjoint buffers for `solver` (an [`FVNewtonSolver`](@ref), on its
device) and the parameter map `map`; `chunk` = number of parameters (scalar
maps) or colours ([`FVViscosityField`](@ref)) per dual residual evaluation.
Thermal sensitivities are not supported by this workspace yet.
"""
function fv_adjoint_workspace(s::FVNewtonSolver, m::FVParameterMap; chunk::Int = 4)
    prob = s.prob
    _fv_require_isothermal_adjoint(prob)
    n = fv_nunknowns(prob)
    D = ForwardDiff.Dual{FVParamTag,Float64,chunk}
    vec0() = fill!(similar(s.R, Float64, n), 0)
    Rd = similar(s.R, D, n)
    buf = _fv_param_buffers(m, prob, D)
    if m isa FVViscosityField
        colors, ncol = fv_nu_coloring(prob)
        nucolors = similar(s.R, Int32, length(colors)); copyto!(nucolors, Int32.(colors))
        sfield = fill!(similar(prob.nu, Float64), 0)
        per = 0
        Pt = nothing; Ph = zeros(0, 0)
    else
        nucolors = nothing; ncol = 0; sfield = nothing
        per = 256
        nt = cld(n, per)
        Pt = fill!(similar(s.R, Float64, chunk, nt), 0)
        Ph = zeros(Float64, chunk, nt)
    end
    law = _fv_law_workspace(s, m)
    return FVAdjointWorkspace{chunk,typeof(m),typeof(s.R),typeof(Rd),typeof(buf),
                              typeof(nucolors),typeof(sfield),typeof(Pt),typeof(law)}(
        m, vec0(), vec0(), vec0(), Rd, buf, nucolors, ncol, sfield, Pt, Ph, per, law)
end

_fv_ws_chunk(::FVAdjointWorkspace{N}) where {N} = N

# ------------------------------------------------------------------ kernels
@kernel function _fv_lambda_dot_kernel!(Pt, @Const(λ), @Const(Rd), n, per, ::Val{N}) where {N}
    t = @index(Global)
    r0 = (t - 1) * per + 1
    r1 = min(t * per, n)
    acc = ntuple(_ -> 0.0, Val(N))
    @inbounds for r in r0:r1
        l = λ[r]
        pv = ForwardDiff.partials(Rd[r])
        # bind the accumulator so the closure does not capture a reassigned
        # variable (Core.Box, rejected by the GPU compiler)
        acc = let a = acc
            ntuple(k -> a[k] + l * pv[k], Val(N))
        end
    end
    @inbounds for k in 1:N
        Pt[k, t] = acc[k]
    end
end

@kernel function _fv_nu_seed_kernel!(nud, @Const(nu), @Const(colors), offset, ::Val{N}) where {N}
    c = @index(Global)
    @inbounds begin
        col = colors[c] - offset
        nud[c] = ForwardDiff.Dual{FVParamTag}(nu[c],
                    ForwardDiff.Partials(ntuple(m -> m == col ? 1.0 : 0.0, Val(N))))
    end
end

@kernel function _fv_nu_direction_kernel!(nud, @Const(nu), @Const(dnu), ::Val{N}) where {N}
    c = @index(Global)
    @inbounds nud[c] = ForwardDiff.Dual{FVParamTag}(nu[c],
                           ForwardDiff.Partials(ntuple(m -> m == 1 ? dnu[c] : 0.0, Val(N))))
end

@kernel function _fv_partial1_kernel!(r, @Const(Rd))
    k = @index(Global)
    @inbounds r[k] = ForwardDiff.partials(Rd[k])[1]
end

@inline function _fv_rows_dot(λ, Rd, g, i, j, col)
    b = fv_index(g, i, j, 0)
    @inbounds result = λ[b + _fv_velocity_slot(g.layout, 1)] * ForwardDiff.partials(Rd[b + _fv_velocity_slot(g.layout, 1)])[col] +
                     λ[b + _fv_velocity_slot(g.layout, 2)] * ForwardDiff.partials(Rd[b + _fv_velocity_slot(g.layout, 2)])[col] +
                     λ[b + _fv_pressure_slot(g.layout)] * ForwardDiff.partials(Rd[b + _fv_pressure_slot(g.layout)])[col]
    # Preserve the original triple's addition order, then extend by scalar rows.
    @inbounds for scalar in 1:_fv_nscalars(g.layout)
        row = b + _fv_scalar_slot(g.layout, scalar)
        result += λ[row] * ForwardDiff.partials(Rd[row])[col]
    end
    return result
end

# s[c] = −λᵀ ∂R/∂ν_c for the cells c of the colours offset+1 .. offset+N:
# rows of c and of its distinct interior face neighbours
@kernel function _fv_nu_gather_kernel!(s, @Const(λ), @Const(Rd), @Const(tag), @Const(colors),
                                       g, offset, ::Val{N}) where {N}
    i, j = @index(Global, NTuple)
    c = i + (j - 1) * g.Nx
    @inbounds col = colors[c] - offset
    if 1 <= col <= N
        acc = 0.0
        @inbounds if tag[i, j] != FV_CELL_SOLID
            acc = _fv_rows_dot(λ, Rd, g, i, j, col)
            for d in 1:4
                kind, ii, jj = _fv_face_kind(g, tag, i, j, d)
                fresh = kind == _FV_F_INT && !(ii == i && jj == j)
                for e in 1:(d - 1)
                    k2, i2, j2 = _fv_face_kind(g, tag, i, j, e)
                    fresh &= !(k2 == _FV_F_INT && i2 == ii && j2 == jj)
                end
                if fresh
                    acc += _fv_rows_dot(λ, Rd, g, ii, jj, col)
                end
            end
        end
        @inbounds s[i, j] = -acc
    end
end

# ------------------------------------------------------------------ ∂R/∂θ products
function _fv_check_law_gradient(mode)
    mode in (:contraction, :duals) ||
        throw(ArgumentError("law_gradient must be :contraction or :duals, got $mode"))
    return nothing
end

function _fv_param_cotangent!(ws::FVAdjointWorkspace, m::FVParameterMap,
                              s::FVNewtonSolver, U, θ; law_gradient = :contraction)
    _fv_check_law_gradient(law_gradient)
    return _fv_param_cotangent_duals!(ws, m, s, U, θ)
end

# −λᵀ ∂R/∂θ for a scalar map: ceil(P / N) dual residual evaluations
function _fv_param_cotangent_duals!(ws::FVAdjointWorkspace{N}, m::FVParameterMap, s::FVNewtonSolver,
                              U, θ) where {N}
    θ === nothing && throw(ArgumentError("θ is required for the parameter map $(typeof(m))"))
    prob = s.prob
    _fv_check_nparams(m, prob, θ)
    θ = θ isa Array ? θ : Array(θ)
    P = length(θ)
    D = ForwardDiff.Dual{FVParamTag,Float64,N}
    out = zeros(Float64, P)
    n = length(U)
    backend = KernelAbstractions.get_backend(U)
    dot! = _fv_lambda_dot_kernel!(backend, 64)
    for off in 0:N:(P - 1)
        θd = D[D(Float64(θ[k]), ForwardDiff.Partials(ntuple(c -> c == k - off ? 1.0 : 0.0, Val(N))))
               for k in 1:P]
        pd = _fv_param_problem!(ws.buf, m, prob, θd)
        fv_residual!(ws.Rd, U, pd; sync = false)
        dot!(ws.Pt, ws.λ, ws.Rd, n, ws.per, Val(N); ndrange = size(ws.Pt, 2))
        KernelAbstractions.synchronize(backend)
        H = ws.Pt isa Array ? ws.Pt : copyto!(ws.Ph, ws.Pt)
        for k in off+1:min(off + N, P)
            out[k] = -sum(@view H[k - off, :])
        end
    end
    return out
end

# −λᵀ ∂R/∂ν (Nx × Ny, on the device) for the viscosity field: ceil(ncolors / N)
# passes of seed → C1 residual on duals → gather
function _fv_param_cotangent_duals!(ws::FVAdjointWorkspace{N}, ::FVViscosityField, s::FVNewtonSolver,
                              U, θ) where {N}
    prob = s.prob
    g = prob.grid
    θ === nothing || _fv_check_nparams(FVViscosityField(), prob, θ)
    backend = KernelAbstractions.get_backend(U)
    seed! = _fv_nu_seed_kernel!(backend, 256)
    gather! = _fv_nu_gather_kernel!(backend, 64)
    pd = _fv_with(prob; nu = ws.buf.nu)
    for off in 0:N:(ws.nncolors - 1)
        seed!(ws.buf.nu, prob.nu, ws.nucolors, off, Val(N); ndrange = length(prob.nu))
        fv_residual!(ws.Rd, U, pd; sync = false)
        gather!(ws.s, ws.λ, ws.Rd, prob.tag, ws.nucolors, g, off, Val(N); ndrange = (g.Nx, g.Ny))
    end
    KernelAbstractions.synchronize(backend)
    return ws.s
end

# ------------------------------------------------------------------ public API
_fv_host_problem(p::FVSteadyProblem2D) = p.nu isa Array ? p : fv_adapt(Array, p)

# finalize = false: the cache must hold the UNSHIFTED J at this very U (shift
# and (ΣU, ΣU²) recorded when the factors were made; two O(N) reductions)
function _fv_check_factors(s::FVNewtonSolver, U)
    shift, su, sq = s.fac_sig
    cur = _fv_state_sig(s.norms, 0.0, U)
    ok(a, b) = isfinite(a) && abs(a - b) <= 1e-12 * max(abs(a), abs(b), 1.0)
    if !(shift == 0.0 && ok(su, cur[2]) && ok(sq, cur[3]))
        why = isnan(shift) ? "no factors recorded for a state (or the parameters changed since)" :
              shift != 0 ? "the factors include a pseudo-transient shift ($shift)" :
              "the factors were made at a different U"
        throw(ArgumentError("fv_steady_adjoint(...; finalize = false): the solver cache is not " *
                            "the unshifted Jacobian at the given U ($why); call " *
                            "fv_newton_finalize!(solver, U) or use finalize = true"))
    end
    return nothing
end

function _fv_check_transpose(s::FVNewtonSolver)
    s.cache === nothing &&
        throw(ArgumentError("the solver has no factorization: run fv_newton! (or fv_newton_finalize!) first"))
    lin_supports_transpose(s.cache) ||
        throw(ArgumentError("this linear-solve backend cannot solve with Jᵀ " *
                            "(cuDSS needs CUDSS.jl >= 1.0, i.e. CUDA.jl 6)"))
    return nothing
end

"""
    fv_steady_adjoint(solver, U, gU; params = FVViscosityField(), θ = nothing,
                      ws = nothing, finalize = true, law_gradient = :contraction) -> (s, stats)

Discrete adjoint of the steady FV residual at the converged state `U` (device
vector of `solver`), for a cotangent `gU = ∂𝒥/∂U` (host or device vector,
length `3 Nx Ny`, unknown ordering of [`fv_index`](@ref)):

1. `finalize = true`: [`fv_newton_finalize!`](@ref) re-assembles the exact
   Jacobian at `U` and refactorizes it — numeric only, the symbolic analysis
   of the Newton solve is reused (`solver.n_factorize` does not change);
2. `Jᵀ λ = gU` with the Newton factors (`lin_solve!(...; transpose = true)`);
3. `s = −λᵀ ∂R/∂θ` through the residual kernel. [`FVLawParameters`](@ref)
   contracts face-viscosity cotangents with local law pullbacks by default;
   `law_gradient = :duals` selects its parameter-dual reference. Other maps
   ([`FVParameterMap`](@ref)) retain their existing parameter-dual path.

Returns `s` — for [`FVViscosityField`](@ref) the per-cell cotangent of ν
(`Nx × Ny`, on the device of `solver`, zero at solid cells; the FV analogue of
the `s` of [`gnf_steady_adjoint`](@ref)), otherwise a host vector of length
`fv_nparams(params)` (`θ` required) — and `stats = (; λ, residual_inf,
t_finalize, t_solve, t_param)`. `residual_inf = ‖R(U)‖∞` at return: the
gradient is that of the discrete problem only if it is at round-off. Any
explicit dependence of the objective on `θ` must be added by the caller (see
[`fv_steady_gradient`](@ref)). `ws` ([`fv_adjoint_workspace`](@ref)) is built
on the fly when omitted; pass one to keep the call free of device allocations.
With `finalize = false` the solver must hold the unshifted Jacobian at `U`
(right after [`fv_newton_finalize!`](@ref)): the shift and a signature of `U`
(ΣU, ΣU²) are recorded when the factors are made and an `ArgumentError` is
thrown on mismatch (shifted factors, other `U`, or `fv_set_params!` since).
"""
function fv_steady_adjoint(s::FVNewtonSolver, U, gU; params::FVParameterMap = FVViscosityField(),
                           θ = nothing, ws = nothing, finalize::Bool = true,
                           law_gradient = :contraction)
    _fv_require_isothermal_adjoint(s.prob)
    _fv_check_law_gradient(law_gradient)
    ws === nothing && (ws = fv_adjoint_workspace(s, params; chunk = fv_jacobian_chunk(s.asm)))
    ws.map isa typeof(params) ||
        throw(ArgumentError("workspace built for $(typeof(ws.map)), called with $(typeof(params))"))
    length(gU) == length(U) || throw(DimensionMismatch("gU must have length $(length(U))"))
    t0 = time_ns()
    finalize ? fv_newton_finalize!(s, U) : _fv_check_factors(s, U)
    _fv_check_transpose(s)
    t1 = time_ns()
    copyto!(ws.g, gU)
    lin_solve!(ws.λ, s.cache, ws.g; transpose = true)
    _fv_sync(U)
    t2 = time_ns()
    cot = _fv_param_cotangent!(ws, params, s, U, θ; law_gradient)
    t3 = time_ns()
    fv_residual!(s.R, U, s.prob)
    _, _, rinf = _fv_norms(s.norms, s.R)
    return cot, (; λ = ws.λ, residual_inf = rinf, t_finalize = (t1 - t0) / 1e9,
                 t_solve = (t2 - t1) / 1e9, t_param = (t3 - t2) / 1e9)
end

"""
    fv_steady_gradient(solver, U, θ, obj; params, ws = nothing, finalize = true,
                       law_gradient = :contraction)
        -> NamedTuple

Value and gradient of the objective `obj` ([`FVObjective`](@ref)) of the
converged steady FV state `U` with respect to the parameters `θ` of `params`
([`FVParameterMap`](@ref)); `solver.prob` must be the problem at `θ` (e.g. set
by [`fv_set_params!`](@ref)) and `U` its converged state ([`fv_newton!`](@ref)).

    d𝒥/dθ = ∂𝒥/∂θ − λᵀ ∂R/∂θ,   Jᵀ λ = ∂𝒥/∂U

`∂𝒥/∂U` and the explicit `∂𝒥/∂θ` come from host duals restricted to the
objective's support. Wall forces use local face stencils for state and
viscosity derivatives, and parameter chunks for other maps. The adjoint is
[`fv_steady_adjoint`](@ref) (one
refactorization of the Newton Jacobian at `U`, no new symbolic analysis, one
transposed solve). For [`FVLawParameters`](@ref), `law_gradient = :contraction`
uses one face-seeded residual traversal followed by local law pullbacks;
`:duals` retains `⌈P/chunk⌉` parameter-dual residual evaluations as a reference.
Other maps keep their parameter/colour dual evaluations. Explicit parameter
derivatives of wall force and torque objectives still use host parameter duals
in both modes; the selector changes only the residual cotangent.

Returns `(; J, grad, gU, explicit, s, λ, residual_inf, t_objective,
t_finalize, t_solve, t_param, t_explicit, t_total)` with `grad` a host vector
(`vec` of the field for [`FVViscosityField`](@ref)), `s = −λᵀ ∂R/∂θ` (as from
`fv_steady_adjoint`) and `explicit = ∂𝒥/∂θ`; times in seconds.
"""
function fv_steady_gradient(s::FVNewtonSolver, U, θ, obj::FVObjective;
                            params::FVParameterMap, ws = nothing, finalize::Bool = true,
                            law_gradient = :contraction)
    _fv_require_isothermal_adjoint(s.prob)
    _fv_check_law_gradient(law_gradient)
    t0 = time_ns()
    hprob = _fv_host_problem(s.prob)
    Uh = Vector{Float64}(Array(U))
    J, gU = _fv_objective_grad_u(obj, Uh, hprob)
    t1 = time_ns()
    cot, st = fv_steady_adjoint(s, U, gU; params, θ, ws, finalize, law_gradient)
    t2 = time_ns()
    θh = params isa FVViscosityField ? vec(Array(hprob.nu)) : Vector{Float64}(Array(θ))
    explicit = _fv_objective_grad_theta(obj, params, Uh, hprob, θh)
    grad = vec(Array(cot)) .+ explicit
    t3 = time_ns()
    return (; J, grad, gU, explicit, s = cot, λ = st.λ, residual_inf = st.residual_inf,
            t_objective = (t1 - t0) / 1e9, t_finalize = st.t_finalize, t_solve = st.t_solve,
            t_param = st.t_param, t_explicit = (t3 - t2) / 1e9, t_total = (t3 - t0) / 1e9)
end

"""
    fv_steady_tangent(solver, U, θ, δθ; params, ws = nothing, finalize = true) -> dU

Forward (tangent-linear) sensitivity of the converged state along the
parameter direction `δθ`: `J dU = −(∂R/∂θ) δθ`, with one directional dual
residual evaluation and one solve with the Newton factors. `dU` is a new
vector on the device of `U`. With `gU = ∂𝒥/∂U`, `gU ⋅ dU + (∂𝒥/∂θ) ⋅ δθ`
is the directional derivative of `𝒥` — the forward check of the adjoint.
"""
function fv_steady_tangent(s::FVNewtonSolver, U, θ, δθ; params::FVParameterMap,
                           ws = nothing, finalize::Bool = true)
    _fv_require_isothermal_adjoint(s.prob)
    ws === nothing && (ws = fv_adjoint_workspace(s, params; chunk = fv_jacobian_chunk(s.asm)))
    finalize ? fv_newton_finalize!(s, U) : _fv_check_factors(s, U)
    s.cache === nothing && throw(ArgumentError("the solver has no factorization"))
    prob = s.prob
    N = _fv_ws_chunk(ws)
    backend = KernelAbstractions.get_backend(U)
    if params isa FVViscosityField
        dnu = similar(prob.nu, Float64); copyto!(dnu, reshape(Vector{Float64}(Array(δθ)), size(dnu)))
        _fv_nu_direction_kernel!(backend, 256)(ws.buf.nu, prob.nu, dnu, Val(N); ndrange = length(dnu))
        pd = _fv_with(prob; nu = ws.buf.nu)
    else
        _fv_check_nparams(params, prob, θ)
        length(δθ) == length(θ) || throw(DimensionMismatch("δθ and θ differ in length"))
        θ = θ isa Array ? θ : Array(θ)
        δθ = δθ isa Array ? δθ : Array(δθ)
        D = ForwardDiff.Dual{FVParamTag,Float64,N}
        θd = D[D(Float64(θ[k]), ForwardDiff.Partials(ntuple(c -> c == 1 ? Float64(δθ[k]) : 0.0, Val(N))))
               for k in eachindex(θ)]
        pd = _fv_param_problem!(ws.buf, params, prob, θd)
    end
    fv_residual!(ws.Rd, U, pd; sync = false)
    _fv_partial1_kernel!(backend, 256)(ws.r, ws.Rd; ndrange = length(ws.r))
    KernelAbstractions.synchronize(backend)
    dU = similar(U, Float64)
    lin_solve!(dU, s.cache, ws.r)
    dU .*= -1
    return dU
end
