"""
    FVEnergy(; alpha, sigma=0, sigma_nom=sigma, bottom=:adiabatic, top=:adiabatic,
               left=:adiabatic, right=:adiabatic, source=nothing,
               scheme=:linear_upwind, face_velocity=nothing, T_ref=nothing)

Opt-in steady temperature equation `div(u*T) - alpha*laplacian(T) =
2*sigma*nu*(S:S) + source` for [`fv_steady_problem`](@ref). `alpha` must be
positive and `sigma` nonnegative. `sigma_nom >= sigma` is the fixed final
heating coefficient used for stopping scales throughout a heating
continuation, including its zero-heating step. A wall accepts a finite prescribed
temperature or `:adiabatic`; periodic flow sides transport temperature
periodically and must keep the `:adiabatic` placeholder. Solid voxel faces
are adiabatic and solid temperature rows are storage identities.

`source` is a constant, an `Nx × Ny` matrix, a cell-centre `(x,y)` function,
or `nothing`. `scheme` is `:linear_upwind` (second-order reconstruction,
first-order fallback when the upstream fluid cell is absent) or `:upwind`.
Thermal convection uses the continuity Rhie--Chow flux even for Stokes flow.
`face_velocity=(uf, vf)` (opt-in, default `nothing`) replaces that flux by
`h*uf`, `h*vf` for the temperature transport: `uf[i,j]` is the velocity of
the east face between cells `(i,j)` and `(i+1,j)` (periodic wrap), `vf[i,j]`
that of the north face between `(i,j)` and `(i,j+1)`, both `Nx × Ny` matrices.
It is meant for energy-only solves with an externally prescribed face field;
the momentum and continuity rows are not affected.
Heating uses the momentum Green--Gauss cell gradient and the existing cell
viscosity law. Viscosity regularisation affects viscosity only, never `S:S`.
The temperature unknown is stored as `theta = T - T_ref` (constant `T_ref`,
default: the mean of the prescribed wall temperatures). At a few hundred kelvin
the spacing of absolute floating-point temperatures (6e-14 K at 468 K) is a
residual floor, and the convective term would carry `T_ref` times the
continuity residual; `theta` has neither problem. The solution does not
depend on `T_ref` (to round-off, for a divergence-free flux). Wall values, `fv_state`,
`fv_fields`, viscosity laws via [`FVStateTemperature`](@ref), the thermal
objectives and the public results all stay in absolute temperature; only the
raw state vector (`U[4:4:end]`, `fv_residual(U, prob)`) holds `theta`. `T_ref=0`
reproduces the absolute-temperature unknown.
Temperature feeds back into viscosity when the model explicitly selects
[`FVStateTemperature`](@ref); prescribed viscosity inputs remain one-way.

Only wall/periodic domains and voxel solids are supported. Each connected
fluid component, including periodic connections, must touch a prescribed
temperature. Immersed geometry and inlet/outlet boundaries are rejected.
"""
struct FVEnergy{T,B,S}
    alpha::T
    sigma::T
    sigma_nom::T
    boundary::B
    source::S
    scheme::Symbol
    T_ref::Union{Nothing,T}
end

function FVEnergy(; alpha::Real, sigma::Real=0, sigma_nom::Real=sigma, bottom=:adiabatic,
                    top=:adiabatic, left=:adiabatic, right=:adiabatic,
                    source=nothing, scheme::Symbol=:linear_upwind, face_velocity=nothing,
                    T_ref=nothing)
    if face_velocity !== nothing
        (face_velocity isa Tuple && length(face_velocity) == 2 &&
         all(f -> f isa AbstractMatrix, face_velocity)) ||
            throw(ArgumentError("face_velocity must be a tuple (uf, vf) of matrices"))
        source = _FVEnergyFaceFlux(source, face_velocity[1], face_velocity[2])
    end
    isfinite(alpha) && alpha > 0 || throw(ArgumentError("alpha must be finite and positive"))
    isfinite(sigma) && sigma >= 0 || throw(ArgumentError("sigma must be finite and nonnegative"))
    isfinite(sigma_nom) && sigma_nom >= sigma ||
        throw(ArgumentError("sigma_nom must be finite and at least sigma"))
    scheme in (:linear_upwind, :upwind) ||
        throw(ArgumentError("energy scheme must be :linear_upwind or :upwind"))
    boundary = (right, left, top, bottom)
    all(b -> b === :adiabatic || (b isa Real && isfinite(b)), boundary) ||
        throw(ArgumentError("thermal walls require a finite temperature or :adiabatic"))
    T_ref === nothing || (T_ref isa Real && isfinite(T_ref)) ||
        throw(ArgumentError("T_ref must be nothing or a finite real"))
    a, s, nominal = promote(float(alpha), float(sigma), float(sigma_nom))
    return FVEnergy(a, s, nominal, boundary, source, scheme,
                    T_ref === nothing ? nothing : convert(typeof(a), T_ref))
end

"""Opt-in carrier of prescribed face velocities, riding in the `source` slot.

`getindex` returns the wrapped cell source, so every other use of `source`
is unchanged; only `_fv_energy_cell` asks it for the convective face flux.
"""
struct _FVEnergyFaceFlux{S,A}
    source::S
    uf::A
    vf::A
end
Base.eltype(::Type{_FVEnergyFaceFlux{S,A}}) where {S,A} = eltype(S)
@inline Base.getindex(f::_FVEnergyFaceFlux, i::Int, j::Int) = f.source[i, j]
function KernelAbstractions.Adapt.adapt_structure(to, f::_FVEnergyFaceFlux)
    adapt = KernelAbstractions.Adapt.adapt
    return _FVEnergyFaceFlux(adapt(to, f.source), adapt(to, f.uf), adapt(to, f.vf))
end
_fv_adapt_energy_source(to, source) = to(source)
_fv_adapt_energy_source(to, f::_FVEnergyFaceFlux) =
    _FVEnergyFaceFlux(to(f.source), to(f.uf), to(f.vf))

# Default: the continuity (Rhie--Chow) flux F of the face. Outward flux of cell
# (i,j) through face d (1 E, 2 W, 3 N, 4 S); (ii,jj) is the neighbour across it.
@inline _fv_energy_face_flux(source, F, g, d, i, j, ii, jj) = F
@inline function _fv_energy_face_flux(f::_FVEnergyFaceFlux, F, g, d, i, j, ii, jj)
    @inbounds d == 1 && return g.h * f.uf[i, j]
    @inbounds d == 2 && return -g.h * f.uf[ii, jj]
    @inbounds d == 3 && return g.h * f.vf[i, j]
    @inbounds return -g.h * f.vf[ii, jj]
end

# Keep arrays separate from the isbits description passed to the KA kernel.
struct _FVEnergySpec{T}
    alpha::T
    sigma::T
    boundary::NTuple{4,T}
    dirichlet::NTuple{4,Bool}
    scheme::UInt8
    Tref::T
end
struct _FVEnergyData{T,A}
    alpha::T
    sigma::T
    sigma_nom::T
    boundary::NTuple{4,T}
    dirichlet::NTuple{4,Bool}
    scheme::UInt8
    source::A
    Tref::T
end

_fv_energy_args(::Nothing) = (nothing, nothing)
_fv_energy_args(e::_FVEnergyData) =
    (_FVEnergySpec(e.alpha, e.sigma, e.boundary, e.dirichlet, e.scheme, e.Tref), e.source)
_fv_energy_eltype(::Nothing) = Bool
_fv_energy_eltype(e::_FVEnergyData) = promote_type(typeof(e.alpha), eltype(e.source))
_fv_adapt_energy(to, ::Nothing) = nothing
_fv_adapt_energy(to, e::_FVEnergyData) =
    _FVEnergyData(e.alpha, e.sigma, e.sigma_nom, e.boundary, e.dirichlet, e.scheme,
                  _fv_adapt_energy_source(to, e.source), e.Tref)
_fv_lower_energy(::Nothing, g, tag, immersed, T) = nothing

function _fv_lower_energy_source(source, g, ::Type{T}) where {T}
    if source === nothing
        return zeros(T, g.Nx, g.Ny)
    elseif source isa Number
        return fill(T(source), g.Nx, g.Ny)
    elseif source isa AbstractMatrix
        size(source) == (g.Nx, g.Ny) || throw(DimensionMismatch("thermal source must match the grid"))
        return T.(source)
    end
    return T[source((i - 0.5) * g.h, (j - 0.5) * g.h) for i in 1:g.Nx, j in 1:g.Ny]
end
function _fv_lower_energy_source(f::_FVEnergyFaceFlux, g, ::Type{T}) where {T}
    size(f.uf) == (g.Nx, g.Ny) && size(f.vf) == (g.Nx, g.Ny) ||
        throw(DimensionMismatch("face velocities must match the grid"))
    all(isfinite, f.uf) && all(isfinite, f.vf) ||
        throw(ArgumentError("face velocities must be finite"))
    return _FVEnergyFaceFlux(_fv_lower_energy_source(f.source, g, T), T.(f.uf), T.(f.vf))
end

function _fv_lower_energy(e::FVEnergy, g, tag, immersed, ::Type{T}) where {T}
    immersed === nothing || throw(ArgumentError("energy does not support immersed geometry"))
    all(d -> _fv_side_bc(g, d) in (FV_BC_WALL, FV_BC_PERIODIC), 1:4) ||
        throw(ArgumentError("energy supports only wall and periodic flow boundaries"))
    dirichlet = map(b -> b !== :adiabatic, e.boundary)
    boundary = map(b -> b === :adiabatic ? zero(T) : T(b), e.boundary)
    for d in 1:4
        _fv_side_bc(g, d) == FV_BC_PERIODIC && dirichlet[d] &&
            throw(ArgumentError("periodic thermal sides must keep the :adiabatic placeholder"))
    end
    a, s, nominal = T(e.alpha), T(e.sigma), T(e.sigma_nom)
    isfinite(a) && a > 0 && isfinite(s) && s >= 0 && isfinite(nominal) && nominal >= s && all(isfinite, boundary) ||
        throw(ArgumentError("thermal coefficients must remain finite and alpha positive in $T"))
    src = _fv_lower_energy_source(e.source, g, T)
    all(isfinite, src isa _FVEnergyFaceFlux ? src.source : src) ||
        throw(ArgumentError("thermal source must be finite"))
    walls = [boundary[d] for d in 1:4 if dirichlet[d]]
    Tref = e.T_ref === nothing ? (isempty(walls) ? zero(T) : sum(walls) / T(length(walls))) : T(e.T_ref)
    isfinite(Tref) || throw(ArgumentError("T_ref must remain finite in $T"))
    data = _FVEnergyData(a, s, nominal, boundary, dirichlet,
                         e.scheme === :linear_upwind ? 0x01 : 0x00, src, Tref)
    _fv_validate_energy_components(data, g, tag)
    return data
end

@inline function _fv_domain_face(g, i, j, d)
    return (d == 1 && i == g.Nx) || (d == 2 && i == 1) ||
           (d == 3 && j == g.Ny) || (d == 4 && j == 1)
end

function _fv_validate_energy_components(energy, g, tag)
    seen = falses(g.Nx, g.Ny)
    stack = Tuple{Int,Int}[]
    for j0 in 1:g.Ny, i0 in 1:g.Nx
        (tag[i0, j0] == FV_CELL_SOLID || seen[i0, j0]) && continue
        seen[i0, j0] = true
        push!(stack, (i0, j0))
        anchored = false
        while !isempty(stack)
            i, j = pop!(stack)
            for d in 1:4
                kind, ii, jj = _fv_face_kind(g, tag, i, j, d)
                if kind == _FV_F_INT
                    if !seen[ii, jj]
                        seen[ii, jj] = true
                        push!(stack, (ii, jj))
                    end
                elseif _fv_domain_face(g, i, j, d) && energy.dirichlet[d]
                    anchored = true
                end
            end
        end
        anchored || throw(ArgumentError("fluid component at ($i0,$j0) has no thermal Dirichlet wall"))
    end
    return nothing
end

"""Two upwind face reconstructions; absent upstream fluid cells use first order."""
@inline function _fv_energy_face_temperature(U, tag, g, i, j, d, ii, jj, scheme)
    slot = _fv_scalar_slot(g.layout, 1)
    TP = _fv_val(U, g, i, j, slot)
    TN = _fv_val(U, g, ii, jj, slot)
    scheme == 0x00 && return TP, TN
    opposite = isodd(d) ? d + 1 : d - 1
    kindO, oi, oj = _fv_face_kind(g, tag, i, j, opposite)
    kindNN, ni, nj = _fv_face_kind(g, tag, ii, jj, d)
    positive = kindO == _FV_F_INT ? TP + (TP - _fv_val(U, g, oi, oj, slot)) / 2 : TP
    negative = kindNN == _FV_F_INT ? TN + (TN - _fv_val(U, g, ni, nj, slot)) / 2 : TN
    return positive, negative
end

"""Cell-integrated physical dissipation, without regularisation in the invariant."""
@inline function _fv_energy_heating_from_gradient(nuP, visc, gP, i, j, sigma, h)
    ux, uy, vx, vy = gP[1], gP[2], gP[3], gP[4]
    sxy = (uy + vx) / 2
    I1 = ux * ux + vy * vy + 2 * sxy * sxy
    nuc = _fv_wall_nu(_fv_heating_viscosity(visc), nuP, ux, uy, vx, vy, i, j)
    return 2 * sigma * nuc * I1 * h * h
end
@inline _fv_heating_viscosity(visc) = visc
@inline function _fv_energy_heating(U, tag, nu, bvel, g, i, j, visc, sigma)
    visc = _fv_bind_viscosity(visc, U, g)
    @inbounds tag[i, j] == FV_CELL_SOLID && return zero(eltype(U))
    gP = _fv_cell_grad(U, tag, bvel, g, i, j)
    @inbounds return _fv_energy_heating_from_gradient(nu[i, j], visc, gP, i, j, sigma, g.h)
end

"""Integrated thermal row using exactly the continuity mass-flux expression."""
@inline function _fv_energy_cell(U, tag, nu, bvel, g, i, j, visc, energy, source)
    visc = _fv_bind_viscosity(visc, U, g)
    slot = _fv_scalar_slot(g.layout, 1)
    TP = _fv_val(U, g, i, j, slot)  # theta = T - T_ref: the state stores theta
    Tref = energy.Tref
    @inbounds nuP = nu[i, j]
    z = zero(TP) + zero(nuP) + zero(eltype(bvel)) + _fv_visc_zero(visc) +
        zero(energy.alpha) + zero(eltype(source))
    @inbounds tag[i, j] == FV_CELL_SOLID && return TP + z
    thP = TP
    uP = _fv_val(U, g, i, j, 1); vP = _fv_val(U, g, i, j, 2)
    pP = _fv_val(U, g, i, j, 3)
    gP = _fv_cell_grad(U, tag, bvel, g, i, j)
    R = z
    for d in 1:4
        face_visc = _fv_face_viscosity(visc, d)
        isx = d <= 2
        sx = (d == 1 || d == 3) ? 1 : -1
        kind, ii, jj = _fv_face_kind(g, tag, i, j, d)
        if kind == _FV_F_INT
            uN = _fv_val(U, g, ii, jj, 1); vN = _fv_val(U, g, ii, jj, 2)
            pN = _fv_val(U, g, ii, jj, 3)
            thN = _fv_val(U, g, ii, jj, slot)
            gN = _fv_cell_grad(U, tag, bvel, g, ii, jj)
            @inbounds nuN = nu[ii, jj]
            dun = (uN - uP) / g.h
            dvn = (vN - vP) / g.h
            if isx
                nuf = _fv_face_nu(face_visc, nuP, nuN, sx * dun, (gP[2] + gN[2]) / 2,
                                  sx * dvn, (gP[4] + gN[4]) / 2, i, j, ii, jj)
            else
                nuf = _fv_face_nu(face_visc, nuP, nuN, (gP[1] + gN[1]) / 2, sx * dun,
                                  (gP[3] + gN[3]) / 2, sx * dvn, i, j, ii, jj)
            end
            F = _fv_interior_mass_flux(g.h, g.conv, uP, vP, pP, uN, vN, pN,
                                       gP, gN, nuf, isx, sx)
            F = _fv_energy_face_flux(source, F, g, d, i, j, ii, jj)
            positive, negative = _fv_energy_face_temperature(U, tag, g, i, j, d, ii, jj, energy.scheme)
            R += _fv_posp(F) * positive + _fv_negp(F) * negative
            R += energy.alpha * (thP - thN)
        else
            # Only domain walls can carry heat; voxel-solid faces are insulated.
            if _fv_domain_face(g, i, j, d)
                uw, vw = _fv_dirichlet_velocity(g, bvel, i, j, d)
                F = g.h * sx * (isx ? uw : vw)
                boundary_th = energy.dirichlet[d] ? energy.boundary[d] - Tref : thP
                R += _fv_posp(F) * thP + _fv_negp(F) * boundary_th
                energy.dirichlet[d] && (R += 2 * energy.alpha * (thP - boundary_th))
            end
        end
    end
    R -= _fv_energy_heating_from_gradient(nuP, visc, gP, i, j, energy.sigma, g.h)
    @inbounds R -= g.h * g.h * source[i, j]
    return R
end

@inline _fv_scalar_residual(energy::_FVEnergySpec, source, U, tag, nu, bvel, g, i, j, visc, scalar) =
    _fv_energy_cell(U, tag, nu, bvel, g, i, j, visc, energy, source)

"""
    fv_state(prob, u, v, p, temperature) -> Vector

Pack flow and temperature cell fields into an energy-active problem.
Temperature accepts a matrix, a cell-centre function, or a constant.
"""
function fv_state(prob::FVSteadyProblem2D, u, v, p, temperature)
    prob.energy === nothing && throw(ArgumentError("temperature requires an active energy model"))
    U = fv_state(prob, u, v, p)
    g = prob.grid
    temperature isa AbstractMatrix && size(temperature) != (g.Nx, g.Ny) &&
        throw(DimensionMismatch("temperature must match the grid"))
    Tref = prob.energy.Tref
    for j in 1:g.Ny, i in 1:g.Nx
        value = temperature isa Number ? temperature : temperature isa AbstractMatrix ?
            temperature[i, j] : temperature((i - 0.5) * g.h, (j - 0.5) * g.h)
        # The state stores theta = T - T_ref; solid rows are storage identities, kept raw.
        prob.tag[i, j] == FV_CELL_SOLID || (value -= Tref)
        U[fv_index(g, i, j, _fv_scalar_slot(g.layout, 1))] = value
    end
    return U
end

# Energy-active problems report absolute temperature T = theta + T_ref
# (fluid cells; solid storage rows are returned as stored).
function fv_fields(prob::FVSteadyProblem2D{<:FVGrid2D,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:_FVEnergyData},
                   U::AbstractVector)
    u, v, p, theta = invoke(fv_fields, Tuple{FVSteadyProblem2D,AbstractVector}, prob, U)
    iszero(prob.energy.Tref) && return u, v, p, theta
    tag = Array(prob.tag)
    return u, v, p, [tag[k] == FV_CELL_SOLID ? theta[k] : theta[k] + prob.energy.Tref
                     for k in CartesianIndices(theta)]
end
