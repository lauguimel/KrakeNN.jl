"""
    FVEnergy(; alpha, sigma=0, sigma_nom=sigma, bottom=:adiabatic, top=:adiabatic,
               left=:adiabatic, right=:adiabatic, solid=:adiabatic, source=nothing,
               scheme=:linear_upwind, face_velocity=nothing, T_ref=nothing)

Opt-in steady temperature equation `div(u*T) - alpha*laplacian(T) =
2*sigma*nu*(S:S) + source` for [`fv_steady_problem`](@ref). `alpha` must be
positive and `sigma` nonnegative. `sigma_nom >= sigma` is the fixed final
heating coefficient used for stopping scales throughout a heating
continuation, including its zero-heating step.

Thermal boundary conditions follow the flow side codes of `fv_steady_problem`
(`bottom`/`top`/`left`/`right` = S/N/W/E):

- `:wall` side: a finite prescribed temperature `T_w` (half-cell diffusive flux
  `2*alpha*(T_P - T_w)`, plus the upwind convective flux of the prescribed
  wall velocity, zero for a no-slip wall) or `:adiabatic`.
- `:inlet` side: a finite inflow temperature `T_in` is required. The face
  carries the upwind convective flux of the prescribed inlet velocity
  (`T_in` for inflow, `T_P` for a reversed face) and the same half-cell
  Dirichlet diffusive flux `2*alpha*(T_P - T_in)`.
- `:outlet` side: zero normal temperature gradient (convective outflow); it
  must keep the `:adiabatic` placeholder. The face carries `F*T_P` with `F`
  the continuity outlet flux (`h*u_P` or `h*v_P`, outward), in either
  direction, and no diffusive flux.
- `:periodic` side: periodic transport, `:adiabatic` placeholder.
- `solid`: voxel-solid faces (masked walls, including a periodic wrap into a
  solid cell) are adiabatic (`:adiabatic`, default) or held at one finite
  temperature `T_s` for every solid face (`2*alpha*(T_P - T_s)`, no
  advection: solid faces are no-slip). A per-cell solid temperature field is
  not supported. Solid temperature rows are storage identities.

`source` is a constant, an `Nx × Ny` matrix, a cell-centre `(x,y)` function,
or `nothing`. `scheme` is `:linear_upwind` (second-order reconstruction,
first-order fallback when the upstream fluid cell is absent) or `:upwind`.
Thermal convection uses the continuity Rhie--Chow flux even for Stokes flow.
`face_velocity=(uf, vf)` (opt-in, default `nothing`) replaces that flux by
`h*uf`, `h*vf` for the temperature transport: `uf[i,j]` is the velocity of
the east face between cells `(i,j)` and `(i+1,j)` (periodic wrap), `vf[i,j]`
that of the north face between `(i,j)` and `(i,j+1)`, both `Nx × Ny` matrices.
It is meant for energy-only solves with an externally prescribed face field;
the momentum and continuity rows are not affected. An outlet face reads the
wrap slot (`uf[Nx,j]` for an E or W outlet, `vf[i,Ny]` for N or S), so two
opposite outlets are rejected with `face_velocity`; wall and inlet faces keep
the prescribed boundary velocity.
Heating uses the momentum Green--Gauss cell gradient and the existing cell
viscosity law. Viscosity regularisation affects viscosity only, never `S:S`.
The temperature unknown is stored as `theta = T - T_ref` (constant `T_ref`,
default: the mean of the prescribed side temperatures and of `solid` when finite). At a few hundred kelvin
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

Each connected fluid component, including periodic connections, must touch
a prescribed temperature (a Dirichlet wall, an inlet or a Dirichlet solid);
an outlet does not anchor.

Immersed walls ([`fv_immersed_walls`](@ref), [`fv_immersed_boundary`](@ref)):
`solid` then sets the wall temperature at the physical cut, either one finite
value or a function `(wall, x, y) -> T` evaluated at each cut point (`wall`
is the wall index of the geometry). The diffusive wall flux uses the velocity
ghost closure, `alpha*(theta_P - theta_G)` with `theta_G` built from the cell,
the opposite fluid cell and the wall value (C1: `2*alpha*(theta_P - theta_w)`),
so the Jacobian pattern is unchanged. The wall face convects the residual's
(M3-corrected) wall mass flux, upwind. `solid=:adiabatic` keeps only that
convective flux. [`fv_immersed_wall_heat`](@ref) sums exactly these fluxes.
"""
struct FVEnergy{T,B,S,W}
    alpha::T
    sigma::T
    sigma_nom::T
    boundary::B
    source::S
    scheme::Symbol
    T_ref::Union{Nothing,T}
    solid::W
end

function FVEnergy(; alpha::Real, sigma::Real=0, sigma_nom::Real=sigma, bottom=:adiabatic,
                    top=:adiabatic, left=:adiabatic, right=:adiabatic, solid=:adiabatic,
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
    solid === :adiabatic || (solid isa Real && isfinite(solid)) || solid isa Function ||
        throw(ArgumentError("solid thermal faces require a finite temperature, a wall function or :adiabatic"))
    T_ref === nothing || (T_ref isa Real && isfinite(T_ref)) ||
        throw(ArgumentError("T_ref must be nothing or a finite real"))
    a, s, nominal = promote(float(alpha), float(sigma), float(sigma_nom))
    return FVEnergy(a, s, nominal, boundary, source, scheme,
                    T_ref === nothing ? nothing : convert(typeof(a), T_ref), solid)
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
# Outlet face: the continuity outlet flux, or the prescribed wrap-slot face velocity.
@inline _fv_energy_outlet_flux(source, F, g, d, i, j) = F
@inline function _fv_energy_outlet_flux(f::_FVEnergyFaceFlux, F, g, d, i, j)
    ii = mod1(i + (d == 1) - (d == 2), g.Nx)
    jj = mod1(j + (d == 3) - (d == 4), g.Ny)
    return _fv_energy_face_flux(f, F, g, d, i, j, ii, jj)
end

"""Immersed-wall temperatures (`theta = T - T_ref` at each cut face, layout of
`qcut`), riding in the `source` slot like `_FVEnergyFaceFlux`; `getindex`
returns the wrapped cell source. `lo`/`hi`: absolute wall temperature range."""
struct _FVEnergyWall{S,A,T}
    source::S
    theta::A
    lo::T
    hi::T
end
Base.eltype(::Type{_FVEnergyWall{S,A,T}}) where {S,A,T} = promote_type(eltype(S), eltype(A))
@inline Base.getindex(f::_FVEnergyWall, i::Int, j::Int) = f.source[i, j]
function KernelAbstractions.Adapt.adapt_structure(to, f::_FVEnergyWall)
    adapt = KernelAbstractions.Adapt.adapt
    return _FVEnergyWall(adapt(to, f.source), adapt(to, f.theta), f.lo, f.hi)
end
_fv_adapt_energy_source(to, f::_FVEnergyWall) =
    _FVEnergyWall(_fv_adapt_energy_source(to, f.source), to(f.theta), f.lo, f.hi)
@inline _fv_energy_face_flux(f::_FVEnergyWall, F, g, d, i, j, ii, jj) =
    _fv_energy_face_flux(f.source, F, g, d, i, j, ii, jj)
@inline _fv_energy_outlet_flux(f::_FVEnergyWall, F, g, d, i, j) =
    _fv_energy_outlet_flux(f.source, F, g, d, i, j)
@inline _fv_energy_wall_theta(f::_FVEnergyWall, d, i, j) = @inbounds f.theta[d, i, j]

# Keep arrays separate from the isbits description passed to the KA kernel.
struct _FVEnergySpec{T}
    alpha::T
    sigma::T
    boundary::NTuple{4,T}
    dirichlet::NTuple{4,Bool}
    scheme::UInt8
    Tref::T
    solid::T
    solid_dirichlet::Bool
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
    solid::T
    solid_dirichlet::Bool
end
# Historical positional form: adiabatic solid faces.
_FVEnergyData(alpha::T, sigma::T, sigma_nom::T, boundary::NTuple{4,T}, dirichlet::NTuple{4,Bool},
              scheme::UInt8, source::A, Tref::T) where {T,A} =
    _FVEnergyData{T,A}(alpha, sigma, sigma_nom, boundary, dirichlet, scheme, source, Tref,
                       zero(T), false)

_fv_energy_args(::Nothing) = (nothing, nothing)
_fv_energy_args(e::_FVEnergyData) =
    (_FVEnergySpec(e.alpha, e.sigma, e.boundary, e.dirichlet, e.scheme, e.Tref,
                   e.solid, e.solid_dirichlet), e.source)
_fv_energy_eltype(::Nothing) = Bool
_fv_energy_eltype(e::_FVEnergyData) = promote_type(typeof(e.alpha), eltype(e.source))
_fv_adapt_energy(to, ::Nothing) = nothing
_fv_adapt_energy(to, e::_FVEnergyData) =
    _FVEnergyData(e.alpha, e.sigma, e.sigma_nom, e.boundary, e.dirichlet, e.scheme,
                  _fv_adapt_energy_source(to, e.source), e.Tref, e.solid, e.solid_dirichlet)

"""Prescribed temperatures of an energy model (Dirichlet/inlet sides, then a Dirichlet solid)."""
function _fv_energy_prescribed_temperatures(e)
    temperatures = [e.boundary[d] for d in 1:4 if e.dirichlet[d]]
    if e.solid_dirichlet
        if e.source isa _FVEnergyWall
            push!(temperatures, e.source.lo, e.source.hi)
        else
            push!(temperatures, e.solid)
        end
    end
    return temperatures
end
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

# Absolute wall temperature at every cut face (layout of qcut), plus the cut
# list and the (min, max, mean) of the prescribed values.
function _fv_immersed_wall_temperature(solid, b, g, ::Type{T}) where {T}
    qcut = b.qcut
    cuts = Tuple{Int,Int,Int}[]
    vals = Any[]
    for j in 1:g.Ny, i in 1:g.Nx, d in 1:4
        q = qcut[d, i, j]
        iszero(_fv_ib_primal(q)) && continue
        ex = (d == 1) - (d == 2)
        ey = (d == 3) - (d == 4)
        x = (i - 0.5 + _fv_ib_primal(q) * ex) * g.h
        y = (j - 0.5 + _fv_ib_primal(q) * ey) * g.h
        k = b.surface === nothing ? 1 : max(1, Int(b.surface.wall[d, i, j]))
        v = solid isa Function ? solid(k, x, y) : T(solid)
        isfinite(_fv_ib_primal(v)) ||
            throw(ArgumentError("immersed wall temperature must be finite; cell ($i, $j)"))
        push!(cuts, (d, i, j))
        push!(vals, v)
    end
    VT = isempty(vals) ? T : promote_type(T, map(typeof, vals)...)
    Tw = zeros(VT, 4, g.Nx, g.Ny)
    for (c, v) in zip(cuts, vals)
        Tw[c...] = v
    end
    prim = isempty(vals) ? T[zero(T)] : T[T(_fv_ib_primal(v)) for v in vals]
    return Tw, cuts, minimum(prim), maximum(prim), sum(prim) / length(prim)
end

function _fv_lower_energy(e::FVEnergy, g, tag, immersed, ::Type{T}) where {T}
    immersed === nothing && e.solid isa Function &&
        throw(ArgumentError("a solid wall-temperature function requires immersed geometry"))
    dirichlet = map(b -> b !== :adiabatic, e.boundary)
    boundary = map(b -> b === :adiabatic ? zero(T) : T(b), e.boundary)
    for d in 1:4
        bc = _fv_side_bc(g, d)
        side = _FV_SIDES[d]
        bc == FV_BC_PERIODIC && dirichlet[d] &&
            throw(ArgumentError("periodic thermal sides must keep the :adiabatic placeholder"))
        bc == FV_BC_OUTLET && dirichlet[d] &&
            throw(ArgumentError("outlet thermal side $side has a zero normal gradient; keep the :adiabatic placeholder"))
        bc == FV_BC_INLET && !dirichlet[d] &&
            throw(ArgumentError("inlet thermal side $side requires a finite inflow temperature"))
    end
    if e.source isa _FVEnergyFaceFlux &&
       ((g.bcE == FV_BC_OUTLET && g.bcW == FV_BC_OUTLET) ||
        (g.bcN == FV_BC_OUTLET && g.bcS == FV_BC_OUTLET))
        throw(ArgumentError("face_velocity cannot serve two opposite outlets (shared wrap slot)"))
    end
    solid_dirichlet = e.solid !== :adiabatic
    wall_data = immersed === nothing ? nothing :
        _fv_immersed_wall_temperature(solid_dirichlet ? e.solid : zero(T), immersed, g, T)
    solid_T = !solid_dirichlet ? zero(T) : e.solid isa Real ? T(e.solid) : T(wall_data[5])
    isfinite(solid_T) || throw(ArgumentError("solid temperature must remain finite in $T"))
    a, s, nominal = T(e.alpha), T(e.sigma), T(e.sigma_nom)
    isfinite(a) && a > 0 && isfinite(s) && s >= 0 && isfinite(nominal) && nominal >= s && all(isfinite, boundary) ||
        throw(ArgumentError("thermal coefficients must remain finite and alpha positive in $T"))
    src = _fv_lower_energy_source(e.source, g, T)
    all(isfinite, src isa _FVEnergyFaceFlux ? src.source : src) ||
        throw(ArgumentError("thermal source must be finite"))
    walls = [boundary[d] for d in 1:4 if dirichlet[d]]
    solid_dirichlet && push!(walls, solid_T)
    Tref = e.T_ref === nothing ? (isempty(walls) ? zero(T) : sum(walls) / T(length(walls))) : T(e.T_ref)
    isfinite(Tref) || throw(ArgumentError("T_ref must remain finite in $T"))
    if wall_data !== nothing
        Tw, cuts, lo, hi, _ = wall_data
        theta = zero(Tw)
        for c in cuts
            theta[c...] = Tw[c...] - Tref
        end
        src = _FVEnergyWall(src, theta, lo, hi)
    end
    data = _FVEnergyData(a, s, nominal, boundary, dirichlet,
                         e.scheme === :linear_upwind ? 0x01 : 0x00, src, Tref,
                         solid_T, solid_dirichlet)
    _fv_validate_energy_components(data, g, tag)
    return data
end

@inline function _fv_domain_face(g, i, j, d)
    return (d == 1 && i == g.Nx) || (d == 2 && i == 1) ||
           (d == 3 && j == g.Ny) || (d == 4 && j == 1)
end
# A non-periodic domain side face (wall or inlet when the face kind is Dirichlet).
# A Dirichlet face that is not a side face is a voxel-solid face.
@inline _fv_energy_side_face(g, i, j, d) =
    _fv_domain_face(g, i, j, d) && _fv_side_bc(g, d) != FV_BC_PERIODIC

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
                elseif kind == _FV_F_DIRICHLET
                    if _fv_energy_side_face(g, i, j, d)
                        energy.dirichlet[d] && (anchored = true)
                    else
                        energy.solid_dirichlet && (anchored = true)
                    end
                end
            end
        end
        anchored || throw(ArgumentError("fluid component at ($i0,$j0) has no prescribed temperature (Dirichlet wall, inlet or solid)"))
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
@inline function _fv_energy_heating(U, tag, nu, bvel, g, i, j, visc, sigma,
                                    qcut = nothing, wvel = nothing, closure = nothing)
    visc = _fv_bind_viscosity(visc, U, g)
    @inbounds tag[i, j] == FV_CELL_SOLID && return zero(eltype(U))
    gP = _fv_cell_grad(U, tag, bvel, g, i, j, qcut, wvel, closure)
    @inbounds return _fv_energy_heating_from_gradient(nu[i, j], visc, gP, i, j, sigma, g.h)
end

# Accumulation of the thermal row. `Val(false)` is the residual itself (the
# grouping of every sum is the historical one, so results are bit-identical);
# `Val(true)` adds the absolute value of each term instead, which is the
# magnitude of what cancels in the row (see `_fv_energy_roundoff_floor!`).
@inline _fv_eacc(::Val{false}, R, a) = R + a
@inline _fv_eacc(::Val{true}, R, a) = R + abs(a)
@inline _fv_eacc(::Val{false}, R, a, b) = R + (a + b)
@inline _fv_eacc(::Val{true}, R, a, b) = R + (abs(a) + abs(b))
# Diffusive flux a*(x - y): in magnitude mode its operands a*x and a*y are what
# cancel (each carries the storage rounding of its temperature), not |a*(x - y)|.
@inline _fv_ediff(::Val{false}, a, x, y) = a * (x - y)
@inline _fv_ediff(::Val{true}, a, x, y) = a * (abs(x) + abs(y))
@inline _fv_esolid(::Val{false}, TP, z) = TP + z
@inline _fv_esolid(::Val{true}, TP, z) = z

"""Integrated thermal row using exactly the continuity mass-flux expression."""
@inline _fv_energy_cell(U, tag, nu, bvel, g, i, j, visc, energy, source) =
    _fv_energy_cell(U, tag, nu, bvel, g, i, j, visc, energy, source, Val(false))

@inline _fv_energy_cell(U, tag, nu, bvel, g, i, j, visc, energy, source, mode::Val) =
    _fv_energy_cell(U, tag, nu, bvel, g, i, j, visc, energy, source, mode,
                    nothing, nothing, nothing, nothing)

"""Thermal flux of one immersed wall face, outward from cell `(i, j)`, in `theta`.

Convection uses the wall mass flux of the flow residual
(`_fv_wall_flux_components`, M3 lambda correction included), upwind:
`theta_P` on outflow, the wall value on inflow (`theta_P` when adiabatic).
Dirichlet diffusion uses the velocity ghost closure,
`alpha*(theta_P - theta_G)` (C1: `2*alpha*(theta_P - theta_w)`).
`Val(true)` returns the sum of term magnitudes (round-off floor).
"""
@inline function _fv_energy_immersed_face(U, bvel, g, i, j, d, qcut, wvel, closure,
                                          face_pair, energy, source, mode::Val)
    slot = _fv_scalar_slot(g.layout, 1)
    thP = _fv_val(U, g, i, j, slot)
    isx = d <= 2
    sx = (d == 1 || d == 3) ? 1 : -1
    uw, vw, _, _, _, uf, vf, _, _ = _fv_wall_data(U, bvel, g, i, j, d, qcut, wvel, closure)
    F = g.h * sx * (closure.mass != 2 ? (isx ? uf : vf) : (isx ? uw : vw))
    F = _fv_corrected_wall_mass(F, U, g, i, j, d, face_pair)
    z = zero(F) * zero(thP)
    if energy.solid_dirichlet
        thw = _fv_energy_wall_theta(source, d, i, j)
        R = _fv_eacc(mode, z, _fv_posp(F) * thP, _fv_negp(F) * thw)
        closure.ghost == 0 && return _fv_eacc(mode, R, _fv_ediff(mode, 2 * energy.alpha, thP, thw))
        oi = i - (d == 1) + (d == 2)
        oj = j - (d == 3) + (d == 4)
        @inbounds q = qcut[d, i, j]
        thG = _fv_ghost_value(_fv_val(U, g, oi, oj, slot), thP, thw, q, closure.ghost)
        return _fv_eacc(mode, R, _fv_ediff(mode, energy.alpha, thP, thG))
    end
    return _fv_eacc(mode, z, F * thP)
end

@inline function _fv_energy_cell(U, tag, nu, bvel, g, i, j, visc, energy, source, mode::Val,
                                 qcut, wvel, closure, face_pair)
    visc = _fv_bind_viscosity(visc, U, g)
    slot = _fv_scalar_slot(g.layout, 1)
    TP = _fv_val(U, g, i, j, slot)  # theta = T - T_ref: the state stores theta
    Tref = energy.Tref
    @inbounds nuP = nu[i, j]
    z = zero(TP) + zero(nuP) + zero(eltype(bvel)) + _fv_visc_zero(visc) +
        zero(energy.alpha) + zero(eltype(source)) + _fv_array_zero(qcut) + _fv_array_zero(wvel)
    @inbounds tag[i, j] == FV_CELL_SOLID && return _fv_esolid(mode, TP, z)
    thP = TP
    uP = _fv_val(U, g, i, j, 1); vP = _fv_val(U, g, i, j, 2)
    pP = _fv_val(U, g, i, j, 3)
    gP = _fv_cell_grad(U, tag, bvel, g, i, j, qcut, wvel, closure)
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
            gN = _fv_cell_grad(U, tag, bvel, g, ii, jj, qcut, wvel, closure)
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
            R = _fv_eacc(mode, R, _fv_posp(F) * positive, _fv_negp(F) * negative)
            R = _fv_eacc(mode, R, _fv_ediff(mode, energy.alpha, thP, thN))
        elseif kind == _FV_F_OUTLET
            # Zero normal gradient: the cell value leaves (or re-enters) with
            # the continuity outlet flux; no diffusive flux.
            F = _fv_energy_outlet_flux(source, g.h * sx * (isx ? uP : vP), g, d, i, j)
            R = _fv_eacc(mode, R, F * thP)
        elseif _fv_energy_side_face(g, i, j, d)
            # Wall or inlet side: upwind convection of the prescribed face
            # velocity, half-cell Dirichlet diffusion when a temperature is set.
            uw, vw = _fv_dirichlet_velocity(g, bvel, i, j, d)
            F = g.h * sx * (isx ? uw : vw)
            boundary_th = energy.dirichlet[d] ? energy.boundary[d] - Tref : thP
            R = _fv_eacc(mode, R, _fv_posp(F) * thP, _fv_negp(F) * boundary_th)
            energy.dirichlet[d] && (R = _fv_eacc(mode, R, _fv_ediff(mode, 2 * energy.alpha, thP, boundary_th)))
        elseif _fv_immersed_face(g, qcut, i, j, d)
            # Immersed wall face: ghost-closure Dirichlet (or adiabatic) + wall mass flux.
            R = R + _fv_energy_immersed_face(U, bvel, g, i, j, d, qcut, wvel, closure,
                                             face_pair, energy, source, mode)
        elseif energy.solid_dirichlet
            # Voxel-solid face (no-slip, no convection) held at the solid temperature.
            R = _fv_eacc(mode, R, _fv_ediff(mode, 2 * energy.alpha, thP, energy.solid - Tref))
        end
    end
    R = _fv_eacc(mode, R, -_fv_energy_heating_from_gradient(nuP, visc, gP, i, j, energy.sigma, g.h))
    @inbounds R = _fv_eacc(mode, R, -(g.h * g.h * source[i, j]))
    return R
end

@inline _fv_scalar_residual(energy::_FVEnergySpec, source, U, tag, nu, bvel, g, i, j, visc, scalar) =
    _fv_energy_cell(U, tag, nu, bvel, g, i, j, visc, energy, source)
@inline _fv_scalar_residual(energy::_FVEnergySpec, source, U, tag, nu, bvel, g, i, j, visc, scalar,
                            qcut, wvel, closure, face_pair) =
    _fv_energy_cell(U, tag, nu, bvel, g, i, j, visc, energy, source, Val(false),
                    qcut, wvel, closure, face_pair)

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

# ------------------------------------------------------------------ round-off floor
"""Multiplier of the per-cell round-off floor (fixed before any measurement, never tuned).

64 = 4 faces x (convective + diffusive term) x a generous 8 for the accumulated
rounding of a short sum of products of already-rounded operands.
"""
const FV_ENERGY_FLOOR_C = 64.0

@kernel function _fv_energy_floor_kernel!(Fl, @Const(U), @Const(tag), @Const(nu), @Const(bvel),
                                          g, visc, energy, @Const(source), C,
                                          @Const(qcut), @Const(wvel), closure, @Const(face_pair))
    i, j = @index(Global, NTuple)
    s = _fv_energy_cell(U, tag, nu, bvel, g, i, j, visc, energy, source, Val(true),
                        qcut, wvel, closure, face_pair)
    @inbounds Fl[fv_index(g, i, j, 0) + _fv_scalar_slot(g.layout, 1)] = C * eps(typeof(s)) * s
end

"""
    _fv_energy_roundoff_floor!(Fl, U, prob; C = FV_ENERGY_FLOOR_C, workgroup = 64) -> Fl

Write into the thermal slot of every cell of `Fl` (length `fv_nunknowns(prob)`,
same backend as `U`) the round-off floor `C * eps * sum(|terms|)` of the energy
row: the same kernel body as the residual, accumulating the absolute value of
each term (convective face fluxes, diffusive fluxes, wall terms, heating, source)
instead of their signed sum. Solid cells get 0. `eps` is the machine epsilon of
`eltype(Fl)`: the terms already carry the magnitude of the temperature.
"""
function _fv_energy_roundoff_floor!(Fl, U, prob::FVSteadyProblem2D; C::Real = FV_ENERGY_FLOOR_C,
                                    workgroup::Int = 64)
    prob.energy === nothing && throw(ArgumentError("round-off floor requires an energy model"))
    g = prob.grid
    backend = KernelAbstractions.get_backend(U)
    energy, energy_source = _fv_energy_args(prob.energy)
    qcut, wvel, closure = _fv_immersed_args(prob.immersed)
    _fv_energy_floor_kernel!(backend, workgroup)(Fl, U, prob.tag, prob.nu, prob.bvel, g, prob.visc,
        energy, energy_source, eltype(Fl)(C), qcut, wvel, closure, _fv_face_pair(prob.immersed);
        ndrange = (g.Nx, g.Ny))
    KernelAbstractions.synchronize(backend)
    return Fl
end

@kernel function _fv_floor_excess_kernel!(Pt, @Const(R), @Const(Fl), layout, ncell, per)
    t = @index(Global)
    excess = -Inf; fmax = 0.0; ratio = 0.0
    @inbounds for cell in ((t - 1) * per + 1):min(t * per, ncell)
        k = _fv_cell_index(layout, cell, 4)
        r = abs(R[k]); f = Fl[k]
        excess = max(excess, r - f)
        fmax = max(fmax, f)
        ratio = max(ratio, f > 0 ? r / f : (r > 0 ? Inf : 0.0))
    end
    @inbounds begin
        Pt[1, t] = excess; Pt[2, t] = fmax; Pt[3, t] = ratio
    end
end

"""
    _fv_energy_floor_check(nb, R, Fl, layout) -> (; passed, excess, max_floor, max_ratio)

Compare `|R|` with the floor `Fl` over all thermal rows (both full-layout
vectors). `passed` iff no cell exceeds its floor; `max_ratio = max |R|/floor`
(`Inf` where the floor is 0 and `R` is not). Uses the preallocated partials
of `nb`; no device allocation.
"""
function _fv_energy_floor_check(nb, R, Fl, layout)
    backend = KernelAbstractions.get_backend(R)
    _fv_floor_excess_kernel!(backend, 64)(nb.dev, R, Fl, layout, nb.ncell, nb.per;
                                         ndrange = size(nb.dev, 2))
    KernelAbstractions.synchronize(backend)
    H = nb.dev isa Array ? nb.dev : copyto!(nb.host, nb.dev)
    excess = -Inf; fmax = 0.0; ratio = 0.0
    @inbounds for t in axes(H, 2)
        excess = max(excess, H[1, t]); fmax = max(fmax, H[2, t]); ratio = max(ratio, H[3, t])
    end
    return (; passed = excess <= 0, excess, max_floor = fmax, max_ratio = ratio)
end
