# Opt-in steady temperature equation of the 3D steady FV problem (plan
# "solveur FV stationnaire 3D", increment I3, step 5): the 3D counterpart of
# fv_energy.jl, fv_thermal_law.jl, fv_thermal_continuation.jl and the thermal
# parts of fv_newton.jl / fv_thermal_adjoint.jl, written as additive methods on
# FVSteadyProblem3D. The 2D files are not changed; a 3D problem built with
# `energy = nothing` keeps the increment I1 path (no method below applies).
#
# Discretisation: the 2D energy row face by face, with face area h^2 and
# cell volume h^3 (6 faces, theta = T - T_ref as the 5th cell unknown):
#   sum_f [ F_f T_upwind + alpha h (theta_P - theta_N) ] + Dirichlet walls
#   2 alpha h (theta_P - theta_b) - 2 sigma nu S:S h^3 - h^3 q = 0,
# F_f the continuity (Rhie-Chow) flux of the 3D residual. For a z-extruded
# state (w = 0, z-uniform data, periodic z) every term is h times its 2D
# counterpart and the z faces add exact zeros, so the 3D energy row equals h
# times the 2D energy row (test/analytical/incns_fv3d_thermal_test.jl).

"""
    FVEnergy3D(; alpha, sigma=0, sigma_nom=sigma, boundary=NamedTuple(),
                 source=nothing, scheme=:linear_upwind, T_ref=nothing)

Opt-in steady temperature equation `div(u*T) - alpha*laplacian(T) =
2*sigma*nu*(S:S) + source` for the 3D [`fv_steady_problem`](@ref)`(Nx, Ny, Nz;
energy = FVEnergy3D(...))`; the 3D form of [`FVEnergy`](@ref), with the same
meaning of `alpha`, `sigma`, `sigma_nom`, `scheme` and `T_ref`.

`boundary` is a named tuple with keys among `E, W, N, S, T, B` (x+, x-, y+,
y-, z+, z-); each value is a finite prescribed wall temperature or
`:adiabatic` (the default of an absent key). Periodic sides must keep
`:adiabatic`. Voxel-solid faces are adiabatic; solid temperature rows are
storage identities. `source` is a constant, an `Nx × Ny × Nz` array, a
cell-centre `(x, y, z)` function, or `nothing`.

The temperature is the fifth cell unknown, stored as `theta = T - T_ref`
(`U[5:5:end]`); `fv_state(prob, u, v, w, p, T)`, `fv_fields`, the thermal
objectives and viscosity laws through [`FVStateTemperature`](@ref) use the
absolute temperature. Only wall and periodic flow sides are supported, and
each connected fluid component must touch a prescribed temperature.
"""
struct FVEnergy3D{T,B,S}
    alpha::T
    sigma::T
    sigma_nom::T
    boundary::B
    source::S
    scheme::Symbol
    T_ref::Union{Nothing,T}
end

function FVEnergy3D(; alpha::Real, sigma::Real = 0, sigma_nom::Real = sigma,
                    boundary::NamedTuple = NamedTuple(), source = nothing,
                    scheme::Symbol = :linear_upwind, T_ref = nothing)
    all(s -> s in _FV3_SIDES, keys(boundary)) ||
        throw(ArgumentError("3D thermal boundary keys must be among $(_FV3_SIDES)"))
    isfinite(alpha) && alpha > 0 || throw(ArgumentError("alpha must be finite and positive"))
    isfinite(sigma) && sigma >= 0 || throw(ArgumentError("sigma must be finite and nonnegative"))
    isfinite(sigma_nom) && sigma_nom >= sigma ||
        throw(ArgumentError("sigma_nom must be finite and at least sigma"))
    scheme in (:linear_upwind, :upwind) ||
        throw(ArgumentError("energy scheme must be :linear_upwind or :upwind"))
    walls = map(s -> get(boundary, s, :adiabatic), _FV3_SIDES)       # E, W, N, S, T, B
    all(b -> b === :adiabatic || (b isa Real && isfinite(b)), walls) ||
        throw(ArgumentError("thermal walls require a finite temperature or :adiabatic"))
    T_ref === nothing || (T_ref isa Real && isfinite(T_ref)) ||
        throw(ArgumentError("T_ref must be nothing or a finite real"))
    a, s, nominal = promote(float(alpha), float(sigma), float(sigma_nom))
    return FVEnergy3D(a, s, nominal, walls, source, scheme,
                      T_ref === nothing ? nothing : convert(typeof(a), T_ref))
end

# Isbits description passed to the kernels; arrays stay in _FVEnergyData3D.
struct _FVEnergySpec3D{T}
    alpha::T
    sigma::T
    boundary::NTuple{6,T}
    dirichlet::NTuple{6,Bool}
    scheme::UInt8
    Tref::T
end
struct _FVEnergyData3D{T,A}
    alpha::T
    sigma::T
    sigma_nom::T
    boundary::NTuple{6,T}
    dirichlet::NTuple{6,Bool}
    scheme::UInt8
    source::A
    Tref::T
end

_fv_energy_args(e::_FVEnergyData3D) =
    (_FVEnergySpec3D(e.alpha, e.sigma, e.boundary, e.dirichlet, e.scheme, e.Tref), e.source)
_fv_energy_eltype(e::_FVEnergyData3D) = promote_type(typeof(e.alpha), eltype(e.source))
_fv_adapt_energy(to, e::_FVEnergyData3D) =
    _FVEnergyData3D(e.alpha, e.sigma, e.sigma_nom, e.boundary, e.dirichlet, e.scheme,
                    to(e.source), e.Tref)

# The 2D builder rejects the 3D model with a clear message.
_fv_lower_energy(e::FVEnergy3D, g, tag, immersed, T) =
    throw(ArgumentError("FVEnergy3D belongs to 3D problems; use FVEnergy in 2D"))

# ------------------------------------------------------------------ lowering (host, once)
_fv3_lower_energy(::Nothing, g, tag, ::Type) = nothing
_fv3_lower_energy(e::FVEnergy, g, tag, ::Type) =
    throw(ArgumentError("3D problems take FVEnergy3D (six sides), not the 2D FVEnergy"))
_fv3_lower_energy(e, g, tag, ::Type) =
    throw(ArgumentError("energy must be nothing or FVEnergy3D, got $(typeof(e))"))

function _fv3_lower_energy_source(source, g, ::Type{T}) where {T}
    dims = (g.Nx, g.Ny, g.Nz)
    source === nothing && return zeros(T, dims)
    source isa Number && return fill(T(source), dims)
    if source isa AbstractArray
        size(source) == dims ||
            throw(DimensionMismatch("thermal source must be $(g.Nx) × $(g.Ny) × $(g.Nz)"))
        return T.(Array(source))
    end
    return T[source((i - 0.5) * g.h, (j - 0.5) * g.h, (k - 0.5) * g.h)
             for i in 1:g.Nx, j in 1:g.Ny, k in 1:g.Nz]
end

function _fv3_lower_energy(e::FVEnergy3D, g::FVGrid3D, tag, ::Type{T}) where {T}
    all(d -> _fv_side_bc(g, d) in (FV_BC_WALL, FV_BC_PERIODIC), 1:6) ||
        throw(ArgumentError("3D energy supports only wall and periodic flow boundaries"))
    dirichlet = map(b -> b !== :adiabatic, e.boundary)
    boundary = map(b -> b === :adiabatic ? zero(T) : T(b), e.boundary)
    for d in 1:6
        _fv_side_bc(g, d) == FV_BC_PERIODIC && dirichlet[d] &&
            throw(ArgumentError("periodic thermal sides must keep the :adiabatic placeholder"))
    end
    a, s, nominal = T(e.alpha), T(e.sigma), T(e.sigma_nom)
    isfinite(a) && a > 0 && isfinite(s) && s >= 0 && isfinite(nominal) && nominal >= s &&
        all(isfinite, boundary) ||
        throw(ArgumentError("thermal coefficients must remain finite and alpha positive in $T"))
    src = _fv3_lower_energy_source(e.source, g, T)
    all(isfinite, src) || throw(ArgumentError("thermal source must be finite"))
    walls = [boundary[d] for d in 1:6 if dirichlet[d]]
    Tref = e.T_ref === nothing ? (isempty(walls) ? zero(T) : sum(walls) / T(length(walls))) : T(e.T_ref)
    isfinite(Tref) || throw(ArgumentError("T_ref must remain finite in $T"))
    data = _FVEnergyData3D(a, s, nominal, boundary, dirichlet,
                           e.scheme === :linear_upwind ? 0x01 : 0x00, src, Tref)
    _fv3_validate_energy_components(data, g, tag)
    return data
end

# Is face d of cell (i, j, k) on the domain boundary?
@inline _fv3_domain_face(g, i, j, k, d) =
    (d == 1 && i == g.Nx) || (d == 2 && i == 1) || (d == 3 && j == g.Ny) ||
    (d == 4 && j == 1) || (d == 5 && k == g.Nz) || (d == 6 && k == 1)

# Every 6-connected fluid component (periodic sides wrapped) needs a
# prescribed-temperature wall, or the temperature level is undetermined.
function _fv3_validate_energy_components(energy, g, tag)
    seen = falses(g.Nx, g.Ny, g.Nz)
    stack = NTuple{3,Int}[]
    for k0 in 1:g.Nz, j0 in 1:g.Ny, i0 in 1:g.Nx
        (tag[i0, j0, k0] == FV_CELL_SOLID || seen[i0, j0, k0]) && continue
        seen[i0, j0, k0] = true
        push!(stack, (i0, j0, k0))
        anchored = false
        while !isempty(stack)
            i, j, k = pop!(stack)
            for d in 1:6
                kind, ii, jj, kk = _fv_face_kind(g, tag, i, j, k, d)
                if kind == _FV_F_INT
                    if !seen[ii, jj, kk]
                        seen[ii, jj, kk] = true
                        push!(stack, (ii, jj, kk))
                    end
                elseif _fv3_domain_face(g, i, j, k, d) && energy.dirichlet[d]
                    anchored = true
                end
            end
        end
        anchored || throw(ArgumentError("fluid component at ($i0,$j0,$k0) has no thermal Dirichlet wall"))
    end
    return nothing
end

const _FVEnergyProblem3D = FVSteadyProblem3D{<:FVGrid3D,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:_FVEnergyData3D}

# Lowering hook, as in 2D: an energy-active problem shifts the state-temperature
# inputs of its viscosity by T_ref (idempotent), through every constructor path
# (fv_steady_problem, fv_adapt, _fv_with).
function FVSteadyProblem3D(grid::FVGrid3D, tag, nu, bvel, src, visc, immersed,
                           energy::_FVEnergyData3D, flowrate)
    v = _fv_shift_viscosity(visc, energy.Tref)
    return FVSteadyProblem3D{typeof(grid),typeof(tag),typeof(nu),typeof(bvel),typeof(src),
                             typeof(v),typeof(immersed),typeof(energy),typeof(flowrate)}(
        grid, tag, nu, bvel, src, v, immersed, energy, flowrate)
end

# ------------------------------------------------------------------ state temperature in 3D laws
struct _FVTemperatureView3{T,U,G} <: AbstractArray{T,3}
    state::U
    grid::G
end
_FVTemperatureView3(U, g) = _FVTemperatureView3{eltype(U),typeof(U),typeof(g)}(U, g)
Base.size(v::_FVTemperatureView3) = (v.grid.Nx, v.grid.Ny, v.grid.Nz)
@inline Base.getindex(v::_FVTemperatureView3, i::Int, j::Int, k::Int) =
    @inbounds v.state[fv_index(v.grid, i, j, k, _fv_scalar_slot(v.grid.layout, 1))]

struct _FVShiftedTemperatureView3{T,U,G,R} <: AbstractArray{T,3}
    state::U
    grid::G
    Tref::R
end
_FVShiftedTemperatureView3(U, g, Tref) =
    _FVShiftedTemperatureView3{eltype(U),typeof(U),typeof(g),typeof(Tref)}(U, g, Tref)
Base.size(v::_FVShiftedTemperatureView3) = (v.grid.Nx, v.grid.Ny, v.grid.Nz)
@inline Base.getindex(v::_FVShiftedTemperatureView3, i::Int, j::Int, k::Int) =
    @inbounds v.state[fv_index(v.grid, i, j, k, _fv_scalar_slot(v.grid.layout, 1))] + v.Tref

@inline _fv_bind_law_input(::FVStateTemperature, U, g::FVGrid3D) = _FVTemperatureView3(U, g)
@inline _fv_bind_law_input(f::_FVShiftedStateTemperature, U, g::FVGrid3D) =
    _FVShiftedTemperatureView3(U, g, f.Tref)
# reference (case-scale) viscosity at a uniform temperature, read with 3 indices
@inline Base.getindex(v::_FVUniformTemperature, i::Int, j::Int, k::Int) = v.value

# Thermal GNF (FVGNFViscosity(...; temperature = FVStateTemperature())) on the
# 9-component face gradient; same invariant order as the isothermal 3D GNF.
@inline _fv_thermal_gnf_nu3(v::_FVThermalGNFViscosity, G, temperature) =
    _fv_thermal_gnf_invariant(v.base.law, _fv_strain_invariant3(G) + v.base.eps * v.base.eps / 2,
                              temperature)
@inline function _fv_face_nu3(v::_FVThermalGNFViscosity, nuP, nuN, G, i, j, k, ii, jj, kk)
    @inbounds temperature = (v.temperature[i, j, k] + v.temperature[ii, jj, kk]) / 2
    return _fv_thermal_gnf_nu3(v, G, temperature)
end
@inline function _fv_wall_nu3(v::_FVThermalGNFViscosity, nuP, G, i, j, k)
    @inbounds return _fv_thermal_gnf_nu3(v, G, v.temperature[i, j, k])
end
_fv3_state_viscosity(::_FVThermalGNFViscosity) = true
function _fv3_validate_viscosity(v::_FVThermalGNFViscosity, g, nu)
    _fv_nfields(g) >= 5 ||
        throw(ArgumentError("thermal GNF viscosity requires an active 3D energy model (FVEnergy3D)"))
    isbitstype(typeof(v.base.law)) || throw(ArgumentError("thermal GNF law must be isbits"))
    return nothing
end
_fv_validate_viscosity(v::_FVThermalGNFViscosity, g::FVGrid3D, nu) = _fv3_validate_viscosity(v, g, nu)

# The adjoint capture of an energy-active 3D problem has 6 face slots plus the
# cell-heating slot 7.
@inline _fv_heating_viscosity(v::FVLawViscosity{<:_FVLawCapture{7}}) = _fv_face_viscosity(v, 7)

# ------------------------------------------------------------------ energy row
@inline _fv3_val(U, g, i, j, k, comp) = @inbounds U[fv_index(g, i, j, k, comp)]

"""Two upwind face reconstructions; absent upstream fluid cells use first order."""
@inline function _fv3_energy_face_temperature(U, tag, g, i, j, k, d, ii, jj, kk, scheme)
    slot = _fv_scalar_slot(g.layout, 1)
    TP = _fv3_val(U, g, i, j, k, slot)
    TN = _fv3_val(U, g, ii, jj, kk, slot)
    scheme == 0x00 && return TP, TN
    opposite = isodd(d) ? d + 1 : d - 1
    kindO, oi, oj, ok = _fv_face_kind(g, tag, i, j, k, opposite)
    kindNN, ni, nj, nk = _fv_face_kind(g, tag, ii, jj, kk, d)
    positive = kindO == _FV_F_INT ? TP + (TP - _fv3_val(U, g, oi, oj, ok, slot)) / 2 : TP
    negative = kindNN == _FV_F_INT ? TN + (TN - _fv3_val(U, g, ni, nj, nk, slot)) / 2 : TN
    return positive, negative
end

"""Cell-integrated physical dissipation 2 sigma nu S:S h^3 (no regularisation in S:S)."""
@inline function _fv3_energy_heating_from_gradient(nuP, visc, gP, i, j, k, sigma, h)
    G = _fv3_first9(gP)
    I1 = _fv_strain_invariant3(G)
    nuc = _fv_wall_nu3(_fv_heating_viscosity(visc), nuP, G, i, j, k)
    return 2 * sigma * nuc * I1 * h * h * h
end

"""
    _fv3_energy_cell(U, tag, nu, bvel, g, i, j, k, visc, energy, source, mode) -> R_T

Integrated energy row of cell `(i, j, k)`, using exactly the continuity mass
flux of `_fv3_cell_residual` (same face viscosity, same Rhie-Chow expression).
`mode = Val(false)`: the residual; `Val(true)`: the sum of the magnitudes of its
terms (round-off floor). Branches depend on integer geometry only.
"""
@inline function _fv3_energy_cell(U, tag, nu, bvel, g, i, j, k, visc, energy, source, mode::Val)
    visc = _fv_bind_viscosity(visc, U, g)
    slot = _fv_scalar_slot(g.layout, 1)
    TP = _fv3_val(U, g, i, j, k, slot)       # theta = T - T_ref
    Tref = energy.Tref
    h = g.h
    @inbounds nuP = nu[i, j, k]
    z = zero(TP) + zero(nuP) + zero(eltype(bvel)) + _fv_visc_zero(visc) +
        zero(energy.alpha) + zero(eltype(source))
    @inbounds tag[i, j, k] == FV_CELL_SOLID && return _fv_esolid(mode, TP, z)
    thP = TP
    uP, vP, wP, pP = _fv3_cell(U, g, i, j, k)
    gP = _fv3_cell_grad(U, tag, bvel, g, i, j, k)
    h2 = h * h
    R = z
    for d in 1:6
        n = _fv3_axis(d)
        sx = _fv3_sign(d)
        vf = _fv_face_viscosity(visc, d)
        uPn = _fv3_pick3(n, uP, vP, wP)
        kind, ii, jj, kk = _fv_face_kind(g, tag, i, j, k, d)
        if kind == _FV_F_INT
            uN, vN, wN, pN = _fv3_cell(U, g, ii, jj, kk)
            gN = _fv3_cell_grad(U, tag, bvel, g, ii, jj, kk)
            @inbounds nuN = nu[ii, jj, kk]
            du = (uN - uP) / h + z
            dv = (vN - vP) / h + z
            dw = (wN - wP) / h + z
            Gt = _fv3_mean9(gP, gN)
            nuf = _fv_face_nu3(vf, nuP, nuN, _fv3_face_grad(Gt, n, sx, du, dv, dw),
                               i, j, k, ii, jj, kk)
            uNn = _fv3_pick3(n, uN, vN, wN)
            F = _fv3_interior_mass_flux(h, g.conv, uPn, uNn, pP, pN,
                                        gP[9 + n], gN[9 + n], nuf, sx)
            thN = _fv3_val(U, g, ii, jj, kk, slot)
            positive, negative = _fv3_energy_face_temperature(U, tag, g, i, j, k, d, ii, jj, kk,
                                                              energy.scheme)
            R = _fv_eacc(mode, R, _fv_posp(F) * positive, _fv_negp(F) * negative)
            R = _fv_eacc(mode, R, _fv_ediff(mode, energy.alpha * h, thP, thN))
        elseif _fv3_domain_face(g, i, j, k, d)
            # Only domain walls carry heat; voxel-solid faces are insulated (a
            # periodic wrap onto a solid cell reads the zero velocity of that side).
            uw, vw, ww = _fv3_dirichlet_velocity(g, bvel, i, j, k, d)
            F = h2 * sx * _fv3_pick3(n, uw, vw, ww)
            boundary_th = energy.dirichlet[d] ? (energy.boundary[d] - Tref) + z : thP + z
            R = _fv_eacc(mode, R, _fv_posp(F) * thP, _fv_negp(F) * boundary_th)
            energy.dirichlet[d] &&
                (R = _fv_eacc(mode, R, _fv_ediff(mode, 2 * energy.alpha * h, thP, boundary_th)))
        end
    end
    R = _fv_eacc(mode, R, -_fv3_energy_heating_from_gradient(nuP, visc, gP, i, j, k, energy.sigma, h))
    @inbounds R = _fv_eacc(mode, R, -(h2 * h * source[i, j, k]))
    return R
end

@inline function _fv3_energy_residual_cell!(R, U, tag, nu, bvel, src, g, visc, energy, energy_source,
                                            i, j, k)
    bound = _fv_bind_viscosity(visc, U, g)
    ru, rv, rw, rc = _fv3_cell_residual(U, tag, nu, bvel, src, g, i, j, k, bound)
    rt = _fv3_energy_cell(U, tag, nu, bvel, g, i, j, k, bound, energy, energy_source, Val(false))
    b = fv_index(g, i, j, k, 0)
    @inbounds begin
        R[b + 1] = ru
        R[b + 2] = rv
        R[b + 3] = rw
        R[b + 4] = rc
        R[b + _fv_scalar_slot(g.layout, 1)] = rt
    end
    return nothing
end

@kernel function _fv3_energy_residual_kernel!(R, @Const(U), @Const(tag), @Const(nu), @Const(bvel),
                                              @Const(src), g, visc, energy, @Const(energy_source))
    i, j, k = @index(Global, NTuple)
    _fv3_energy_residual_cell!(R, U, tag, nu, bvel, src, g, visc, energy, energy_source, i, j, k)
end

"""
    fv_residual!(R, U, prob::FVSteadyProblem3D; ...)   (energy active)

Energy-active 3D residual: one kernel writes the four flow rows (with the
viscosity bound to the state temperature) and the energy row of every cell,
then the optional mean-flow rows; same contract as the isothermal method.
"""
function fv_residual!(R, U, prob::_FVEnergyProblem3D; workgroup::Int = 64, sync::Bool = true,
                      mass_rows::Bool = true)
    g = prob.grid
    n = fv_nunknowns(prob)
    length(U) == n || throw(DimensionMismatch("length(U) = $(length(U)), expected $n"))
    length(R) == n || throw(DimensionMismatch("length(R) = $(length(R)), expected $n"))
    backend = KernelAbstractions.get_backend(U)
    energy, energy_source = _fv_energy_args(prob.energy)
    _fv3_energy_residual_kernel!(backend, workgroup)(R, U, prob.tag, prob.nu, prob.bvel, prob.src,
        g, prob.visc, energy, energy_source; ndrange = (g.Nx, g.Ny, g.Nz))
    _fv_flow_residual!(R, U, prob, backend, workgroup; constraint = mass_rows)
    sync && KernelAbstractions.synchronize(backend)
    return R
end

function fv_residual(U, prob::_FVEnergyProblem3D)
    T = promote_type(eltype(U), eltype(prob.nu), eltype(prob.bvel), eltype(prob.src),
                     _fv_visc_eltype(prob.visc), _fv_energy_eltype(prob.energy))
    R = similar(U, T)
    return fv_residual!(R, U, prob)
end

# ------------------------------------------------------------------ Jacobian pattern
"""
    fv_jacobian_pattern(prob::FVSteadyProblem3D)   (energy active)

Isothermal 3D pattern plus the energy rows: temperature on the star and the
second neighbours along each axis (linear upwind), velocity on the star
(mass flux and heating gradient) widened by the face-strain stencil for a
state-dependent viscosity, pressure as in continuity (Rhie-Chow); and, when
the viscosity reads [`FVStateTemperature`](@ref), temperature on the star of
the momentum and unpinned continuity rows. Independent of `sigma`.
"""
function fv_jacobian_pattern(prob::_FVEnergyProblem3D)
    g = prob.grid
    tag = Array(prob.tag)
    n = fv_nunknowns(prob)
    I = Int[]; J = Int[]
    law = _fv3_state_viscosity(prob.visc)
    thermal_law = _fv_has_state_temperature(prob.visc)
    sizehint!(I, (law ? 80 : 50) * n); sizehint!(J, (law ? 80 : 50) * n)
    slot = _fv_scalar_slot(g.layout, 1)
    add(row, comp, cells) = for c in cells
        c[1] == 0 && continue
        push!(I, row); push!(J, fv_index(g, c[1], c[2], c[3], comp))
    end
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        _fv3_push_newtonian!(I, J, g, tag, i, j, k)
        law && _fv3_push_viscosity!(I, J, g, tag, i, j, k)
        tag[i, j, k] == FV_CELL_SOLID && continue
        s = _fv3_newtonian_dependencies(g, tag, i, j, k)
        star = (s.P, s.nbs...)
        if thermal_law
            for a in 1:3
                add(fv_index(g, i, j, k, a), slot, star)
            end
            tag[i, j, k] == FV_CELL_PINNED || add(fv_index(g, i, j, k, 4), slot, star)
        end
        row = fv_index(g, i, j, k, slot)
        add(row, slot, s.pres)
        for a in 1:3
            add(row, a, star)
        end
        add(row, 4, s.pres)
        law || continue
        P = s.P
        for d in 1:6
            kind, ii, jj, kk = _fv_face_kind(g, tag, i, j, k, d)
            kind == _FV_F_INT || continue
            td = _fv3_tangential_dirs(d)
            N = (ii, jj, kk)
            cells = (P, N, map(e -> _fv3_nb(g, tag, P, e), td)..., map(e -> _fv3_nb(g, tag, N, e), td)...)
            for a in 1:3
                add(row, a, cells)
            end
        end
    end
    _fv_flow_pattern!(I, J, prob)
    return sparse(I, J, trues(length(I)), n, n, |)
end

# ------------------------------------------------------------------ round-off floor
"""Multiplier of the 3D per-cell round-off floor (fixed before any measurement, never tuned).

96 = 6 faces x (convective + diffusive term) x the same generous 8 as the 2D
[`FV_ENERGY_FLOOR_C`](@ref) for the rounding of a short sum of products.
"""
const FV_ENERGY_FLOOR_C3 = 96.0

@kernel function _fv3_energy_floor_kernel!(Fl, @Const(U), @Const(tag), @Const(nu), @Const(bvel),
                                           g, visc, energy, @Const(source), C)
    i, j, k = @index(Global, NTuple)
    s = _fv3_energy_cell(U, tag, nu, bvel, g, i, j, k, visc, energy, source, Val(true))
    @inbounds Fl[fv_index(g, i, j, k, _fv_scalar_slot(g.layout, 1))] = C * eps(typeof(s)) * s
end

function _fv_energy_roundoff_floor!(Fl, U, prob::FVSteadyProblem3D; C::Real = FV_ENERGY_FLOOR_C3,
                                    workgroup::Int = 64)
    prob.energy === nothing && throw(ArgumentError("round-off floor requires an energy model"))
    g = prob.grid
    backend = KernelAbstractions.get_backend(U)
    energy, energy_source = _fv_energy_args(prob.energy)
    _fv3_energy_floor_kernel!(backend, workgroup)(Fl, U, prob.tag, prob.nu, prob.bvel, g, prob.visc,
        energy, energy_source, eltype(Fl)(C); ndrange = (g.Nx, g.Ny, g.Nz))
    KernelAbstractions.synchronize(backend)
    return Fl
end

@kernel function _fv3_floor_excess_kernel!(Pt, @Const(R), @Const(Fl), layout, ncell, per)
    t = @index(Global)
    excess = -Inf; fmax = 0.0; ratio = 0.0
    @inbounds for cell in ((t - 1) * per + 1):min(t * per, ncell)
        q = _fv_cell_index(layout, cell, _fv_scalar_slot(layout, 1))
        r = abs(R[q]); f = Fl[q]
        excess = max(excess, r - f)
        fmax = max(fmax, f)
        ratio = max(ratio, f > 0 ? r / f : (r > 0 ? Inf : 0.0))
    end
    @inbounds begin
        Pt[1, t] = excess; Pt[2, t] = fmax; Pt[3, t] = ratio
    end
end

function _fv_energy_floor_check(nb, R, Fl, layout::FVFieldLayout3D)
    backend = KernelAbstractions.get_backend(R)
    _fv3_floor_excess_kernel!(backend, 64)(nb.dev, R, Fl, layout, nb.ncell, nb.per;
                                          ndrange = size(nb.dev, 2))
    KernelAbstractions.synchronize(backend)
    H = nb.dev isa Array ? nb.dev : copyto!(nb.host, nb.dev)
    excess = -Inf; fmax = 0.0; ratio = 0.0
    @inbounds for t in axes(H, 2)
        excess = max(excess, H[1, t]); fmax = max(fmax, H[2, t]); ratio = max(ratio, H[3, t])
    end
    return (; passed = excess <= 0, excess, max_floor = fmax, max_ratio = ratio)
end

# ------------------------------------------------------------------ thermal Newton (3D scales)
@kernel function _fv3_temperature_increment_kernel!(Pt, @Const(dU), layout, ncell, per)
    t = @index(Global)
    value = 0.0
    @inbounds for cell in ((t - 1) * per + 1):min(t * per, ncell)
        value = max(value, abs(dU[_fv_cell_index(layout, cell, _fv_scalar_slot(layout, 1))]))
    end
    @inbounds Pt[1, t] = value
end
function _fv_temperature_increment(nb::_FVNormBuffers{<:Any,<:Any,<:FVFieldLayout3D}, dU)
    backend = KernelAbstractions.get_backend(dU)
    _fv3_temperature_increment_kernel!(backend, 64)(nb.dev, dU, nb.layout, nb.ncell, nb.per;
                                                   ndrange = size(nb.dev, 2))
    KernelAbstractions.synchronize(backend)
    H = nb.dev isa Array ? nb.dev : copyto!(nb.host, nb.dev)
    return maximum(view(H, 1, :))
end

"""Pass-3 physical scales of a 3D energy problem, from case data only.

Same definitions as the 2D `_fv_energy_scales` (H = Ny h, nu_ref at rest at
the coldest imposed wall temperature, U_c from wall speed, body force and
flow-rate target, DeltaT with a one-kelvin floor), with the 3D cell measure:
momentum nu_ref U_c h^3/H^2, continuity U_c h^2, energy alpha DeltaT h^3/H^2.
"""
function _fv_energy_scales(prob::FVSteadyProblem3D)
    p = prob.nu isa Array ? prob : fv_adapt(Array, prob)
    g = p.grid; e = p.energy; H = g.Ny * g.h
    temperatures = [e.boundary[d] for d in 1:6 if e.dirichlet[d]]
    reference_visc = _fv_reference_viscosity(p.visc, minimum(temperatures))
    G0 = ntuple(_ -> 0.0, Val(9))
    nu_sum = 0.0; count = 0; force = 0.0
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        p.tag[i, j, k] == FV_CELL_SOLID && continue
        nu_sum += _fv_wall_nu3(reference_visc, p.nu[i, j, k], G0, i, j, k)
        count += 1
        force = max(force, hypot(p.src[fv_index(g, i, j, k, 1)], p.src[fv_index(g, i, j, k, 2)],
                                 p.src[fv_index(g, i, j, k, 3)]))
    end
    nu_ref = _fv_reference_nu(p.visc, nu_sum / count)
    isfinite(nu_ref) && nu_ref > 0 ||
        throw(ArgumentError("energy scales require positive finite reference viscosity"))
    wall = maximum(hypot(p.bvel[1, l1, l2, d], p.bvel[2, l1, l2, d], p.bvel[3, l1, l2, d])
                   for l1 in axes(p.bvel, 2), l2 in axes(p.bvel, 3), d in 1:6)
    target = p.flowrate === nothing ? 0.0 : abs(p.flowrate.target)
    Uc = max(wall, force * H^2 / (12nu_ref), target, 1e-12)
    spread = maximum(temperatures) - minimum(temperatures)
    deltaT = max(e.sigma_nom * nu_ref * Uc^2 / e.alpha, spread, 1.0)
    momentum = nu_ref * Uc * g.h^3 / H^2
    continuity = Uc * g.h^2
    energy = e.alpha * deltaT * g.h^3 / H^2
    equations = (momentum, momentum, momentum, continuity, energy)
    p.flowrate === nothing || (equations = (equations..., Uc))
    return (; Uc, nu_ref, deltaT, equations)
end

# ------------------------------------------------------------------ state, fields
"""
    fv_state(prob::FVSteadyProblem3D, u, v, w, p, temperature) -> Vector

Pack flow and absolute-temperature cell fields of an energy-active 3D problem
(`Nx × Ny × Nz` arrays, cell-centre `(x, y, z)` functions or constants). Fluid
rows store `theta = T - T_ref`; solid rows are storage identities, kept raw.
"""
function fv_state(prob::FVSteadyProblem3D, u, v, w, p, temperature)
    prob.energy === nothing && throw(ArgumentError("temperature requires an active energy model"))
    lift(f) = f isa Number ? ((x, y, z) -> f) : f
    U = fv_state(prob, lift(u), lift(v), lift(w), lift(p))
    g = prob.grid
    temperature isa AbstractArray && size(temperature) != (g.Nx, g.Ny, g.Nz) &&
        throw(DimensionMismatch("temperature must be $(g.Nx) × $(g.Ny) × $(g.Nz)"))
    Tref = prob.energy.Tref
    tag = Array(prob.tag)
    slot = _fv_scalar_slot(g.layout, 1)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        value = temperature isa Number ? temperature : temperature isa AbstractArray ?
            temperature[i, j, k] : temperature((i - 0.5) * g.h, (j - 0.5) * g.h, (k - 0.5) * g.h)
        tag[i, j, k] == FV_CELL_SOLID || (value -= Tref)
        U[fv_index(g, i, j, k, slot)] = value
    end
    return U
end

"""
    fv_fields(prob::FVSteadyProblem3D, U) -> (u, v, w, p, T)   (energy active)

Host `Nx × Ny × Nz` copies; `T` is absolute on fluid cells (`theta + T_ref`),
solid storage rows are returned as stored.
"""
function fv_fields(prob::_FVEnergyProblem3D, U::AbstractVector)
    g = prob.grid
    length(U) == fv_nunknowns(prob) || throw(DimensionMismatch("state length does not match problem"))
    A = reshape(Array(U)[1:fv_nunknowns(g)], _fv_nfields(g), g.Nx, g.Ny, g.Nz)
    u = A[1, :, :, :]; v = A[2, :, :, :]; w = A[3, :, :, :]; p = A[4, :, :, :]
    theta = A[_fv_scalar_slot(g.layout, 1), :, :, :]
    tag = Array(prob.tag); Tref = prob.energy.Tref
    T = [tag[c] == FV_CELL_SOLID ? theta[c] : theta[c] + Tref for c in CartesianIndices(theta)]
    return u, v, w, p, T
end

# ------------------------------------------------------------------ heating continuation
function _fv_heating_problem(p::FVSteadyProblem3D, sigma)
    e = p.energy
    energy = _FVEnergyData3D(e.alpha, typeof(e.sigma)(sigma), e.sigma_nom,
                             e.boundary, e.dirichlet, e.scheme, e.source, e.Tref)
    return _fv_with(p; energy)
end

"""
    fv_thermal_continuation(prob::FVSteadyProblem3D; U0=nothing, initial_step=0.25,
                            max_halvings=5, maxit=30, to=Array,
                            linsolve=CPUBackendTag(), chunk=4, verbose=false)

3D form of the heating continuation (same contract as the 2D method: start at
zero heating from the isothermal state of the common imposed wall temperature,
then steps towards `sigma_nom` with at most five halvings). The result also
carries the `w` field.
"""
function fv_thermal_continuation(prob::FVSteadyProblem3D; U0 = nothing,
        initial_step::Real = 0.25, max_halvings::Int = 5, maxit::Int = 30,
        singular_iterations::Int = 12, to = Array,
        linsolve::LinearSolveBackend = CPUBackendTag(), chunk::Int = 4, verbose::Bool = false)
    prob.energy === nothing && throw(ArgumentError("heating continuation requires FVEnergy3D"))
    0 < initial_step <= 1 || throw(ArgumentError("initial_step must lie in (0,1]"))
    0 <= max_halvings <= 5 || throw(ArgumentError("max_halvings must lie in 0:5"))
    1 <= maxit <= 30 || throw(ArgumentError("maxit must lie in 1:30"))
    singular_iterations > 0 || throw(ArgumentError("singular_iterations must be positive"))
    p = prob.nu isa Array ? prob : fv_adapt(Array, prob)
    e = p.energy
    walls = [e.boundary[d] for d in 1:6 if e.dirichlet[d]]
    Tw = first(walls)
    all(==(Tw), walls) || throw(ArgumentError("isothermal continuation start requires equal imposed temperatures"))
    all(iszero, e.source) || throw(ArgumentError("isothermal continuation start requires zero independent energy source"))
    scales = _fv_energy_scales(p)
    initial = U0 === nothing ? _fv_initial_force!(zeros(Float64, fv_nunknowns(p)), p) : Float64.(Array(U0))
    length(initial) == fv_nunknowns(p) || throw(DimensionMismatch("U0 does not match the problem"))
    slot = _fv_scalar_slot(p.grid.layout, 1)
    for cell in eachindex(p.tag)
        initial[_fv_cell_index(p.grid, cell, slot)] = p.tag[cell] == FV_CELL_SOLID ? 0.0 : Tw - e.Tref
    end
    s = fv_newton_solver(_fv_heating_problem(p, 0); to, linsolve, chunk)
    U = to(initial); saved = copy(U)
    x = to([sin(Float64(k)) + cos(sqrt(2.0) * k) for k in eachindex(initial)])
    y = similar(U); z = similar(U)
    history = NamedTuple[]; attempts = NamedTuple[]
    last_sigma = nothing; last_failure = nothing; halvings = 0
    nominal = Float64(e.sigma_nom); step = initial_step * nominal; trial_sigma = 0.0
    reason = :initial_failure
    while true
        s.prob = _fv_heating_problem(s.prob, trial_sigma)
        copyto!(U, saved)
        attempt = _fv_continuation_attempt!(s, U, scales; maxit, verbose)
        estimate = NaN
        if attempt.converged
            try
                fv_newton_finalize!(s, U)
                estimate = _fv_smallest_singular_estimate!(s, x, y, z; iterations = singular_iterations)
            catch error
                _fv_continuation_failure(error) || rethrow()
                attempt = (; converged = false, reason = :singular_factorization, stats = attempt.stats,
                           gate = :none, error = sprint(showerror, error))
            end
        end
        push!(attempts, (; sigma = trial_sigma, step, halvings, attempt...))
        if attempt.converged
            push!(history, (; sigma = trial_sigma, singular_min = estimate, gate = attempt.gate,
                            stats = attempt.stats))
            copyto!(saved, U); last_sigma = trial_sigma
            if trial_sigma >= nominal
                reason = :converged
                break
            end
            trial_sigma = min(nominal, last_sigma + step)
        else
            last_failure = attempt.reason
            last_sigma === nothing && break
            if halvings >= max_halvings
                reason = :step_halving_limit
                break
            end
            halvings += 1; step /= 2
            trial_sigma = min(nominal, last_sigma + step)
        end
    end
    copyto!(U, saved)
    if last_sigma !== nothing
        s.prob = _fv_heating_problem(s.prob, last_sigma)
        reason == :converged || fv_newton_finalize!(s, U)
    end
    u, v, w, pressure, temperature = fv_fields(p, U)
    return (; converged = reason == :converged, reason, last_sigma, last_failure, halvings,
            history, attempts, gates = [h.gate for h in history], U, u, v, w, p = pressure,
            T = temperature, solver = s, thermal_scales = scales)
end

# ------------------------------------------------------------------ adjoint through T
# Law parameters of an energy-active 3D problem: the full state transpose
# already carries temperature; the parameter pullback captures 6 face
# viscosities plus the cell-heating viscosity (slot 7) in one traversal.
_fv_require_adjoint_map(prob::_FVLawProblem3D, ::FVLawParameters) = nothing
_fv_law_problem_slots(p::_FVEnergyProblem3D) = Val(7)

@kernel function _fv3_energy_law_capture_kernel!(inputs, sensitivity, @Const(U), @Const(lambda),
                                                 @Const(tag), @Const(nu), @Const(bvel), @Const(src),
                                                 g, visc, energy, @Const(energy_source),
                                                 ::Val{N}) where {N}
    i, j, k = @index(Global, NTuple)
    c = i + (j - 1) * g.Nx + (k - 1) * g.Nx * g.Ny
    @inbounds for d in 1:N, a in 1:law_ninputs(visc.law)
        inputs[c, d, a] = 0.0
    end
    capture = _FVLawCapture(visc.law, inputs, c, 0, Val(N))
    v = _fv_bind_viscosity(FVLawViscosity(capture, visc.theta, visc.fields, visc.eps), U, g)
    ru, rv, rw, rc = _fv3_cell_residual(U, tag, nu, bvel, src, g, i, j, k, v)
    rt = _fv3_energy_cell(U, tag, nu, bvel, g, i, j, k, v, energy, energy_source, Val(false))
    pu, pv = ForwardDiff.partials(ru), ForwardDiff.partials(rv)
    pw, pc = ForwardDiff.partials(rw), ForwardDiff.partials(rc)
    pt = ForwardDiff.partials(rt)
    b = fv_index(g, i, j, k, 0)
    slot = _fv_scalar_slot(g.layout, 1)
    @inbounds for d in 1:N
        sensitivity[c, d] = -(lambda[b + 1] * pu[d] + lambda[b + 2] * pv[d] +
                              lambda[b + 3] * pw[d] + lambda[b + 4] * pc[d] +
                              lambda[b + slot] * pt[d])
    end
end

function _fv_law_capture!(backend, w, U, λ, prob::_FVEnergyProblem3D, v)
    g = prob.grid
    energy, energy_source = _fv_energy_args(prob.energy)
    _fv3_energy_law_capture_kernel!(backend, 64)(w.inputs, w.sensitivity, U, λ, prob.tag, prob.nu,
        prob.bvel, prob.src, g, v, energy, energy_source, Val(7); ndrange = (g.Nx, g.Ny, g.Nz))
    return nothing
end

# ------------------------------------------------------------------ thermal objectives (3D)
function _fv_temperature_indices(prob::FVSteadyProblem3D)
    prob.energy === nothing && throw(ArgumentError("temperature objectives require an active energy model"))
    g = prob.grid
    cells = findall(!=(FV_CELL_SOLID), vec(Array(prob.tag)))
    isempty(cells) && throw(ArgumentError("temperature objectives require fluid cells"))
    return [_fv_cell_index(g, c, _fv_scalar_slot(g.layout, 1)) for c in cells], cells
end

"""
    fv_mean_temperature(prob::FVSteadyProblem3D; T_wall, target=0) -> FVStateObs

3D form of the squared mean fluid-temperature-rise misfit
`0.5 * (mean_fluid(T - T_wall) - target)^2` (solid cells excluded).
"""
function fv_mean_temperature(prob::FVSteadyProblem3D; T_wall::Real, target::Real = 0)
    isfinite(T_wall) && isfinite(target) || throw(ArgumentError("T_wall and target must be finite"))
    index, _ = _fv_temperature_indices(prob)
    return FVStateObs(reshape(index, :, 1), fill(inv(Float64(length(index))), length(index), 1),
                      [Float64(T_wall + target - prob.energy.Tref)], 1.0)
end

function _fv3_temperature_data(value, prob, cells, name)
    if value isa Real
        data = fill(Float64(value), length(cells))
    elseif value isa AbstractArray{<:Any,3}
        size(value) == size(prob.tag) || throw(DimensionMismatch("$name must match the cell grid"))
        all(isfinite, Array(value)) || throw(ArgumentError("$name must be finite"))
        data = Float64.(vec(Array(value))[cells])
    else
        throw(ArgumentError("$name must be a scalar or an Nx × Ny × Nz array"))
    end
    all(isfinite, data) || throw(ArgumentError("$name must be finite"))
    return data
end

"""
    fv_temperature_l2(prob::FVSteadyProblem3D, target; weights=1) -> FVObjective

3D form of the weighted temperature misfit `0.5 * sum_fluid(weights * (T-target)^2)`;
`target` and `weights` are scalars or `Nx × Ny × Nz` arrays (absolute temperature).
"""
function fv_temperature_l2(prob::FVSteadyProblem3D, target; weights = 1)
    index, cells = _fv_temperature_indices(prob)
    target_values = _fv3_temperature_data(target, prob, cells, "target") .- prob.energy.Tref
    weight_values = _fv3_temperature_data(weights, prob, cells, "weights")
    all(>=(0), weights isa AbstractArray ? Array(weights) : (weights,)) ||
        throw(ArgumentError("temperature weights must be nonnegative"))
    return _FVTemperatureL2(index, target_values, weight_values)
end
