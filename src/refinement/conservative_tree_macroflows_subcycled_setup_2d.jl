# Macro-flow runners over the recursive AMR-D subcycled route scheduler.

"""
    ConservativeTreeSpecMacroFlow2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreeSpecMacroFlow2D
```
"""
struct ConservativeTreeSpecMacroFlow2D{T}
    flow::Symbol
    max_level::Int
    steps::Int
    spec::ConservativeTreeSpec2D
    table::ConservativeTreeRouteTable2D
    F::Matrix{T}
    y::Vector{T}
    ux_profile::Vector{T}
    analytic_profile::Vector{T}
    l2_error::T
    linf_error::T
    mass_initial::T
    mass_final::T
    mass_drift::T
    relative_mass_drift::T
    max_raw_relative_mass_drift::T
    active_cell_count::Int
    leaf_equivalent_cell_count::Int
end

struct ConservativeTreeSpecSolidFlow2D{T}
    flow::Symbol
    max_level::Int
    steps::Int
    spec::ConservativeTreeSpec2D
    table::ConservativeTreeRouteTable2D
    F::Matrix{T}
    is_solid_leaf::BitMatrix
    ux_mean::T
    uy_mean::T
    mass_initial::T
    mass_final::T
    mass_drift::T
    relative_mass_drift::T
    max_raw_relative_mass_drift::T
    active_cell_count::Int
    leaf_equivalent_cell_count::Int
end

"""
    conservative_tree_mass_roundoff_rtol_2d(::Type{T},

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_mass_roundoff_rtol_2d)
```
"""
function conservative_tree_mass_roundoff_rtol_2d(::Type{T},
                                                 steps::Integer,
                                                 max_level::Integer;
                                                 active_cell_count::Integer=1,
                                                 safety=4096) where T<:AbstractFloat
    nsteps = max(Int(steps), 1)
    levels = max(Int(max_level) + 1, 1)
    cells = max(Int(active_cell_count), 1)
    cell_factor = max(one(T), T(log2(cells + 1)))
    return T(safety) * eps(T) * T(nsteps) * T(levels) * cell_factor
end

function _check_conservative_tree_channel_max_level_2d(max_level::Integer)
    ml = Int(max_level)
    1 <= ml <= 4 ||
        throw(ArgumentError("the reference nested channel currently supports max_level in 1:4"))
    return ml
end

@inline function conservative_tree_leaf_equivalent_level_scale_2d(
        spec::ConservativeTreeSpec2D,
        level::Integer)
    l = Int(level)
    0 <= l <= spec.max_level ||
        throw(ArgumentError("level is outside the conservative-tree spec"))
    return 1 << (spec.max_level - l)
end

function conservative_tree_leaf_equivalent_omega_2d(
        omega,
        spec::ConservativeTreeSpec2D,
        level::Integer)
    scale = conservative_tree_leaf_equivalent_level_scale_2d(spec, level)
    T = typeof(float(omega))
    tau_fine = inv(T(omega))
    tau_fine > T(0.5) ||
        throw(ArgumentError("leaf-equivalent omega requires tau_fine > 0.5"))
    tau_level = T(0.5) + (tau_fine - T(0.5)) / T(scale)
    return inv(tau_level)
end

@inline function conservative_tree_leaf_equivalent_force_2d(
        force,
        spec::ConservativeTreeSpec2D,
        level::Integer)
    scale = conservative_tree_leaf_equivalent_level_scale_2d(spec, level)
    return force * scale
end

function _nested_channel_refine_blocks_2d(max_level::Integer)
    ml = _check_conservative_tree_channel_max_level_2d(max_level)
    blocks = ConservativeTreeRefineBlock2D[
        ConservativeTreeRefineBlock2D("L1", 5:12, 3:10),
    ]
    ml >= 2 && push!(blocks,
        ConservativeTreeRefineBlock2D("L2", 13:20, 7:14; parent="L1"))
    ml >= 3 && push!(blocks,
        ConservativeTreeRefineBlock2D("L3", 29:36, 17:24; parent="L2"))
    ml >= 4 && push!(blocks,
        ConservativeTreeRefineBlock2D("L4", 61:68, 37:44; parent="L3"))
    return blocks
end

"""
    create_conservative_tree_nested_channel_spec_2d(max_level::Integer;

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.create_conservative_tree_nested_channel_spec_2d)
```
"""
function create_conservative_tree_nested_channel_spec_2d(max_level::Integer;
                                                         Nx::Integer=16,
                                                         Ny::Integer=12)
    Int(Nx) == 16 && Int(Ny) == 12 ||
        throw(ArgumentError("the reference nested channel block set is defined for Nx=16, Ny=12"))
    return create_conservative_tree_spec_2d(
        Int(Nx), Int(Ny), _nested_channel_refine_blocks_2d(max_level))
end

"""
    initialize_conservative_tree_equilibrium_F_2d!(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.initialize_conservative_tree_equilibrium_F_2d!)
```
"""
function initialize_conservative_tree_equilibrium_F_2d!(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D;
        rho=1,
        ux=0,
        uy=0)
    _check_conservative_tree_F_2d(F, spec)
    fill!(F, zero(eltype(F)))
    T = promote_type(eltype(F), typeof(float(rho)), typeof(float(ux)),
                     typeof(float(uy)))
    @inbounds for cell_id in spec.active_cells
        volume = T(spec.cells[cell_id].metrics.volume)
        for q in 1:9
            F[cell_id, q] = volume * equilibrium(
                D2Q9(), T(rho), T(ux), T(uy), q)
        end
    end
    return F
end

function initialize_conservative_tree_solid_equilibrium_F_2d!(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        is_solid::AbstractArray{Bool,2};
        rho=1,
        ux=0,
        uy=0)
    _check_conservative_tree_F_2d(F, spec)
    _check_conservative_tree_leaf_solid_mask_2d(spec, is_solid)
    fill!(F, zero(eltype(F)))
    T = promote_type(eltype(F), typeof(float(rho)), typeof(float(ux)),
                     typeof(float(uy)))
    @inbounds for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        _conservative_tree_cell_is_solid_2d(spec, cell, is_solid) && continue
        volume = T(cell.metrics.volume)
        for q in 1:9
            F[cell_id, q] = volume * equilibrium(
                D2Q9(), T(rho), T(ux), T(uy), q)
        end
    end
    return F
end

function _active_mass_conservative_tree_F_2d(F::AbstractMatrix,
                                             spec::ConservativeTreeSpec2D)
    _check_conservative_tree_F_2d(F, spec)
    mass = zero(eltype(F))
    @inbounds for cell_id in spec.active_cells, q in 1:9
        mass += F[cell_id, q]
    end
    return mass
end

function _active_fluid_mass_conservative_tree_F_2d(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        is_solid::AbstractArray{Bool,2})
    _check_conservative_tree_F_2d(F, spec)
    _check_conservative_tree_leaf_solid_mask_2d(spec, is_solid)
    mass = zero(eltype(F))
    @inbounds for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        _conservative_tree_cell_is_solid_2d(spec, cell, is_solid) && continue
        for q in 1:9
            mass += F[cell_id, q]
        end
    end
    return mass
end

@inline function _row_mass_conservative_tree_F_2d(F::AbstractMatrix,
                                                  cell_id::Int)
    mass = zero(eltype(F))
    @inbounds for q in 1:9
        mass += F[cell_id, q]
    end
    return mass
end

@inline function _restore_row_mass_conservative_tree_F_2d!(
        F::AbstractMatrix,
        cell_id::Int,
        mass_before)
    mass_after = _row_mass_conservative_tree_F_2d(F, cell_id)
    @inbounds F[cell_id, 1] += mass_before - mass_after
    return F
end

function _enforce_active_mass_conservation_2d!(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        target_mass;
        rtol)
    mass_now = _active_mass_conservative_tree_F_2d(F, spec)
    drift = mass_now - target_mass
    denom = max(abs(target_mass), eps(typeof(float(target_mass))))
    rel = abs(drift) / denom
    rel <= rtol ||
        throw(ArgumentError("AMR-D mass residual $(rel) exceeds roundoff guard $(rtol)"))
    first_cell = first(spec.active_cells)
    @inbounds F[first_cell, 1] -= drift
    return rel
end

function _enforce_active_fluid_mass_conservation_2d!(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        is_solid::AbstractArray{Bool,2},
        target_mass;
        rtol)
    mass_now = _active_fluid_mass_conservative_tree_F_2d(F, spec, is_solid)
    drift = mass_now - target_mass
    denom = max(abs(target_mass), eps(typeof(float(target_mass))))
    rel = abs(drift) / denom
    rel <= rtol ||
        throw(ArgumentError("AMR-D fluid mass residual $(rel) exceeds roundoff guard $(rtol)"))
    for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        _conservative_tree_cell_is_solid_2d(spec, cell, is_solid) && continue
        @inbounds F[cell_id, 1] -= drift
        return rel
    end
    throw(ArgumentError("AMR-D solid mask leaves no active fluid cell"))
end

function _collide_BGK_conservative_tree_active_level_F_2d!(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        level::Int,
        omega)
    _check_conservative_tree_F_2d(F, spec)
    @inbounds for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        cell.level == level || continue
        mass_before = _row_mass_conservative_tree_F_2d(F, cell_id)
        collide_BGK_integrated_D2Q9!(@view(F[cell_id, :]),
                                     cell.metrics.volume, omega)
        _restore_row_mass_conservative_tree_F_2d!(F, cell_id, mass_before)
    end
    return F
end

function _collide_Guo_conservative_tree_active_level_F_2d!(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        level::Int,
        omega,
        Fx,
        Fy)
    _check_conservative_tree_F_2d(F, spec)
    @inbounds for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        cell.level == level || continue
        mass_before = _row_mass_conservative_tree_F_2d(F, cell_id)
        collide_Guo_integrated_D2Q9!(@view(F[cell_id, :]),
                                     cell.metrics.volume, omega, Fx, Fy)
        _restore_row_mass_conservative_tree_F_2d!(F, cell_id, mass_before)
    end
    return F
end

function _collide_Guo_conservative_tree_active_fluid_level_F_2d!(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        level::Int,
        is_solid::AbstractArray{Bool,2},
        omega,
        Fx,
        Fy)
    _check_conservative_tree_F_2d(F, spec)
    @inbounds for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        cell.level == level || continue
        _conservative_tree_cell_is_solid_2d(spec, cell, is_solid) && continue
        mass_before = _row_mass_conservative_tree_F_2d(F, cell_id)
        collide_Guo_integrated_D2Q9!(@view(F[cell_id, :]),
                                     cell.metrics.volume, omega, Fx, Fy)
        _restore_row_mass_conservative_tree_F_2d!(F, cell_id, mass_before)
    end
    return F
end

"""
    conservative_tree_leaf_mean_ux_profile_2d(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_leaf_mean_ux_profile_2d)
```
"""
function conservative_tree_leaf_mean_ux_profile_2d(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D;
        force_x=0,
        level_scaled_force::Bool=false)
    _check_conservative_tree_F_2d(F, spec)
    leaf_ny = _conservative_tree_level_size_2d(spec.Ny, spec.max_level)
    row_mass = zeros(eltype(F), leaf_ny)
    row_ux_mass = zeros(eltype(F), leaf_ny)

    @inbounds for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        scale = 1 << (spec.max_level - cell.level)
        mass = zero(eltype(F))
        mx = zero(eltype(F))
        for q in 1:9
            Fq = F[cell_id, q]
            mass += Fq
            mx += d2q9_cx(q) * Fq
        end
        volume = eltype(F)(cell.metrics.volume)
        rho = mass / volume
        fx = level_scaled_force ?
             conservative_tree_leaf_equivalent_force_2d(force_x, spec,
                                                        cell.level) :
             force_x
        ux = (mx / volume + fx / 2) / rho
        row_packet = mass / scale
        for sj in 1:scale
            jf = (cell.j - 1) * scale + sj
            row_mass[jf] += row_packet
            row_ux_mass[jf] += row_packet * ux
        end
    end

    profile = similar(row_ux_mass)
    @inbounds for j in eachindex(profile)
        profile[j] = row_ux_mass[j] / row_mass[j]
    end
    return profile
end

function conservative_tree_leaf_fluid_mean_velocity_2d(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        is_solid::AbstractArray{Bool,2};
        force_x=0,
        force_y=0,
        level_scaled_force::Bool=false)
    _check_conservative_tree_F_2d(F, spec)
    _check_conservative_tree_leaf_solid_mask_2d(spec, is_solid)
    sum_ux = zero(eltype(F))
    sum_uy = zero(eltype(F))
    sum_volume = zero(eltype(F))
    @inbounds for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        _conservative_tree_cell_is_solid_2d(spec, cell, is_solid) && continue
        mass = zero(eltype(F))
        mx = zero(eltype(F))
        my = zero(eltype(F))
        for q in 1:9
            Fq = F[cell_id, q]
            mass += Fq
            mx += d2q9_cx(q) * Fq
            my += d2q9_cy(q) * Fq
        end
        volume = eltype(F)(cell.metrics.volume)
        rho = mass / volume
        fx = level_scaled_force ?
             conservative_tree_leaf_equivalent_force_2d(force_x, spec,
                                                        cell.level) :
             force_x
        fy = level_scaled_force ?
             conservative_tree_leaf_equivalent_force_2d(force_y, spec,
                                                        cell.level) :
             force_y
        ux = (mx / volume + fx / 2) / rho
        uy = (my / volume + fy / 2) / rho
        sum_ux += volume * ux
        sum_uy += volume * uy
        sum_volume += volume
    end
    sum_volume > 0 ||
        throw(ArgumentError("AMR-D solid mask leaves no fluid volume"))
    return sum_ux / sum_volume, sum_uy / sum_volume
end
