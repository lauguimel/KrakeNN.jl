using KernelAbstractions

# ===========================================================================
# Patch-based static grid refinement for LBM
#
# Each RefinementPatch is an independent uniform grid that reuses existing
# stream/collide/BC kernels unchanged. Ghost-layer exchange between levels
# uses Filippova-Hanel non-equilibrium rescaling.
#
# References:
# - Filippova & Hanel (1998) doi:10.1006/jcph.1998.6057
# - Dupuis & Chopard (2003) doi:10.1016/S0378-4371(03)00281-4
# ===========================================================================

"""
    RefinementPatch{T}

A rectangular sub-domain at a given refinement level. Each patch is a
self-contained uniform grid with its own LBM state arrays, allocated on the
same GPU backend as the base grid. Existing kernels operate on patches without
any modification.

# Fields
- `name`: human-readable label (matches .krk `Refine` block name)
- `level`: refinement level (0 = base, 1 = first refinement, ...)
- `ratio`: refinement ratio relative to parent (typically 2)
- `Nx, Ny`: lattice dimensions including ghost cells
- `Nx_inner, Ny_inner`: lattice dimensions excluding ghost cells
- `dx`: physical grid spacing at this level
- `x_min, y_min, x_max, y_max`: physical extent (inner region, no ghosts)
- `parent_i_range, parent_j_range`: indices in parent grid covered by this patch
- `n_ghost`: ghost layer width (2 for D2Q9)
- `f_in, f_out`: distribution function arrays [Nx, Ny, Q]
- `rho, ux, uy`: macroscopic fields [Nx, Ny]
- `is_solid`: obstacle mask [Nx, Ny]
- `omega`: relaxation parameter (rescaled from parent)

Grid refinement is under development and not validated.
"""
struct RefinementPatch{T}
    name::String
    level::Int
    ratio::Int
    Nx::Int
    Ny::Int
    Nx_inner::Int
    Ny_inner::Int
    dx::T
    x_min::T
    y_min::T
    x_max::T
    y_max::T
    parent_i_range::UnitRange{Int}
    parent_j_range::UnitRange{Int}
    n_ghost::Int
    # LBM state (GPU arrays)
    f_in::AbstractArray{T, 3}
    f_out::AbstractArray{T, 3}
    rho::AbstractMatrix{T}
    ux::AbstractMatrix{T}
    uy::AbstractMatrix{T}
    is_solid::AbstractMatrix{Bool}
    omega::T
    # Temporal interpolation buffers (parent macroscopic at step n)
    rho_prev::AbstractMatrix{T}
    ux_prev::AbstractMatrix{T}
    uy_prev::AbstractMatrix{T}
    f_prev::AbstractArray{T, 3}
end

"""
    RefinedDomain{T}

Multi-level domain with a base grid and nested refinement patches.

Grid refinement is under development and not validated.
"""
struct RefinedDomain{T}
    base_Nx::Int
    base_Ny::Int
    base_dx::T
    base_omega::T
    patches::Vector{RefinementPatch{T}}
    parent_of::Dict{Int, Int}      # patch index -> parent patch index (0 = base)
    children_of::Dict{Int, Vector{Int}}
end

"""
    rescaled_omega(omega_parent, ratio) -> Float64

Compute the relaxation parameter at the fine level to preserve physical
viscosity across refinement levels.

    tau_fine = ratio * (tau_parent - 0.5) + 0.5
    omega_fine = 1 / tau_fine
"""
function rescaled_omega(omega_parent::Real, ratio::Int)
    tau_parent = 1.0 / omega_parent
    tau_fine = ratio * (tau_parent - 0.5) + 0.5
    return 1.0 / tau_fine
end

"""
    rescaling_factor_c2f(omega_coarse, omega_fine, ratio) -> Float64

Non-equilibrium rescaling factor for coarse-to-fine ghost fill (Filippova-Hanel).

    alpha = (tau_fine - 0.5) / (tau_coarse - 0.5)

Singularity-free for all valid omega in (0, 2).
"""
function rescaling_factor_c2f(omega_coarse::Real, omega_fine::Real, ratio::Int)
    tau_c = 1.0 / omega_coarse
    tau_f = 1.0 / omega_fine
    return (tau_f - 0.5) / (tau_c - 0.5)
end

"""
    rescaling_factor_f2c(omega_coarse, omega_fine, ratio) -> Float64

Non-equilibrium rescaling factor for fine-to-coarse restriction (inverse).

    alpha = (tau_coarse - 0.5) / (tau_fine - 0.5)
"""
function rescaling_factor_f2c(omega_coarse::Real, omega_fine::Real, ratio::Int)
    tau_c = 1.0 / omega_coarse
    tau_f = 1.0 / omega_fine
    return (tau_c - 0.5) / (tau_f - 0.5)
end

"""
    create_patch(name, level, ratio, region, parent_Nx, parent_Ny, parent_dx,
                 parent_omega, n_ghost, backend, T) -> RefinementPatch

Create a refinement patch from a physical region specification.

# Arguments
- `region`: (x_min, y_min, x_max, y_max) physical coordinates
- `parent_Nx, parent_Ny`: parent grid dimensions
- `parent_dx`: parent grid spacing
- `parent_omega`: parent relaxation parameter
- `n_ghost`: ghost layer width (default 2 for D2Q9)

Grid refinement is under development and not validated.
"""
function create_patch(name::String, level::Int, ratio::Int,
                      region::NTuple{4, Float64},
                      parent_Nx::Int, parent_Ny::Int, parent_dx::Real,
                      parent_omega::Real,
                      ::Type{T}=Float64;
                      n_ghost::Int=2,
                      backend=KernelAbstractions.CPU()) where T
    x_min, y_min, x_max, y_max = region

    # Map physical region to parent grid indices (cell-center convention)
    i_start = max(1, floor(Int, x_min / parent_dx) + 1)
    j_start = max(1, floor(Int, y_min / parent_dx) + 1)
    i_end = min(parent_Nx, ceil(Int, x_max / parent_dx))
    j_end = min(parent_Ny, ceil(Int, y_max / parent_dx))

    parent_i_range = i_start:i_end
    parent_j_range = j_start:j_end

    # Fine grid dimensions (inner = ratio * parent cells covered)
    Nx_inner = (i_end - i_start + 1) * ratio
    Ny_inner = (j_end - j_start + 1) * ratio
    Nx = Nx_inner + 2 * n_ghost
    Ny = Ny_inner + 2 * n_ghost

    dx_fine = T(parent_dx / ratio)
    omega_fine = T(rescaled_omega(parent_omega, ratio))

    # Snap physical extent to parent cell boundaries
    x_min_snap = T((i_start - 1) * parent_dx)
    y_min_snap = T((j_start - 1) * parent_dx)
    x_max_snap = T(i_end * parent_dx)
    y_max_snap = T(j_end * parent_dx)

    # Allocate GPU arrays
    f_in  = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
    f_out = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
    rho   = KernelAbstractions.ones(backend, T, Nx, Ny)
    ux    = KernelAbstractions.zeros(backend, T, Nx, Ny)
    uy    = KernelAbstractions.zeros(backend, T, Nx, Ny)
    is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny)

    # Initialize f to equilibrium (rho=1, u=0)
    w = weights(D2Q9())
    f_cpu = zeros(T, Nx, Ny, 9)
    for q in 1:9
        f_cpu[:, :, q] .= T(w[q])
    end
    copyto!(f_in, f_cpu)
    copyto!(f_out, f_cpu)

    # Temporal interpolation buffers (parent-size region + halo for interpolation)
    n_parent_i = length(parent_i_range) + 2  # +2 for bilinear stencil margin
    n_parent_j = length(parent_j_range) + 2
    rho_prev = KernelAbstractions.ones(backend, T, n_parent_i, n_parent_j)
    ux_prev  = KernelAbstractions.zeros(backend, T, n_parent_i, n_parent_j)
    uy_prev  = KernelAbstractions.zeros(backend, T, n_parent_i, n_parent_j)
    f_prev   = KernelAbstractions.zeros(backend, T, n_parent_i, n_parent_j, 9)

    return RefinementPatch{T}(
        name, level, ratio, Nx, Ny, Nx_inner, Ny_inner,
        dx_fine, x_min_snap, y_min_snap, x_max_snap, y_max_snap,
        parent_i_range, parent_j_range, n_ghost,
        f_in, f_out, rho, ux, uy, is_solid, omega_fine,
        rho_prev, ux_prev, uy_prev, f_prev
    )
end

"""
    create_refined_domain(base_Nx, base_Ny, base_dx, base_omega, patches) -> RefinedDomain

Build a RefinedDomain from a base grid specification and a list of patches.
Patches must be ordered by level (level 1 first, then level 2, etc.).

Grid refinement is under development and not validated.
"""
function create_refined_domain(base_Nx::Int, base_Ny::Int, base_dx::Real,
                               base_omega::Real,
                               patches::Vector{RefinementPatch{T}};
                               parent_map::Dict{Int, Int}=Dict{Int, Int}()) where T
    # Default: all patches have the base grid as parent
    parent_of = copy(parent_map)
    for (idx, _) in enumerate(patches)
        if !haskey(parent_of, idx)
            parent_of[idx] = 0
        end
    end

    # Build reverse mapping
    children_of = Dict{Int, Vector{Int}}()
    for (child, parent) in parent_of
        if !haskey(children_of, parent)
            children_of[parent] = Int[]
        end
        push!(children_of[parent], child)
    end

    return RefinedDomain{T}(
        base_Nx, base_Ny, T(base_dx), T(base_omega),
        patches, parent_of, children_of
    )
end
