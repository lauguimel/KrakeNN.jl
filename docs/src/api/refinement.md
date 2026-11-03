# Grid refinement (under development)

Grid refinement (patch and conservative AMR) is under development and not validated; it is not part of the supported feature set. This page is not listed in the navigation and is kept for reference only.

Kraken supports patch-based nested refinement (2:1 ratio) with
Filippova-Hänel distribution rescaling across levels. A
`RefinedDomain` bundles the coarse base grid with one or more
`RefinementPatch`es; `advance_refined_step!` performs one coarse
step plus the required fine sub-steps, including prolongation,
restriction, and temporal interpolation at patch boundaries.


## Quick reference

| Symbol | Purpose |
|--------|---------|
| `ThermalPatchArrays` | Per-patch thermal DDF arrays |
| `create_thermal_patch_arrays` | Allocate thermal arrays for refined patches |
| `advance_thermal_refined_step!` | Advance one refined thermal step |
| `RefinementPatch` | Descriptor for one nested-refinement patch |
| `RefinedDomain` | Container bundling coarse grid + patches |
| `create_patch` | Build a `RefinementPatch` from a bounding box |
| `create_refined_domain` | Assemble a `RefinedDomain` from multiple patches |
| `rescaled_omega` | Filippova-Hänel τ rescaling across levels |
| `rescaling_factor_c2f` | Distribution rescaling factor coarse→fine |
| `rescaling_factor_f2c` | Distribution rescaling factor fine→coarse |
| `prolongate_f_rescaled_2d!` | Interpolate f from coarse to fine (rescaled) |
| `restrict_f_rescaled_2d!` | Average f from fine to coarse (rescaled) |
| `temporal_interpolate_2d!` | Temporal interpolation at patch boundaries |
| `copy_macroscopic_overlap_2d!` | Copy ρ, u in the overlap region |
| `advance_refined_step!` | Advance one coarse step + nested fine sub-steps |

## Details

### `RefinementPatch`

**Source:** `src/refinement/refinement.jl`

```julia
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
```


### `RefinedDomain`

**Source:** `src/refinement/refinement.jl`

```julia
"""
    RefinedDomain{T}

Multi-level domain with a base grid and nested refinement patches.
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
```


### `create_refined_domain`

**Source:** `src/refinement/refinement.jl`

```julia
"""
    create_refined_domain(base_Nx, base_Ny, base_dx, base_omega, patches) -> RefinedDomain

Build a RefinedDomain from a base grid specification and a list of patches.
Patches must be ordered by level (level 1 first, then level 2, etc.).
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
```


### `advance_refined_step!`

**Source:** `src/refinement/time_stepping.jl`

```julia
"""
    advance_refined_step!(domain, f_in, f_out, rho, ux, uy, is_solid;
                          stream_fn, collide_fn, macro_fn, bc_base_fn,
                          bc_patch_fns)

Advance the full refined domain by one coarse timestep with sub-cycling.

# Algorithm
1. Save coarse state at time n for each patch
2. Advance coarse grid one step
3. For each patch, sub-cycle `ratio` fine steps with temporal interpolation
4. Restrict fine results back to coarse overlap

Returns updated (f_in, f_out) for the base grid (swapped).
"""
function advance_refined_step!(domain::RefinedDomain{T},
                               f_in, f_out, rho, ux, uy, is_solid;
                               stream_fn,
                               collide_fn,
                               macro_fn,
                               bc_base_fn=nothing,
                               bc_patch_fns=nothing) where T
    Nx = domain.base_Nx
    Ny = domain.base_Ny

    # 1. Save coarse state at time n for all patches
    for patch in domain.patches
        save_coarse_state!(patch, f_in, rho, ux, uy)
    end

    # 2. Advance coarse grid one step
    stream_fn(f_out, f_in, Nx, Ny)
    if bc_base_fn !== nothing
        bc_base_fn(f_out)
    end
    collide_fn(f_out, is_solid)
    macro_fn(rho, ux, uy, f_out)
    f_in, f_out = f_out, f_in

    # 3. Sub-cycle each patch
    for (pidx, patch) in enumerate(domain.patches)
        ratio = patch.ratio

        for sub_step in 1:ratio
            # Temporal interpolation: sub_step=1 -> t_frac=0, sub_step=ratio -> (ratio-1)/ratio
            t_frac = T((sub_step - 1) / ratio)

            if t_frac > zero(T)
                _fill_ghost_interpolated!(patch, f_in, rho, ux, uy,
                                         Float64(domain.base_omega),
                                         Nx, Ny, t_frac)
            else
                # First sub-step: use coarse state at time n
                fill_ghost_from_coarse!(patch, f_in, rho, ux, uy,
                                        Float64(domain.base_omega), Nx, Ny)
            end

            # Advance patch one fine step
            bc_fn = bc_patch_fns !== nothing ? get(bc_patch_fns, pidx, nothing) : nothing
            advance_patch!(patch;
                          stream_fn=stream_fn,
                          collide_fn=(f, is_s) -> collide_2d!(f, is_s, patch.omega),
                          macro_fn=compute_macroscopic_2d!,
                          bc_fn=bc_fn)

            # Copy f_out -> f_in for next sub-step
            copyto!(patch.f_in, patch.f_out)
        end
    end

    # 4. Restrict fine results back to coarse
    for patch in domain.patches
        restrict_to_coarse!(patch, f_in, rho, ux, uy,
                           Float64(domain.base_omega))
    end

    return f_in, f_out
end
```


### `rescaled_omega`

**Source:** `src/refinement/refinement.jl`

```julia
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
```


