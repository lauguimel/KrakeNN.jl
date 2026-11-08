# ===========================================================================
# 3D Patch-based grid refinement (D3Q19)
#
# Port of the 2D refinement system to 3D with trilinear interpolation,
# D3Q19 Filippova-Hanel rescaling, and ratio³ block-averaging.
# ===========================================================================

"""
    RefinementPatch3D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.RefinementPatch3D
```

Grid refinement is under development and not validated.
"""
struct RefinementPatch3D{T}
    name::String
    level::Int
    ratio::Int
    Nx::Int; Ny::Int; Nz::Int               # Total dimensions (including ghosts)
    Nx_inner::Int; Ny_inner::Int; Nz_inner::Int  # Inner dimensions (no ghosts)
    dx::T
    x_min::T; y_min::T; z_min::T
    x_max::T; y_max::T; z_max::T
    parent_i_range::UnitRange{Int}
    parent_j_range::UnitRange{Int}
    parent_k_range::UnitRange{Int}
    n_ghost::Int                              # Ghost width (2 for D3Q19)
    f_in::AbstractArray{T, 4}                 # [Nx, Ny, Nz, 19]
    f_out::AbstractArray{T, 4}
    rho::AbstractArray{T, 3}
    ux::AbstractArray{T, 3}
    uy::AbstractArray{T, 3}
    uz::AbstractArray{T, 3}
    is_solid::AbstractArray{Bool, 3}
    omega::T
    # Temporal interpolation buffers (parent overlap + 1-cell margin)
    rho_prev::AbstractArray{T, 3}
    ux_prev::AbstractArray{T, 3}
    uy_prev::AbstractArray{T, 3}
    uz_prev::AbstractArray{T, 3}
    f_prev::AbstractArray{T, 4}
end

"""
    RefinedDomain3D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.RefinedDomain3D
```

Grid refinement is under development and not validated.
"""
struct RefinedDomain3D{T}
    base_Nx::Int; base_Ny::Int; base_Nz::Int
    base_dx::Float64
    base_omega::Float64
    patches::Vector{RefinementPatch3D{T}}
end

"""
    create_patch_3d(name::String, level::Int, ratio::Int,

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.create_patch_3d)
```

Grid refinement is under development and not validated.
"""
function create_patch_3d(name::String, level::Int, ratio::Int,
                          region::NTuple{6, Float64},  # (x0, y0, z0, x1, y1, z1)
                          parent_Nx::Int, parent_Ny::Int, parent_Nz::Int,
                          parent_dx::Float64, parent_omega::Float64,
                          ::Type{T}; backend=KernelAbstractions.CPU()) where T
    x_min, y_min, z_min, x_max, y_max, z_max = region
    n_ghost = 2  # Same as 2D

    # Map physical region to parent cell indices
    i_start = max(1, floor(Int, x_min / parent_dx) + 1)
    j_start = max(1, floor(Int, y_min / parent_dx) + 1)
    k_start = max(1, floor(Int, z_min / parent_dx) + 1)
    i_end = min(parent_Nx, ceil(Int, x_max / parent_dx))
    j_end = min(parent_Ny, ceil(Int, y_max / parent_dx))
    k_end = min(parent_Nz, ceil(Int, z_max / parent_dx))

    parent_i_range = i_start:i_end
    parent_j_range = j_start:j_end
    parent_k_range = k_start:k_end

    Nx_inner = (i_end - i_start + 1) * ratio
    Ny_inner = (j_end - j_start + 1) * ratio
    Nz_inner = (k_end - k_start + 1) * ratio
    Nx = Nx_inner + 2 * n_ghost
    Ny = Ny_inner + 2 * n_ghost
    Nz = Nz_inner + 2 * n_ghost

    dx_fine = parent_dx / ratio
    omega_fine = T(rescaled_omega(parent_omega, ratio))

    # Snapped physical extent
    x_min_snap = (i_start - 1) * parent_dx
    y_min_snap = (j_start - 1) * parent_dx
    z_min_snap = (k_start - 1) * parent_dx
    x_max_snap = i_end * parent_dx
    y_max_snap = j_end * parent_dx
    z_max_snap = k_end * parent_dx

    # Allocate arrays
    f_in  = KernelAbstractions.zeros(backend, T, Nx, Ny, Nz, 19)
    f_out = KernelAbstractions.zeros(backend, T, Nx, Ny, Nz, 19)
    rho   = KernelAbstractions.ones(backend, T, Nx, Ny, Nz)
    ux    = KernelAbstractions.zeros(backend, T, Nx, Ny, Nz)
    uy    = KernelAbstractions.zeros(backend, T, Nx, Ny, Nz)
    uz    = KernelAbstractions.zeros(backend, T, Nx, Ny, Nz)
    is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny, Nz)

    # Initialize f to equilibrium (ρ=1, u=0)
    w = weights(D3Q19())
    f_cpu = zeros(T, Nx, Ny, Nz, 19)
    for q in 1:19
        f_cpu[:, :, :, q] .= T(w[q])
    end
    copyto!(f_in, f_cpu)
    copyto!(f_out, f_cpu)

    # Temporal buffers (+1 margin on each side for trilinear stencil)
    n_pi = length(parent_i_range) + 2
    n_pj = length(parent_j_range) + 2
    n_pk = length(parent_k_range) + 2
    rho_prev = KernelAbstractions.zeros(backend, T, n_pi, n_pj, n_pk)
    ux_prev  = KernelAbstractions.zeros(backend, T, n_pi, n_pj, n_pk)
    uy_prev  = KernelAbstractions.zeros(backend, T, n_pi, n_pj, n_pk)
    uz_prev  = KernelAbstractions.zeros(backend, T, n_pi, n_pj, n_pk)
    f_prev   = KernelAbstractions.zeros(backend, T, n_pi, n_pj, n_pk, 19)

    return RefinementPatch3D{T}(
        name, level, ratio, Nx, Ny, Nz, Nx_inner, Ny_inner, Nz_inner,
        T(dx_fine),
        T(x_min_snap), T(y_min_snap), T(z_min_snap),
        T(x_max_snap), T(y_max_snap), T(z_max_snap),
        parent_i_range, parent_j_range, parent_k_range,
        n_ghost, f_in, f_out, rho, ux, uy, uz, is_solid, omega_fine,
        rho_prev, ux_prev, uy_prev, uz_prev, f_prev)
end

"""
    create_refined_domain_3d(Nx, Ny, Nz, dx, omega, patches)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.create_refined_domain_3d)
```

Grid refinement is under development and not validated.
"""
function create_refined_domain_3d(Nx, Ny, Nz, dx, omega, patches)
    T = isempty(patches) ? Float64 : eltype(patches[1].dx)
    return RefinedDomain3D{T}(Nx, Ny, Nz, Float64(dx), Float64(omega), patches)
end

# --- Save coarse state at time n for temporal interpolation ---

function save_coarse_state_3d!(patch::RefinementPatch3D{T},
                                f_coarse, rho_c, ux_c, uy_c, uz_c) where T
    Nx_c = size(rho_c, 1); Ny_c = size(rho_c, 2); Nz_c = size(rho_c, 3)

    i_lo = max(first(patch.parent_i_range) - 1, 1)
    i_hi = min(last(patch.parent_i_range) + 1, Nx_c)
    j_lo = max(first(patch.parent_j_range) - 1, 1)
    j_hi = min(last(patch.parent_j_range) + 1, Ny_c)
    k_lo = max(first(patch.parent_k_range) - 1, 1)
    k_hi = min(last(patch.parent_k_range) + 1, Nz_c)

    ni = i_hi - i_lo + 1; nj = j_hi - j_lo + 1; nk = k_hi - k_lo + 1

    copyto!(@view(patch.rho_prev[1:ni, 1:nj, 1:nk]), @view(rho_c[i_lo:i_hi, j_lo:j_hi, k_lo:k_hi]))
    copyto!(@view(patch.ux_prev[1:ni, 1:nj, 1:nk]),  @view(ux_c[i_lo:i_hi, j_lo:j_hi, k_lo:k_hi]))
    copyto!(@view(patch.uy_prev[1:ni, 1:nj, 1:nk]),  @view(uy_c[i_lo:i_hi, j_lo:j_hi, k_lo:k_hi]))
    copyto!(@view(patch.uz_prev[1:ni, 1:nj, 1:nk]),  @view(uz_c[i_lo:i_hi, j_lo:j_hi, k_lo:k_hi]))
    copyto!(@view(patch.f_prev[1:ni, 1:nj, 1:nk, :]), @view(f_coarse[i_lo:i_hi, j_lo:j_hi, k_lo:k_hi, :]))
end

# --- Ghost fill using temporal prolongation ---

function fill_ghost_temporal_3d!(patch::RefinementPatch3D{T},
                                  f_coarse, rho_c, ux_c, uy_c, uz_c,
                                  omega_c, Nx_c, Ny_c, Nz_c, t_frac) where T
    prolongate_f_rescaled_temporal_3d!(
        patch.f_in,
        f_coarse, rho_c, ux_c, uy_c, uz_c,
        patch.f_prev, patch.rho_prev, patch.ux_prev, patch.uy_prev, patch.uz_prev,
        patch.ratio, patch.Nx_inner, patch.Ny_inner, patch.Nz_inner,
        patch.n_ghost,
        first(patch.parent_i_range), first(patch.parent_j_range), first(patch.parent_k_range),
        Nx_c, Ny_c, Nz_c, omega_c, Float64(patch.omega), t_frac)
end

# --- Restrict fine → coarse ---

function restrict_to_coarse_3d!(patch::RefinementPatch3D{T},
                                 f_coarse, rho_c, ux_c, uy_c, uz_c,
                                 omega_c) where T
    restrict_f_rescaled_3d!(
        f_coarse, rho_c, ux_c, uy_c, uz_c,
        patch.f_in, patch.rho, patch.ux, patch.uy, patch.uz,
        patch.ratio, patch.n_ghost,
        first(patch.parent_i_range), first(patch.parent_j_range), first(patch.parent_k_range),
        length(patch.parent_i_range), length(patch.parent_j_range), length(patch.parent_k_range),
        omega_c, Float64(patch.omega))
end

# --- Advance one coarse timestep with 3D sub-cycling ---

"""
    advance_refined_step_3d!(domain::RefinedDomain3D{T},

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.advance_refined_step_3d!)
```
"""
function advance_refined_step_3d!(domain::RefinedDomain3D{T},
                                    f_in, f_out, rho, ux, uy, uz, is_solid;
                                    stream_fn, collide_fn, macro_fn,
                                    bc_base_fn=nothing,
                                    bc_patch_fns=nothing,
                                    bc_coarse_fn=nothing,
                                    patch_collide_fns=nothing) where T
    Nx = domain.base_Nx; Ny = domain.base_Ny; Nz = domain.base_Nz

    # 1. Save coarse state at time n
    for patch in domain.patches
        save_coarse_state_3d!(patch, f_in, rho, ux, uy, uz)
    end

    # 2. Advance coarse grid one step
    stream_fn(f_out, f_in, Nx, Ny, Nz)
    bc_base_fn !== nothing && bc_base_fn(f_out)
    collide_fn(f_out, is_solid)
    macro_fn(rho, ux, uy, uz, f_out)
    f_in, f_out = f_out, f_in

    # 3. Sub-cycle each patch
    for (pidx, patch) in enumerate(domain.patches)
        ratio = patch.ratio
        coll_fn = patch_collide_fns !== nothing ?
            get(patch_collide_fns, pidx, nothing) : nothing

        for sub_step in 1:ratio
            t_frac = T((sub_step - 1) / ratio)

            fill_ghost_temporal_3d!(patch, f_in, rho, ux, uy, uz,
                                    domain.base_omega, Nx, Ny, Nz, t_frac)

            stream_3d!(patch.f_out, patch.f_in, patch.Nx, patch.Ny, patch.Nz)

            if bc_patch_fns !== nothing
                bc_fn = get(bc_patch_fns, pidx, nothing)
                bc_fn !== nothing && bc_fn(patch.f_out, patch.Nx, patch.Ny, patch.Nz)
            end

            if coll_fn !== nothing
                coll_fn(patch.f_out, patch.is_solid)
            else
                collide_3d!(patch.f_out, patch.is_solid, patch.omega)
            end
            compute_macroscopic_3d!(patch.rho, patch.ux, patch.uy, patch.uz, patch.f_out)

            copyto!(patch.f_in, patch.f_out)
        end
    end

    # 4. Restrict fine → coarse
    for patch in domain.patches
        restrict_to_coarse_3d!(patch, f_in, rho, ux, uy, uz, domain.base_omega)
    end

    # 5. Re-apply coarse BCs (restriction may overwrite wall cells)
    if bc_coarse_fn !== nothing
        bc_coarse_fn(f_in, Nx, Ny, Nz)
    end
    compute_macroscopic_3d!(rho, ux, uy, uz, f_in)

    return f_in, f_out
end
