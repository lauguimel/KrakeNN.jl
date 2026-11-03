
# --- Auto-detect 3D patch faces touching domain walls ---

"""
    build_patch_flow_bcs_3d(patches, Lx, Ly, Lz, Nx; wall_faces=[:west,:east,:south,:north,:bottom,:top])

Build BC closures for 3D patch faces that touch domain walls.
Returns Dict{Int, Function} mapping patch index → BC closure `(f, Nx_p, Ny_p, Nz_p) -> ...`.
"""
# ===========================================================================
# Thermal extension for 3D patch-based grid refinement (D3Q19)
#
# Adds thermal DDF (g populations) support to the 3D refinement system.
# g uses D3Q19 with simpler prolongation (trilinear without FH, since g_eq
# is linear in T) and restriction (block averaging over ratio³ cells).
# ===========================================================================

struct ThermalPatchArrays3D{T}
    g_in::AbstractArray{T, 4}       # [Nx, Ny, Nz, 19]
    g_out::AbstractArray{T, 4}
    Temp::AbstractArray{T, 3}
    omega_T::T
    # Temporal interpolation buffers (parent overlap + 1-cell margin)
    Temp_prev::AbstractArray{T, 3}
    g_prev::AbstractArray{T, 4}
end

"""
    create_thermal_patch_arrays_3d(patch::RefinementPatch3D{T},

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using Kraken

methods(Kraken.create_thermal_patch_arrays_3d)
```
"""
function create_thermal_patch_arrays_3d(patch::RefinementPatch3D{T},
                                         omega_T_parent::Real;
                                         T_init::Real=0.5,
                                         backend=KernelAbstractions.CPU()) where T
    Nx, Ny, Nz = patch.Nx, patch.Ny, patch.Nz

    omega_T_fine = T(rescaled_omega(omega_T_parent, patch.ratio))

    g_in  = KernelAbstractions.zeros(backend, T, Nx, Ny, Nz, 19)
    g_out = KernelAbstractions.zeros(backend, T, Nx, Ny, Nz, 19)
    Temp  = KernelAbstractions.zeros(backend, T, Nx, Ny, Nz)

    # Initialize g to equilibrium at T_init (g_eq = w_q * T for u=0)
    w = weights(D3Q19())
    g_cpu = zeros(T, Nx, Ny, Nz, 19)
    for q in 1:19
        g_cpu[:, :, :, q] .= T(w[q]) * T(T_init)
    end
    copyto!(g_in, g_cpu)
    copyto!(g_out, g_cpu)

    # Temporal buffers (same parent-region size as flow patch)
    n_pi = size(patch.rho_prev, 1)
    n_pj = size(patch.rho_prev, 2)
    n_pk = size(patch.rho_prev, 3)
    Temp_prev = KernelAbstractions.zeros(backend, T, n_pi, n_pj, n_pk)
    g_prev    = KernelAbstractions.zeros(backend, T, n_pi, n_pj, n_pk, 19)

    return ThermalPatchArrays3D{T}(g_in, g_out, Temp, omega_T_fine,
                                    Temp_prev, g_prev)
end

function save_thermal_coarse_state_3d!(patch::RefinementPatch3D{T},
                                        thermal::ThermalPatchArrays3D{T},
                                        g_coarse, Temp_c) where T
    Nx_c = size(Temp_c, 1); Ny_c = size(Temp_c, 2); Nz_c = size(Temp_c, 3)

    i_lo = max(first(patch.parent_i_range) - 1, 1)
    i_hi = min(last(patch.parent_i_range) + 1, Nx_c)
    j_lo = max(first(patch.parent_j_range) - 1, 1)
    j_hi = min(last(patch.parent_j_range) + 1, Ny_c)
    k_lo = max(first(patch.parent_k_range) - 1, 1)
    k_hi = min(last(patch.parent_k_range) + 1, Nz_c)

    ni = i_hi - i_lo + 1; nj = j_hi - j_lo + 1; nk = k_hi - k_lo + 1

    copyto!(@view(thermal.Temp_prev[1:ni, 1:nj, 1:nk]),
            @view(Temp_c[i_lo:i_hi, j_lo:j_hi, k_lo:k_hi]))
    copyto!(@view(thermal.g_prev[1:ni, 1:nj, 1:nk, :]),
            @view(g_coarse[i_lo:i_hi, j_lo:j_hi, k_lo:k_hi, :]))
end

function _restore_coarse_overlap_3d!(f_coarse, g_coarse,
                                      patch::RefinementPatch3D{T},
                                      thermal::ThermalPatchArrays3D{T}) where T
    i_range = patch.parent_i_range
    j_range = patch.parent_j_range
    k_range = patch.parent_k_range
    i_lo = max(first(i_range) - 1, 1)
    j_lo = max(first(j_range) - 1, 1)
    k_lo = max(first(k_range) - 1, 1)
    Ni = length(i_range); Nj = length(j_range); Nk = length(k_range)
    i_off = first(i_range); j_off = first(j_range); k_off = first(k_range)

    backend = KernelAbstractions.get_backend(f_coarse)
    kernel! = _restore_overlap_3d_kernel!(backend)
    kernel!(f_coarse, g_coarse, patch.f_prev, thermal.g_prev,
            i_off, j_off, k_off, i_lo, j_lo, k_lo; ndrange=(Ni, Nj, Nk))
end

@kernel function _restore_overlap_3d_kernel!(f_c, g_c, @Const(f_prev), @Const(g_prev),
                                              i_off, j_off, k_off, i_lo, j_lo, k_lo)
    li, lj, lk = @index(Global, NTuple)
    @inbounds begin
        i = li + i_off - 1
        j = lj + j_off - 1
        k = lk + k_off - 1
        ip = i - i_lo + 1
        jp = j - j_lo + 1
        kp = k - k_lo + 1
        for q in 1:19
            f_c[i, j, k, q] = f_prev[ip, jp, kp, q]
            g_c[i, j, k, q] = g_prev[ip, jp, kp, q]
        end
    end
end

"""
    fill_thermal_ghost_3d!(patch::RefinementPatch3D{T},

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using Kraken

methods(Kraken.fill_thermal_ghost_3d!)
```
"""
function fill_thermal_ghost_3d!(patch::RefinementPatch3D{T},
                                 thermal::ThermalPatchArrays3D{T},
                                 g_coarse,
                                 Nx_c::Int, Ny_c::Int, Nz_c::Int;
                                 t_frac::Real=zero(T)) where T
    _fill_thermal_ghost_temporal_3d!(
        thermal.g_in, g_coarse, thermal.g_prev,
        patch.ratio, patch.Nx_inner, patch.Ny_inner, patch.Nz_inner,
        patch.n_ghost,
        first(patch.parent_i_range), first(patch.parent_j_range), first(patch.parent_k_range),
        Nx_c, Ny_c, Nz_c, T(t_frac))
end

"""
    fill_thermal_full_3d!(patch::RefinementPatch3D{T},

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using Kraken

methods(Kraken.fill_thermal_full_3d!)
```
"""
function fill_thermal_full_3d!(patch::RefinementPatch3D{T},
                                thermal::ThermalPatchArrays3D{T},
                                g_coarse, Nx_c::Int, Ny_c::Int, Nz_c::Int) where T
    backend = KernelAbstractions.get_backend(thermal.g_in)
    Nx_f = patch.Nx_inner + 2 * patch.n_ghost
    Ny_f = patch.Ny_inner + 2 * patch.n_ghost
    Nz_f = patch.Nz_inner + 2 * patch.n_ghost
    kernel! = _fill_thermal_full_3d_kernel!(backend)
    kernel!(thermal.g_in, g_coarse, patch.ratio, patch.n_ghost,
            first(patch.parent_i_range), first(patch.parent_j_range), first(patch.parent_k_range),
            Nx_c, Ny_c, Nz_c; ndrange=(Nx_f, Ny_f, Nz_f))
    copyto!(thermal.g_out, thermal.g_in)
end

"""
    restrict_thermal_to_coarse_3d!(patch::RefinementPatch3D{T},

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using Kraken

methods(Kraken.restrict_thermal_to_coarse_3d!)
```
"""
function restrict_thermal_to_coarse_3d!(patch::RefinementPatch3D{T},
                                         thermal::ThermalPatchArrays3D{T},
                                         g_coarse, Temp_c) where T
    Nx_ov = length(patch.parent_i_range)
    Ny_ov = length(patch.parent_j_range)
    Nz_ov = length(patch.parent_k_range)
    backend = KernelAbstractions.get_backend(g_coarse)
    kernel! = _restrict_thermal_simple_3d_kernel!(backend)
    kernel!(g_coarse, Temp_c, thermal.g_in, thermal.Temp,
            patch.ratio, patch.n_ghost,
            first(patch.parent_i_range), first(patch.parent_j_range), first(patch.parent_k_range);
            ndrange=(Nx_ov, Ny_ov, Nz_ov))
end

# --- Thermal refined sub-cycling (3D) ---

"""
    advance_thermal_refined_step_3d!(domain::RefinedDomain3D{T},

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using Kraken

methods(Kraken.advance_thermal_refined_step_3d!)
```
"""
function advance_thermal_refined_step_3d!(domain::RefinedDomain3D{T},
                                            thermals::Vector{ThermalPatchArrays3D{T}},
                                            f_in, f_out, g_in, g_out,
                                            rho, ux, uy, uz, Temp, is_solid;
                                            fused_step_fn,
                                            omega_T_coarse::Real,
                                            β_g::Real=zero(T),
                                            T_ref_buoy::Real=zero(T),
                                            bc_thermal_patch_fns=nothing,
                                            bc_flow_patch_fns=nothing,
                                            bc_coarse_fn=nothing) where T
    Nx = domain.base_Nx; Ny = domain.base_Ny; Nz = domain.base_Nz

    # 1. Save coarse state at time n (flow + thermal)
    for (pidx, patch) in enumerate(domain.patches)
        save_coarse_state_3d!(patch, f_in, rho, ux, uy, uz)
        save_thermal_coarse_state_3d!(patch, thermals[pidx], g_in, Temp)
    end

    # 2. Advance coarse grid one step (fused kernel)
    fused_step_fn(f_out, f_in, g_out, g_in, Temp, Nx, Ny, Nz)
    f_in, f_out = f_out, f_in
    g_in, g_out = g_out, g_in

    # Restore patch-covered cells from saved state
    for (pidx, patch) in enumerate(domain.patches)
        _restore_coarse_overlap_3d!(f_in, g_in, patch, thermals[pidx])
    end

    # Recover macroscopic on coarse for ghost fill
    compute_macroscopic_3d!(rho, ux, uy, uz, f_in)
    compute_temperature_3d!(Temp, g_in)

    # 3. Sub-cycle each patch
    for (pidx, patch) in enumerate(domain.patches)
        thermal = thermals[pidx]
        ratio = patch.ratio

        for sub_step in 1:ratio
            t_frac = T((sub_step - 1) / ratio)

            # Flow ghost fill: temporal interpolation with FH rescaling
            fill_ghost_temporal_3d!(patch, f_in, rho, ux, uy, uz,
                                    domain.base_omega, Nx, Ny, Nz, t_frac)
            # Thermal ghost fill: trilinear temporal (no FH)
            fill_thermal_ghost_3d!(patch, thermal, g_in, Nx, Ny, Nz;
                                    t_frac=t_frac)

            # Stream both f and g on patch
            stream_3d!(patch.f_out, patch.f_in, patch.Nx, patch.Ny, patch.Nz)
            stream_3d!(thermal.g_out, thermal.g_in, patch.Nx, patch.Ny, patch.Nz)

            # Flow BCs on patch
            if bc_flow_patch_fns !== nothing
                bc_f_fn = get(bc_flow_patch_fns, pidx, nothing)
                if bc_f_fn !== nothing
                    bc_f_fn(patch.f_out, patch.Nx, patch.Ny, patch.Nz)
                end
            end

            # Thermal BCs on patch
            if bc_thermal_patch_fns !== nothing
                bc_fn = get(bc_thermal_patch_fns, pidx, nothing)
                if bc_fn !== nothing
                    bc_fn(thermal.g_out, patch.Nx, patch.Ny, patch.Nz)
                end
            end

            # Recover temperature and macroscopic flow
            compute_temperature_3d!(thermal.Temp, thermal.g_out)
            compute_macroscopic_3d!(patch.rho, patch.ux, patch.uy, patch.uz, patch.f_out)

            # Collide thermal (BGK on g)
            collide_thermal_3d!(thermal.g_out, patch.ux, patch.uy, patch.uz, thermal.omega_T)
            # Collide flow with Boussinesq (acoustic scaling: β_g / ratio)
            collide_boussinesq_3d!(patch.f_out, thermal.Temp, patch.is_solid,
                                    patch.omega, T(β_g) / T(patch.ratio),
                                    T(T_ref_buoy))

            # Swap for next sub-step
            copyto!(patch.f_in, patch.f_out)
            copyto!(thermal.g_in, thermal.g_out)
        end
    end

    # 4. Restrict fine results back to coarse
    for (pidx, patch) in enumerate(domain.patches)
        thermal = thermals[pidx]
        restrict_to_coarse_3d!(patch, f_in, rho, ux, uy, uz, domain.base_omega)
        restrict_thermal_to_coarse_3d!(patch, thermal, g_in, Temp)
    end

    # 5. Re-apply coarse BCs (restriction may overwrite wall values)
    if bc_coarse_fn !== nothing
        bc_coarse_fn(f_in, g_in, Temp, Nx, Ny, Nz)
    end
    compute_macroscopic_3d!(rho, ux, uy, uz, f_in)
    compute_temperature_3d!(Temp, g_in)

    return f_in, f_out, g_in, g_out
end

# --- Auto-detect 3D patch faces touching domain walls for thermal BCs ---

"""
    build_patch_thermal_bcs_3d(patches, Lx, Ly, Lz, Nx, temp_bcs)

Build BC closures for 3D patch faces that touch domain walls with Dirichlet
temperature BCs. `temp_bcs` is a Dict mapping face symbols to wall temperatures,
e.g. `Dict(:west => 1.0, :east => 0.0)`.

Returns Dict{Int, Function} mapping patch index → BC closure `(g, Nx_p, Ny_p, Nz_p) -> ...`.
"""
function build_patch_thermal_bcs_3d(patches::Vector{RefinementPatch3D{T}},
                                     Lx, Ly, Lz, Nx,
                                     temp_bcs::Dict{Symbol, <:Real}) where T
    tol = Lx / Nx * 0.01

    face_checks = [
        (:west,   p -> Float64(p.x_min) <= tol),
        (:east,   p -> Float64(p.x_max) >= Lx - tol),
        (:south,  p -> Float64(p.y_min) <= tol),
        (:north,  p -> Float64(p.y_max) >= Ly - tol),
        (:bottom, p -> Float64(p.z_min) <= tol),
        (:top,    p -> Float64(p.z_max) >= Lz - tol),
    ]

    # One closure per patch applies all the Dirichlet faces the patch touches
    # in a single `apply_fixed_temp_3d!`-style call, so a patch along an edge
    # of two Dirichlet faces flips the edge slots once.
    patch_bcs = Dict{Int, Function}()
    for (pidx, patch) in enumerate(patches)
        touched = Dict{Symbol, T}()
        for (face, check_fn) in face_checks
            check_fn(patch) || continue
            haskey(temp_bcs, face) || continue
            touched[face] = T(temp_bcs[face])
        end

        if !isempty(touched)
            let tt = _fixed_temp_set_3d(touched), tf = _FIXED_TEMP_FACES_3D
                patch_bcs[pidx] = (g, Nx_p, Ny_p, Nz_p) ->
                    _apply_fixed_temp_faces_3d!(g, Nx_p, Ny_p, Nz_p, tt, tf)
            end
        end
    end
    return patch_bcs
end

"""
    build_patch_flow_bcs_3d(patches::Vector{RefinementPatch3D{T}},

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using Kraken

methods(Kraken.build_patch_flow_bcs_3d)
```
"""
function build_patch_flow_bcs_3d(patches::Vector{RefinementPatch3D{T}},
                                  Lx, Ly, Lz, Nx;
                                  wall_faces=Symbol[]) where T
    tol = Lx / Nx * 0.01

    patch_bcs = Dict{Int, Function}()
    for (pidx, patch) in enumerate(patches)
        closures = Function[]

        for (face, condition) in [
            (:west,   Float64(patch.x_min) <= tol),
            (:east,   Float64(patch.x_max) >= Lx - tol),
            (:south,  Float64(patch.y_min) <= tol),
            (:north,  Float64(patch.y_max) >= Ly - tol),
            (:bottom, Float64(patch.z_min) <= tol),
            (:top,    Float64(patch.z_max) >= Lz - tol),
        ]
            condition || continue
            face in wall_faces || continue
            let f_sym = face
                push!(closures, (f, Nx_p, Ny_p, Nz_p) ->
                    apply_bounce_back_wall_3d!(f, Nx_p, Ny_p, Nz_p, f_sym))
            end
        end

        if !isempty(closures)
            let cls = closures
                patch_bcs[pidx] = (f, Nx_p, Ny_p, Nz_p) -> begin
                    for cl in cls
                        cl(f, Nx_p, Ny_p, Nz_p)
                    end
                end
            end
        end
    end
    return patch_bcs
end
