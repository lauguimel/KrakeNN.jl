# WP-MESH-6 approach (E-full) — 8-block body-fitted O-grid-in-rectangle
# cylinder Re=100. This is the paper-killer figure: multi-block
# structured LBM with LI-BB per block on a body-fitted grid that
# reproduces the cylinder as a true curved boundary.
#
# ⚠️ WORK-IN-PROGRESS — NOT YET VALIDATED. Two infra issues identified
# during Metal smoke (steps labelled {I1, I2} in NEXT_SESSION_V0_3_B2.md):
#
#   (I1) `layout=:topological` walker starts from an arbitrary corner,
#        so a loaded block's ξ/η may map to {radial, angular} in a
#        way that makes the loader's corner-matching assign edge
#        tags to the WRONG curve. E.g., a block's `:west` edge (i=1)
#        after topological walking can coincide geometrically with a
#        spoke (neighbour interface) but inherit the `:cylinder` tag
#        from the actually-nearby arc. This must be fixed either by
#        (a) respecting the Transfinite Surface corner ordering
#        from the .geo, or (b) re-identifying physical BCs post-load
#        via geometry (e.g., "cells at radius R_in are cylinder").
#
#   (I2) `extend_mesh_2d` uses linear extrapolation, which FOLDS the
#        Jacobian on blocks with tight curved boundaries. For an
#        O-grid ring block with inner arc at R_in = 0.025, extending
#        the mesh by 1 ghost row "inward" (toward the cylinder
#        centre) produces a fold at some cells when the radial step
#        is comparable to R_in. Fix: mirror extension across the
#        wall (geometric reflection) OR skip extension on the wall-
#        facing edge and have the SLBM kernel operate on the interior
#        size (size mismatch with the extended state buffers — this
#        would need a "virtual ghost" geometry layer instead of a
#        rebuilt extended mesh).
#
# Do NOT submit this to Aqua until (I1) and (I2) are resolved.
# See `NEXT_SESSION_V0_3_B2.md` for the recommended fix order.
#
# Topology (`tmp/gen_ogrid_rect_8block.jl`) — 8 ring blocks:
#   ring_k (k=0..7) spans angular sector [k·45°, (k+1)·45°] from the
#   cylinder surface (R_in=0.025) to a point on the rectangle boundary
#   (either a corner or an edge midpoint, alternating). Each ring
#   block is 4-sided, cylinder arc on one edge, rectangle segment on
#   the opposite edge, 2 spokes shared with neighbours as interfaces.
#
# After `autoreorient_blocks`, every ring_k has its cylinder arc on
# `:west`, its outer tag on `:east`, and interfaces on `:south`/`:north`.
#
# Step pipeline per block:
#   exchange_ghost_shared_node_2d!(mbm, states)   # 8 interfaces
#   fill_physical_wall_ghost_2d!(mbm, states)     # cylinder + outer walls
#   for each block:
#       slbm_trt_libb_step_local_2d!(f_out[k], states[k].f, ρ[k], ux[k], uy[k],
#                                      is_solid_ext[k], q_wall_ext[k], uw_*,
#                                      geom_ext[k], sp[k], sm[k])
#       apply_bc_rebuild_2d!(interior view of f_out[k], ..., bcspec[k], ν, Nξ, Nη;
#                              sp_field=sp_int, sm_field=sm_int)
#   swap states[k].f ↔ f_out[k]
#
# Per-block BCSpec (after autoreorient, west=cylinder, east=outer):
#   ring_0, ring_7 : west=HalfwayBB (cyl, wall_ghost handles), east=ZouHePressure
#   ring_1, ring_2 : west=HalfwayBB (cyl),                    east=HalfwayBB (wall_top)
#   ring_3, ring_4 : west=HalfwayBB (cyl),                    east=ZouHeVelocity(u_in)
#   ring_5, ring_6 : west=HalfwayBB (cyl),                    east=HalfwayBB (wall_bot)
#   south/north    : HalfwayBB (interfaces, no-op)
#
# Drag aggregation: sum compute_drag_libb_mei_2d on each of the 8
# ring blocks' INTERIOR views (each contributes its 45° arc contribution).

using KrakeNN, CUDA, KernelAbstractions, Gmsh
using FFTW: rfft, rfftfreq

include(joinpath(@__DIR__, "..", "tmp", "gen_ogrid_rect_8block.jl"))

const Lx, Ly         = 1.0, 0.5
const cx_p, cy_p     = 0.5, 0.245
const R_p            = 0.025
const Re_target      = 100.0
const u_max          = 0.04
const u_mean         = u_max
const Cd_ref         = 1.4
const Cl_ref         = 0.33
const St_ref         = 0.165
T = Float64

backend = CUDABackend()

function _strouhal(cl, dt_lu, D_lu)
    cl_z = cl .- (sum(cl) / length(cl))
    F = abs.(rfft(cl_z))
    fs = rfftfreq(length(cl_z), 1 / dt_lu)
    idx = argmax(F[2:end]) + 1
    return fs[idx] * D_lu / Float64(u_mean)
end

function _stats(cd, cl)
    cd_mean = sum(cd) / length(cd)
    cl_mean = sum(cl) / length(cl)
    cl_rms  = sqrt(sum((cl .- cl_mean).^2) / length(cl))
    return cd_mean, cl_rms
end

function _format(label, cells, cd, clr, St, mlups, elapsed)
    err_cd  = 100*abs(cd - Cd_ref)/Cd_ref
    err_clr = 100*abs(clr - Cl_ref)/Cl_ref
    err_st  = 100*abs(St - St_ref)/St_ref
    println("[$label cells=$cells] Cd=$(round(cd,digits=3)) (err $(round(err_cd,digits=2))%)  ",
            "Cl_RMS=$(round(clr,digits=3)) (err $(round(err_clr,digits=2))%)  ",
            "St=$(round(St,digits=3)) (err $(round(err_st,digits=2))%)  | ",
            "$(round(mlups,digits=0)) MLUPS  $(round(elapsed,digits=1))s")
end

# Per-block BCSpec from the block's :east tag (which carries the
# outer physical-boundary tag after autoreorient). :west (cylinder)
# and :south/:north (interfaces) are all HalfwayBB — the kernel's
# pull + wall_ghost_fill + exchange_ghost_shared_node_2d! already
# handle them. Only the :east side needs a real apply_bc_rebuild call.
function _build_bcspec(block, u_prof_cu, T)
    east_tag = block.boundary_tags.east
    east_bc = if east_tag === :inlet
        ZouHeVelocity(u_prof_cu)
    elseif east_tag === :outlet
        ZouHePressure(one(T))
    else  # :wall_top, :wall_bot, :cylinder (should not happen on east after reorient)
        HalfwayBB()
    end
    return BCSpec2D(; west=HalfwayBB(), east=east_bc,
                      south=HalfwayBB(), north=HalfwayBB())
end

run_Efull(D_lu, steps, sw, se; N_arc=8, N_radial_factor=2) = begin
    # N_arc: nodes per block along the angular direction (8 blocks cover
    #        2π, so total angular resolution ≈ 8·(N_arc-1) cells).
    # N_radial: nodes per block along the radial direction.
    # We scale both with D_lu so the cell count tracks the other matrix
    # rows at the same D_lu.
    N_arc_k    = max(8, round(Int, D_lu / 2))   # rough scaling
    N_radial_k = max(8, round(Int, D_lu * N_radial_factor))
    # Adaptive Progression with CORRECT direction (sign):
    # Spoke is Line(p_k → q_k), p_k = cylinder (inner, START), q_k =
    # rectangle (outer, END). gmsh convention: Progression d > 1 ⇒
    # cells smaller at START (= cylinder side). For body-fitted
    # resolution we want finest cells on the cylinder surface, so
    # d > 1 is required. target_ratio = dx_outer/dx_inner (>1).
    # Previous (-1.0 exponent) gave d < 1 which clustered at the
    # RECTANGLE instead, collapsing radial cells at the outer wall
    # to 1e-9 and breaking SLBM.
    target_ratio = 50.0
    radial_prog  = target_ratio^(+1.0 / (N_radial_k - 1))

    mktempdir() do dir
        geo_path = joinpath(dir, "ogrid_rect_8block.geo")
        write_ogrid_rect_8block_geo(geo_path;
                                      Lx=Lx, Ly=Ly, cx_p=cx_p, cy_p=cy_p,
                                      R_in=R_p,
                                      N_arc=N_arc_k, N_radial=N_radial_k,
                                      radial_progression=radial_prog)

        mbm_raw, _ = load_gmsh_multiblock_2d(geo_path; FT=T, layout=:topological)
        mbm = autoreorient_blocks(mbm_raw; verbose=false)
        issues = sanity_check_multiblock(mbm; verbose=false)
        any(iss -> iss.severity === :error, issues) &&
            error("approach (E-full) sanity has :error issues: $issues")

        # dx_ref (for ν and drag normalization): pick the smallest cell
        # across all ring blocks, which is typically on the cylinder arc
        # edge. Same interpretation as (D): ν in "lattice units relative
        # to dx_ref".
        dx_ref = minimum(b.mesh.dx_ref for b in mbm.blocks)
        R_lu   = R_p / dx_ref
        ν      = u_mean * (2 * R_lu) / Re_target    # u·D/Re in lattice units

        # --- Per-block state, SLBM geometry, LI-BB precompute ---
        ng = 1
        n_blocks = length(mbm.blocks)
        states = [allocate_block_state_2d(b; n_ghost=ng, backend=backend)
                  for b in mbm.blocks]

        geom_ext = Vector{Any}(undef, n_blocks)
        sp_ext   = Vector{Any}(undef, n_blocks)
        sm_ext   = Vector{Any}(undef, n_blocks)
        qwall_ext = Vector{Any}(undef, n_blocks)
        uwx_ext   = Vector{Any}(undef, n_blocks)
        uwy_ext   = Vector{Any}(undef, n_blocks)
        solid_ext = Vector{Any}(undef, n_blocks)
        sp_int_cu = Vector{Any}(undef, n_blocks)   # for apply_bc_rebuild local-τ
        sm_int_cu = Vector{Any}(undef, n_blocks)

        for (k, blk) in enumerate(mbm.blocks)
            mesh_ext, g_h = build_block_slbm_geometry_extended(blk; n_ghost=ng,
                                                                   local_cfl=true)
            geom_ext[k] = transfer_slbm_geometry(g_h, backend)
            # Local-τ fields on the extended mesh.
            sp_h, sm_h = compute_local_omega_2d(mesh_ext; ν=Float64(ν),
                                                  scaling=:quadratic, τ_floor=0.51)
            sp_ext[k] = CuArray(T.(sp_h))
            sm_ext[k] = CuArray(T.(sm_h))
            # LI-BB precompute on the INTERIOR mesh of this ring block.
            is_solid_int = zeros(Bool, blk.mesh.Nξ, blk.mesh.Nη)
            for j in 1:blk.mesh.Nη, i in 1:blk.mesh.Nξ
                x = blk.mesh.X[i, j]; y = blk.mesh.Y[i, j]
                (x - cx_p)^2 + (y - cy_p)^2 ≤ R_p^2 && (is_solid_int[i, j] = true)
            end
            qw_int_h, uwx_int_h, uwy_int_h =
                precompute_q_wall_slbm_cylinder_2d(blk.mesh, is_solid_int,
                                                     cx_p, cy_p, R_p)
            # Lift to extended (zeros on ghost layer).
            solid_ext[k]  = CuArray(extend_interior_field_2d(is_solid_int, ng))
            qwall_ext[k]  = CuArray(T.(extend_interior_field_2d(qw_int_h, ng)))
            uwx_ext[k]    = CuArray(T.(extend_interior_field_2d(uwx_int_h, ng)))
            uwy_ext[k]    = CuArray(T.(extend_interior_field_2d(uwy_int_h, ng)))
            # Interior sp/sm for apply_bc_rebuild (takes (Nξ,Nη) field,
            # not extended — the kernel writes only the boundary row).
            sp_int_cu[k] = CuArray(T.(sp_h[(ng+1):(ng+blk.mesh.Nξ),
                                              (ng+1):(ng+blk.mesh.Nη)]))
            sm_int_cu[k] = CuArray(T.(sm_h[(ng+1):(ng+blk.mesh.Nξ),
                                              (ng+1):(ng+blk.mesh.Nη)]))
        end

        # --- Initial condition: uniform flow u=u_max everywhere ---
        u_prof_cu = CuArray(fill(T(u_max), 2))  # placeholder; see below
        for (k, blk) in enumerate(mbm.blocks)
            Nξk, Nηk = blk.mesh.Nξ, blk.mesh.Nη
            f_int = zeros(T, Nξk, Nηk, 9)
            usq = u_max * u_max
            for j in 1:Nηk, i in 1:Nξk, q in 1:9
                x = blk.mesh.X[i, j]; y = blk.mesh.Y[i, j]
                inside = (x - cx_p)^2 + (y - cy_p)^2 ≤ R_p^2
                u = inside ? 0.0 : Float64(u_max)
                f_int[i, j, q] = T(KrakeNN.equilibrium(D2Q9(), 1.0, u, 0.0, q))
            end
            int_view = interior_f(states[k])
            copyto!(int_view, CuArray(f_int))
        end

        # --- Per-block f_out buffers + per-block bcspec ---
        f_out = [similar(st.f) for st in states]
        for k in 1:n_blocks; fill!(f_out[k], zero(T)); end

        # Build per-block inlet profile sized to Nη of that block.
        u_prof_per_block = [CuArray(fill(T(u_max), blk.mesh.Nη))
                             for blk in mbm.blocks]
        bcspecs = [_build_bcspec(blk, u_prof_per_block[k], T)
                   for (k, blk) in enumerate(mbm.blocks)]

        # --- Time loop ---
        cd_h = Float64[]; cl_h = Float64[]
        norm_drag = u_mean^2 * (R_lu * 2)

        t0 = time()
        for step in 1:steps
            exchange_ghost_shared_node_2d!(mbm, states)
            fill_physical_wall_ghost_2d!(mbm, states)
            for k in 1:n_blocks
                slbm_trt_libb_step_local_2d!(f_out[k], states[k].f,
                                               states[k].ρ, states[k].ux, states[k].uy,
                                               solid_ext[k], qwall_ext[k],
                                               uwx_ext[k], uwy_ext[k],
                                               geom_ext[k], sp_ext[k], sm_ext[k])
            end
            for (k, blk) in enumerate(mbm.blocks)
                Nξp = states[k].Nξ_phys; Nηp = states[k].Nη_phys
                ngk = states[k].n_ghost
                int_f_out = view(f_out[k],        (ngk+1):(ngk+Nξp), (ngk+1):(ngk+Nηp), :)
                int_f_in  = view(states[k].f,     (ngk+1):(ngk+Nξp), (ngk+1):(ngk+Nηp), :)
                apply_bc_rebuild_2d!(int_f_out, int_f_in, bcspecs[k], ν, Nξp, Nηp;
                                       sp_field=sp_int_cu[k], sm_field=sm_int_cu[k])
            end
            # Drag sampling: aggregate over all 8 ring blocks' INTERIOR views.
            if step > steps - sw && step % se == 0
                Fx_tot = zero(T); Fy_tot = zero(T)
                for (k, blk) in enumerate(mbm.blocks)
                    Nξp = states[k].Nξ_phys; Nηp = states[k].Nη_phys
                    ngk = states[k].n_ghost
                    int_f_out = view(f_out[k],  (ngk+1):(ngk+Nξp), (ngk+1):(ngk+Nηp), :)
                    int_q  = view(qwall_ext[k], (ngk+1):(ngk+Nξp), (ngk+1):(ngk+Nηp), :)
                    int_ux = view(uwx_ext[k],   (ngk+1):(ngk+Nξp), (ngk+1):(ngk+Nηp), :)
                    int_uy = view(uwy_ext[k],   (ngk+1):(ngk+Nξp), (ngk+1):(ngk+Nηp), :)
                    Fx, Fy = compute_drag_libb_mei_2d(int_f_out, int_q, int_ux, int_uy,
                                                       Nξp, Nηp)
                    Fx_tot += T(Fx); Fy_tot += T(Fy)
                end
                push!(cd_h, 2 * Float64(Fx_tot) / norm_drag)
                push!(cl_h, 2 * Float64(Fy_tot) / norm_drag)
            end
            for k in 1:n_blocks
                states[k].f, f_out[k] = f_out[k], states[k].f
            end
        end
        CUDA.synchronize()
        elapsed = time() - t0
        cd, clr = _stats(cd_h, cl_h)
        St = _strouhal(cl_h, se, 2 * R_lu)
        cells = sum(b.mesh.Nξ * b.mesh.Nη for b in mbm.blocks)
        mlups = cells * steps / elapsed / 1e6
        _format("Efull D=$D_lu", cells, cd, clr, St, mlups, elapsed)
        println("  (Efull): 8 blocks, N_arc=$(N_arc_k)×N_radial=$(N_radial_k) per block, " *
                "total $cells cells,  dx_ref=$(round(dx_ref, sigdigits=4))")
        return (; cells, cd, cl_rms=clr, St, mlups, elapsed)
    end
end

println("=== WP-MESH-6 (E-full) — 8-block O-grid body-fitted cylinder Re=100 ===")
println("Reference (Williamson 1996): Cd≈$Cd_ref  Cl_RMS≈$Cl_ref  St≈$St_ref")
println("Domain $Lx × $Ly  cylinder ($cx_p, $cy_p) R=$R_p\n")

for D_lu in (20, 40, 80)
    steps = D_lu == 20 ? 80_000 : D_lu == 40 ? 160_000 : 320_000
    sw = steps ÷ 2
    se = D_lu == 80 ? 20 : 10
    println("--- D_lu=$D_lu  steps=$steps  sample_every=$se  window=$sw ---")
    try
        run_Efull(D_lu, steps, sw, se)
    catch e
        println("  Efull FAILED: ", sprint(showerror, e)[1:min(end, 400)])
    end
end

println("\n=== Done ===")
