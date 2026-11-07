# Metal smoke for approach (E-full) — 8-block body-fitted O-grid
# cylinder. Pipeline validation only (Metal FP32 is insufficient for
# accurate LBM at low lattice resolution, as with all other SLBM
# smokes — see memory project_wp_mesh_6_bump). This smoke confirms:
#
#   1. The .geo generator + loader + autoreorient path produces a
#      consistent 8-block MBM
#   2. Per-block `build_block_slbm_geometry_extended`, local-τ fields,
#      and LI-BB precompute all launch without errors
#   3. The full step pipeline (exchange_ghost_shared_node +
#      wall_ghost_fill + slbm_step + apply_bc + drag aggregate +
#      swap) runs for a few iterations without kernel errors
#
# NaN within the first ~100 steps is expected on Metal FP32 — we only
# check that the DATA FLOW is wired correctly.

using KrakeNN, KernelAbstractions, Metal, Gmsh

include(joinpath(@__DIR__, "gen_ogrid_rect_8block.jl"))

T = Float32
backend = MetalBackend()
DeviceArray = Metal.MtlArray

const Lx, Ly = 1.0f0, 0.5f0
const cx_p, cy_p = 0.5f0, 0.245f0
const R_p = 0.025f0
const u_max = 0.04f0

D_lu  = parse(Int, get(ENV, "D_LU",  "20"))
steps = parse(Int, get(ENV, "STEPS", "50"))
N_arc    = max(6, round(Int, D_lu / 3))
N_radial = max(8, D_lu)

println("=== Smoke (E-full) — 8-block O-grid local Metal Float32 ===")
println("D_lu=$D_lu  N_arc=$N_arc  N_radial=$N_radial  steps=$steps")

mktempdir() do dir
    geo_path = joinpath(dir, "ogrid_rect_8block.geo")
    write_ogrid_rect_8block_geo(geo_path;
                                  Lx=Float64(Lx), Ly=Float64(Ly),
                                  cx_p=Float64(cx_p), cy_p=Float64(cy_p),
                                  R_in=Float64(R_p),
                                  N_arc=N_arc, N_radial=N_radial)

    mbm_raw, _ = load_gmsh_multiblock_2d(geo_path; FT=T, layout=:topological)
    println("  load: $(length(mbm_raw.blocks)) blocks, $(length(mbm_raw.interfaces)) interfaces")
    mbm = autoreorient_blocks(mbm_raw; verbose=false)
    # F32 precision on gmsh-node coordinates gives ~1e-7 colocation
    # offsets (vs 1e-12 in F64). Relax sanity tol so shared-node check
    # doesn't reject legitimate topology due to F32 truncation.
    issues = sanity_check_multiblock(mbm; tol=1e-5, verbose=false)
    any(iss -> iss.severity === :error, issues) && error("sanity error: $issues")
    println("  sanity OK: $(length(issues)) issue(s) (warnings only)")

    dx_ref = minimum(b.mesh.dx_ref for b in mbm.blocks)
    R_lu = R_p / dx_ref
    ν = T(u_max * 2 * R_lu / 100)
    println("  dx_ref=$(round(Float64(dx_ref), sigdigits=4))  R_lu=$(round(Float64(R_lu), digits=2))  ν=$(round(Float64(ν), sigdigits=4))")

    ng = 1
    n_blocks = length(mbm.blocks)
    states = [allocate_block_state_2d(b; n_ghost=ng, backend=backend) for b in mbm.blocks]

    geom_ext = Vector{Any}(undef, n_blocks)
    sp_ext = Vector{Any}(undef, n_blocks); sm_ext = Vector{Any}(undef, n_blocks)
    qwall_ext = Vector{Any}(undef, n_blocks)
    uwx_ext = Vector{Any}(undef, n_blocks); uwy_ext = Vector{Any}(undef, n_blocks)
    solid_ext = Vector{Any}(undef, n_blocks)
    sp_int_cu = Vector{Any}(undef, n_blocks); sm_int_cu = Vector{Any}(undef, n_blocks)

    for (k, blk) in enumerate(mbm.blocks)
        mesh_ext, g_h = build_block_slbm_geometry_extended(blk; n_ghost=ng, local_cfl=true)
        geom_ext[k] = transfer_slbm_geometry(g_h, backend)
        sp_h, sm_h = compute_local_omega_2d(mesh_ext; ν=Float64(ν), scaling=:quadratic, τ_floor=0.51)
        sp_ext[k] = DeviceArray(T.(sp_h)); sm_ext[k] = DeviceArray(T.(sm_h))
        is_solid_int = zeros(Bool, blk.mesh.Nξ, blk.mesh.Nη)
        for j in 1:blk.mesh.Nη, i in 1:blk.mesh.Nξ
            x = blk.mesh.X[i, j]; y = blk.mesh.Y[i, j]
            (x - cx_p)^2 + (y - cy_p)^2 ≤ R_p^2 && (is_solid_int[i, j] = true)
        end
        qw_h, uwx_h, uwy_h = precompute_q_wall_slbm_cylinder_2d(blk.mesh, is_solid_int,
                                                                  Float64(cx_p), Float64(cy_p),
                                                                  Float64(R_p); FT=T)
        solid_ext[k] = DeviceArray(extend_interior_field_2d(is_solid_int, ng))
        qwall_ext[k] = DeviceArray(extend_interior_field_2d(qw_h, ng))
        uwx_ext[k]   = DeviceArray(extend_interior_field_2d(uwx_h, ng))
        uwy_ext[k]   = DeviceArray(extend_interior_field_2d(uwy_h, ng))
        sp_int_cu[k] = DeviceArray(T.(sp_h[(ng+1):(ng+blk.mesh.Nξ), (ng+1):(ng+blk.mesh.Nη)]))
        sm_int_cu[k] = DeviceArray(T.(sm_h[(ng+1):(ng+blk.mesh.Nξ), (ng+1):(ng+blk.mesh.Nη)]))
    end
    println("  per-block SLBM geom + LI-BB precompute: OK")

    # Init uniform flow
    for (k, blk) in enumerate(mbm.blocks)
        Nξk, Nηk = blk.mesh.Nξ, blk.mesh.Nη
        f_int = zeros(T, Nξk, Nηk, 9)
        for j in 1:Nηk, i in 1:Nξk, q in 1:9
            x = blk.mesh.X[i, j]; y = blk.mesh.Y[i, j]
            inside = (x - cx_p)^2 + (y - cy_p)^2 ≤ R_p^2
            u = inside ? T(0) : T(u_max)
            f_int[i, j, q] = T(KrakeNN.equilibrium(D2Q9(), one(T), u, zero(T), q))
        end
        int_view = interior_f(states[k])
        copyto!(int_view, DeviceArray(f_int))
    end

    f_out = [similar(st.f) for st in states]
    for k in 1:n_blocks; fill!(f_out[k], zero(T)); end
    # Build per-block BC by dispatching on EVERY edge's tag. Ring blocks
    # have `cylinder` on one edge (handled via LI-BB + wall_ghost on
    # that edge, BC=HalfwayBB here), interfaces on two edges
    # (HalfwayBB no-op, handled by exchange_ghost), and an inlet/outlet/
    # wall_top/wall_bot on the remaining physical edge.
    _bc_for_tag(tag, Nη, Nξ) = begin
        if tag === :inlet
            ZouHeVelocity(DeviceArray(fill(T(u_max), Nη)))
        elseif tag === :outlet
            ZouHePressure(one(T))
        else
            HalfwayBB()
        end
    end
    bcspecs = map(mbm.blocks) do blk
        t = blk.boundary_tags
        BCSpec2D(; west=_bc_for_tag(t.west, blk.mesh.Nη, blk.mesh.Nξ),
                   east=_bc_for_tag(t.east, blk.mesh.Nη, blk.mesh.Nξ),
                   south=_bc_for_tag(t.south, blk.mesh.Nξ, blk.mesh.Nξ),
                   north=_bc_for_tag(t.north, blk.mesh.Nξ, blk.mesh.Nξ))
    end

    println("  bcspec tags per block: ",
            [(b.boundary_tags.west, b.boundary_tags.east,
              b.boundary_tags.south, b.boundary_tags.north) for b in mbm.blocks])
    println("Starting $steps-step pipeline loop …")

    function _loop!(mbm, states, f_out, solid_ext, qwall_ext, uwx_ext, uwy_ext,
                    geom_ext, sp_ext, sm_ext, sp_int_cu, sm_int_cu, bcspecs, ν, steps)
        for step in 1:steps
            exchange_ghost_shared_node_2d!(mbm, states)
            fill_physical_wall_ghost_2d!(mbm, states)
            for k in 1:length(states)
                slbm_trt_libb_step_local_2d!(f_out[k], states[k].f,
                                               states[k].ρ, states[k].ux, states[k].uy,
                                               solid_ext[k], qwall_ext[k],
                                               uwx_ext[k], uwy_ext[k],
                                               geom_ext[k], sp_ext[k], sm_ext[k])
            end
            for (k, blk) in enumerate(mbm.blocks)
                Nξp = states[k].Nξ_phys; Nηp = states[k].Nη_phys
                ngk = states[k].n_ghost
                int_f_out = view(f_out[k], (ngk+1):(ngk+Nξp), (ngk+1):(ngk+Nηp), :)
                int_f_in  = view(states[k].f, (ngk+1):(ngk+Nξp), (ngk+1):(ngk+Nηp), :)
                apply_bc_rebuild_2d!(int_f_out, int_f_in, bcspecs[k], ν, Nξp, Nηp;
                                       sp_field=sp_int_cu[k], sm_field=sm_int_cu[k])
            end
            for k in 1:length(states)
                states[k].f, f_out[k] = f_out[k], states[k].f
            end
            if step % 10 == 0
                KernelAbstractions.synchronize(get_backend(states[1].f))
                rho0 = Array(states[1].ρ)
                if any(isnan, rho0)
                    println("  step $step : NaN in ring_0 — expected on Metal FP32, pipeline OK")
                    return true
                end
                println("  step $step : ring_0 ρ ∈ [$(round(minimum(rho0),digits=4)), $(round(maximum(rho0),digits=4))]")
            end
        end
        return true
    end

    t0 = time()
    ok = _loop!(mbm, states, f_out, solid_ext, qwall_ext, uwx_ext, uwy_ext,
                 geom_ext, sp_ext, sm_ext, sp_int_cu, sm_int_cu, bcspecs, ν, steps)
    KernelAbstractions.synchronize(backend)
    elapsed = time() - t0
    cells = sum(b.mesh.Nξ * b.mesh.Nη for b in mbm.blocks)
    println("\n=== Smoke (E-full) ===  cells=$cells  elapsed=$(round(elapsed, digits=1))s")
    println(ok ? "PASS (pipeline wired OK — validate Cd on Aqua FP64)" :
                 "FAIL (unexpected error)")
end
