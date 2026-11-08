# Reproduce E-full locally CPU F64 to find when NaN appears
using KrakeNN, Gmsh, KernelAbstractions

include(joinpath(@__DIR__, "gen_ogrid_rect_8block.jl"))

const T = Float64
const backend = KernelAbstractions.CPU()
const Lx, Ly = 1.0, 0.5
const cx_p, cy_p = 0.5, 0.245
const R_p = 0.025
const u_max = 0.04
const u_mean = u_max
const Re_target = 20.0

D_lu = 20
N_arc_k = max(8, round(Int, D_lu/2))    # 10
N_radial_k = max(8, D_lu*2)              # 40
radial_prog = 50.0^(+1.0/(N_radial_k - 1))  # fine cells at cylinder, coarse at rectangle
steps = 200

mktempdir() do dir
    geo_path = joinpath(dir, "o.geo")
    write_ogrid_rect_8block_geo(geo_path; Lx=Lx, Ly=Ly, cx_p=cx_p, cy_p=cy_p,
                                  R_in=R_p, N_arc=N_arc_k, N_radial=N_radial_k,
                                  radial_progression=radial_prog)
    mbm_raw, _ = load_gmsh_multiblock_2d(geo_path; FT=T, layout=:topological)
    mbm = autoreorient_blocks(mbm_raw; verbose=false)
    dx_ref = minimum(b.mesh.dx_ref for b in mbm.blocks)
    R_lu = R_p/dx_ref
    ν = u_mean * 2 * R_lu / Re_target
    println("D_lu=$D_lu dx_ref=$(round(dx_ref,sigdigits=4)) R_lu=$(round(R_lu,sigdigits=4)) ν=$(round(ν,sigdigits=4)) τ=$(round(3ν+0.5,sigdigits=4))")

    ng = 1
    n_blocks = length(mbm.blocks)
    states = [allocate_block_state_2d(b; n_ghost=ng, backend=backend) for b in mbm.blocks]
    geom_ext = Vector{Any}(undef, n_blocks)
    sp_ext = Vector{Any}(undef, n_blocks); sm_ext = Vector{Any}(undef, n_blocks)
    qwall_ext = Vector{Any}(undef, n_blocks); solid_ext = Vector{Any}(undef, n_blocks)
    uwx_ext = Vector{Any}(undef, n_blocks); uwy_ext = Vector{Any}(undef, n_blocks)
    sp_int = Vector{Any}(undef, n_blocks); sm_int = Vector{Any}(undef, n_blocks)
    for (k, blk) in enumerate(mbm.blocks)
        mesh_ext, g_h = build_block_slbm_geometry_extended(blk; n_ghost=ng, local_cfl=false)
        geom_ext[k] = g_h
        sp_h, sm_h = compute_local_omega_2d(mesh_ext; ν=Float64(ν), scaling=:quadratic, τ_floor=0.51)
        sp_ext[k] = T.(sp_h); sm_ext[k] = T.(sm_h)
        is_solid_int = zeros(Bool, blk.mesh.Nξ, blk.mesh.Nη)
        for j in 1:blk.mesh.Nη, i in 1:blk.mesh.Nξ
            x = blk.mesh.X[i,j]; y = blk.mesh.Y[i,j]
            (x-cx_p)^2 + (y-cy_p)^2 ≤ R_p^2 && (is_solid_int[i,j] = true)
        end
        qw_h, uwx_h, uwy_h = precompute_q_wall_slbm_cylinder_2d(blk.mesh, is_solid_int,
                                                                 Float64(cx_p), Float64(cy_p),
                                                                 Float64(R_p); FT=T)
        solid_ext[k] = extend_interior_field_2d(is_solid_int, ng)
        qwall_ext[k] = extend_interior_field_2d(qw_h, ng)
        uwx_ext[k] = extend_interior_field_2d(uwx_h, ng)
        uwy_ext[k] = extend_interior_field_2d(uwy_h, ng)
        sp_int[k] = T.(sp_h[(ng+1):(ng+blk.mesh.Nξ), (ng+1):(ng+blk.mesh.Nη)])
        sm_int[k] = T.(sm_h[(ng+1):(ng+blk.mesh.Nξ), (ng+1):(ng+blk.mesh.Nη)])
        s = sum(is_solid_int); nz = count(x->x>0, qw_h)
        println("Block $k :$(blk.id) Nξ=$(blk.mesh.Nξ) Nη=$(blk.mesh.Nη) solid=$s cut_links=$nz  east=$(blk.boundary_tags.east)")
    end

    for (k, blk) in enumerate(mbm.blocks)
        Nξk, Nηk = blk.mesh.Nξ, blk.mesh.Nη; ngk = states[k].n_ghost
        f_int = zeros(T, Nξk, Nηk, 9)
        solid_int = @view solid_ext[k][(ngk+1):(ngk+Nξk), (ngk+1):(ngk+Nηk)]
        for j in 1:Nηk, i in 1:Nξk, q in 1:9
            u = solid_int[i, j] ? zero(T) : T(u_max)
            f_int[i, j, q] = T(KrakeNN.equilibrium(D2Q9(), 1.0, u, 0.0, q))
        end
        int_view = interior_f(states[k]); int_view .= f_int
    end
    f_out = [similar(st.f) for st in states]
    for k in 1:n_blocks; fill!(f_out[k], zero(T)); end
    # Parabolic inlet profile u(y) = u_max * 4y(Ly-y)/Ly² with u=0 at walls.
    # Per-block inlet profile is sampled at that block's east-edge y coords.
    function _inlet_profile_east(blk)
        Nη = blk.mesh.Nη; Nx = blk.mesh.Nξ
        prof = zeros(T, Nη)
        for j in 1:Nη
            y = blk.mesh.Y[Nx, j]
            prof[j] = T(u_max * 4 * y * (Ly - y) / Ly^2)
        end
        return prof
    end
    bcspecs = map(mbm.blocks) do blk
        t = blk.boundary_tags
        east_bc = if t.east === :inlet
            ZouHeVelocity(_inlet_profile_east(blk))
        elseif t.east === :outlet
            ZouHePressure(one(T))
        else
            HalfwayBB()
        end
        BCSpec2D(; west=HalfwayBB(),
                   east=east_bc,
                   south=HalfwayBB(),
                   north=HalfwayBB())
    end

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
            Nξp = states[k].Nξ_phys; Nηp = states[k].Nη_phys; ngk = states[k].n_ghost
            int_out = view(f_out[k], (ngk+1):(ngk+Nξp), (ngk+1):(ngk+Nηp), :)
            int_in  = view(states[k].f, (ngk+1):(ngk+Nξp), (ngk+1):(ngk+Nηp), :)
            apply_bc_rebuild_2d!(int_out, int_in, bcspecs[k], ν, Nξp, Nηp;
                                   sp_field=sp_int[k], sm_field=sm_int[k])
        end
        for k in 1:n_blocks
            states[k].f, f_out[k] = f_out[k], states[k].f
        end
        if step in (1, 2, 5, 10, 20, 50)
            nan_any = false
            println("step $step (INTERIOR only):")
            for k in 1:n_blocks
                Nξp = states[k].Nξ_phys; Nηp = states[k].Nη_phys; ngk = states[k].n_ghost
                int_ux = view(states[k].ux, (ngk+1):(ngk+Nξp), (ngk+1):(ngk+Nηp))
                int_uy = view(states[k].uy, (ngk+1):(ngk+Nξp), (ngk+1):(ngk+Nηp))
                int_ρ = view(states[k].ρ, (ngk+1):(ngk+Nξp), (ngk+1):(ngk+Nηp))
                um = maximum(sqrt.(int_ux.^2 .+ int_uy.^2))
                ρmin, ρmax = extrema(int_ρ)
                nan_any = nan_any || any(isnan, int_ρ)
                println("  block $k :$(mbm.blocks[k].id) u_max=$(round(um, sigdigits=4)) ρ=[$(round(ρmin, sigdigits=4)), $(round(ρmax, sigdigits=4))]")
            end
            nan_any && break
        end
    end
end