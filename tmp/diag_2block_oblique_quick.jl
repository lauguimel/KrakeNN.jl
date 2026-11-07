# Quick 2-block oblique Poiseuille — higher viscosity to avoid τ→0.5
using Gmsh, KrakeNN, KernelAbstractions

const FT = Float64
const Lx, Ly = 90.0, 90.0
const u_max = 0.01
const ν = 0.1
const STEPS = 5000
const ng = 2

prof_full(y) = u_max * 4 * y * (Ly - y) / Ly^2

function write_geo(path)
    gmsh.initialize(); gmsh.model.add("ob2")
    p1=gmsh.model.geo.addPoint(0,0,0); p2=gmsh.model.geo.addPoint(Lx,0,0)
    p3=gmsh.model.geo.addPoint(Lx,70,0); p4=gmsh.model.geo.addPoint(0,20,0)
    p5=gmsh.model.geo.addPoint(Lx,Ly,0); p6=gmsh.model.geo.addPoint(0,Ly,0)
    l1=gmsh.model.geo.addLine(p1,p2); l2=gmsh.model.geo.addLine(p2,p3)
    l3=gmsh.model.geo.addLine(p3,p4); l4=gmsh.model.geo.addLine(p4,p1)
    l5=gmsh.model.geo.addLine(p3,p5); l6=gmsh.model.geo.addLine(p5,p6)
    l7=gmsh.model.geo.addLine(p6,p4)
    cl_A=gmsh.model.geo.addCurveLoop([l1,l2,l3,l4]); sf_A=gmsh.model.geo.addPlaneSurface([cl_A])
    cl_B=gmsh.model.geo.addCurveLoop([-l3,l5,l6,l7]); sf_B=gmsh.model.geo.addPlaneSurface([cl_B])
    for l in [l1,l3,l6]; gmsh.model.geo.mesh.setTransfiniteCurve(l, 30); end
    for l in [l4,l2,l7,l5]; gmsh.model.geo.mesh.setTransfiniteCurve(l, 15); end
    for sf in [sf_A,sf_B]; gmsh.model.geo.mesh.setTransfiniteSurface(sf); gmsh.model.geo.mesh.setRecombine(2,sf); end
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1,[l1],-1,"wall_bot"); gmsh.model.addPhysicalGroup(1,[l6],-1,"wall_top")
    gmsh.model.addPhysicalGroup(1,[l4,l7],-1,"inlet"); gmsh.model.addPhysicalGroup(1,[l2,l5],-1,"outlet")
    gmsh.model.addPhysicalGroup(1,[l3],-1,"interface")
    gmsh.model.addPhysicalGroup(2,[sf_A],-1,"block_A"); gmsh.model.addPhysicalGroup(2,[sf_B],-1,"block_B")
    gmsh.model.geo.synchronize(); gmsh.model.mesh.generate(2); gmsh.write(path); gmsh.finalize()
end

mktempdir() do dir
    geo = joinpath(dir, "ob2.msh")
    write_geo(geo)
    mbm_raw, _ = load_gmsh_multiblock_2d(geo; FT=FT, layout=:topological)
    mbm = autoreorient_blocks(mbm_raw; verbose=false)
    dx_ref = minimum(b.mesh.dx_ref for b in mbm.blocks)
    nb = length(mbm.blocks)
    println("angle=29°, dx_ref=$(round(dx_ref, digits=3)), ν=$ν, steps=$STEPS")

    states = map(mbm.blocks) do blk
        Nξ, Nη = blk.mesh.Nξ, blk.mesh.Nη
        BlockState2D{FT, Array{FT,3}, Array{FT,2}}(
            zeros(FT, Nξ+2ng, Nη+2ng, 9), ones(FT, Nξ+2ng, Nη+2ng),
            zeros(FT, Nξ+2ng, Nη+2ng), zeros(FT, Nξ+2ng, Nη+2ng), Nξ, Nη, ng)
    end

    geom_ext = Vector{Any}(undef, nb)
    sp_ext = Vector{Any}(undef, nb); sm_ext = Vector{Any}(undef, nb)
    sp_int = Vector{Any}(undef, nb); sm_int = Vector{Any}(undef, nb)

    for (k, blk) in enumerate(mbm.blocks)
        me, gh = build_block_slbm_geometry_extended(blk; n_ghost=ng, local_cfl=false, dx_ref=dx_ref)
        geom_ext[k] = gh
        mt = CurvilinearMesh(me.X, me.Y; periodic_ξ=false, periodic_η=false,
                             type=me.type, dx_ref=dx_ref, skip_validate=true, FT=FT)
        sph, smh = compute_local_omega_2d(mt; ν=ν, scaling=:quadratic, τ_floor=0.51)
        sp_ext[k] = sph; sm_ext[k] = smh
        Nξ, Nη = blk.mesh.Nξ, blk.mesh.Nη
        sp_int[k] = sph[ng+1:ng+Nξ, ng+1:ng+Nη]
        sm_int[k] = smh[ng+1:ng+Nξ, ng+1:ng+Nη]
        tau = @. 1.0 / sph
        println("  $(blk.id): τ ∈ [$(round(minimum(tau), sigdigits=4)), $(round(maximum(tau), sigdigits=4))]")
    end

    solid_ext = [zeros(Bool, size(st.f,1), size(st.f,2)) for st in states]
    qw_ext = [zeros(FT, size(st.f)...) for st in states]
    uwx_ext = [zeros(FT, size(st.f)...) for st in states]
    uwy_ext = [zeros(FT, size(st.f)...) for st in states]

    for (k, blk) in enumerate(mbm.blocks)
        m = blk.mesh
        fi = zeros(FT, m.Nξ, m.Nη, 9)
        for j in 1:m.Nη, i in 1:m.Nξ, q in 1:9
            fi[i,j,q] = KrakeNN.equilibrium(D2Q9(), 1.0, prof_full(m.Y[i,j]), 0.0, q)
        end
        copyto!(interior_f(states[k]), fi)
    end

    f_out = [zeros(FT, size(st.f)...) for st in states]

    function _bc(tag, mesh, face)
        tag === :interface && return InterfaceBC()
        if tag === :inlet
            if face === :west
                return ZouHeVelocity(FT[prof_full(mesh.Y[1,j]) for j in 1:mesh.Nη])
            elseif face === :north
                return ZouHeVelocity(FT[prof_full(mesh.Y[i,mesh.Nη]) for i in 1:mesh.Nξ])
            elseif face === :south
                return ZouHeVelocity(FT[prof_full(mesh.Y[i,1]) for i in 1:mesh.Nξ])
            else
                return ZouHeVelocity(FT[prof_full(mesh.Y[mesh.Nξ,j]) for j in 1:mesh.Nη], :west)
            end
        end
        tag === :outlet && return ZouHePressure(one(FT))
        return HalfwayBB()
    end

    bcs = map(mbm.blocks) do blk
        t = blk.boundary_tags; m = blk.mesh
        BCSpec2D(; west=_bc(t.west, m, :west), east=_bc(t.east, m, :east),
                   south=_bc(t.south, m, :south), north=_bc(t.north, m, :north))
    end

    t0 = time()
    nan_step = 0
    for nn in 1:STEPS
        exchange_ghost_shared_node_2d!(mbm, states)
        fill_ghost_corners_2d!(mbm, states)
        fill_slbm_wall_ghost_2d!(mbm, states)
        fill_physical_wall_ghost_2d!(mbm, states)
        for k in 1:nb
            slbm_trt_libb_step_local_biquad_2d!(f_out[k], states[k].f,
                states[k].ρ, states[k].ux, states[k].uy,
                solid_ext[k], qw_ext[k], uwx_ext[k], uwy_ext[k],
                geom_ext[k], sp_ext[k], sm_ext[k])
        end
        for (k, blk) in enumerate(mbm.blocks)
            Np = states[k].Nξ_phys; Nq = states[k].Nη_phys
            apply_bc_rebuild_2d!(
                view(f_out[k], ng+1:ng+Np, ng+1:ng+Nq, :),
                view(states[k].f, ng+1:ng+Np, ng+1:ng+Nq, :),
                bcs[k], ν, Np, Nq; sp_field=sp_int[k], sm_field=sm_int[k])
        end
        for k in 1:nb
            states[k].f, f_out[k] = f_out[k], states[k].f
        end
        if any(k -> any(isnan, states[k].f), 1:nb)
            nan_step = nn
            println("NaN at step $(nn)")
            break
        end
        if nn % 1000 == 0
            println("  step $nn ($(round(time()-t0, digits=1))s)")
        end
    end
    nan_step == 0 && println("Done $STEPS steps ($(round(time()-t0, digits=1))s)")

    # Profile at x≈45
    all_y = FT[]; all_ux = FT[]
    for (k, blk) in enumerate(mbm.blocks)
        m = blk.mesh
        uxa = Array(states[k].ux)[ng+1:ng+m.Nξ, ng+1:ng+m.Nη]
        for j in 1:m.Nη, i in 1:m.Nξ
            abs(m.X[i,j] - 45.0) < 3.0 || continue
            push!(all_y, m.Y[i,j])
            push!(all_ux, uxa[i,j])
        end
    end
    sp = sortperm(all_y); all_y = all_y[sp]; all_ux = all_ux[sp]
    err = maximum(abs, all_ux .- prof_full.(all_y))
    println("max|ux - analytical| = $(round(err, sigdigits=3)) ($(round(100*err/u_max, digits=2))%)")
end
