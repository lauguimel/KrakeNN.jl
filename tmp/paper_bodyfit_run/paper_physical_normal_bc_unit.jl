using KrakeNN
using Printf

include(joinpath(@__DIR__, "paper_gmsh_bc_semantics.jl"))

const PB_FT = Float64
const PB_LAT = D2Q9()
const PB_CX = PB_FT[0, 1, 0, -1, 0, 1, -1, -1, 1]
const PB_CY = PB_FT[0, 0, 1, 0, -1, 1, 1, -1, -1]
const PB_W = PB_FT[4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9,
                   1 / 36, 1 / 36, 1 / 36, 1 / 36]
const PB_OPP = (1, 4, 5, 2, 3, 8, 9, 6, 7)

pb_fmt(x) = @sprintf("%.3e", Float64(x))

function pb_axis_velocity(normal::Symbol, value)
    if normal === :west || normal === :east
        return PB_FT(value), zero(PB_FT)
    elseif normal === :south || normal === :north
        return zero(PB_FT), PB_FT(value)
    end
    error("unknown physical normal $(normal)")
end

function pb_unknown_pops(normal::Symbol)
    normal === :west  && return (2, 6, 9)  # c . n_out < 0 for n_out=(-1,0)
    normal === :east  && return (4, 7, 8)
    normal === :south && return (3, 6, 7)
    normal === :north && return (5, 8, 9)
    error("unknown physical normal $(normal)")
end

function pb_fill_equilibrium!(f; rho=1.0, ux=0.0, uy=0.0)
    nx, ny = size(f, 1), size(f, 2)
    for j in 1:ny, i in 1:nx, q in 1:9
        f[i, j, q] = equilibrium(PB_LAT, PB_FT(rho), PB_FT(ux), PB_FT(uy), q)
    end
    return f
end

function pb_moments_node(f, i, j)
    rho = sum(f[i, j, q] for q in 1:9)
    ux = sum(PB_CX[q] * f[i, j, q] for q in 1:9) / rho
    uy = sum(PB_CY[q] * f[i, j, q] for q in 1:9) / rho
    return rho, ux, uy
end

function pb_logical_edge_indices(edge::Symbol, nx::Int, ny::Int)
    edge === :west  && return ((1, j) for j in 1:ny)
    edge === :east  && return ((nx, j) for j in 1:ny)
    edge === :south && return ((i, 1) for i in 1:nx)
    edge === :north && return ((i, ny) for i in 1:nx)
    error("unknown logical edge $(edge)")
end

function pb_corrupt_unknowns!(f, logical_edge::Symbol, physical_normal::Symbol)
    nx, ny = size(f, 1), size(f, 2)
    for (i, j) in pb_logical_edge_indices(logical_edge, nx, ny)
        for q in pb_unknown_pops(physical_normal)
            f[i, j, q] = PB_FT(NaN)
        end
    end
    return f
end

function pb_zouhe_velocity_node!(fq, normal::Symbol, ux::PB_FT, uy::PB_FT)
    if normal === :west
        rho = (fq[1] + fq[3] + fq[5] + 2 * (fq[4] + fq[7] + fq[8])) / (1 - ux)
        fq[2] = fq[4] + PB_FT(2 / 3) * rho * ux
        fq[6] = fq[8] - PB_FT(0.5) * (fq[3] - fq[5]) + PB_FT(1 / 6) * rho * ux
        fq[9] = fq[7] + PB_FT(0.5) * (fq[3] - fq[5]) + PB_FT(1 / 6) * rho * ux
    elseif normal === :east
        rho = (fq[1] + fq[3] + fq[5] + 2 * (fq[2] + fq[6] + fq[9])) / (1 + ux)
        fq[4] = fq[2] - PB_FT(2 / 3) * rho * ux
        fq[7] = fq[9] - PB_FT(0.5) * (fq[3] - fq[5]) - PB_FT(1 / 6) * rho * ux
        fq[8] = fq[6] + PB_FT(0.5) * (fq[3] - fq[5]) - PB_FT(1 / 6) * rho * ux
    elseif normal === :south
        rho = (fq[1] + fq[2] + fq[4] + 2 * (fq[5] + fq[8] + fq[9])) / (1 - uy)
        fq[3] = fq[5] + PB_FT(2 / 3) * rho * uy
        fq[6] = fq[8] + PB_FT(0.5) * (fq[4] - fq[2]) + PB_FT(1 / 6) * rho * uy
        fq[7] = fq[9] + PB_FT(0.5) * (fq[2] - fq[4]) + PB_FT(1 / 6) * rho * uy
    elseif normal === :north
        rho = (fq[1] + fq[2] + fq[4] + 2 * (fq[3] + fq[6] + fq[7])) / (1 + uy)
        fq[5] = fq[3] - PB_FT(2 / 3) * rho * uy
        fq[8] = fq[6] + PB_FT(0.5) * (fq[2] - fq[4]) - PB_FT(1 / 6) * rho * uy
        fq[9] = fq[7] + PB_FT(0.5) * (fq[4] - fq[2]) - PB_FT(1 / 6) * rho * uy
    else
        error("unknown physical normal $(normal)")
    end
    return fq
end

function pb_zouhe_pressure_node!(fq, normal::Symbol, rho::PB_FT)
    if normal === :west
        ux = 1 - (fq[1] + fq[3] + fq[5] + 2 * (fq[4] + fq[7] + fq[8])) / rho
        return pb_zouhe_velocity_node!(fq, normal, ux, zero(PB_FT))
    elseif normal === :east
        ux = -1 + (fq[1] + fq[3] + fq[5] + 2 * (fq[2] + fq[6] + fq[9])) / rho
        return pb_zouhe_velocity_node!(fq, normal, ux, zero(PB_FT))
    elseif normal === :south
        uy = 1 - (fq[1] + fq[2] + fq[4] + 2 * (fq[5] + fq[8] + fq[9])) / rho
        return pb_zouhe_velocity_node!(fq, normal, zero(PB_FT), uy)
    elseif normal === :north
        uy = -1 + (fq[1] + fq[2] + fq[4] + 2 * (fq[3] + fq[6] + fq[7])) / rho
        return pb_zouhe_velocity_node!(fq, normal, zero(PB_FT), uy)
    end
    error("unknown physical normal $(normal)")
end

function pb_ladd_wall_node!(fq, normal::Symbol, rho::PB_FT, uxw::PB_FT, uyw::PB_FT)
    for q in pb_unknown_pops(normal)
        qo = PB_OPP[q]
        fq[q] = fq[qo] + 6 * PB_W[q] * rho * (PB_CX[q] * uxw + PB_CY[q] * uyw)
    end
    return fq
end

function pb_apply_velocity!(f, logical_edge::Symbol, physical_normal::Symbol;
                            value=0.037)
    nx, ny = size(f, 1), size(f, 2)
    ux, uy = pb_axis_velocity(physical_normal, value)
    scratch = zeros(PB_FT, 9)
    for (i, j) in pb_logical_edge_indices(logical_edge, nx, ny)
        for q in 1:9
            scratch[q] = f[i, j, q]
        end
        pb_zouhe_velocity_node!(scratch, physical_normal, ux, uy)
        for q in 1:9
            f[i, j, q] = scratch[q]
        end
    end
    return ux, uy
end

function pb_apply_pressure!(f, logical_edge::Symbol, physical_normal::Symbol;
                            rho=1.021)
    nx, ny = size(f, 1), size(f, 2)
    scratch = zeros(PB_FT, 9)
    for (i, j) in pb_logical_edge_indices(logical_edge, nx, ny)
        for q in 1:9
            scratch[q] = f[i, j, q]
        end
        pb_zouhe_pressure_node!(scratch, physical_normal, PB_FT(rho))
        for q in 1:9
            f[i, j, q] = scratch[q]
        end
    end
    return PB_FT(rho)
end

function pb_apply_ladd_wall!(f, logical_edge::Symbol, physical_normal::Symbol;
                             rho=1.0, uxw=0.0, uyw=0.0)
    nx, ny = size(f, 1), size(f, 2)
    scratch = zeros(PB_FT, 9)
    for (i, j) in pb_logical_edge_indices(logical_edge, nx, ny)
        for q in 1:9
            scratch[q] = f[i, j, q]
        end
        pb_ladd_wall_node!(scratch, physical_normal, PB_FT(rho), PB_FT(uxw), PB_FT(uyw))
        for q in 1:9
            f[i, j, q] = scratch[q]
        end
    end
    return nothing
end

function pb_measure_edge(f, logical_edge::Symbol; rho=1.0, ux=0.0, uy=0.0)
    nx, ny = size(f, 1), size(f, 2)
    max_pop_err = zero(PB_FT)
    max_moment_err = zero(PB_FT)
    for (i, j) in pb_logical_edge_indices(logical_edge, nx, ny)
        r, u, v = pb_moments_node(f, i, j)
        max_moment_err = max(max_moment_err, abs(r - rho), abs(u - ux), abs(v - uy))
        for q in 1:9
            expected = equilibrium(PB_LAT, PB_FT(rho), PB_FT(ux), PB_FT(uy), q)
            max_pop_err = max(max_pop_err, abs(f[i, j, q] - expected))
        end
    end
    return (; max_pop_err, max_moment_err)
end

function pb_physical_normal_from_tag(tag::Symbol)
    tag === :inlet && return :west
    tag === :outlet && return :east
    tag === :wall_bot && return :south
    tag === :wall_top && return :north
    error("no physical normal for tag $(tag)")
end

function pb_case(logical_edge::Symbol, physical_normal::Symbol, kind::Symbol;
                 nx=9, ny=8, rho=1.021, value=0.037)
    f = zeros(PB_FT, nx, ny, 9)
    if kind === :velocity
        ux, uy = pb_axis_velocity(physical_normal, value)
        pb_fill_equilibrium!(f; rho=1.0, ux=ux, uy=uy)
        pb_corrupt_unknowns!(f, logical_edge, physical_normal)
        pb_apply_velocity!(f, logical_edge, physical_normal; value=value)
        return merge((kind=kind, logical_edge=logical_edge, physical_normal=physical_normal),
                     pb_measure_edge(f, logical_edge; rho=1.0, ux=ux, uy=uy))
    elseif kind === :pressure
        ux, uy = pb_axis_velocity(physical_normal, value)
        pb_fill_equilibrium!(f; rho=rho, ux=ux, uy=uy)
        pb_corrupt_unknowns!(f, logical_edge, physical_normal)
        pb_apply_pressure!(f, logical_edge, physical_normal; rho=rho)
        return merge((kind=kind, logical_edge=logical_edge, physical_normal=physical_normal),
                     pb_measure_edge(f, logical_edge; rho=rho, ux=ux, uy=uy))
    elseif kind === :wall
        uxw, uyw = physical_normal in (:south, :north) ? (PB_FT(value), zero(PB_FT)) :
                   (zero(PB_FT), PB_FT(value))
        pb_fill_equilibrium!(f; rho=1.0, ux=uxw, uy=uyw)
        pb_corrupt_unknowns!(f, logical_edge, physical_normal)
        pb_apply_ladd_wall!(f, logical_edge, physical_normal; rho=1.0, uxw=uxw, uyw=uyw)
        return merge((kind=kind, logical_edge=logical_edge, physical_normal=physical_normal),
                     pb_measure_edge(f, logical_edge; rho=1.0, ux=uxw, uy=uyw))
    end
    error("unknown case kind $(kind)")
end

function pb_run_all()
    logical_edges = (:west, :east, :south, :north)
    physical_normals = (:west, :east, :south, :north)
    velocity = [pb_case(le, pn, :velocity) for le in logical_edges for pn in physical_normals]
    pressure = [pb_case(le, pn, :pressure) for le in logical_edges for pn in physical_normals]
    wall = [pb_case(le, pn, :wall) for le in logical_edges for pn in physical_normals]

    demo = bm_demo_mbm()
    gmsh = NamedTuple[]
    for r in bm_audit_records(demo.mbm)
        pn = pb_physical_normal_from_tag(r.tag)
        kind = r.tag === :inlet ? :velocity : r.tag === :outlet ? :pressure : :wall
        push!(gmsh, pb_case(r.edge, pn, kind))
    end
    return (; velocity, pressure, wall, gmsh)
end

function pb_group_error(group)
    return (maximum(r.max_pop_err for r in group),
            maximum(r.max_moment_err for r in group))
end

function main()
    results = pb_run_all()
    println("=== Physical-normal BC unit diagnostic ===")
    println("The physical normal is intentionally independent of the logical edge.")
    println(rpad("group", 12), rpad("cases", 8), rpad("max_pop_err", 14), "max_moment_err")
    for (name, group) in ((:velocity, results.velocity),
                          (:pressure, results.pressure),
                          (:wall, results.wall),
                          (:gmsh_tags, results.gmsh))
        pop_err, moment_err = pb_group_error(group)
        println(rpad(String(name), 12), rpad(string(length(group)), 8),
                rpad(pb_fmt(pop_err), 14), pb_fmt(moment_err))
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
