using KrakeNN
using Gmsh
using KernelAbstractions
using LinearAlgebra
using Printf

const FT = Float64
const LAT = D2Q9()
const CX = FT[0, 1, 0, -1, 0, 1, -1, -1, 1]
const CY = FT[0, 0, 1, 0, -1, 1, 1, -1, -1]

const HAS_CAIRO = try
    @eval using CairoMakie
    true
catch err
    @warn "CairoMakie unavailable; PNG generation will be skipped" exception = err
    false
end

fmt(x) = @sprintf("%.6e", Float64(x))

function env_int(name, default)
    raw = strip(get(ENV, name, string(default)))
    isempty(raw) && return default
    return parse(Int, raw)
end

function env_float(name, default)
    raw = strip(get(ENV, name, string(default)))
    isempty(raw) && return FT(default)
    return parse(FT, raw)
end

function write_convergent_msh(path::AbstractString;
                              L=FT(2.0), H_in=FT(1.0), H_out=FT(0.5),
                              Nx::Int=121, Ny::Int=61)
    mkpath(dirname(path))
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("single_block_convergent")

        p1 = gmsh.model.geo.addPoint(0, 0, 0)
        p2 = gmsh.model.geo.addPoint(L, 0, 0)
        p3 = gmsh.model.geo.addPoint(L, H_out, 0)
        p4 = gmsh.model.geo.addPoint(0, H_in, 0)

        l_bot = gmsh.model.geo.addLine(p1, p2)
        l_out = gmsh.model.geo.addLine(p2, p3)
        l_top = gmsh.model.geo.addLine(p3, p4)
        l_in = gmsh.model.geo.addLine(p4, p1)

        loop = gmsh.model.geo.addCurveLoop([l_bot, l_out, l_top, l_in])
        surf = gmsh.model.geo.addPlaneSurface([loop])

        gmsh.model.geo.mesh.setTransfiniteCurve(l_bot, Nx)
        gmsh.model.geo.mesh.setTransfiniteCurve(l_top, Nx)
        gmsh.model.geo.mesh.setTransfiniteCurve(l_in, Ny)
        gmsh.model.geo.mesh.setTransfiniteCurve(l_out, Ny)
        gmsh.model.geo.mesh.setTransfiniteSurface(surf)
        gmsh.model.geo.mesh.setRecombine(2, surf)

        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, [l_in], -1, "inlet")
        gmsh.model.addPhysicalGroup(1, [l_out], -1, "outlet")
        gmsh.model.addPhysicalGroup(1, [l_bot], -1, "wall_bot")
        gmsh.model.addPhysicalGroup(1, [l_top], -1, "wall_top")
        gmsh.model.addPhysicalGroup(2, [surf], -1, "block_convergent")

        gmsh.model.mesh.generate(2)
        gmsh.write(path)
    finally
        gmsh.finalize()
    end
    return path
end

function write_convergent_krk(path::AbstractString, msh_path::AbstractString;
                              L=FT(2.0), H_in=FT(1.0), H_out=FT(0.5),
                              Nx::Int=121, Ny::Int=61,
                              u_max=FT(0.02), Re=FT(20.0), steps::Int=2000)
    mkpath(dirname(path))
    rel_mesh = relpath(msh_path, dirname(path))
    open(path, "w") do io
        println(io, "Simulation single_block_convergent D2Q9")
        println(io, "Module slbm_drag")
        println(io)
        println(io, "Define L = $(L)")
        println(io, "Define H_in = $(H_in)")
        println(io, "Define H_out = $(H_out)")
        println(io, "Define U = $(u_max)")
        println(io)
        println(io, "Domain L = L x H_in  N = $(Nx) x $(Ny)")
        println(io, "Mesh gmsh(file = \"$rel_mesh\", layout = topological, multiblock = true)")
        println(io)
        println(io, "Physics Re = $(Re) u_max = U")
        println(io)
        println(io, "Boundary west velocity(ux = U, uy = 0)")
        println(io, "Boundary east pressure(rho = 1.0)")
        println(io, "Boundary south wall")
        println(io, "Boundary north wall")
        println(io)
        println(io, "Run $(steps) steps")
    end
    return path
end

height_at_x(x; L=FT(2.0), H_in=FT(1.0), H_out=FT(0.5)) =
    H_in + (H_out - H_in) * x / L

function inlet_profile(block; u_max=FT(0.02), H_in=FT(1.0))
    T = eltype(block.mesh.X)
    prof = zeros(T, block.mesh.Nη)
    y0 = minimum(block.mesh.Y[1, :])
    @inbounds for j in 1:block.mesh.Nη
        y = clamp((block.mesh.Y[1, j] - y0) / H_in, zero(T), one(T))
        prof[j] = T(4) * T(u_max) * y * (one(T) - y)
    end
    return prof
end

function initial_u(block, i, j; L=FT(2.0), H_in=FT(1.0),
                   H_out=FT(0.5), u_max=FT(0.02), height_fn=nothing)
    x = block.mesh.X[i, j]
    h = height_fn === nothing ? height_at_x(x; L=L, H_in=H_in, H_out=H_out) :
        height_fn(x)
    y = clamp(block.mesh.Y[i, j] / h, zero(FT), one(FT))
    # Continuity-friendly initial guess: same volumetric flux as inlet.
    return u_max * (H_in / h) * 4 * y * (1 - y)
end

function seed_equilibrium!(mbm, states; L=FT(2.0), H_in=FT(1.0),
                           H_out=FT(0.5), u_max=FT(0.02), height_fn=nothing)
    for (k, block) in enumerate(mbm.blocks)
        state = states[k]
        ng = state.n_ghost
        fill!(state.f, zero(eltype(state.f)))
        fill!(state.ρ, one(eltype(state.ρ)))
        fill!(state.ux, zero(eltype(state.ux)))
        fill!(state.uy, zero(eltype(state.uy)))
        for j in 1:block.mesh.Nη, i in 1:block.mesh.Nξ
            ii = ng + i
            jj = ng + j
            ux = eltype(state.f)(initial_u(block, i, j;
                                           L=L, H_in=H_in,
                                           H_out=H_out, u_max=u_max,
                                           height_fn=height_fn))
            state.ux[ii, jj] = ux
            state.uy[ii, jj] = zero(eltype(state.f))
            for q in 1:9
                state.f[ii, jj, q] = equilibrium(LAT, one(eltype(state.f)),
                                                  ux, zero(eltype(state.f)), q)
            end
        end
    end
    return states
end

function moments_from_f(f, i, j)
    rho = zero(eltype(f))
    mx = zero(eltype(f))
    my = zero(eltype(f))
    @inbounds for q in 1:9
        fq = f[i, j, q]
        rho += fq
        mx += CX[q] * fq
        my += CY[q] * fq
    end
    return rho, mx / rho, my / rho
end

function recompute_physical_moments!(state)
    ng = state.n_ghost
    @inbounds for j in 1:state.Nη_phys, i in 1:state.Nξ_phys
        ii = ng + i
        jj = ng + j
        rho, ux, uy = moments_from_f(state.f, ii, jj)
        state.ρ[ii, jj] = rho
        state.ux[ii, jj] = ux
        state.uy[ii, jj] = uy
    end
    return state
end

function physical_density_bounds(states)
    rho_min = Inf
    rho_max = -Inf
    for state in states
        ng = state.n_ghost
        r = Array(view(state.ρ,
                       (ng + 1):(ng + state.Nξ_phys),
                       (ng + 1):(ng + state.Nη_phys)))
        any(!isfinite, r) && return (NaN, NaN)
        rho_min = min(rho_min, minimum(r))
        rho_max = max(rho_max, maximum(r))
    end
    return Float64(rho_min), Float64(rho_max)
end

function physical_speeds(mbm, states)
    xs = FT[]
    ys = FT[]
    rho = FT[]
    ux = FT[]
    uy = FT[]
    speed = FT[]
    for (k, block) in enumerate(mbm.blocks)
        state = states[k]
        ng = state.n_ghost
        for j in 1:block.mesh.Nη, i in 1:block.mesh.Nξ
            ii = ng + i
            jj = ng + j
            push!(xs, block.mesh.X[i, j])
            push!(ys, block.mesh.Y[i, j])
            push!(rho, state.ρ[ii, jj])
            push!(ux, state.ux[ii, jj])
            push!(uy, state.uy[ii, jj])
            push!(speed, hypot(state.ux[ii, jj], state.uy[ii, jj]))
        end
    end
    return (; xs, ys, rho, ux, uy, speed)
end

function edge_normal_velocity(block, state, edge::Symbol)
    ng = state.n_ghost
    vals = FT[]
    if edge === :south || edge === :north
        j = edge === :south ? 1 : block.mesh.Nη
        jj = ng + j
        for i in 1:block.mesh.Nξ
            if i == 1
                tx = block.mesh.X[i + 1, j] - block.mesh.X[i, j]
                ty = block.mesh.Y[i + 1, j] - block.mesh.Y[i, j]
            elseif i == block.mesh.Nξ
                tx = block.mesh.X[i, j] - block.mesh.X[i - 1, j]
                ty = block.mesh.Y[i, j] - block.mesh.Y[i - 1, j]
            else
                tx = block.mesh.X[i + 1, j] - block.mesh.X[i - 1, j]
                ty = block.mesh.Y[i + 1, j] - block.mesh.Y[i - 1, j]
            end
            len = hypot(tx, ty)
            nx, ny = -ty / len, tx / len
            ii = ng + i
            push!(vals, abs(state.ux[ii, jj] * nx + state.uy[ii, jj] * ny))
        end
    else
        i = edge === :west ? 1 : block.mesh.Nξ
        ii = ng + i
        for j in 1:block.mesh.Nη
            if j == 1
                tx = block.mesh.X[i, j + 1] - block.mesh.X[i, j]
                ty = block.mesh.Y[i, j + 1] - block.mesh.Y[i, j]
            elseif j == block.mesh.Nη
                tx = block.mesh.X[i, j] - block.mesh.X[i, j - 1]
                ty = block.mesh.Y[i, j] - block.mesh.Y[i, j - 1]
            else
                tx = block.mesh.X[i, j + 1] - block.mesh.X[i, j - 1]
                ty = block.mesh.Y[i, j + 1] - block.mesh.Y[i, j - 1]
            end
            len = hypot(tx, ty)
            nx, ny = -ty / len, tx / len
            jj = ng + j
            push!(vals, abs(state.ux[ii, jj] * nx + state.uy[ii, jj] * ny))
        end
    end
    return vals
end

function edge_unit_normal(block, edge::Symbol, r::Int)
    if edge === :south || edge === :north
        j = edge === :south ? 1 : block.mesh.Nη
        i = r
        if i == 1
            tx = block.mesh.X[i + 1, j] - block.mesh.X[i, j]
            ty = block.mesh.Y[i + 1, j] - block.mesh.Y[i, j]
        elseif i == block.mesh.Nξ
            tx = block.mesh.X[i, j] - block.mesh.X[i - 1, j]
            ty = block.mesh.Y[i, j] - block.mesh.Y[i - 1, j]
        else
            tx = block.mesh.X[i + 1, j] - block.mesh.X[i - 1, j]
            ty = block.mesh.Y[i + 1, j] - block.mesh.Y[i - 1, j]
        end
    else
        i = edge === :west ? 1 : block.mesh.Nξ
        j = r
        if j == 1
            tx = block.mesh.X[i, j + 1] - block.mesh.X[i, j]
            ty = block.mesh.Y[i, j + 1] - block.mesh.Y[i, j]
        elseif j == block.mesh.Nη
            tx = block.mesh.X[i, j] - block.mesh.X[i, j - 1]
            ty = block.mesh.Y[i, j] - block.mesh.Y[i, j - 1]
        else
            tx = block.mesh.X[i, j + 1] - block.mesh.X[i, j - 1]
            ty = block.mesh.Y[i, j + 1] - block.mesh.Y[i, j - 1]
        end
    end
    len = hypot(tx, ty)
    return -ty / len, tx / len
end

function project_wall_velocity_edge!(f, block, state, edge::Symbol; mode::Symbol)
    n = edge === :west || edge === :east ? block.mesh.Nη : block.mesh.Nξ
    @inbounds for r in 1:n
        i, j = edge_node_index(block, edge, r)
        ii, jj = edge_ext_index(state, edge, r)
        rho, ux, uy = moments_from_f(f, ii, jj)
        if mode === :normal || mode === :normal_corners
            nx, ny = edge_unit_normal(block, edge, r)
            un = ux * nx + uy * ny
            ux_target = ux - un * nx
            uy_target = uy - un * ny
            if mode === :normal_corners && (r == 1 || r == n)
                ux_target = zero(ux)
                uy_target = zero(uy)
            end
        elseif mode === :noslip
            ux_target = zero(ux)
            uy_target = zero(uy)
        else
            error("unknown wall projection mode $(mode)")
        end
        for q in 1:9
            f[ii, jj, q] += equilibrium(LAT, rho, ux_target, uy_target, q) -
                            equilibrium(LAT, rho, ux, uy, q)
        end
    end
    return nothing
end

function project_tagged_wall_velocity_2d!(mbm, states, fbufs; mode::Symbol=:normal)
    for (k, block) in enumerate(mbm.blocks)
        for edge in EDGE_SYMBOLS_2D
            tag = getproperty(block.boundary_tags, edge)
            (tag === :wall || tag === :wall_bot || tag === :wall_top) || continue
            project_wall_velocity_edge!(fbufs[k], block, states[k], edge; mode=mode)
        end
    end
    return nothing
end

function trapezoid_integral(y, u)
    length(y) < 2 && return zero(eltype(u))
    total = zero(eltype(u))
    for k in 1:(length(y) - 1)
        total += (y[k + 1] - y[k]) * (u[k + 1] + u[k]) / 2
    end
    return total
end

function edge_flow_rates(block, state)
    ng = state.n_ghost
    y_in = vec(block.mesh.Y[1, :])
    u_in = [state.ux[ng + 1, ng + j] for j in 1:block.mesh.Nη]
    y_out = vec(block.mesh.Y[end, :])
    u_out = [state.ux[ng + block.mesh.Nξ, ng + j] for j in 1:block.mesh.Nη]
    return (;
        Q_in=Float64(trapezoid_integral(y_in, u_in)),
        Q_out=Float64(trapezoid_integral(y_out, u_out)),
        y_in=FT.(y_in), u_in=FT.(u_in),
        y_out=FT.(y_out), u_out=FT.(u_out),
    )
end

function tagged_vertical_flow_rates(mbm, states)
    q_in = FT(0)
    q_out = FT(0)
    for (k, block) in enumerate(mbm.blocks)
        state = states[k]
        ng = state.n_ghost
        for edge in (:west, :east)
            tag = getproperty(block.boundary_tags, edge)
            (tag === :inlet || tag === :outlet) || continue
            i = edge === :west ? 1 : block.mesh.Nξ
            ii = ng + i
            y = vec(block.mesh.Y[i, :])
            u = [state.ux[ii, ng + j] for j in 1:block.mesh.Nη]
            q = trapezoid_integral(y, u)
            tag === :inlet ? (q_in += q) : (q_out += q)
        end
    end
    return (; Q_in=Float64(q_in), Q_out=Float64(q_out))
end

function edge_node_index(block, edge::Symbol, r::Int)
    edge === :west && return 1, r
    edge === :east && return block.mesh.Nξ, r
    edge === :south && return r, 1
    edge === :north && return r, block.mesh.Nη
    error("unknown edge $edge")
end

function edge_profile(block, edge::Symbol; u_max=FT(0.02))
    n = edge === :west || edge === :east ? block.mesh.Nη : block.mesh.Nξ
    T = eltype(block.mesh.X)
    prof = zeros(T, n)
    ys = T[]
    for r in 1:n
        i, j = edge_node_index(block, edge, r)
        push!(ys, block.mesh.Y[i, j])
    end
    y0 = minimum(ys)
    height = max(maximum(ys) - y0, eps(T))
    @inbounds for r in 1:n
        y = clamp((ys[r] - y0) / height, zero(T), one(T))
        prof[r] = T(4) * T(u_max) * y * (one(T) - y)
    end
    return prof
end

function block_bcspec(block; u_max=FT(0.02))
    function bc_for(edge::Symbol, tag::Symbol)
        tag === INTERFACE_TAG && return InterfaceBC()
        tag === :interface && return InterfaceBC()
        tag === :inlet && return ZouHeVelocity(edge_profile(block, edge; u_max=u_max), edge)
        tag === :outlet && return ZouHePressure(one(eltype(block.mesh.X)), edge)
        return HalfwayBB()
    end
    tags = block.boundary_tags
    return BCSpec2D(; west=bc_for(:west, tags.west),
                    east=bc_for(:east, tags.east),
                    south=bc_for(:south, tags.south),
                    north=bc_for(:north, tags.north))
end

function orient_single_convergent(mbm)
    length(mbm.blocks) == 1 ||
        error("orient_single_convergent expects exactly one block")
    block = mbm.blocks[1]
    target = (west=:inlet, east=:outlet, south=:wall_bot, north=:wall_top)
    for transpose in (false, true), flip_η in (false, true), flip_ξ in (false, true)
        candidate = reorient_block(block; transpose=transpose,
                                   flip_ξ=flip_ξ, flip_η=flip_η)
        tags = candidate.boundary_tags
        good_tags = tags.west === target.west &&
                    tags.east === target.east &&
                    tags.south === target.south &&
                    tags.north === target.north
        good_j = minimum(candidate.mesh.J) > 0
        if good_tags && good_j
            return MultiBlockMesh2D([candidate]; interfaces=Interface[])
        end
    end
    error("could not orient convergent block to W=inlet E=outlet S=wall_bot N=wall_top with J>0")
end

function interior_f_array(f, state)
    ng = state.n_ghost
    return view(f,
                (ng + 1):(ng + state.Nξ_phys),
                (ng + 1):(ng + state.Nη_phys),
                :)
end

function edge_ext_index(state, edge::Symbol, r::Int)
    ng = state.n_ghost
    if edge === :west
        return ng + 1, ng + r
    elseif edge === :east
        return ng + state.Nξ_phys, ng + r
    elseif edge === :south
        return ng + r, ng + 1
    elseif edge === :north
        return ng + r, ng + state.Nη_phys
    end
    error("unknown edge $(edge)")
end

function sync_shared_interface_nodes_2d!(mbm, states, fbufs)
    for iface in mbm.interfaces
        ia = mbm.block_by_id[iface.from[1]]
        ib = mbm.block_by_id[iface.to[1]]
        block_a = mbm.blocks[ia]
        block_b = mbm.blocks[ib]
        edge_a = iface.from[2]
        edge_b = iface.to[2]
        xa, ya = edge_coords(block_a, edge_a)
        xb, yb = edge_coords(block_b, edge_b)
        same = maximum(hypot.(xa .- xb, ya .- yb))
        flipped = maximum(hypot.(xa .- reverse(xb), ya .- reverse(yb)))
        reverse_b = flipped < same
        nrun = edge_length(block_a, edge_a)
        scratch = zeros(eltype(fbufs[ia]), 9)
        for r in 1:nrun
            rb = reverse_b ? nrun - r + 1 : r
            iae, jae = edge_ext_index(states[ia], edge_a, r)
            ibe, jbe = edge_ext_index(states[ib], edge_b, rb)
            for q in 1:9
                scratch[q] = (fbufs[ia][iae, jae, q] + fbufs[ib][ibe, jbe, q]) / 2
            end
            for q in 1:9
                fbufs[ia][iae, jae, q] = scratch[q]
                fbufs[ib][ibe, jbe, q] = scratch[q]
            end
        end
    end
    return nothing
end

function run_convergent(mbm; steps::Int=2000, ng::Int=2, ν=FT(0.04),
                        L=FT(2.0), H_in=FT(1.0), H_out=FT(0.5),
                        u_max=FT(0.02), departure::Symbol=:q1_newton,
                        height_fn=nothing, sync_interface::Bool=true,
                        shared_exchange::Bool=true,
                        wall_ghost_mode::Symbol=:copy_halfway,
                        check_every_override=nothing,
                        diagnostic_cb=nothing)
    backend = KernelAbstractions.CPU()
    states = [allocate_block_state_2d(block; n_ghost=ng, backend=backend)
              for block in mbm.blocks]
    seed_equilibrium!(mbm, states; L=L, H_in=H_in, H_out=H_out,
                      u_max=u_max, height_fn=height_fn)

    dx_ref = minimum(block.mesh.dx_ref for block in mbm.blocks)
    mesh_ext = Any[]
    geom_ext = Any[]
    buffers = Any[]
    solid_ext = Any[]
    qwall_ext = Any[]
    uwx_ext = Any[]
    uwy_ext = Any[]
    sp_ext = Any[]
    sm_ext = Any[]
    sp_int = Any[]
    sm_int = Any[]
    bcs = BCSpec2D[]
    for block in mbm.blocks
        mesh_e, geom_e = build_block_slbm_geometry_extended(
            block; n_ghost=ng, local_cfl=false, dx_ref=dx_ref,
            departure=departure)
        sp, sm = compute_local_omega_2d(mesh_e; ν=ν, scaling=:none, τ_floor=0.51)
        push!(mesh_ext, mesh_e)
        push!(geom_ext, geom_e)
        push!(solid_ext, falses(mesh_e.Nξ, mesh_e.Nη))
        push!(qwall_ext, zeros(FT, mesh_e.Nξ, mesh_e.Nη, 9))
        push!(uwx_ext, zeros(FT, mesh_e.Nξ, mesh_e.Nη, 9))
        push!(uwy_ext, zeros(FT, mesh_e.Nξ, mesh_e.Nη, 9))
        push!(sp_ext, sp)
        push!(sm_ext, sm)
        push!(sp_int, view(sp, (ng + 1):(ng + block.mesh.Nξ),
                              (ng + 1):(ng + block.mesh.Nη)))
        push!(sm_int, view(sm, (ng + 1):(ng + block.mesh.Nξ),
                              (ng + 1):(ng + block.mesh.Nη)))
        push!(bcs, block_bcspec(block; u_max=u_max))
    end
    buffers = [similar(state.f) for state in states]

    history = NamedTuple[]
    check_every = check_every_override === nothing ?
                  max(1, min(50, steps ÷ 8)) :
                  Int(check_every_override)
    t0 = time()
    for step in 1:steps
        if shared_exchange
            exchange_ghost_shared_node_2d!(mbm, states)
        else
            exchange_ghost_2d!(mbm, states)
        end
        if wall_ghost_mode === :copy_only
            fill_slbm_wall_ghost_2d!(mbm, states)
        elseif wall_ghost_mode === :copy_halfway ||
               wall_ghost_mode === :project_normal ||
               wall_ghost_mode === :project_normal_corners ||
               wall_ghost_mode === :project_noslip
            fill_slbm_wall_ghost_2d!(mbm, states)
            fill_physical_wall_ghost_2d!(mbm, states)
        elseif wall_ghost_mode === :reflect_walls
            fill_slbm_wall_ghost_2d!(mbm, states)
            fill_physical_wall_ghost_2d!(mbm, states)
            fill_tagged_reflection_ghost_2d!(mbm, states, :wall)
            fill_tagged_reflection_ghost_2d!(mbm, states, :wall_bot)
            fill_tagged_reflection_ghost_2d!(mbm, states, :wall_top)
        else
            error("unknown wall_ghost_mode=$(wall_ghost_mode)")
        end
        fill_ghost_corners_2d!(mbm, states)

        for k in eachindex(states)
            slbm_trt_libb_step_local_2d!(
                buffers[k], states[k].f, states[k].ρ, states[k].ux, states[k].uy,
                solid_ext[k], qwall_ext[k], uwx_ext[k], uwy_ext[k],
                geom_ext[k], sp_ext[k], sm_ext[k])
            apply_bc_rebuild_2d!(interior_f_array(buffers[k], states[k]), interior_f(states[k]),
                                 bcs[k], ν,
                                 states[k].Nξ_phys, states[k].Nη_phys;
                                 sp_field=sp_int[k], sm_field=sm_int[k])
        end
        if wall_ghost_mode === :project_normal
            project_tagged_wall_velocity_2d!(mbm, states, buffers; mode=:normal)
        elseif wall_ghost_mode === :project_normal_corners
            project_tagged_wall_velocity_2d!(mbm, states, buffers; mode=:normal_corners)
        elseif wall_ghost_mode === :project_noslip
            project_tagged_wall_velocity_2d!(mbm, states, buffers; mode=:noslip)
        end
        sync_interface && sync_shared_interface_nodes_2d!(mbm, states, buffers)
        for k in eachindex(states)
            states[k].f, buffers[k] = buffers[k], states[k].f
            recompute_physical_moments!(states[k])
        end
        if step % check_every == 0 || step == steps
            rho_min, rho_max = physical_density_bounds(states)
            flows = length(mbm.blocks) == 1 ?
                edge_flow_rates(mbm.blocks[1], states[1]) :
                tagged_vertical_flow_rates(mbm, states)
            q_rel = abs(flows.Q_out - flows.Q_in) / max(abs(flows.Q_in), eps())
            push!(history, (; step, rho_min, rho_max,
                            Q_in=Float64(flows.Q_in),
                            Q_out=Float64(flows.Q_out),
                            Q_rel_err=Float64(q_rel)))
            diagnostic_cb === nothing ||
                diagnostic_cb(step, mbm, states, history[end])
            if !isfinite(rho_min) || !isfinite(rho_max)
                error("non-finite physical density at step $step")
            end
        end
    end
    elapsed_s = time() - t0
    return (; states, history, dx_ref, mesh_ext, geom_ext, elapsed_s)
end

function plot_convergent(path, mbm, states, run; L=FT(2.0), H_in=FT(1.0),
                         H_out=FT(0.5), u_max=FT(0.02), ν=FT(0.04))
    HAS_CAIRO || return nothing
    mkpath(dirname(path))
    data = physical_speeds(mbm, states)
    block = mbm.blocks[1]
    state = states[1]
    flows = edge_flow_rates(block, state)
    wall_bot = edge_normal_velocity(block, state, :south)
    wall_top = edge_normal_velocity(block, state, :north)

    fig = Figure(size=(1550, 920))
    ax_mesh = Axis(fig[1, 1], aspect=DataAspect(), title="single body-fitted block",
                   xlabel="x", ylabel="y")
    ax_ux = Axis(fig[1, 2], aspect=DataAspect(), title="ux",
                 xlabel="x", ylabel="y")
    ax_speed = Axis(fig[1, 3], aspect=DataAspect(), title="|u|",
                    xlabel="x", ylabel="y")
    ax_rho = Axis(fig[2, 1], aspect=DataAspect(), title="rho",
                  xlabel="x", ylabel="y")
    ax_prof = Axis(fig[2, 2], title="inlet/outlet profiles",
                   xlabel="ux", ylabel="y")
    ax_hist = Axis(fig[2, 3], title="density bounds",
                   xlabel="step", ylabel="rho")

    mesh = block.mesh
    for j in 1:mesh.Nη
        lines!(ax_mesh, mesh.X[:, j], mesh.Y[:, j], color=(:gray35, 0.45), linewidth=0.4)
    end
    for i in 1:mesh.Nξ
        lines!(ax_mesh, mesh.X[i, :], mesh.Y[i, :], color=(:gray35, 0.45), linewidth=0.4)
    end
    lines!(ax_mesh, mesh.X[1, :], mesh.Y[1, :], color=:seagreen, linewidth=4, label="inlet")
    lines!(ax_mesh, mesh.X[end, :], mesh.Y[end, :], color=:royalblue, linewidth=4, label="outlet")
    lines!(ax_mesh, mesh.X[:, 1], mesh.Y[:, 1], color=:black, linewidth=4, label="walls")
    lines!(ax_mesh, mesh.X[:, end], mesh.Y[:, end], color=:black, linewidth=4)
    axislegend(ax_mesh; position=:rt)

    uxlim = max(maximum(abs, data.ux), FT(1e-10))
    sc_ux = scatter!(ax_ux, data.xs, data.ys; color=data.ux, markersize=4,
                     colormap=:balance, colorrange=(-uxlim, uxlim))
    Colorbar(fig[1, 4], sc_ux)
    sc_speed = scatter!(ax_speed, data.xs, data.ys; color=data.speed,
                        markersize=4, colormap=:viridis)
    Colorbar(fig[1, 5], sc_speed)
    rho_delta = max(maximum(abs.(data.rho .- 1)), FT(1e-8))
    sc_rho = scatter!(ax_rho, data.xs, data.ys; color=data.rho, markersize=4,
                      colormap=:balance, colorrange=(1 - rho_delta, 1 + rho_delta))
    Colorbar(fig[2, 4], sc_rho)

    yline = range(0, H_in; length=300)
    inlet_ref = [4 * u_max * y / H_in * (1 - y / H_in) for y in yline]
    lines!(ax_prof, inlet_ref, yline; color=:black, linewidth=2, label="target inlet")
    scatter!(ax_prof, flows.u_in, flows.y_in; color=:seagreen, markersize=5, label="inlet")
    scatter!(ax_prof, flows.u_out, flows.y_out; color=:royalblue, markersize=5, label="outlet")
    axislegend(ax_prof; position=:rt)

    steps = [h.step for h in run.history]
    rho_min = [h.rho_min for h in run.history]
    rho_max = [h.rho_max for h in run.history]
    lines!(ax_hist, steps, rho_min; color=:firebrick, linewidth=2, label="rho min")
    lines!(ax_hist, steps, rho_max; color=:navy, linewidth=2, label="rho max")
    axislegend(ax_hist; position=:rb)

    Label(fig[3, 1:3],
          "L=$(L), H_in=$(H_in), H_out=$(H_out), Umax=$(u_max), nu=$(ν), " *
          "dx_ref=$(fmt(run.dx_ref)), Q_in=$(fmt(flows.Q_in)), Q_out=$(fmt(flows.Q_out)), " *
          "max wall |u.n|=$(fmt(max(maximum(wall_bot), maximum(wall_top))))";
          tellwidth=false, fontsize=16)

    save(path, fig; px_per_unit=2)
    return path
end

function write_summary(path, mbm, states, run; steps, ν, u_max, L, H_in, H_out)
    mkpath(dirname(path))
    block = mbm.blocks[1]
    state = states[1]
    flows = edge_flow_rates(block, state)
    wall_bot = edge_normal_velocity(block, state, :south)
    wall_top = edge_normal_velocity(block, state, :north)
    rho_min, rho_max = physical_density_bounds(states)
    open(path, "w") do io
        println(io, "case,steps,nu,u_max,L,H_in,H_out,Nx,Ny,dx_ref,rho_min,rho_max,Q_in,Q_out,Q_rel_err,max_wall_un")
        println(io, join(Any[
            "single_block_convergent", steps, ν, u_max, L, H_in, H_out,
            block.mesh.Nξ, block.mesh.Nη, run.dx_ref, rho_min, rho_max,
            flows.Q_in, flows.Q_out,
            abs(flows.Q_out - flows.Q_in) / max(abs(flows.Q_in), eps()),
            max(maximum(wall_bot), maximum(wall_top)),
        ], ","))
    end
    return path
end

function print_block_summary(mbm)
    for block in mbm.blocks
        println("  $(block.id): $(block.mesh.Nξ)x$(block.mesh.Nη) ",
                "W=$(block.boundary_tags.west) E=$(block.boundary_tags.east) ",
                "S=$(block.boundary_tags.south) N=$(block.boundary_tags.north) ",
                "dx_ref=$(fmt(block.mesh.dx_ref))")
    end
end

function main()
    L = env_float("KRK_CONV_L", 2.0)
    H_in = env_float("KRK_CONV_H_IN", 1.0)
    H_out = env_float("KRK_CONV_H_OUT", 0.5)
    Nx = env_int("KRK_CONV_NX", 121)
    Ny = env_int("KRK_CONV_NY", 61)
    steps = env_int("KRK_CONV_STEPS", 2000)
    ng = env_int("KRK_CONV_NG", 2)
    u_max = env_float("KRK_CONV_UMAX", 0.02)
    ν = env_float("KRK_CONV_NU", 0.04)
    departure = Symbol(get(ENV, "KRK_CONV_DEPARTURE", "q1_newton"))

    case_dir = joinpath(@__DIR__, "convergence_cases", "single_block_convergent")
    mesh_dir = joinpath(case_dir, "meshes")
    krk_dir = joinpath(case_dir, "krk")
    plot_dir = joinpath(@__DIR__, "plots")
    table_dir = joinpath(@__DIR__, "paper_tables")
    msh_path = joinpath(mesh_dir, "single_block_convergent.msh")
    krk_path = joinpath(krk_dir, "single_block_convergent.krk")
    png_path = joinpath(plot_dir, "bodyfit_convergent_singleblock.png")
    csv_path = joinpath(table_dir, "bodyfit_convergent_singleblock_summary.csv")

    write_convergent_msh(msh_path; L=L, H_in=H_in, H_out=H_out, Nx=Nx, Ny=Ny)
    write_convergent_krk(krk_path, msh_path; L=L, H_in=H_in, H_out=H_out,
                         Nx=Nx, Ny=Ny,
                         u_max=u_max, Re=20.0, steps=steps)

    mbm_raw, groups = load_gmsh_multiblock_2d(msh_path; FT=FT, layout=:topological)
    mbm = orient_single_convergent(mbm_raw)
    issues = sanity_check_multiblock(mbm; verbose=false)
    n_errors = count(issue -> issue.severity === :error, issues)
    n_errors == 0 || error("single-block convergent sanity check has $n_errors error(s)")

    println("=== Single-block body-fitted convergent ===")
    println("physical groups: ", join(sort(collect(keys(groups.by_name))), ", "))
    print_block_summary(mbm)
    println("steps=$steps ng=$ng nu=$(ν) u_max=$(u_max) departure=$(departure)")

    run = run_convergent(mbm; steps=steps, ng=ng, ν=ν, L=L, H_in=H_in,
                         H_out=H_out, u_max=u_max, departure=departure)
    png = plot_convergent(png_path, mbm, run.states, run;
                          L=L, H_in=H_in, H_out=H_out, u_max=u_max, ν=ν)
    summary = write_summary(csv_path, mbm, run.states, run;
                            steps=steps, ν=ν, u_max=u_max,
                            L=L, H_in=H_in, H_out=H_out)

    rho_min, rho_max = physical_density_bounds(run.states)
    flows = edge_flow_rates(mbm.blocks[1], run.states[1])
    println("rho physical: [$(fmt(rho_min)), $(fmt(rho_max))]")
    println("Q_in=$(fmt(flows.Q_in)) Q_out=$(fmt(flows.Q_out)) ",
            "rel_err=$(fmt(abs(flows.Q_out - flows.Q_in) / max(abs(flows.Q_in), eps())))")
    println("wrote:")
    println("  ", relpath(msh_path, pwd()))
    println("  ", relpath(krk_path, pwd()))
    png !== nothing && println("  ", relpath(png, pwd()))
    println("  ", relpath(summary, pwd()))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
