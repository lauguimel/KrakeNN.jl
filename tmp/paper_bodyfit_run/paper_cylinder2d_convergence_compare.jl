using KrakeNN
using KernelAbstractions
using Printf
using Statistics

const ROOT = @__DIR__
const CASE_ROOT = joinpath(ROOT, "convergence_cases")
const TABLEDIR = joinpath(ROOT, "paper_tables")
const PLOTDIR = joinpath(ROOT, "plots")
const REQUESTED_BACKEND = lowercase(strip(get(ENV, "KRK_CYL_CONV_BACKEND", "cpu")))
mkpath(CASE_ROOT)
mkpath(TABLEDIR)
mkpath(PLOTDIR)

if REQUESTED_BACKEND == "cuda"
    using CUDA
elseif REQUESTED_BACKEND == "metal"
    using Metal
end

include(joinpath(ROOT, "..", "gen_ogrid_rect_8block.jl"))

const HAS_CAIRO = if get(ENV, "KRK_CYL_CONV_SKIP_PLOTS", "0") == "1"
    false
else
    try
        @eval using CairoMakie
        true
    catch err
        @warn "CairoMakie unavailable; PNG generation will be skipped" exception = err
        false
    end
end

fmt2(x) = @sprintf("%.2f", Float64(x))
fmt3(x) = @sprintf("%.3e", Float64(x))
fmt4(x) = @sprintf("%.4f", Float64(x))
fmt6(x) = @sprintf("%.6f", Float64(x))

function env_int(name, default)
    return parse(Int, strip(get(ENV, name, string(default))))
end

function env_float(name, default)
    return parse(Float64, strip(get(ENV, name, string(default))))
end

function parse_float_list(raw)
    vals = Float64[]
    for part in split(raw, ',')
        s = strip(part)
        isempty(s) && continue
        push!(vals, parse(Float64, s))
    end
    isempty(vals) && error("empty float list: $raw")
    return vals
end

function parse_ogrid_specs(raw)
    specs = NamedTuple[]
    for part in split(raw, ',')
        s = lowercase(strip(part))
        isempty(s) && continue
        bits = split(s, 'x')
        length(bits) == 2 || error("invalid O-grid spec '$s'; expected N_arcxN_radial")
        push!(specs, (; n_arc=parse(Int, bits[1]), n_radial=parse(Int, bits[2])))
    end
    isempty(specs) && error("empty O-grid spec list: $raw")
    return specs
end

function select_float_type(backend_label)
    raw = strip(get(ENV, "KRK_CYL_CONV_FT", ""))
    if raw == "Float64"
        return Float64
    elseif raw == "Float32"
        return Float32
    elseif !isempty(raw)
        error("unknown KRK_CYL_CONV_FT=$raw; expected Float64 or Float32")
    end
    return backend_label == "metal" ? Float32 : Float64
end

function select_backend()
    if REQUESTED_BACKEND == "cuda"
        CUDA.functional() || error("KRK_CYL_CONV_BACKEND=cuda requested, but CUDA.functional() is false")
        return (; backend=CUDABackend(), T=select_float_type("cuda"), label="cuda")
    elseif REQUESTED_BACKEND == "metal"
        Metal.functional() || error("KRK_CYL_CONV_BACKEND=metal requested, but Metal.functional() is false")
        return (; backend=MetalBackend(), T=select_float_type("metal"), label="metal")
    elseif REQUESTED_BACKEND == "cpu"
        return (; backend=KernelAbstractions.CPU(), T=select_float_type("cpu"), label="cpu")
    end
    error("unknown KRK_CYL_CONV_BACKEND=$REQUESTED_BACKEND; expected cpu, cuda, or metal")
end

function allocate_copy(backend, T, host)
    if occursin("MetalBackend", string(typeof(backend)))
        return Metal.MtlArray(T.(host))
    end
    dev = KernelAbstractions.allocate(backend, T, size(host)...)
    copyto!(dev, T.(host))
    return dev
end

function allocate_copy_bool(backend, host)
    if occursin("MetalBackend", string(typeof(backend)))
        return Metal.MtlArray(host)
    end
    dev = KernelAbstractions.allocate(backend, Bool, size(host)...)
    copyto!(dev, host)
    return dev
end

function allocate_filled(backend, T, value, dims::Int...)
    if occursin("MetalBackend", string(typeof(backend)))
        return Metal.MtlArray(fill(T(value), dims...))
    end
    dev = KernelAbstractions.allocate(backend, T, dims...)
    fill!(dev, T(value))
    return dev
end

function output_suffix()
    raw = strip(get(ENV, "KRK_CYL_CONV_TAG", ""))
    isempty(raw) && return ""
    clean = replace(raw, r"[^A-Za-z0-9_\-]" => "_")
    return "_" * clean
end

function cfg_from_env()
    steps = env_int("KRK_CYL_CONV_STEPS", 10_000)
    avg_window = min(env_int("KRK_CYL_CONV_AVG_WINDOW", 3_000), steps)
    sample_every = env_int("KRK_CYL_CONV_SAMPLE_EVERY", 50)
    return (;
        Lx=env_float("KRK_CYL_CONV_LX", 2.2),
        Ly=env_float("KRK_CYL_CONV_LY", 0.41),
        cx=env_float("KRK_CYL_CONV_CX", 0.2),
        cy=env_float("KRK_CYL_CONV_CY", 0.2),
        R=env_float("KRK_CYL_CONV_R", 0.05),
        Re=env_float("KRK_CYL_CONV_RE", 20.0),
        u_max=env_float("KRK_CYL_CONV_UMAX", 0.04),
        steps=steps,
        avg_window=avg_window,
        sample_every=sample_every,
        check_every=env_int("KRK_CYL_CONV_CHECK_EVERY", 500),
        cart_deffs=parse_float_list(get(ENV, "KRK_CYL_CONV_CART_DEFFS", "20,30,40,50")),
        ogrid_specs=parse_ogrid_specs(get(ENV, "KRK_CYL_CONV_OGRID_SPECS", "20x16,28x20,36x24")),
        radial_progression=env_float("KRK_CYL_CONV_RADIAL_PROGRESSION", 0.92),
        bodyfit_reflect_ghost=env_float("KRK_CYL_CONV_BODYFIT_REFLECT_GHOST", 0.0),
        bodyfit_cylinder_ghost_alpha=env_float("KRK_CYL_CONV_CYLINDER_GHOST_ALPHA", 0.10),
        ref_cd=env_float("KRK_CYL_CONV_REF_CD", 5.57953523384),
        ref_cl=env_float("KRK_CYL_CONV_REF_CL", 0.010618948146),
    )
end

function check_rho!(rho, step, label)
    rho_h = Array(rho)
    if any(isnan, rho_h)
        error("NaN density in $label at step $step")
    end
    return Float64(minimum(rho_h)), Float64(maximum(rho_h))
end

function sample_stats(xs)
    isempty(xs) && return (; mean=NaN, std=NaN)
    return (; mean=Float64(mean(xs)), std=length(xs) > 1 ? Float64(std(xs)) : 0.0)
end

function row_with_errors(row, cfg)
    cl_err = abs(row.Cl - cfg.ref_cl)
    cl_flip_err = abs(-row.Cl - cfg.ref_cl)
    return merge(row, (;
        Cd_ref=cfg.ref_cd,
        Cl_ref=cfg.ref_cl,
        Cd_abs_error=abs(row.Cd - cfg.ref_cd),
        Cl_abs_error=cl_err,
        Cl_abs_error_flipped=cl_flip_err,
        Cl_sign_note=cl_flip_err < cl_err ? "flipped-sign-closer" : "native-sign",
    ))
end

function classify_error(err)
    msg = sprint(showerror, err)
    if occursin("non-finite density", msg) || occursin("NaN density", msg)
        return "failed_nonfinite_density"
    elseif occursin("Gmsh", msg)
        return "failed_gmsh"
    end
    return "failed_runtime"
end

function failed_row(; method, resolution, backend_info, cfg, blocks=0, Nx=0, Ny=0,
                    n_arc=0, n_radial=0, nodes=0, dx_ref=NaN, D_eff=NaN,
                    mesh_file="", status)
    row = (;
        method=method, resolution=resolution, backend=backend_info.label,
        precision=string(backend_info.T), steps=cfg.steps,
        avg_window=cfg.avg_window, sample_every=cfg.sample_every,
        blocks=blocks, Nx=Nx, Ny=Ny, n_arc=n_arc, n_radial=n_radial,
        nodes=nodes, solid_cells=0, dx_ref=dx_ref, D_eff=D_eff, Re=cfg.Re,
        u_max=cfg.u_max, u_ref=(2.0 / 3.0) * cfg.u_max, nu=NaN,
        Cd=NaN, Cd_std=NaN, Cl=NaN, Cl_std=NaN,
        rho_min=NaN, rho_max=NaN, elapsed_s=NaN, MLUPs=NaN,
        mesh_file=mesh_file, status=status)
    return row_with_errors(row, cfg)
end

function run_cartesian_libb_case(cfg, backend_info, deff_target)
    backend, T = backend_info.backend, backend_info.T
    dx = 2.0 * cfg.R / Float64(deff_target)
    Nx = round(Int, cfg.Lx / dx) + 1
    Ny = round(Int, cfg.Ly / dx) + 1
    radius = cfg.R / dx
    cx = cfg.cx / dx
    cy = cfg.cy / dx
    D_eff = 2.0 * radius
    u_ref = (2.0 / 3.0) * cfg.u_max
    nu = u_ref * D_eff / cfg.Re

    q_wall_h, is_solid_h = precompute_q_wall_cylinder(Nx, Ny, cx, cy, radius; FT=T)
    uw_h = zeros(T, Nx, Ny, 9)
    profile_h = [T(4) * T(cfg.u_max) * T(j - 1) * T(Ny - j) / T(Ny - 1)^2
                 for j in 1:Ny]
    f_h = zeros(T, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        u = is_solid_h[i, j] ? zero(T) : profile_h[j]
        f_h[i, j, q] = T(KrakeNN.equilibrium(D2Q9(), one(T), u, zero(T), q))
    end

    q_wall = allocate_copy(backend, T, q_wall_h)
    is_solid = allocate_copy_bool(backend, is_solid_h)
    uw_x = allocate_copy(backend, T, uw_h)
    uw_y = allocate_copy(backend, T, uw_h)
    f_in = allocate_copy(backend, T, f_h)
    f_out = allocate_filled(backend, T, zero(T), Nx, Ny, 9)
    rho = allocate_filled(backend, T, one(T), Nx, Ny)
    ux = allocate_filled(backend, T, zero(T), Nx, Ny)
    uy = allocate_filled(backend, T, zero(T), Nx, Ny)
    profile = allocate_copy(backend, T, profile_h)
    bc = BCSpec2D(; west=ZouHeVelocity(profile), east=ZouHePressure(one(T)),
                    south=HalfwayBB(), north=HalfwayBB())

    cd_samples = Float64[]
    cl_samples = Float64[]
    history = NamedTuple[]
    rho_min = 1.0
    rho_max = 1.0
    t0 = time()
    for step in 1:cfg.steps
        fused_trt_libb_v2_step!(f_out, f_in, rho, ux, uy, is_solid,
                                 q_wall, uw_x, uw_y, Nx, Ny, T(nu))
        apply_bc_rebuild_2d!(f_out, f_in, bc, nu, Nx, Ny)
        if step > cfg.steps - cfg.avg_window &&
                (step % cfg.sample_every == 0 || step == cfg.steps)
            drag = compute_drag_libb_mei_2d(f_out, q_wall, uw_x, uw_y, Nx, Ny)
            Cd = 2.0 * Float64(drag.Fx) / (u_ref^2 * D_eff)
            Cl = 2.0 * Float64(drag.Fy) / (u_ref^2 * D_eff)
            push!(cd_samples, Cd)
            push!(cl_samples, Cl)
            push!(history, (; method="cartesian_libb", resolution=string(round(Int, D_eff)),
                            step=step, Cd=Cd, Cl=Cl, Fx=Float64(drag.Fx),
                            Fy=Float64(drag.Fy)))
        end
        f_in, f_out = f_out, f_in
        if step % cfg.check_every == 0 || step == cfg.steps
            rho_min, rho_max = check_rho!(rho, step, "cartesian_libb_D$(D_eff)")
        end
    end
    KernelAbstractions.synchronize(backend)
    elapsed = time() - t0
    rho_min, rho_max = check_rho!(rho, cfg.steps, "cartesian_libb_D$(D_eff)")
    cds = sample_stats(cd_samples)
    cls = sample_stats(cl_samples)
    nodes = Nx * Ny
    row = (;
        method="cartesian_libb", resolution="D$(round(Int, D_eff))",
        backend=backend_info.label, precision=string(T), steps=cfg.steps,
        avg_window=cfg.avg_window, sample_every=cfg.sample_every,
        blocks=1, Nx=Nx, Ny=Ny, n_arc=0, n_radial=0, nodes=nodes,
        solid_cells=count(identity, is_solid_h), dx_ref=dx, D_eff=D_eff,
        Re=cfg.Re, u_max=cfg.u_max, u_ref=u_ref, nu=nu,
        Cd=cds.mean, Cd_std=cds.std, Cl=cls.mean, Cl_std=cls.std,
        rho_min=rho_min, rho_max=rho_max, elapsed_s=elapsed,
        MLUPs=nodes * cfg.steps / max(elapsed, eps()) / 1e6,
        mesh_file="", status="ok")
    return (; row=row_with_errors(row, cfg), history)
end

function write_ogrid_msh(case_dir, cfg, spec)
    mesh_dir = joinpath(case_dir, "meshes")
    mkpath(mesh_dir)
    stem = "cylinder_ogrid_a$(spec.n_arc)_r$(spec.n_radial)"
    geo_path = joinpath(mesh_dir, stem * ".geo")
    msh_path = joinpath(mesh_dir, stem * ".msh")
    write_ogrid_rect_8block_geo(geo_path; Lx=cfg.Lx, Ly=cfg.Ly,
        cx_p=cfg.cx, cy_p=cfg.cy, R_in=cfg.R, N_arc=spec.n_arc,
        N_radial=spec.n_radial, radial_progression=cfg.radial_progression)

    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(geo_path)
        gmsh.model.mesh.generate(2)
        gmsh.write(msh_path)
    finally
        gmsh.finalize()
    end
    return (; geo_path, msh_path)
end

function bodyfit_case_dir(tag, spec)
    return joinpath(CASE_ROOT, isempty(tag) ? "latest" : tag,
                    "bodyfit_a$(spec.n_arc)_r$(spec.n_radial)")
end

function bodyfit_mesh_path(tag, spec)
    stem = "cylinder_ogrid_a$(spec.n_arc)_r$(spec.n_radial)"
    return joinpath(bodyfit_case_dir(tag, spec), "meshes", stem * ".msh")
end

function estimate_bodyfit_mesh_row(cfg, spec, tag)
    msh = bodyfit_mesh_path(tag, spec)
    if !isfile(msh)
        return (; blocks=8, nodes=8 * spec.n_arc * spec.n_radial,
                dx_ref=NaN, D_eff=NaN, mesh_file=msh)
    end
    mbm, _ = load_gmsh_multiblock_2d(msh; FT=Float64, layout=:topological)
    dx_ref = minimum(block.mesh.dx_ref for block in mbm.blocks)
    nodes = sum(block.mesh.Nξ * block.mesh.Nη for block in mbm.blocks)
    return (; blocks=length(mbm.blocks), nodes=nodes, dx_ref=Float64(dx_ref),
            D_eff=2.0 * cfg.R / Float64(dx_ref), mesh_file=msh)
end

function write_bodyfit_krk(case_dir, cfg, spec, mesh_paths)
    krk_dir = joinpath(case_dir, "krk")
    mkpath(krk_dir)
    krk_path = joinpath(krk_dir, "cylinder_ogrid_a$(spec.n_arc)_r$(spec.n_radial).krk")
    rel_mesh = relpath(mesh_paths.msh_path, krk_dir)
    open(krk_path, "w") do io
        println(io, "Simulation cylinder_ogrid_a$(spec.n_arc)_r$(spec.n_radial) D2Q9")
        println(io, "Module slbm_drag")
        println(io)
        println(io, "Define Lx = $(cfg.Lx)")
        println(io, "Define Ly = $(cfg.Ly)")
        println(io, "Define U = $(cfg.u_max)")
        println(io, "Define cx = $(cfg.cx)")
        println(io, "Define cy = $(cfg.cy)")
        println(io, "Define R = $(cfg.R)")
        println(io)
        println(io, "Domain L = Lx x Ly  N = $(8 * spec.n_arc) x $(spec.n_radial)")
        println(io, "Mesh gmsh(file = \"$rel_mesh\", layout = topological, multiblock = true)")
        println(io)
        println(io, "Physics Re = $(cfg.Re) u_max = U cx = cx cy = cy R = R avg_window = $(cfg.avg_window) sample_every = $(cfg.sample_every) check_every = $(cfg.check_every) bodyfit_reflect_ghost = $(cfg.bodyfit_reflect_ghost) bodyfit_cylinder_ghost_alpha = $(cfg.bodyfit_cylinder_ghost_alpha)")
        println(io)
        println(io, "Boundary west velocity(ux = U, uy = 0)")
        println(io, "Boundary east pressure(rho = 1.0)")
        println(io, "Boundary south wall")
        println(io, "Boundary north wall")
        println(io)
        println(io, "Diagnostics every $(cfg.sample_every) [step, drag, lift]")
        println(io, "Run $(cfg.steps) steps")
    end
    return krk_path
end

function run_bodyfit_ogrid_case(cfg, backend_info, spec, tag)
    case_dir = bodyfit_case_dir(tag, spec)
    mkpath(case_dir)
    mesh_paths = write_ogrid_msh(case_dir, cfg, spec)
    krk_path = write_bodyfit_krk(case_dir, cfg, spec, mesh_paths)

    result = run_simulation(krk_path; backend=backend_info.backend,
                            T=backend_info.T, max_steps=cfg.steps)
    cds = sample_stats(result.Cd_samples)
    cls = sample_stats(result.Cl_samples)
    row = (;
        method="bodyfit_gmsh_slbm",
        resolution="a$(spec.n_arc)_r$(spec.n_radial)",
        backend=backend_info.label, precision=string(backend_info.T),
        steps=result.steps, avg_window=result.avg_window,
        sample_every=result.sample_every, blocks=result.blocks,
        Nx=0, Ny=0, n_arc=spec.n_arc, n_radial=spec.n_radial,
        nodes=result.nodes, solid_cells=result.solid_cells,
        dx_ref=result.dx_ref, D_eff=result.D_eff, Re=result.Re,
        u_max=result.u_max, u_ref=result.u_ref, nu=result.nu,
        Cd=cds.mean, Cd_std=cds.std, Cl=cls.mean, Cl_std=cls.std,
        rho_min=result.rho_min, rho_max=result.rho_max,
        elapsed_s=result.elapsed_s, MLUPs=result.MLUPs,
        mesh_file=result.mesh_file, status="ok")
    history = [merge(h, (; method="bodyfit_gmsh_slbm",
                         resolution="a$(spec.n_arc)_r$(spec.n_radial)"))
               for h in result.history]
    return (; row=row_with_errors(row, cfg), history)
end

function write_summary_csv(path, rows)
    open(path, "w") do io
        println(io, "method,resolution,backend,precision,steps,avg_window,sample_every,blocks,Nx,Ny,n_arc,n_radial,nodes,solid_cells,dx_ref,D_eff,Re,u_max,u_ref,nu,Cd,Cd_std,Cl,Cl_std,Cd_ref,Cl_ref,Cd_abs_error,Cl_abs_error,Cl_abs_error_flipped,Cl_sign_note,rho_min,rho_max,elapsed_s,MLUPs,mesh_file,status")
        for r in rows
            println(io, join((r.method, r.resolution, r.backend, r.precision,
                r.steps, r.avg_window, r.sample_every, r.blocks, r.Nx, r.Ny,
                r.n_arc, r.n_radial, r.nodes, r.solid_cells, fmt3(r.dx_ref),
                fmt4(r.D_eff), fmt2(r.Re), fmt4(r.u_max), fmt4(r.u_ref),
                fmt3(r.nu), fmt6(r.Cd), fmt3(r.Cd_std), fmt6(r.Cl),
                fmt3(r.Cl_std), fmt6(r.Cd_ref), fmt6(r.Cl_ref),
                fmt6(r.Cd_abs_error), fmt6(r.Cl_abs_error),
                fmt6(r.Cl_abs_error_flipped), r.Cl_sign_note,
                fmt4(r.rho_min), fmt4(r.rho_max), fmt3(r.elapsed_s),
                fmt4(r.MLUPs), r.mesh_file, r.status), ','))
        end
    end
end

function write_history_csv(path, rows)
    extra(r, key) = key in propertynames(r) ? getproperty(r, key) : NaN
    open(path, "w") do io
        println(io, "method,resolution,step,Cd,Cl,Fx,Fy,Fx_pressure,Fy_pressure,Fx_viscous,Fy_viscous")
        for r in rows
            println(io, join((r.method, r.resolution, r.step, fmt6(r.Cd),
                              fmt6(r.Cl), fmt6(r.Fx), fmt6(r.Fy),
                              fmt6(extra(r, :Fx_pressure)),
                              fmt6(extra(r, :Fy_pressure)),
                              fmt6(extra(r, :Fx_viscous)),
                              fmt6(extra(r, :Fy_viscous))), ','))
        end
    end
end

function write_summary_md(path, rows, cfg)
    open(path, "w") do io
        println(io, "# Cylinder 2D Body-Fitted vs Cartesian Convergence")
        println(io)
        println(io, "GPU comparison for Schaefer-Turek 2D-1 style cylinder flow. The body-fitted route is the production `.krk -> Mesh gmsh(.msh) -> Module slbm_drag` path; the Cartesian baseline is LI-BB on a uniform grid.")
        println(io)
        println(io, "Reference values used for error columns:")
        println(io)
        println(io, "- `Cd_ref = $(fmt6(cfg.ref_cd))`")
        println(io, "- `Cl_ref = $(fmt6(cfg.ref_cl))`")
        println(io)
        println(io, "| method | resolution | D_eff | nodes | Cd | |Cd-Cd_ref| | Cl | |Cl-Cl_ref| | rho range | elapsed |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows
            println(io, "| $(r.method) | $(r.resolution) | $(fmt4(r.D_eff)) | $(r.nodes) | $(fmt6(r.Cd)) | $(fmt6(r.Cd_abs_error)) | $(fmt6(r.Cl)) | $(fmt6(r.Cl_abs_error)) | $(fmt4(r.rho_min))-$(fmt4(r.rho_max)) | $(fmt3(r.elapsed_s))s |")
        end
        println(io)
        println(io, "For lift, `Cl_abs_error_flipped` is also written in the CSV to detect a force-sign convention mismatch; the Markdown table reports the native KrakeNN sign.")
        println(io)
        println(io, "This table is useful only if the force histories are temporally settled over the averaging window. Check the companion history CSV before using the errors as paper numbers.")
    end
end

function finite_rows(rows, fields)
    return filter(rows) do r
        all(field -> isfinite(Float64(getfield(r, field))), fields)
    end
end

function write_convergence_plot(path, rows)
    HAS_CAIRO || return false
    rows = finite_rows(rows, (:D_eff, :Cd_abs_error, :Cl_abs_error))
    isempty(rows) && return false
    fig = Figure(size=(980, 440))
    ax1 = Axis(fig[1, 1], xlabel="D_eff", ylabel="|Cd - Cd_ref|",
               title="Drag convergence")
    ax2 = Axis(fig[1, 2], xlabel="D_eff", ylabel="|Cl - Cl_ref|",
               title="Lift convergence")
    for method in unique([r.method for r in rows])
        data = sort(filter(r -> r.method == method, rows), by=r -> r.D_eff)
        lines!(ax1, [r.D_eff for r in data], [r.Cd_abs_error for r in data],
               linewidth=2.5, label=method)
        scatter!(ax1, [r.D_eff for r in data], [r.Cd_abs_error for r in data])
        lines!(ax2, [r.D_eff for r in data], [r.Cl_abs_error for r in data],
               linewidth=2.5, label=method)
        scatter!(ax2, [r.D_eff for r in data], [r.Cl_abs_error for r in data])
    end
    axislegend(ax1, position=:rt, framevisible=false)
    axislegend(ax2, position=:rt, framevisible=false)
    save(path, fig)
    return true
end

function write_coeff_plot(path, rows)
    HAS_CAIRO || return false
    rows = finite_rows(rows, (:D_eff, :Cd, :Cl))
    isempty(rows) && return false
    fig = Figure(size=(980, 440))
    ax1 = Axis(fig[1, 1], xlabel="D_eff", ylabel="Cd",
               title="Drag coefficient")
    ax2 = Axis(fig[1, 2], xlabel="D_eff", ylabel="Cl",
               title="Lift coefficient")
    for method in unique([r.method for r in rows])
        data = sort(filter(r -> r.method == method, rows), by=r -> r.D_eff)
        lines!(ax1, [r.D_eff for r in data], [r.Cd for r in data],
               linewidth=2.5, label=method)
        scatter!(ax1, [r.D_eff for r in data], [r.Cd for r in data])
        lines!(ax2, [r.D_eff for r in data], [r.Cl for r in data],
               linewidth=2.5, label=method)
        scatter!(ax2, [r.D_eff for r in data], [r.Cl for r in data])
    end
    axislegend(ax1, position=:rt, framevisible=false)
    axislegend(ax2, position=:rt, framevisible=false)
    save(path, fig)
    return true
end

function write_history_plot(path, rows)
    HAS_CAIRO || return false
    rows = finite_rows(rows, (:step, :Cd, :Cl))
    isempty(rows) && return false
    fig = Figure(size=(1050, 520))
    ax1 = Axis(fig[1, 1], xlabel="step", ylabel="Cd",
               title="Drag history")
    ax2 = Axis(fig[2, 1], xlabel="step", ylabel="Cl",
               title="Lift history")
    for key in unique([(r.method, r.resolution) for r in rows])
        data = sort(filter(r -> (r.method, r.resolution) == key, rows),
                    by=r -> r.step)
        label = "$(key[1]) $(key[2])"
        lines!(ax1, [r.step for r in data], [r.Cd for r in data],
               linewidth=2.0, label=label)
        lines!(ax2, [r.step for r in data], [r.Cl for r in data],
               linewidth=2.0, label=label)
    end
    axislegend(ax1, position=:rt, framevisible=false, labelsize=10)
    save(path, fig)
    return true
end

function _parse_float_cell(s)
    isempty(strip(s)) && return NaN
    return parse(Float64, strip(s))
end

function _read_csv_rows(path)
    isfile(path) || error("missing CSV file: $path")
    lines = collect(eachline(path))
    isempty(lines) && return Dict{Symbol,String}[]
    header = Symbol.(split(lines[1], ','))
    rows = Dict{Symbol,String}[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        cells = split(line, ','; keepempty=true)
        length(cells) == length(header) ||
            error("invalid CSV row in $path: expected $(length(header)) cells, got $(length(cells))")
        push!(rows, Dict(header[i] => String(cells[i]) for i in eachindex(header)))
    end
    return rows
end

function read_summary_plot_rows(path)
    return [(
        method=r[:method],
        resolution=r[:resolution],
        D_eff=_parse_float_cell(r[:D_eff]),
        Cd=_parse_float_cell(r[:Cd]),
        Cl=_parse_float_cell(r[:Cl]),
        Cd_abs_error=_parse_float_cell(r[:Cd_abs_error]),
        Cl_abs_error=_parse_float_cell(r[:Cl_abs_error]),
    ) for r in _read_csv_rows(path)]
end

function read_history_plot_rows(path)
    return [(
        method=r[:method],
        resolution=r[:resolution],
        step=parse(Int, r[:step]),
        Cd=_parse_float_cell(r[:Cd]),
        Cl=_parse_float_cell(r[:Cl]),
    ) for r in _read_csv_rows(path)]
end

function plot_existing_outputs(suffix)
    summary_csv = joinpath(TABLEDIR, "cylinder2d_convergence_compare$(suffix).csv")
    history_csv = joinpath(TABLEDIR, "cylinder2d_convergence_history$(suffix).csv")
    conv_plot_png = joinpath(PLOTDIR, "paper_cylinder2d_convergence_compare$(suffix).png")
    coeff_plot_png = joinpath(PLOTDIR, "paper_cylinder2d_coefficients$(suffix).png")
    history_plot_png = joinpath(PLOTDIR, "paper_cylinder2d_force_history$(suffix).png")
    rows = read_summary_plot_rows(summary_csv)
    history = read_history_plot_rows(history_csv)
    made_conv_plot = write_convergence_plot(conv_plot_png, rows)
    made_coeff_plot = write_coeff_plot(coeff_plot_png, rows)
    made_history_plot = write_history_plot(history_plot_png, history)
    println("Wrote:")
    made_conv_plot && println("  ", relpath(conv_plot_png, pwd()))
    made_coeff_plot && println("  ", relpath(coeff_plot_png, pwd()))
    made_history_plot && println("  ", relpath(history_plot_png, pwd()))
    return made_conv_plot || made_coeff_plot || made_history_plot
end

function main()
    suffix = output_suffix()
    if get(ENV, "KRK_CYL_CONV_PLOT_ONLY", "0") == "1"
        plot_existing_outputs(suffix)
        return nothing
    end

    cfg = cfg_from_env()
    backend_info = select_backend()
    tag = isempty(suffix) ? "latest" : suffix[2:end]

    println("=== Cylinder 2D convergence: body-fitted vs Cartesian LI-BB ===")
    println("backend=$(backend_info.label) precision=$(backend_info.T)")
    println("steps=$(cfg.steps) avg_window=$(cfg.avg_window) sample_every=$(cfg.sample_every)")
    println("cy=$(cfg.cy) bodyfit_cylinder_ghost_alpha=$(cfg.bodyfit_cylinder_ghost_alpha)")
    cart_list = join(cfg.cart_deffs, ",")
    ogrid_list = join(["$(s.n_arc)x$(s.n_radial)" for s in cfg.ogrid_specs], ",")
    println("cartesian D_eff targets=$cart_list")
    println("O-grid specs=$ogrid_list")

    rows = NamedTuple[]
    history = NamedTuple[]
    for deff in cfg.cart_deffs
        println("-- cartesian_libb D_eff target=$(fmt2(deff))")
        try
            result = run_cartesian_libb_case(cfg, backend_info, deff)
            push!(rows, result.row)
            append!(history, result.history)
            println("   Cd=$(fmt6(result.row.Cd)) Cl=$(fmt6(result.row.Cl)) errors=($(fmt3(result.row.Cd_abs_error)), $(fmt3(result.row.Cl_abs_error))) elapsed=$(fmt3(result.row.elapsed_s))s")
        catch err
            status = classify_error(err)
            dx = 2.0 * cfg.R / Float64(deff)
            Nx = round(Int, cfg.Lx / dx) + 1
            Ny = round(Int, cfg.Ly / dx) + 1
            row = failed_row(; method="cartesian_libb",
                resolution="D$(round(Int, deff))", backend_info, cfg,
                blocks=1, Nx=Nx, Ny=Ny, nodes=Nx * Ny, dx_ref=dx,
                D_eff=Float64(deff), status=status)
            push!(rows, row)
            @warn "cartesian_libb failed" deff status exception = err
        end
    end
    for spec in cfg.ogrid_specs
        println("-- bodyfit_gmsh_slbm $(spec.n_arc)x$(spec.n_radial)")
        try
            result = run_bodyfit_ogrid_case(cfg, backend_info, spec, tag)
            push!(rows, result.row)
            append!(history, result.history)
            println("   D_eff=$(fmt4(result.row.D_eff)) Cd=$(fmt6(result.row.Cd)) Cl=$(fmt6(result.row.Cl)) errors=($(fmt3(result.row.Cd_abs_error)), $(fmt3(result.row.Cl_abs_error))) elapsed=$(fmt3(result.row.elapsed_s))s")
        catch err
            status = classify_error(err)
            mesh_info = estimate_bodyfit_mesh_row(cfg, spec, tag)
            row = failed_row(; method="bodyfit_gmsh_slbm",
                resolution="a$(spec.n_arc)_r$(spec.n_radial)",
                backend_info, cfg, blocks=mesh_info.blocks, n_arc=spec.n_arc,
                n_radial=spec.n_radial, nodes=mesh_info.nodes,
                dx_ref=mesh_info.dx_ref, D_eff=mesh_info.D_eff,
                mesh_file=mesh_info.mesh_file, status=status)
            push!(rows, row)
            @warn "bodyfit_gmsh_slbm failed" spec status exception = err
        end
    end

    summary_csv = joinpath(TABLEDIR, "cylinder2d_convergence_compare$(suffix).csv")
    history_csv = joinpath(TABLEDIR, "cylinder2d_convergence_history$(suffix).csv")
    summary_md = joinpath(TABLEDIR, "cylinder2d_convergence_compare$(suffix).md")
    conv_plot_png = joinpath(PLOTDIR, "paper_cylinder2d_convergence_compare$(suffix).png")
    coeff_plot_png = joinpath(PLOTDIR, "paper_cylinder2d_coefficients$(suffix).png")
    history_plot_png = joinpath(PLOTDIR, "paper_cylinder2d_force_history$(suffix).png")
    write_summary_csv(summary_csv, rows)
    write_history_csv(history_csv, history)
    write_summary_md(summary_md, rows, cfg)
    made_conv_plot = write_convergence_plot(conv_plot_png, rows)
    made_coeff_plot = write_coeff_plot(coeff_plot_png, rows)
    made_history_plot = write_history_plot(history_plot_png, history)

    println("Wrote:")
    for path in (summary_csv, history_csv, summary_md)
        println("  ", relpath(path, pwd()))
    end
    made_conv_plot && println("  ", relpath(conv_plot_png, pwd()))
    made_coeff_plot && println("  ", relpath(coeff_plot_png, pwd()))
    made_history_plot && println("  ", relpath(history_plot_png, pwd()))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
