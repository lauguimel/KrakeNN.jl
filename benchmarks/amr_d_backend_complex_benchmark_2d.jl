#!/usr/bin/env julia

using Dates
using KernelAbstractions
using KrakeNN
using Printf

const REQUESTED_BACKEND = lowercase(get(ENV, "KRK_AMR_D_BACKEND", "cpu"))
const PRECISION_NAME = lowercase(get(
    ENV, "KRK_AMR_D_BENCH_T", REQUESTED_BACKEND == "metal" ? "float32" : "float64"))
const RUN_T = PRECISION_NAME in ("float32", "f32") ? Float32 : Float64
const STEPS = parse(Int, get(ENV, "KRK_AMR_D_BENCH_STEPS", "240"))
const AVG_WINDOW = parse(Int, get(ENV, "KRK_AMR_D_BENCH_AVG_WINDOW", "60"))
const WARMUP_STEPS = parse(Int, get(ENV, "KRK_AMR_D_BENCH_WARMUP", "10"))
const FLOWS = Symbol.(split(get(ENV, "KRK_AMR_D_BENCH_FLOWS",
                                "bfs,square,cylinder"), ','))
const OUTDIR = get(ENV, "KRK_AMR_D_BENCH_OUTDIR",
                   joinpath(dirname(@__DIR__), "benchmarks", "results"))
const TAG = get(ENV, "KRK_AMR_D_BENCH_TAG",
                "local_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS"))

mkpath(OUTDIR)

function _load_cuda_module()
    return Base.require(Base.PkgId(
        Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))
end

function _load_metal_module()
    return Base.require(Base.PkgId(
        Base.UUID("dde4c033-4e86-420c-a63e-0dd931031962"), "Metal"))
end

function _device_array(backend_name::AbstractString, host)
    if backend_name == "cpu"
        return copy(host)
    elseif backend_name == "cuda"
        cuda = _load_cuda_module()
        return Base.invokelatest(cuda.CuArray, host)
    elseif backend_name == "metal"
        metal = _load_metal_module()
        return Base.invokelatest(metal.MtlArray, host)
    end
    throw(ArgumentError("unknown backend $backend_name"))
end

function _host_array(backend_name::AbstractString, dev)
    backend_name == "cpu" && return copy(dev)
    return Base.invokelatest(Array, dev)
end

function _launch_kernel!(kernel!, args...; ndrange)
    return Base.invokelatest(kernel!, args...; ndrange=ndrange)
end

function _sync_backend(backend)
    return Base.invokelatest(KernelAbstractions.synchronize, backend)
end

@inline function _bench_feq(w, rho, ux, uy, usq, cx, cy)
    T = typeof(rho)
    cu = T(cx) * ux + T(cy) * uy
    return w * rho * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end

@kernel function _collide_guo_periodic_solid_2d_kernel!(
        f, @Const(is_solid), omega, Fx, Fy)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if is_solid[i, j] == UInt8(0)
            T = eltype(f)
            f1 = f[i, j, 1]; f2 = f[i, j, 2]; f3 = f[i, j, 3]
            f4 = f[i, j, 4]; f5 = f[i, j, 5]; f6 = f[i, j, 6]
            f7 = f[i, j, 7]; f8 = f[i, j, 8]; f9 = f[i, j, 9]
            fx = T(Fx)
            fy = T(Fy)
            om = T(omega)

            rho = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
            ux = ((f2 - f4 + f6 - f7 - f8 + f9) + fx / T(2)) / rho
            uy = ((f3 - f5 + f6 + f7 - f8 - f9) + fy / T(2)) / rho
            usq = ux * ux + uy * uy
            guo_pref = one(T) - om / T(2)

            w1 = T(4) / T(9)
            w2 = T(1) / T(9)
            w6 = T(1) / T(36)

            sq = w1 * (T(3) * ((-ux) * fx + (-uy) * fy))
            f[i, j, 1] = f1 - om * (f1 - _bench_feq(w1, rho, ux, uy, usq, 0, 0)) +
                         guo_pref * sq

            sq = w2 * (T(3) * ((one(T) - ux) * fx + (-uy) * fy) +
                       T(9) * ux * fx)
            f[i, j, 2] = f2 - om * (f2 - _bench_feq(w2, rho, ux, uy, usq, 1, 0)) +
                         guo_pref * sq

            sq = w2 * (T(3) * ((-ux) * fx + (one(T) - uy) * fy) +
                       T(9) * uy * fy)
            f[i, j, 3] = f3 - om * (f3 - _bench_feq(w2, rho, ux, uy, usq, 0, 1)) +
                         guo_pref * sq

            sq = w2 * (T(3) * ((-one(T) - ux) * fx + (-uy) * fy) +
                       T(9) * ux * fx)
            f[i, j, 4] = f4 - om * (f4 - _bench_feq(w2, rho, ux, uy, usq, -1, 0)) +
                         guo_pref * sq

            sq = w2 * (T(3) * ((-ux) * fx + (-one(T) - uy) * fy) +
                       T(9) * uy * fy)
            f[i, j, 5] = f5 - om * (f5 - _bench_feq(w2, rho, ux, uy, usq, 0, -1)) +
                         guo_pref * sq

            sq = w6 * (T(3) * ((one(T) - ux) * fx + (one(T) - uy) * fy) +
                       T(9) * (ux + uy) * (fx + fy))
            f[i, j, 6] = f6 - om * (f6 - _bench_feq(w6, rho, ux, uy, usq, 1, 1)) +
                         guo_pref * sq

            sq = w6 * (T(3) * ((-one(T) - ux) * fx + (one(T) - uy) * fy) +
                       T(9) * (-ux + uy) * (-fx + fy))
            f[i, j, 7] = f7 - om * (f7 - _bench_feq(w6, rho, ux, uy, usq, -1, 1)) +
                         guo_pref * sq

            sq = w6 * (T(3) * ((-one(T) - ux) * fx + (-one(T) - uy) * fy) +
                       T(9) * (-ux - uy) * (-fx - fy))
            f[i, j, 8] = f8 - om * (f8 - _bench_feq(w6, rho, ux, uy, usq, -1, -1)) +
                         guo_pref * sq

            sq = w6 * (T(3) * ((one(T) - ux) * fx + (-one(T) - uy) * fy) +
                       T(9) * (ux - uy) * (fx - fy))
            f[i, j, 9] = f9 - om * (f9 - _bench_feq(w6, rho, ux, uy, usq, 1, -1)) +
                         guo_pref * sq
        end
    end
end

@kernel function _stream_periodic_x_wall_y_solid_2d_kernel!(
        f_out, @Const(f_in), @Const(is_solid), nx, ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if is_solid[i, j] != UInt8(0)
            f_out[i, j, 1] = f_in[i, j, 1]
            f_out[i, j, 2] = f_in[i, j, 2]
            f_out[i, j, 3] = f_in[i, j, 3]
            f_out[i, j, 4] = f_in[i, j, 4]
            f_out[i, j, 5] = f_in[i, j, 5]
            f_out[i, j, 6] = f_in[i, j, 6]
            f_out[i, j, 7] = f_in[i, j, 7]
            f_out[i, j, 8] = f_in[i, j, 8]
            f_out[i, j, 9] = f_in[i, j, 9]
        else

            f_out[i, j, 1] = f_in[i, j, 1]

            isrc = ifelse(i > 1, i - 1, nx)
            f_out[i, j, 2] = is_solid[isrc, j] != UInt8(0) ? f_in[i, j, 4] :
                             f_in[isrc, j, 2]

            jsrc = j - 1
            f_out[i, j, 3] = jsrc < 1 || is_solid[i, jsrc] != UInt8(0) ?
                             f_in[i, j, 5] : f_in[i, jsrc, 3]

            isrc = ifelse(i < nx, i + 1, 1)
            f_out[i, j, 4] = is_solid[isrc, j] != UInt8(0) ? f_in[i, j, 2] :
                             f_in[isrc, j, 4]

            jsrc = j + 1
            f_out[i, j, 5] = jsrc > ny || is_solid[i, jsrc] != UInt8(0) ?
                             f_in[i, j, 3] : f_in[i, jsrc, 5]

            isrc = ifelse(i > 1, i - 1, nx)
            jsrc = j - 1
            f_out[i, j, 6] = jsrc < 1 || is_solid[isrc, jsrc] != UInt8(0) ?
                             f_in[i, j, 8] : f_in[isrc, jsrc, 6]

            isrc = ifelse(i < nx, i + 1, 1)
            jsrc = j - 1
            f_out[i, j, 7] = jsrc < 1 || is_solid[isrc, jsrc] != UInt8(0) ?
                             f_in[i, j, 9] : f_in[isrc, jsrc, 7]

            isrc = ifelse(i < nx, i + 1, 1)
            jsrc = j + 1
            f_out[i, j, 8] = jsrc > ny || is_solid[isrc, jsrc] != UInt8(0) ?
                             f_in[i, j, 6] : f_in[isrc, jsrc, 8]

            isrc = ifelse(i > 1, i - 1, nx)
            jsrc = j + 1
            f_out[i, j, 9] = jsrc > ny || is_solid[isrc, jsrc] != UInt8(0) ?
                             f_in[i, j, 7] : f_in[isrc, jsrc, 9]
        end
    end
end

function _resolve_backend(name::AbstractString)
    if name == "cpu"
        return KernelAbstractions.CPU(), "cpu"
    elseif name == "cuda"
        try
            cuda = _load_cuda_module()
            Base.invokelatest(cuda.functional) || error("CUDA is not functional")
            return Base.invokelatest(cuda.CUDABackend), "cuda"
        catch err
            error("requested CUDA backend is unavailable: $err")
        end
    elseif name == "metal"
        try
            metal = _load_metal_module()
            Base.invokelatest(metal.functional) || error("Metal is not functional")
            return Base.invokelatest(metal.MetalBackend), "metal"
        catch err
            error("requested Metal backend is unavailable: $err")
        end
    elseif name == "auto"
        try
            cuda = _load_cuda_module()
            Base.invokelatest(cuda.functional) &&
                return Base.invokelatest(cuda.CUDABackend), "cuda"
        catch
        end
        try
            metal = _load_metal_module()
            Base.invokelatest(metal.functional) &&
                return Base.invokelatest(metal.MetalBackend), "metal"
        catch
        end
        return KernelAbstractions.CPU(), "cpu"
    end
    throw(ArgumentError("KRK_AMR_D_BACKEND must be cpu, metal, cuda, or auto"))
end

function _fill_equilibrium_host!(f::Array{T,3}; rho=T(1), ux=T(0), uy=T(0)) where T
    usq = ux * ux + uy * uy
    ws = (T(4) / T(9), T(1) / T(9), T(1) / T(9), T(1) / T(9), T(1) / T(9),
          T(1) / T(36), T(1) / T(36), T(1) / T(36), T(1) / T(36))
    cxs = (0, 1, 0, -1, 0, 1, -1, -1, 1)
    cys = (0, 0, 1, 0, -1, 1, 1, -1, -1)
    @inbounds for q in 1:9, j in axes(f, 2), i in axes(f, 1)
        f[i, j, q] = _bench_feq(ws[q], rho, ux, uy, usq, cxs[q], cys[q])
    end
    return f
end

function _solid_mask_for_periodic_flow(flow::Symbol, nx::Int, ny::Int)
    if flow == :square
        return square_solid_mask_leaf_2d(nx, ny, 22:27, 12:17)
    elseif flow == :cylinder
        return cylinder_solid_mask_leaf_2d(nx, ny, (nx + 1) / 2, (ny + 1) / 2, 3.0)
    end
    throw(ArgumentError("backend periodic-solid benchmark supports square and cylinder only"))
end

function _fluid_metrics_density(f::Array{T,3}, solid::AbstractMatrix{UInt8};
        force_x=T(0), force_y=T(0)) where T
    mass = zero(T)
    ux_sum = zero(T)
    uy_sum = zero(T)
    n = 0
    @inbounds for j in axes(f, 2), i in axes(f, 1)
        solid[i, j] != UInt8(0) && continue
        f1 = f[i, j, 1]; f2 = f[i, j, 2]; f3 = f[i, j, 3]
        f4 = f[i, j, 4]; f5 = f[i, j, 5]; f6 = f[i, j, 6]
        f7 = f[i, j, 7]; f8 = f[i, j, 8]; f9 = f[i, j, 9]
        rho = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
        ux = ((f2 - f4 + f6 - f7 - f8 + f9) + force_x / T(2)) / rho
        uy = ((f3 - f5 + f6 + f7 - f8 - f9) + force_y / T(2)) / rho
        mass += rho
        ux_sum += ux
        uy_sum += uy
        n += 1
    end
    return (; mass, ux_mean=ux_sum / T(n), uy_mean=uy_sum / T(n), fluid_cells=n)
end

function _active_composite_sites(result)
    coarse = getproperty(result, :coarse_F)
    patch = getproperty(result, :patch)
    np = length(patch.parent_i_range) * length(patch.parent_j_range)
    return size(coarse, 1) * size(coarse, 2) - np + 4 * np
end

function _mass_rel_drift(result)
    m0 = Float64(getproperty(result, :mass_initial))
    dm = Float64(getproperty(result, :mass_drift))
    return abs(dm) / max(abs(m0), eps(Float64))
end

function _safe_get(result, field::Symbol, default=NaN)
    return hasproperty(result, field) ? getproperty(result, field) : default
end

function _flow_force(flow::Symbol)
    flow == :bfs && return (0.0, 0.0)
    return (2e-5, 0.0)
end

function _run_route_flow(flow::Symbol)
    if flow == :bfs
        return run_conservative_tree_bfs_route_native_2d(; steps=STEPS, T=RUN_T)
    elseif flow == :square
        return run_conservative_tree_square_obstacle_route_native_2d(; steps=STEPS, T=RUN_T)
    elseif flow == :cylinder
        return run_conservative_tree_cylinder_obstacle_route_native_2d(
            ; steps=STEPS, avg_window=min(AVG_WINDOW, STEPS), T=RUN_T)
    end
    throw(ArgumentError("unknown flow $flow"))
end

function _run_cart_flow(flow::Symbol)
    if flow == :bfs
        return run_conservative_tree_bfs_macroflow_2d(; steps=STEPS, T=RUN_T)
    elseif flow == :square
        return run_conservative_tree_square_obstacle_macroflow_2d(; steps=STEPS, T=RUN_T)
    elseif flow == :cylinder
        return run_conservative_tree_cylinder_macroflow_2d(
            ; steps=STEPS, avg_window=min(AVG_WINDOW, STEPS), T=RUN_T)
    end
    throw(ArgumentError("unknown flow $flow"))
end

function _result_row(; flow, method, requested_backend, backend_used, precision,
        nx_leaf, ny_leaf, steps, kernel_cells, active_cells, ux_mean, uy_mean,
        cd, mass_rel_drift, elapsed_s, mlups, status)
    return (timestamp=string(Dates.now()), tag=TAG, flow=String(flow),
            method=String(method), requested_backend=requested_backend,
            backend_used=backend_used, precision=precision, Nx_leaf=nx_leaf,
            Ny_leaf=ny_leaf, steps=steps, kernel_cells=kernel_cells,
            active_cells=active_cells, ux_mean=Float64(ux_mean),
            uy_mean=Float64(uy_mean), Cd=Float64(cd),
            mass_rel_drift=Float64(mass_rel_drift), elapsed_s=Float64(elapsed_s),
            MLUPs=Float64(mlups), status=status)
end

function _cpu_result_rows(flow::Symbol)
    rows = NamedTuple[]

    route_elapsed = @elapsed route = _run_route_flow(flow)
    nx_leaf = 2 * size(getproperty(route, :coarse_F), 1)
    ny_leaf = 2 * size(getproperty(route, :coarse_F), 2)
    route_sites = _active_composite_sites(route)
    route_ux = _safe_get(route, :ux_mean, _safe_get(route, :u_ref))
    route_uy = _safe_get(route, :uy_mean, 0.0)
    push!(rows, _result_row(
        flow=flow, method=:amr_d_route_native_cpu,
        requested_backend=REQUESTED_BACKEND, backend_used="cpu",
        precision=string(RUN_T), nx_leaf=nx_leaf, ny_leaf=ny_leaf,
        steps=STEPS, kernel_cells=route_sites, active_cells=route_sites,
        ux_mean=route_ux, uy_mean=route_uy, cd=_safe_get(route, :Cd),
        mass_rel_drift=_mass_rel_drift(route), elapsed_s=route_elapsed,
        mlups=route_sites * STEPS / max(route_elapsed, eps()) / 1e6,
        status="ok_cpu_only_refined"))

    cart_elapsed = @elapsed cart = _run_cart_flow(flow)
    cart_sites = 4 * size(getproperty(cart, :coarse_F), 1) *
                 size(getproperty(cart, :coarse_F), 2)
    cart_ux = _safe_get(cart, :ux_mean, _safe_get(cart, :u_ref))
    cart_uy = _safe_get(cart, :uy_mean, 0.0)
    push!(rows, _result_row(
        flow=flow, method=:leaf_cartesian_reference_cpu,
        requested_backend=REQUESTED_BACKEND, backend_used="cpu",
        precision=string(RUN_T), nx_leaf=nx_leaf, ny_leaf=ny_leaf,
        steps=STEPS, kernel_cells=cart_sites, active_cells=cart_sites,
        ux_mean=cart_ux, uy_mean=cart_uy, cd=_safe_get(cart, :Cd),
        mass_rel_drift=_mass_rel_drift(cart), elapsed_s=cart_elapsed,
        mlups=cart_sites * STEPS / max(cart_elapsed, eps()) / 1e6,
        status="ok_leaf_reference"))

    return rows
end

function _backend_periodic_solid_row(flow::Symbol, backend, backend_name::String)
    nx = 48
    ny = 28
    Fx, Fy = _flow_force(flow)
    solid_bool = _solid_mask_for_periodic_flow(flow, nx, ny)
    solid_host = Array{UInt8}(solid_bool)
    f_host = zeros(RUN_T, nx, ny, 9)
    _fill_equilibrium_host!(f_host)

    f_a = _device_array(backend_name, f_host)
    f_b = _device_array(backend_name, f_host)
    solid_dev = _device_array(backend_name, solid_host)

    collide! = _collide_guo_periodic_solid_2d_kernel!(backend)
    stream! = _stream_periodic_x_wall_y_solid_2d_kernel!(backend)
    for _ in 1:WARMUP_STEPS
        _launch_kernel!(collide!, f_a, solid_dev, RUN_T(1), RUN_T(Fx), RUN_T(Fy);
                        ndrange=(nx, ny))
        _launch_kernel!(stream!, f_b, f_a, solid_dev, nx, ny; ndrange=(nx, ny))
        f_a, f_b = f_b, f_a
    end
    _sync_backend(backend)

    start_host = _host_array(backend_name, f_a)
    start_metrics = _fluid_metrics_density(start_host, solid_host;
                                           force_x=RUN_T(Fx), force_y=RUN_T(Fy))

    elapsed = @elapsed begin
        for _ in 1:STEPS
            _launch_kernel!(
                collide!, f_a, solid_dev, RUN_T(1), RUN_T(Fx), RUN_T(Fy);
                ndrange=(nx, ny))
            _launch_kernel!(stream!, f_b, f_a, solid_dev, nx, ny;
                            ndrange=(nx, ny))
            f_a, f_b = f_b, f_a
        end
        _sync_backend(backend)
    end

    finish_host = _host_array(backend_name, f_a)
    metrics = _fluid_metrics_density(finish_host, solid_host;
                                     force_x=RUN_T(Fx), force_y=RUN_T(Fy))
    mass_rel = abs(Float64(metrics.mass - start_metrics.mass)) /
               max(abs(Float64(start_metrics.mass)), eps(Float64))
    kernel_cells = nx * ny
    return _result_row(
        flow=flow, method=:leaf_cartesian_backend_periodic_solid,
        requested_backend=REQUESTED_BACKEND, backend_used=backend_name,
        precision=string(RUN_T), nx_leaf=nx, ny_leaf=ny, steps=STEPS,
        kernel_cells=kernel_cells, active_cells=metrics.fluid_cells,
        ux_mean=metrics.ux_mean, uy_mean=metrics.uy_mean, cd=NaN,
        mass_rel_drift=mass_rel, elapsed_s=elapsed,
        mlups=kernel_cells * STEPS / max(elapsed, eps()) / 1e6,
        status="ok_backend_periodic_solid")
end

function _backend_unsupported_row(flow::Symbol, backend_name::String)
    return _result_row(
        flow=flow, method=:leaf_cartesian_backend_periodic_solid,
        requested_backend=REQUESTED_BACKEND, backend_used=backend_name,
        precision=string(RUN_T), nx_leaf=56, ny_leaf=28, steps=STEPS,
        kernel_cells=0, active_cells=0, ux_mean=NaN, uy_mean=NaN, cd=NaN,
        mass_rel_drift=NaN, elapsed_s=0.0, mlups=NaN,
        status="unsupported_open_boundary_gpu_parity")
end

function _write_csv(rows)
    path = joinpath(OUTDIR, "amr_d_backend_complex_benchmark_2d_$(TAG).csv")
    header = (:timestamp, :tag, :flow, :method, :requested_backend, :backend_used,
              :precision, :Nx_leaf, :Ny_leaf, :steps, :kernel_cells,
              :active_cells, :ux_mean, :uy_mean, :Cd, :mass_rel_drift,
              :elapsed_s, :MLUPs, :status)
    open(path, "w") do io
        println(io, join(String.(header), ","))
        for r in rows
            values = (getproperty(r, h) for h in header)
            println(io, join(values, ","))
        end
    end
    return path
end

function main()
    backend, backend_name = _resolve_backend(REQUESTED_BACKEND)
    rows = NamedTuple[]
    println("AMR-D backend benchmark tag=$TAG backend=$backend_name T=$RUN_T steps=$STEPS")
    for flow in FLOWS
        append!(rows, _cpu_result_rows(flow))
        if flow in (:square, :cylinder)
            push!(rows, _backend_periodic_solid_row(flow, backend, backend_name))
        elseif flow == :bfs
            push!(rows, _backend_unsupported_row(flow, backend_name))
        else
            throw(ArgumentError("unknown flow $flow"))
        end
    end
    path = _write_csv(rows)
    println("wrote ", path)
    for r in rows
        @printf("%-9s %-38s %-5s %10.3f MLUPS status=%s\n",
                r.flow, r.method, r.backend_used, r.MLUPs, r.status)
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
