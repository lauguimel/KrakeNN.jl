using KernelAbstractions
using Random
using Test

const INCNS_GDL_OPERATOR_PATH = joinpath(
    @__DIR__, "..", "..", "src", "fvfd", "operators_2d_grad_div_laplacian.jl",
)
const INCNS_GDL_POISSON_PATH = joinpath(@__DIR__, "..", "..", "src", "solve", "poisson_embedded.jl")

if !isdefined(@__MODULE__, :gdl_divergence_2d!)
    include(INCNS_GDL_OPERATOR_PATH)
end

# Guard on the EXPORTED assemble_poisson_embedded (tilted_half_plane_fractions
# is unexported, so probing it would always re-include under `using KrakeNN`).
if !isdefined(@__MODULE__, :assemble_poisson_embedded)
    include(INCNS_GDL_POISSON_PATH)
end

const INCNS_GDL_NS = (16, 32, 64, 128)
const INCNS_GDL_RESULTS = Dict{Symbol,Any}()

incns_gdl_x(i, h) = (Float64(i) - 0.5) * h
incns_gdl_periodic_bc() = (KrakeNN.FVFD_BC_PERIODIC, KrakeNN.FVFD_BC_PERIODIC,
                           KrakeNN.FVFD_BC_PERIODIC, KrakeNN.FVFD_BC_PERIODIC)
incns_gdl_wall_bc() = (KrakeNN.FVFD_BC_WALL, KrakeNN.FVFD_BC_WALL,
                       KrakeNN.FVFD_BC_WALL, KrakeNN.FVFD_BC_WALL)

function incns_gdl_regular_geometry(N::Integer)
    return falses(N, N)
end

function incns_gdl_full_fractions(N::Integer)
    west = ones(Float64, N, N)
    east = ones(Float64, N, N)
    south = ones(Float64, N, N)
    north = ones(Float64, N, N)
    cell = ones(Float64, N, N)
    return west, east, south, north, cell
end

function incns_gdl_l2_error(actual, exact::Function, N::Integer)
    h = 1.0 / Float64(N)
    err2 = 0.0
    for j in 1:N, i in 1:N
        x = incns_gdl_x(i, h)
        y = incns_gdl_x(j, h)
        diff = actual[i, j] - exact(x, y)
        err2 += diff * diff
    end
    return sqrt(h * h * err2)
end

function incns_gdl_convergence_result(error_case)
    errors = Float64[]
    orders = Float64[]
    previous_error = NaN
    for N in INCNS_GDL_NS
        err = error_case(N)
        push!(errors, err)
        if !isnan(previous_error)
            push!(orders, log2(previous_error / err))
        end
        previous_error = err
    end
    mean_order = sum(orders) / length(orders)
    @test all(order -> 1.8 <= order <= 2.2, orders)
    return (; N=INCNS_GDL_NS, errors, orders, mean_order)
end

function incns_gdl_divergence_mms(; backend=CPU())
    k = 2.0 * pi
    result = incns_gdl_convergence_result() do N
        h = 1.0 / Float64(N)
        is_solid = incns_gdl_regular_geometry(N)
        ux = Matrix{Float64}(undef, N, N)
        uy = zeros(Float64, N, N)
        divu = zeros(Float64, N, N)
        for j in 1:N, i in 1:N
            x = incns_gdl_x(i, h)
            y = incns_gdl_x(j, h)
            ux[i, j] = sin(k * x) * sin(k * y)
        end
        gdl_divergence_2d!(divu, ux, uy, is_solid, h, h, incns_gdl_periodic_bc()...; backend)
        return incns_gdl_l2_error(divu, (x, y) -> k * cos(k * x) * sin(k * y), N)
    end
    return result
end

function incns_gdl_pressure_gradient_mms(; backend=CPU())
    k = 2.0 * pi
    x_result = incns_gdl_convergence_result() do N
        h = 1.0 / Float64(N)
        is_solid = incns_gdl_regular_geometry(N)
        p = Matrix{Float64}(undef, N, N)
        gpx = zeros(Float64, N, N)
        gpy = zeros(Float64, N, N)
        for j in 1:N, i in 1:N
            x = incns_gdl_x(i, h)
            y = incns_gdl_x(j, h)
            p[i, j] = sin(k * x) * sin(k * y)
        end
        gdl_pressure_gradient_2d!(gpx, gpy, p, is_solid, h, h, incns_gdl_periodic_bc()...; backend)
        return incns_gdl_l2_error(gpx, (x, y) -> k * cos(k * x) * sin(k * y), N)
    end

    y_result = incns_gdl_convergence_result() do N
        h = 1.0 / Float64(N)
        is_solid = incns_gdl_regular_geometry(N)
        p = Matrix{Float64}(undef, N, N)
        gpx = zeros(Float64, N, N)
        gpy = zeros(Float64, N, N)
        for j in 1:N, i in 1:N
            x = incns_gdl_x(i, h)
            y = incns_gdl_x(j, h)
            p[i, j] = sin(k * x) * sin(k * y)
        end
        gdl_pressure_gradient_2d!(gpx, gpy, p, is_solid, h, h, incns_gdl_periodic_bc()...; backend)
        return incns_gdl_l2_error(gpy, (x, y) -> k * sin(k * x) * cos(k * y), N)
    end

    return (; x=x_result, y=y_result)
end

function incns_gdl_laplacian_mms(; backend=CPU())
    k = 2.0 * pi
    result = incns_gdl_convergence_result() do N
        h = 1.0 / Float64(N)
        is_solid = incns_gdl_regular_geometry(N)
        u = Matrix{Float64}(undef, N, N)
        lap = zeros(Float64, N, N)
        for j in 1:N, i in 1:N
            x = incns_gdl_x(i, h)
            y = incns_gdl_x(j, h)
            u[i, j] = sin(k * x) * sin(k * y)
        end
        gdl_laplacian_apply_2d!(lap, u, is_solid, h, h, incns_gdl_periodic_bc()...; backend)
        return incns_gdl_l2_error(lap, (x, y) -> -2.0 * k * k * sin(k * x) * sin(k * y), N)
    end
    return result
end

function incns_gdl_duality_gold(; N::Integer=32, backend=CPU())
    Random.seed!(0x1c_7d15)
    h = 1.0 / Float64(N)
    is_solid = incns_gdl_regular_geometry(N)
    west, east, south, north, cell = incns_gdl_full_fractions(N)
    ux = randn(Float64, N, N)
    uy = randn(Float64, N, N)
    p = randn(Float64, N, N)
    divu = zeros(Float64, N, N)
    gpx = zeros(Float64, N, N)
    gpy = zeros(Float64, N, N)

    bc = incns_gdl_periodic_bc()
    gdl_divergence_embedded_2d!(
        divu, ux, uy, is_solid, west, east, south, north, cell, h, h, bc...; backend,
    )
    gdl_pressure_gradient_embedded_2d!(
        gpx, gpy, p, is_solid, west, east, south, north, cell, h, h, bc...; backend,
    )

    residual = h * h * (
        sum(cell .* divu .* p) +
        sum(cell .* (ux .* gpx .+ uy .* gpy))
    )
    @test abs(residual) < 1.0e-10
    return (; N, residual=abs(residual))
end

function incns_gdl_cell_fractions_from_staggered(face_frac_x, face_frac_y, vol_frac)
    N = size(vol_frac, 1)
    west = Matrix{Float64}(undef, N, N)
    east = Matrix{Float64}(undef, N, N)
    south = Matrix{Float64}(undef, N, N)
    north = Matrix{Float64}(undef, N, N)
    is_solid = falses(N, N)
    for j in 1:N, i in 1:N
        west[i, j] = face_frac_x[i, j]
        east[i, j] = face_frac_x[i + 1, j]
        south[i, j] = face_frac_y[i, j]
        north[i, j] = face_frac_y[i, j + 1]
        is_solid[i, j] = vol_frac[i, j] <= 0.0
    end
    return is_solid, west, east, south, north
end

function incns_gdl_full_fluid_max_abs(field, vol_frac)
    N = size(vol_frac, 1)
    max_abs = 0.0
    count = 0
    for j in 1:N, i in 1:N
        if vol_frac[i, j] >= 1.0 - 1.0e-12
            max_abs = max(max_abs, abs(field[i, j]))
            count += 1
        end
    end
    count > 0 || throw(ErrorException("cut-cell smoke found no fully fluid cells"))
    return max_abs, count
end

function incns_gdl_cut_cell_smoke(; N::Integer=32, backend=CPU())
    face_frac_x, face_frac_y, vol_frac = KrakeNN.tilted_half_plane_fractions(N)
    is_solid, west, east, south, north =
        incns_gdl_cell_fractions_from_staggered(face_frac_x, face_frac_y, vol_frac)
    h = 1.0 / Float64(N)
    bc = incns_gdl_wall_bc()

    ux = fill(1.25, N, N)
    uy = fill(-0.5, N, N)
    divu = zeros(Float64, N, N)
    gpx = zeros(Float64, N, N)
    gpy = zeros(Float64, N, N)
    lap_ux = zeros(Float64, N, N)
    lap_uy = zeros(Float64, N, N)
    p = [sin(2.0 * pi * incns_gdl_x(i, h)) * cos(2.0 * pi * incns_gdl_x(j, h))
         for i in 1:N, j in 1:N]

    gdl_divergence_embedded_2d!(
        divu, ux, uy, is_solid, west, east, south, north, vol_frac, h, h, bc...; backend,
    )
    gdl_pressure_gradient_embedded_2d!(
        gpx, gpy, p, is_solid, west, east, south, north, vol_frac, h, h, bc...; backend,
    )
    gdl_laplacian_apply_embedded_2d!(
        lap_ux, lap_uy, ux, uy, is_solid, west, east, south, north, h, h, bc...; backend,
    )

    max_const_div, full_fluid_cells = incns_gdl_full_fluid_max_abs(divu, vol_frac)
    @test max_const_div < 1.0e-12
    @test all(isfinite, divu)
    @test all(isfinite, gpx)
    @test all(isfinite, gpy)
    @test all(isfinite, lap_ux)
    @test all(isfinite, lap_uy)
    return (; N, max_const_div, full_fluid_cells)
end

function incns_gdl_static_gpu_readiness()
    source = read(INCNS_GDL_OPERATOR_PATH, String)
    has_kernels = all(
        name -> occursin("@kernel function $(name)", source),
        (
            "gdl_divergence_2d_kernel!",
            "gdl_pressure_gradient_2d_kernel!",
            "gdl_laplacian_apply_2d_kernel!",
        ),
    )
    ka_only = occursin("using KernelAbstractions", source) && !occursin("using KrakeNN", source)
    no_cuda_dep = !occursin("using CUDA", source) && !occursin("CUDA.", source)
    backend_launch = occursin("KernelAbstractions.get_backend", source) &&
                     occursin("ndrange=(Nx, Ny)", source)
    @test has_kernels
    @test ka_only
    @test no_cuda_dep
    @test backend_launch
    return (; has_kernels, ka_only, no_cuda_dep, backend_launch)
end

@testset "IncNS grad/div/laplacian FVFD operators" begin
    backend = CPU()

    divergence = incns_gdl_divergence_mms(; backend)
    INCNS_GDL_RESULTS[:divergence] = divergence

    gradient = incns_gdl_pressure_gradient_mms(; backend)
    INCNS_GDL_RESULTS[:gradient] = gradient

    laplacian = incns_gdl_laplacian_mms(; backend)
    INCNS_GDL_RESULTS[:laplacian] = laplacian

    duality = incns_gdl_duality_gold(; backend)
    INCNS_GDL_RESULTS[:duality] = duality

    cut_cell = incns_gdl_cut_cell_smoke(; backend)
    INCNS_GDL_RESULTS[:cut_cell] = cut_cell

    static_gpu = incns_gdl_static_gpu_readiness()
    INCNS_GDL_RESULTS[:static_gpu] = static_gpu
end
