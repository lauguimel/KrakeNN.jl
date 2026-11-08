# Analytical validation of IncNS momentum advection order.
#
# Manufactured constant-flux advection of a smooth cell-centred field:
#   div(F phi) = F dphi/dx,  F = 1.
# The error is measured away from the boundary fallback bands so the test
# isolates the interior donor-cell vs linear-upwind reconstruction.

using Test

if !isdefined(@__MODULE__, :solve_incns_manifold)
    include(joinpath(@__DIR__, "..", "..", "src", "methods", "inc_ns",
                     "manifold_flow.jl"))
end

const INCNS_MOMENTUM_ADVECTION_ORDER_RESULTS = Dict{Symbol,Any}()

function _incns_order_symbol(name::Symbol)
    if isdefined(@__MODULE__, name)
        return getfield(@__MODULE__, name)
    end
    return getfield(KrakeNN, name)
end

function momentum_advection_order_case(nx::Integer;
                                       ny::Integer = 4,
                                       momentum_advection::Symbol = :linear_upwind)
    nx = Int(nx); ny = Int(ny)
    Lx = 1.0
    Ly = 1.0
    dx = Lx / nx
    dy = Ly / ny
    xcenters = [(i - 0.5) * dx for i in 1:nx]
    exact(x) = 1.0 + 0.2 * sin(2.0 * pi * x)
    dexact(x) = 0.4 * pi * cos(2.0 * pi * x)

    u = [exact(xcenters[i]) for i in 1:nx, j in 1:ny]
    v = zeros(Float64, nx, ny)
    uf = fill(1.0, nx, ny)
    vf = zeros(Float64, nx, ny)
    uwest = fill(1.0, ny)
    solid = falses(nx, ny)
    conv_u = zeros(Float64, nx, ny)
    conv_v = zeros(Float64, nx, ny)

    boundary_spec = _incns_order_symbol(:_mf_boundary_spec)
    convection! = _incns_order_symbol(:_mf_convection!)
    bc = boundary_spec(ny, (; side=:west, j0=1, j1=ny, u=1.0),
                       (; side=:east, j0=1, j1=ny))
    convection!(conv_u, conv_v, u, v, uf, vf, uwest, dx, dy, nx, ny, solid, bc,
                momentum_advection)

    i0 = max(4, round(Int, 0.2 * nx))
    i1 = min(nx - 3, round(Int, 0.8 * nx))
    err2 = 0.0
    nerr = 0
    @inbounds for j in 1:ny, i in i0:i1
        err2 += (conv_u[i, j] - dexact(xcenters[i]))^2
        nerr += 1
    end
    l2 = sqrt(err2 / nerr)
    return (; nx, ny, l2)
end

function observed_momentum_advection_order(momentum_advection::Symbol)
    coarse = momentum_advection_order_case(64; momentum_advection)
    medium = momentum_advection_order_case(128; momentum_advection)
    fine = momentum_advection_order_case(256; momentum_advection)
    order_64_128 = log2(coarse.l2 / medium.l2)
    order_128_256 = log2(medium.l2 / fine.l2)
    return (; coarse, medium, fine, order_64_128, order_128_256)
end

@testset "IncNS implicit upwind operator consistency" begin
    nx = 7
    ny = 5
    dx = 0.2
    dy = 0.3
    solid = falses(nx, ny)
    solid[4, 3] = true
    u = [0.1 + 0.03 * i - 0.02 * j + 0.01 * sin(i + j) for i in 1:nx, j in 1:ny]
    v = [-0.05 + 0.01 * i + 0.04 * j + 0.02 * cos(2i - j) for i in 1:nx, j in 1:ny]
    uf = [0.08 * sin(0.7 * i - 0.2 * j) for i in 1:nx, j in 1:ny]
    vf = [0.06 * cos(0.4 * i + 0.5 * j) for i in 1:nx, j in 1:ny]
    uwest = [0.05 * (-1)^j for j in 1:ny]
    conv_u = zeros(Float64, nx, ny)
    conv_v = zeros(Float64, nx, ny)

    boundary_spec = _incns_order_symbol(:_mf_boundary_spec)
    upwind! = _incns_order_symbol(:_mf_convection_upwind!)
    assemble = _incns_order_symbol(:_mf_assemble_upwind_convection_operator)
    bc = boundary_spec(ny, (; side=:west, j0=2, j1=4, u=0.25),
                       (; side=:east, j0=1, j1=ny))

    upwind!(conv_u, conv_v, u, v, uf, vf, uwest, dx, dy, nx, ny, solid, bc)
    adv = assemble(nx, ny, dx, dy, solid, bc, uf, vf, uwest)

    @test reshape(adv.Cu * vec(u), nx, ny) .+ adv.src_u ≈ conv_u
    @test reshape(adv.Cv * vec(v), nx, ny) ≈ conv_v
end

@testset "IncNS momentum advection order" begin
    up = observed_momentum_advection_order(:upwind)
    lu = observed_momentum_advection_order(:linear_upwind)
    INCNS_MOMENTUM_ADVECTION_ORDER_RESULTS[:upwind] = up
    INCNS_MOMENTUM_ADVECTION_ORDER_RESULTS[:linear_upwind] = lu

    @test 0.85 <= up.order_128_256 <= 1.15
    @test lu.order_64_128 >= 1.8
    @test lu.order_128_256 >= 1.8
    @test lu.fine.l2 < up.fine.l2
end

if (abspath(PROGRAM_FILE) == (@__FILE__)) || (get(ENV, "INCNS_REPORT", "0") == "1")
    up = get(INCNS_MOMENTUM_ADVECTION_ORDER_RESULTS, :upwind, nothing)
    lu = get(INCNS_MOMENTUM_ADVECTION_ORDER_RESULTS, :linear_upwind, nothing)
    if up !== nothing && lu !== nothing
        @info "incns momentum advection order" upwind_64_128=up.order_64_128 upwind_128_256=up.order_128_256 linear_64_128=lu.order_64_128 linear_128_256=lu.order_128_256
    end
end
