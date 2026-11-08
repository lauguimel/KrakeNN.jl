# Analytical validation: decoupled steady scalar advection–diffusion
# ("thermal transport"). Mirrors test/analytical/incns_poiseuille.jl.
#
# Three rungs:
#   1. Pure conduction (u=v=0): one wall :flux q, opposite wall :dirichlet T0,
#      sides :outflow -> analytic LINEAR profile, recovered to machine precision
#      (the discrete operator is exact for linear fields).
#   2. Second-order convergence of the conduction L2 error across two grids
#      (assert order in [1.7, 2.3]).
#   3. Constant-flux parallel-plate channel, thermally-developed Nusselt number:
#      frozen Poiseuille parabola u(y) as uf (vf=0), both plates :flux q, inlet
#      :dirichlet 0, outlet :outflow. Assert |Nu - 8.235|/8.235 <= 0.02.

using Test

if !isdefined(@__MODULE__, :solve_scalar_transport)
    include(joinpath(@__DIR__, "..", "..", "src", "methods",
                     "scalar_transport", "thermal_transport.jl"))
end

const SCALAR_TRANSPORT_RESULTS = Dict{Symbol,Any}()

# ---------------------------------------------------------------------------
# Rung 1+2: pure conduction across the channel height (y).
#
# Domain height H in y, walls at y=0 (south, :flux q) and y=H (north,
# :dirichlet T0). Sides (west/east) :outflow (zero-gradient -> 1D in y).
# With u=v=0 the steady balance is -DT T'' = 0 -> T linear in y.
# Conductive flux q = DT * dT/dy (heat injected at the south wall flows up to the
# fixed-T north wall). The analytic profile that satisfies T(H)=T0 and the south
# flux is  T(y) = T0 + (q/DT)*(H - y).
# ---------------------------------------------------------------------------
function conduction_case(; nx::Integer = 4, ny::Integer = 64,
                         H::Real = 1.0, DT::Real = 1.0,
                         q::Real = 1.0, T0::Real = 0.0)
    dx = H / nx          # square-ish cells; x is homogeneous here
    dy = H / ny
    uf = zeros(Float64, nx, ny)
    vf = zeros(Float64, nx, ny)
    bc = (west  = (kind = :outflow,   value = 0.0),
          east  = (kind = :outflow,   value = 0.0),
          south = (kind = :flux,      value = q),
          north = (kind = :dirichlet, value = T0))
    res = solve_scalar_transport(; nx, ny, dx, dy, uf, vf, DT, bc)

    # Analytic linear profile T(y) = T0 + (q/DT)*(H - y).
    Tan = [T0 + (q / DT) * (H - y) for y in res.ycenters]
    # x-averaged numerical profile (must be x-independent for this 1D problem).
    Tprof = vec(sum(res.T; dims = 1)) ./ nx

    l2_abs = sqrt(sum(abs2, Tprof .- Tan) / ny)
    linf = maximum(abs.(Tprof .- Tan))
    Tspan = maximum(Tan) - minimum(Tan)
    l2_rel = l2_abs / max(Tspan, eps())

    return (; res, Tprof, Tan, l2_abs, l2_rel, linf, Tspan)
end

# ---------------------------------------------------------------------------
# Manufactured pure-conduction case for the SECOND-ORDER convergence rung.
#
# The linear conduction profile is recovered EXACTLY by the discrete operator
# (machine precision on every grid), so it carries no measurable truncation
# error and no convergence order. To exercise the diffusion operator's formal
# second-order accuracy we manufacture a SMOOTH NON-LINEAR conduction field
#   T_exact(y) = cos(π y / H)
# satisfying  -DT T'' = S(y)  with the volumetric source
#   S(y) = DT (π/H)^2 cos(π y / H).
# Pure conduction (u=v=0). Both plates carry the Dirichlet value of T_exact at
# the wall; sides :outflow (1D in y). The 5-point Laplacian then converges at
# O(h^2) in the L2 norm.
# ---------------------------------------------------------------------------
function manufactured_conduction_case(; nx::Integer = 4, ny::Integer = 32,
                                      H::Real = 1.0, DT::Real = 1.0)
    dx = H / nx
    dy = H / ny
    ycenters = [(j - 0.5) * dy for j in 1:ny]
    Texact(y) = cos(pi * y / H)

    uf = zeros(Float64, nx, ny)
    vf = zeros(Float64, nx, ny)

    # Volumetric source S = DT*(π/H)^2 * cos(π y / H), cell-centred.
    src = zeros(Float64, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        src[i, j] = DT * (pi / H)^2 * cos(pi * ycenters[j] / H)
    end

    # Dirichlet wall values = T_exact AT the plate faces (y=0 and y=H).
    Tw_south = Texact(0.0)
    Tw_north = Texact(H)
    bc = (west  = (kind = :outflow,   value = 0.0),
          east  = (kind = :outflow,   value = 0.0),
          south = (kind = :dirichlet, value = Tw_south),
          north = (kind = :dirichlet, value = Tw_north))

    res = solve_scalar_transport(; nx, ny, dx, dy, uf, vf, DT, bc, source = src)

    Tan = [Texact(y) for y in res.ycenters]
    Tprof = vec(sum(res.T; dims = 1)) ./ nx
    l2_abs = sqrt(sum(abs2, Tprof .- Tan) / ny)
    return (; res, Tprof, Tan, l2_abs)
end

# ---------------------------------------------------------------------------
# Rung 3: constant-wall-flux parallel-plate channel, developed Nu = 8.235.
#
# Channel height H in y (plates at y=0 and y=H), length Lx in x. Frozen
# Poiseuille parabola u(y) = u_max*4*(y/H)*(1-y/H) as the streamwise face
# velocity uf (vf=0). Inlet (west) :dirichlet 0, outlet (east) :outflow, both
# plates :flux q (heated). Far downstream the flow is thermally developed:
# constant Nu based on the hydraulic diameter D_h = 2H,
#   Nu = q * D_h / (k * (T_wall - T_bulk)),  with k = DT here (rho*cp=1).
# Analytic developed value for both walls constant flux: Nu = 8.235.
#
# uf[i,j] = east face of cell (i,j); for a uniform streamwise velocity profile
# every x-face in a row carries the same u(y_j). We set uf in EVERY column
# (including the last, used as the outlet face velocity) to the row parabola.
# ---------------------------------------------------------------------------
function channel_nusselt_case(; nx::Integer = 400, ny::Integer = 60,
                              H::Real = 1.0, Lx::Real = 40.0,
                              u_max::Real = 1.0, DT::Real = 0.1,
                              q::Real = 1.0,
                              advection::Symbol = :linear_upwind,
                              deferred_passes::Integer = 4)
    dx = Lx / nx
    dy = H / ny
    ycenters = [(j - 0.5) * dy for j in 1:ny]

    # Frozen Poiseuille parabola on the cell-centred rows, applied to all x-faces.
    uprof = [u_max * 4.0 * (y / H) * (1.0 - y / H) for y in ycenters]
    uf = zeros(Float64, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        uf[i, j] = uprof[j]
    end
    vf = zeros(Float64, nx, ny)

    bc = (west  = (kind = :dirichlet, value = 0.0),   # inlet T=0
          east  = (kind = :outflow,   value = 0.0),   # outlet zero-gradient
          south = (kind = :flux,      value = q),      # heated plate
          north = (kind = :flux,      value = q))      # heated plate

    res = solve_scalar_transport(; nx, ny, dx, dy, uf, vf, DT, bc,
                                 advection, deferred_passes)
    T = res.T

    # Bulk (mixing-cup) temperature per streamwise station:
    #   T_bulk(x) = ∫ u T dy / ∫ u dy.
    # Wall temperature: average of the two near-wall cell extrapolations to the
    # plate (cell centre + half the local gradient toward the wall, but for a
    # constant-flux wall the wall value is cell + q*(dy/2)/DT from the flux BC's
    # zero-gradient-plus-source discretisation; we read the wall temperature
    # consistently from the cell value plus the flux jump over the half cell).
    udy = sum(uprof) * dy
    Nu_x = zeros(Float64, nx)
    Twall_x = zeros(Float64, nx)
    Tbulk_x = zeros(Float64, nx)
    for i in 1:nx
        col = @view T[i, :]
        Tbulk = (sum(uprof .* col) * dy) / udy
        # Wall temperature from the constant-flux condition: the conductive flux
        # q = DT*(T_wall - T_firstcell)/(dy/2) at each plate -> T_wall = T_cell +
        # q*(dy/2)/DT. Average the two plates.
        Tw_s = col[1]  + q * (dy / 2) / DT
        Tw_n = col[ny] + q * (dy / 2) / DT
        Twall = 0.5 * (Tw_s + Tw_n)
        Dh = 2.0 * H
        Nu_x[i] = q * Dh / (DT * (Twall - Tbulk))
        Twall_x[i] = Twall
        Tbulk_x[i] = Tbulk
    end

    # Developed value: average over the thermally-developed window, downstream of
    # the thermal entry length and upstream of the outlet (the zero-gradient
    # outflow contaminates the last few percent of cells). 50%–95% of the channel
    # sits squarely in the developed region for the grid/Pe used here.
    i0 = max(1, round(Int, 0.50 * nx))
    i1 = min(nx, round(Int, 0.95 * nx))
    Nu_dev = sum(@view Nu_x[i0:i1]) / (i1 - i0 + 1)

    return (; res, Nu_x, Nu_dev, Twall_x, Tbulk_x, dx, dy, nx, ny, Lx, H, DT, q)
end

# ---------------------------------------------------------------------------
# Localized same-side inlet/outlet energy balance.
#
# A frozen U-turn velocity field enters through the lower west segment and exits
# through the upper west segment. East/north/south are impermeable. The scalar
# source is volumetric, so the integrated source must leave as outflow enthalpy
# through the localized west outlet when that segment reads the boundary face
# velocity instead of assuming Fw=0.
# ---------------------------------------------------------------------------
function segment_outlet_energy_case(; nx::Integer = 48, ny::Integer = 32,
                                    Lx::Real = 4.0, Ly::Real = 2.0,
                                    U::Real = 1.0, DT::Real = 1e-4,
                                    S::Real = 1.0,
                                    advection::Symbol = :linear_upwind,
                                    deferred_passes::Integer = 4)
    dx = Lx / nx
    dy = Ly / ny
    split = ny ÷ 2

    phi(y) = y^2 * (Ly - y)^2
    dphi(y) = 2.0 * y * (Ly - y) * (Ly - 2.0 * y)
    f(x) = (1.0 - x / Lx)^2
    df(x) = -2.0 * (1.0 - x / Lx) / Lx

    ycenters = [(j - 0.5) * dy for j in 1:ny]
    xfaces = [(i - 1) * Lx / (nx - 1) for i in 1:nx]
    scale = U / maximum(abs, dphi.(ycenters))

    uf = zeros(Float64, nx, ny)
    vf = zeros(Float64, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        uf[i, j] = scale * f(xfaces[i]) * dphi(ycenters[j])
        yface = (j - 1) * Ly / (ny - 1)
        xcenter = (i - 0.5) * dx
        vf[i, j] = -scale * df(xcenter) * phi(yface)
    end
    vf[:, 1] .= 0.0
    vf[:, ny] .= 0.0

    source = zeros(Float64, nx, ny)
    source[nx ÷ 3:nx, :] .= S

    bc = (west = [(lo = 1, hi = split, kind = :dirichlet, value = 0.0),
                  (lo = split + 1, hi = ny, kind = :outflow, value = 0.0)],
          east = (kind = :flux, value = 0.0),
          south = (kind = :flux, value = 0.0),
          north = (kind = :flux, value = 0.0))

    res = solve_scalar_transport(; nx, ny, dx, dy, uf, vf, DT, bc, source,
                                 advection, deferred_passes)
    solid = falses(nx, ny)
    Q_in = sum(source) * dx * dy
    # Dual-mode: standalone include defines the helper in this module; under
    # runtests the include auto-skips and the helper lives in KrakeNN.
    westflux = isdefined(@__MODULE__, :_st_west_boundary_flux) ?
        _st_west_boundary_flux : KrakeNN._st_west_boundary_flux
    enthalpy_out = sum(max(-westflux(uf, vf, solid, nx, ny,
                                     dx, dy, j), 0.0) *
                       res.T[1, j] * dy for j in split + 1:ny)
    rel = abs(Q_in - enthalpy_out) / Q_in
    return (; res, Q_in, enthalpy_out, rel, split)
end

# ---------------------------------------------------------------------------
# Solid exclusion: a full-height solid strip separates two conductive regions.
# Source is applied only on the left. With fluid-solid faces removed, the right
# side remains exactly at its east Dirichlet value and solid rows stay pinned.
# ---------------------------------------------------------------------------
function solid_barrier_case(; nx::Integer = 30, ny::Integer = 16,
                            Lx::Real = 3.0, Ly::Real = 1.6,
                            DT::Real = 0.5, S::Real = 1.0)
    dx = Lx / nx
    dy = Ly / ny
    uf = zeros(Float64, nx, ny)
    vf = zeros(Float64, nx, ny)
    solid = falses(nx, ny)
    ib = nx ÷ 2 + 1
    solid[ib, :] .= true

    source = zeros(Float64, nx, ny)
    source[1:ib - 1, :] .= S

    bc = (west = (kind = :dirichlet, value = 0.0),
          east = (kind = :dirichlet, value = 0.0),
          south = (kind = :flux, value = 0.0),
          north = (kind = :flux, value = 0.0))
    res = solve_scalar_transport(; nx, ny, dx, dy, uf, vf, DT,
                                 is_solid = solid, bc, source)

    Lleft = (ib - 1) * dx
    wall_extrapolated_max = S * Lleft^2 / (2.0 * DT)
    right_max = maximum(abs, @view res.T[ib + 1:nx, :])
    solid_max = maximum(abs, res.T[solid])
    fluid_max = maximum(res.T[.!solid])
    return (; res, solid, ib, right_max, solid_max, fluid_max,
            wall_extrapolated_max)
end

@testset "Scalar transport: heated channel" begin

    @testset "pure conduction (machine precision)" begin
        c = conduction_case()
        SCALAR_TRANSPORT_RESULTS[:conduction] = c
        @test c.res.converged
        @test c.res.iters == 1
        # Residual ‖A·T − b‖ at machine noise.
        @test c.res.residual_history[1] <= 1e-8
        # Linear field recovered to machine precision (operator is exact).
        @test c.l2_rel <= 1e-10
        @test c.linf <= 1e-10
    end

    @testset "second-order convergence" begin
        # The LINEAR conduction profile is exact on every grid (machine noise),
        # so it carries no measurable order. We measure the diffusion operator's
        # formal O(h^2) on a SMOOTH NON-LINEAR manufactured conduction field
        # (T = cos(πy/H), matching volumetric source) across two grids.
        coarse = manufactured_conduction_case(; ny = 32)
        fine = manufactured_conduction_case(; ny = 64)
        SCALAR_TRANSPORT_RESULTS[:order_coarse] = coarse
        SCALAR_TRANSPORT_RESULTS[:order_fine] = fine
        order = log2(coarse.l2_abs / fine.l2_abs)
        SCALAR_TRANSPORT_RESULTS[:order] = order
        @test 1.7 <= order <= 2.3
    end

    @testset "developed Nusselt (constant flux)" begin
        c = channel_nusselt_case()
        SCALAR_TRANSPORT_RESULTS[:nusselt] = c
        @test c.res.converged
        rel = abs(c.Nu_dev - 8.235) / 8.235
        SCALAR_TRANSPORT_RESULTS[:nu_rel] = rel
        @test rel <= 0.02
    end

    @testset "localized same-side outlet energy balance" begin
        c = segment_outlet_energy_case()
        SCALAR_TRANSPORT_RESULTS[:segment_energy] = c
        @test c.res.converged
        @test c.rel <= 0.02
    end

    @testset "solid block exclusion" begin
        c = solid_barrier_case()
        SCALAR_TRANSPORT_RESULTS[:solid_barrier] = c
        @test c.res.converged
        @test c.solid_max == 0.0
        @test c.right_max <= 1e-10
        @test c.fluid_max <= c.wall_extrapolated_max * (1.0 + 1e-10)
    end

    @testset "legacy upwind parity fingerprints" begin
        c = channel_nusselt_case(; nx = 80, ny = 20, Lx = 8.0,
                                 H = 1.0, u_max = 1.3, DT = 0.07,
                                 q = 0.8, advection = :upwind)
        @test c.res.advection === :upwind
        @test c.res.iters == 1
        @test sum(c.res.T) ≈ 12431.075011137844 rtol = 1e-14
        @test maximum(c.res.T) ≈ 16.834358480047527 rtol = 1e-14
        @test c.Nu_dev ≈ 8.240863030731946 rtol = 1e-14
        # Residual-type quantities are near-cancellations: bound them instead of
        # pinning (FP reassociation, e.g. Pkg.test --check-bounds, shifts them).
        @test c.res.residual_history[1] <= 1e-11

        s = segment_outlet_energy_case(; nx = 24, ny = 16, Lx = 4.0,
                                       Ly = 2.0, U = 0.8, DT = 2e-4,
                                       S = 0.7, advection = :upwind)
        @test s.res.advection === :upwind
        @test s.res.iters == 1
        @test sum(s.res.T) ≈ 17071.81594621023 rtol = 1e-14
        @test maximum(s.res.T) ≈ 751.6923570728646 rtol = 1e-14
        @test s.enthalpy_out ≈ 3.9666666663171792 rtol = 1e-14
        @test s.rel <= 1e-10
    end
end

# Report (printed when run as a standalone script).
if (abspath(PROGRAM_FILE) == (@__FILE__)) || (get(ENV, "ST_REPORT", "0") == "1")
    cd = get(SCALAR_TRANSPORT_RESULTS, :conduction, nothing)
    od = get(SCALAR_TRANSPORT_RESULTS, :order, nothing)
    nu = get(SCALAR_TRANSPORT_RESULTS, :nusselt, nothing)
    nr = get(SCALAR_TRANSPORT_RESULTS, :nu_rel, nothing)
    eb = get(SCALAR_TRANSPORT_RESULTS, :segment_energy, nothing)
    sb = get(SCALAR_TRANSPORT_RESULTS, :solid_barrier, nothing)
    if cd !== nothing
        @info "conduction" l2_rel=cd.l2_rel linf=cd.linf residual=cd.res.residual_history[1] Pe=cd.res.Pe_cell
    end
    if od !== nothing
        @info "convergence order" order=od
    end
    if nu !== nothing
        @info "developed Nusselt" Nu_dev=nu.Nu_dev rel=nr nx=nu.nx ny=nu.ny Lx=nu.Lx Pe=nu.res.Pe_cell
    end
    if eb !== nothing
        @info "segment energy balance" Q_in=eb.Q_in enthalpy_out=eb.enthalpy_out rel=eb.rel
    end
    if sb !== nothing
        @info "solid barrier" solid_max=sb.solid_max right_max=sb.right_max fluid_max=sb.fluid_max bound=sb.wall_extrapolated_max
    end
end
