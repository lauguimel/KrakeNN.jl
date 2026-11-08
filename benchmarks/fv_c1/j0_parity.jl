# Parity of the C1 steady FV residual (src/methods/inc_ns/fv_steady_residual.jl)
# with the J0 prototype (benchmarks/fv_newton_j0/FVJ0.jl on branch
# feat/fv-newton-j0), on random states, for every boundary configuration the
# prototype supports. The prototype is the independent reference: it was
# written and validated (D-RC study, dense-Jacobian checks) before C1.
#
#   FVJ0_PATH=<path to FVJ0.jl> julia --project benchmarks/fv_c1/j0_parity.jl
#
# The prototype's two sparsity packages (SparseConnectivityTracer,
# SparseMatrixColorings) are not KrakeNN dependencies; their `using` lines are
# stripped before loading, which is safe because the residual never calls them.

using KrakeNN, Random, LinearAlgebra, Printf

path = get(ENV, "FVJ0_PATH", "")
isfile(path) || error("set FVJ0_PATH to benchmarks/fv_newton_j0/FVJ0.jl of feat/fv-newton-j0")
src = read(path, String)
src = replace(src, "using SparseConnectivityTracer\n" => "", "using SparseMatrixColorings\n" => "")
Base.include_string(Main, src, path)
using .FVJ0

const BCSYM = Dict(FVJ0.BC_WALL => :wall, FVJ0.BC_INLET => :inlet,
                   FVJ0.BC_OUTLET => :outlet, FVJ0.BC_PERIODIC => :periodic)

function c1_problem(cs, solid)
    bv(s, x, y) = if BCSYM[cs.bcW] === :inlet && s === :W
        (cs.uin * (1 - abs(2 * y / (cs.Ny * cs.h) - 1)^cs.inexp), 0.0)
    elseif s === :N
        (cs.ulid, 0.0)
    else
        (0.0, 0.0)
    end
    nu = cs.model.nu
    p = fv_steady_problem(cs.Nx, cs.Ny; h = cs.h, nu = nu, conv = cs.conv != 0,
                          bc = (W = BCSYM[cs.bcW], E = BCSYM[cs.bcE], S = BCSYM[cs.bcS],
                                N = BCSYM[cs.bcN]),
                          boundary_velocity = bv, solid = solid,
                          source = (x, y) -> (cs.fx, cs.fy, 0.0))
    return p
end

function compare(name, cs, solid)
    p = c1_problem(cs, solid)
    pins_c1 = findall(==(FV_CELL_PINNED), vec(p.tag))
    pins_j0 = cs.pin == 0 ? Int[] : [cs.pin]
    worst = 0.0
    rng = MersenneTwister(7)
    for trial in 1:5
        U = randn(rng, fv_nunknowns(p))
        Rj = FVJ0.residual!(similar(U), U, solid, cs)
        Rc = fv_residual(U, p)
        # continuity rows of cells pinned by only one of the two codes differ
        # by construction (C1 pins every closed pocket, J0 one given cell)
        mask = trues(length(U))
        for c in symdiff(pins_c1, pins_j0)
            mask[3 * (c - 1) + 3] = false
        end
        worst = max(worst, norm((Rj .- Rc)[mask], Inf) / norm(Rj[mask], Inf))
    end
    @printf("%-40s pins C1 %-10s J0 %-6s  max rel |R_C1 - R_J0|_inf = %.2e\n",
            name, string(pins_c1), string(pins_j0), worst)
    return worst
end

worst = 0.0
for N in (8, 13)
    z = zeros(UInt8, N, N)
    global worst = max(worst, compare("cavity NS Re=100 $(N)²", cavity_case(N; Re = 100.0), z))
    global worst = max(worst, compare("cavity Stokes $(N)²", cavity_case(N; conv = 0.0), z))
end
for (Nx, Ny) in ((12, 6), (9, 7))
    z = zeros(UInt8, Nx, Ny)
    global worst = max(worst, compare("channel in/out Stokes $(Nx)x$(Ny)", channel_case(Nx, Ny), z))
    global worst = max(worst, compare("channel in/out NS $(Nx)x$(Ny)",
                                      channel_case(Nx, Ny; conv = 1.0, model = Newt(0.05)), z))
    global worst = max(worst, compare("periodic channel NS fx $(Nx)x$(Ny)",
                                      periodic_channel_case(Nx, Ny; model = Newt(0.1)), z))
end
m = manifold_mask(40, 20)
global worst = max(worst, compare("manifold mask NS 40x20",
                                  channel_case(40, 20; conv = 1.0, model = Newt(0.02)), m))
println("WORST = ", worst)
