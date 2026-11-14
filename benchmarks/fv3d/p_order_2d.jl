# 2D analogue of v0p_mms.jl: 2D MMS of test/analytical/fv_steady_helpers.jl,
# same side sets, Stokes, CPU. Also splits the pressure error into the wall
# band (cells within 2 of a wall side) and the rest.
#   FV3D_BACKEND=cpu FV3D_NS=16,32,64,128 sh tmp/julia.sh --project=. benchmarks/fv3d/p_order_2d.jl
ENV["FV3D_BACKEND"] = "cpu"
include(joinpath(@__DIR__, "fv3d_common.jl"))
include(joinpath(@__DIR__, "..", "..", "test", "analytical", "fv_steady_helpers.jl"))
const SIDES2 = (("walls", (W = :wall, E = :wall, S = :wall, N = :wall)),
                ("periodic_x", (W = :periodic, E = :periodic, S = :wall, N = :wall)),
                ("periodic_xy", (W = :periodic, E = :periodic, S = :periodic, N = :periodic)))
function case2(N, bc)
    p = fv_steady_problem(N, N; nu = (x, y) -> _fvc1_nuex([x, y]), bc, conv = false,
                          boundary_velocity = (s, x, y) -> Tuple(_fvc1_uex([x, y])),
                          source = (x, y) -> _fvc1_source([x, y], false))
    Ue = fv_state(p, (x, y) -> _fvc1_uex([x, y])[1], (x, y) -> _fvc1_uex([x, y])[2], (x, y) -> _fvc1_pex([x, y]))
    r = fv3d_solve(p; U0 = Ue, atol = 1e-11, rtol = 1e-13, maxit = 20)
    u, v, pp = fv_fields(p, r.U); ue, ve, pe = fv_fields(p, Ue)
    du = sqrt.((u .- ue) .^ 2 .+ (v .- ve) .^ 2)
    d = (pp .- sum(pp) / N^2) .- (pe .- sum(pe) / N^2)
    band = falses(N, N)
    bc.S === :wall && (band[:, 1:2] .= true; band[:, N-1:N] .= true)
    bc.W === :wall && (band[1:2, :] .= true; band[N-1:N, :] .= true)
    rms(a) = isempty(a) ? NaN : sqrt(sum(abs2, a) / N^2)
    return (; N, eu2 = rms(du), euinf = maximum(du), ep2 = rms(d), epinf = maximum(abs, d),
            epband = rms(d[band]), epin = rms(d[.!band]), it = r.stats.iterations)
end
Ns = fv3d_ints("FV3D_NS", "16,32,64,128")
for (name, bc) in SIDES2
    rows = [case2(N, bc) for N in Ns]
    for (k, c) in enumerate(rows)
        o(f) = k == 1 ? "  -  " : @sprintf("%.2f", fv3d_order(f(rows[k-1]), f(c), rows[k-1].N, c.N))
        @printf("%-12s N=%3d eu2=%.2e(%s) euinf=%.2e(%s) ep2=%.2e(%s) epinf=%.2e(%s) | p band=%.2e(%s) interior=%.2e(%s) it=%d\n",
                name, c.N, c.eu2, o(r -> r.eu2), c.euinf, o(r -> r.euinf), c.ep2, o(r -> r.ep2),
                c.epinf, o(r -> r.epinf), c.epband, o(r -> r.epband), c.epin, o(r -> r.epin), c.it)
        flush(stdout)
    end
end
