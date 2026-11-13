# V0' — 3D manufactured solutions on the GPU (plan "solveur FV stationnaire
# 3D", I1): divergence-free velocity
#   u = sin kx (c cos ky − b cos kz), v = sin ky (a cos kz − c cos kx),
#   w = sin kz (b cos kx − a cos ky),   k = 2π, (a, b, c) = (1, 0.7, −0.4),
# p = ½ cos kx sin 2ky cos kz, ν = 1 + 0.4 sin k(x + 2y + 3z) (variable, so the
# transpose stress term is active), on [0, 1]³, N³ cells, Newton from the exact
# cell state. Four side sets (all walls; periodic x; periodic x and y; periodic
# x, y and z = no wall anywhere; there the momentum source is made zero-mean and
# the mean velocity (a free mode) is removed from the error; Stokes only, the
# Navier–Stokes Jacobian is singular there), Stokes and Navier–Stokes elsewhere. L2 (rms) and L∞ errors of velocity and pressure (pressure
# mean removed), observed orders between successive grids. Pressure diagnostic
# per case: index of the largest pressure error, its distance (in cells) to the
# nearest wall face (-1: no wall), and the share of the squared pressure error
# held by cells within 2 cells of a wall (0 without walls). A grid that runs out
# of GPU memory is reported as skipped (larger grids of that case are skipped
# too) and is not a failure.
#
# Gates (finest pair, L2 velocity): Stokes ≥ 1.8, NS ≥ 0.9. Pressure (Stokes):
# target 1.5 reported, gate ≥ 1.27 = the 2D MMS pressure order 1.47 (x-periodic
# Stokes, 20 → 40, test/analytical/incns_fv_steady_residual.jl) − 0.2.
# Energy rows of V0' belong to increment I3.
#
#   FV3D_SIDES=walls,periodic_xyz restricts the side sets (default: all four).
#   FV3D_BACKEND=cuda FV3D_NS=16,24,32,48 julia --project=<env> benchmarks/fv3d/v0p_mms.jl
#   FV3D_BACKEND=cpu  FV3D_NS=4,6        julia --project benchmarks/fv3d/v0p_mms.jl   (smoke)
if lowercase(get(ENV, "FV3D_BACKEND", "cuda")) == "cuda"
    using CUDA
    CUDA.functional() || error("CUDA not functional")
    using CUDSS
end
include(joinpath(@__DIR__, "fv3d_common.jl"))
const ForwardDiff = KrakeNN.ForwardDiff

const K = 2pi
const ABC = (1.0, 0.7, -0.4)
function uex(x)
    a, b, c = ABC
    sx, sy, sz = sin(K * x[1]), sin(K * x[2]), sin(K * x[3])
    cx, cy, cz = cos(K * x[1]), cos(K * x[2]), cos(K * x[3])
    return [sx * (c * cy - b * cz), sy * (a * cz - c * cx), sz * (b * cx - a * cy)]
end
pex(x) = 0.5 * cos(K * x[1]) * sin(2 * K * x[2]) * cos(K * x[3])
nuex(x) = 1 + 0.4 * sin(K * (x[1] + 2 * x[2] + 3 * x[3]))
function mms_source(x, conv)
    G(y) = ForwardDiff.jacobian(uex, y)
    σ(y) = (Gy = G(y); vec(nuex(y) .* (Gy .+ Gy')))
    Dσ = ForwardDiff.jacobian(σ, x)
    divσ = ntuple(r -> sum(Dσ[r + 3 * (c - 1), c] for c in 1:3), 3)
    gp = ForwardDiff.gradient(pex, x)
    cv = conv ? G(x) * uex(x) : zeros(3)
    return (cv[1] + gp[1] - divσ[1], cv[2] + gp[2] - divσ[2], cv[3] + gp[3] - divσ[3], 0.0)
end

const SIDES = (("walls", (W = :wall, E = :wall, S = :wall, N = :wall, B = :wall, T = :wall)),
               ("periodic_x", (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall)),
               ("periodic_xy", (W = :periodic, E = :periodic, S = :periodic, N = :periodic, B = :wall, T = :wall)),
               ("periodic_xyz", (W = :periodic, E = :periodic, S = :periodic, N = :periodic, B = :periodic, T = :periodic)))
const SIDE_FILTER = split(get(ENV, "FV3D_SIDES", ""), ','; keepempty = false)

# pressure-error location: (index, distance in cells to the nearest wall face, wall-layer share)
function perr_location(e, bc, N)
    I = Tuple(argmax(abs.(e)))
    function dwall(idx)
        d = Inf
        for a in 1:3
            bc[(:W, :S, :B)[a]] === :wall && (d = min(d, idx[a] - 1))
            bc[(:E, :N, :T)[a]] === :wall && (d = min(d, N - idx[a]))
        end
        return d
    end
    tot = sum(abs2, e); wall = 0.0
    for c in CartesianIndices(e)
        dwall(Tuple(c)) <= 1 && (wall += e[c]^2)
    end
    return (; loc = I, dwall = (d = dwall(I); isinf(d) ? -1.0 : d), wallshare = tot > 0 ? wall / tot : 0.0)
end

function mms_case(N, bc, conv)
    p = fv_steady_problem(N, N, N; nu = (x, y, z) -> nuex([x, y, z]), bc, conv,
                          boundary_velocity = (s, x, y, z) -> Tuple(uex([x, y, z])),
                          source = (x, y, z) -> mms_source([x, y, z], conv))
    allper = all(==(:periodic), values(bc))
    if allper   # no wall: momentum source must have zero mean (discrete compatibility)
        g = p.grid; src = copy(p.src)
        for c in 1:3
            idx = [KrakeNN.fv_index(g, i, j, k, c) for k in 1:N, j in 1:N, i in 1:N]
            src[idx] .-= sum(src[idx]) / N^3
        end
        p = KrakeNN.FVSteadyProblem3D(p.grid, p.tag, p.nu, p.bvel, src, p.visc, p.immersed, p.energy, p.flowrate)
    end
    Ue = fv_state(p, ((x, y, z) -> uex([x, y, z])[c] for c in 1:3)..., (x, y, z) -> pex([x, y, z]))
    r = fv3d_solve(p; U0 = Ue, atol = 1e-11, rtol = 1e-13, maxit = 20)
    r.stats.converged || println("  not converged: ‖R‖₂ history ", round.(r.stats.history[max(1, end - 5):end]; sigdigits = 3),
                                 "  stagnated=", r.stats.stagnated)
    f = fv_fields(p, r.U); fe = fv_fields(p, Ue)
    # all-periodic: the mean velocity is a free mode, compare the fluctuation only
    dm(c) = allper ? sum(f[c] .- fe[c]) / N^3 : 0.0
    du = sqrt.(sum((f[c] .- fe[c] .- dm(c)) .^ 2 for c in 1:3))
    pp = f[4] .- sum(f[4]) / N^3; pe = fe[4] .- sum(fe[4]) / N^3
    loc = perr_location(pp - pe, bc, N)
    return (; N, loc, eu2 = sqrt(sum(abs2, du) / N^3), euinf = maximum(du),
            ep2 = sqrt(sum(abs2, pp - pe) / N^3), epinf = maximum(abs, pp - pe),
            converged = r.stats.converged, iterations = r.stats.iterations, seconds = r.seconds)
end

fv3d_banner("v0p_mms")
Ns = fv3d_ints("FV3D_NS", FV3D_GPU ? "16,24,32,48" : "4,6")
results = Dict{String,Any}("backend" => FV3D_BACKEND, "Ns" => Ns,
                           "gates" => Dict("stokes_u" => 1.8, "ns_u" => 0.9, "stokes_p_floor" => 1.27,
                                           "stokes_p_target" => 1.5))
failures = String[]
skipped = String[]
for (sname, bc) in SIDES, conv in (false, true)
    isempty(SIDE_FILTER) || sname in SIDE_FILTER || continue
    if conv && all(==(:periodic), values(bc))
        # no wall and no mean-flow row: the mean velocity is a free mode, the Navier–Stokes
        # Jacobian is singular (3 left and right null vectors) and Newton stalls; Stokes
        # still solves (consistent right-hand side)
        println("SKIPPED ns_", sname, ": singular Jacobian (free mean velocity)"); continue
    end
    name = (conv ? "ns_" : "stokes_") * sname
    rows = Any[]
    for N in Ns
        c = try
            mms_case(N, bc, conv)
        catch err
            fv3d_free()
            if fv3d_isoom(err)
                println("SKIPPED ", name, " N=", N, " and larger: out of GPU memory"); flush(stdout)
                push!(skipped, "$name N=$N")
            else
                println("ERROR ", name, " N=", N, ": ", sprint(showerror, err)); flush(stdout)
                push!(failures, "$name N=$N error")
            end
            nothing
        end
        c === nothing && break
        fv3d_free()
        push!(rows, c)
        @printf("%-20s N=%3d  eu2=%.3e euinf=%.3e ep2=%.3e epinf=%.3e  it=%d conv=%s  %.1f s  pmax at %s, wall distance %s, wall-layer share %.2f\n",
                name, N, c.eu2, c.euinf, c.ep2, c.epinf, c.iterations, c.converged, c.seconds,
                string(c.loc.loc), c.loc.dwall < 0 ? "no wall" : string(Int(c.loc.dwall)) * " cells", c.loc.wallshare)
        flush(stdout)
        c.converged || push!(failures, "$name N=$N not converged")
    end
    length(rows) >= 2 || continue
    ord(f) = [fv3d_order(f(rows[k]), f(rows[k + 1]), rows[k].N, rows[k + 1].N) for k in 1:length(rows) - 1]
    ou2, ouinf, op2, opinf = ord(r -> r.eu2), ord(r -> r.euinf), ord(r -> r.ep2), ord(r -> r.epinf)
    gate_u = conv ? 0.9 : 1.8
    ok_u = ou2[end] >= gate_u
    ok_p = conv || op2[end] >= 1.27
    ok_u || push!(failures, @sprintf("%s velocity order %.2f < %.2f", name, ou2[end], gate_u))
    ok_p || push!(failures, @sprintf("%s pressure order %.2f < 1.27", name, op2[end]))
    @printf("SUMMARY %-20s order u L2 %s  Linf %s | p L2 %s  Linf %s | %s\n", name,
            join((@sprintf("%.2f", o) for o in ou2), ","), join((@sprintf("%.2f", o) for o in ouinf), ","),
            join((@sprintf("%.2f", o) for o in op2), ","), join((@sprintf("%.2f", o) for o in opinf), ","),
            ok_u && ok_p ? "ok" : "BELOW GATE")
    results[name] = Dict("N" => [r.N for r in rows], "eu2" => [r.eu2 for r in rows],
                         "euinf" => [r.euinf for r in rows], "ep2" => [r.ep2 for r in rows],
                         "epinf" => [r.epinf for r in rows], "iterations" => [r.iterations for r in rows],
                         "seconds" => [r.seconds for r in rows], "order_u_l2" => ou2,
                         "order_u_linf" => ouinf, "order_p_l2" => op2, "order_p_linf" => opinf,
                         "pass" => ok_u && ok_p,
                         "pmax_dist_wall_cells" => [r.loc.dwall for r in rows],
                         "pmax_wall_share" => [r.loc.wallshare for r in rows],
                         "pmax_index" => [collect(r.loc.loc) for r in rows])
end
results["failures"] = failures
results["skipped_oom"] = skipped
fv3d_write(fv3d_out("v0p_mms"), results)
exit(fv3d_verdict("v0p_mms", failures))
