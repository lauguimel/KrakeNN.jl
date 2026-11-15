# Planar and 3D cases of the 3D FV conformation adjoint (plan "discrete
# viscoelastic adjoints", step F5). Giesekus (alpha = 0.3, nu_s = nu_p = 1,
# h = 1/N, imposed mean axial velocity 1, so De = lambda), De continuation
# 0.1, 0.25, ... up to FV3D_VE_DE (default 0.5) from rest. Parameters
# theta = (lambda, alpha) through FVRelaxationParameters(:lambda, :theta), step
# scales (lambda, 1). Objective as in the derivative gate (PIV plane + Fx + N1
# terms, data from the state at theta + 0.1 s).
#
# planar (F5.1): periodic array of cylinders between two walls, N x N x Nz cells
#   (x periodic, y walls, z periodic), one cylinder of solid cells (centre of the
#   cell, radius 0.25), Nz = 1 then Nz = 2. Objective: z-normal PIV plane at the
#   first cell centres (u, v), Fx, N1.
#   Gates: (a) Nz = 1 and Nz = 2 give the same state: max |f1 - f2| / max |f1|
#   <= 1e-12 for u, v, p (minus its mean over the fluid cells), c_xx, c_xy,
#   c_yy, c_zz on every layer of Nz = 2, and the body force; (b) w, c_xz, c_yz
#   <= 1e-13 (absolute) on both; (c) G1 and G2 on alpha and lambda, for each Nz.
#   If Nz = 1 fails (a), the Nz = 2 rows still run (plan: use Nz = 2 and say so).
#   (d) J(Nz = 1) vs J(Nz = 2) <= 1e-12 relative, gradient <= 1e-10 relative
#   (PIV weights are normalised by the sampled plane, not by the 3D field).
# duct (F5.2): square duct N^3 periodic in x, walls y and z, constriction of
#   solid cells (x in (3N/8, 5N/8], section outside the central square
#   (N/4, 3N/4]^2 solid: half the side open). Objective: x-normal stereo PIV
#   plane at the first cell centres, Fx, N1 (ve_objective of the gate).
#   Gates: G1 and G2 on alpha and lambda; every transposed GMRES solve
#   converged. Reported: adjoint time / Newton time (last continuation stage,
#   and per Newton iteration), adjoint with refactorisation (finalize = true)
#   and with the factors / preconditioner reused (finalize = false).
#
# Gates G1/G2 and their thresholds are those of the derivative gate
# (test/analytical/incns_fv3d_conformation_gate_test.jl, VE_G2_* of
# fv3d_conformation_gate_setup.jl): G1 |<g, d> - (<gU, dU> + <explicit, d>)|
# <= 1e-10 |<g, d>| + 1e-14 for each parameter and one random direction; G2
# min over h in VE_G2_STEPS of |FD - g_k| <= VE_G2_REL |g_k| + VE_G2_ABS |J| / s_k.
#
# Linear solvers per row (FV3D_VE_PLANAR_LINSOLVE, FV3D_VE_DUCT_LINSOLVE,
# comma-separated): direct (cuDSS on GPU, UMFPACK on CPU) or lsc (GMRES + block
# LSC, precond passed explicitly: precond = :mg, block = :upper, schur_approx =
# :lsc, mg_cycles = 4; the :auto default is not used and not changed).
# The lsc rows use rtol = rtol_transpose = FV3D_VE_KRYLOV_RTOL (default 1e-12)
# for the Newton steps, the tangent and the adjoint, so that G1 compares two
# solves accurate well below its 1e-10 threshold; maxiter 2000, restart 50.
# Defaults: planar direct; duct lsc,direct on GPU, direct on CPU.
#
#   FV3D_BACKEND=cuda julia --project=<env with CUDA, CUDSS> benchmarks/fv3d/ve_adjoint_cases.jl
#   FV3D_BACKEND=cpu FV3D_VE_PLANAR_N=8 FV3D_VE_DUCT_N=8 julia --project benchmarks/fv3d/ve_adjoint_cases.jl  (smoke)
# Options: FV3D_VE_CASES (planar,duct), FV3D_VE_PLANAR_N (32), FV3D_VE_DUCT_N
# (24), FV3D_VE_DE (0.5), FV3D_VE_KRYLOV_RTOL, FV3D_OUT.
if lowercase(get(ENV, "FV3D_BACKEND", "cuda")) == "cuda"
    using CUDA
    CUDA.functional() || error("CUDA not functional")
    using CUDSS
end
include(joinpath(@__DIR__, "fv3d_common.jl"))
include(joinpath(@__DIR__, "..", "..", "test", "analytical", "fv3d_conformation_gate_setup.jl"))

const SCRIPT = "ve_adjoint_cases"
const CASES = Symbol.(split(get(ENV, "FV3D_VE_CASES", "planar,duct"), ','; keepempty = false))
const PLANAR_N = parse(Int, get(ENV, "FV3D_VE_PLANAR_N", "32"))
const DUCT_N = parse(Int, get(ENV, "FV3D_VE_DUCT_N", "24"))
const DE = parse(Float64, get(ENV, "FV3D_VE_DE", "0.5"))
const RAMP = [filter(<(DE), [0.1, 0.25, 0.5, 0.75]); DE]
const KRTOL = parse(Float64, get(ENV, "FV3D_VE_KRYLOV_RTOL", "1e-12"))
const PLANAR_LS = split(get(ENV, "FV3D_VE_PLANAR_LINSOLVE", "direct"), ','; keepempty = false)
const DUCT_LS = split(get(ENV, "FV3D_VE_DUCT_LINSOLVE", FV3D_GPU ? "lsc,direct" : "direct"), ','; keepempty = false)

const PNAMES = [:lambda, :alpha]
pmap() = FVRelaxationParameters(:lambda, :theta)  # :theta = slot 2 of PolynomialRelaxation = alpha
theta0() = [DE, VE_ALPHA]
scales() = [DE, 1.0]
giesekus(lam, a = VE_ALPHA) = FVConformation3D(; lambda = lam, nu_p = 1.0, alpha = a)

lsc_tag() = KrylovBackendTag(; rtol = KRTOL, rtol_transpose = KRTOL, restart = 50, maxiter = 2000,
                              strict = false, precond = :mg, block = :upper, schur_approx = :lsc,
                              mg_cycles = 4)
linsolve_of(name) = name == "direct" ? fv3d_linsolve() : name == "lsc" ? lsc_tag() :
                    error("linear solver must be direct or lsc, got $name")

# ------------------------------------------------------------------ geometry
const PLANAR_BC = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :periodic, T = :periodic)
function planar_problem(N, Nz, De)
    h = 1 / N
    solid = falses(N, N, Nz)
    for k in 1:Nz, j in 1:N, i in 1:N
        x = (i - 0.5) * h; y = (j - 0.5) * h
        solid[i, j, k] = (x - 0.5)^2 + (y - 0.5)^2 <= 0.25^2
    end
    return fv_steady_problem(N, N, Nz; h, nu = 1.0, bc = PLANAR_BC, conformation = giesekus(De), solid,
                             source = (x, y, z) -> (50.0, 0.0, 0.0, 0.0), u_mean_target = 1.0)
end
function duct_problem(N, De)
    solid = falses(N, N, N)
    gap = (N ÷ 4 + 1):(3N ÷ 4)
    for k in 1:N, j in 1:N, i in (3N ÷ 8 + 1):(5N ÷ 8)
        solid[i, j, k] = !(j in gap && k in gap)
    end
    return fv_steady_problem(N, N, N; h = 1 / N, nu = 1.0, bc = VE_BC, conformation = giesekus(De), solid,
                             source = (x, y, z) -> (50.0, 0.0, 0.0, 0.0), u_mean_target = 1.0)
end
rest_state(p) = (z = zeros(p.grid.Nx, p.grid.Ny, p.grid.Nz); fv_state(p, z, z, z, z))

# continuation from rest; returns solver, device state and the last stage's
# Newton time and iteration count
function solve_state(p, linsolve)
    to = fv3d_to()
    s = fv_newton_solver(p; to, linsolve)
    U = to(rest_state(p))
    t_last = 0.0; it_last = 0
    for lam in RAMP
        fv_set_conformation!(s, giesekus(lam))
        t_last = @elapsed (st = ve_solve!(s, U); fv3d_sync())
        it_last = length(st.inf) - 1
    end
    fv_set_params!(s, pmap(), theta0())          # same coefficients, through the map
    ve_solve!(s, U)
    fv_check_conformation_domain(s.prob, U)
    return s, U, t_last, it_last
end

function data_state(s, U, linsolve)
    sd = fv_newton_solver(KrakeNN._fv_host_problem(s.prob); to = fv3d_to(), linsolve)
    fv_set_params!(sd, pmap(), theta0() .+ 0.1 .* scales())
    Ud = copy(U)
    ve_solve!(sd, Ud)
    fv_check_conformation_domain(sd.prob, Ud)
    Uh = Vector{Float64}(Array(Ud))
    fv3d_release_all!(sd)
    return Uh
end

# z-normal plane at the first cell centres: (u, v) at every (x, y) cell centre
function planar_objective(p, Ud::Vector{Float64}; to = fv3d_to())
    ph = KrakeNN._fv_host_problem(p)
    g = ph.grid
    u, v, w, _ = fv_fields(ph, Ud)
    xs = ((1:g.Nx) .- 0.5) .* g.h; ys = ((1:g.Ny) .- 0.5) .* g.h
    plane = PIVPlane(; normal = :z, position = 0.5 * g.h, xs, ys)
    op = piv_operator(plane, to(zeros(g.Nx, g.Ny, g.Nz)); h = (g.h, g.h, g.h))
    data = piv_apply(KrakeNN._fv_piv_host_op(op), (u, v, w))
    # weights from the sampled plane (not from the 3D field: a 3D sum doubles with
    # Nz = 2 and would halve the PIV term against Fx and N1)
    mask = (fill(1 / sum(abs2, data[1]), g.Nx, g.Ny), fill(1 / sum(abs2, data[2]), g.Nx, g.Ny))
    Fs = Ud[KrakeNN._fv_force_index(ph)]
    n1 = ve_n1_functional(ph)
    N1s = fv_objective(n1, Ud, ph)
    terms = ((fv_flow_force_objective(p), 1 / Fs^2, Fs), (n1, 1 / N1s^2, N1s))
    return FVPIVObjective(p, (op,), (data,); mask = (mask,), terms)
end

# ------------------------------------------------------------------ G1 / G2
function gate_g1g2(label, s, U, obj, linsolve)
    to = fv3d_to(); m = pmap(); θ = theta0(); sk = scales()
    fails = String[]
    ws = fv_adjoint_workspace(s, m; chunk = 4)
    res = fv_steady_gradient(s, U, θ, obj; params = m, ws)
    grad = copy(res.grad); gU = Vector{Float64}(Array(copy(res.gU))); explicit = copy(res.explicit); J = res.J
    out = Dict{String,Any}("J" => J, "grad" => grad, "residual_inf" => res.residual_inf)
    all(isfinite, grad) || push!(fails, "$label: non-finite gradient")
    for k in eachindex(θ)
        abs(grad[k]) > 1e-8 * J / sk[k] || push!(fails, "$label: zero sensitivity $(PNAMES[k])")
    end
    rd = randn(MersenneTwister(13), length(θ)); rd ./= norm(rd, Inf)
    for (d, name) in [[(Float64.(1:2 .== k), String(PNAMES[k])) for k in 1:2]; (rd, "random")]
        dU = fv_steady_tangent(s, U, θ, d; params = m, ws)
        lhs = dot(grad, d)
        e = abs(lhs - (dot(gU, Array(dU)) + dot(explicit, d)))
        tol = 1e-10 * abs(lhs) + 1e-14
        @printf("   %s G1 %-7s <g,d> %+.6e  err %.2e  tol %.2e  %s\n", label, name, lhs, e, tol, e <= tol ? "ok" : "FAIL")
        out["G1_$name"] = e / abs(lhs)
        e <= tol || push!(fails, @sprintf("%s: G1 %s %.2e > %.2e", label, name, e, tol))
    end
    sf = fv_newton_solver(KrakeNN._fv_host_problem(s.prob); to, linsolve)
    for k in 1:2
        errs = Float64[]
        for h in VE_G2_STEPS
            fd, _ = ve_fd(sf, m, θ, k, h, sk[k], U, obj)
            push!(errs, abs(fd - grad[k]))
        end
        tol = VE_G2_REL * abs(grad[k]) + VE_G2_ABS * J / sk[k]
        e = minimum(errs)
        @printf("   %s G2 %-7s g %+.6e  err %.2e (rel %.1e)  tol %.2e  %s\n", label, PNAMES[k], grad[k], e,
                e / abs(grad[k]), tol, e <= tol ? "ok" : "FAIL")
        out["G2_$(PNAMES[k])"] = e / abs(grad[k])
        e <= tol || push!(fails, @sprintf("%s: G2 %s %.2e > %.2e", label, PNAMES[k], e, tol))
    end
    fv3d_release_all!(sf)
    if linsolve isa KrylovBackendTag
        lg = lin_krylov_log(s.cache)
        nbad = count(e -> e.transpose && !e.converged, lg)
        out["transpose_unconverged"] = nbad
        out["transpose_iters_max"] = maximum((e.iters for e in lg if e.transpose); init = 0)
        nbad == 0 || push!(fails, "$label: $nbad transposed GMRES solve(s) not converged")
    end
    return out, fails, ws
end

# ------------------------------------------------------------------ planar
function layer_compare(p1, U1, p2, U2)
    ph1 = KrakeNN._fv_host_problem(p1); ph2 = KrakeNN._fv_host_problem(p2)
    f1 = fv_fields(ph1, U1); f2 = fv_fields(ph2, U2)
    c1 = fv_conformation_fields(ph1, U1); c2 = fv_conformation_fields(ph2, U2)
    fl1 = Array(ph1.tag) .!= KrakeNN.FV_CELL_SOLID; fl2 = Array(ph2.tag) .!= KrakeNN.FV_CELL_SOLID
    gauge(p, fl) = p .- sum(p[fl]) / count(fl)
    pairs = [("u", f1[1], f2[1]), ("v", f1[2], f2[2]), ("p", gauge(f1[4], fl1), gauge(f2[4], fl2)),
             ("c_xx", c1.xx, c2.xx), ("c_xy", c1.xy, c2.xy), ("c_yy", c1.yy, c2.yy), ("c_zz", c1.zz, c2.zz)]
    rel = Dict{String,Float64}()
    for (name, a, b) in pairs
        sc = max(maximum(abs, a), 1e-300)
        rel[name] = maximum(maximum(abs, a[:, :, 1] .- b[:, :, k]) for k in axes(b, 3)) / sc
    end
    iF1 = KrakeNN._fv_force_index(ph1); iF2 = KrakeNN._fv_force_index(ph2)
    rel["force"] = abs(U1[iF1] - U2[iF2]) / abs(U1[iF1])
    zs = Dict("w" => max(maximum(abs, f1[3]), maximum(abs, f2[3])),
                "c_xz" => max(maximum(abs, c1.xz), maximum(abs, c2.xz)),
                "c_yz" => max(maximum(abs, c1.yz), maximum(abs, c2.yz)))
    return rel, zs
end

function run_planar(lsname)
    linsolve = linsolve_of(lsname)
    out = Dict{String,Any}(); fails = String[]
    states = Dict{Int,Any}()
    for Nz in (1, 2)
        label = "planar $(PLANAR_N)x$(PLANAR_N)x$Nz $lsname"
        try
            p = planar_problem(PLANAR_N, Nz, RAMP[1])
            s, U, t_n, it_n = solve_state(p, linsolve)
            Ud = data_state(s, U, linsolve)
            obj = planar_objective(s.prob, Ud)
            r, f, _ = gate_g1g2(label, s, U, obj, linsolve)
            r["t_newton_last_stage"] = t_n; r["newton_iters_last_stage"] = it_n
            out["Nz$Nz"] = r; append!(fails, f)
            states[Nz] = (KrakeNN._fv_host_problem(s.prob), Vector{Float64}(Array(U)), r["grad"], r["J"])
            fv3d_release_all!(s)
        catch err
            msg = first(sprint(showerror, err), 300)
            println("   $label: ERROR ", msg)
            out["Nz$Nz"] = Dict{String,Any}("error" => msg)
            push!(fails, "$label: error")
        end
        flush(stdout)
    end
    if haskey(states, 1) && haskey(states, 2)
        (p1, U1, g1, J1), (p2, U2, g2, J2) = states[1], states[2]
        rel, zs = layer_compare(p1, U1, p2, U2)
        for (k, v) in sort(collect(rel))
            @printf("   planar %s Nz=1 vs Nz=2 %-5s rel %.2e  %s\n", lsname, k, v, v <= 1e-12 ? "ok" : "FAIL")
            v <= 1e-12 || push!(fails, @sprintf("planar %s: Nz=1 vs Nz=2 %s %.2e > 1e-12", lsname, k, v))
        end
        for (k, v) in sort(collect(zs))
            @printf("   planar %s max |%s| (both Nz) %.2e  %s\n", lsname, k, v, v <= 1e-13 ? "ok" : "FAIL")
            v <= 1e-13 || push!(fails, @sprintf("planar %s: |%s| %.2e > 1e-13", lsname, k, v))
        end
        dJ = abs(J1 - J2) / abs(J1)
        @printf("   planar %s J Nz=1 vs Nz=2: rel diff %.2e  tol 1e-12  %s\n", lsname, dJ, dJ <= 1e-12 ? "ok" : "FAIL")
        dJ <= 1e-12 || push!(fails, @sprintf("planar %s: J Nz=1 vs Nz=2 %.2e > 1e-12", lsname, dJ))
        dg = maximum(abs.(g1 .- g2) ./ abs.(g2))
        @printf("   planar %s gradient Nz=1 vs Nz=2: max rel diff %.2e  tol 1e-10  %s\n", lsname, dg, dg <= 1e-10 ? "ok" : "FAIL")
        dg <= 1e-10 || push!(fails, @sprintf("planar %s: gradient Nz=1 vs Nz=2 %.2e > 1e-10", lsname, dg))
        out["same_state"] = rel; out["zero_fields"] = zs; out["J_reldiff_nz"] = dJ; out["grad_reldiff_nz"] = dg
    end
    return out, fails
end

# ------------------------------------------------------------------ duct
function run_duct(lsname)
    linsolve = linsolve_of(lsname)
    label = "duct $(DUCT_N)^3 $lsname"
    out = Dict{String,Any}(); fails = String[]
    s = nothing
    try
        p = duct_problem(DUCT_N, RAMP[1])
        s, U, t_n, it_n = solve_state(p, linsolve)
        Ud = data_state(s, U, linsolve)
        obj, _ = ve_objective(s.prob, Ud; to = fv3d_to())
        r, f, ws = gate_g1g2(label, s, U, obj, linsolve)
        append!(fails, f)
        m = pmap(); θ = theta0()
        fv_steady_gradient(s, U, θ, obj; params = m, ws)                     # warm (compiled above)
        t_adj = @elapsed (fv_steady_gradient(s, U, θ, obj; params = m, ws); fv3d_sync())
        t_reuse = @elapsed (fv_steady_gradient(s, U, θ, obj; params = m, ws, finalize = false); fv3d_sync())
        per_it = t_n / max(it_n, 1)
        @printf("   %s Newton last stage (De %.2f): %d iterations, %.3f s (%.3f s / iteration)\n", label, DE, it_n, t_n, per_it)
        @printf("   %s adjoint (2 parameters): %.3f s with refactorisation = %.2f x last-stage Newton, %.2f x one Newton iteration; %.3f s with factors reused\n",
                label, t_adj, t_adj / t_n, t_adj / per_it, t_reuse)
        merge!(r, Dict{String,Any}("t_newton_last_stage" => t_n, "newton_iters_last_stage" => it_n,
                                   "t_adjoint" => t_adj, "t_adjoint_reuse" => t_reuse,
                                   "adjoint_over_newton" => t_adj / t_n, "adjoint_over_newton_iter" => t_adj / per_it,
                                   "device_used_bytes" => fv3d_devused()))
        out = r
    catch err
        msg = first(sprint(showerror, err), 300)
        println("   $label: ERROR ", msg)
        out["error"] = msg
        push!(fails, fv3d_isoom(err) ? "$label: out of device memory" : "$label: error")
    end
    fv3d_release_all!(s)
    flush(stdout)
    return out, fails
end

function main()
    fv3d_banner(SCRIPT)
    println("cases ", join(CASES, ","), "  De ramp ", join(RAMP, "->"), "  planar N ", PLANAR_N, " (",
            join(PLANAR_LS, ","), ")  duct N ", DUCT_N, " (", join(DUCT_LS, ","), ")  Krylov rtol ", KRTOL)
    data = Dict{String,Any}("De" => DE, "planar_N" => PLANAR_N, "duct_N" => DUCT_N, "krylov_rtol" => KRTOL)
    failures = String[]
    for c in CASES
        c in (:planar, :duct) || error("unknown case $c (planar, duct)")
        for ls in (c === :planar ? PLANAR_LS : DUCT_LS)
            out, f = c === :planar ? run_planar(ls) : run_duct(ls)
            data["$(c)_$ls"] = out
            append!(failures, f)
            fv3d_write(fv3d_out(SCRIPT), data)
        end
    end
    return fv3d_verdict(SCRIPT, failures)
end

exit(main())
