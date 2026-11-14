# I2 measurement of the 3D steady FV discrete adjoint on the GPU (plan
# "solveur FV stationnaire 3D", step 4).
#
# (0) Parity on a small grid (FV3D_CHECK_N, default 8 on cuda, 6 on cpu):
#     periodic-x duct with an off-centre obstacle, x source and a moving top
#     wall, plus a flow-rate-controlled law duct (TinyMLP law with a prescribed
#     field). The state is solved once on the host and copied to the device so
#     both backends differentiate the same U. For every map (uniform ν, ν field,
#     side velocity, x source, law parameters) the gradient of the composite
#     PIV objective (three planes with thickness / window / mask, flow-rate or
#     Fx term, wall-force term) on the device must equal the host gradient
#     within 1e-12 relative (∞-norm). Device bytes (CUDA.@allocated, after a
#     warm-up, with a prebuilt workspace and objective): fv_objective_gradient!
#     and fv_steady_gradient with an all-linear PIV objective must allocate at
#     most LINEAR_BYTES_MAX = 1024 bytes (a few small host-visible scalars, not
#     a field: the bound is the same at every grid size); with the wall-force
#     term (host duals) the bytes are reported only.
# (1) Cost, for N in FV3D_NS (default 32 on cuda, 8 on cpu): N × N/2 × N/2
#     duct with obstacle, from rest; median time per Newton iteration and the
#     gradient time (finalize = true: refactorisation at U included; finalize =
#     false: factors reused), split into objective / transposed solve /
#     parameter cotangent; ratio gradient / Newton iteration. Gated on Newton
#     convergence and, at the largest N, on the device bytes of the all-linear
#     path: <= LINEAR_BYTES_MAX and EQUAL to the bytes measured at FV3D_CHECK_N
#     (uniform ν and ν field maps of the duct); a difference means an
#     allocation that grows with the grid.
#
#   FV3D_BACKEND=cuda FV3D_CHECK_N=8 FV3D_NS=32,48 julia --project=<env> benchmarks/fv3d/i2_adjoint_measure.jl
#   FV3D_BACKEND=cpu  FV3D_CHECK_N=6 FV3D_NS=8     julia --project benchmarks/fv3d/i2_adjoint_measure.jl   (smoke)
if lowercase(get(ENV, "FV3D_BACKEND", "cuda")) == "cuda"
    using CUDA
    CUDA.functional() || error("CUDA not functional")
    using CUDSS
end
include(joinpath(@__DIR__, "fv3d_common.jl"))
using Statistics
using Random

const LINEAR_BYTES_MAX = 1024
# device bytes of the all-linear path measured by `parity` (key => (objective, gradient))
const LIN_BYTES = Dict{String,Tuple{Int,Int}}()

const BC = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall)
top(s, x, y, z) = s === :T ? (0.2, 0.1, 0.0) : (0.0, 0.0, 0.0)

function obstacle(Nx, Ny, Nz)
    solid = falses(Nx, Ny, Nz)
    i0 = max(2, Nx ÷ 2); j1 = max(2, Ny ÷ 2); k0 = max(1, Nz ÷ 2)
    solid[i0, 2:j1, k0] .= true
    solid[i0 + 1, 2, k0] = true
    return solid
end

function duct(Nx, Ny, Nz; law = false)
    h = 1.0 / Ny
    solid = obstacle(Nx, Ny, Nz)
    if law
        field = [0.2 * i / Nx + 0.1 * j / Ny - 0.15 * k / Nz for i in 1:Nx, j in 1:Ny, k in 1:Nz]
        nn = LearnedViscosity(TinyMLP{2,2,1}(); nu_ref = 0.05, monotone = false)
        θ = law_init(nn; rng = MersenneTwister(33)); θ[3:4] .= (0.3, -0.2)
        p = fv_steady_problem(Nx, Ny, Nz; h, nu = 0.05, bc = BC, solid, u_mean_target = 0.4,
                              viscosity = FVLawViscosity(nn, θ, (field,); eps = 0.02))
        return p, θ
    end
    p = fv_steady_problem(Nx, Ny, Nz; h, nu = (x, y, z) -> 0.05 * (1 + 0.3x - 0.2y + 0.1z), bc = BC,
                          solid, source = (x, y, z) -> (1.0, 0.0, 0.0, 0.0), boundary_velocity = top)
    return p, nothing
end

# three planes inside the box [0, Nx h] × [0, Ny h] × [0, Nz h]
function planes(p)
    g = p.grid; h = g.h; Lx, Ly, Lz = g.Nx * h, g.Ny * h, g.Nz * h
    r(L, n) = range(0.1L, 0.9L; length = n)
    return (PIVPlane(; normal = :z, position = 0.43Lz, xs = r(Lx, 5), ys = r(Ly, 4),
                     window = (1.0h, 0.75h), thickness = 0.4h, stereo = true),
            PIVPlane(; normal = :x, position = 0.53Lx, xs = r(Ly, 5), ys = r(Lz, 4)),
            PIVPlane(; normal = :y, position = 0.41Ly, xs = r(Lx, 4), ys = r(Lz, 3),
                     window = (0.5h, 0.0), thickness = 0.25h))
end

function piv_setup(p, U)
    g = p.grid; h = g.h
    tmpl = zeros(g.Nx, g.Ny, g.Nz)
    ops = map(pl -> piv_operator(pl, tmpl; h = (h, h, h)), planes(p))
    u, v, w, _ = fv_fields(p, U)
    pert(f) = 1.05 .* f .+ 0.02 .* [sin(1.3i + 0.7j - 0.4k) for i in 1:g.Nx, j in 1:g.Ny, k in 1:g.Nz]
    data = Any[piv_apply(op, (pert(u), pert(v), pert(w))) for op in ops]
    m1 = ones(size(data[1][1])); m1[2, 3] = 0
    data[1] = map(d -> (d = copy(d); d[m1 .== 0] .= NaN; d), data[1])
    return ops, Tuple(data), (m1, nothing, nothing)
end

function objectives(p, dp, U, linear_term)
    ops, data, mask = piv_setup(p, U)
    to = fv3d_to()
    g = p.grid
    dops = map(pl -> piv_operator(pl, to(zeros(g.Nx, g.Ny, g.Nz)); h = (g.h, g.h, g.h)), planes(p))
    wall = (FVWallForce(1; region = :solid), 0.5, 0.01)
    mk(q, o, t) = FVPIVObjective(q, o, data; mask, sigma_u = 0.05, terms = t)
    return (; host_full = mk(p, ops, (linear_term, wall)), dev_full = mk(dp, dops, (linear_term, wall)),
            host_lin = mk(p, ops, (linear_term,)), dev_lin = mk(dp, dops, (linear_term,)))
end

function parity(N, failures)
    out = Dict{String,Any}()
    to = fv3d_to()
    for law in (false, true)
        p, θlaw = duct(N, N - 1, N - 2; law)
        sc = fv_newton_solver(p)
        U = law ? KrakeNN._fv_initial_force!(zeros(fv_nunknowns(p)), p) : zeros(fv_nunknowns(p))
        st = fv_newton!(sc, U; atol = 1e-12, rtol = 1e-14, maxit = 40)
        st.converged || st.stagnated || push!(failures, "parity host solve law=$law")
        dp = fv_adapt(to, p)
        sg = fv_newton_solver(dp; to, linsolve = fv3d_linsolve())
        dU = to(copy(U))
        term = law ? (fv_flow_force_objective(p), 4.0, 1.1 * fv_flow_force(p, U)) : (fv_flow_rate(p, 2), 0.3)
        objs = objectives(p, dp, U, term)
        maps = law ? (("law NN", FVLawParameters(length(θlaw)), θlaw), ("uniform nu", FVUniformViscosity(), [0.05])) :
               (("uniform nu", FVUniformViscosity(), [0.05]), ("nu field", FVViscosityField(), nothing),
                ("side velocity T", FVSideVelocity(p, :T; shape = (x, y, z) -> (1.0, 0.5, 0.0)), [0.2]),
                ("x source", FVUniformSource(1), [1.0]))
        for (mname, map, θ) in maps
            key = (law ? "law_" : "duct_") * replace(mname, " " => "_")
            row = Dict{String,Any}()
            try
                # uniform-ν map on the varying-ν duct: solve at the uniform value
                pm, dpm, Um, dUm, scm, sgm = p, dp, U, dU, sc, sg
                if map isa FVUniformViscosity && !law
                    pm = fv_param_problem(map, p, θ); dpm = fv_adapt(to, pm)
                    scm = fv_newton_solver(pm); Um = copy(U)
                    fv_newton!(scm, Um; atol = 1e-12, rtol = 1e-14, maxit = 40)
                    sgm = fv_newton_solver(dpm; to, linsolve = fv3d_linsolve()); dUm = to(copy(Um))
                end
                o = map isa FVUniformViscosity && !law ? objectives(pm, dpm, Um, term) : objs
                wsg = fv_adjoint_workspace(sgm, map)
                for (oname, oh, od) in (("full", o.host_full, o.dev_full), ("linear", o.host_lin, o.dev_lin))
                    rc = fv_steady_gradient(scm, Um, θ, oh; params = map)
                    rg = fv_steady_gradient(sgm, dUm, θ, od; params = map, ws = wsg)       # warm-up
                    rel = norm(rg.grad - rc.grad, Inf) / max(norm(rc.grad, Inf), eps())
                    relJ = abs(rg.J - rc.J) / max(abs(rc.J), eps())
                    bytes_obj = fv3d_devalloc(() -> fv_objective_gradient!(od, dUm, dpm))
                    bytes = fv3d_devalloc(() -> fv_steady_gradient(sgm, dUm, θ, od; params = map, ws = wsg))
                    ok = rel <= 1e-12 && relJ <= 1e-12
                    if oname == "linear"
                        ok &= bytes_obj <= LINEAR_BYTES_MAX && bytes <= LINEAR_BYTES_MAX
                        LIN_BYTES[key] = (bytes_obj, bytes)
                    end
                    @printf("SUMMARY parity %-6s %-16s %-6s grad rel %.2e  J rel %.2e  device bytes objective %d gradient %d  %s\n",
                            law ? "law" : "duct", mname, oname, rel, relJ, bytes_obj, bytes, ok ? "ok" : "FAIL")
                    flush(stdout)
                    ok || push!(failures, "parity $key $oname")
                    row[oname] = Dict("grad_rel" => rel, "J_rel" => relJ, "device_bytes_objective" => bytes_obj,
                                      "device_bytes_gradient" => bytes, "pass" => ok)
                end
            catch err
                println("ERROR parity ", key, ": ", first(sprint(showerror, err), 600)); flush(stdout)
                push!(failures, "parity $key error")
            end
            out[key] = row
        end
    end
    return out
end

function cost(N, failures; largest = false)
    p, _ = duct(N, max(4, N ÷ 2), max(4, N ÷ 2))
    to = fv3d_to()
    dp = fv_adapt(to, p)
    s = fv_newton_solver(dp; to, linsolve = fv3d_linsolve())
    U = to(zeros(fv_nunknowns(p)))
    t = @elapsed begin
        st = fv_newton!(s, U; atol = 1e-10, rtol = 1e-12, maxit = 60)
        fv3d_sync()
    end
    st.converged || push!(failures, "cost N=$N not converged")
    per_it = st.t_assembly .+ st.t_factor .+ st.t_solve .+ st.t_linesearch
    it_med = median(per_it[min(2, end):end])
    Uh = Vector{Float64}(Array(U))
    o = objectives(p, dp, Uh, (fv_flow_rate(p, 2), 0.3))
    row = Dict{String,Any}("N" => N, "unknowns" => fv_nunknowns(p), "iterations" => st.iterations,
                           "t_newton" => t, "t_iteration_median" => it_med)
    for (mname, map, θ) in (("uniform nu", FVUniformViscosity(), [0.05]), ("nu field", FVViscosityField(), nothing))
        ws = fv_adjoint_workspace(s, map)
        for fin in (true, false)
            fv_steady_gradient(s, U, θ, o.dev_lin; params = map, ws, finalize = fin)   # warm-up
            r = fv_steady_gradient(s, U, θ, o.dev_lin; params = map, ws, finalize = fin)
            fv3d_sync()
            k = replace(mname, " " => "_") * (fin ? "_finalize" : "_reuse")
            if largest && fin
                # all-linear device bytes at this size vs the parity size (same calls)
                bo = fv3d_devalloc(() -> fv_objective_gradient!(o.dev_lin, U, dp))
                bg = fv3d_devalloc(() -> fv_steady_gradient(s, U, θ, o.dev_lin; params = map, ws))
                ref = get(LIN_BYTES, "duct_" * replace(mname, " " => "_"), nothing)
                okb = bo <= LINEAR_BYTES_MAX && bg <= LINEAR_BYTES_MAX && ref !== nothing && ref == (bo, bg)
                @printf("SUMMARY bytes N=%3d %-10s objective %d gradient %d  (parity size: %s)  %s\n",
                        N, mname, bo, bg, ref === nothing ? "missing" : string(ref), okb ? "ok" : "FAIL")
                okb || push!(failures, "linear device bytes N=$N $mname $((bo, bg)) vs parity $(ref)")
                row[k * "_device_bytes"] = Dict("objective" => bo, "gradient" => bg,
                                                "parity_objective" => ref === nothing ? -2 : ref[1],
                                                "parity_gradient" => ref === nothing ? -2 : ref[2])
            end
            row[k] = Dict("t_total" => r.t_total, "t_objective" => r.t_objective, "t_finalize" => r.t_finalize,
                          "t_solve" => r.t_solve, "t_param" => r.t_param, "ratio_to_iteration" => r.t_total / it_med)
            @printf("SUMMARY cost N=%3d n=%d %-10s %-8s gradient %.3f s (objective %.4f, finalize %.3f, solve %.3f, param %.3f)  Newton iteration %.3f s  ratio %.2f\n",
                    N, fv_nunknowns(p), mname, fin ? "finalize" : "reuse", r.t_total, r.t_objective,
                    r.t_finalize, r.t_solve, r.t_param, it_med, r.t_total / it_med)
            flush(stdout)
        end
    end
    return row
end

fv3d_banner("i2_adjoint_measure")
failures = String[]
results = Dict{String,Any}("backend" => FV3D_BACKEND)
results["parity"] = parity(parse(Int, get(ENV, "FV3D_CHECK_N", FV3D_GPU ? "8" : "6")), failures)
const NS = fv3d_ints("FV3D_NS", FV3D_GPU ? "32" : "8")
for N in NS
    key = "cost_N" * string(N)
    try
        results[key] = cost(N, failures; largest = N == maximum(NS))
    catch err
        println("ERROR ", key, ": ", first(sprint(showerror, err), 600)); flush(stdout)
        push!(failures, "$key error")
    end
    fv3d_free()
end
results["failures"] = failures
fv3d_write(fv3d_out("i2_adjoint_measure"), results)
exit(fv3d_verdict("i2_adjoint_measure", failures))
