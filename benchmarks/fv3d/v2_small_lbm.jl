# V2-small — 3D steady FV against the 3D LBM drivers (plan "solveur FV
# stationnaire 3D", I1). Gate: relative L2 velocity deviation ≤ 3 % per case.
#
# (1) Newtonian lid-driven cavity vs `run_cavity_3d` (D3Q19 BGK, halfway
#     bounce-back on the five static walls, Zou–He lid on the top node layer
#     k = Nz). The LBM domain is therefore x, y ∈ [½, N + ½], z ∈ [½, N] in
#     lattice units (height N − ½). The FV grid with h = ½ lattice unit,
#     2N × 2N × (2N − 1) cells, covers it exactly; LBM node (i, j, k) sits on
#     the corner shared by FV cells 2i−1:2i × 2j−1:2j × 2k−1:2k and is
#     compared with their mean (k = 1 … N − 1; the lid layer is excluded).
#     u_lid = FV3D_LID (0.1), Re = u_lid N / ν = FV3D_RE. The LBM runs
#     FV3D_LBM_STEPS steps; a second run at 80 % of them gives the reported
#     steadiness (relative change between the two).
# (2) GNF (Carreau–Yasuda) plane channel extruded in x and y vs
#     `run_gnf_channel_3d` (flow x, walls z, halfway walls q_w = ½, so the LBM
#     nodes are the FV cell centres with h = 1), same law and body force as
#     test/analytical/gnf3d_reference.jl (:carreau). FV grid 4 × 4 × N,
#     periodic x and y. Both are also compared with the analytic profile.
#
# Memory: the FV cavity grid is 2N × 2N × (2N − 1) cells (h = ½), i.e. (2N)³ cells,
# not N³; the direct factorisation grows like cells^(4/3) (about 25 GB at 48³, more
# than 80 GB at 64³). N = 32 means a 64³ FV grid and runs out of memory on an 80 GB
# GPU; the default N = 24 is a 48³ FV grid (about 25 GB). The LBM arrays are small.
#
#   FV3D_BACKEND=cuda FV3D_LBM_N=24 FV3D_RE=100 FV3D_GNF_N=32 \
#       julia --project=<env> benchmarks/fv3d/v2_small_lbm.jl
#   FV3D_BACKEND=cpu  (defaults N=5, Re=10, GNF N=8: smoke, gate not meaningful)
if lowercase(get(ENV, "FV3D_BACKEND", "cuda")) == "cuda"
    using CUDA
    CUDA.functional() || error("CUDA not functional")
    using CUDSS
end
include(joinpath(@__DIR__, "fv3d_common.jl"))
using KrakeNN.KernelAbstractions
include(joinpath(@__DIR__, "..", "..", "test", "analytical", "gnf3d_reference.jl"))

lbm_backend() = FV3D_GPU ? CUDA.CUDABackend() : KernelAbstractions.CPU()
rel_l2(a, b) = sqrt(sum(abs2, a .- b)) / sqrt(sum(abs2, b))

function cavity_case(N, Re, ulid, steps)
    nu = ulid * N / Re
    lbm(nsteps) = run_cavity_3d(LBMConfig(D3Q19(); Nx = N, Ny = N, Nz = N, ν = nu, u_lid = ulid,
                                          max_steps = nsteps); backend = lbm_backend())
    t_lbm = @elapsed L = lbm(steps)
    L0 = lbm(round(Int, 0.8 * steps))
    lid = (s, x, y, z) -> s === :T ? (ulid, 0.0, 0.0) : (0.0, 0.0, 0.0)
    p = fv_steady_problem(2N, 2N, 2N - 1; h = 0.5, nu, boundary_velocity = lid)
    r = fv3d_solve(p; maxit = 60, dtau0 = 1.0, dtau_max = 1e4)
    u, v, w, _ = fv_fields(p, r.U)
    m(f, i, j, k) = sum(f[a, b, c] for a in 2i-1:2i, b in 2j-1:2j, c in 2k-1:2k) / 8
    K = 1:N-1
    fv = [m(f, i, j, k) for f in (u, v, w), i in 1:N, j in 1:N, k in K]
    lb = [f[i, j, k] for f in (L.ux, L.uy, L.uz), i in 1:N, j in 1:N, k in K]
    lb0 = [f[i, j, k] for f in (L0.ux, L0.uy, L0.uz), i in 1:N, j in 1:N, k in K]
    stats = (; converged = r.stats.converged, iterations = r.stats.iterations, t_fv = r.seconds)
    r = nothing; L = nothing; L0 = nothing; p = nothing; fv3d_free()
    return (; dev = rel_l2(lb, fv), steadiness = rel_l2(lb, lb0), converged = stats.converged,
            iterations = stats.iterations, t_fv = stats.t_fv, t_lbm, nu)
end

function gnf_case(N, steps)
    ref = _gnf3d_reference(:carreau, N, 0.5)
    t_lbm = @elapsed L = KrakeNN.run_gnf_channel_3d(; rheology = ref.rheology, N, q_w = 0.5, F = ref.F,
                                             nu_init = ref.nu, u_init = ref.u, max_steps = steps,
                                             tol_ss = 1e-9, check_every = 200, backend = lbm_backend(),
                                             FT = Float64)
    p = fv_steady_problem(4, 4, N; h = 1.0, nu = ref.rheology.eta_0, rheology = ref.rheology,
                          bc = (W = :periodic, E = :periodic, S = :periodic, N = :periodic, B = :wall, T = :wall),
                          source = (x, y, z) -> (ref.F, 0.0, 0.0, 0.0))
    U0 = fv_state(p, [ref.u[k] for i in 1:4, j in 1:4, k in 1:N], zeros(4, 4, N), zeros(4, 4, N),
                  zeros(4, 4, N))
    r = fv3d_solve(p; U0, maxit = 40)
    u, _, _, _ = fv_fields(p, r.U)
    prof = [sum(u[:, :, k]) / 16 for k in 1:N]
    return (; dev = rel_l2(L.u, prof), fv_vs_exact = rel_l2(prof, ref.u), lbm_vs_exact = rel_l2(L.u, ref.u),
            lbm_converged = L.converged, lbm_steps = L.steps, converged = r.stats.converged,
            iterations = r.stats.iterations, t_fv = r.seconds, t_lbm)
end

fv3d_banner("v2_small_lbm")
N = parse(Int, get(ENV, "FV3D_LBM_N", FV3D_GPU ? "24" : "5"))
Re = parse(Float64, get(ENV, "FV3D_RE", FV3D_GPU ? "100" : "10"))
ulid = parse(Float64, get(ENV, "FV3D_LID", "0.1"))
steps = parse(Int, get(ENV, "FV3D_LBM_STEPS", FV3D_GPU ? "60000" : "1500"))
GN = parse(Int, get(ENV, "FV3D_GNF_N", FV3D_GPU ? "32" : "8"))
gsteps = parse(Int, get(ENV, "FV3D_GNF_STEPS", FV3D_GPU ? "120000" : "20000"))
failures = String[]
results = Dict{String,Any}("backend" => FV3D_BACKEND, "gates" => Dict("l2_dev" => 0.03))

c = try
    cavity_case(N, Re, ulid, steps)
catch err
    fv3d_free()
    fv3d_isoom(err) || rethrow()
    println("SKIPPED cavity N=", N, ": out of GPU memory (FV grid ", 2N, "^3)")
    push!(failures, "cavity out of GPU memory at N=$N (FV grid $(2N)^3): lower FV3D_LBM_N")
    nothing
end
if c !== nothing
@printf("SUMMARY cavity LBM N=%d (FV %dx%dx%d, h=0.5) Re=%g nu=%.4g  dev L2=%.4f  LBM steadiness=%.2e  FV it=%d conv=%s  FV %.1f s  LBM %.1f s\n",
        N, 2N, 2N, 2N - 1, Re, c.nu, c.dev, c.steadiness, c.iterations, c.converged, c.t_fv, c.t_lbm)
c.converged || push!(failures, "cavity FV not converged")
c.dev <= 0.03 || push!(failures, @sprintf("cavity deviation %.4f > 0.03", c.dev))
results["cavity"] = Dict("N_lbm" => N, "Re" => Re, "u_lid" => ulid, "lbm_steps" => steps,
                         "dev_l2" => c.dev, "lbm_steadiness" => c.steadiness, "fv_iterations" => c.iterations,
                         "fv_seconds" => c.t_fv, "lbm_seconds" => c.t_lbm, "pass" => c.converged && c.dev <= 0.03)
end
flush(stdout)

gcase = gnf_case(GN, gsteps)
@printf("SUMMARY gnf channel N=%d  dev L2=%.4f  FV-exact=%.2e  LBM-exact=%.2e  LBM conv=%s (%d steps)  FV it=%d conv=%s  FV %.1f s  LBM %.1f s\n",
        GN, gcase.dev, gcase.fv_vs_exact, gcase.lbm_vs_exact, gcase.lbm_converged, gcase.lbm_steps,
        gcase.iterations, gcase.converged, gcase.t_fv, gcase.t_lbm)
gcase.converged || push!(failures, "gnf FV not converged")
gcase.dev <= 0.03 || push!(failures, @sprintf("gnf deviation %.4f > 0.03", gcase.dev))
results["gnf_channel"] = Dict("N" => GN, "lbm_steps" => gcase.lbm_steps, "lbm_converged" => gcase.lbm_converged,
                              "dev_l2" => gcase.dev, "fv_vs_exact" => gcase.fv_vs_exact,
                              "lbm_vs_exact" => gcase.lbm_vs_exact, "fv_iterations" => gcase.iterations,
                              "pass" => gcase.converged && gcase.dev <= 0.03)
results["failures"] = failures
fv3d_write(fv3d_out("v2_small_lbm"), results)
exit(fv3d_verdict("v2_small_lbm", failures))
