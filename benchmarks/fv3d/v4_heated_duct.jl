# V4 — 3D energy equation on the GPU (plan "solveur FV stationnaire 3D",
# increment I3, step 5).
#
# The plan's reference (the 3D thermal lattice-Boltzmann solver, θ within 3 %)
# is not available for this geometry: the 3D thermal LBM driver covers closed
# cavities only, without a periodic body-force duct or a volumetric heat source.
# Such a driver is estimated at half a day of development plus its GPU runs; it
# was not written. This script uses instead:
#
# (a) heated square duct vs a separated-variable reference. Section [0, 1]²
#     (y, z), periodic x of length 2 (Nx = 2N, Ny = Nz = N, h = 1/N), walls
#     S/N/B/T at T_w = 1, uniform body force G = 1, ν = 0.05 (u_max ≈ 1.47, the
#     flow is unidirectional), no viscous heating (σ = 0), α = 0.05, heat source
#     q = q0 + q1 cos kx (k = π, q0 = q1 = 1). The exact temperature is
#       T = T_w + θ0(y, z) + Re[Θ1(y, z) e^(ikx)],
#       −α ∇⊥²θ0 = q0,   α (∇⊥²Θ1 − k² Θ1) − i k u(y, z) Θ1 = −q1,
#     with θ0 = Θ1 = 0 on the walls and u the Fourier series of v1_duct_cavity.jl.
#     θ0 and Θ1 are solved on a fine section of M = r N cells per side (r odd,
#     M ≥ 300, so the coarse cell centres are fine cell centres) with the
#     cell-centred 5-point Laplacian and a complex sparse LU (CPU).
#     Gate: relative L2 and L∞ errors of θ = T − T_w ≤ 3 % on the finest grid;
#     orders and the velocity error are reported.
#
# (b) extrusion on the device (V0a at solver level): a 2D x-periodic channel
#     with a square obstacle, walls S (T = 1) and N (T = 1.5), body force,
#     thermal Carreau–Yasuda law with an Arrhenius shift (a_T = exp(2(1/T − 1))),
#     viscous heating σ = 0.3 and source 0.2; the 2D problem and its Nz = 3
#     z-periodic extrusion are solved on the backend. Gates: (b1) with pins
#     cleared, at a perturbed copy of the 2D solution, the device 3D residual
#     of the extruded state equals h times the 2D residual to 1e-14 of each
#     equation's scale (w row ≤ 1e-14 of the momentum scale); (b2) the 3D
#     solve equals the extruded 2D solve to 1e-7 (relative max, T measured
#     from the S wall temperature), w ≤ 1e-7 max|u|.
#
# Per solve: Newton iterations, stopping gate, setup and Newton time, device
# memory in use after the solve (memory pool included).
#
# Linear solver of the Newton steps of (a) (fv3d_common.jl): FV3D_LINSOLVE=direct
# (default, cuDSS) or krylov (branch K: GMRES + P3L, the V5 configuration, for
# ducts whose factorisation does not fit, N = 48 on 80 GB). With krylov, per
# grid: GMRES iterations of every Newton step and their maximum, any
# unconverged GMRES solve (a failure), and for N <= FV3D_DIRECT_MAX (24) a
# second solve with direct: relative max difference of the state ≤ 1e-6 (a
# failure otherwise). (b) always uses the direct solver: its size is set by
# FV3D_EXT_N, not FV3D_NS, and its 1e-7 solve gate compares two direct solves.
#
#   FV3D_BACKEND=cuda FV3D_NS=16,24    FV3D_EXT_N=32 julia --project=<env> benchmarks/fv3d/v4_heated_duct.jl  (10 GB)
#   FV3D_BACKEND=cuda FV3D_NS=16,24,32 FV3D_EXT_N=64 julia --project=<env> benchmarks/fv3d/v4_heated_duct.jl  (80 GB)
#   FV3D_BACKEND=cuda FV3D_LINSOLVE=krylov FV3D_NS=16,24,32,48 FV3D_EXT_N=64 julia --project=<env> benchmarks/fv3d/v4_heated_duct.jl  (80 GB)
#   FV3D_BACKEND=cpu  FV3D_NS=4,6      FV3D_EXT_N=8  julia --project benchmarks/fv3d/v4_heated_duct.jl  (smoke; (a) above 3 % expected)
if lowercase(get(ENV, "FV3D_BACKEND", "cuda")) == "cuda"
    using CUDA
    CUDA.functional() || error("CUDA not functional")
    using CUDSS
end
include(joinpath(@__DIR__, "fv3d_common.jl"))

# ---- (a) heated duct
const NU = 0.05
const GX = 1.0
const ALPHA = 0.05
const Q0 = 1.0
const Q1 = 1.0
const LX = 2.0
const KX = 2pi / LX
const TW = 1.0
const GATE_A = 0.03

# square-duct Fourier series, as in v1_duct_cavity.jl
_cosh_ratio(x, z, b) = exp(x * (abs(z) - b)) * (1 + exp(-2x * abs(z))) / (1 + exp(-2x * b))
function duct_series(y, z; G = 1.0, nu = 1.0, a = 0.5, b = 0.5, terms = 401)
    yc, zc = y - a, z - b
    s = 0.0
    for i in 1:2:terms
        x = i * pi / (2a)
        s += (-1)^((i - 1) ÷ 2) * (1 - _cosh_ratio(x, zc, b)) * cos(x * yc) / i^3
    end
    return 16a^2 * G / (nu * pi^3) * s
end

# cell-centred 5-point Laplacian on the unit square, zero Dirichlet walls at the faces
function section_laplacian(M)
    hf2 = (1 / M)^2
    idx(j, l) = j + (l - 1) * M
    I = Int[]; J = Int[]; V = Float64[]
    for l in 1:M, j in 1:M
        c = idx(j, l); diag = 0.0
        for (dj, dl) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            jj, ll = j + dj, l + dl
            if 1 <= jj <= M && 1 <= ll <= M
                push!(I, c); push!(J, idx(jj, ll)); push!(V, 1 / hf2); diag -= 1 / hf2
            else
                diag -= 2 / hf2                  # wall value 0 half a cell away
            end
        end
        push!(I, c); push!(J, c); push!(V, diag)
    end
    return sparse(I, J, V, M * M, M * M)
end

function duct_reference(N)
    r = max(3, cld(300, N)); iseven(r) && (r += 1)
    M = r * N; hf = 1 / M
    t = @elapsed begin
        L = section_laplacian(M)
        u = [duct_series((j - 0.5) * hf, (l - 0.5) * hf; G = GX, nu = NU) for j in 1:M, l in 1:M]
        theta0 = lu(L) \ fill(-Q0 / ALPHA, M * M)
        A = convert(SparseMatrixCSC{ComplexF64,Int}, ALPHA * L) -
            spdiagm(0 => ALPHA * KX^2 .+ im * KX .* vec(u))
        theta1 = lu(A) \ fill(complex(-Q1), M * M)
    end
    return (; r, M, u, theta0 = reshape(theta0, M, M), theta1 = reshape(theta1, M, M), seconds = t)
end

function duct_case(N)
    ref = duct_reference(N)
    Nx = 2N; h = 1 / N
    bc = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall)
    energy = FVEnergy3D(; alpha = ALPHA, boundary = (S = TW, N = TW, B = TW, T = TW),
                        source = (x, y, z) -> Q0 + Q1 * cos(KX * x))
    p = fv_steady_problem(Nx, N, N; h, nu = NU, bc, source = (x, y, z) -> (GX, 0.0, 0.0, 0.0), energy)
    m0 = fv3d_devused()
    r, mem, kd = fv3d_solve_selected(p, N)
    u, v, w, _, T = fv_fields(p, r.U)
    rr = ref.r
    jf(j) = rr * (j - 1) + (rr + 1) ÷ 2         # fine cell whose centre is the coarse centre
    theta_ref = [ref.theta0[jf(j), jf(l)] + real(ref.theta1[jf(j), jf(l)] * cis(KX * (i - 0.5) * h))
                 for i in 1:Nx, j in 1:N, l in 1:N]
    uref = [ref.u[jf(j), jf(l)] for i in 1:Nx, j in 1:N, l in 1:N]
    d = (T .- TW) .- theta_ref
    du = sqrt.((u .- uref) .^ 2 .+ v .^ 2 .+ w .^ 2)
    st = r.stats
    return (; N, M = ref.M, e2 = norm(d) / norm(theta_ref), einf = maximum(abs, d) / maximum(abs, theta_ref),
            eu2 = norm(du) / norm(uref), theta_max = maximum(abs, theta_ref),
            converged = st.converged, gate = st.gate, iterations = st.iterations,
            seconds = r.seconds, t_setup = r.t_setup, t_reference = ref.seconds,
            mem_gb = mem / 1e9, mem_delta_gb = (mem - m0) / 1e9, krylov = r.krylov, kd)
end

# ---- (b) extrusion
function extrusion_problems(M2)
    Nx, Ny = 2M2, M2
    visc = FVGNFViscosity(CarreauYasuda(0.3, 0.05, 1.0, 2.0, 0.7; thermal = ArrheniusCoupling(1.0, 2.0)), 0.0;
                          temperature = FVStateTemperature())
    solid = falses(Nx, Ny)
    b = max(1, M2 ÷ 4)
    solid[(Nx ÷ 4 + 1):(Nx ÷ 4 + b), (Ny ÷ 2 - b ÷ 2 + 1):(Ny ÷ 2 - b ÷ 2 + b)] .= true
    p2 = fv_steady_problem(Nx, Ny; h = 1 / Ny, bc = (W = :periodic, E = :periodic, S = :wall, N = :wall),
                           solid, source = (x, y) -> (1.0, 0.0, 0.0), viscosity = visc,
                           energy = FVEnergy(; alpha = 0.05, sigma = 0.3, source = 0.2, bottom = 1.0, top = 1.5))
    p3 = fv_steady_problem(Nx, Ny, 3; h = 1 / Ny, solid = repeat(solid, 1, 1, 3),
                           bc = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :periodic, T = :periodic),
                           source = (x, y, z) -> (1.0, 0.0, 0.0, 0.0), viscosity = visc,
                           energy = FVEnergy3D(; alpha = 0.05, sigma = 0.3, source = 0.2,
                                               boundary = (S = 1.0, N = 1.5)))
    return p2, p3
end

function extrusion_case(M2)
    p2, p3 = extrusion_problems(M2)
    g2, g3 = p2.grid, p3.grid
    Nx, Ny = g2.Nx, g2.Ny
    m0 = fv3d_devused()
    r2 = fv3d_solve(p2)
    r3 = fv3d_solve(p3)
    mem = fv3d_devused()
    u2, v2, pr2, T2 = fv_fields(p2, r2.U)
    u3, v3, w3, pr3, T3 = fv_fields(p3, r3.U)
    ext(a) = repeat(a, 1, 1, 3)
    umax = maximum(abs, u2)
    solve_diff = (u = maximum(abs, u3 - ext(u2)) / umax, v = maximum(abs, v3 - ext(v2)) / umax,
                  w = maximum(abs, w3) / umax,
                  p = maximum(abs, pr3 - ext(pr2)) / max(maximum(abs, pr2), 1e-300),
                  T = maximum(abs, T3 - ext(T2)) / maximum(abs, T2 .- 1.0))
    # (b1) device residuals at a perturbed copy of the 2D solution (a converged
    # state has round-off residuals only) and at its extrusion, pins cleared on
    # both sides; both states are packed from the same absolute temperatures.
    q2 = deepcopy(p2); q3 = deepcopy(p3)
    replace!(q2.tag, FV_CELL_PINNED => FV_CELL_FLUID); replace!(q3.tag, FV_CELL_PINNED => FV_CELL_FLUID)
    up, vp, prp, Tp = fv_fields(p2, r2.U .* (1 .+ 0.1 .* sin.(1:length(r2.U))))
    U2 = fv_state(q2, up, vp, prp, Tp)
    U3 = fv_state(q3, ext(up), ext(vp), zeros(Nx, Ny, 3), ext(prp), ext(Tp))
    to = fv3d_to()
    R2 = Array(fv_residual(to(U2), fv_adapt(to, q2)))
    R3 = Array(fv_residual(to(U3), fv_adapt(to, q3)))
    h = g3.h
    slot_pairs = ((1, 1), (2, 2), (4, 3), (5, 4))
    scales = [h * maximum(abs(R2[fv_index(g2, i, j, b)]) for j in 1:Ny, i in 1:Nx if q2.tag[i, j] != FV_CELL_SOLID)
              for (_, b) in slot_pairs]
    worst = zeros(length(slot_pairs)); worst_w = 0.0
    for k in 1:3, j in 1:Ny, i in 1:Nx
        hs = q3.tag[i, j, k] == FV_CELL_SOLID ? 1.0 : h
        for (c, (a, b)) in enumerate(slot_pairs)
            worst[c] = max(worst[c], abs(R3[fv_index(g3, i, j, k, a)] - hs * R2[fv_index(g2, i, j, b)]))
        end
        worst_w = max(worst_w, abs(R3[fv_index(g3, i, j, k, 3)]))
    end
    residual_rel = (; u = worst[1] / scales[1], v = worst[2] / scales[2], p = worst[3] / scales[3],
                    T = worst[4] / scales[4], w = worst_w / max(scales[1], scales[2]))
    return (; Nx, Ny, solve_diff, residual_rel, converged = r2.stats.converged && r3.stats.converged,
            gates = (r2.stats.gate, r3.stats.gate), iterations = (r2.stats.iterations, r3.stats.iterations),
            seconds = (r2.seconds, r3.seconds), mem_gb = mem / 1e9, mem_delta_gb = (mem - m0) / 1e9,
            Tmax = maximum(T2))
end

fv3d_banner("v4_heated_duct")
println("   linear solver of (a) ", fv3d_linsolve_kind(),
        fv3d_linsolve_kind() == "krylov" ? "  (GMRES + P3L; direct check for N <= $(fv3d_direct_max()))" : "",
        "; (b) direct")
println("NOTE 3D thermal LBM reference: not run (the driver covers closed cavities only; a periodic heated-duct driver is not written)")
Ns = fv3d_ints("FV3D_NS", FV3D_GPU ? "16,24" : "4,6")
M2 = only(fv3d_ints("FV3D_EXT_N", FV3D_GPU ? "32" : "8"))
results = Dict{String,Any}("backend" => FV3D_BACKEND, "Ns" => Ns, "ext_N" => M2,
                           "linsolve_duct" => fv3d_linsolve_kind(), "direct_max" => fv3d_direct_max(),
                           "gates" => Dict("duct_theta_rel" => GATE_A, "ext_residual" => 1e-14, "ext_solve" => 1e-7),
                           "lbm_reference" => "not run: the 3D thermal LBM driver covers closed cavities only")
failures = String[]
skipped = String[]

# (a)
rows = Any[]
krows = Dict{String,Any}[]
for N in Ns
    c = try
        duct_case(N)
    catch err
        fv3d_free()
        if fv3d_isoom(err)
            println("SKIPPED duct N=", N, " and larger: out of GPU memory"); flush(stdout)
            push!(skipped, "duct N=$N")
        else
            println("ERROR duct N=", N, ": ", sprint(showerror, err)); flush(stdout)
            push!(failures, "duct N=$N error")
        end
        nothing
    end
    c === nothing && break
    fv3d_free()
    push!(rows, c)
    @printf("duct N=%3d (%d×%d×%d, reference M=%d)  theta rel L2 %.3e Linf %.3e  u rel L2 %.3e  theta_max %.3f  it=%d gate=%s conv=%s  setup %.1f s  Newton %.1f s  reference %.1f s  device %.2f GB (+%.2f)\n",
            c.N, 2c.N, c.N, c.N, c.M, c.e2, c.einf, c.eu2, c.theta_max, c.iterations, c.gate, c.converged,
            c.t_setup, c.seconds, c.t_reference, c.mem_gb, c.mem_delta_gb)
    flush(stdout)
    c.converged || push!(failures, "duct N=$N not converged")
    push!(krows, fv3d_report_krylov!(failures, "duct N=$N", c, c.kd))
end
if isempty(rows)
    push!(failures, "duct: no grid solved")
else
    last_row = rows[end]
    ok_a = last_row.e2 <= GATE_A && last_row.einf <= GATE_A
    ok_a || push!(failures, @sprintf("duct N=%d theta error L2 %.3e / Linf %.3e > %.2f", last_row.N,
                                    last_row.e2, last_row.einf, GATE_A))
    ord(f) = [fv3d_order(f(rows[k]), f(rows[k + 1]), rows[k].N, rows[k + 1].N) for k in 1:length(rows) - 1]
    o2, oinf = ord(r -> r.e2), ord(r -> r.einf)
    @printf("SUMMARY duct  finest N=%d  theta rel L2 %.3e Linf %.3e (gate %.2f) %s | orders L2 %s Linf %s | max device %.2f GB, max Newton %.1f s\n",
            last_row.N, last_row.e2, last_row.einf, GATE_A, ok_a ? "ok" : "ABOVE GATE",
            join((@sprintf("%.2f", o) for o in o2), ","), join((@sprintf("%.2f", o) for o in oinf), ","),
            maximum(r.mem_gb for r in rows), maximum(r.seconds for r in rows))
    results["duct"] = Dict("N" => [r.N for r in rows], "M_reference" => [r.M for r in rows],
                           "theta_rel_l2" => [r.e2 for r in rows], "theta_rel_linf" => [r.einf for r in rows],
                           "u_rel_l2" => [r.eu2 for r in rows], "iterations" => [r.iterations for r in rows],
                           "gate" => [string(r.gate) for r in rows], "seconds" => [r.seconds for r in rows],
                           "t_setup" => [r.t_setup for r in rows], "device_gb" => [r.mem_gb for r in rows],
                           "order_l2" => o2, "order_linf" => oinf, "pass" => ok_a)
    any(!isempty, krows) && (results["duct"]["branch_k"] = krows)
end

# (b)
ex = try
    extrusion_case(M2)
catch err
    fv3d_free()
    println("ERROR extrusion N=", M2, ": ", sprint(showerror, err)); flush(stdout)
    push!(failures, fv3d_isoom(err) ? "extrusion N=$M2 out of GPU memory" : "extrusion N=$M2 error")
    nothing
end
fv3d_free()
if ex !== nothing
    ok_b1 = all(<=(1e-14), values(ex.residual_rel))
    ok_b2 = all(<=(1e-7), values(ex.solve_diff))
    ex.converged || push!(failures, "extrusion not converged")
    ok_b1 || push!(failures, "extrusion residual above 1e-14: " * string(ex.residual_rel))
    ok_b2 || push!(failures, "extrusion solve difference above 1e-7: " * string(ex.solve_diff))
    @printf("SUMMARY extrusion %d×%d (×3)  residual rel u %.1e v %.1e p %.1e T %.1e w %.1e %s | solve rel u %.1e v %.1e w %.1e p %.1e T %.1e %s | it 2D/3D %d/%d gates %s/%s  Newton %.1f/%.1f s  Tmax %.3f  device %.2f GB (+%.2f)\n",
            ex.Nx, ex.Ny, ex.residual_rel.u, ex.residual_rel.v, ex.residual_rel.p, ex.residual_rel.T, ex.residual_rel.w,
            ok_b1 ? "ok" : "ABOVE GATE", ex.solve_diff.u, ex.solve_diff.v, ex.solve_diff.w, ex.solve_diff.p,
            ex.solve_diff.T, ok_b2 ? "ok" : "ABOVE GATE", ex.iterations[1], ex.iterations[2], ex.gates[1], ex.gates[2],
            ex.seconds[1], ex.seconds[2], ex.Tmax,
            ex.mem_gb, ex.mem_delta_gb)
    results["extrusion"] = Dict("Nx" => ex.Nx, "Ny" => ex.Ny,
                                "residual_rel" => Dict(string(k) => v for (k, v) in pairs(ex.residual_rel)),
                                "solve_rel" => Dict(string(k) => v for (k, v) in pairs(ex.solve_diff)),
                                "iterations" => collect(ex.iterations), "seconds" => collect(ex.seconds),
                                "device_gb" => ex.mem_gb, "pass" => ok_b1 && ok_b2 && ex.converged)
end
results["failures"] = failures
results["skipped_oom"] = skipped
fv3d_write(fv3d_out("v4_heated_duct"), results)
exit(fv3d_verdict("v4_heated_duct", failures))
