# V0' energy rows — 3D manufactured temperature on the GPU (plan "solveur FV
# stationnaire 3D", increment I3, step 5). The energy equation of FVEnergy3D,
#   div(u T) − α ∇²T = 2σ ν S:S + q,
# on [0, 1]³, N³ cells, coupled Newton (five fields, branch D) from the exact
# cell state. q is computed with ForwardDiff from the exact fields.
#   conduction: u = 0 (walls at rest, no momentum source), σ = 0, α = 0.1;
#   convection: the V0' Navier–Stokes flow of v0p_mms.jl (divergence-free
#               trigonometric velocity, variable ν, exact pressure) with viscous
#               heating σ = 0.2 and α = 0.02 (cell Péclet about 50/N).
# Two temperature / side sets:
#   walls:      T = T0 + ΔT sin πx sin πy cos πz, six walls, Dirichlet T0 on
#               W/E/S/N, adiabatic B/T (∂T/∂z = 0 there);
#   periodic_x: T = T0 + ΔT cos 2πx sin πy cos πz, x periodic, Dirichlet T0 on
#               S/N, adiabatic B/T.
# T0 = 1, ΔT = 0.5. L2 (rms) and L∞ errors of T, observed orders between
# successive grids; velocity errors of the convection cases are printed for
# reference (their gate is v0p_mms.jl). Per grid: Newton iterations, stopping
# gate, setup and Newton time, device memory in use after the solve (memory
# pool included). A grid that runs out of GPU memory is reported as skipped
# (larger grids of that case are skipped too) and is not a failure.
#
# Conduction: u = 0 is exact, so the case-data velocity scale of the thermal
# Newton gate is its 1e-12 floor and round-off velocity increments would read
# as large scaled momentum residuals; these cases pass a unit velocity scale
# (`thermal_scales`), the energy scale is unchanged.
#
# Gates (finest pair, L2 and L∞ of T): conduction ≥ 1.8, convection ≥ 0.9.
#
# Linear solver of the Newton steps (fv3d_common.jl): FV3D_LINSOLVE=direct
# (default, cuDSS) or krylov (branch K: GMRES + P3L, the V5 configuration, for
# grids whose factorisation does not fit, 48³ and above on 80 GB). With krylov,
# per grid: GMRES iterations of every Newton step and their maximum, any
# unconverged GMRES solve (a failure), and for N <= FV3D_DIRECT_MAX (24) a
# second solve with direct: relative max difference of the state ≤ 1e-6 (a
# failure otherwise). The order gates do not depend on the solver.
#
#   FV3D_CASES=conduction_walls,convection_periodic_x restricts the cases (default: all four).
#   FV3D_BACKEND=cuda FV3D_NS=12,16,24    julia --project=<env> benchmarks/fv3d/v0p_energy_mms.jl  (10 GB)
#   FV3D_BACKEND=cuda FV3D_NS=16,24,32,48 julia --project=<env> benchmarks/fv3d/v0p_energy_mms.jl  (80 GB)
#   FV3D_BACKEND=cuda FV3D_LINSOLVE=krylov FV3D_NS=16,24,32,48,64 julia --project=<env> benchmarks/fv3d/v0p_energy_mms.jl  (80 GB)
#   FV3D_BACKEND=cpu  FV3D_NS=4,6         julia --project benchmarks/fv3d/v0p_energy_mms.jl        (smoke)
if lowercase(get(ENV, "FV3D_BACKEND", "cuda")) == "cuda"
    using CUDA
    CUDA.functional() || error("CUDA not functional")
    using CUDSS
end
include(joinpath(@__DIR__, "fv3d_common.jl"))
const ForwardDiff = KrakeNN.ForwardDiff

# ---- flow of v0p_mms.jl (same formulas)
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
function mms_source(x)
    G(y) = ForwardDiff.jacobian(uex, y)
    σ(y) = (Gy = G(y); vec(nuex(y) .* (Gy .+ Gy')))
    Dσ = ForwardDiff.jacobian(σ, x)
    divσ = ntuple(r -> sum(Dσ[r + 3 * (c - 1), c] for c in 1:3), 3)
    gp = ForwardDiff.gradient(pex, x)
    cv = G(x) * uex(x)
    return (cv[1] + gp[1] - divσ[1], cv[2] + gp[2] - divσ[2], cv[3] + gp[3] - divσ[3], 0.0)
end

# ---- temperature
const T0 = 1.0
const DT = 0.5
const ALPHA = (conduction = 0.1, convection = 0.02)
const SIGMA = 0.2
Tex_walls(x) = T0 + DT * sin(pi * x[1]) * sin(pi * x[2]) * cos(pi * x[3])
Tex_px(x) = T0 + DT * cos(2pi * x[1]) * sin(pi * x[2]) * cos(pi * x[3])

function energy_source(x, Te, conv)
    lap = tr(ForwardDiff.hessian(Te, x))
    conv || return -ALPHA.conduction * lap
    G = ForwardDiff.jacobian(uex, x)
    S = (G .+ G') ./ 2
    return dot(uex(x), ForwardDiff.gradient(Te, x)) - ALPHA.convection * lap -
           2 * SIGMA * nuex(x) * sum(abs2, S)
end

const CASES = (("walls", (W = :wall, E = :wall, S = :wall, N = :wall, B = :wall, T = :wall),
                (W = T0, E = T0, S = T0, N = T0), Tex_walls),
               ("periodic_x", (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall),
                (S = T0, N = T0), Tex_px))
const CASE_FILTER = split(get(ENV, "FV3D_CASES", ""), ','; keepempty = false)

function energy_case(N, bc, walls, Te, conv)
    X(x, y, z) = [x, y, z]
    if conv
        energy = FVEnergy3D(; alpha = ALPHA.convection, sigma = SIGMA, boundary = walls,
                            source = (x, y, z) -> energy_source(X(x, y, z), Te, true))
        p = fv_steady_problem(N, N, N; nu = (x, y, z) -> nuex(X(x, y, z)), bc, conv = true,
                              boundary_velocity = (s, x, y, z) -> Tuple(uex(X(x, y, z))),
                              source = (x, y, z) -> mms_source(X(x, y, z)), energy)
        Ue = fv_state(p, ((x, y, z) -> uex(X(x, y, z))[c] for c in 1:3)...,
                      (x, y, z) -> pex(X(x, y, z)), (x, y, z) -> Te(X(x, y, z)))
        newton = (;)
    else
        energy = FVEnergy3D(; alpha = ALPHA.conduction, boundary = walls,
                            source = (x, y, z) -> energy_source(X(x, y, z), Te, false))
        p = fv_steady_problem(N, N, N; nu = 1.0, bc, energy)
        Ue = fv_state(p, 0.0, 0.0, 0.0, 0.0, (x, y, z) -> Te(X(x, y, z)))
        sc = KrakeNN._fv_energy_scales(p)
        h = p.grid.h; H = p.grid.Ny * h
        mom = sc.nu_ref * h^3 / H^2                      # Uc = 1
        newton = (; thermal_scales = (; Uc = 1.0, nu_ref = sc.nu_ref, deltaT = sc.deltaT,
                                      equations = (mom, mom, mom, h^2, sc.equations[5])))
    end
    m0 = fv3d_devused()
    r, mem, kd = fv3d_solve_selected(p, N; U0 = Ue, maxit = 20, newton...)
    st = r.stats
    st.converged || println("  not converged: gate ", st.gate, ", scaled equation max ",
                            round.(st.equation_inf[end]; sigdigits = 3))
    f = fv_fields(p, r.U)
    hc = 1 / N
    xc(i, j, k) = [(i - 0.5) * hc, (j - 0.5) * hc, (k - 0.5) * hc]
    eT = [f[5][i, j, k] - Te(xc(i, j, k)) for i in 1:N, j in 1:N, k in 1:N]
    eu2 = eu_inf = 0.0
    if conv
        du = [sqrt(sum(abs2, (f[1][i, j, k], f[2][i, j, k], f[3][i, j, k]) .- Tuple(uex(xc(i, j, k)))))
              for i in 1:N, j in 1:N, k in 1:N]
        eu2 = sqrt(sum(abs2, du) / N^3); eu_inf = maximum(du)
    end
    return (; N, eT2 = sqrt(sum(abs2, eT) / N^3), eTinf = maximum(abs, eT), eu2, eu_inf,
            converged = st.converged, gate = st.gate, iterations = st.iterations,
            seconds = r.seconds, t_setup = r.t_setup, mem_gb = mem / 1e9, mem_delta_gb = (mem - m0) / 1e9,
            krylov = r.krylov, kd)
end

fv3d_banner("v0p_energy_mms")
println("   linear solver ", fv3d_linsolve_kind(),
        fv3d_linsolve_kind() == "krylov" ? "  (GMRES + P3L; direct check for N <= $(fv3d_direct_max()))" : "")
Ns = fv3d_ints("FV3D_NS", FV3D_GPU ? "12,16,24" : "4,6")
results = Dict{String,Any}("backend" => FV3D_BACKEND, "Ns" => Ns, "linsolve" => fv3d_linsolve_kind(),
                           "direct_max" => fv3d_direct_max(),
                           "gates" => Dict("conduction_T" => 1.8, "convection_T" => 0.9),
                           "alpha" => Dict("conduction" => ALPHA.conduction, "convection" => ALPHA.convection),
                           "sigma_convection" => SIGMA)
failures = String[]
skipped = String[]
for (sname, bc, walls, Te) in CASES, conv in (false, true)
    name = (conv ? "convection_" : "conduction_") * sname
    isempty(CASE_FILTER) || name in CASE_FILTER || continue
    rows = Any[]
    krows = Dict{String,Any}[]
    for N in Ns
        c = try
            energy_case(N, bc, walls, Te, conv)
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
        @printf("%-24s N=%3d  eT2=%.3e eTinf=%.3e  eu2=%.3e euinf=%.3e  it=%d gate=%s conv=%s  setup %.1f s  Newton %.1f s  device %.2f GB (+%.2f)\n",
                name, N, c.eT2, c.eTinf, c.eu2, c.eu_inf, c.iterations, c.gate, c.converged,
                c.t_setup, c.seconds, c.mem_gb, c.mem_delta_gb)
        flush(stdout)
        c.converged || push!(failures, "$name N=$N not converged")
        push!(krows, fv3d_report_krylov!(failures, "$name N=$N", c, c.kd))
    end
    length(rows) >= 2 || continue
    ord(f) = [fv3d_order(f(rows[k]), f(rows[k + 1]), rows[k].N, rows[k + 1].N) for k in 1:length(rows) - 1]
    o2, oinf = ord(r -> r.eT2), ord(r -> r.eTinf)
    gate = conv ? 0.9 : 1.8
    ok = o2[end] >= gate && oinf[end] >= gate
    o2[end] >= gate || push!(failures, @sprintf("%s T L2 order %.2f < %.2f", name, o2[end], gate))
    oinf[end] >= gate || push!(failures, @sprintf("%s T Linf order %.2f < %.2f", name, oinf[end], gate))
    @printf("SUMMARY %-24s order T L2 %s  Linf %s | gate %.1f %s | max device %.2f GB, max Newton %.1f s\n", name,
            join((@sprintf("%.2f", o) for o in o2), ","), join((@sprintf("%.2f", o) for o in oinf), ","),
            gate, ok ? "ok" : "BELOW GATE", maximum(r.mem_gb for r in rows), maximum(r.seconds for r in rows))
    results[name] = Dict("N" => [r.N for r in rows], "eT2" => [r.eT2 for r in rows],
                         "eTinf" => [r.eTinf for r in rows], "eu2" => [r.eu2 for r in rows],
                         "euinf" => [r.eu_inf for r in rows], "iterations" => [r.iterations for r in rows],
                         "gate" => [string(r.gate) for r in rows], "seconds" => [r.seconds for r in rows],
                         "t_setup" => [r.t_setup for r in rows], "device_gb" => [r.mem_gb for r in rows],
                         "order_T_l2" => o2, "order_T_linf" => oinf, "pass" => ok)
    any(!isempty, krows) && (results[name]["branch_k"] = krows)
end
results["failures"] = failures
results["skipped_oom"] = skipped
fv3d_write(fv3d_out("v0p_energy_mms"), results)
exit(fv3d_verdict("v0p_energy_mms", failures))
