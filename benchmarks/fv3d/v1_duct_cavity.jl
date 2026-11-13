# V1 — 3D steady FV on the GPU (plan "solveur FV stationnaire 3D", I1):
#
# (1) Newtonian square duct [0, 1]² section, periodic x (Nx = FV3D_DUCT_NX
#     cells), walls in y and z, uniform body force G = 1, ν = 1, Navier–Stokes
#     (u·∇u = 0 for the exact unidirectional flow). Reference: the classical
#     Fourier series (e.g. White, Viscous Fluid Flow, eq. 3-48) with
#     a = b = ½ and centred coordinates,
#       u = 16 a² G / (ν π³) Σ_{i odd} (−1)^((i−1)/2) [1 − cosh(iπz/2a) / cosh(iπb/2a)]
#           cos(iπy/2a) / i³.
#     Gate: L2 velocity order ≥ 1.8 on the finest pair of FV3D_DUCT_NS sections
#     (at least 3 grids required; a grid that runs out of GPU memory is skipped).
#
# (2) Lid-driven cubic cavity [0, 1]³, lid z = 1 moving at u = 1 in +x, Re =
#     1/ν in FV3D_RE, N³ cells for N in FV3D_CAVITY_N (direct-solver limit about 48³ on an
#     80 GB GPU: 64³ runs out of memory). Re = 100 from rest with
#     pseudo-transient continuation; each next Re starts from the previous one.
#     Reported (no gate): u on the vertical centreline (x = y = ½) and w on the
#     horizontal centreline (y = z = ½), their extrema and positions. With
#     FV3D_CAVITY_REF = a TOML file holding, per "re<Re>" table, any of u_min,
#     z_umin, w_max, x_wmax, w_min, x_wmin (e.g. digitised from Ku, Hirsh &
#     Taylor 1987, or Albensoeder & Kuhlmann 2005, mapped to this orientation),
#     the deviations are printed; otherwise NOREF. No reference numbers are
#     shipped in the repository.
#
#   FV3D_BACKEND=cuda FV3D_DUCT_NS=32,64,128 FV3D_CAVITY_N=32,48 FV3D_RE=100,400 \
#       julia --project=<env> benchmarks/fv3d/v1_duct_cavity.jl
#   FV3D_BACKEND=cpu  (defaults 6,8,12 / 6 / 100: smoke)
# The duct no longer reads FV3D_NS (shared with other scripts): use FV3D_DUCT_NS.
#    julia --project benchmarks/fv3d/v1_duct_cavity.jl
if lowercase(get(ENV, "FV3D_BACKEND", "cuda")) == "cuda"
    using CUDA
    CUDA.functional() || error("CUDA not functional")
    using CUDSS
end
include(joinpath(@__DIR__, "fv3d_common.jl"))

# cosh(x z) / cosh(x b) for |z| <= b without overflow
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

function duct_case(N, Nx)
    bc = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall)
    p = fv_steady_problem(Nx, N, N; h = 1 / N, nu = 1.0, bc, source = (x, y, z) -> (1.0, 0.0, 0.0, 0.0))
    r = fv3d_solve(p)
    u, v, w, _ = fv_fields(p, r.U)
    h = 1 / N
    ue = [duct_series((j - 0.5) * h, (k - 0.5) * h) for i in 1:Nx, j in 1:N, k in 1:N]
    du = sqrt.((u .- ue) .^ 2 .+ v .^ 2 .+ w .^ 2)
    umax = maximum(ue)
    return (; N, e2 = sqrt(sum(abs2, du) / length(du)) / umax, einf = maximum(du) / umax,
            umax_fv = maximum(u), umax_ref = umax, converged = r.stats.converged,
            iterations = r.stats.iterations, seconds = r.seconds)
end

# centreline samples (mean of the 2 or 4 cells around the line)
function centrelines(p, U)
    u, v, w, _ = fv_fields(p, U)
    N = p.grid.Nx
    h = p.grid.h
    mid = iseven(N) ? (N ÷ 2, N ÷ 2 + 1) : (N ÷ 2 + 1,)
    avg(f, I, J, K) = sum(f[i, j, k] for i in I, j in J, k in K) / (length(I) * length(J) * length(K))
    zs = [(k - 0.5) * h for k in 1:N]
    uz = [avg(u, mid, mid, (k,)) for k in 1:N]
    wx = [avg(w, (i,), mid, mid) for i in 1:N]
    return zs, uz, wx
end

function cavity_runs(N, Res, ref)
    out = Dict{String,Any}()
    U0 = nothing
    lid = (s, x, y, z) -> s === :T ? (1.0, 0.0, 0.0) : (0.0, 0.0, 0.0)
    for Re in Res
        p = fv_steady_problem(N, N, N; h = 1 / N, nu = 1 / Re, boundary_velocity = lid)
        r = fv3d_solve(p; U0, maxit = 60, dtau0 = 1.0, dtau_max = 1e4)
        U0 = r.U
        xs, uz, wx = centrelines(p, r.U)
        iu = argmin(uz); iwmax = argmax(wx); iwmin = argmin(wx)
        row = Dict{String,Any}("u_min" => uz[iu], "z_umin" => xs[iu], "w_max" => wx[iwmax],
                               "x_wmax" => xs[iwmax], "w_min" => wx[iwmin], "x_wmin" => xs[iwmin],
                               "centre" => xs, "u_vertical" => uz, "w_horizontal" => wx,
                               "converged" => r.stats.converged, "iterations" => r.stats.iterations,
                               "seconds" => r.seconds)
        key = "re$(round(Int, Re))"
        refline = if ref !== nothing && haskey(ref, key)
            join(("$q: $(@sprintf("%+.4f", row[q] - ref[key][q]))" for q in keys(ref[key]) if haskey(row, q)), "  ")
        else
            "NOREF"
        end
        @printf("SUMMARY cavity N=%d Re=%g  u_min=%.4f at z=%.3f  w_max=%.4f at x=%.3f  w_min=%.4f at x=%.3f  it=%d conv=%s  %.1f s  dev: %s\n",
                N, Re, row["u_min"], row["z_umin"], row["w_max"], row["x_wmax"], row["w_min"],
                row["x_wmin"], r.stats.iterations, r.stats.converged, r.seconds, refline)
        flush(stdout)
        out[key] = row
        r = nothing; fv3d_free()   # drop this Re's factorisation before the next one
    end
    return out
end

fv3d_banner("v1_duct_cavity")
Ns = fv3d_ints("FV3D_DUCT_NS", FV3D_GPU ? "32,64,128" : "6,8,12")
Nx = parse(Int, get(ENV, "FV3D_DUCT_NX", "4"))
CNs = fv3d_ints("FV3D_CAVITY_N", FV3D_GPU ? "32,48" : "6")
Res = fv3d_floats("FV3D_RE", FV3D_GPU ? "100,400" : "100")
refpath = get(ENV, "FV3D_CAVITY_REF", "")
ref = isempty(refpath) ? nothing : TOML.parsefile(refpath)
failures = String[]
skipped = String[]
results = Dict{String,Any}("backend" => FV3D_BACKEND, "gates" => Dict("duct_u_order" => 1.8))

println("duct sections ", Ns, "  Nx=", Nx, "  cavity N ", CNs, "  Re ", Res); flush(stdout)
duct = Any[]
for N in Ns
    c = try
        duct_case(N, Nx)
    catch err
        fv3d_free()
        if fv3d_isoom(err)
            println("SKIPPED duct N=", N, ": out of GPU memory"); push!(skipped, "duct N=$N")
        else
            println("ERROR duct N=", N, ": ", sprint(showerror, err)); push!(failures, "duct N=$N error")
        end
        flush(stdout); continue
    end
    fv3d_free()
    push!(duct, c)
    @printf("duct N=%3d  e2/umax=%.3e einf/umax=%.3e  umax fv=%.5f ref=%.5f  it=%d conv=%s  %.1f s\n",
            N, c.e2, c.einf, c.umax_fv, c.umax_ref, c.iterations, c.converged, c.seconds)
    flush(stdout)
    c.converged || push!(failures, "duct N=$N not converged")
end
o2 = [fv3d_order(duct[k].e2, duct[k + 1].e2, duct[k].N, duct[k + 1].N) for k in 1:length(duct) - 1]
oinf = [fv3d_order(duct[k].einf, duct[k + 1].einf, duct[k].N, duct[k + 1].N) for k in 1:length(duct) - 1]
ok = length(duct) >= 3 && o2[end] >= 1.8
ok || push!(failures, length(duct) < 3 ? "duct needs three grids, got $(length(duct)) of $(Ns) (set FV3D_DUCT_NS)" : @sprintf("duct velocity order %.2f < 1.8", o2[end]))
println("SUMMARY duct  order L2 ", join((@sprintf("%.2f", o) for o in o2), ","),
        "  Linf ", join((@sprintf("%.2f", o) for o in oinf), ","), "  ", ok ? "ok" : "BELOW GATE")
results["duct"] = Dict("N" => [c.N for c in duct], "Nx" => Nx, "e2_rel" => [c.e2 for c in duct],
                       "einf_rel" => [c.einf for c in duct], "order_l2" => o2, "order_linf" => oinf,
                       "iterations" => [c.iterations for c in duct], "seconds" => [c.seconds for c in duct],
                       "pass" => ok)
for N in CNs
    cav = try
        cavity_runs(N, Res, ref)
    catch err
        fv3d_free()
        if fv3d_isoom(err)
            println("SKIPPED cavity N=", N, ": out of GPU memory"); push!(skipped, "cavity N=$N")
        else
            println("ERROR cavity N=", N, ": ", sprint(showerror, err)); push!(failures, "cavity N=$N error")
        end
        flush(stdout); nothing
    end
    fv3d_free()
    cav === nothing && continue
    for (k, row) in cav
        row["converged"] || push!(failures, "cavity N=$N $k not converged")
    end
    results["cavity_N$N"] = cav
end
results["failures"] = failures
results["skipped_oom"] = skipped
fv3d_write(fv3d_out("v1_duct_cavity"), results)
exit(fv3d_verdict("v1_duct_cavity", failures))
