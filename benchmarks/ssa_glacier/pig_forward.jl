# Steady SSA on a 2 km crop of Pine Island Glacier (BedMachine v4 + MEaSUREs v2).
# Data: run prep_pig.py first (writes benchmarks/ssa_glacier/data/*.bin, gitignored).
#
#   julia --project=. benchmarks/ssa_glacier/pig_forward.jl
#
# Environment:
#   KRAKEN_SSA_BACKEND = cpu (default) | cuda     (cuda needs CUDA.jl + CUDSS.jl in the env)
#   KRAKEN_PIG_COARSEN = 1 (default, 2 km) | 2 (4 km, quick CPU check)
#   KRAKEN_PIG_NWEIGHTS = number of MLP weights checked against finite differences (default 4)
# Steps: (1) Weertman C sweep + forward solve, (2) learned drag F_theta(|u|, N): adjoint
# gradient vs central finite differences, (3) fields written to benchmarks/ssa_glacier/out/.
# Units: m, yr, kPa (see src/methods/ssa/ssa.jl).
using KrakeNN
using LinearAlgebra
using Random
using Printf
using Statistics
# Load CUDA at top level: an @eval inside a function leaves its bindings in a newer world.
if lowercase(get(ENV, "KRAKEN_SSA_BACKEND", "cpu")) == "cuda"
    using CUDA, CUDSS
end

const DATA = get(ENV, "KRAKEN_PIG_DATA", joinpath(@__DIR__, "data"))
const OUT = joinpath(@__DIR__, "out")
const RHO_W_G = 1028.0 * 9.81 / 1000.0          # kPa/m
const SSA_A = 7.57e-8                           # 2.4e-24 Pa^-3 s^-1 in kPa^-3 yr^-1, as in the tests

# ------------------------------------------------------------------ data
function read_header(path)
    h = Dict{String,String}()
    for l in eachline(path)
        occursin('=', l) || continue
        k, v = strip.(split(l, '='; limit = 2)); h[k] = v
    end
    return h
end

function load_pig(dir = DATA; coarsen::Int = 1)
    h = read_header(joinpath(dir, "pig_header.txt"))
    Nx = parse(Int, h["Nx"]); Ny = parse(Int, h["Ny"]); dx = parse(Float64, h["dx"])
    f = Dict{Symbol,Matrix{Float64}}()
    for name in split(h["fields"])
        raw = Vector{Float64}(undef, Nx * Ny)
        read!(joinpath(dir, name * ".bin"), raw)
        f[Symbol(name)] = reshape(raw, Nx, Ny)
    end
    k = coarsen
    if k > 1
        (Nx % k == 0 && Ny % k == 0) || error("grid $Nx x $Ny not divisible by $k")
        blk(a) = [sum(@view a[(i-1)k+1:i*k, (j-1)k+1:j*k]) / k^2 for i in 1:(Nx÷k), j in 1:(Ny÷k)]
        for s in (:H, :s, :bed, :vx, :vy, :verr)
            f[s] = blk(f[s])
        end
        # mask: ice if any cell of the block is ice; floating if the majority of ice cells floats
        ice = blk(Float64.(f[:mask] .> 0)); fl = blk(Float64.(f[:mask] .== 3))
        f[:mask] = [ice[i, j] < 0.5 ? 0.0 : (fl[i, j] >= 0.5 * ice[i, j] ? 3.0 : 2.0) for i in axes(ice, 1), j in axes(ice, 2)]
        f[:vvalid] = Float64.(blk(f[:vvalid]) .> 0.5)
        Nx ÷= k; Ny ÷= k; dx *= k
    end
    return (; Nx, Ny, dx, f...)
end

"""Build the SSA problem on the crop. `drag` is the basal law. Returns (problem, fields)."""
function pig_problem(d; drag, eps_reg = 1e-5, Hmin = 10.0, Nmin = 10.0)
    Nx, Ny = d.Nx, d.Ny
    H = max.(d.H, Hmin)
    N = max.(KrakeNN.SSA_RHOG .* H .- RHO_W_G .* max.(0.0, -d.bed), Nmin)
    dmask = Float64.(d.mask .!= 3)                 # floating ice: no basal drag
    # Dirichlet = observed velocity of the edge cell on all four sides (E, W, N, S)
    side(c1, c2) = vcat(reshape(c1, 1, :), reshape(c2, 1, :))
    bv = (side(d.vx[Nx, :], d.vy[Nx, :]), side(d.vx[1, :], d.vy[1, :]),
          side(d.vx[:, Ny], d.vy[:, Ny]), side(d.vx[:, 1], d.vy[:, 1]))
    p = ssa_problem(Nx, Ny; Lx = Nx * d.dx, Ly = Ny * d.dx, H, s = d.s, A = SSA_A, n = 3,
                    drag, N, eps_reg, boundary_velocity = bv, drag_mask = dmask)
    return p
end

observed_state(d) = (U = zeros(2 * d.Nx * d.Ny); U[1:2:end] .= vec(d.vx); U[2:2:end] .= vec(d.vy); U)

# misfit weights: ice cells with an observed velocity
misfit_weights(d) = Float64.((d.mask .> 0) .& (d.vvalid .> 0.5))

function misfit_stats(d, U)
    u = U[1:2:end]; v = U[2:2:end]
    w = vec(misfit_weights(d)); ice = vec(d.mask .> 0)
    du = u .- vec(d.vx); dv = v .- vec(d.vy)
    sm = hypot.(u, v); so = hypot.(vec(d.vx), vec(d.vy))
    n = sum(w)
    wg = w .* vec(d.mask .== 2); wf = w .* vec(d.mask .== 3)
    return (rms_grounded = sqrt(sum(wg .* (sm .- so) .^ 2) / sum(wg)),
            rms_floating = sqrt(sum(wf .* (sm .- so) .^ 2) / sum(wf)),
            rms_vec = sqrt(sum(w .* (du .^ 2 .+ dv .^ 2)) / n),
            rms_speed = sqrt(sum(w .* (sm .- so) .^ 2) / n),
            max_model = maximum(sm[ice]), max_obs = maximum(so[ice]), n = Int(n))
end

# ------------------------------------------------------------------ backend
function select_backend()
    if lowercase(get(ENV, "KRAKEN_SSA_BACKEND", "cpu")) == "cuda"
        Main.CUDA.functional() || error("KRAKEN_SSA_BACKEND=cuda but CUDA is not functional")
        return (to = Main.CUDA.CuArray, linsolve = CUDABackendTag(), name = "cuda")
    end
    return (to = Array, linsolve = CPUBackendTag(), name = "cpu")
end

"""Newton with continuation in the regularisation; returns the stats of every stage."""
function solve_continuation!(s, U, eps_regs; atol = 1e-8, rtol = 1e-13, maxit = 60)
    st = ssa_newton_continuation!(s, U, eps_regs; atol, rtol, maxit)
    return st
end

const EPS_LADDER = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)

# ------------------------------------------------------------------ main
function main()
    be = select_backend()
    k = parse(Int, get(ENV, "KRAKEN_PIG_COARSEN", "1"))
    d = load_pig(; coarsen = k)
    ice = d.mask .> 0
    @printf("PIG crop: %d x %d at %.0f m, %d floating of %d cells (%.1f%%), %d non-ice, backend %s\n",
            d.Nx, d.Ny, d.dx, count(==(3), d.mask), length(d.mask), 100 * count(==(3), d.mask) / length(d.mask),
            count(==(0), d.mask), be.name)
    Uobs = observed_state(d)
    w = misfit_weights(d)
    W = zeros(length(Uobs)); W[1:2:end] .= vec(w); W[2:2:end] .= vec(w)

    # (1) Weertman C sweep: Newton from the observed velocity, eps continuation
    println("\n== Weertman sweep (m = 1/3, floating cells without drag)")
    Cs = Tuple(parse.(Float64, split(get(ENV, "KRAKEN_PIG_CS", "5,10,20,40,80,160"), ",")))
    best = nothing
    for C in Cs
        p = pig_problem(d; drag = SSAWeertman(C, 1 / 3))
        s = ssa_newton_solver(p; to = be.to, linsolve = be.linsolve)
        U = be.to(copy(Uobs))
        t = @elapsed st = solve_continuation!(s, U, EPS_LADDER)
        ok = st[end].converged            # intermediate eps stages may stall; the last one is what counts
        m = misfit_stats(d, Array(U))
        @printf("C = %5.1f  converged = %s  its = %s  |R|inf = %.2e  RMS speed all/grounded/floating = %.0f/%.0f/%.0f  max model/obs = %.0f/%.0f  (%.1f s)\n",
                C, ok, join(getfield.(st, :iterations), "+"), st[end].inf[end], m.rms_speed, m.rms_grounded, m.rms_floating, m.max_model, m.max_obs, t)
        # C only acts on grounded cells: choose it on the grounded-cell RMS
        if ok && (best === nothing || m.rms_grounded < best.m.rms_grounded)
            best = (; C, m, U = Array(U), st)
        end
    end
    best === nothing && error("no Weertman case converged")
    @printf("best C = %.1f (grounded-cell RMS %.1f, floating-cell RMS %.1f): RMS speed misfit %.1f m/yr (RMS vector %.1f), Newton its per stage %s, final |R|inf %.2e kPa, max speed model %.0f vs obs %.0f m/yr\n",
            best.C, best.m.rms_grounded, best.m.rms_floating, best.m.rms_speed, best.m.rms_vec, join(getfield.(best.st, :iterations), "+"),
            best.st[end].inf[end], best.m.max_model, best.m.max_obs)
    mkpath(OUT)
    u, v = best.U[1:2:end], best.U[2:2:end]
    for (nm, a) in (("model_speed", reshape(hypot.(u, v), d.Nx, d.Ny)), ("obs_speed", hypot.(d.vx, d.vy)),
                    ("mask", d.mask), ("weights", w))
        write(joinpath(OUT, nm * ".bin"), Float64.(a))
    end
    open(joinpath(OUT, "pig_out.txt"), "w") do io
        println(io, "Nx = $(d.Nx)\nNy = $(d.Ny)\ndx = $(d.dx)\nC = $(best.C)")
    end

    # (2) learned drag F_theta(|u|, N): adjoint gradient vs central FD
    println("\n== Learned drag TinyMLP(|u|, N): adjoint vs central FD")
    mlp = TinyMLP{2,4,1}(; shift = (0.0, 0.0), scale = (1000.0, 1e4))
    pweert = pig_problem(d; drag = SSAWeertman(best.C, 1 / 3))
    # F0 chosen so softplus(z0) ~ 1 reproduces the Weertman drag at the median grounded speed
    ug = filter(x -> x > 0, hypot.(d.vx, d.vy)[d.mask .== 2])
    F0 = best.C * median(ug)^(1 / 3) / log(2)
    theta0 = law_init(mlp; rng = MersenneTwister(1))
    drag = SSALearnedDrag(mlp, theta0; F0)
    pl = ssa_with(pweert; drag, eps_reg = 1e-5)
    sl = ssa_newton_solver(pl; to = be.to, linsolve = be.linsolve)
    U = be.to(copy(best.U)); Ub = be.to(copy(Uobs)); Wd = be.to(W)
    # continuation from the Weertman state through eps_reg (state at theta0)
    st = solve_continuation!(sl, U, EPS_LADDER)
    ok = st[end].converged            # intermediate eps stages may stall; the last one is what counts
    @printf("learned-drag Newton at theta0: converged = %s, its = %s, |R|inf = %.2e\n", ok,
            join(getfield.(st, :iterations), "+"), st[end].inf[end])
    ok || error("learned-drag Newton did not converge at theta0")
    ms = misfit_stats(d, Array(U))
    @printf("learned drag at theta0 (untrained): RMS speed misfit %.1f m/yr\n", ms.rms_speed)
    misfit(Ucur) = (g = Wd .* (Ucur .- Ub); (sum(g .* (Ucur .- Ub)) / 2, g))
    L0, g0 = misfit(U)
    U0 = copy(U)
    # FD re-solves restart from the theta0 state (a small perturbation), same regularisation
    function L_at(theta)
        ssa_set_theta!(sl, theta)
        copyto!(U, U0)
        r = ssa_newton!(sl, U; atol = 1e-9, rtol = 1e-14, maxit = 60)
        r.converged || error("FD Newton failed")
        return misfit(U)[1]
    end
    L0b = L_at(theta0)
    copyto!(U, U0); ssa_set_theta!(sl, theta0)
    ssa_newton!(sl, U; atol = 1e-9, rtol = 1e-14, maxit = 60)
    _, g = misfit(U)
    grad = ssa_theta_gradient(sl, U, g)
    @printf("L(theta0) = %.6e (re-solve %.6e), |grad|inf = %.3e, factorizations = %d\n", L0, L0b, maximum(abs, grad), sl.n_factorize)
    nw = parse(Int, get(ENV, "KRAKEN_PIG_NWEIGHTS", "4"))
    chosen = sortperm(abs.(grad); rev = true)[1:nw]
    for kq in chosen, h in (1e-4, 1e-6)
        tp = copy(theta0); tm = copy(theta0); tp[kq] += h; tm[kq] -= h
        fd = (L_at(tp) - L_at(tm)) / (2h)
        @printf("weight %2d  h = %.0e  adjoint %+.8e  FD %+.8e  rel err %.2e\n", kq, h, grad[kq], fd, abs(fd - grad[kq]) / abs(grad[kq]))
    end
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
