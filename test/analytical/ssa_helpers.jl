# Shared builders for the SSA tests (units: m, yr, kPa; see src/methods/ssa/ssa.jl).
using KrakeNN
using LinearAlgebra
using Random
using KrakeNN.SparseArrays
using KrakeNN.ForwardDiff

const SSA_A = 7.57e-8          # 2.4e-24 Pa^-3 s^-1 in kPa^-3 yr^-1
const SSA_C = 4.3              # Weertman coefficient, kPa (m/yr)^(-1/3): ~20 kPa at 100 m/yr
const SSA_C_STREAM = 10.0      # 2D stream case: ~46 kPa at 100 m/yr, speeds of 50-100 m/yr

# Problem size for the Newton / learned-law tests: 24 on the Mac, 64+ on the cluster.
ssa_test_n() = parse(Int, get(ENV, "KRAKEN_SSA_N", "24"))

# Flat-bed ice-stream rectangle 40 km x 40 km: thickness (= surface) with a mean
# slope and a bump, ice entering/leaving the x edges with a sin profile, zero velocity
# on the lateral (y) margins.
function ssa_stream_problem(N::Integer; drag = SSAWeertman(SSA_C_STREAM, 1 / 3), n = 3, eps_reg = 1e-5)
    L = 40e3
    Hf(x, y) = 1500.0 - 0.003 * x + 20.0 * sin(2pi * x / L) * cos(pi * y / L)
    bv(x, y) = (x == 0.0 || x == L) ? (60.0 * sin(pi * y / L), 0.0) : (0.0, 0.0)
    return ssa_problem(N, N; Lx = L, Ly = L, H = Hf, s = Hf, A = SSA_A, n, drag, eps_reg,
                       boundary_velocity = bv)
end

# TinyMLP drag: inputs (|u|, N) normalised by (100 m/yr, 1e4 kPa); 2 inputs, 4 hidden, 1 layer
ssa_test_mlp() = TinyMLP{2,4,1}(; shift = (0.0, 0.0), scale = (100.0, 1e4))
function ssa_learned_drag(; seed = 1)
    mlp = ssa_test_mlp()
    theta = law_init(mlp; rng = MersenneTwister(seed))
    return SSALearnedDrag(mlp, theta; F0 = 30.0), theta
end

"""Misfit and device state at weights `theta` (re-solves from the observed state)."""
function ssa_misfit_at!(s, U, Uobs, theta)
    ssa_set_theta!(s, theta)
    copyto!(U, Uobs)
    st = ssa_newton!(s, U; atol = 1e-11, rtol = 1e-14, maxit = 40)
    st.converged || error("SSA Newton did not converge for the FD solve")
    return ssa_misfit(U, Uobs)[1]
end

function ssa_learned_gradient_check(; N = ssa_test_n(), to = Array, linsolve = CPUBackendTag(),
                                      nweights = 17, steps = (1e-2, 1e-5, 1e-10), seed = 1)
    pw = ssa_stream_problem(N)
    obs = ssa_solve(pw; atol = 1e-11, rtol = 1e-14)
    @assert obs.converged
    Uobs = Vector{Float64}(Array(obs.U))
    drag, theta0 = ssa_learned_drag(; seed)
    pl = ssa_with(pw; drag)
    s = ssa_newton_solver(pl; to, linsolve)
    U = to(copy(Uobs)); Uo = to(copy(Uobs))
    L0 = ssa_misfit_at!(s, U, Uo, theta0)
    # state at theta0, gradient by the adjoint
    _, g = ssa_misfit(U, Uo)
    nit = s.n_refactorize
    grad, glogA = ssa_theta_gradient(s, U, g; logA = true)
    # d/dlogA by central FD of the re-solved misfit (Bn = Bn0 exp(-logA/n))
    p0 = s.prob
    fdA(h) = begin
        vals = map((-h, h)) do dl
            s.prob = ssa_with(p0; Bn = p0.Bn * exp(-dl / p0.n_glen))
            ssa_misfit_at!(s, U, Uobs_d, theta0)
        end
        (vals[2] - vals[1]) / (2h)
    end
    Uobs_d = to(copy(Uobs))
    fdlogA = fdA(1e-5)
    s.prob = p0
    # weights to check: the largest gradient entries, spread over the layout
    order = sortperm(abs.(grad); rev = true)
    chosen = order[1:nweights]
    fd(k, h) = begin
        tp = copy(theta0); tm = copy(theta0); tp[k] += h; tm[k] -= h
        Lp = ssa_misfit_at!(s, U, Uo, tp); Lm = ssa_misfit_at!(s, U, Uo, tm)
        (Lp - Lm) / (2h)
    end
    rel = Float64[]
    for k in chosen
        e = abs(fd(k, steps[2]) - grad[k]) / abs(grad[k])
        push!(rel, e)
    end
    kV = chosen[1]
    vshape = [abs(fd(kV, h) - grad[kV]) / abs(grad[kV]) for h in steps]
    return (; L0, grad, glogA, fdlogA, chosen, rel, vshape, steps, solver = s, nweights = length(theta0), U)
end
