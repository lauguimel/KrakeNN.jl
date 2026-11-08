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
