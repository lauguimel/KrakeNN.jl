#!/usr/bin/env julia
# =====================================================================
# make_crossslot_csv.jl — PLANAR STAGNATION-POINT (cross-slot core)
# comparison of the 4 Kraken viscoelastic constitutive models
# (Oldroyd-B, FENE-P, Giesekus, PTT) using the REAL Kraken FVFD
# log-conformation solver.
#
#   Page : docs/src/users/benchmarks/ve-constitutive-models.md
#   Out  : ve_crossslot_fields.csv  (next to this script)
#
# The cross-slot's iconic central birefringent "strand" is a planar
# stagnation-point flow effect. With the imposed straining field
#
#     u = (ε̇·x, −ε̇·y, 0)
#
# fluid elements near the stagnation streamline (the inflow y-axis)
# decelerate toward the centre, then accelerate out along the outflow
# x-axis: they spend a LONG residence time in the extensional region, so
# the conformation tensor C builds a high-stretch STRAND along the
# outflow (x) axis while staying near equilibrium along the inflow (y)
# axis. This is driven by conformation ADVECTION (the M1 transport term),
# so even a spatially uniform imposed strain rate produces a strongly
# NON-uniform trace(C) field.
#
# `run_viscoelastic_fvfd_extensional_3d(velocity_mode=:imposed)` imposes
# exactly this stagnation field (cheap: no LBM flow solve) and runs the
# full FVFD conformation pipeline (advect + constitutive + Guo coupling),
# so the strand emerges from the real solver.
#
# Expectation: Oldroyd-B (unbounded stretch) → most intense strand;
# FENE-P / Giesekus / PTT → weaker strand, capped by finite extensibility /
# extension-thinning.
#
# Run (from repo root):
#   julia --project=. benchmarks/results/repro/ve-constitutive-models/make_crossslot_csv.jl
# CPU Float64, no GPU; CI-reproducible (~a few minutes at N=32, λε̇=0.35).
# =====================================================================
using Kraken
using Printf

const HERE = @__DIR__

# --- operating point (matched G/β to the analytical & Poiseuille figs) ----
const FT       = Float64
const NX, NY, NZ = 32, 32, 4          # 2-D in the z-mid plane (periodic z)
const NU_TOTAL = 0.1
const BETA     = 0.5
const NU_S     = BETA * NU_TOTAL
const NU_P     = (1 - BETA) * NU_TOTAL
# The imposed stagnation field is u=(ε̇·x,−ε̇·y,0): u_max = ε̇·(N/2). For the
# coupled D3Q19 solvent step (which stays live in :imposed mode) to remain
# below the lattice stability limit, ε̇ must be SMALL in lattice units. We use
# a small ε̇ and a large λ so the extension number λε̇ = 0.35 still drives a
# strong (sub-pole) Oldroyd-B stretch — exactly the validated extensional
# canary regime (driver default ε̇=0.005, λ=50 → λε̇=0.25).
const EPS_DOT  = 0.005
const LAMBDA   = 70.0                 # λε̇ = 0.35 → strong OB stretch, below the 0.5 pole
const LE       = LAMBDA * EPS_DOT
const MAX_STEPS = 1500                # strand is quasi-steady within the residence time
const GMOD     = FT(NU_P / LAMBDA)

# The 4 validated models at the SAME parameters as the other figures.
models = [
    ("oldroydb", LogConfOldroydB(G=GMOD, λ=FT(LAMBDA))),
    ("fenep",    LogConfFENEP(G=GMOD, λ=FT(LAMBDA), L²=FT(50.0))),
    ("giesekus", LogConfGiesekus(G=GMOD, λ=FT(LAMBDA), α=FT(0.2))),
    ("ptt",      LogConfPTT(G=GMOD, λ=FT(LAMBDA), ε=FT(0.25), variant=:linear)),
]

# Default backend is KernelAbstractions.CPU() → CPU Float64, no GPU.
run_crossslot(model) = Kraken.run_viscoelastic_fvfd_extensional_3d(;
    Nx=NX, Ny=NY, Nz=NZ, epsilon_dot=EPS_DOT,
    ν_s=NU_S, ν_p=nothing, lambda=LAMBDA, polymer_model=model,
    max_steps=MAX_STEPS, FT=FT, velocity_mode=:imposed,
    advection_scheme=:muscl_superbee)

@printf("Cross-slot (planar stagnation) setup: Nx=%d Ny=%d Nz=%d ε̇=%g λε̇=%g β=%g steps=%d\n",
        NX, NY, NZ, EPS_DOT, LE, BETA, MAX_STEPS)

const KZ = NZ ÷ 2 + 1                  # z-mid plane
const IC = NX ÷ 2 + 1                  # stagnation point (x index)
const JC = NY ÷ 2 + 1                  # stagnation point (y index)

# --- run + collect the z-mid trace(C) field ---------------------------
fields = Dict{String,Matrix{Float64}}()
peak_trC = Dict{String,Float64}()

for (name, model) in models
    res = run_crossslot(model)
    @assert res.completed_steps == MAX_STEPS
    trC3 = res.C_xx .+ res.C_yy .+ res.C_zz
    @assert all(isfinite, trC3) "non-finite trace(C) for $name"
    sl = trC3[:, :, KZ]                 # (Nx, Ny)
    fields[name] = sl
    peak_trC[name] = maximum(sl)

    # Anisotropy at r = NX÷4 along the two axes (outflow x vs inflow y).
    r = NX ÷ 4
    outm = (sl[IC + r, JC] + sl[IC - r, JC]) / 2
    inm  = (sl[IC, JC + r] + sl[IC, JC - r]) / 2
    @printf("  %-9s  centre trC=%6.3f  peak trC=%7.3f  outflow(r=%d)=%6.3f  inflow=%6.3f  aniso=%5.2f×\n",
            name, sl[IC, JC], maximum(sl), r, outm, inm, outm / inm)
end

# --- strand ranking (the headline) ------------------------------------
println("\n== peak trace(C) ranking (strand intensity) ==")
ranked = sort(collect(peak_trC); by=p -> -p[2])
for (name, v) in ranked
    @printf("  %-9s  peak trC=%7.3f\n", name, v)
end

# --- write CSV: full z-mid trace(C) field per model -------------------
# Long format: model, i, j, x, y, trC. Coordinates are LU offsets from the
# stagnation point (x = i − IC, y = j − JC) so the figure is centred.
out = joinpath(HERE, "ve_crossslot_fields.csv")
open(out, "w") do io
    println(io, "# Kraken viscoelastic constitutive models — PLANAR STAGNATION-POINT (cross-slot core)")
    @printf(io, "# Nx=%d Ny=%d Nz=%d epsilon_dot=%g lambda_edot=%g beta=%g lambda=%g steps=%d (z-mid plane k=%d)\n",
            NX, NY, NZ, EPS_DOT, LE, BETA, LAMBDA, MAX_STEPS, KZ)
    println(io, "# imposed u=(edot*x,-edot*y,0): outflow along x, inflow along y, stagnation at centre")
    println(io, "# models: oldroydb, fenep(L2=50), giesekus(alpha=0.2), ptt(eps=0.25,linear)")
    println(io, "# x,y are LU offsets from the stagnation point (x=i-ic, y=j-jc)")
    println(io, "model,i,j,x,y,trC")
    for (name, _model) in models
        sl = fields[name]
        for j in 1:NY, i in 1:NX
            @printf(io, "%s,%d,%d,%d,%d,%.10g\n",
                    name, i, j, i - IC, j - JC, sl[i, j])
        end
    end
end
println("\nwrote $(out)  ($(length(models) * NX * NY) rows)")
