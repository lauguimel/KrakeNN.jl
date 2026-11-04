#!/usr/bin/env julia
# =====================================================================
# make_poiseuille_csv.jl — IN-FLOW comparison of the 4 Kraken
# viscoelastic constitutive models (Oldroyd-B, FENE-P, Giesekus, PTT)
# in a fully COUPLED planar Poiseuille channel.
#
#   Page : docs/src/users/benchmarks/ve-constitutive-models.md
#   Out  : ve_poiseuille_profiles.csv  (next to this script)
#
# Where make_csv.jl drives a single homogeneous cell, this driver solves
# the real coupled problem: the D3Q19 solvent + the FVFD log-conformation
# polymer transport feed back on each other through F_poly = ∇·τ_p, so the
# shear rate γ̇(y) is set by the flow, not imposed. The same operating
# point is used for all four models — only the constitutive `polymer_model`
# spec changes — so the y-profiles differ ONLY by the constitutive closure.
# This is exactly the coupled-driver pattern of the constitutive-coupling
# canaries test/test_fvfd_{fenep,giesekus,ptt}_coupled_3d.jl and the
# payoff test test/test_fvfd_poiseuille_payoff_3d.jl.
#
# Channel (matched to the coupled canaries):
#   periodic x/z, half-way bounce-back no-slip y walls, constant body force
#   Fx, D3Q19 solvent at ν_s, polymer at ν_p (β = ν_s/ν_total = 0.5).
#   The shear rate γ̇(y) is ≈0 at the centre and peaks at the walls, so the
#   near-wall band is the high-Wi region where the models separate:
#   tr C = C_xx + C_yy + C_zz is largest for Oldroyd-B (unbounded stretch),
#   thinned/bounded for FENE-P / Giesekus / PTT.
#
# Run (from repo root):
#   julia --project=. benchmarks/results/repro/ve-constitutive-models/make_poiseuille_csv.jl
# CPU Float64, no GPU; CI-reproducible (~1-2 min at Ny=32).
# =====================================================================
using Kraken
using Printf

const HERE = @__DIR__

# --- channel / flow operating point (matched to the coupled canaries) -
const FT       = Float64
const NX, NY, NZ = 6, 32, 6
const NU_TOTAL = 0.1
const BETA     = 0.5
const NU_S     = BETA * NU_TOTAL
const NU_P     = (1 - BETA) * NU_TOTAL
const FX       = 1.5e-5
# Wall shear of the unperturbed parabola sets λ so Wi_wall ≈ 1.
const GAMMA_WALL = FX / (2 * NU_TOTAL) * (NY - 1)
const LAMBDA   = 1.0 / GAMMA_WALL
const GMOD     = FT(NU_P / LAMBDA)
const MAX_STEPS = 10_000

# The 4 validated models at the SAME parameters as the analytical figure.
models = [
    ("oldroydb", LogConfOldroydB(G=GMOD, λ=FT(LAMBDA))),
    ("fenep",    LogConfFENEP(G=GMOD, λ=FT(LAMBDA), L²=FT(50.0))),
    ("giesekus", LogConfGiesekus(G=GMOD, λ=FT(LAMBDA), α=FT(0.2))),
    ("ptt",      LogConfPTT(G=GMOD, λ=FT(LAMBDA), ε=FT(0.25), variant=:linear)),
]

# Default backend is KernelAbstractions.CPU() → CPU Float64, no GPU.
run_poiseuille(model) = Kraken.run_viscoelastic_fvfd_poiseuille_3d(;
    Nx=NX, Ny=NY, Nz=NZ, Fx=FX, ν_s=NU_S, ν_p=nothing, lambda=LAMBDA,
    polymer_model=model, max_steps=MAX_STEPS,
    FT=FT, advection_scheme=:muscl_superbee)

@printf("In-flow Poiseuille setup: Nx=%d Ny=%d Nz=%d Fx=%g ν_total=%g β=%g λ=%g Wi_wall≈%.3f steps=%d\n",
        NX, NY, NZ, FX, NU_TOTAL, BETA, LAMBDA, LAMBDA * GAMMA_WALL, MAX_STEPS)

# --- run + collect y-profiles -----------------------------------------
rows = Vector{NamedTuple}()
nearwall_trC = Dict{String,Float64}()

for (name, model) in models
    res = run_poiseuille(model)
    @assert res.completed_steps == MAX_STEPS
    @assert all(isfinite, res.profile) "non-finite velocity for $name"
    @assert all(isfinite, res.Cxx_prof) && all(isfinite, res.Cyy_prof) &&
            all(isfinite, res.Czz_prof) "non-finite conformation for $name"

    trC = res.Cxx_prof .+ res.Cyy_prof .+ res.Czz_prof
    # Near-wall station (j=4) is the high-shear band where models separate;
    # centre (j=Ny÷2+1) is the low-shear equilibrium band (tr C → 3).
    near_j = 4
    nearwall_trC[name] = trC[near_j]
    @printf("  %-9s  trC[wall j=%d]=%8.4f  trC[centre]=%7.4f  N1_peak=%8.4f  u_max=%9.5f\n",
            name, near_j, trC[near_j], trC[NY ÷ 2 + 1],
            maximum(abs.(res.N1_prof)), maximum(res.profile))

    for j in 1:NY
        push!(rows, (; model=name, j=j,
                       u=res.profile[j],
                       gamma=res.gamma_dot_meas_prof[j],
                       Cxx=res.Cxx_prof[j], Cyy=res.Cyy_prof[j],
                       Czz=res.Czz_prof[j], Cxy=res.Cxy_prof[j],
                       trC=trC[j], N1=res.N1_prof[j]))
    end
end

# --- near-wall ranking (the headline) ---------------------------------
println("\n== near-wall (j=4) tr C ranking ==")
ranked = sort(collect(nearwall_trC); by=p -> -p[2])
for (name, v) in ranked
    @printf("  %-9s  trC=%8.4f\n", name, v)
end

# --- write CSV --------------------------------------------------------
out = joinpath(HERE, "ve_poiseuille_profiles.csv")
open(out, "w") do io
    println(io, "# Kraken viscoelastic constitutive models — COUPLED planar Poiseuille y-profiles")
    @printf(io, "# Nx=%d Ny=%d Nz=%d Fx=%g nu_total=%g beta=%g lambda=%g Wi_wall=%.4f steps=%d\n",
            NX, NY, NZ, FX, NU_TOTAL, BETA, LAMBDA, LAMBDA * GAMMA_WALL, MAX_STEPS)
    println(io, "# models: oldroydb, fenep(L2=50), giesekus(alpha=0.2), ptt(eps=0.25,linear)")
    println(io, "# j is the wall-normal row 1..Ny (1 and Ny touch the no-slip walls)")
    println(io, "model,j,u,gamma,Cxx,Cyy,Czz,Cxy,trC,N1")
    for r in rows
        @printf(io, "%s,%d,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g\n",
                r.model, r.j, r.u, r.gamma, r.Cxx, r.Cyy, r.Czz, r.Cxy, r.trC, r.N1)
    end
end
println("\nwrote $(out)  ($(length(rows)) rows)")
