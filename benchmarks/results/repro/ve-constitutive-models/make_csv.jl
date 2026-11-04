#!/usr/bin/env julia
# =====================================================================
# make_csv.jl — analytical comparison of the 4 Kraken viscoelastic
# constitutive models (Oldroyd-B, FENE-P, Giesekus, PTT) in two
# homogeneous flows, using the REAL Kraken log-conformation constitutive
# solver (no hardcoded model formulas).
#
#   Page : docs/src/users/benchmarks/ve-constitutive-models.md
#   Out  : ve_constitutive_models.csv  (next to this script)
#
# For each model we drive a single conformation cell with an imposed,
# spatially-uniform velocity gradient and integrate the constitutive ODE
# (the same RK2 / sub-stepped log-conformation kernels the 3D drivers
# call: `logfv_constitutive_step_log_{,_fenep,_giesekus,_ptt}_3d!`) to its
# steady fixed point, then reconstruct C = exp(Ψ) via `mat_exp_sym3x3`.
# This is exactly the single-cell pattern of the constitutive canaries
# test/test_fvfd_{logconf,fenep,giesekus,ptt}_3d.jl — so the figure shows
# what Kraken actually computes, not a closed form.
#
#   Simple shear     : gradU = γ̇·(e_x ⊗ e_y),  Wi = λγ̇  ∈ sweep
#       → N1 = G(C_xx − C_yy),  η_p,app = G·C_xy/γ̇  (here G=1, λ=1)
#   Planar extension : gradU = ε̇·diag(1,−1,0),  λε̇ ∈ [0, 0.49]
#       → C_xx (OB diverges at λε̇=0.5: C_xx = 1/(1−2λε̇)), trace C
#
# Run (from repo root):
#   julia --project=. benchmarks/results/repro/ve-constitutive-models/make_csv.jl
# =====================================================================
using Kraken
using Printf

const HERE = @__DIR__

# --- single-cell fields (1,1,1) ---------------------------------------
field3(v) = fill(Float64(v), (1, 1, 1))

psi_state(; xx=0.0, xy=0.0, xz=0.0, yy=0.0, yz=0.0, zz=0.0) =
    (field3(xx), field3(xy), field3(xz), field3(yy), field3(yz), field3(zz))

velocity_gradient(; duxdx=0.0, duxdy=0.0, duxdz=0.0,
                    duydx=0.0, duydy=0.0, duydz=0.0,
                    duzdx=0.0, duzdy=0.0, duzdz=0.0) =
    (field3(duxdx), field3(duxdy), field3(duxdz),
     field3(duydx), field3(duydy), field3(duydz),
     field3(duzdx), field3(duzdy), field3(duzdz))

psi_values(psi) = ntuple(c -> psi[c][1, 1, 1], 6)
conformation_from_psi(psi) = Kraken.mat_exp_sym3x3(psi_values(psi)...)

# --- one constitutive step per model (dispatch on the model spec) -----
# Each writes Ψ_out in place; returns max-abs Ψ increment for convergence.
function step!(psi, grad, lambda, dt, n_sub, model)
    before = psi_values(psi)
    out = ntuple(c -> similar(psi[c]), 6)
    if model isa LogConfOldroydB
        Kraken.logfv_constitutive_step_log_3d!(
            out..., psi..., grad..., lambda, dt, n_sub; sync=true)
    elseif model isa LogConfFENEP
        Kraken.logfv_constitutive_step_log_fenep_3d!(
            out..., psi..., grad..., lambda, dt, model.L², n_sub; sync=true)
    elseif model isa LogConfGiesekus
        Kraken.logfv_constitutive_step_log_giesekus_3d!(
            out..., psi..., grad..., lambda, dt, model.α, n_sub; sync=true)
    elseif model isa LogConfPTT
        Kraken.logfv_constitutive_step_log_ptt_3d!(
            out..., psi..., grad..., lambda, dt, model.ε, n_sub;
            variant=model.variant, sync=true)
    else
        error("unhandled model $(typeof(model))")
    end
    for c in 1:6
        psi[c] .= out[c]
    end
    after = psi_values(psi)
    return maximum(abs(after[c] - before[c]) for c in 1:6)
end

# --- integrate a homogeneous flow to its constitutive fixed point -----
function run_to_steady(model, grad; lambda=1.0, dt=0.02)
    psi = psi_state()
    max_grad_norm = Kraken.logfv_max_grad_norm_3d(grad...)
    n_sub = 8 * Kraken.logfv_recommended_oldroydb_substeps_3d(max_grad_norm, lambda, dt)
    max_steps = ceil(Int, 400lambda / dt)
    delta = Inf
    steps = 0
    while steps < max_steps && delta >= 1e-13
        delta = step!(psi, grad, lambda, dt, n_sub, model)
        steps += 1
    end
    C = conformation_from_psi(psi)          # (Cxx,Cxy,Cxz,Cyy,Cyz,Czz)
    trC = C[1] + C[4] + C[6]
    return (; C, trC, steps, delta, n_sub)
end

# G=1, λ=1 ⇒ Wi=γ̇ and λε̇=ε̇ directly; N1=C_xx−C_yy, η_p,app=C_xy/γ̇.
const LAMBDA = 1.0

# The 4 validated models at representative non-Newtonian parameters.
models = [
    ("oldroydb", LogConfOldroydB(G=1.0, λ=LAMBDA)),
    ("fenep",    LogConfFENEP(G=1.0, λ=LAMBDA, L²=50.0)),
    ("giesekus", LogConfGiesekus(G=1.0, λ=LAMBDA, α=0.2)),
    ("ptt",      LogConfPTT(G=1.0, λ=LAMBDA, ε=0.25, variant=:linear)),
]

# --- sweeps -----------------------------------------------------------
# Simple shear: Wi = λγ̇.
wi_shear = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0,
            3.0, 4.0, 5.0, 7.0, 10.0]
# Planar extension: λε̇ ∈ [0, 0.49] (OB pole at 0.5).
le_ext = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
          0.43, 0.45, 0.47, 0.48, 0.49]

rows = Vector{NamedTuple}()

println("== simple shear (Wi = λγ̇) ==")
for (name, model) in models
    for Wi in wi_shear
        gdot = Wi / LAMBDA
        grad = velocity_gradient(duxdy=gdot)
        r = run_to_steady(model, grad; lambda=LAMBDA)
        Cxx, Cxy, _, Cyy, _, _ = r.C
        N1 = Cxx - Cyy                          # G(Cxx−Cyy), G=1
        eta_p = Wi == 0 ? (1.0 - 0.0) : Cxy / gdot  # η_p,app = G·Cxy/γ̇
        push!(rows, (; flow="shear", model=name, control=Wi,
                       Cxx, Cxy, Cyy, Czz=r.C[6], trC=r.trC,
                       N1, eta_p, steps=r.steps))
        @printf("  %-9s Wi=%5.2f  N1=%10.4f  Cxy=%8.4f  ηp=%7.4f  trC=%8.4f\n",
                name, Wi, N1, Cxy, eta_p, r.trC)
    end
end

println("== planar extension (λε̇) ==")
for (name, model) in models
    for le in le_ext
        edot = le / LAMBDA
        grad = velocity_gradient(duxdx=edot, duydy=-edot)
        r = run_to_steady(model, grad; lambda=LAMBDA)
        Cxx, Cxy, _, Cyy, _, Czz = r.C
        push!(rows, (; flow="extension", model=name, control=le,
                       Cxx, Cxy, Cyy, Czz, trC=r.trC,
                       N1=Cxx - Cyy, eta_p=NaN, steps=r.steps))
        @printf("  %-9s λε̇=%5.2f  Cxx=%10.4f  trC=%10.4f  steps=%d\n",
                name, le, Cxx, r.trC, r.steps)
    end
end

# --- write CSV --------------------------------------------------------
out = joinpath(HERE, "ve_constitutive_models.csv")
open(out, "w") do io
    println(io, "# Kraken viscoelastic constitutive-model comparison (single-cell, real solver)")
    println(io, "# G=1, lambda=1; shear control=Wi=lambda*gammadot; extension control=lambda*edot")
    println(io, "# models: oldroydb, fenep(L2=50), giesekus(alpha=0.2), ptt(eps=0.25,linear)")
    println(io, "flow,model,control,Cxx,Cxy,Cyy,Czz,trC,N1,eta_p,steps")
    for r in rows
        @printf(io, "%s,%s,%.6g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%d\n",
                r.flow, r.model, r.control, r.Cxx, r.Cxy, r.Cyy, r.Czz,
                r.trC, r.N1, r.eta_p, r.steps)
    end
end
println("wrote $(out)  ($(length(rows)) rows)")
