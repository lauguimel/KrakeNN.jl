# Direct-call smoke for the 3D Oldroyd-B confined-sphere driver
# (run_conformation_sphere_libb_3d), ported onto release/v0.2.
#
# Tiny CPU Float64 config: R=4 sphere so the LI-BB cut links are genuinely
# exercised (a closed q_wall=0.5 box would collapse Bouzidi-FL to halfway-BB
# and hide cut-link bugs — see feedback_smoke_must_exercise_cutlinks). Low
# Wi=0.01, a few hundred steps. Mirrors the parameter relations of
# bench/viscoelastic_audit/step5_3d_diagnostic/5c_sphere_wi_sweep.jl, scaled
# down for a sub-minute CPU run.
#
# Asserts: runs NaN-free AND returns a finite drag Cd.

using Kraken, Printf

backend = Kraken.KernelAbstractions.CPU()
FT = Float64

R_s = 4
Nx = 12 * R_s            # 48
Ny = 4 * R_s             # 16
Nz = 4 * R_s             # 16
cx = 4 * R_s             # 16
cy = Ny ÷ 2             # 8
cz = Nz ÷ 2             # 8

β = 0.5
u_in = 0.02
ν_total = u_in * (2 * R_s) / 1.0     # Re ~ O(1)
ν_s = β * ν_total
ν_p = (1 - β) * ν_total

Wi = 0.01
λ  = Wi * R_s / u_in                  # = 2.0
m_OB = OldroydBSpec(G = ν_p / λ, λ = λ)

max_steps = 300
avg_window = 100

println("="^70)
println("Smoke: 3D Oldroyd-B sphere (LI-BB V2) — R=$R_s grid $(Nx)x$(Ny)x$(Nz), Wi=$Wi")
println("="^70)

t0 = time()
r = run_conformation_sphere_libb_3d(;
        Nx = Nx, Ny = Ny, Nz = Nz, radius = R_s,
        cx = cx, cy = cy, cz = cz,
        u_in = u_in, ν_s = ν_s,
        inlet = :uniform, ρ_out = 1.0, tau_plus = 1.0,
        polymer_bc = CNEBB(), polymer_model = m_OB,
        max_steps = max_steps, avg_window = avg_window,
        backend = backend, FT = FT)
dt = time() - t0

# --- Assertions ---
@assert !any(isnan, r.ρ)   "density field contains NaN"
@assert !any(isnan, r.ux)  "ux field contains NaN"
@assert !any(isnan, r.C_xx) "C_xx field contains NaN"
@assert isfinite(r.Cd)     "Cd is not finite: $(r.Cd)"
# Cut-link sanity: a sub-cell-radius sphere on this grid must have some
# fractional q_wall entries in (0,1) (true cut links), not all 0/0.5/1.
n_cut = count(q -> 0 < q < 1, r.q_wall)
@assert n_cut > 0          "no fractional cut links — sphere not resolved"

@printf("\nRESULT  Cd=%.6f  Re=%.4f  Wi=%.4f  beta=%.3f\n", r.Cd, r.Re, r.Wi, r.beta)
@printf("        fractional cut links = %d   wall time = %.1fs\n", n_cut, dt)
println("SMOKE_OK")
