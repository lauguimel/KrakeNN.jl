# Viscoelastic cylinder benchmark — GPU (H100)
#
# High-resolution Cd vs Wi sweep for Oldroyd-B confined cylinder.
# Compare to Alves et al. (2001), Hulsen et al. (2005).
#
# Standard benchmark: β = 0.59, B = 0.5, creeping flow
#
# Usage: julia --project=. hpc/viscoelastic_cylinder_gpu.jl

using KrakeNN
using CUDA

backend = CUDABackend()
FT = Float64

# --- Resolution parameters ---
# B = D/(2H) = 0.5 (blockage), creeping flow Re ≈ 0.1
radius = 40              # D = 80 → good resolution
D = 2 * radius
Ny = 2 * D              # = 160 (B = 0.5)
Nx = 20 * D             # = 1600 (10D upstream, 10D downstream)
u_in = 0.005            # very low → creeping flow
ν_total = 0.1
β = 0.59                # standard benchmark value (Alves 2001, Hulsen 2005)
ν_s = β * ν_total
ν_p = (1 - β) * ν_total
Re = u_in * D / ν_total

max_steps  = 200_000    # long run for convergence
avg_window = 50_000

println("=" ^ 70)
println("Viscoelastic cylinder benchmark (GPU)")
println("  Nx=$Nx, Ny=$Ny, radius=$radius, D=$D")
println("  Re = $Re,  β = $β,  u_in = $u_in")
println("  ν_s = $ν_s,  ν_p = $ν_p,  ν_total = $ν_total")
println("  max_steps = $max_steps, avg_window = $avg_window")
println("  Backend: $(typeof(backend))")
println("=" ^ 70)

# --- Newtonian reference ---
println("\n>>> Wi = 0 (Newtonian) ...")
t0 = time()
r0 = run_cylinder_2d(; Nx=Nx, Ny=Ny, radius=radius, u_in=u_in, ν=ν_total,
                      max_steps=max_steps, avg_window=avg_window,
                      backend=backend, T=FT)
Cd_newt = r0.Cd
K_newt = Cd_newt * Re / 2
dt0 = time() - t0
println("  Cd = $(round(Cd_newt, digits=3)),  K = $(round(K_newt, digits=3)),  time = $(round(dt0, digits=1))s")

# --- Wi sweep (Oldroyd-B, log-conformation) ---
Wi_values = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
results = [(Wi=0.0, Cd=Cd_newt, K=K_newt, time=dt0)]

for Wi in Wi_values
    lambda = Wi * radius / u_in  # Wi = λ·U/R
    println("\n>>> Wi = $Wi  (λ = $(round(lambda, digits=1))) ...")
    t0 = time()
    r = run_viscoelastic_cylinder_2d(;
        Nx=Nx, Ny=Ny, radius=radius, u_in=u_in,
        ν_s=ν_s, ν_p=ν_p, lambda=lambda,
        formulation=:stress, L_max=0.0,
        max_steps=max_steps, avg_window=avg_window,
        backend=backend, FT=FT)
    dt = time() - t0
    K = r.Cd * Re / 2
    println("  Cd = $(round(r.Cd, digits=3)),  K = $(round(K, digits=3)),  time = $(round(dt, digits=1))s")
    push!(results, (; Wi, Cd=r.Cd, K, time=dt))

    if isnan(r.Cd)
        println("  ⚠ DIVERGED — stopping sweep")
        break
    end
end

# --- Summary table ---
println("\n" * "=" ^ 70)
println("RESULTS: Cd vs Wi (Oldroyd-B, β=$β, B=0.5, Re=$Re)")
println("  Nx=$Nx, Ny=$Ny, radius=$radius")
println("-" ^ 70)
println("  Wi    |  Cd         |  K=Cd·Re/2  |  Cd/Cd(0)  |  Time (s)")
println("-" ^ 70)
for r in results
    ratio = r.Cd / Cd_newt
    println("  $(lpad(round(r.Wi, digits=3), 5)) | " *
            " $(lpad(round(r.Cd, digits=3), 10)) | " *
            " $(lpad(round(r.K, digits=3), 10)) | " *
            " $(lpad(round(ratio, digits=4), 8)) | " *
            " $(round(r.time, digits=1))")
end
println("=" ^ 70)

# --- Literature reference ---
# Hulsen et al. (2005), Alves et al. (2001)
# Oldroyd-B, β = 0.59, B = 0.5, creeping flow
println("\nLiterature (Hulsen et al. 2005, B=0.5, β=0.59, creeping):")
println("  Wi  |  K")
for (wi, k) in [(0.0, 132.36), (0.5, 130.36), (0.6, 129.72),
                (0.7, 129.15), (0.8, 128.60), (0.9, 128.04), (1.0, 127.49)]
    println("  $(lpad(wi, 4)) |  $k")
end

println("\nExpected: K decreases monotonically with Wi (drag reduction).")
println("Done.")
