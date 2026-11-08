# =============================================================================
# Schäfer-Turek 2D-2 benchmark (unsteady cylinder, Re=100, parabolic inflow)
#
# Reference (Schäfer & Turek 1996):
#   Cd_max  = 3.22 .. 3.24
#   Cl_max  = 0.98 .. 1.02
#   Strouhal St = f D / U_mean = 0.295 .. 0.305
#
# Same geometry as 2D-1, but U_max = 1.5 m/s (U_mean = 1.0), ν = 10⁻³,
# Re = 100. Flow is periodic vortex shedding after a ~5T transient.
#
# Measurement protocol:
#   1. Run ~20 shedding periods (T_lu = D_lu / (St · u_mean_lu) ≈ 3300 for
#      D=40, so ~65k steps).
#   2. Discard first ~5 periods (transient).
#   3. Record Cd(t), Cl(t) each step in the measurement window.
#   4. Cd_mean, Cd_amp, Cl_amp from time series.
#   5. Strouhal from FFT of Cl(t).
# =============================================================================

using Pkg; Pkg.activate(dirname(@__DIR__))
using KrakeNN
using KrakeNN: fused_trt_libb_v2_step!, compute_drag_libb_mei_2d,
              precompute_q_wall_cylinder, D2Q9,
              apply_bc_rebuild_2d!, BCSpec2D, ZouHeVelocity, NeqExtrapolationPressure
using Metal, KernelAbstractions
using FFTW

backend = Metal.MetalBackend()
T = Float32

# Parabolic parabolic convention (Schäfer-Turek 2D-2):
#   u(y) = 4 U_max y(H-y)/H², U_max = 1.5 (phys), ν = 1e-3, D = 0.1
#   U_mean = (2/3) U_max = 1.0, Re = U_mean D / ν = 100

D_lu = 40
Ny = Int(round(0.41 / 0.1 * D_lu))     # ≈ 164
Nx = Int(round(2.2 / 0.1 * D_lu))      # = 880
cx = 2 * D_lu
cy = 2 * D_lu
radius = D_lu ÷ 2

u_max  = 0.06                          # Ma ≈ 0.1
u_mean = (2/3) * u_max                 # = 0.04
Re     = 100.0
ν      = u_mean * D_lu / Re            # = 0.016

# Expected period (from literature St ≈ 0.30):
St_ref = 0.30
T_period = D_lu / (St_ref * u_mean)    # ≈ 3333 lattice steps
transient_periods = 5
measure_periods   = 15
max_steps = Int(round((transient_periods + measure_periods) * T_period))
measure_start = Int(round(transient_periods * T_period))

println("ST 2D-2: D_lu=$D_lu Nx×Ny=$(Nx)×$(Ny) ν=$ν max_steps=$max_steps measure_start=$measure_start")

q_wall_h, is_solid_h = precompute_q_wall_cylinder(Nx, Ny, cx, cy, radius; FT=T)
u_prof_h = [T(4)*T(u_max)*T(j-1)*T(Ny-j)/T(Ny-1)^2 for j in 1:Ny]

q_wall   = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny)
uw_x     = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
uw_y     = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
f_in     = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
f_out    = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
ρ = KernelAbstractions.allocate(backend, T, Nx, Ny)
ux = KernelAbstractions.allocate(backend, T, Nx, Ny)
uy = KernelAbstractions.allocate(backend, T, Nx, Ny)
u_prof = KernelAbstractions.allocate(backend, T, Ny)
copyto!(q_wall, q_wall_h); copyto!(is_solid, is_solid_h)
fill!(uw_x, zero(T)); fill!(uw_y, zero(T))
copyto!(u_prof, u_prof_h)
f_in_h = zeros(T, Nx, Ny, 9)
for j in 1:Ny, i in 1:Nx, q in 1:9
    f_in_h[i,j,q] = KrakeNN.equilibrium(D2Q9(), one(T), u_prof_h[j], zero(T), q)
end
copyto!(f_in, f_in_h); fill!(f_out, zero(T))
fill!(ρ, one(T)); fill!(ux, zero(T)); fill!(uy, zero(T))

bcspec = BCSpec2D(; west = ZouHeVelocity(u_prof),
                    east = NeqExtrapolationPressure(T(1.0)))

n_rec = max_steps - measure_start + 1
Cd_t = Float64[]
Cl_t = Float64[]

u_ref = u_mean
D_phys = 2 * radius

t0 = time()
let f_in = f_in, f_out = f_out
    for step in 1:max_steps
        fused_trt_libb_v2_step!(f_out, f_in, ρ, ux, uy, is_solid,
                                 q_wall, uw_x, uw_y, Nx, Ny, T(ν))
        apply_bc_rebuild_2d!(f_out, f_in, bcspec, ν, Nx, Ny)
        if step >= measure_start
            drag = compute_drag_libb_mei_2d(f_out, q_wall, uw_x, uw_y, Nx, Ny)
            push!(Cd_t, 2 * drag.Fx / (u_ref^2 * D_phys))
            push!(Cl_t, 2 * drag.Fy / (u_ref^2 * D_phys))
        end
        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)
end
dt = time() - t0
mlups = max_steps * Nx * Ny / dt / 1e6

println("Runtime: $(round(dt, digits=1)) s ($(round(mlups, digits=1)) MLUPS)")

# Time-series statistics. The raw per-step MEA drag signal has
# high-frequency sampling noise (pop-level fluctuations) on top of the
# physical vortex-shedding oscillation. Apply a short moving-average
# (~1 step per 100 period) before extracting peak values.
function _smooth(x, w)
    n = length(x); out = similar(x, Float64)
    for i in 1:n
        a = max(1, i - w ÷ 2); b = min(n, i + w ÷ 2)
        out[i] = sum(Float64, @view x[a:b]) / (b - a + 1)
    end
    return out
end
smooth_w = max(Int(round(T_period / 100)), 20)
Cd_s = _smooth(Cd_t, smooth_w)
Cl_s = _smooth(Cl_t, smooth_w)

Cd_mean = sum(Cd_s) / length(Cd_s)
Cd_max  = maximum(Cd_s)
Cl_max  = maximum(abs.(Cl_s))

# Strouhal: FFT of Cl
N = length(Cl_t)
F = abs.(fft(Cl_t .- sum(Cl_t)/N))
# Peak frequency (positive side only, skip DC)
f_peak_idx = argmax(F[2:N÷2]) + 1
f_peak_lu  = (f_peak_idx - 1) / N           # cycles per step
St = f_peak_lu * D_phys / u_mean

println()
println("=" ^ 72)
println("Schäfer-Turek 2D-2 results (ref: Cd_max=3.22-3.24, Cl_max=0.98-1.02,")
println("                           St = 0.295-0.305)")
println("=" ^ 72)
println("  D_lu      = $D_lu")
println("  Cd_mean   = $(round(Cd_mean, digits=3))")
println("  Cd_max    = $(round(Cd_max,  digits=3))")
println("  Cl_max    = $(round(Cl_max,  digits=3))")
println("  Strouhal  = $(round(St,      digits=3))")
println("  MLUPS     = $(round(mlups,   digits=1))")
