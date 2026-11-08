using KrakeNN, Enzyme

# ===========================================================================
# SLBM AD proof-of-concept on Aqua (CPU only — Enzyme GPU is experimental).
#
# 1. Kinetic energy decay: dKE/dν on Taylor-Green-like perturbation
# 2. Shape derivative: dFx/dR on cylinder flow (the flagship result)
#
# Runs on CPU (faster compile, Enzyme mature) but on Aqua's fast cores.
# ===========================================================================

# --- Test 1: dKE/dν (kinetic energy vs viscosity) ---

function forward_ke(ν_val::Float64; Nx=40, Ny=30, steps=200)
    mesh = cartesian_mesh(; x_min=0.0, x_max=Float64(Nx-1),
                            y_min=0.0, y_max=Float64(Ny-1), Nx=Nx, Ny=Ny)
    geom = build_slbm_geometry(mesh)
    ω = 1.0 / (3.0 * ν_val + 0.5)

    f_in = zeros(Nx, Ny, 9); f_out = similar(f_in)
    ρ = ones(Nx, Ny); ux = zeros(Nx, Ny); uy = zeros(Nx, Ny)
    is_solid = zeros(Bool, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        x = (i-1.0)/(Nx-1); y = (j-1.0)/(Ny-1)
        u0 = 0.01 * sin(2π*x) * cos(2π*y)
        v0 = -0.01 * cos(2π*x) * sin(2π*y)
        for q in 1:9
            f_in[i,j,q] = KrakeNN.equilibrium(D2Q9(), 1.0, u0, v0, q)
        end
    end

    for step in 1:steps
        slbm_bgk_step!(f_out, f_in, ρ, ux, uy, is_solid, geom, ω)
        f_in, f_out = f_out, f_in
    end

    ke = 0.0
    for j in 1:Ny, i in 1:Nx
        ke += ρ[i,j] * (ux[i,j]^2 + uy[i,j]^2)
    end
    return ke
end

println("=== Test 1: dKE/dν on decaying Taylor-Green ===")
ν0 = 0.1
val = forward_ke(ν0)
println("KE(ν=$ν0) = $(round(val, sigdigits=8))")

eps = 1e-7
t_fd = time()
dke_fd = (forward_ke(ν0+eps) - forward_ke(ν0-eps)) / (2eps)
t_fd = time() - t_fd
println("dKE/dν (fin. diff.) = $(round(dke_fd, sigdigits=6))  [$(round(t_fd, digits=1))s]")

t_ad = time()
result = Enzyme.autodiff(Enzyme.Reverse, forward_ke, Active, Active(ν0))
dke_ad = result[1][1]
t_ad = time() - t_ad
println("dKE/dν (Enzyme)     = $(round(dke_ad, sigdigits=6))  [$(round(t_ad, digits=1))s]")
rel_err = abs(dke_ad - dke_fd) / max(abs(dke_fd), 1e-15)
println("Relative error      = $(round(100*rel_err, digits=4))%")

# --- Test 2: dFx/dR (drag vs cylinder radius) ---

function forward_drag(R_val::Float64; Nx=120, Ny=40, steps=10_000)
    cx = Float64(Nx)/4; cy = Float64(Ny)/2
    q_wall, is_solid = precompute_q_wall_cylinder(Nx, Ny, cx, cy, R_val)
    mesh = cartesian_mesh(; x_min=0.0, x_max=Float64(Nx-1),
                            y_min=0.0, y_max=Float64(Ny-1), Nx=Nx, Ny=Ny)
    geom = build_slbm_geometry(mesh)

    u_in = 0.04; ν = 0.04
    ω = 1.0 / (3.0*ν + 0.5)

    f_in = zeros(Nx, Ny, 9); f_out = similar(f_in)
    ρ = ones(Nx, Ny); ux = zeros(Nx, Ny); uy = zeros(Nx, Ny)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        u_prof = 4*u_in*(j-1)*(Ny-j)/(Ny-1)^2
        f_in[i,j,q] = KrakeNN.equilibrium(D2Q9(), 1.0, u_prof, 0.0, q)
    end

    for step in 1:steps
        slbm_bgk_step!(f_out, f_in, ρ, ux, uy, is_solid, geom, ω)
        for j in 2:Ny-1
            u_j = 4*u_in*(j-1)*(Ny-j)/(Ny-1)^2
            for q in 1:9
                f_out[1,j,q] = KrakeNN.equilibrium(D2Q9(), 1.0, u_j, 0.0, q)
            end
        end
        f_in, f_out = f_out, f_in
    end

    # Simple MEA drag
    Fx = 0.0
    cxs = KrakeNN.velocities_x(D2Q9())
    for j in 1:Ny, i in 1:Nx
        is_solid[i,j] && continue
        for q in 2:9
            q_wall[i,j,q] > 0 || continue
            qbar = q <= 5 ? (q <= 3 ? q+2 : q-2) :
                   (q == 6 ? 8 : q == 7 ? 9 : q == 8 ? 6 : 7)
            Fx += (f_in[i,j,q] + f_in[i,j,qbar]) * cxs[q]
        end
    end
    return Fx
end

println("\n=== Test 2: dFx/dR (shape derivative) ===")
R0 = 8.0
println("Computing Fx(R=$R0)...")
t = time()
Fx0 = forward_drag(R0)
println("Fx(R=$R0) = $(round(Fx0, sigdigits=6)) [$(round(time()-t, digits=1))s]")

eps = 0.1
println("Finite difference dFx/dR...")
t_fd = time()
dFx_fd = (forward_drag(R0+eps) - forward_drag(R0-eps)) / (2eps)
t_fd = time() - t_fd
println("dFx/dR (fin. diff.) = $(round(dFx_fd, sigdigits=4)) [$(round(t_fd, digits=1))s]")

println("Enzyme shape derivative...")
t_ad = time()
try
    result = Enzyme.autodiff(Enzyme.Reverse, forward_drag, Active, Active(R0))
    dFx_ad = result[1][1]
    t_ad = time() - t_ad
    println("dFx/dR (Enzyme)     = $(round(dFx_ad, sigdigits=4)) [$(round(t_ad, digits=1))s]")
    if abs(dFx_fd) > 1e-10
        rel = abs(dFx_ad - dFx_fd)/abs(dFx_fd)
        println("Relative error      = $(round(100*rel, digits=2))%")
    end
catch e
    s = sprint(showerror, e)
    println("Enzyme failed: ", s[1:min(end,500)])
end

println("\n=== Done ===")
