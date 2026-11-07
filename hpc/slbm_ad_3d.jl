using KrakeNN, Enzyme
using KernelAbstractions

# ===========================================================================
# WP-3D-5 — Enzyme AD on SLBM 3D (D3Q19, BGK, CPU).
#
# Taylor-Green-like vortex in 3D, decaying. We compute dKE/dν via:
#   1. Central finite differences (reference)
#   2. Enzyme.autodiff(Reverse) on the same forward
#
# Mirror of `slbm_ad_cuda.jl` (2D), now on D3Q19 — validates that the
# semi-Lagrangian 3D path is differentiable end-to-end.
# ===========================================================================

function forward_ke_3d(ν_val::Float64; Nx=24, Ny=24, Nz=24, steps=100)
    mesh = cartesian_mesh_3d(; x_min=0.0, x_max=Float64(Nx - 1),
                                y_min=0.0, y_max=Float64(Ny - 1),
                                z_min=0.0, z_max=Float64(Nz - 1),
                                Nx=Nx, Ny=Ny, Nz=Nz, FT=Float64)
    geom = build_slbm_geometry_3d(mesh; local_cfl=false)
    ω = 1.0 / (3.0 * ν_val + 0.5)

    f_in  = zeros(Nx, Ny, Nz, 19); f_out = similar(f_in)
    ρ     = ones(Nx, Ny, Nz)
    ux    = zeros(Nx, Ny, Nz)
    uy    = zeros(Nx, Ny, Nz)
    uz    = zeros(Nx, Ny, Nz)
    is_solid = zeros(Bool, Nx, Ny, Nz)

    # Divergence-free 3D Taylor-Green-like initial field
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        x = (i - 1) / (Nx - 1); y = (j - 1) / (Ny - 1); z = (k - 1) / (Nz - 1)
        u0 =  0.01 * sin(2π * x) * cos(2π * y) * cos(2π * z)
        v0 = -0.01 * cos(2π * x) * sin(2π * y) * cos(2π * z)
        w0 =  0.0
        for q in 1:19
            f_in[i, j, k, q] = KrakeNN.equilibrium(D3Q19(), 1.0, u0, v0, w0, q)
        end
    end

    for step in 1:steps
        slbm_bgk_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid, geom, ω)
        f_in, f_out = f_out, f_in
    end

    ke = 0.0
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ke += ρ[i, j, k] * (ux[i, j, k]^2 + uy[i, j, k]^2 + uz[i, j, k]^2)
    end
    return ke
end

println("=== WP-3D-5 — dKE/dν on 3D Taylor-Green decay (D3Q19) ===")
ν0 = 0.1

t0 = time()
val = forward_ke_3d(ν0)
println("KE(ν=$ν0) = $(round(val, sigdigits=8))   [forward $(round(time()-t0, digits=1))s]")

eps = 1e-7
t_fd = time()
ke_p = forward_ke_3d(ν0 + eps)
ke_m = forward_ke_3d(ν0 - eps)
dke_fd = (ke_p - ke_m) / (2eps)
t_fd = time() - t_fd
println("dKE/dν (fin. diff.) = $(round(dke_fd, sigdigits=6))   [$(round(t_fd, digits=1))s]")

println("Enzyme reverse-mode...")
let t_ad = time()
    try
        result = Enzyme.autodiff(Enzyme.Reverse, forward_ke_3d, Active, Active(ν0))
        dke_ad = result[1][1]
        t_ad = time() - t_ad
        println("dKE/dν (Enzyme)     = $(round(dke_ad, sigdigits=6))   [$(round(t_ad, digits=1))s]")
        if abs(dke_fd) > 1e-15
            rel_err = abs(dke_ad - dke_fd) / abs(dke_fd)
            println("Relative error      = $(round(100*rel_err, digits=4))%")
        end
    catch e
        s = sprint(showerror, e)
        println("Enzyme failed: ", s[1:min(end, 800)])
    end
end

println("\n=== Done ===")
