using KrakeNN, CUDA, KernelAbstractions

# ===========================================================================
# WP-3D-3 — sphere drag in a duct via SLBM 3D on Aqua H100 (CUDA Float64).
#
# Two paths benchmarked:
#   A. Uniform Cartesian: SLBM 3D + LI-BB sphere, isotropic mesh
#   B. Stretched box: SLBM 3D + LI-BB sphere with x-stretching
#      → fewer cells for the same near-body resolution
#
# For each run we report (N, Cd, MLUPS, runtime). Reference Cd ≈ 3.0 for
# Re=20 in a confined duct H/D=4 (Clift-Gauvin 2.84 free + ~5–10%
# blockage). Headline number: cell-count ratio at fixed Cd accuracy.
# ===========================================================================

FT = Float64

# Re=20 sphere physical parameters (lattice units derived from D).
# We use the parabolic-inlet PEAK velocity as the reference everywhere:
#   Re = u_max · D / ν   and   Cd = 2·F / (u_max² · A)
# This keeps Re and Cd consistent and matches the Clift-Gauvin Cd∞=2.84
# free-stream reference at Re=20 (after small confinement correction).
const Re_ref  = 20.0
const u_in_lu = 0.04                    # PEAK inlet velocity (lattice)
const ρ_out   = 1.0
const Cd_ref  = 2.84                    # Clift-Gauvin free-stream at Re=20

# Physical box: 12D long, 4D wide & tall, sphere centered at (3D, 2D, 2D)
function _box_geom(D)
    L_x = 12.0 * D
    L_yz = 4.0 * D
    cx = 3.0 * D; cy = L_yz / 2; cz = L_yz / 2
    R = D / 2
    return (; L_x, L_yz, cx, cy, cz, R)
end

function _build_geom(Nx, Ny, Nz, D; x_str=0.0)
    g = _box_geom(D)
    if x_str > 0
        mesh = stretched_box_mesh_3d(;
            x_min=0.0, x_max=g.L_x, y_min=0.0, y_max=g.L_yz,
            z_min=0.0, z_max=g.L_yz,
            Nx=Nx, Ny=Ny, Nz=Nz,
            x_stretch=x_str, x_stretch_dir=:left, FT=FT)
    else
        mesh = cartesian_mesh_3d(; x_min=0.0, x_max=g.L_x,
                                    y_min=0.0, y_max=g.L_yz,
                                    z_min=0.0, z_max=g.L_yz,
                                    Nx=Nx, Ny=Ny, Nz=Nz, FT=FT)
    end
    geom_h = build_slbm_geometry_3d(mesh; local_cfl=(x_str > 0))
    return mesh, geom_h, g
end

function _tag_solid(mesh, g)
    Nx, Ny, Nz = mesh.Nξ, mesh.Nη, mesh.Nζ
    is_solid_h = zeros(Bool, Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        dx = mesh.X[i,j,k] - g.cx
        dy = mesh.Y[i,j,k] - g.cy
        dz = mesh.Z[i,j,k] - g.cz
        if dx*dx + dy*dy + dz*dz ≤ g.R^2
            is_solid_h[i,j,k] = true
        end
    end
    return is_solid_h
end

function run_sphere_3d_slbm(; Nx, Ny, Nz, D, x_str=0.0, label,
                              steps=15_000, avg_window=4_000)
    mesh, geom_h, g = _build_geom(Nx, Ny, Nz, D; x_str=x_str)
    is_solid_h = _tag_solid(mesh, g)
    n_solid = count(is_solid_h)

    qw_h, uwx_h, uwy_h, uwz_h =
        precompute_q_wall_slbm_sphere_3d(mesh, is_solid_h, g.cx, g.cy, g.cz, g.R; FT=FT)
    n_cut = count(qw_h .> 0)

    # Re=20 calibration
    ν = u_in_lu * D / Re_ref

    # Local-tau on stretched, uniform on Cartesian
    use_local = x_str > 0
    sp_h, sm_h = compute_local_omega_3d(mesh; ν=ν, scaling=:quadratic)

    backend = CUDABackend()
    is_solid = CuArray(is_solid_h)
    q_wall   = CuArray(FT.(qw_h))
    uw_x     = CuArray(FT.(uwx_h)); uw_y = CuArray(FT.(uwy_h)); uw_z = CuArray(FT.(uwz_h))

    geom = transfer_slbm_geometry_3d(geom_h, backend)
    sp = use_local ? CuArray(FT.(sp_h)) : nothing
    sm = use_local ? CuArray(FT.(sm_h)) : nothing

    # Inlet profile (parabolic in y and z to avoid corner singularity)
    Hy = FT(Ny - 1); Hz = FT(Nz - 1)
    u_prof_h = zeros(FT, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny
        yy = FT(j - 1); zz = FT(k - 1)
        u_prof_h[j,k] = FT(16) * FT(u_in_lu) *
                        yy * (Hy - yy) * zz * (Hz - zz) /
                        (Hy^2 * Hz^2)
    end
    u_max = maximum(u_prof_h)                    # = u_in_lu by construction
    u_ref = Float64(u_in_lu)                     # peak velocity, matches Re definition
    u_prof = CuArray(u_prof_h)
    bcspec = BCSpec3D(; west = ZouHeVelocity(u_prof),
                        east = ZouHePressure(FT(ρ_out)))

    # Init populations to local equilibrium with the inlet profile (and zero in solid)
    f_init = zeros(FT, Nx, Ny, Nz, 19)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx, q in 1:19
        if is_solid_h[i,j,k]
            f_init[i,j,k,q] = FT(KrakeNN.equilibrium(D3Q19(), 1.0, 0.0, 0.0, 0.0, q))
        else
            f_init[i,j,k,q] = FT(KrakeNN.equilibrium(D3Q19(), 1.0,
                                                     Float64(u_prof_h[j,k]),
                                                     0.0, 0.0, q))
        end
    end
    f_in  = CuArray(f_init); f_out = similar(f_in)
    ρ  = CuArray(ones(FT, Nx, Ny, Nz))
    ux = CuArray(zeros(FT, Nx, Ny, Nz))
    uy = CuArray(zeros(FT, Nx, Ny, Nz))
    uz = CuArray(zeros(FT, Nx, Ny, Nz))

    Fx_sum = 0.0; n_avg = 0
    t0 = time()
    for step in 1:steps
        if use_local
            slbm_trt_libb_step_local_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                           q_wall, uw_x, uw_y, uw_z, geom, sp, sm)
            apply_bc_rebuild_3d!(f_out, f_in, bcspec, ν, Nx, Ny, Nz;
                                 sp_field=sp, sm_field=sm,
                                 apply_transverse=true)
        else
            slbm_trt_libb_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                    q_wall, uw_x, uw_y, uw_z, geom, ν)
            apply_bc_rebuild_3d!(f_out, f_in, bcspec, ν, Nx, Ny, Nz;
                                 apply_transverse=true)
        end
        if step > steps - avg_window
            drag = compute_drag_libb_3d(f_out, q_wall, Nx, Ny, Nz)
            Fx_sum += drag.Fx
            n_avg += 1
        end
        f_in, f_out = f_out, f_in
    end
    CUDA.synchronize()
    elapsed = time() - t0
    cells = Nx * Ny * Nz
    mlups = cells * steps / elapsed / 1e6

    Fx_avg = Fx_sum / n_avg
    A = π * (D/2)^2
    Cd = 2.0 * Fx_avg / (u_ref^2 * A)
    err = 100 * abs(Cd - Cd_ref) / Cd_ref

    println(rpad(label, 30), " ", lpad("$Nx×$Ny×$Nz", 14),
            " (", lpad(cells, 9), " cells)",
            " Cd=", round(Cd, digits=3),
            " err=", round(err, digits=2), "%",
            " MLUPS=", round(mlups, digits=0),
            " (", round(elapsed, digits=1), "s, ",
            n_solid, " solid, ", n_cut, " cuts)")
    return (; label, Nx, Ny, Nz, cells, Cd, err, mlups, elapsed)
end

println("=== WP-3D-3 — Sphere Re=20 in a duct (CUDA Float64) ===\n")
println("Reference: Cd ≈ $Cd_ref (Clift-Gauvin + ~5% confinement, H/D=4)\n")

println("--- Uniform Cartesian convergence ---")
run_sphere_3d_slbm(Nx=121, Ny=41, Nz=41, D=10.0, label="Uniform D=10")
run_sphere_3d_slbm(Nx=241, Ny=81, Nz=81, D=20.0, label="Uniform D=20")
run_sphere_3d_slbm(Nx=361, Ny=121, Nz=121, D=30.0, label="Uniform D=30")

# NOTE: stretched_box_mesh_3d with x_stretch_dir=:left clusters cells
# toward x=0 (the inlet), but the sphere sits at x=3D (25% of the
# channel). The dense cells are therefore at the wrong location, which
# explains the under-prediction of Cd on the stretched runs below. A
# 3D port of `cylinder_focused_mesh` (clustering around an interior
# point) is required to make stretched-mesh comparisons fair; deferred
# to v0.2. For now the uniform runs above are the publishable numbers.
println("\n--- Stretched (local-CFL, illustrative — see NOTE above) ---")
run_sphere_3d_slbm(Nx=181, Ny=61, Nz=61, D=20.0, x_str=0.5, label="Stretch D=20 s=0.5")
run_sphere_3d_slbm(Nx=181, Ny=61, Nz=61, D=20.0, x_str=1.0, label="Stretch D=20 s=1.0")
run_sphere_3d_slbm(Nx=241, Ny=81, Nz=81, D=30.0, x_str=0.5, label="Stretch D=30 s=0.5")
run_sphere_3d_slbm(Nx=241, Ny=81, Nz=81, D=30.0, x_str=1.0, label="Stretch D=30 s=1.0")

println("\n=== Done ===")
