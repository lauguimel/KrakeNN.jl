#!/usr/bin/env julia
# 3D LBM reference ("truth") for a GNF flow in a duct with a voxel obstacle.
# Everything is set by environment variables (defaults in `_duct_config`);
# see README.md. Writes raw Float64 fields + case.toml in the layout of the
# 2D E5 export (tmp/e5_export/<grid>/): little-endian, column-major, x fastest,
# `<name>.bin` read with reshape(reinterpret(Float64, read(path)), Nx, Ny, Nz).
using KrakeNN
using KernelAbstractions
using TOML
using Dates

_envs(name, default) = get(ENV, name, default)
_envi(name, default) = parse(Int, _envs(name, string(default)))
_envf(name, default) = parse(Float64, _envs(name, string(default)))
_envlist(name, default) = parse.(Float64, split(_envs(name, default), ','))

const _DUCT_BACKEND = lowercase(_envs("DUCT_BACKEND", "cuda"))
# CUDA is loaded at top level (never inside a function: world-age).
if _DUCT_BACKEND == "cuda"
    using CUDA
end

function _duct_config()
    return (;
        Nx=_envi("DUCT_NX", 128), Ny=_envi("DUCT_NY", 64), Nz=_envi("DUCT_NZ", 64),
        law=lowercase(_envs("DUCT_LAW", "carreau")),
        # Lattice-unit law parameters (nu in h^2/dt, lambda/K in dt).
        nu0=_envf("DUCT_NU0", 0.1), nuinf=_envf("DUCT_NUINF", 0.02),
        lambda=_envf("DUCT_LAMBDA", 3000.0), a=_envf("DUCT_A", 2.0),
        n=_envf("DUCT_N", 0.5), K=_envf("DUCT_K", 0.1),
        numin=_envf("DUCT_NUMIN", 1e-3), numax=_envf("DUCT_NUMAX", 1.0),
        F=_envf("DUCT_F", 1e-5),
        steps=_envi("DUCT_STEPS", 200_000), tol=_envf("DUCT_TOL", 1e-9),
        check=_envi("DUCT_CHECK_EVERY", 500),
        obstacle=lowercase(_envs("DUCT_OBSTACLE", "cuboid")),
        # Obstacle centre and size as fractions of (Lx, Ly, Lz) (cuboid) or
        # centre fractions and diameter as a fraction of min(Ly, Lz) (sphere).
        obs_center=_envlist("DUCT_OBS_CENTER", "0.5,0.5,0.5"),
        obs_size=_envlist("DUCT_OBS_SIZE", "0.15,0.4,0.4"),
        FT=_envs("DUCT_FT", "Float64") == "Float32" ? Float32 : Float64,
        h=_envf("DUCT_H", 1.0), dt=_envf("DUCT_DT", 1.0),
        outdir=abspath(_envs("DUCT_OUT", "duct_out")))
end

function _duct_rheology(c)
    c.law == "newtonian" && return Newtonian(c.nu0)
    c.law == "carreau" && return CarreauYasuda(c.nu0, c.nuinf, c.lambda, c.a, c.n)
    c.law == "cross" && return Cross(c.nu0, c.nuinf, c.K, c.n)
    c.law == "powerlaw" && return PowerLaw(c.K, c.n; nu_min=c.numin, nu_max=c.numax)
    error("DUCT_LAW must be newtonian, carreau, cross or powerlaw; got $(c.law)")
end

# Mask as a function of the cell centre in units of h (see run_gnf_duct_obstacle_3d).
function _duct_mask(c)
    c.obstacle == "none" && return nothing
    Lx, Ly, Lz = c.Nx, c.Ny - 1 + 1.0, c.Nz - 1 + 1.0   # q_w = 1/2: Ly = Ny, Lz = Nz
    cx, cy, cz = c.obs_center .* (Lx, Ly, Lz)
    if c.obstacle == "cuboid"
        sx, sy, sz = c.obs_size .* (Lx, Ly, Lz) ./ 2
        return (x, y, z) -> abs(x - cx) < sx && abs(y - cy) < sy && abs(z - cz) < sz
    elseif c.obstacle == "sphere"
        r = c.obs_size[1] * min(Ly, Lz) / 2
        return (x, y, z) -> (x - cx)^2 + (y - cy)^2 + (z - cz)^2 < r^2
    end
    error("DUCT_OBSTACLE must be none, cuboid or sphere; got $(c.obstacle)")
end

_write_bin(path, a) = open(io -> write(io, Float64.(vec(a))), path, "w")

function _duct_main()
    c = _duct_config()
    backend = if _DUCT_BACKEND == "cuda"
        CUDA.functional() || error("A functional CUDA device is required; set DUCT_BACKEND=cpu for a CPU run")
        CUDA.CUDABackend()
    elseif _DUCT_BACKEND == "cpu"
        any(id.name == "CUDA" for id in keys(Base.loaded_modules)) &&
            error("CPU run must not load CUDA")
        KernelAbstractions.CPU()
    else
        error("DUCT_BACKEND must be cuda or cpu")
    end
    mkpath(c.outdir)
    rheology = _duct_rheology(c)
    println("config: ", c)
    wall_s = @elapsed out = run_gnf_duct_obstacle_3d(; rheology, Nx=c.Nx, Ny=c.Ny,
        Nz=c.Nz, F=c.F, max_steps=c.steps, mask=_duct_mask(c), tol_ss=c.tol,
        check_every=c.check, h=c.h, dt=c.dt, backend, FT=c.FT)
    for (name, a) in (("ux", out.ux), ("uy", out.uy), ("uz", out.uz),
                      ("nu", out.nu), ("rho", out.rho), ("is_solid", out.mask))
        _write_bin(joinpath(c.outdir, name * ".bin"), a)
    end
    nfluid = count(!, out.mask)
    mf = out.mass_flux_planes
    summary = Dict(
        "grid" => "$(c.Nx)x$(c.Ny)x$(c.Nz)",
        "description" => "3D LBM (D3Q19 TRT, Guo force, LI-BB) GNF duct with voxel obstacle, steady state; reference for FV comparison",
        "provenance" => Dict("julia_version" => string(VERSION),
            "backend" => string(typeof(backend)), "created_at" => string(now()),
            "driver" => "run_gnf_duct_obstacle_3d", "script" => "benchmarks/lbm3d_duct/run_duct_gpu.jl"),
        "format" => Dict(
            "dtype" => "Float64 little-endian, raw, no header",
            "shape" => [c.Nx, c.Ny, c.Nz],
            "order" => "Julia column-major (x fastest): A[i,j,k] at offset (i-1)+(j-1)*Nx+(k-1)*Nx*Ny; numpy: np.fromfile(f,'<f8').reshape((Nx,Ny,Nz),order='F')",
            "units" => "ux, uy, uz, nu, rho in lattice units (multiply by the *_scale entries of [constants]); is_solid 1.0 = obstacle voxel, 0.0 = fluid",
            "solid_cells" => "ux, uy, uz, nu are 0 in obstacle voxels",
            "index" => "i = x (periodic flow direction), j = y, k = z; cell centre ((i-1/2)dx, (j-1+q_w)dx, (k-1+q_w)dx), duct walls at y=0, Ly, z=0, Lz",
            "velocity" => "cell-centred physical velocity, Guo half-force included; nu from the last-step TRT rate"),
        "law" => Dict("name" => c.law, "parameters_lattice" => Dict(
            "nu0" => c.nu0, "nuinf" => c.nuinf, "lambda" => c.lambda, "a" => c.a,
            "n" => c.n, "K" => c.K, "nu_min" => c.numin, "nu_max" => c.numax)),
        "obstacle" => Dict("kind" => c.obstacle, "center_fraction" => c.obs_center,
            "size_fraction" => c.obs_size, "solid_voxels" => count(out.mask)),
        "constants" => Dict(
            "Fx" => out.F, "Fx_unit" => "lattice units (dx/dt^2), uniform body force per unit mass in +x",
            "q_w" => out.q_w, "dx" => out.scaling.h, "dt" => out.scaling.dt,
            "velocity_scale" => out.scaling.velocity, "nu_scale" => out.scaling.nu,
            "force_scale" => out.scaling.acceleration, "shear_rate_scale" => out.scaling.shear_rate,
            "flow_rate_scale" => out.scaling.flow_rate,
            "Nx" => c.Nx, "Ny" => c.Ny, "Nz" => c.Nz, "Hy" => out.Hy, "Hz" => out.Hz,
            "physical_Lx_m" => out.physical.Lx, "physical_Ly_m" => out.physical.Ly,
            "physical_Lz_m" => out.physical.Lz),
        "results" => Dict(
            "flow_rate_lattice" => out.flow_rate, "flow_rate_physical" => out.physical.flow_rate,
            "u_bulk_lattice" => out.u_bulk, "u_max_lattice" => out.u_max,
            "Mach_max" => out.u_max * sqrt(3),
            "nu_min_lattice" => minimum(out.nu[.!out.mask]), "nu_max_lattice" => maximum(out.nu[.!out.mask]),
            "mass_flux_plane_spread" => (maximum(mf) - minimum(mf)) / maximum(mf),
            "fluid_cells" => nfluid),
        "convergence" => Dict("steps" => out.steps, "converged" => out.converged,
            "relative_change" => out.convergence, "tol_ss" => c.tol,
            "check_every" => c.check, "wall_seconds" => wall_s,
            "criterion" => "max change of any velocity component over check_every steps <= tol_ss * max speed"))
    open(io -> TOML.print(io, summary), joinpath(c.outdir, "case.toml"), "w")
    println("steps=$(out.steps) converged=$(out.converged) Q=$(out.flow_rate) umax=$(out.u_max) wall_s=$(round(wall_s; digits=2))")
    println("wrote $(c.outdir)")
    return nothing
end

_duct_main()
