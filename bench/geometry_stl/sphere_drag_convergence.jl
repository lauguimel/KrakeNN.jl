# M-GEO-7 (b) — 3D STL sphere drag convergence toward the free-stream Clift
# (1978) Cd ≈ 2.6 at Re=20, on Aqua GPU (F64, CUDA).
#
# Strategy: fix R_LU=16 (D=32, half the registration sensitivity of R=8) and
# sweep the lateral blockage D/W from ~20% down to ~8% (larger laterally) so
# the confined Cd relaxes toward the unbounded free-stream value. A small R=8
# smoke case runs first to validate the pipeline (CUDA + the geometry path)
# before the expensive sweep.
#
# Each case: generate an STL sphere, write a .krk, run via run_simulation,
# read the frontal-area Cd. Output: bench/geometry_stl/sphere_drag_conv.csv.
#
# Memory (F64): the LI-BB driver holds ~6 arrays of N×19 ⇒ ~912 B/cell, so
# ~80 GB H100 caps near ~80M cells; the sweep tops out at ~61M (blk 8%).

using KrakeNN
using KernelAbstractions
using Printf

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(ROOT, "examples", "geometry_stl", "make_sphere_stl.jl"))

# --- backend: CUDA F64 (this benchmark is GPU-only; fail fast otherwise) ---
@eval using CUDA
@assert CUDA.functional() "CUDA not functional — sphere_drag_convergence requires a GPU node"
const BACKEND = CUDA.CUDABackend()
const FT = Float64

# --- case list: (label, R_LU, Nx, Nyz, steps) ; sphere centred at (Nx÷3, Nyz÷2, Nyz÷2) ---
# Re=20 ⇒ nu = u_in*2R/Re, u_in=0.04 fixed.
const U_IN = 0.04
cases = [
    # smoke first (cheap) — validates CUDA + the path
    (label="smoke_R8",   R=8,  Nx=160, Nyz=80,  steps=8000),
    # R=16 blockage sweep toward free-stream (D=32)
    (label="R16_blk20",  R=16, Nx=384, Nyz=160, steps=30000),
    (label="R16_blk14",  R=16, Nx=384, Nyz=224, steps=30000),
    (label="R16_blk10",  R=16, Nx=384, Nyz=320, steps=36000),
    (label="R16_blk08",  R=16, Nx=384, Nyz=400, steps=40000),
]

const STL_DIR = joinpath(ROOT, "bench", "geometry_stl", "stl")
const KRK_DIR = joinpath(ROOT, "bench", "geometry_stl", "krk")
mkpath(STL_DIR); mkpath(KRK_DIR)
const CSV = joinpath(@__DIR__, "sphere_drag_conv.csv")
open(CSV, "w") do io
    println(io, "label,R_LU,Nx,Nyz,blockage_pct,Re,nu,u_in,steps,Cd,Fx,A_frontal,Vsolid,backend")
end

bname = BACKEND isa CPU ? "cpu" : "cuda"
@info "sphere_drag_convergence start" backend=bname FT ncases=length(cases)

for c in cases
    R, Nx, Nyz, steps = c.R, c.Nx, c.Nyz, c.steps
    D = 2R
    blk = 100.0 * D / Nyz
    nu = U_IN * D / 20.0                      # Re = 20
    cx, cy, cz = Nx ÷ 3, Nyz ÷ 2, Nyz ÷ 2
    stl = joinpath(STL_DIR, "sphere_$(c.label).stl")
    write_sphere_stl(stl; radius=Float64(R), cx=Float64(cx), cy=Float64(cy),
                     cz=Float64(cz), latitudes=max(48, 6R), longitudes=max(96, 12R))
    krk = joinpath(KRK_DIR, "sphere_$(c.label).krk")
    open(krk, "w") do io
        print(io, """
# M-GEO-7(b) convergence case $(c.label): R_LU=$R, blockage=$(round(blk,digits=1))%, Re=20.
Simulation drag_$(c.label) D3Q19
Domain L = $Nx x $Nyz x $Nyz  N = $Nx x $Nyz x $Nyz
Physics nu = $nu
Obstacle sph wall=libb stl(file = "$(relpath(stl, ROOT))")
Boundary west  velocity(ux = $U_IN, uy = 0)
Boundary east  pressure(rho = 1.0)
Boundary south wall
Boundary north wall
Run $steps steps
""")
    end

    @info "running case" label=c.label R Nx Nyz blockage_pct=round(blk,digits=1) nu steps cells=Nx*Nyz*Nyz
    t0 = time()
    res = cd(ROOT) do
        run_simulation(load_kraken(krk); backend=BACKEND, T=FT)
    end
    dt = time() - t0
    Vsolid = count(res.is_solid)
    @info "case done" label=c.label Cd=round(res.Cd,sigdigits=5) Fx=round(res.Fx,sigdigits=4) Vsolid sec=round(dt,digits=1)
    open(CSV, "a") do io
        @printf(io, "%s,%d,%d,%d,%.2f,%.1f,%.5f,%.4f,%d,%.6f,%.6f,%.1f,%d,%s\n",
                c.label, R, Nx, Nyz, blk, 20.0, nu, U_IN, steps,
                res.Cd, res.Fx, res.A, Vsolid, bname)
    end
end

@info "sphere_drag_convergence DONE — free-stream Clift target Cd≈2.6" csv=CSV
