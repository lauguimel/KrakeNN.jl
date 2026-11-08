# M-GEO-7(b) follow-up — the two low-blockage points that OOM'd on the 40GB
# A100 (the first run captured 20/14/10%). Run on an 80GB GPU (H100), CUDA F64.
# 8% keeps the consistent Nx=384; 6% needs a shorter streamwise box (Nx=256)
# to fit 80 GB F64 (109M cells at Nx=384 would need ~99 GB) — noted in the CSV.
# Appends to a SEPARATE CSV so the first run's results stay intact.

using KrakeNN
using KernelAbstractions
using Printf

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(ROOT, "examples", "geometry_stl", "make_sphere_stl.jl"))

@eval using CUDA
@assert CUDA.functional() "CUDA not functional — requires a GPU node"
const BACKEND = CUDA.CUDABackend()
const FT = Float64
const U_IN = 0.04

cases = [
    (label="R16_blk08", R=16, Nx=384, Nyz=400, steps=40000),  # 8%,  61M cells ~58GB
    (label="R16_blk06", R=16, Nx=256, Nyz=533, steps=40000),  # 6%,  73M cells ~69GB (Nx shorter)
]

const STL_DIR = joinpath(ROOT, "bench", "geometry_stl", "stl")
const KRK_DIR = joinpath(ROOT, "bench", "geometry_stl", "krk")
mkpath(STL_DIR); mkpath(KRK_DIR)
const CSV = joinpath(@__DIR__, "sphere_drag_conv_lowblock.csv")
open(CSV, "w") do io
    println(io, "label,R_LU,Nx,Nyz,blockage_pct,Re,nu,u_in,steps,Cd,Fx,A_frontal,Vsolid,backend")
end

@info "sphere_drag_lowblock start" ncases=length(cases)
for c in cases
    R, Nx, Nyz, steps = c.R, c.Nx, c.Nyz, c.steps
    D = 2R
    blk = 100.0 * D / Nyz
    nu = U_IN * D / 20.0
    cx, cy, cz = Nx ÷ 3, Nyz ÷ 2, Nyz ÷ 2
    stl = joinpath(STL_DIR, "sphere_$(c.label).stl")
    write_sphere_stl(stl; radius=Float64(R), cx=Float64(cx), cy=Float64(cy),
                     cz=Float64(cz), latitudes=max(48, 6R), longitudes=max(96, 12R))
    krk = joinpath(KRK_DIR, "sphere_$(c.label).krk")
    open(krk, "w") do io
        print(io, """
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
    @info "running case" label=c.label R Nx Nyz blockage_pct=round(blk,digits=1) cells=Nx*Nyz*Nyz
    t0 = time()
    res = cd(ROOT) do
        run_simulation(load_kraken(krk); backend=BACKEND, T=FT)
    end
    @info "case done" label=c.label Cd=round(res.Cd,sigdigits=5) Fx=round(res.Fx,sigdigits=4) sec=round(time()-t0,digits=1)
    open(CSV, "a") do io
        @printf(io, "%s,%d,%d,%d,%.2f,%.1f,%.5f,%.4f,%d,%.6f,%.6f,%.1f,%d,%s\n",
                c.label, R, Nx, Nyz, blk, 20.0, nu, U_IN, steps,
                res.Cd, res.Fx, res.A, count(res.is_solid), "cuda")
    end
end
@info "sphere_drag_lowblock DONE" csv=CSV
