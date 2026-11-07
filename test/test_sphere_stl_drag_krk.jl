using Test
using KrakeNN
using KernelAbstractions

# ==========================================================================
# M-GEO-7 — quantitative drag for a 3D STL sphere driven through the generic
# `.krk` path (run_obstacle_libb_3d), validated as a TWIN against the
# analytic-sphere scaffold `run_sphere_libb_3d`, which is itself validated
# vs Clift et al. 1978 free-stream Cd≈2.6 at Re=20 (test_sphere_libb.jl).
#
# The STL path and the analytic path share kernel + drag reducer + sub-cell
# q_w class; the only physics deltas are the geometry source (triangulated
# vs analytic) and a half-cell centre convention. The twin tolerance is
# ±10% relative Cd, plus both in the Clift band [1, 8].
#
# GPU is canonical (project convention). On CPU the full 120³/10k run is far
# too expensive, so CPU is a parse+voxelize code-path smoke only.
# ==========================================================================

const _DRAG_METAL = try
    @eval using Metal
    Metal.MetalBackend()
catch
    nothing
end

const _DRAG_CUDA = if _DRAG_METAL === nothing
    try
        @eval using CUDA
        CUDA.functional() ? CUDA.CUDABackend() : nothing
    catch
        nothing
    end
else
    nothing
end

_pick_drag_backend() =
    _DRAG_METAL !== nothing ? (:Metal, _DRAG_METAL) :
    _DRAG_CUDA  !== nothing ? (:CUDA,  _DRAG_CUDA)  :
                               (:CPU,   CPU())

@testset "Sphere STL drag via .krk — twin vs analytic scaffold (M-GEO-7)" begin
    bname, backend = _pick_drag_backend()
    on_gpu = bname !== :CPU
    FT = on_gpu && bname === :Metal ? Float32 : Float64

    root = normpath(joinpath(@__DIR__, ".."))
    krk = joinpath(root, "examples", "geometry_stl", "sphere_stl_3d_drag.krk")

    if !on_gpu
        # CPU: parse + voxelize smoke only (no flow — 120³/10k is GPU-class).
        # cd(root): the .krk references the STL fixture by a path relative to
        # the repo root, so resolve it from there regardless of the test cwd.
        cd(root) do
            setup = load_kraken(krk)
            dom = setup.domain
            mask = falses(dom.Nx, dom.Ny, dom.Nz)
            KrakeNN._apply_geometry_3d!(mask, setup, dom.Lx / dom.Nx)
            @test count(mask) > 0
            @test count(mask) < length(mask)
        end
    else
        cd(root) do
            res = run_simulation(load_kraken(krk); backend=backend, T=FT)
            # REGISTRATION: the STL voxelizer is cell-centred ((i-0.5)·dx) while
            # the analytic q_wall is node-centred (xf=i-1) — a half-cell frame
            # difference. The STL sphere is at physical (30,30,30); to place the
            # analytic sphere on the SAME lattice cells, its centre is 30-0.5.
            # (Verified: the two q_wall arrays then correlate at 1.0; LI-BB drag
            #  at R=8 is strongly registration-sensitive, so the twin MUST match
            #  registration to be apples-to-apples.)
            ref = run_sphere_libb_3d(; Nx=120, Ny=60, Nz=60,
                                       cx=29.5, cy=29.5, cz=29.5, radius=8,
                                       u_in=FT(0.04), ν=FT(0.032),
                                       max_steps=10_000, avg_window=2_000,
                                       backend=backend, T=FT)
            Cd_krk = res.Cd
            Cd_ref = ref.Cd
            rel = abs(Cd_krk - Cd_ref) / Cd_ref

            @test !any(isnan, res.ux)
            @test 1.0 < Cd_ref < 8.0          # scaffold in the Clift band
            @test 1.0 < Cd_krk < 8.0          # STL path in the Clift band
            @test rel < 0.05                  # twin: STL ≡ analytic (registered)

            @info "M-GEO-7 sphere STL drag (twin)" backend=bname Cd_krk Cd_ref rel
        end
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    println("PASS M-GEO-7")
end
