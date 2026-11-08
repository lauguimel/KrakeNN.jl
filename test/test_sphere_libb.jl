using Test
using KrakeNN
using KernelAbstractions

# ==========================================================================
# 3D sphere in uniform x-flow with LI-BB V2 (Bouzidi pre-phase + TRT).
#
# Tries Metal GPU first (M-series Macs), else CUDA, else CPU with a
# much smaller grid so runtests.jl stays cheap. GPU path is canonical
# (project convention: always GPU locally).
# ==========================================================================

const _METAL_BACKEND = try
    @eval using Metal
    Metal.MetalBackend()
catch
    nothing
end

const _CUDA_BACKEND = if _METAL_BACKEND === nothing
    try
        @eval using CUDA
        CUDA.functional() ? CUDA.CUDABackend() : nothing
    catch
        nothing
    end
else
    nothing
end

_pick_3d_backend() =
    _METAL_BACKEND !== nothing ? (:Metal, _METAL_BACKEND) :
    _CUDA_BACKEND  !== nothing ? (:CUDA,  _CUDA_BACKEND)  :
                                  (:CPU,  CPU())

@testset "Sphere LI-BB V2 — 3D scaffold" begin
    bname, backend = _pick_3d_backend()
    on_gpu = bname !== :CPU
    FT = on_gpu && bname === :Metal ? Float32 : Float64

    # GPU: full-size benchmark (~10s). CPU fallback: small grid, few
    # steps — only verifies the code path, no quantitative asserts.
    Nx, Ny, Nz, radius, steps, window = on_gpu ?
        (120, 60, 60, 8, 10_000, 2_000) :
        (16, 8, 8, 2, 200, 100)

    u_in = FT(0.04)
    Re = 20.0
    D = 2 * radius
    ν = FT(u_in * D / Re)

    result = run_sphere_libb_3d(; Nx=Nx, Ny=Ny, Nz=Nz,
                                  cx=Nx÷4, cy=Ny÷2, cz=Nz÷2,
                                  radius=radius, u_in=u_in, ν=ν,
                                  max_steps=steps, avg_window=window,
                                  backend=backend, T=FT)

    @test !any(isnan, result.ρ)
    @test !any(isnan, result.ux)
    @test !any(isnan, result.uy)
    @test !any(isnan, result.uz)

    if on_gpu
        # Axis symmetry (small transverse lift ratio)
        @test abs(result.Fy / result.Fx) < 0.1
        @test abs(result.Fz / result.Fx) < 0.1
        # Density bounded
        @test maximum(result.ρ) < 1.6
        @test minimum(result.ρ) > 0.6
        # Flow develops: u_avg in the sphere plane should match
        # mass-conservation expectation within 30 %.
        sol = result.is_solid
        cx = Nx ÷ 4
        gap = Float64[]
        for k in 1:Nz, j in 1:Ny
            if !sol[cx, j, k]; push!(gap, Float64(result.ux[cx, j, k])); end
        end
        u_gap_mean = sum(gap) / length(gap)
        A_sphere = π * radius^2
        u_gap_expected = Float64(u_in) * Ny * Nz / (Ny * Nz - A_sphere)
        @test 0.7 * u_gap_expected < u_gap_mean < 1.3 * u_gap_expected
        # Drag: 3D sphere uniform inflow Re=20 free-stream Cd ≈ 2.6
        # (Clift et al. 1978). With moderate blockage and the LI-BB
        # sub-cell q_w, expected range 1 .. 8.
        @test 1.0 < result.Cd < 8.0
    end

    @info "Sphere LI-BB Re=20 (scaffold)" backend=bname Cd=result.Cd maxρm1=maximum(abs.(result.ρ .- 1))
end
