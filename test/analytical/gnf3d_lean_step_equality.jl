# Performance options of the 3D D3Q19 GNF step `fused_trt_gnf_libb_guo_field_step_3d!`
# (steps C1, C6, N1 of .orchestrator/3d-kernel-plan.md) leave the results unchanged:
#
#   (1) variant = :lean (CollideTRTGNFGuoFieldLean_3D: equilibria recomputed at write
#       time, moments and force read back) == variant = :reference;
#   (2) an explicit workgroupsize == the automatic launch;
#   (3) a law closure with its parameters held by value (`lbm_law_by_value`) == the
#       same closure with a parameter array;
#   (4) the register cap (maxregs = :auto, the default: 128 regs/thread Float64,
#       96 Float32 on CUDA) == no cap (maxregs = 0).
#
# Case: 10×12×14 lattice with a voxel sphere, LI-BB with random q_w in (0.05, 1) on
# every fluid→solid link, nonzero random wall velocity, non-uniform Guo force (zero
# on a slab, so both branches of the half-force shift run), random near-equilibrium
# populations, per-cell TRT rates. Laws: native CarreauYasuda, ClassicalLaw(CY) and
# DenseNet through LBMLawViscosity. 20 steps (and periodic_z = true for one law).
# Outputs compared: f, sp, sm, rho, ux, uy, uz.
#
# Gate for (1), (3), (4), CPU and CUDA (decision 2026-11-13: relative tolerance
# instead of bitwise equality; the lean step differs from the reference by round-off
# on the GPU, max |diff| 5.6e-17 measured on Aqua):
#   for each of the 7 output arrays separately (f, sp, sm, rho, ux, uy, uz, all cells,
#   solid and ghost included), norm-wise, not elementwise:
#       max_i |ref_i - x_i| <= rtol * max_i |ref_i|
#   with rtol = 1e-12 in Float64. A reference array that is all zero therefore
#   requires the compared array to be exactly zero (no absolute floor); a NaN or Inf
#   in the compared array fails the gate. Elementwise relative errors are not used
#   because velocities pass through zero.
#   Float32: rtol = 1e-4 (about 840 eps(Float32)). 64 eps (7.6e-6) failed on an H100
#   (job 28190399, 2026-11-14): after 20 steps the lean and by-value runs differ from
#   the reference by 1.7e-5 to 2.3e-5, round-off accumulated over the steps. 1e-12 in
#   Float64 is about 4500 eps(Float64), so 1e-4 is the milder of the two gates.
# Production defaults (2026-11-13): variant = :lean; on CUDA maxregs 128 + workgroup
# (32, 4, 2) in Float64, maxregs 96 + automatic workgroup in Float32. Runs here pass
# the variant explicitly (reference by default) and use the production launch
# defaults (workgroupsize = :auto, maxregs = :auto) unless stated.
#   The number of differing values is printed (@info), so a bitwise result stays visible.
# (2) explicit workgroup vs automatic: bitwise (`isequal`) on CPU and CUDA (same
# compiled kernel, only the launch shape changes). (4) on CPU: bitwise, the cap is
# ignored there (plain launch through the fallback of KrakeNN.lbm_launch!).
# Command (cluster): KRAKEN_REQUIRE_GPU=1 julia --project=. test/analytical/gnf3d_lean_step_equality.jl
using Test, KrakeNN, Random

const _G3L_REQUIRE_GPU = get(ENV, "KRAKEN_REQUIRE_GPU", "0") in ("1", "true")
# CUDA is loaded here, at top level, never inside a function.
const _G3L_GPU_OK = (try
    @eval Main using CUDA
    @eval Main CUDA.functional()
catch
    false
end) === true

const _G3L_DIMS = (10, 12, 14)
const _G3L_LAMBDA = 3 / 16

function _g3l_case(::Type{T}; seed=11) where {T}
    K = KrakeNN
    rng = MersenneTwister(seed)
    Nx, Ny, Nz = _G3L_DIMS
    ctr = (5.3, 6.6, 7.4)
    is_solid = Array{Bool}(undef, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        is_solid[i, j, k] = (i - ctr[1])^2 + (j - ctr[2])^2 + (k - ctr[3])^2 <= 2.7^2
    end
    q_wall = zeros(T, Nx, Ny, Nz, 19)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        is_solid[i, j, k] && continue
        for q in 2:19
            ii, jj, kk = i + K._G3_CX[q], j + K._G3_CY[q], k + K._G3_CZ[q]
            (1 <= ii <= Nx && 1 <= jj <= Ny && 1 <= kk <= Nz) || continue
            is_solid[ii, jj, kk] && (q_wall[i, j, k, q] = T(0.05 + 0.95 * rand(rng)))
        end
    end
    uw = ntuple(_ -> T.(1e-3 .* randn(rng, Nx, Ny, Nz, 19)), 3)
    F = ntuple(_ -> T.(1e-4 .* randn(rng, Nx, Ny, Nz)), 3)
    for a in F
        a[is_solid] .= zero(T)
        a[:, :, 1:3] .= zero(T)          # zero-force slab: unshifted branch
    end
    f = zeros(T, Nx, Ny, Nz, 19)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        rho = 1 + 0.01 * randn(rng)
        u = 0.02 .* (randn(rng), randn(rng), randn(rng))
        for q in 1:19
            f[i, j, k, q] = T(K.equilibrium(D3Q19(), rho, u..., q) * (1 + 1e-3 * randn(rng)))
        end
    end
    nu = 0.05 .+ 0.1 .* rand(rng, Nx, Ny, Nz)
    sp = T.(1 ./ (3 .* nu .+ 0.5))
    sm = T.(1 ./ (_G3L_LAMBDA ./ (3 .* nu) .+ 0.5))
    return (; is_solid, q_wall, uw, F, f, sp, sm)
end

function _g3l_laws(::Type{T}, to) where {T}
    cy = CarreauYasuda(0.17, 0.0034, 20.0, 2.0, 0.7)
    lcy = ClassicalLaw(cy)
    net = DenseNet(1, (8, 8); activation=:tanh, output=:exp, shift=(zero(T),), scale=(T(5),))
    theta = law_init(net; rng=MersenneTwister(3))
    theta[end] = log(0.1)
    return (; cy,
            classical=LBMLawViscosity(lcy, to(law_init(lcy)); eps=0.0),
            densenet=LBMLawViscosity(net, to(theta); eps=1e-3, gamma_scale=100.0))
end

# Runs `steps` steps from the case; returns host copies of (f, sp, sm, rho, ux, uy, uz).
function _g3l_run(case, rheology, ::Type{T}, to; steps=20, variant=:reference,
                  periodic_z=false, workgroupsize=:auto, maxregs=:auto) where {T}
    Nx, Ny, Nz = _G3L_DIMS
    f_in = to(copy(case.f))
    f_out = similar(f_in)
    sp, sm = to(copy(case.sp)), to(copy(case.sm))
    rho, ux, uy, uz = (to(zeros(T, Nx, Ny, Nz)) for _ in 1:4)
    is_solid, q_wall = to(case.is_solid), to(case.q_wall)
    uwx, uwy, uwz = map(to, case.uw)
    Fx, Fy, Fz = map(to, case.F)
    for _ in 1:steps
        KrakeNN.fused_trt_gnf_libb_guo_field_step_3d!(f_out, f_in, rho, ux, uy, uz, is_solid,
            q_wall, uwx, uwy, uwz, Fx, Fy, Fz, sp, sm, Nx, Ny, Nz, rheology;
            Λ=_G3L_LAMBDA, periodic_z, variant, workgroupsize, maxregs)
        f_in, f_out = f_out, f_in
    end
    return map(Array, (f_in, sp, sm, rho, ux, uy, uz))
end

const _G3L_NAMES = ("f", "sp", "sm", "rho", "ux", "uy", "uz")

_g3l_bitwise(a, b) = all(map(isequal, a, b))
_g3l_ndiff(a, b) = sum(count(.!isequal.(x, y)) for (x, y) in zip(a, b))
_g3l_rtol(::Type{Float64}) = 1e-12
_g3l_rtol(::Type{Float32}) = 1f-4
# Gate of the header, array by array: max|ref - x| <= rtol * max|ref| (an all-zero
# reference requires x == 0; NaN in x fails since NaN <= r is false).
_g3l_close(ref, out, rtol) = all(maximum(abs.(x .- y)) <= rtol * maximum(abs, x)
                                 for (x, y) in zip(ref, out))
# Largest per-array relative difference, for the @info lines only.
_g3l_maxrel(a, b) = maximum(maximum(abs.(x .- y)) / max(maximum(abs, x), floatmin(eltype(x)))
                            for (x, y) in zip(a, b))

@testset "3D GNF step: lean variant, workgroup size, parameters by value" begin
    @test KrakeNN.spec_args(KrakeNN._TRT_GNF_LIBB_GUO_FIELD_LEAN_SPEC_3D) ==
          KrakeNN.spec_args(KrakeNN._TRT_GNF_LIBB_GUO_FIELD_SPEC_3D)
    @test KrakeNN.spec_args(KrakeNN._TRT_GNF_LIBB_GUO_FIELD_LEAN_SPEC_Z_PERIODIC_3D) ==
          KrakeNN.spec_args(KrakeNN._TRT_GNF_LIBB_GUO_FIELD_SPEC_3D)
    @test_throws ArgumentError KrakeNN._gnf3d_step_spec(false, :fast)
    @test KrakeNN._gnf3d_resolve_maxregs(Float64, :auto) == 128
    @test KrakeNN._gnf3d_resolve_maxregs(Float32, :auto) == 96
    @test KrakeNN._gnf3d_resolve_maxregs(Float64, 0) == 0
    @test KrakeNN._gnf3d_resolve_maxregs(Float64, nothing) === nothing
    @test_throws ArgumentError KrakeNN._gnf3d_resolve_maxregs(Float64, -1)
    @test_throws ArgumentError KrakeNN._gnf3d_resolve_maxregs(Float64, :max)
    @test KrakeNN._gnf3d_step_workgroup(KrakeNN.KernelAbstractions.CPU(), Float64) === nothing

    @testset "CPU $T" for T in (Float64, Float32)
        rtol = _g3l_rtol(T)
        case = _g3l_case(T)
        laws = _g3l_laws(T, identity)
        for (name, law) in pairs(laws)
            ref = _g3l_run(case, law, T, identity)
            @test all(a -> all(isfinite, a), ref)
            lean = _g3l_run(case, law, T, identity; variant=:lean)
            @info "GNF3D lean vs reference, CPU" T name n_diff = _g3l_ndiff(ref, lean) max_rel = _g3l_maxrel(ref, lean)
            @test _g3l_close(ref, lean, rtol)
            name === :cy || continue
            wg = _g3l_run(case, law, T, identity; workgroupsize=(4, 4, 2))
            @test _g3l_bitwise(ref, wg)
            # The cap is ignored on the CPU (fallback launch): bitwise.
            @test _g3l_bitwise(ref, _g3l_run(case, law, T, identity; maxregs=0))
            @test _g3l_bitwise(ref, _g3l_run(case, law, T, identity; maxregs=64))
            refz = _g3l_run(case, law, T, identity; periodic_z=true)
            leanz = _g3l_run(case, law, T, identity; periodic_z=true, variant=:lean)
            @test _g3l_close(refz, leanz, rtol)
        end
        for name in (:classical, :densenet)
            law = laws[name]
            byv = KrakeNN.lbm_law_by_value(law)
            ref = _g3l_run(case, law, T, identity)
            @test _g3l_close(ref, _g3l_run(case, byv, T, identity), rtol)
            @test _g3l_close(ref, _g3l_run(case, byv, T, identity; variant=:lean), rtol)
        end
        @test_throws ArgumentError KrakeNN.lbm_law_by_value(laws.densenet; max_params=10)
    end

    if _G3L_GPU_OK
        to = Main.CUDA.CuArray
        @test KrakeNN._gnf3d_step_workgroup(Main.CUDA.CUDABackend(), Float64) == (32, 4, 2)
        @test KrakeNN._gnf3d_step_workgroup(Main.CUDA.CUDABackend(), Float32) === nothing
        @testset "CUDA $T" for T in (Float64, Float32)
            case = _g3l_case(T)
            laws = _g3l_laws(T, to)
            rtol = _g3l_rtol(T)
            for (name, law) in pairs(laws), steps in (1, 20)
                ref = _g3l_run(case, law, T, to; steps)          # reference, capped (default)
                @test all(a -> all(isfinite, a), ref)
                lean = _g3l_run(case, law, T, to; steps, variant=:lean)
                @info "GNF3D lean vs reference, CUDA" T name steps n_diff = _g3l_ndiff(ref, lean) max_rel = _g3l_maxrel(ref, lean)
                @test _g3l_close(ref, lean, rtol)
                for variant in (:reference, :lean)
                    capped = variant === :reference ? ref : lean
                    uncapped = _g3l_run(case, law, T, to; steps, variant, maxregs=0)
                    @info "GNF3D register cap vs no cap, CUDA" T name steps variant n_diff = _g3l_ndiff(uncapped, capped) max_rel = _g3l_maxrel(uncapped, capped)
                    @test _g3l_close(uncapped, capped, rtol)
                end
                if name === :cy
                    for wg in ((32, 4, 2), (64, 4, 1), (256, 1, 1))
                        @test _g3l_bitwise(ref, _g3l_run(case, law, T, to; steps, workgroupsize=wg))
                    end
                else
                    byv = KrakeNN.lbm_law_by_value(law)
                    for variant in (:reference, :lean)
                        out = _g3l_run(case, byv, T, to; steps, variant)
                        @info "GNF3D by-value parameters vs array, CUDA" T name steps variant n_diff = _g3l_ndiff(ref, out) max_rel = _g3l_maxrel(ref, out)
                        @test _g3l_close(ref, out, rtol)
                    end
                end
            end
        end
    elseif _G3L_REQUIRE_GPU
        @test false   # KRAKEN_REQUIRE_GPU=1 but CUDA is not functional
    else
        @info "gnf3d_lean_step_equality: CUDA not functional, GPU part skipped"
    end
end
