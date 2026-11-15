# Wall slip and central momentum convection on CUDA/cuDSS (opt-in: KRAKEN_TEST_GPU=1,
# KRAKEN_REQUIRE_GPU=1 makes a skip fail). Float64.
#
# Frozen gates (2026-11-13, set before the first run), CPU vs GPU on the same problem:
# - Newton converges on both; states agree to 1e-10 relative (the linear
#   solvers differ, so not to round-off); residual of the GPU state evaluated on
#   the host <= 1e-10 relative to the forcing scale;
# - adjoint gradient of the slip parameters: GPU vs CPU <= 1e-8 relative.
# Cases: plane channel with Navier slip on the domain walls (upwind); inclined
# immersed channel with cubic slip, quadratic ghost + M3, convection = :central.
isdefined(@__MODULE__, :_isl_problem) || include(joinpath(@__DIR__, "incns_fv_slip_helpers.jl"))

const _SLGPU_REQUIRED = get(ENV, "KRAKEN_REQUIRE_GPU", "0") in ("1", "true")
const _SLGPU_ENABLED = _SLGPU_REQUIRED || get(ENV, "KRAKEN_TEST_GPU", "0") in ("1", "true")
const _SLGPU_READY = _SLGPU_ENABLED && (try
    @eval Main using CUDA, CUDSS
    Main.CUDA.functional()
catch
    false
end)

function _slgpu_case(p, U0, θ, pmap, obj; label)
    CU = Main.CUDA
    opts = (; atol = 1e-12, rtol = 1e-13, maxit = 30)
    sc = fv_newton_solver(p)
    sg = fv_newton_solver(p; to = CU.CuArray, linsolve = CUDABackendTag())
    Uc = copy(U0); Ug = CU.CuArray(copy(U0))
    stc = fv_newton!(sc, Uc; opts...)
    stg = fv_newton!(sg, Ug; opts...)
    @test stc.converged
    @test stg.converged
    Uh = Array(Ug)
    @test norm(Uh - Uc, Inf) <= 1e-10 * norm(Uc, Inf)
    scale = max(norm(p.src, Inf) * p.grid.h^2, 1e-300)
    @test norm(fv_residual(Uh, p), Inf) <= 1e-10 * max(scale, 1.0)
    rc = fv_steady_gradient(sc, Uc, θ, obj; params = pmap)
    rg = fv_steady_gradient(sg, Ug, θ, obj; params = pmap)
    @test norm(Array(rg.grad) - rc.grad, Inf) <= 1e-8 * norm(rc.grad, Inf)
    @info "slip GPU" label iterations_cpu = stc.iterations iterations_gpu = stg.iterations state = norm(Uh - Uc, Inf) / norm(Uc, Inf) grad_cpu = rc.grad grad_gpu = Array(rg.grad)
end

@testset "FV wall slip + central convection (GPU opt-in)" begin
    if !_SLGPU_READY
        if _SLGPU_REQUIRED
            @test _SLGPU_READY
        else
            @test_skip "Requires explicit GPU opt-in and functional CUDA/cuDSS"
        end
    else
        Main.CUDA.allowscalar(false)
        @testset "plane channel, Navier slip" begin
            p = _sl_channel(32; law = FVNavierSlip(0.1), conv = true)
            _slgpu_case(p, zeros(fv_nunknowns(p)), [0.1], FVProblemMap(fv_with_slip_parameters, 1),
                fv_flow_rate(p, 1); label = "channel")
        end
        @testset "inclined immersed channel, cubic slip, central" begin
            law = _SlipCubic(0.02, 0.002)
            p = _isl_problem(1; law, conv = true, convection = :central)
            g = p.grid; i = g.Nx ÷ 2
            idx = Int[]; w = Float64[]
            for j in 1:g.Ny
                p.tag[i, j] == FV_CELL_SOLID && continue
                push!(idx, fv_index(g, i, j, 1)); push!(w, g.h * ((j - 0.5) * g.h - 0.75)^2)
            end
            _slgpu_case(p, _isl_state(p, _isl_us(law)), [0.02, 0.002], FVProblemMap(fv_with_slip_parameters, 2),
                FVLinearFunctional(idx, w); label = "inclined central")
        end
    end
end
