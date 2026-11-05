using Test
using LinearAlgebra
using KernelAbstractions

# Carreau calibration script (benchmarks/validation/gnf_calibration), CPU:
#   O1  the BFGS minimiser on an ill-conditioned quadratic (exact minimum,
#       counters, step cap), with an infeasible region (f = Inf) on the path,
#       and with a target already met at the start
#   O2  command-line helpers of the scripts
#   C1  synthetic data on the 16×8 obstacle channel (three forcings converged,
#       Mach < 0.15, no zero-shear node) and reproducible noise
#   C2  smoke calibration: one start, three iterations, misfit decreases
# The script is included in a module of its own; it does not run its `main`
# (guarded by PROGRAM_FILE) and does not load CUDA.

module GNFCalibrationScript
include(joinpath(@__DIR__, "..", "..", "benchmarks", "validation", "gnf_calibration",
                 "calibrate_carreau.jl"))
end

const _GCS = GNFCalibrationScript

@testset "GNF Carreau calibration script" begin
    @testset "O1 BFGS on a quadratic" begin
        # f(x) = ½ (x − x*)ᵀ A (x − x*), A = Q diag(1, 1e2, 1e4) Qᵀ.
        Q = Matrix(qr([1.0 2 0; 0 1 3; 1 0 1]).Q)
        A = Q * Diagonal([1.0, 1e2, 1e4]) * Q'
        A = (A + A') / 2
        xs = [0.3, -1.2, 2.0]
        fq(x) = 0.5 * dot(x .- xs, A * (x .- xs))
        gq(x) = A * (x .- xs)
        x0 = [1.0, 1.0, 1.0]
        res = _GCS.gc_bfgs(fq, gq, x0; maxiter=200, gtol_rel=1e-12, xtol=0.0, ftol=0.0,
                           max_step=10.0)
        @test res.stop === :gradient
        @test norm(res.x .- xs, Inf) < 1e-8
        @test res.iterations < 60
        @test res.n_fgrad == res.iterations + 1              # one gradient per accepted point
        @test length(res.history) == res.iterations + 1
        @test res.n_fval >= res.n_fgrad
        @test all(diff([r.f for r in res.history]) .< 0)    # Armijo: strict decrease
        @test res.n_fail == 0

        # Per-component step cap: no accepted step exceeds it.
        cap = [0.05, 0.1, 0.2]
        rc = _GCS.gc_bfgs(fq, gq, x0; maxiter=300, gtol_rel=1e-10, xtol=0.0, ftol=0.0, max_step=cap)
        @test rc.stop === :gradient
        @test norm(rc.x .- xs, Inf) < 1e-6
        steps = [abs.(rc.history[k+1].x .- rc.history[k].x) for k in 1:rc.iterations]
        @test all(s -> all(s .<= cap .* (1 + 1e-12)), steps)

        # Infeasible region on the path (f = Inf for x₁ > 0.5): the line search
        # shortens the step and the minimiser still converges.
        fi(x) = x[1] > 0.5 ? Inf : fq(x)
        x1 = [0.45, 0.0, 0.0]
        calls = Ref(0)
        gi(x) = (calls[] += 1; @assert isfinite(fi(x)); gq(x))
        ri = _GCS.gc_bfgs(fi, gi, x1; maxiter=300, gtol_rel=1e-10, xtol=0.0, ftol=0.0, max_step=10.0)
        @test ri.n_fail >= 1
        @test ri.stop === :gradient
        @test norm(ri.x .- xs, Inf) < 1e-6
        @test calls[] == ri.n_fgrad

        # Target met at the start: no iteration.
        rt = _GCS.gc_bfgs(fq, gq, xs .+ 1e-6; ftarget=1.0)
        @test rt.stop === :target
        @test rt.iterations == 0
        @test_throws ErrorException _GCS.gc_bfgs(x -> Inf, gq, x0)
    end

    @testset "O2 command-line helpers" begin
        @test _GCS.gc_parse_grid("32x16") == 4
        @test _GCS.gc_parse_grid("16x8") == 2
        @test_throws ErrorException _GCS.gc_parse_grid("32x32")
        @test _GCS.gc_parse_ints("1,3:5") == [1, 3, 4, 5]
        o = _GCS.gc_options(["--grid=16x8", "--mode=smoke"], _GCS.CAL_DEFAULTS)
        @test o["grid"] == "16x8" && o["mode"] == "smoke" && o["backend"] == "cuda"
        @test_throws ErrorException _GCS.gc_options(["--nope=1"], _GCS.CAL_DEFAULTS)
        @test _GCS.gc_wants_cuda(String[], "cuda")
        @test !_GCS.gc_wants_cuda(["--backend=cpu"], "cuda")
        @test _GCS.gc_backend("cpu") isa KernelAbstractions.CPU
        @test_throws ErrorException _GCS.gc_backend("metal")
        @test length(_GCS.CAL_STARTS) == 5
        @test length(_GCS.CAL_NOISE_SEEDS) == 10 && allunique(_GCS.CAL_NOISE_SEEDS)
        # Starts 20–50 % away from the true values, component by component.
        @test all(s -> all(f -> 0.2 - 1e-12 <= abs(f - 1) <= 0.5 + 1e-12, s), _GCS.CAL_STARTS)
    end

    cpu = KernelAbstractions.CPU()
    case = _GCS.gc_case(2)                       # 16×8
    data = _GCS.cal_data(case; backend=cpu, T=Float64)

    @testset "C1 synthetic data and noise" begin
        @test length(data) == 3
        @test all(d -> d.dist < _GCS.GC_FWD_TOL, data)
        @test all(d -> d.mach < 0.15, data)
        @test all(d -> d.zero_shear == 0, data)
        @test issorted([d.uref for d in data])
        n1 = _GCS.cal_noisy(case, data, 0.01, _GCS.CAL_NOISE_SEEDS[1])
        n2 = _GCS.cal_noisy(case, data, 0.01, _GCS.CAL_NOISE_SEEDS[1])
        @test n1[2].ux == n2[2].ux && n1[3].uy == n2[3].uy
        r = vcat([(n1[s].ux[case.nodes] .- data[s].ux[case.nodes]) ./ data[s].uref for s in 1:3]...)
        @test 0.008 < sqrt(sum(abs2, r) / length(r)) < 0.012
        @test all(s -> n1[s].ux[case.solid] == data[s].ux[case.solid], 1:3)
    end

    @testset "C2 smoke calibration (16x8, one start, three iterations)" begin
        fields = [(; ux=d.ux, uy=d.uy) for d in data]
        hist = IOBuffer()
        out = _GCS.cal_run(case, data, fields, _GCS.cal_start(1); backend=cpu, T=Float64,
                           solver=:gmres, maxiter=3, noise_free=true, run_id="test", hist=hist)
        res = out.res; obj = out.obj
        @test res.f < out.J0
        @test out.rms < 0.5 * out.rms0
        @test out.rms_truth ≈ out.rms rtol=1e-12       # noise-free: same data
        @test res.iterations == 3 || res.stop === :target
        @test all(isfinite, res.x)
        @test obj.n_adj == 3 * res.n_fgrad                  # one adjoint per forcing and gradient
        @test obj.n_fwd_fail == 0 && obj.n_domain == 0 && obj.n_adj_unconv == 0
        @test obj.n_fwd <= 3 * res.n_fval
        @test obj.adj_maxrel <= _GCS.GC_ADJ_TOL
        @test count(==('\n'), String(take!(hist))) == res.iterations + 1
        csv = IOBuffer()
        redirect_stdout(devnull) do
            _GCS.cal_csvrow(csv, out, (; run_id="test", kind="noise-free", start=1, level=0.0,
                                       seed=nothing, grid="16x8", backend="cpu", T=Float64,
                                       solver=:gmres))
        end
        line = String(take!(csv))
        @test count(==(','), line) == length(_GCS.CAL_CSV_HEADER) - 1
    end
end
