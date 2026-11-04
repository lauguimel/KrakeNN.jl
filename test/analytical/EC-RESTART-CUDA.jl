using Test, Kraken

# Explicit opt-in: a missing GPU is an error when GPU acceptance is requested.
# KRAKEN_TEST_EC_RESTART_CUDA=true julia --project test/analytical/EC-RESTART-CUDA.jl
if get(ENV, "KRAKEN_TEST_EC_RESTART_CUDA", "false") == "true"
    using CUDA
    CUDA.functional() || error("EC restart CUDA acceptance requested but CUDA is unavailable")
    CUDA.allowscalar(false)
    if !isdefined(@__MODULE__, :snapshot_differences)
        include(joinpath(@__DIR__, "..", "platform", "state_contract_suite.jl"))
    end
    @testset "EC-RESTART CUDA same-backend" begin
        for FT in (Float64, Float32), wall in (:free_slip, :no_slip)
            @testset "$FT $wall" begin
                # Representative MRT/regularized-charge path; no host Poisson
                # solve or cuDSS dependency. CPU coverage includes direct phi.
                cfg = (; Nx=10, Ny=16, C=10.0, M=10.0, T=190.0,
                    Ma_E=0.01, alpha=1e-4, phi_scheme=:lbm, phi_substeps=2,
                    ns_scheme=:mrt, charge_scheme=:regularized, sidewall_bc=wall,
                    history_interval=3, backend=CUDA.CUDABackend(), FT)
                make() = init_state(ECState; cfg...)
                ref = advance!(make(), 20)
                s = advance!(make(), 7)
                @test s.f_in isa CUDA.CuArray
                @test s.q_f_in isa CUDA.CuArray
                @test s.phi_f_in isa CUDA.CuArray
                mid = export_state(s)
                mktempdir() do dir
                    path = joinpath(dir, "ec.h5")
                    save_checkpoint(path, s)
                    r = load_checkpoint(ECState, path; backend=cfg.backend)
                    @test r.config.sidewall_bc === wall
                    @test r.f_in isa CUDA.CuArray
                    @test isempty(snapshot_differences(export_state(r), mid))
                    a, b = solution(s).result, solution(r).result
                    for k in (:rho, :ux, :uy, :q, :phi, :Ex, :Ey, :Fx, :Fy)
                        @test isequal(getproperty(a, k), getproperty(b, k))
                    end
                    advance!(r, 13)
                    final, continuous = export_state(r), export_state(ref)
                    diffs = snapshot_differences(final, continuous)
                    println("CUDA restart FT=", FT, " wall=", wall, " differing keys=", diffs)
                    for k in sort!(collect(keys(final.fields)))
                        err = maximum(abs.(final.fields[k] .- continuous.fields[k]))
                        println("  ", k, " max_abs_restart_difference=", err)
                    end
                    # First measurement: require exactness, do not pre-authorise
                    # an arbitrary GPU tolerance. Preserve/report any mismatch.
                    @test isempty(diffs)
                    r = load_checkpoint(ECState, path; backend=cfg.backend)
                    for state in (s, r)
                        before = export_state(state)
                        @test state.config.T == 190.0
                        @test state.parameter_cycles == [0]
                        @test state.parameter_values == [190.0]
                        dt = state.p.dt_star
                        update_parameter!(state, :T, 180.0)
                        @test state.config.T == 180.0
                        @test state.config.sidewall_bc === wall
                        @test state.parameter_cycles == [0, 7]
                        @test state.parameter_values == [190.0, 180.0]
                        @test isempty(snapshot_differences(before, export_state(state);
                            classes=(:fields, :scalars, :identity)))
                        @test state.p.dt_star === dt
                        @test state.cycle == 7
                        println("CUDA T provenance FT=", FT, " wall=", wall,
                            " cycle=", state.cycle, " history=",
                            collect(zip(state.parameter_cycles, state.parameter_values)))
                        advance!(state, 13)
                        @test state.cycle == 20
                        @test state.config.sidewall_bc === wall
                        @test state.parameter_cycles == [0, 7]
                        @test state.parameter_values == [190.0, 180.0]
                    end
                    @test isempty(snapshot_differences(export_state(s), export_state(r)))
                    save_checkpoint(path, r)
                    @test isempty(snapshot_differences(export_state(r),
                        export_state(load_checkpoint(ECState, path; backend=cfg.backend))))
                end
            end
        end
    end
    CUDA.synchronize()
else
    @info "EC restart CUDA checks not run (set KRAKEN_TEST_EC_RESTART_CUDA=true on a GPU node)"
end
