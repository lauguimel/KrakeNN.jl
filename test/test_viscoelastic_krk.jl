using KrakeNN
using Test

@testset "Viscoelastic .krk (FVFD log-conformation)" begin
    root = normpath(joinpath(@__DIR__, ".."))
    case_path = joinpath(root, "benchmarks", "krk", "viscoelastic", "cylinder_oldroyd_b.krk")

    cd(root) do
        setup = load_kraken(case_path)
        @test setup.lattice == :D2Q9
        @test :viscoelastic in setup.modules
        @test setup.max_steps == 20

        r = run_simulation(case_path)

        @test haskey(r, :rho)
        @test haskey(r, :ux)
        @test haskey(r, :uy)
        @test all(isfinite, r.rho)
        @test all(isfinite, r.ux)
        @test all(isfinite, r.uy)
        @test all(rho_val -> 0.9 < rho_val < 1.1, r.rho)

        max_abs_ux = maximum(abs, r.ux)
        inlet = r.u_mean
        @test isfinite(inlet)
        @test inlet > 0
        @test max_abs_ux > 0
        @test max_abs_ux > 0.5 * inlet

        # Scalar drag outputs only; `Cd_last` is NaN by design when the steady
        # stop rule took no sample (`stop = :none`, the .krk default).
        cd_keys = [k for k in keys(r) if occursin("cd", lowercase(String(k))) &&
                   getproperty(r, k) isa Real && k != :Cd_last]
        @test !isempty(cd_keys)
        @test all(k -> isfinite(getproperty(r, k)), cd_keys)
        if haskey(r, :cd_history)
            @test all(s -> isfinite(last(s)), r.cd_history)
            if !isempty(r.cd_history)
                @test isfinite(r.Cd_last)
            end
        end
        if haskey(r, :n_drag_samples)
            @test r.n_drag_samples > 0
        end
    end
end
