using Test
using KrakeNN

@testset "Benchmark" begin
    @testset "MLUPs CPU" begin
        results = benchmark_mlups(; Ns=[32, 64], steps=50)

        for (N, mlups) in results
            @test mlups > 0
            @info "CPU MLUPs at N=$N: $(round(mlups, digits=1))"
        end

        # Performance should scale roughly with grid size (memory bandwidth limited)
        @test results[1][2] > 0  # at least runs
    end
end
