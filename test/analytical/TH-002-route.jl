using Test
using KernelAbstractions
using Kraken

# TH-002 — conduction `.krk` fallback: does it run the case the file describes?
#
# Reported as issue #19: the conduction branch of `_run_thermal` forwarded only
# Nx, Ny, Pr and the step count, so `nu`, `alpha` and the thermal faces given in
# the file were replaced by the driver defaults (ν = 0.05, α = ν/Pr, hot south /
# cold north). Everything below goes through `run_simulation(path)` on a `.krk`
# file, which is the path that exposed the hole.
#
# CPU Float64, non-refined `Module thermal`, case name selecting the conduction
# branch.

"""Write `body` to a temporary `.krk` file and return its path."""
function _th_002_krk(body::AbstractString)
    path = tempname() * ".krk"
    write(path, body)
    return path
end

"""Small west/east conduction fixture with the given `Physics` line."""
function _th_002_horizontal(physics::AbstractString, steps::Int)
    return """
Simulation heat_conduction_route_test D2Q9
Domain L = 1.0 x 0.5 N = 12 x 8
$(physics)
Module thermal
Boundary west wall T = 0.8
Boundary east wall T = 0.2
Boundary south wall
Boundary north wall
Run $(steps) steps
"""
end

@testset "TH-002 public conduction parameter mapping" begin

    @testset "supplied nu and alpha reach the driver" begin
        path = _th_002_krk(_th_002_horizontal("Physics nu = 0.1 alpha = 0.017", 1))
        result = run_simulation(path; backend=KernelAbstractions.CPU(),
                                T=Float64, max_steps=0)
        ν = getproperty(result, Symbol("ν"))
        α = getproperty(result, Symbol("α"))

        # Exact equality: this tests propagation of scalar inputs, not PDE
        # accuracy. Before the fix these came back as 0.05 and 0.05/0.71.
        @test ν == 0.1
        @test α == 0.017
        @test size(result.Temp) == (12, 8)
        @test all(isfinite, result.Temp)
    end

    @testset "supplied wall temperatures and orientation are honoured" begin
        path = _th_002_krk(_th_002_horizontal("Physics nu = 0.1 alpha = 0.02", 4000))
        result = run_simulation(path; backend=KernelAbstractions.CPU(), T=Float64)

        # The file describes a west/east problem. Before the fix the run was
        # south/north with T = 1 and T = 0.
        @test result.orientation === :horizontal
        @test result.T_hot == 0.8
        @test result.T_cold == 0.2

        Temp = result.Temp
        Nx, Ny = size(Temp)
        profile = [sum(Temp[i, :]) / Ny for i in 1:Nx]

        # Analytical reference: steady 1D conduction between two fixed-temperature
        # walls is linear. The walls are half-way, at x = 0.5 and x = Nx + 0.5
        # with node i at x = i, so T_i = T_hot + (T_cold - T_hot)·(i - 0.5)/Nx.
        # Measured 2026-11-03 on CPU Float64: max deviation 1.2e-4, set by the
        # adiabatic south/north walls (the on-node walls used before gave
        # 2.5e-2 against this profile).
        analytic = [0.8 + (0.2 - 0.8) * (i - 0.5) / Nx for i in 1:Nx]
        @test maximum(abs.(profile .- analytic)) < 5e-4

        # Wall values are the ones the file asked for: the linear extrapolation
        # of the first two nodes to the half-way wall. Measured 6.9e-5 on both
        # walls (2.8e-2 with the former on-node walls).
        @test abs(1.5 * profile[1] - 0.5 * profile[2] - 0.8) < 2e-4
        @test abs(1.5 * profile[end] - 0.5 * profile[end-1] - 0.2) < 2e-4

        # Ra ≈ 0: buoyancy must not drive any flow. Measured max|uy| ≈ 4e-15.
        @test maximum(abs, result.uy) < 1e-10
    end

    @testset "alpha is the diffusivity that actually runs" begin
        # Exact equality above proves the number is forwarded; this proves it
        # is the number the thermal relaxation uses. The vertical (south/north)
        # initial condition carries a 1 % perturbation on the (kx, ky) = (2π/Nx,
        # π/Ny) mode, whose amplitude decays as exp(-α k² t) while the fluid is
        # at rest. Fitting that decay between two step counts recovers α.
        function _decay_alpha(alpha::Float64)
            function _amplitude(steps::Int)
                body = """
Simulation heat_conduction_decay D2Q9
Domain L = 1.0 x 1.0 N = 32 x 32
Physics nu = 0.05 alpha = $(alpha)
Module thermal
Boundary south wall T = 1.0
Boundary north wall T = 0.0
Boundary x periodic
Run $(steps) steps
"""
                result = run_simulation(_th_002_krk(body);
                                        backend=KernelAbstractions.CPU(), T=Float64)
                Temp = result.Temp
                Nx, Ny = size(Temp)
                num = 0.0
                den = 0.0
                for j in 1:Ny, i in 1:Nx
                    basis = sin(2π * i / Nx) * sin(π * j / Ny)
                    num += Temp[i, j] * basis
                    den += basis^2
                end
                return num / den
            end

            a1 = _amplitude(50)
            a2 = _amplitude(150)
            k2 = (2π / 32)^2 + (π / 32)^2
            return -log(abs(a2 / a1)) / 100 / k2
        end

        # Gate: the lattice introduces an O(k²) discretisation bias on the
        # measured diffusivity. Measured 2026-11-03 on CPU Float64 with the
        # half-way walls: α = 0.02 → 0.020136 (+0.68 %), α = 0.05 → 0.050266
        # (+0.53 %); the former on-node walls gave +2.2 % and +1.9 %.
        for alpha in (0.02, 0.05)
            measured = _decay_alpha(alpha)
            @test abs(measured / alpha - 1) < 0.02
        end
    end

    @testset "defaults unchanged when nothing is supplied" begin
        # No `alpha` in the file: α must still fall back to ν / Pr, and ν to the
        # historical 0.05 when `nu` is absent from the thermal parameters.
        path = _th_002_krk(_th_002_horizontal("Physics nu = 0.05 Pr = 0.71", 1))
        result = run_simulation(path; backend=KernelAbstractions.CPU(),
                                T=Float64, max_steps=0)
        @test getproperty(result, Symbol("ν")) == 0.05
        @test getproperty(result, Symbol("α")) == 0.05 / 0.71
        @test result.Pr == 0.71
    end

    @testset "configurations the fallback cannot represent raise" begin
        # Silent substitution is the defect. Each of these describes a problem
        # the Ra ≈ 0 Boussinesq fallback has no way to run.
        both_pairs = """
Simulation heat_conduction_case D2Q9
Domain L = 1.0 x 0.5 N = 12 x 8
Physics nu = 0.1 alpha = 0.017
Module thermal
Boundary west wall T = 0.8
Boundary east wall T = 0.2
Boundary south wall T = 1.0
Boundary north wall T = 0.0
Run 1 steps
"""
        single_face = """
Simulation heat_conduction_case D2Q9
Domain L = 1.0 x 0.5 N = 12 x 8
Physics nu = 0.1 alpha = 0.017
Module thermal
Boundary west wall T = 0.8
Boundary east wall
Boundary south wall
Boundary north wall
Run 1 steps
"""
        equal_temps = """
Simulation heat_conduction_case D2Q9
Domain L = 1.0 x 0.5 N = 12 x 8
Physics nu = 0.1 alpha = 0.017
Module thermal
Boundary west wall T = 0.5
Boundary east wall T = 0.5
Boundary south wall
Boundary north wall
Run 1 steps
"""
        for body in (both_pairs, single_face, equal_temps)
            path = _th_002_krk(body)
            @test_throws ArgumentError run_simulation(path;
                                                      backend=KernelAbstractions.CPU(),
                                                      T=Float64, max_steps=0)
        end

        # A negative diffusivity is not runnable either. Now that `alpha` is
        # actually forwarded, the `.krk` sanity check sees the real thermal tau
        # and rejects it (an ErrorException, raised before the driver's own
        # positivity guard). Before the fix the value was dropped on the way to
        # the driver, so the run would have proceeded with α = ν/Pr.
        bad_alpha = _th_002_krk(_th_002_horizontal("Physics nu = 0.1 alpha = -0.01", 1))
        @test_throws ErrorException run_simulation(bad_alpha;
                                                   backend=KernelAbstractions.CPU(),
                                                   T=Float64, max_steps=0)
    end
end
