using Test
using KernelAbstractions
using KrakeNN

# H2-004 — west/east pressure-driven channel through the public `.krk` runner.
#
# Reported as issue #18: `_apply_pressure_bc!` handled only `:east`, so a west
# pressure inlet reached a no-op and the run silently solved a different
# problem. Everything below drives the code through `run_simulation(path)` on a
# `.krk` file, which is the path a user takes; calling the driver directly is
# what let the hole survive.
#
# Generic, non-refined D2Q9 BGK route, CPU. Domain is a plain channel:
# pressure on west and east, solid south and north.

"""Write `body` to a temporary `.krk` file and return its path."""
function _h2_004_krk(body::AbstractString)
    path = tempname() * ".krk"
    write(path, body)
    return path
end

const _H2_004_CONTRACT = """
Simulation pressure_channel D2Q9
Domain L = 2.0 x 1.0 N = 32 x 16
Physics nu = 0.1
Boundary west pressure(rho = 1.0002)
Boundary east pressure(rho = 0.9998)
Boundary south wall
Boundary north wall
Run 40 steps
"""

"""Pressure-driven channel fixture: `4N x N` nodes, ρ_west/ρ_east = 1±4e-4."""
function _h2_004_channel(N::Int, steps::Int)
    return """
Simulation pressure_channel D2Q9
Domain L = 4.0 x 1.0 N = $(4N) x $(N)
Physics nu = 0.1
Boundary west pressure(rho = 1.0004)
Boundary east pressure(rho = 0.9996)
Boundary south wall
Boundary north wall
Run $(steps) steps
"""
end

@testset "H2-004 public west/east pressure mapping" begin

    @testset "imposed face densities (CPU Float64)" begin
        path = _h2_004_krk(_H2_004_CONTRACT)
        result = run_simulation(path; backend=KernelAbstractions.CPU(), T=Float64)
        ρ = getproperty(result, Symbol("ρ"))

        # Corner nodes are shared with the south/north bounce-back walls and are
        # not part of the pressure contract, hence 2:end-1 (14 interior nodes).
        west = sum(ρ[1, 2:end-1]) / 14
        east = sum(ρ[end, 2:end-1]) / 14
        imposed_drop = 1.0002 - 0.9998
        drop_error = abs((west - east) - imposed_drop) / imposed_drop

        # Gates: Zou-He imposes the face density exactly, so the only error is
        # floating-point accumulation. Measured 2026-09-14 on CPU Float64:
        # |west - 1.0002| = 2.2e-16, |east - 0.9998| = 1.1e-16,
        # relative drop error = 9.4e-13. Gates from the issue #18 draft.
        @test isapprox(west, 1.0002; atol=5e-13, rtol=0)
        @test isapprox(east, 0.9998; atol=5e-13, rtol=0)
        @test drop_error < 1e-9

        # Density is higher at west, so the flow must go towards east. A sign
        # error in the reconstructed normal velocity would flip this.
        @test sum(result.ux[2:end-1, 2:end-1]) > 0
    end

    @testset "CPU Float32 parity" begin
        path = _h2_004_krk(_H2_004_CONTRACT)
        r64 = run_simulation(path; backend=KernelAbstractions.CPU(), T=Float64)
        r32 = run_simulation(path; backend=KernelAbstractions.CPU(), T=Float32)
        ρ32 = getproperty(r32, Symbol("ρ"))

        west32 = sum(Float64.(ρ32[1, 2:end-1])) / 14
        east32 = sum(Float64.(ρ32[end, 2:end-1])) / 14

        # Gate: Float32 cannot hold 1.0002 more accurately than eps(Float32)≈1.2e-7.
        # Measured 2026-09-14: |west - 1.0002| = |east - 0.9998| = 1.52e-7.
        @test isapprox(west32, 1.0002; atol=1e-6, rtol=0)
        @test isapprox(east32, 0.9998; atol=1e-6, rtol=0)

        flux64 = sum(r64.ux[2:end-1, 2:end-1])
        flux32 = Float64(sum(r32.ux[2:end-1, 2:end-1]))
        # Gate: CPU Float64/Float32 parity on the integrated flux.
        # Measured 2026-09-14: relative difference 2.7e-4.
        @test abs(flux32 - flux64) / abs(flux64) < 1e-3
    end

    @testset "Poiseuille profile and grid convergence" begin
        # Analytical reference: plane Poiseuille driven by a constant pressure
        # gradient, u(y) = G/(2ν) · y·(H - y), with G = dp/dx = c_s²·Δρ/(Nx-1),
        # c_s² = 1/3, and H = Ny because half-way bounce-back puts the walls at
        # j = 0.5 and j = Ny + 0.5. Node j sits at y = j - 0.5.
        ν = 0.1
        Δρ = 1.0004 - 0.9996

        function _profile_error(N::Int, steps::Int)
            path = _h2_004_krk(_h2_004_channel(N, steps))
            result = run_simulation(path; backend=KernelAbstractions.CPU(), T=Float64)
            Nx = 4N
            G = (Δρ / 3) / (Nx - 1)
            H = float(N)
            analytic = [G / (2ν) * ((j - 0.5) * (H - (j - 0.5))) for j in 1:N]
            measured = result.ux[Nx ÷ 2, :]
            return maximum(abs.(measured .- analytic)) / maximum(analytic)
        end

        err16 = _profile_error(16, 8000)
        err32 = _profile_error(32, 16000)

        # Gates from the analytical solution above, measured 2026-09-14 on CPU
        # Float64 (macOS, Julia 1.11): err16 = 2.03e-3, err32 = 5.15e-4.
        @test err16 < 3e-3
        @test err32 < 8e-4
        # Half-way bounce-back plus Zou-He is second order; measured ratio 3.95.
        @test err16 / err32 > 3.0
    end

    @testset "unsupported pressure face raises" begin
        # Silence is the defect being fixed: a face the runner cannot honour
        # must say so rather than fall through to a no-op.
        body = replace(_H2_004_CONTRACT,
                       "Boundary north wall" => "Boundary north pressure(rho = 1.0)")
        path = _h2_004_krk(body)
        @test_throws ErrorException run_simulation(path;
                                                   backend=KernelAbstractions.CPU(),
                                                   T=Float64)

        err = try
            run_simulation(path; backend=KernelAbstractions.CPU(), T=Float64)
            nothing
        catch e
            e
        end
        msg = sprint(showerror, err)
        @test occursin("north", msg)
        @test occursin("west", msg) && occursin("east", msg)
    end
end
