using Test
using KrakeNN

@testset "Taylor-Green vortex 2D" begin
    N = 64
    ν = 0.01
    u0 = 0.01
    k = 2π / N
    decay_rate = 2 * ν * k^2

    # Run long enough for significant decay
    max_steps = 1000
    t_final = max_steps  # Δt = 1 in lattice units

    result = run_taylor_green_2d(; N=N, ν=ν, u0=u0, max_steps=max_steps)

    # Analytical velocity at t_final
    decay_factor = exp(-decay_rate * t_final)

    # Compute L2 error of ux field
    ux_analytical = zeros(N, N)
    for j in 1:N, i in 1:N
        x = Float64(i - 1)
        y = Float64(j - 1)
        ux_analytical[i, j] = -u0 * cos(k * x) * sin(k * y) * decay_factor
    end

    diff = result.ux .- ux_analytical
    l2_err = sqrt(sum(diff .^ 2) / sum(ux_analytical .^ 2))

    @test l2_err < 0.05  # 5% L2 relative error
    @info "Taylor-Green 2D: L2 relative error = $(round(l2_err, digits=5)), decay factor = $(round(decay_factor, digits=4))"

    # Check that velocity has actually decayed
    max_ux = maximum(abs.(result.ux))
    @test max_ux < u0  # must have decayed
    @test max_ux > u0 * decay_factor * 0.5  # but not too much (sanity check)
end
