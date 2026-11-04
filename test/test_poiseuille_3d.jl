using Test
using Kraken
using KernelAbstractions

# Uses the package-level `stream_periodic_xz_wall_y_3d!` (periodic x/z, no-slip
# bounce-back y-walls) — the same reusable streamer the 3D viscoelastic Couette
# canary builds on. This test pins it against the analytical parabolic profile.

@testset "Poiseuille 3D body force" begin
    Nx, Ny, Nz = 4, 32, 4
    ν = 0.1
    Fx = 1e-5
    max_steps = 10000

    config = LBMConfig(D3Q19(); Nx=Nx, Ny=Ny, Nz=Nz, ν=ν, u_lid=0.0, max_steps=max_steps)
    state = initialize_3d(config, Float64)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux_field, uy_field, uz_field = state.ρ, state.ux, state.uy, state.uz
    is_solid = state.is_solid
    ω = Float64(omega(config))

    for step in 1:max_steps
        stream_periodic_xz_wall_y_3d!(f_out, f_in, Nx, Ny, Nz)
        collide_guo_3d!(f_out, is_solid, ω, Float64(Fx), 0.0, 0.0)
        compute_macroscopic_forced_3d!(ρ, ux_field, uy_field, uz_field, f_out, Float64(Fx), 0.0, 0.0)
        f_in, f_out = f_out, f_in
    end

    ρ_cpu = Array(ρ)
    ux_cpu = Array(ux_field)

    # Analytical parabolic profile (half-way bounce-back: walls at y=0.5, y=Ny+0.5)
    u_analytical = [Fx / (2ν) * (j - 0.5) * (Ny + 0.5 - j) for j in 1:Ny]
    u_numerical = ux_cpu[2, :, 2]  # extract at mid-plane (any x,z slice — periodic)

    # Interior points only (skip wall-adjacent nodes j=1 and j=Ny)
    u_max = maximum(u_analytical)
    errors = abs.(u_numerical[2:end-1] .- u_analytical[2:end-1])
    max_rel_err = maximum(errors) / u_max

    @test max_rel_err < 0.02  # 2% L∞ relative error
    @info "Poiseuille 3D: L∞ relative error = $(round(max_rel_err, digits=5))"

    # Mass conservation
    @test abs(sum(ρ_cpu) - Nx * Ny * Nz) / (Nx * Ny * Nz) < 0.001
end
