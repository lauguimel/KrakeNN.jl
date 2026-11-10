using Test
using KrakeNN

# GNF3D duct with a voxel obstacle (run_gnf_duct_obstacle_3d), CPU Float64.
# Newtonian square duct against the rectangular-duct Fourier series, then a
# Carreau duct with a symmetric cuboid obstacle: no flux leak, no velocity in
# the solid, mirror symmetry in y and z. Sizes are tiny (a few seconds).

# u(y,z) = 16 F a^2/(nu pi^3) sum_{n odd} (-1)^((n-1)/2)/n^3
#          [1 - cosh(n pi z/2a)/cosh(n pi b/2a)] cos(n pi y/2a),
# duct |y|<a, |z|<b, walls at the cell faces for q_w = 1/2 (a = Ny/2, b = Nz/2).
function _gnf3d_duct_series(Ny, Nz, q_w, F, nu)
    a, b = (Ny - 1 + 2q_w) / 2, (Nz - 1 + 2q_w) / 2
    u = zeros(Ny, Nz)
    for c in 1:Nz, j in 1:Ny
        y, z = (j - 1 + q_w) - a, (c - 1 + q_w) - b
        s = sum((-1)^((n - 1) ÷ 2) / n^3 *
                (1 - cosh(n * pi * z / (2a)) / cosh(n * pi * b / (2a))) *
                cos(n * pi * y / (2a)) for n in 1:2:201)
        u[j, c] = 16 * F * a^2 / (nu * pi^3) * s
    end
    return u
end

@testset "GNF3D duct with voxel obstacle" begin
    @testset "Newtonian square duct, no obstacle" begin
        F, nu = 1e-5, 0.1
        out = KrakeNN.run_gnf_duct_obstacle_3d(; rheology=Newtonian(nu),
            Nx=4, Ny=12, Nz=12, F, max_steps=20_000, tol_ss=1e-10)
        ref = _gnf3d_duct_series(12, 12, 0.5, F, nu)
        profile = out.ux[2, :, :]
        err = sqrt(sum(abs2, profile .- ref) / sum(abs2, ref))
        @info "GNF3D duct vs series" err steps=out.steps flow_rate=out.flow_rate
        @test out.converged
        # Measured 1.62e-3 (12x12 cells, macOS arm64 CPU, Julia 1.12.5,
        # 2026-11-09); gate at about twice that.
        @test err <= 3.5e-3
        # Invariance along x, exchange of y and z, flow rate from the profile.
        @test maximum(abs, out.ux .- out.ux[1:1, :, :]) <= 1e-12 * out.u_max
        @test maximum(abs, profile .- profile') <= 1e-12 * out.u_max
        @test isapprox(out.flow_rate, sum(ref); rtol=3.5e-3)
        @test all(==(0.1), round.(out.nu; digits=12))
        @test out.scaling.velocity == 1.0 && out.physical.flow_rate == out.flow_rate
    end

    @testset "Carreau duct with a symmetric cuboid" begin
        rheology = CarreauYasuda(0.1, 0.02, 3000.0, 2.0, 0.5)
        F, Nx, Ny, Nz = 6e-5, 16, 10, 10
        mask = falses(Nx, Ny, Nz)
        mask[7:10, 4:7, 4:7] .= true
        run(m; kw...) = KrakeNN.run_gnf_duct_obstacle_3d(; rheology, Nx, Ny, Nz,
            F, mask=m, max_steps=30_000, tol_ss=1e-9, kw...)
        free = run(nothing)
        out = run(mask)
        fn = run((x, y, z) -> 6 < x < 10 && 3 < y < 7 && 3 < z < 7;
                 h=2e-3, dt=1e-4)
        fluid = .!out.mask
        @info "GNF3D duct obstacle" steps=out.steps conv=out.convergence Q=out.flow_rate Qfree=free.flow_rate
        @test out.converged && free.converged
        @test out.mask == mask && fn.mask == mask   # function and array agree
        @test fn.ux == out.ux                        # h, dt only change scaling
        @test fn.physical.F ≈ F * 2e-3 / 1e-4^2
        @test fn.physical.Lx ≈ Nx * 2e-3
        for f in (out.ux, out.uy, out.uz, out.nu)
            @test all(isfinite, f)
            @test all(iszero, f[out.mask])
        end
        # Obstacle obstructs; the law is genuinely non-Newtonian here.
        @test out.flow_rate < 0.8 * free.flow_rate
        @test minimum(out.nu[fluid]) < 0.9 * 0.1
        # Mass conservation: flux through each x-plane is the same (mass flux
        # spread measured 2.3e-11 at tol_ss=1e-9; gate 1e-8).
        mf = out.mass_flux_planes
        @test (maximum(mf) - minimum(mf)) / maximum(mf) <= 1e-8
        # Mirror symmetry in y and z (measured ~2e-16 relative to u_max).
        flipy(a) = a[:, end:-1:1, :]
        flipz(a) = a[:, :, end:-1:1]
        tolsym = 1e-12 * out.u_max
        @test maximum(abs, out.ux .- flipy(out.ux)) <= tolsym
        @test maximum(abs, out.ux .- flipz(out.ux)) <= tolsym
        @test maximum(abs, out.uy .+ flipy(out.uy)) <= tolsym
        @test maximum(abs, out.uz .+ flipz(out.uz)) <= tolsym
        @test maximum(abs, out.uy) > 1e-3 * out.u_max   # the obstacle deflects flow
    end

    @testset "Argument checks" begin
        law = Newtonian(0.1)
        @test_throws ArgumentError KrakeNN.run_gnf_duct_obstacle_3d(;
            rheology=law, Nx=4, Ny=6, Nz=6, F=1e-5, max_steps=1, mask=trues(4, 6, 6))
        @test_throws ArgumentError KrakeNN.run_gnf_duct_obstacle_3d(;
            rheology=law, Nx=4, Ny=6, Nz=6, F=1e-5, max_steps=1, mask=falses(4, 6, 5))
        @test_throws ArgumentError KrakeNN.run_gnf_duct_obstacle_3d(;
            rheology=Bingham(0.1, 1.0), Nx=4, Ny=6, Nz=6, F=1e-5, max_steps=1)
    end
end
