using Test
using Kraken

# Boundary stencils of the 3D FVFD viscoelastic path (#51).
#
# 1. `compute_polymeric_force_3d!` at x and z WALLS (`wall_x`, `wall_z`) must
#    use the same second-order one-sided difference as the y walls; a clamped
#    central difference returns (τ₂ − τ₁)/2 there, half the derivative. An
#    open face (neither periodic nor wall) keeps the clamp: a one-sided
#    difference would amplify the jump of a conformation reset or copy.
# 2. MUSCL-Superbee advection must stay second order next to a periodic z
#    face. With a uniform velocity the operator then commutes with any
#    circular shift in z; a fallback to first-order upwind within two cells
#    of the z faces breaks that. (x and y keep their fallback on purpose: the
#    z-slab of the 3D operator must equal the 2D operator.)
# 3. `logfv_max_grad_norm_3d` must reduce on the arrays' own device instead of
#    copying nine fields to the host every step; its value must not change.

function bs3_quadratic_stress(Nx, Ny, Nz)
    # Quadratic in each index, so central and one-sided second-order
    # differences are both exact: every expected force is linear in i, j, k.
    tau = [zeros(Nx, Ny, Nz) for _ in 1:6]  # xx xy xz yy yz zz
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        tau[1][i, j, k] = 0.3 * i^2 + i
        tau[2][i, j, k] = -0.2 * i^2
        tau[3][i, j, k] = 0.7 * k^2
        tau[4][i, j, k] = 0.4 * j^2
        tau[5][i, j, k] = -0.5 * k^2
        tau[6][i, j, k] = 0.1 * k^2
    end
    return tau
end

function bs3_polymer_force(; Nx=8, Ny=7, Nz=6, kwargs...)
    tau = bs3_quadratic_stress(Nx, Ny, Nz)
    F = [zeros(Nx, Ny, Nz) for _ in 1:3]
    compute_polymeric_force_3d!(F..., tau...; kwargs...)
    Fx_exact = [2 * 0.3 * i + 1 + 2 * 0.7 * k for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    Fy_exact = [-2 * 0.2 * i + 2 * 0.4 * j - 2 * 0.5 * k for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    Fz_exact = [2 * 0.1 * k for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    return F, (Fx_exact, Fy_exact, Fz_exact), tau
end

function bs3_advect(phi, ux, uy, uz; scheme=:muscl_superbee,
                    bc=(:periodic, :periodic, :periodic, :periodic, :periodic, :periodic),
                    is_solid=falses(size(phi)))
    Nx, Ny, Nz = size(phi)
    out = similar(phi)
    bnd = (zeros(Ny, Nz), zeros(Ny, Nz), zeros(Nx, Nz), zeros(Nx, Nz),
           zeros(Nx, Ny), zeros(Nx, Ny))
    ux_face = fill(ux, Nx + 1, Ny, Nz)
    uy_face = fill(uy, Nx, Ny + 1, Nz)
    uz_face = fill(uz, Nx, Ny, Nz + 1)
    Kraken.fvfd_advect_upwind_3d!(
        out, phi, bnd..., ux_face, uy_face, uz_face, is_solid,
        1.0, 1.0, 1.0, bc..., 0.1; advection_scheme=scheme,
    )
    return out
end

function bs3_host_grad_norm(g...)
    m = 0.0
    for idx in eachindex(g[1])
        m = max(m, sum(Float64(a[idx]) * Float64(a[idx]) for a in g))
    end
    return sqrt(m)
end

@testset "FVFD 3D boundary stencils" begin
    @testset "polymer force, x and z walls" begin
        F, Fexact, _ = bs3_polymer_force(; periodic_x=false, periodic_z=false,
                                         wall_x=true, wall_z=true)
        for c in 1:3
            @test maximum(abs.(F[c] .- Fexact[c])) < 1e-12
        end
        # The first cell next to an x face is where the clamped stencil halved
        # the derivative; check it explicitly.
        @test F[1][1, 4, 3] ≈ Fexact[1][1, 4, 3] atol = 1e-12
        @test F[1][end, 4, 3] ≈ Fexact[1][end, 4, 3] atol = 1e-12
        @test F[3][4, 4, 1] ≈ Fexact[3][4, 4, 1] atol = 1e-12
    end

    @testset "polymer force, open x and z faces keep the clamp" begin
        F, Fexact, tau = bs3_polymer_force(; periodic_x=false, periodic_z=false)
        # Interior and y-wall cells are exact; the first and last x / z cells
        # use the clamped central difference, (τ[2] − τ[1]) / 2 and
        # (τ[N] − τ[N−1]) / 2.
        Nx, Ny, Nz = size(F[1])
        dxc(t, i, j, k) = (t[min(i + 1, Nx), j, k] - t[max(i - 1, 1), j, k]) / 2
        dzc(t, i, j, k) = (t[i, j, min(k + 1, Nz)] - t[i, j, max(k - 1, 1)]) / 2
        Fx_clamp = [dxc(tau[1], i, j, k) + dzc(tau[3], i, j, k) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
        Fz_clamp = [dxc(tau[3], i, j, k) + dzc(tau[6], i, j, k) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
        @test maximum(abs.(F[1] .- Fx_clamp)) < 1e-12
        @test maximum(abs.(F[3] .- Fz_clamp)) < 1e-12
        @test maximum(abs.(F[1][2:end-1, :, 2:end-1] .- Fexact[1][2:end-1, :, 2:end-1])) < 1e-12
    end

    @testset "polymer force, argument checks" begin
        tau = bs3_quadratic_stress(2, 7, 6)
        F = [zeros(2, 7, 6) for _ in 1:3]
        @test_throws ArgumentError compute_polymeric_force_3d!(F..., tau...;
                                                               periodic_x=false, wall_x=true)
        @test_throws ArgumentError compute_polymeric_force_3d!(F..., tau...; wall_x=true)
        tau = bs3_quadratic_stress(8, 2, 6)
        F = [zeros(8, 2, 6) for _ in 1:3]
        @test_throws ArgumentError compute_polymeric_force_3d!(F..., tau...)
    end

    @testset "polymer force, periodic x and z unchanged" begin
        # A field periodic in x and z: the wrapped central difference is the
        # reference, and must be what the periodic path still returns.
        Nx, Ny, Nz = 8, 7, 6
        tau = [zeros(Nx, Ny, Nz) for _ in 1:6]
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            tau[1][i, j, k] = sin(2π * i / Nx) + 0.1 * j
            tau[3][i, j, k] = cos(2π * k / Nz)
        end
        F = [zeros(Nx, Ny, Nz) for _ in 1:3]
        compute_polymeric_force_3d!(F..., tau...; periodic_x=true, periodic_z=true)
        Fx_ref = [(sin(2π * mod1(i + 1, Nx) / Nx) - sin(2π * mod1(i - 1, Nx) / Nx)) / 2 +
                  (cos(2π * mod1(k + 1, Nz) / Nz) - cos(2π * mod1(k - 1, Nz) / Nz)) / 2
                  for i in 1:Nx, j in 1:Ny, k in 1:Nz]
        @test maximum(abs.(F[1] .- Fx_ref)) < 1e-13
    end

    @testset "MUSCL advection commutes with z shifts on a periodic z" begin
        # Nz = 4 is the thin periodic direction of a quasi-2D run: with the
        # old fallback no cell of such a box was ever advected at second order.
        for (Nx, Ny, Nz) in ((9, 8, 4), (9, 8, 6))
            phi = [sin(0.7 * i) + 0.3 * cos(1.3 * j) + 0.2 * sin(2π * k / Nz) + 0.05 * (i * j % 5) +
                   0.1 * ((i + 2k) % 3) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
            ref = bs3_advect(phi, 0.3, -0.2, 0.25)
            for s in ((0, 0, 1), (0, 0, 2), (0, 0, -1))
                shifted = bs3_advect(circshift(phi, s), 0.3, -0.2, 0.25)
                @test maximum(abs.(shifted .- circshift(ref, s))) < 1e-14
            end
        end
    end

    @testset "MUSCL z-shift invariance with walls, open faces and a solid" begin
        # x and y walls or open faces keep their fallback, which does not
        # depend on k, so a periodic z still commutes with z shifts; a solid
        # cell against a z face must move with the field.
        Nx, Ny, Nz = 9, 8, 6
        phi = [sin(0.7 * i) + 0.3 * cos(1.3 * j) + 0.2 * sin(2π * k / Nz) + 0.1 * ((i + 2k) % 3)
               for i in 1:Nx, j in 1:Ny, k in 1:Nz]
        solid = falses(Nx, Ny, Nz)
        solid[5, 4, 1] = true
        for bc in ((:periodic, :periodic, :wall, :wall, :periodic, :periodic),
                   (:open, :open, :wall, :wall, :periodic, :periodic))
            ref = bs3_advect(phi, 0.3, -0.2, 0.25; bc=bc, is_solid=solid)
            for s in ((0, 0, 1), (0, 0, 3))
                shifted = bs3_advect(circshift(phi, s), 0.3, -0.2, 0.25; bc=bc,
                                     is_solid=circshift(solid, s))
                @test maximum(abs.(shifted .- circshift(ref, s))) < 1e-14
            end
        end
    end

    @testset "MUSCL on a thin periodic z differs from first-order upwind" begin
        # Guards against the shift test passing because every cell fell back
        # to upwind (which also commutes with shifts).
        Nx, Ny, Nz = 9, 8, 4
        phi = [sin(0.7 * i) + 0.3 * cos(1.3 * j) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
        muscl = bs3_advect(phi, 0.3, -0.2, 0.0)
        upwind = bs3_advect(phi, 0.3, -0.2, 0.0; scheme=:rusanov)
        @test maximum(abs.(muscl .- upwind)) > 1e-4
    end

    @testset "max velocity-gradient norm" begin
        for T in (Float64, Float32)
            g = [T.(randn(5, 4, 3)) for _ in 1:9]
            @test Kraken.logfv_max_grad_norm_3d(g...) isa Float64
            @test Kraken.logfv_max_grad_norm_3d(g...) ≈ bs3_host_grad_norm(g...) rtol = (T == Float64 ? 1e-15 : 1e-6)
        end
    end
end
