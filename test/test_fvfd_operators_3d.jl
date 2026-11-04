using Test
using Printf
using Kraken

const G1_TOL = 1.0e-13
const G2_TOL = 1.0e-12
const G3_TOL = 1.0e-12
const CHANNEL3_BC = (:periodic, :periodic, :wall, :wall, :periodic, :periodic)
const PERIODIC3_BC = (:periodic, :periodic, :periodic, :periodic, :periodic, :periodic)

function boundary_arrays_3d(Nx, Ny, Nz; T=Float64)
    return (
        zeros(T, Ny, Nz), zeros(T, Ny, Nz),
        zeros(T, Nx, Nz), zeros(T, Nx, Nz),
        zeros(T, Nx, Ny), zeros(T, Nx, Ny),
    )
end

function velocity_faces_3d(ux, uy, uz, is_solid, bc)
    Nx, Ny, Nz = size(ux)
    ux_face = zeros(eltype(ux), Nx + 1, Ny, Nz)
    uy_face = zeros(eltype(ux), Nx, Ny + 1, Nz)
    uz_face = zeros(eltype(ux), Nx, Ny, Nz + 1)
    vbcs = boundary_arrays_3d(Nx, Ny, Nz; T=eltype(ux))
    Kraken.fvfd_cell_velocity_to_faces_3d!(
        ux_face, uy_face, uz_face, ux, uy, uz, is_solid, vbcs..., bc...,
    )
    return ux_face, uy_face, uz_face
end

function advect3!(out, phi, faces, is_solid, dx, dy, dz, bc, dt, scheme)
    Nx, Ny, Nz = size(phi)
    sbcs = boundary_arrays_3d(Nx, Ny, Nz; T=eltype(phi))
    Kraken.fvfd_advect_upwind_3d!(
        out, phi, sbcs..., faces..., is_solid, dx, dy, dz, bc..., dt;
        advection_scheme=scheme,
    )
    return out
end

function advect3_steps(phi0, faces, is_solid, dx, dy, dz, bc, dt, steps, scheme)
    a = copy(phi0)
    b = similar(a)
    for _ in 1:steps
        advect3!(b, a, faces, is_solid, dx, dy, dz, bc, dt, scheme)
        a, b = b, a
    end
    return a
end

function compact_cosine_blob_3d(Nx, Ny, Nz; cx, cy, cz, rx, ry, rz)
    phi = zeros(Float64, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        r2 = ((i - cx) / rx)^2 + ((j - cy) / ry)^2 + ((k - cz) / rz)^2
        if r2 < 1.0
            phi[i, j, k] = 0.5 * (1.0 + cos(pi * sqrt(r2)))
        end
    end
    return phi
end

function arbitrary_velocity_3d(Nx, Ny, Nz)
    ux = zeros(Float64, Nx, Ny, Nz)
    uy = similar(ux)
    uz = similar(ux)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        x = 2.0 * pi * i / Nx
        y = 2.0 * pi * j / Ny
        z = 2.0 * pi * k / Nz
        ux[i, j, k] = 0.11 + 0.03 * sin(x) + 0.02 * cos(z)
        uy[i, j, k] = -0.04 + 0.02 * cos(y) + 0.01 * sin(x + z)
        uz[i, j, k] = 0.07 + 0.025 * sin(z) - 0.015 * cos(x - y)
    end
    return ux, uy, uz
end

function velocity_faces_2d(ux, uy, is_solid, bc)
    Nx, Ny = size(ux)
    ux_face = zeros(eltype(ux), Nx + 1, Ny)
    uy_face = zeros(eltype(ux), Nx, Ny + 1)
    Kraken.fvfd_cell_velocity_to_faces_2d!(
        ux_face, uy_face, ux, uy, is_solid,
        zeros(eltype(ux), Ny), zeros(eltype(ux), Ny),
        zeros(eltype(ux), Nx), zeros(eltype(ux), Nx),
        bc,
    )
    return ux_face, uy_face
end

function advect2!(out, phi, faces, is_solid, dx, dy, bc, dt, scheme)
    Nx, Ny = size(phi)
    phi_bc = Kraken.FVFDFieldBC2D(
        zeros(eltype(phi), Ny), zeros(eltype(phi), Ny),
        zeros(eltype(phi), Nx), zeros(eltype(phi), Nx),
    )
    Kraken.fvfd_advect_upwind_2d!(
        out, phi, phi_bc, faces..., is_solid, dx, dy, bc, dt;
        advection_scheme=scheme,
    )
    return out
end

function z_slab_fields(Nx, Ny, Nz)
    phi2 = zeros(Float64, Nx, Ny)
    ux2 = similar(phi2)
    uy2 = similar(phi2)
    for j in 1:Ny, i in 1:Nx
        x = 2.0 * pi * i / Nx
        y = 2.0 * pi * j / Ny
        phi2[i, j] = 0.4 + 0.2 * sin(x) * cos(y) + 0.05 * cos(2.0 * x - y)
        ux2[i, j] = 0.09 + 0.02 * sin(x) + 0.01 * cos(y)
        uy2[i, j] = 0.03 * sin(x - y)
    end
    phi3 = repeat(reshape(phi2, Nx, Ny, 1), 1, 1, Nz)
    ux3 = repeat(reshape(ux2, Nx, Ny, 1), 1, 1, Nz)
    uy3 = repeat(reshape(uy2, Nx, Ny, 1), 1, 1, Nz)
    uz3 = zeros(Float64, Nx, Ny, Nz)
    return phi2, ux2, uy2, phi3, ux3, uy3, uz3
end

g1_errors = Dict{Symbol,Float64}()
g2_errors = Dict{Symbol,Float64}()
g3_errors = Dict{Symbol,Float64}()
g4_retention = Dict{Symbol,Float64}()

@testset "FVFD 3D regular-grid transport operators" begin
    @testset "G1 uniform-field invariance" begin
        Nx, Ny, Nz = 16, 16, 8
        is_solid = falses(Nx, Ny, Nz)
        ux, uy, uz = arbitrary_velocity_3d(Nx, Ny, Nz)
        faces = velocity_faces_3d(ux, uy, uz, is_solid, CHANNEL3_BC)
        phi = ones(Float64, Nx, Ny, Nz)
        out = similar(phi)
        for scheme in (:rusanov, :muscl_superbee)
            advect3!(out, phi, faces, is_solid, 1.0, 1.0, 1.0, CHANNEL3_BC, 0.05, scheme)
            err = maximum(abs.(out .- phi))
            g1_errors[scheme] = err
            @test err <= G1_TOL
        end
    end

    @testset "G2 conservation" begin
        Nx, Ny, Nz = 16, 16, 8
        is_solid = falses(Nx, Ny, Nz)
        ux = fill(0.2, Nx, Ny, Nz)
        uy = zeros(Float64, Nx, Ny, Nz)
        uz = zeros(Float64, Nx, Ny, Nz)
        faces = velocity_faces_3d(ux, uy, uz, is_solid, PERIODIC3_BC)
        phi0 = compact_cosine_blob_3d(
            Nx, Ny, Nz; cx=8.5, cy=8.5, cz=4.5, rx=3.0, ry=3.0, rz=1.0,
        )
        mass0 = sum(phi0)
        for scheme in (:rusanov, :muscl_superbee)
            phi = advect3_steps(phi0, faces, is_solid, 1.0, 1.0, 1.0, PERIODIC3_BC, 0.05, 50, scheme)
            err = abs(sum(phi) - mass0)
            g2_errors[scheme] = err
            @test err <= G2_TOL
        end
    end

    @testset "G3 2D equals 3D z-slab equivalence" begin
        Nx, Ny, Nz = 16, 16, 8
        kslab = 4
        is_solid2 = falses(Nx, Ny)
        is_solid3 = falses(Nx, Ny, Nz)
        phi2, ux2, uy2, phi3, ux3, uy3, uz3 = z_slab_fields(Nx, Ny, Nz)
        bc2 = Kraken.FVFDDomainBC2D(; west=:periodic, east=:periodic, south=:wall, north=:wall)
        faces2 = velocity_faces_2d(ux2, uy2, is_solid2, bc2)
        faces3 = velocity_faces_3d(ux3, uy3, uz3, is_solid3, CHANNEL3_BC)
        out2 = similar(phi2)
        out3 = similar(phi3)
        for scheme in (:rusanov, :muscl_superbee)
            advect2!(out2, phi2, faces2, is_solid2, 1.0, 1.0, bc2, 0.05, scheme)
            advect3!(out3, phi3, faces3, is_solid3, 1.0, 1.0, 1.0, CHANNEL3_BC, 0.05, scheme)
            err = maximum(abs.(view(out3, :, :, kslab) .- out2))
            g3_errors[scheme] = err
            @test err <= G3_TOL
        end
    end

    @testset "G4 MUSCL anti-diffusion property" begin
        Nx, Ny, Nz = 32, 16, 8
        is_solid = falses(Nx, Ny, Nz)
        ux = fill(0.45, Nx, Ny, Nz)
        uy = zeros(Float64, Nx, Ny, Nz)
        uz = zeros(Float64, Nx, Ny, Nz)
        faces = velocity_faces_3d(ux, uy, uz, is_solid, PERIODIC3_BC)
        phi0 = compact_cosine_blob_3d(
            Nx, Ny, Nz; cx=11.0, cy=8.0, cz=4.5, rx=4.0, ry=3.0, rz=1.2,
        )
        peak0 = maximum(phi0)
        for scheme in (:rusanov, :muscl_superbee)
            phi = advect3_steps(phi0, faces, is_solid, 1.0, 1.0, 1.0, PERIODIC3_BC, 0.35, 50, scheme)
            g4_retention[scheme] = maximum(phi) / peak0
        end
        rusanov_loss = 1.0 - g4_retention[:rusanov]
        muscl_loss = 1.0 - g4_retention[:muscl_superbee]
        @test muscl_loss < 0.5 * rusanov_loss
    end

    @testset "sym3 wrapper uniform-field smoke" begin
        Nx, Ny, Nz = 8, 8, 8
        is_solid = falses(Nx, Ny, Nz)
        ux, uy, uz = arbitrary_velocity_3d(Nx, Ny, Nz)
        faces = velocity_faces_3d(ux, uy, uz, is_solid, CHANNEL3_BC)
        fields = ntuple(n -> fill(Float64(n), Nx, Ny, Nz), 6)
        outs = ntuple(_ -> zeros(Float64, Nx, Ny, Nz), 6)
        bcs = ntuple(_ -> boundary_arrays_3d(Nx, Ny, Nz), 6)
        Kraken.fvfd_sym3_advect_upwind_3d!(
            outs..., fields..., bcs..., faces..., is_solid,
            1.0, 1.0, 1.0, CHANNEL3_BC..., 0.05;
            advection_scheme=:muscl_superbee,
        )
        for n in 1:6
            @test maximum(abs.(outs[n] .- fields[n])) <= G1_TOL
        end
    end
end

@printf("G1 max errors: rusanov=%.3e muscl=%.3e\n", g1_errors[:rusanov], g1_errors[:muscl_superbee])
@printf("G2 mass errors: rusanov=%.3e muscl=%.3e\n", g2_errors[:rusanov], g2_errors[:muscl_superbee])
@printf("G3 slab errors: rusanov=%.3e muscl=%.3e\n", g3_errors[:rusanov], g3_errors[:muscl_superbee])
@printf(
    "G4 peak retention: rusanov=%.6f muscl=%.6f losses=(%.3e, %.3e)\n",
    g4_retention[:rusanov],
    g4_retention[:muscl_superbee],
    1.0 - g4_retention[:rusanov],
    1.0 - g4_retention[:muscl_superbee],
)
