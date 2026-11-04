using Test
using Printf
using Kraken

const CHANNEL3_BC = (:periodic, :periodic, :wall, :wall, :periodic, :periodic)
const CHANNEL2_BC = Kraken.FVFDDomainBC2D(;
    west=:periodic, east=:periodic, south=:wall, north=:wall,
)
const W1_TOL = 1.0e-12
const W2_TOL = 1.0e-12
const W3_TOL = 1.0e-13
const W4_TOL = 1.0e-12

const W1_WALL_ERROR = Ref(NaN)
const W2_MAX_ERROR = Ref(NaN)
const W3_MAX_ERROR = Ref(NaN)
const W4_MAX_ERROR = Ref(NaN)

coords_y(j, dy) = (j - 0.5) * dy

function gradient_outputs_3d(T, Nx, Ny, Nz)
    return ntuple(_ -> zeros(T, Nx, Ny, Nz), 9)
end

function run_gradient_3d(ux, uy, uz, is_solid, dx, dy, dz; bc=CHANNEL3_BC)
    Nx, Ny, Nz = size(ux)
    out = gradient_outputs_3d(eltype(ux), Nx, Ny, Nz)
    Kraken.fvfd_velocity_gradient_3d!(
        out..., ux, uy, uz, is_solid, dx, dy, dz, bc...,
    )
    return out
end

function max_abs_tuple(arrays)
    err = 0.0
    for a in arrays
        err = max(err, maximum(abs.(a)))
    end
    return err
end

function max_abs_to(array, value)
    return maximum(abs.(array .- value))
end

function w2_y_affine_error(Nx, Ny, Nz, dx, dy, dz)
    is_solid = falses(Nx, Ny, Nz)
    ux = zeros(Float64, Nx, Ny, Nz)
    uy = similar(ux)
    uz = similar(ux)
    ay = (0.73, -0.42, 0.18)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        y = coords_y(j, dy)
        ux[i, j, k] = 1.1 + ay[1] * y
        uy[i, j, k] = -0.6 + ay[2] * y
        uz[i, j, k] = 0.4 + ay[3] * y
    end
    out = run_gradient_3d(ux, uy, uz, is_solid, dx, dy, dz)
    zero_err = max_abs_tuple((out[1], out[3], out[4], out[6], out[7], out[9]))
    return max(
        max_abs_to(out[2], ay[1]),
        max_abs_to(out[5], ay[2]),
        max_abs_to(out[8], ay[3]),
        zero_err,
    )
end

function w2_x_periodic_error(Nx, Ny, Nz, dx, dy, dz)
    is_solid = falses(Nx, Ny, Nz)
    ux = zeros(Float64, Nx, Ny, Nz)
    uy = similar(ux)
    uz = similar(ux)
    mode = 2
    phase = 0.4
    h = 2.0 * pi * mode / Nx
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        theta = h * (i - 1)
        ux[i, j, k] = 0.31 * sin(theta)
        uy[i, j, k] = -0.27 * cos(theta)
        uz[i, j, k] = 0.19 * sin(theta + phase)
    end
    out = run_gradient_3d(ux, uy, uz, is_solid, dx, dy, dz)
    err = max_abs_tuple((out[2], out[3], out[5], out[6], out[8], out[9]))
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        theta = h * (i - 1)
        err = max(
            err,
            abs(out[1][i, j, k] - 0.31 * cos(theta) * sin(h) / dx),
            abs(out[4][i, j, k] - 0.27 * sin(theta) * sin(h) / dx),
            abs(out[7][i, j, k] - 0.19 * cos(theta + phase) * sin(h) / dx),
        )
    end
    return err
end

function w2_z_periodic_error(Nx, Ny, Nz, dx, dy, dz)
    is_solid = falses(Nx, Ny, Nz)
    ux = zeros(Float64, Nx, Ny, Nz)
    uy = similar(ux)
    uz = similar(ux)
    mode = 1
    phase1 = 0.2
    phase2 = -0.3
    h = 2.0 * pi * mode / Nz
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        theta = h * (k - 1)
        ux[i, j, k] = 0.22 * cos(theta + phase1)
        uy[i, j, k] = -0.11 * sin(theta)
        uz[i, j, k] = 0.09 * cos(theta + phase2)
    end
    out = run_gradient_3d(ux, uy, uz, is_solid, dx, dy, dz)
    err = max_abs_tuple((out[1], out[2], out[4], out[5], out[7], out[8]))
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        theta = h * (k - 1)
        err = max(
            err,
            abs(out[3][i, j, k] + 0.22 * sin(theta + phase1) * sin(h) / dz),
            abs(out[6][i, j, k] + 0.11 * cos(theta) * sin(h) / dz),
            abs(out[9][i, j, k] + 0.09 * sin(theta + phase2) * sin(h) / dz),
        )
    end
    return err
end

@testset "FVFD 3D velocity gradient" begin
    @testset "W1 linear shear exact at wall rows" begin
        Nx, Ny, Nz = 12, 16, 8
        dx, dy, dz = 0.25, 0.125, 0.5
        a = 1.375
        is_solid = falses(Nx, Ny, Nz)
        ux = zeros(Float64, Nx, Ny, Nz)
        uy = zeros(Float64, Nx, Ny, Nz)
        uz = zeros(Float64, Nx, Ny, Nz)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            ux[i, j, k] = a * coords_y(j, dy)
        end
        out = run_gradient_3d(ux, uy, uz, is_solid, dx, dy, dz)
        W1_WALL_ERROR[] = max(
            maximum(abs.(view(out[2], :, 1, :) .- a)),
            maximum(abs.(view(out[2], :, Ny, :) .- a)),
        )
        all_shear_err = max_abs_to(out[2], a)
        other_err = max_abs_tuple((out[1], out[3], out[4], out[5], out[6], out[7], out[8], out[9]))
        @test W1_WALL_ERROR[] <= W1_TOL
        @test all_shear_err <= W1_TOL
        @test other_err <= W1_TOL
    end

    @testset "W2 affine and periodic gradients" begin
        Nx, Ny, Nz = 12, 16, 8
        dx, dy, dz = 0.37, 0.2, 0.41
        W2_MAX_ERROR[] = max(
            w2_y_affine_error(Nx, Ny, Nz, dx, dy, dz),
            w2_x_periodic_error(Nx, Ny, Nz, dx, dy, dz),
            w2_z_periodic_error(Nx, Ny, Nz, dx, dy, dz),
        )
        @test W2_MAX_ERROR[] <= W2_TOL
    end

    @testset "W3 constant field zero gradient" begin
        Nx, Ny, Nz = 12, 16, 8
        dx, dy, dz = 0.25, 0.125, 0.5
        is_solid = falses(Nx, Ny, Nz)
        ux = fill(1.25, Nx, Ny, Nz)
        uy = fill(-0.5, Nx, Ny, Nz)
        uz = fill(0.75, Nx, Ny, Nz)
        out = run_gradient_3d(ux, uy, uz, is_solid, dx, dy, dz)
        W3_MAX_ERROR[] = max_abs_tuple(out)
        @test W3_MAX_ERROR[] <= W3_TOL
    end

    @testset "W4 z-slab matches 2D operator" begin
        Nx, Ny, Nz = 12, 16, 8
        dx, dy, dz = 0.23, 0.31, 0.47
        kslab = 4
        is_solid2 = falses(Nx, Ny)
        is_solid3 = falses(Nx, Ny, Nz)
        ux2 = zeros(Float64, Nx, Ny)
        uy2 = similar(ux2)
        for j in 1:Ny, i in 1:Nx
            x = 2.0 * pi * (i - 1) / Nx
            y = coords_y(j, dy)
            ux2[i, j] = 0.1 + 0.3 * sin(x) + 0.2 * y
            uy2[i, j] = -0.2 + 0.17 * cos(2.0 * x) - 0.04 * y^2
        end
        ux3 = repeat(reshape(ux2, Nx, Ny, 1), 1, 1, Nz)
        uy3 = repeat(reshape(uy2, Nx, Ny, 1), 1, 1, Nz)
        uz3 = zeros(Float64, Nx, Ny, Nz)
        out3 = run_gradient_3d(ux3, uy3, uz3, is_solid3, dx, dy, dz)
        out2 = ntuple(_ -> zeros(Float64, Nx, Ny), 4)
        Kraken.fvfd_velocity_gradient_2d!(
            out2..., ux2, uy2, is_solid2, dx, dy, CHANNEL2_BC,
        )
        W4_MAX_ERROR[] = max(
            maximum(abs.(view(out3[1], :, :, kslab) .- out2[1])),
            maximum(abs.(view(out3[2], :, :, kslab) .- out2[2])),
            maximum(abs.(view(out3[4], :, :, kslab) .- out2[3])),
            maximum(abs.(view(out3[5], :, :, kslab) .- out2[4])),
            max_abs_tuple((out3[3], out3[6], out3[7], out3[8], out3[9])),
        )
        @test W4_MAX_ERROR[] <= W4_TOL
    end
end

@printf("W1 wall-row error: %.3e\n", W1_WALL_ERROR[])
@printf("W2 max error: %.3e\n", W2_MAX_ERROR[])
@printf("W3 max error: %.3e\n", W3_MAX_ERROR[])
@printf("W4 max error: %.3e\n", W4_MAX_ERROR[])
println("EXIT=0")
