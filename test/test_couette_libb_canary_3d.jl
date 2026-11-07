using Test
using KrakeNN

# ==========================================================================
# Planar Couette 3D (D3Q19) — LI-BB V2 canary.
#
# Walls in z: k=1 and k=Nz solid. Top wall at k=Nz moves at (u_top, 0, 0).
# Bottom wall at k=1 stationary. Periodic x and y simulated via ghost
# columns (i=1, i=Nx, j=1, j=Ny).
#
# Expected: linear profile u_x(z) = u_top · (z − z_bot) / H, u_y = u_z = 0.
# Ginzburg-exact for halfway-BB + TRT Λ=3/16.
# ==========================================================================

function planar_couette_3d_setup(Nx::Int, Ny::Int, Nz::Int, u_top::Real;
                                  FT::Type{<:AbstractFloat}=Float64,
                                  q_w::Real=0.5)
    is_solid = zeros(Bool, Nx, Ny, Nz)
    is_solid[:, :, 1]  .= true
    is_solid[:, :, Nz] .= true
    q_wall = zeros(FT, Nx, Ny, Nz, 19)
    uw_x   = zeros(FT, Nx, Ny, Nz, 19)
    uw_y   = zeros(FT, Nx, Ny, Nz, 19)
    uw_z   = zeros(FT, Nx, Ny, Nz, 19)

    # Bottom-adjacent fluid (k=2): all links going -z are cut.
    # D3Q19 -z links: q=7 (-z axis), q=14 (+x-z), q=15 (-x-z),
    #                 q=18 (+y-z), q=19 (-y-z)
    for j in 1:Ny, i in 1:Nx
        for q in (7, 14, 15, 18, 19)
            q_wall[i, j, 2, q] = FT(q_w)
        end
    end
    # Top-adjacent fluid (k=Nz-1): all links going +z are cut.
    # D3Q19 +z links: q=6 (+z axis), q=12 (+x+z), q=13 (-x+z),
    #                 q=16 (+y+z), q=17 (-y+z)
    for j in 1:Ny, i in 1:Nx
        for q in (6, 12, 13, 16, 17)
            q_wall[i, j, Nz-1, q] = FT(q_w)
            uw_x[i, j, Nz-1, q] = FT(u_top)
        end
    end
    return is_solid, q_wall, uw_x, uw_y, uw_z
end

function wrap_periodic_xy_3d!(f::AbstractArray{T,4}) where {T}
    Nx, Ny, Nz, Q = size(f)
    @inbounds for q in 1:Q, k in 1:Nz
        for j in 1:Ny
            f[1,  j, k, q] = f[Nx-1, j, k, q]
            f[Nx, j, k, q] = f[2,    j, k, q]
        end
        for i in 1:Nx
            f[i, 1,  k, q] = f[i, Ny-1, k, q]
            f[i, Ny, k, q] = f[i, 2,    k, q]
        end
    end
    return f
end

function run_planar_couette_libb_3d(; Nx::Int=4, Ny::Int=4, Nz::Int=33,
                                      ν::Real=0.1, u_top::Real=0.01,
                                      steps::Int=5000, q_w::Real=0.5)
    is_solid, qw, uw_x, uw_y, uw_z = planar_couette_3d_setup(Nx, Ny, Nz, u_top; q_w=q_w)
    f_in = zeros(Float64, Nx, Ny, Nz, 19)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx, q in 1:19
        f_in[i, j, k, q] = KrakeNN.equilibrium(D3Q19(), 1.0, 0.0, 0.0, 0.0, q)
    end
    wrap_periodic_xy_3d!(f_in)
    f_out = similar(f_in)
    ρ  = ones(Nx, Ny, Nz); ux = zeros(Nx, Ny, Nz)
    uy = zeros(Nx, Ny, Nz); uz = zeros(Nx, Ny, Nz)
    for _ in 1:steps
        fused_trt_libb_v2_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                    qw, uw_x, uw_y, uw_z, Nx, Ny, Nz, ν)
        wrap_periodic_xy_3d!(f_out)
        f_in, f_out = f_out, f_in
    end
    return (; Nx, Ny, Nz, u_top, ρ, ux, uy, uz, is_solid)
end

@testset "Planar Couette 3D — LI-BB V2 (DSL, D3Q19)" begin

    for q_w in (0.3, 0.5, 0.7)
        out = run_planar_couette_libb_3d(; Nx=4, Ny=4, Nz=33,
                                          ν=0.1, u_top=0.01, steps=10_000,
                                          q_w=q_w)
        Nz, u_top = out.Nz, out.u_top
        H = Float64(Nz - 3) + 2 * q_w
        u_ana = [u_top * (k - 2 + q_w) / H for k in 2:Nz-1]

        i_mid, j_mid = out.Nx ÷ 2, out.Ny ÷ 2
        u_num = out.ux[i_mid, j_mid, 2:Nz-1]
        uy_num = out.uy[i_mid, j_mid, 2:Nz-1]
        uz_num = out.uz[i_mid, j_mid, 2:Nz-1]
        errs = u_num .- u_ana
        L2_rel = sqrt(sum(errs .^ 2) / sum(u_ana .^ 2))
        Linf_rel = maximum(abs.(errs)) / u_top

        @info "Planar Couette 3D V2" q_w Nz L2_rel Linf_rel

        @test all(isfinite.(out.ux))
        @test all(isfinite.(out.uy))
        @test all(isfinite.(out.uz))
        @test maximum(abs.(uy_num)) / u_top < 1e-4
        @test maximum(abs.(uz_num)) / u_top < 1e-4
        # Ginzburg-exact target across the full q_w range.
        @test L2_rel < 1e-4
        @test Linf_rel < 1e-4
    end

end
