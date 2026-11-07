using Test
using KrakeNN
using KernelAbstractions

isdefined(@__MODULE__, :_gnf3d_reference) || include("gnf3d_reference.jl")

function _gnf3d_constant_viscosity_pair(rheology; steps=256)
    state = KrakeNN._setup_gnf_channel_3d(;
        rheology, N=16, q_w=0.3, F=1e-5, nu_init=0.1,
        backend=KernelAbstractions.CPU(), FT=Float64)
    Nx, Ny, Nz = state.dims
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        z = (k - 1) / (Nz - 1)
        rho = 1 + 1e-4 * cospi(i - 2) * sinpi(z)
        u = (0.008 * sinpi(z) + 1e-4 * cospi(i - 2),
             0.001 * sinpi(z), 0.001 * sinpi(z))
        for q in 1:19
            state.f_in[i,j,k,q] = KrakeNN.equilibrium(D3Q19(), rho, u..., q)
        end
    end
    KrakeNN._wrap_gnf_periodic_3d!(state.f_in, state.periodic_axes)
    f_gnf, out_gnf = copy(state.f_in), similar(state.f_in)
    f_ref, out_ref = copy(state.f_in), similar(state.f_in)
    rho_ref, ux_ref, uy_ref, uz_ref = (copy(state.rho), copy(state.ux),
                                     copy(state.uy), copy(state.uz))
    for _ in 1:steps
        KrakeNN.fused_trt_gnf_libb_guo_field_step_3d!(
            out_gnf, f_gnf, state.rho, state.ux, state.uy, state.uz, state.is_solid,
            state.q_wall, state.uw_x, state.uw_y, state.uw_z,
            state.Fx, state.Fy, state.Fz, state.sp, state.sm, Nx, Ny, Nz, rheology)
        KrakeNN.fused_trt_libb_v2_guo_field_step_3d!(
            out_ref, f_ref, rho_ref, ux_ref, uy_ref, uz_ref, state.is_solid,
            state.q_wall, state.uw_x, state.uw_y, state.uw_z,
            state.Fx, state.Fy, state.Fz, Nx, Ny, Nz, 0.1)
        for array in (out_gnf, out_ref, state.sp, state.sm)
            KrakeNN._wrap_gnf_periodic_3d!(array, state.periodic_axes)
        end
        f_gnf, out_gnf = out_gnf, f_gnf
        f_ref, out_ref = out_ref, f_ref
    end
    return maximum(abs, f_gnf .- f_ref)
end

@testset "GNF3D wrapper contracts and parity" begin
    expected_args = [:f_out, :ρ_out, :ux_out, :uy_out, :uz_out, :f_in,
        :is_solid, :q_wall, :uw_link_x, :uw_link_y, :uw_link_z, :Nx, :Ny, :Nz,
        :Fx_field, :Fy_field, :Fz_field, :rheology, :sp_field, :sm_field, :trt_lambda]
    @test KrakeNN.spec_args(KrakeNN._TRT_GNF_LIBB_GUO_FIELD_SPEC_3D) == expected_args
    @test KrakeNN.spec_args(KrakeNN._TRT_GNF_LIBB_GUO_FIELD_SPEC_Z_PERIODIC_3D) == expected_args

    @testset "Constant-viscosity population parity" begin
        for rheology in (Newtonian(0.1), PowerLaw(0.1, 1.0))
            delta = _gnf3d_constant_viscosity_pair(rheology)
            @info "GNF3D constant-viscosity parity" rheology delta
            # Same population update algebra; allowance covers Float64 rounding.
            @test delta <= 1e-12
        end
    end

    @testset "Rotation of flow and wall axes" begin
        ref = _gnf3d_reference(:carreau, 16, 0.3)
        outputs = [KrakeNN.run_gnf_channel_3d(;
            rheology=ref.rheology, N=ref.N, q_w=ref.q_w, F=ref.F,
            nu_init=ref.nu, u_init=ref.u, flow_axis, wall_axis,
            max_steps=400, tol_ss=0.0, check_every=400,
            backend=KernelAbstractions.CPU(), FT=Float64)
            for (flow_axis, wall_axis) in ((:x, :z), (:x, :y), (:y, :z))]
        for out in outputs[2:end]
            delta_u = _gnf3d_rel_l2(out.u, outputs[1].u)
            delta_nu = _gnf3d_rel_l2(out.nu, outputs[1].nu)
            @info "GNF3D rotated channel" delta_u delta_nu
            # Measured max 2.3531e-14 on macOS arm64 CPU Float64, Julia 1.12.5,
            # 2026-11-02, base c99f6f889; frozen allowance for roundoff.
            @test delta_u <= 1e-12
            @test delta_nu <= 1e-12
        end
    end

    @testset "Zou-He channel with local TRT rates" begin
        ref = _gnf3d_reference(:carreau, 16, 0.5)
        out = KrakeNN.run_gnf_channel_3d(;
            rheology=ref.rheology, N=ref.N, q_w=ref.q_w, drive=:zou_he,
            flow_axis=:x, wall_axis=:z, Nx_open=32, F=0.0,
            nu_init=ref.nu, u_init=ref.u, u_inlet=ref.u,
            max_steps=6000, tol_ss=1e-8, check_every=200,
            backend=KernelAbstractions.CPU(), FT=Float64)
        middle = out.dims[1] ÷ 2
        # Profiles use the midstream plane, avoiding entrance/outlet averages.
        profile = vec(sum(out.fields.ux[middle, 2:end-1, 2:end-1]; dims=1)) ./ 2
        nu_field = (inv.(out.fields.sp) .- 0.5) ./ 3
        nu_profile = vec(sum(nu_field[middle, 2:end-1, 2:end-1]; dims=1)) ./ 2
        # Conserved mass flux, rather than volume flux, includes compressibility.
        flux = [sum(out.fields.rho[i, 2:end-1, 2:end-1] .*
                    out.fields.ux[i, 2:end-1, 2:end-1])
                for i in (2, middle, out.dims[1] - 1)]
        error_u = _gnf3d_rel_l2(profile, ref.u)
        error_nu = _gnf3d_rel_l2(nu_profile, ref.nu)
        flux_spread = (maximum(flux) - minimum(flux)) / maximum(flux)
        @info "GNF3D Zou-He local rates" error_u error_nu flux_spread steps=out.steps
        @test out.converged
        for array in (out.fields.f, out.fields.rho, out.fields.ux, out.fields.uy,
                      out.fields.uz, out.fields.sp, out.fields.sm)
            @test all(isfinite, array)
        end
        @test all(iszero, out.fields.Fx)
        @test all(iszero, out.fields.Fy)
        @test all(iszero, out.fields.Fz)
        # Frozen at twice measured errors, 2026-11-02, macOS arm64 CPU
        # Float64, Julia 1.12.5, base c99f6f889. Original measurements were
        # 6.012940299086424e-3, 2.799217216555376e-4, 5.123443827546978e-10.
        @test error_u <= 1.202588059817285e-2
        @test error_nu <= 5.598434433110752e-4
        @test flux_spread <= 1.024688765509396e-9
        @test _gnf3d_inside_bounds(nu_field[.!out.fields.is_solid],
                                  ref.rheology.eta_inf, ref.rheology.eta_0)
    end
end

@testset "GNF3D backend copy never hands a BitArray to the device" begin
    # A BitArray has no bulk device transfer; on CUDA it fell back to scalar
    # indexing. The copy must expand it to a dense Bool array first.
    mask = falses(3, 4, 5)
    mask[2, 2, 2] = true
    copied = KrakeNN._gnf_backend_copy_3d(KernelAbstractions.CPU(), mask)
    @test copied isa Array{Bool,3}
    @test copied == mask
    @test count(copied) == 1
end
