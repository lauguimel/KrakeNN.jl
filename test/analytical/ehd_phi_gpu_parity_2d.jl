using Test
using KernelAbstractions
using KrakeNN

const _EHD_CUDSS_GPU_OK = (try
    @eval Main using CUDA, CUDSS
    @eval Main CUDA.functional()
catch
    false
end) === true

function _ehd_frozen_charge_field()
    Nx, Ny = 31, 48
    C = 10.0
    p = KrakeNN._ehd_ec_lattice_params(Ny, C, 10.0, 175.0, 1e-2, 1e-4, 1.0, 0.3;
                                      FT=Float64)
    analytic = KrakeNN.ehd_hydrostatic_profiles(C, Ny; FT=Float64)
    q_profile = p.q_inj .* analytic.q_star
    q_cpu = zeros(Float64, Nx, Ny)
    for j in 1:Ny
        q_cpu[:, j] .= q_profile[j]
    end
    q_cpu[:, 1] .= p.q_inj
    return Nx, Ny, p, q_cpu
end

@testset "EHD direct phi CPU frozen-charge contract" begin
    Nx, Ny, p, q_cpu = _ehd_frozen_charge_field()
    backend = KernelAbstractions.CPU()

    setup = KrakeNN.ehd_poisson_setup(Nx, Ny, p.eps; xbc=:neumann, backend=backend)
    @test setup isa KrakeNN.EhdPoissonSetup

    qfield = KernelAbstractions.zeros(backend, Float64, Nx, Ny)
    phi = KernelAbstractions.zeros(backend, Float64, Nx, Ny)
    copyto!(qfield, q_cpu)

    KrakeNN.ehd_poisson_solve!(phi, setup, qfield)
    phi_cpu = Array(phi)
    @test maximum(abs.(phi_cpu[:, 1] .- 1.0)) <= 1e-12
    @test maximum(abs.(phi_cpu[:, Ny])) <= 1e-12

    phi_profile = [sum(@view phi_cpu[:, j]) / Nx for j in 1:Ny]
    xvar_phi = maximum(abs.(phi_cpu .- reshape(phi_profile, 1, Ny)))
    @test xvar_phi <= 1000eps(Float64) * maximum(abs, phi_cpu)

    A = KrakeNN._ehd_poisson_matrix(Nx, Ny, :neumann)
    rhs = similar(setup.rhs)
    KrakeNN._ehd_poisson_fill_rhs!(rhs, q_cpu, setup)
    residual_linf = maximum(abs.(A * vec(phi_cpu) .- rhs))
    @test residual_linf <= 1e-10

    phi_second = KernelAbstractions.zeros(backend, Float64, Nx, Ny)
    KrakeNN.ehd_poisson_solve!(phi_second, setup, qfield)
    @test reinterpret(UInt64, vec(Array(phi_second))) ==
          reinterpret(UInt64, vec(phi_cpu))
end

@testset "EHD direct phi GPU parity" begin
    if !_EHD_CUDSS_GPU_OK
        @info "Skipping EHD cuDSS GPU phi parity test (CUDA+CUDSS not functional here)"
        @test_skip "EHD GPU direct phi path requires CUDA.functional() + CUDSS"
    else
        Nx, Ny, p, q_cpu = _ehd_frozen_charge_field()

        cpu_backend = KernelAbstractions.CPU()
        cpu_setup = KrakeNN.ehd_poisson_setup(Nx, Ny, p.eps; xbc=:neumann,
                                             backend=cpu_backend)
        q_cpu_field = KernelAbstractions.zeros(cpu_backend, Float64, Nx, Ny)
        phi_cpu_field = KernelAbstractions.zeros(cpu_backend, Float64, Nx, Ny)
        copyto!(q_cpu_field, q_cpu)
        KrakeNN.ehd_poisson_solve!(phi_cpu_field, cpu_setup, q_cpu_field)
        phi_cpu = Array(phi_cpu_field)

        backend = Main.CUDA.CUDABackend()
        setup_gpu = KrakeNN.ehd_poisson_setup(Nx, Ny, p.eps; xbc=:neumann,
                                             backend=backend)
        @test setup_gpu isa KrakeNN.EhdPoissonSetupGPU

        q_gpu = KernelAbstractions.zeros(backend, Float64, Nx, Ny)
        phi_gpu = KernelAbstractions.zeros(backend, Float64, Nx, Ny)
        copyto!(q_gpu, q_cpu)
        KrakeNN.ehd_poisson_solve!(phi_gpu, setup_gpu, q_gpu)

        linf = maximum(abs.(Array(phi_gpu) .- phi_cpu))
        @test linf <= 1e-8
        @info "EHD cuDSS GPU phi parity" Nx=Nx Ny=Ny Linf=linf
    end
end
