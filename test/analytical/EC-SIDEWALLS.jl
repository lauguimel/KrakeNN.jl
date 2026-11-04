# Issue #21: sidewall capability, not electroconvection onset qualification.
# Gates fixed before execution; tests derive population constraints and the
# plane-Poiseuille reference independently of the boundary implementation.
module ECSidewallTests
using Test
using LinearAlgebra
using KernelAbstractions
using Kraken

const CX = (0, 1, 0, -1, 0, 1, -1, -1, 1)
const CY = (0, 0, 1, 0, -1, 1, 1, -1, -1)

function on_backend(backend, a)
    out = KernelAbstractions.zeros(backend, eltype(a), size(a)...)
    copyto!(out, a)
    return out
end

function operators(backend, FT)
    Nx, Ny = 6, 8
    original = [FT(1 + (i + 8j + 64d) / 1024)
                for i in 1:Nx, j in 1:Ny, d in 1:9]
    expected = copy(original)
    for (i, inward) in ((1, 1), (Nx, -1)), j in 2:Ny-1, d in 1:9
        if CX[d] * inward > 0
            reflected = only(k for k in 1:9 if CX[k] == -CX[d] && CY[k] == CY[d])
            expected[i, j, d] = original[i, j, reflected]
        end
    end
    f = on_backend(backend, original)
    Kraken.apply_free_slip_sidewalls_2d!(f, Nx, Ny)
    KernelAbstractions.synchronize(backend)
    actual = Array(f)
    @test actual == expected # Includes untouched interior, plates and corners.
    for i in (1, Nx), j in 2:Ny-1
        @test sum(CX[d] * actual[i, j, d] for d in 1:9) == zero(FT)
        @test sum(CY[d] * actual[i, j, d] for d in 1:9) ==
              (i == 1 ? FT(-1/4) : FT(-1/2))
    end
    ux0 = [FT((i + 8j) / 1024) for i in 1:Nx, j in 1:Ny]
    uy0 = [FT((2i + 8j) / 1024) for i in 1:Nx, j in 1:Ny]
    uxe, uye = copy(ux0), copy(uy0)
    for (i, neighbour) in ((1, 2), (Nx, Nx-1)), j in 2:Ny-1
        uxe[i, j] = 0
        uye[i, j] = uy0[neighbour, j]
    end
    ux, uy = on_backend(backend, ux0), on_backend(backend, uy0)
    Kraken.enforce_free_side_macros_2d!(ux, uy, Nx, Ny)
    @test Array(ux) == uxe
    @test Array(uy) == uye

    # Both force components, both signs, non-equilibrium populations. The
    # independent oracle solves mass-independent raw-momentum constraints and
    # axial reflection as a 3x3 linear system, rather than copying the kernel.
    for force_sign in (-1, 0, 1)
        fx = [FT(force_sign * (i+j)/128) for i in 1:Nx, j in 1:Ny]
        fy = [FT(-force_sign * (2i+j)/64) for i in 1:Nx, j in 1:Ny]
        Fx, Fy = on_backend(backend, fx), on_backend(backend, fy)
        f = on_backend(backend, original)
        expected = Float64.(original)
        for (i, inward) in ((1, 1), (Nx, -1)), j in 2:Ny-1
            incoming = findall(d -> CX[d] * inward > 0, 1:9)
            known = setdiff(1:9, incoming)
            axial = only(d for d in incoming if CY[d] == 0)
            opposite = only(d for d in known if CX[d] == -CX[axial] && CY[d] == 0)
            Jx, Jy = -Float64(fx[i,j])/2, -Float64(fy[i,j])/2
            A = [Float64[CX[d] for d in incoming]'; Float64[CY[d] for d in incoming]';
                 Float64.(incoming .== axial)']
            rhs = [Jx - sum(CX[d]*original[i,j,d] for d in known),
                   Jy - sum(CY[d]*original[i,j,d] for d in known),
                   original[i,j,opposite] + (2/3)*CX[axial]*Jx]
            expected[i,j,incoming] = A \ rhs
        end
        Kraken.apply_no_slip_sidewalls_2d!(f, Fx, Fy, Nx, Ny)
        KernelAbstractions.synchronize(backend)
        actual = Array(f)
        # 64 ulps on O(1) populations covers sums and divisions in both types.
        @test maximum(abs, Float64.(actual) - expected) <= 64eps(FT)
        for i in 1:Nx, j in 1:Ny, d in 1:9
            changed = 1 < j < Ny && ((i == 1 && CX[d] > 0) || (i == Nx && CX[d] < 0))
            changed || @test actual[i,j,d] == original[i,j,d]
        end
        for i in (1, Nx), j in 2:Ny-1
            @test abs(sum(CX[d]*actual[i,j,d] for d in 1:9) + fx[i,j]/2) <= 64eps(FT)
            @test abs(sum(CY[d]*actual[i,j,d] for d in 1:9) + fy[i,j]/2) <= 64eps(FT)
        end
        # Verify the production Guo reconstruction WITHOUT a velocity override.
        rho = similar(ux)
        Kraken.compute_macroscopic_guo_field_2d!(rho, ux, uy, f, Fx, Fy, Nx, Ny)
        @test maximum(abs, Array(ux)[[1,Nx],2:Ny-1]) <= 64eps(FT)
        @test maximum(abs, Array(uy)[[1,Nx],2:Ny-1]) <= 64eps(FT)
        @test Array(Fx) == fx
        @test Array(Fy) == fy
    end
end

function poiseuille(FT, scheme, L, viscosity=1/6)
    # On-node sidewalls x=0,L; y=2:4 are periodic physical rows, y=1,5
    # are copied halos. The production sidewall kernel leaves those halos
    # alone, just as it leaves electrode rows alone in the EC driver.
    # Fully periodic pull is valid for the interior; lateral incoming values
    # are then reconstructed by the SAME kernel as the production driver.
    Nx, Ny = L+1, 5
    nu, umax = FT(viscosity), FT(0.005) # Peak Ma < 0.009.
    g = FT(8)*nu*umax/FT(L^2)
    f = zeros(FT, Nx, Ny, 9)
    weights = (4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36)
    for d in 1:9
        f[:,:,d] .= FT(weights[d])
    end
    out = similar(f)
    solid = zeros(Bool, Nx, Ny)
    Fx, Fy = zeros(FT, Nx, Ny), fill(g, Nx, Ny)
    rho, ux, uy = ones(FT, Nx, Ny), zeros(FT, Nx, Ny), zeros(FT, Nx, Ny)
    previous = zeros(FT, Nx)
    # Fo=2 leaves the slowest startup mode below exp(-2*pi^2)<3e-9.
    nsteps = ceil(Int, 2L^2/nu)
    for step in 1:nsteps
        if scheme === :bgk
            Kraken.collide_guo_field_2d!(f, solid, Fx, Fy, inv(3nu+FT(0.5)))
        else
            Kraken.ehd_collide_mrt_2d!(f, Fx, Fy, solid, nu)
        end
        Kraken.stream_fully_periodic_2d!(out, f, Nx, Ny)
        Kraken.apply_no_slip_sidewalls_2d!(out, Fx, Fy, Nx, Ny)
        out[:,1,:] .= out[:,4,:]
        out[:,5,:] .= out[:,2,:]
        f, out = out, f
        if step == nsteps-100
            Kraken.compute_macroscopic_guo_field_2d!(rho, ux, uy, f, Fx, Fy, Nx, Ny)
            previous .= uy[:,3]
        end
    end
    Kraken.compute_macroscopic_guo_field_2d!(rho, ux, uy, f, Fx, Fy, Nx, Ny)
    x = FT.(0:L)
    reference = g .* x .* (FT(L) .- x) ./ (2nu)
    profile = uy[:,3]
    err = norm(profile-reference)/norm(reference)
    residual = maximum(abs, profile-previous)/umax
    @info "EC sidewall Poiseuille" FT scheme L nu nsteps err residual
    # Prospective relative error budget: 0.01% (Float64), 0.5% (Float32).
    # Residual must be 10x smaller; do not infer order from an exact parabola.
    gate = FT === Float64 ? 1e-4 : 5e-3
    @test all(isfinite, f)
    @test err <= gate
    @test residual <= gate/10
    @test maximum(abs, profile[[1,Nx]]) <= 64eps(FT)
    @test maximum(abs, ux[:,3]) <= 64eps(FT)
    @test maximum(abs, rho[:,3] .- 1) <= gate
end

function hydrostatic_sidewalls(FT, scheme, force_sign; backend=CPU())
    # PR #37 review: independent physical check of the wall-NORMAL force.
    # Constant force density, not acceleration: dp/dx=Fx, p=rho/3.
    # Start uniform, allowing the pressure gradient to develop dynamically.
    # No fitted slope/wall shift, force projection or velocity overwrite.
    L, Ny = 16, 5
    Nx = L + 1
    nu, force = FT(0.1), FT(force_sign) * FT(1e-4)
    f = zeros(FT, Nx, Ny, 9)
    weights = (4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36)
    for d in 1:9
        f[:,:,d] .= FT(weights[d])
    end
    # Transfer the identical initial populations once; every time-step kernel
    # dispatches from these arrays, so a CUDA request cannot run on CPU arrays.
    f = on_backend(backend, f)
    out = similar(f)
    solid = on_backend(backend, zeros(Bool, Nx, Ny))
    Fx = on_backend(backend, fill(force, Nx, Ny))
    Fy = on_backend(backend, zeros(FT, Nx, Ny))
    rho = on_backend(backend, ones(FT, Nx, Ny))
    ux = on_backend(backend, zeros(FT, Nx, Ny))
    uy = similar(ux)
    fill!(uy, zero(FT))
    @test all(a -> typeof(KernelAbstractions.get_backend(a)) === typeof(backend),
              (f, out, solid, Fx, Fy, rho, ux, uy))
    @test eltype(f) === FT
    previous_rho, previous_ux, previous_uy = (zeros(FT, Nx, 3) for _ in 1:3)
    nsteps = ceil(Int, 4L^2/nu)
    for step in 1:nsteps
        if scheme === :bgk
            Kraken.collide_guo_field_2d!(f, solid, Fx, Fy, inv(3nu+FT(0.5)))
        else
            Kraken.ehd_collide_mrt_2d!(f, Fx, Fy, solid, nu)
        end
        Kraken.stream_fully_periodic_2d!(out, f, Nx, Ny)
        Kraken.apply_no_slip_sidewalls_2d!(out, Fx, Fy, Nx, Ny)
        @views out[:,1,:] .= out[:,4,:]
        @views out[:,5,:] .= out[:,2,:]
        f, out = out, f
        if step == nsteps-100
            Kraken.compute_macroscopic_guo_field_2d!(rho, ux, uy, f, Fx, Fy, Nx, Ny)
            KernelAbstractions.synchronize(backend)
            previous_rho .= Array(rho)[:,2:4]
            previous_ux .= Array(ux)[:,2:4]
            previous_uy .= Array(uy)[:,2:4]
        end
    end
    Kraken.compute_macroscopic_guo_field_2d!(rho, ux, uy, f, Fx, Fy, Nx, Ny)
    # Assess in Float64 to avoid adding low-precision cancellation to the
    # measured error. Solver populations and arithmetic still use FT.
    KernelAbstractions.synchronize(backend)
    actual = Float64.(Array(rho)[:,2:4])
    actual_ux = Float64.(Array(ux)[:,2:4])
    actual_uy = Float64.(Array(uy)[:,2:4])
    mean_density = sum(actual)/length(actual)
    reference = 3Float64(force) .* (collect(0:L) .- L/2)
    expected = repeat(reshape(reference, Nx, 1), 1, 3)
    profile_error = norm(actual .- mean_density .- expected)/norm(expected)
    gradient_error = maximum(abs, diff(actual; dims=1) ./ (3Float64(force)) .- 1)
    speed = maximum(hypot.(actual_ux, actual_uy))
    density_change = maximum(abs, actual-Float64.(previous_rho))/(3abs(Float64(force))*L)
    velocity_change = maximum(hypot.(actual_ux-Float64.(previous_ux),
                                     actual_uy-Float64.(previous_uy)))
    @info "EC sidewall hydrostatic" backend population_type=typeof(f) FT scheme force L nu nsteps profile_error gradient_error speed density_change velocity_change mean_density
    # Prospective budgets, frozen 2026-09-22 before execution: 0.01%/0.5%
    # profile and local-gradient error. Absolute speed in lattice units.
    # This is NOT an electrostatic, corner or onset validation claim.
    # CUDA coverage exists only when the explicit opt-in below is executed.
    gate = FT === Float64 ? 1e-4 : 5e-3
    speed_gate = FT === Float64 ? 1e-8 : 2e-6
    # Prospective Float32 revision agreed in PR #37, 2026-09-22:
    # https://github.com/lauguimel/Kraken.jl/pull/37#issuecomment-5774111607
    # Momentum cancellation from O(rho) populations/moments gives a velocity
    # noise scale of O(ulp(rho)) per step near rho=1, not eps of the tiny u.
    # Run 35686553469 (CPU Julia 1.11.9/1.12.7) recorded max du100=8.32e-7
    # over logged Fo=2..16 samples. Use the agreed 1e-6 LU budget (~8 ulp(1));
    # this is bounded fixture evidence, not a universal roundoff bound.
    # Float64 and all spatial/absolute-speed gates remain unchanged.
    velocity_gate = FT === Float64 ? speed_gate/10 : 1e-6
    @test all(isfinite, f)
    @test all(isfinite, rho) && all(isfinite, ux) && all(isfinite, uy)
    @test minimum(actual) > 0
    @test profile_error <= gate
    @test gradient_error <= gate
    @test speed <= speed_gate
    # Retain the original density-change gate at Fo=4. Float32 mean-density
    # drift is logged above; this local change check is NOT mass conservation
    # or arbitrary-duration stationarity. Reassess, do not silently relax,
    # if future platforms or longer trajectories exceed this budget.
    @test density_change <= gate/10
    @test velocity_change <= velocity_gate
    return (; density=actual .- mean_density, ux=actual_ux, uy=actual_uy,
            reference_norm=norm(expected), profile_gate=gate, speed_gate,
            velocity_change, mean_density)
end

function coupled(backend, FT)
    # Fixed potential work isolates this capability from known stopping #23.
    # Short-run checks are not physical onset/steady-state acceptance.
    common = (Nx=9, Ny=17, C=0.1, M=10.0, T=175.0, Ma_E=0.01,
              alpha=1e-4, max_cycles=4, phi_substeps=32,
              backend=backend, FT=FT)
    for scheme in (:bgk, :mrt)
        default = Kraken.run_electroconvection_2d(; common..., ns_scheme=scheme)
        free = Kraken.run_electroconvection_2d(; common..., ns_scheme=scheme, sidewall_bc=:free_slip)
        rigid = Kraken.run_electroconvection_2d(; common..., ns_scheme=scheme, sidewall_bc=:no_slip)
        @test default.sidewall_bc === free.sidewall_bc === :free_slip_ported
        @test rigid.sidewall_bc === :no_slip
        for name in (:ux, :uy, :rho, :q, :phi, :Ex, :Ey, :Fx, :Fy)
            @test getproperty(default, name) == getproperty(free, name)
            @test all(isfinite, getproperty(rigid, name))
        end
        @test rigid.steps == 4
        @test rigid.A == FT(0.5)
        @test maximum(abs, rigid.ux[[1,end],2:end-1]) <= 64eps(FT)
        @test maximum(abs, rigid.uy[[1,end],2:end-1]) <= 64eps(FT)
        @test maximum(abs, free.uy[[1,end],2:end-1]) > 0
        @test rigid.ux[:,[1,end]] == zeros(FT,9,2)
        @test rigid.uy[:,[1,end]] == zeros(FT,9,2)
    end
end

function public_dispatch()
    base = """
    Simulation electroconvection_sidewalls D2Q9
    Domain L = 1.0 x 1.0 N = 9 x 17
    Physics C = 0.1 M = 10.0 T = 175.0 Ma_E = 0.01 alpha = 1e-4 phi_scheme = direct
    Module ehd
    Run 2 steps
    """
    # Exercise the real parser + public runner, not just the lowering helper.
    mktempdir() do dir
        fixture = joinpath(dir, "electroconvection_sidewalls.krk")
        for (decl, mode) in (("", :free_slip),
                             ("Boundary west symmetry\nBoundary east symmetry\n", :free_slip),
                             ("Boundary west wall\nBoundary east wall\n", :no_slip))
            write(fixture, base * decl)
            actual = Kraken.run_simulation(fixture; backend=CPU(), T=Float64)
            expected = Kraken.run_electroconvection_2d(Nx=9, Ny=17, C=0.1,
                M=10.0, T=175.0, Ma_E=0.01, alpha=1e-4, phi_scheme=:direct,
                max_cycles=2, sidewall_bc=mode, backend=CPU(), FT=Float64)
            @test actual.sidewall_bc === expected.sidewall_bc
            for name in (:ux, :uy, :rho, :q, :phi, :Ex, :Ey)
                @test getproperty(actual, name) == getproperty(expected, name)
            end
        end
        for decl in ("Boundary north wall\n",
                     "Boundary south wall\n",
                     "Boundary west wall\nBoundary east wall\nBoundary north wall\n",
                     "Boundary west symmetry\nBoundary east symmetry\nBoundary south wall\n",
                     "Boundary west wall\n",
                     "Boundary west wall\nBoundary east symmetry\n",
                     "Boundary west wall\nBoundary east wall\nBoundary west wall\n",
                     "Boundary x periodic\n",
                     "Boundary west pressure(rho=1)\nBoundary east wall\n",
                     "Boundary west wall(ux=0.1)\nBoundary east wall\n")
            write(fixture, base * decl)
            @test_throws ArgumentError Kraken.run_simulation(fixture; backend=CPU(), T=Float64)
        end
    end
    for mode in (:invalid, :free_slip_ported, "no_slip")
        @test_throws ArgumentError Kraken.run_electroconvection_2d(Nx=9, Ny=17, sidewall_bc=mode)
    end
end

if !isdefined(Kraken, :_legacy_run_electroconvection_2d)
    Base.include(Kraken, joinpath(@__DIR__, "..", "reference", "ehd_ec_legacy.jl"))
end

function state_segments(FT)
    # Same backend/type/order: exact equality, including populations and carried
    # force. No tolerance can hide a reset or lost boundary selection here.
    for mode in (:free_slip, :no_slip), scheme in (:bgk, :mrt), path in (:lbm, :direct)
        common = (Nx=9, Ny=17, C=0.1, M=10.0, T=175.0, Ma_E=0.01,
                  alpha=1e-4, phi_scheme=path,
                  phi_substeps=(path === :lbm ? 32 : nothing),
                  ns_scheme=scheme, history_interval=2, backend=CPU(), FT=FT)
        whole = init_state(ECState; common..., sidewall_bc=mode)
        split = init_state(ECState; common..., sidewall_bc=mode)
        @test whole.config.sidewall_bc === split.config.sidewall_bc === mode
        advance!(whole, 7; sample_final=true)
        advance!(split, 3)
        intermediate = solution(split).result
        @test intermediate.sidewall_bc === (mode === :free_slip ? :free_slip_ported : :no_slip)
        advance!(split, 4; sample_final=true)
        @test split.cycle == 7
        @test split.cycle_history == [2, 4, 6, 7]
        @test Kraken.at_boundary(split)
        for name in (:phi_f_in, :q_f_in, :f_in, :Fx_prev, :Fy_prev,
                     :phi, :qfield, :phi_iters_last, :phi_rel_last, :q_rel_last,
                     :cycle, :cycle_history, :umax_history)
            @test isequal(getproperty(whole, name), getproperty(split, name))
        end
        a, b = solution(whole).result, solution(split).result
        wrapped = Kraken.run_electroconvection_2d(; common..., sidewall_bc=mode, max_cycles=7)
        for name in keys(a)
            name === :loop_ms_per_step && continue
            @test isequal(a[name], b[name])
            @test isequal(a[name], wrapped[name])
        end
        if mode === :free_slip
            legacy = Kraken._legacy_run_electroconvection_2d(; common..., max_cycles=7)
            @test keys(a) == keys(legacy)
            for name in keys(a)
                name === :loop_ms_per_step && continue
                @test isequal(a[name], legacy[name])
            end
        else
            # Check raw populations with carried force, not only solution fields.
            Kraken.compute_macroscopic_guo_field_2d!(split.rho, split.ux, split.uy,
                split.f_in, split.Fx_prev, split.Fy_prev, 9, 17)
            @test maximum(abs, split.ux[[1,end],2:end-1]) <= 64eps(FT)
            @test maximum(abs, split.uy[[1,end],2:end-1]) <= 64eps(FT)
        end
        # Preserve the upstream failed-state contract for each selected mode.
        split.at_boundary = false
        saved = map(name -> copy(getproperty(split, name)),
                    (:f_in, :q_f_in, :phi_f_in, :Fx_prev, :Fy_prev))
        @test_throws ArgumentError advance!(split, 0)
        @test_throws ArgumentError advance!(split, 1)
        @test_throws ArgumentError solution(split)
        @test split.cycle == 7
        @test !Kraken.at_boundary(split)
        for (name, before) in zip((:f_in, :q_f_in, :phi_f_in, :Fx_prev, :Fy_prev), saved)
            @test isequal(getproperty(split, name), before)
        end
    end
    @test_throws ArgumentError init_state(ECState; Nx=9, Ny=17, FT=FT, sidewall_bc=:invalid)
end

const HYDROSTATIC_CPU = Dict{Tuple{DataType,Symbol,Int},Any}()

@testset "EC-SIDEWALLS CPU" begin
    for FT in (Float64, Float32)
        @testset "population operators $FT" begin
            operators(CPU(), FT)
        end
        @testset "Poiseuille $FT $scheme L=$L nu=$nu" for scheme in (:bgk, :mrt), L in (8, 16), nu in (1/6, 0.1)
            poiseuille(FT, scheme, L, nu)
        end
        @testset "hydrostatic $FT $scheme sign=$force_sign" for scheme in (:bgk, :mrt), force_sign in (-1, 1)
            HYDROSTATIC_CPU[(FT, scheme, force_sign)] =
                hydrostatic_sidewalls(FT, scheme, force_sign)
        end
        @testset "coupled $FT" begin
            coupled(CPU(), FT)
        end
        @testset "state segments $FT" begin
            state_segments(FT)
        end
    end
    @testset "public configuration" begin
        public_dispatch()
    end
end

# Explicit opt-in: no CUDA qualification is claimed by ordinary CPU CI.
# Run from the repository root on a GPU compute node (not the Pkg.test sandbox):
# export JULIA_CUDA_USE_COMPAT=false
# julia --project=. -e 'using Pkg; Pkg.instantiate()'
# KRAKEN_TEST_EHD_SIDEWALLS_CUDA=true julia --project=. -e 'include("test/analytical/EC-SIDEWALLS.jl")'
if get(ENV, "KRAKEN_TEST_EHD_SIDEWALLS_CUDA", "false") == "true"
    @eval using CUDA
    CUDA.functional() || error("Requested EC sidewall CUDA tests, but CUDA is unavailable")
    CUDA.allowscalar(false)
    @info "EC-SIDEWALLS CUDA identity" julia_version=VERSION kraken_path=pathof(Kraken) device=CUDA.device()
    CUDA.versioninfo()
    @testset "EC-SIDEWALLS CUDA" for FT in (Float64, Float32)
        operators(CUDA.CUDABackend(), FT)
        coupled(CUDA.CUDABackend(), FT)
        @testset "hydrostatic $FT $scheme sign=$force_sign" for scheme in (:bgk, :mrt), force_sign in (-1, 1)
            gpu = hydrostatic_sidewalls(FT, scheme, force_sign; backend=CUDA.CUDABackend())
            cpu = HYDROSTATIC_CPU[(FT, scheme, force_sign)]
            # Both backends must independently satisfy the analytical gates.
            # Triangle-inequality parity budgets are frozen before CUDA runs;
            # no bitwise agreement or long-time mass-conservation claim.
            density_parity = norm(gpu.density-cpu.density)/cpu.reference_norm
            velocity_parity = maximum(hypot.(gpu.ux-cpu.ux, gpu.uy-cpu.uy))
            @info "EC sidewall hydrostatic CPU/CUDA parity" FT scheme force_sign density_parity velocity_parity cpu_du100=cpu.velocity_change cuda_du100=gpu.velocity_change
            @test density_parity <= 2cpu.profile_gate
            @test velocity_parity <= 2cpu.speed_gate
        end
    end
end
end # module
