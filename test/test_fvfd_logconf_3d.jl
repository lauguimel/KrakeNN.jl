using Test
using KernelAbstractions
using Kraken

const CELL_3D = (1, 1, 1)

field3(value) = fill(Float64(value), CELL_3D)

function psi_state(; xx=0.0, xy=0.0, xz=0.0, yy=0.0, yz=0.0, zz=0.0)
    return (
        field3(xx), field3(xy), field3(xz),
        field3(yy), field3(yz), field3(zz),
    )
end

function velocity_gradient(;
    duxdx=0.0, duxdy=0.0, duxdz=0.0,
    duydx=0.0, duydy=0.0, duydz=0.0,
    duzdx=0.0, duzdy=0.0, duzdz=0.0,
)
    return (
        field3(duxdx), field3(duxdy), field3(duxdz),
        field3(duydx), field3(duydy), field3(duydz),
        field3(duzdx), field3(duzdy), field3(duzdz),
    )
end

function psi_values(psi)
    return ntuple(c -> psi[c][1, 1, 1], 6)
end

function conformation_from_psi(psi)
    return Kraken.mat_exp_sym3x3(psi_values(psi)...)
end

function apply_constitutive_step!(psi, grad, lambda, dt, n_sub)
    before = psi_values(psi)
    out = ntuple(c -> similar(psi[c]), 6)
    Kraken.logfv_constitutive_step_log_3d!(
        out...,
        psi...,
        grad...,
        lambda, dt, n_sub;
        sync=true,
    )
    for c in 1:6
        psi[c] .= out[c]
    end
    after = psi_values(psi)
    return maximum(abs(after[c] - before[c]) for c in 1:6)
end

function run_uniform_shear_fixed_point(Wi; lambda=1.0, dt=0.05)
    gamma_dot = Wi / lambda
    psi = psi_state()
    grad = velocity_gradient(duxdy=gamma_dot)
    max_grad_norm = Kraken.logfv_max_grad_norm_3d(grad...)
    estimate = Kraken.logfv_oldroydb_subcycle_estimate_3d(max_grad_norm, lambda, dt)
    n_sub = estimate.recommended
    max_steps = ceil(Int, 30lambda / dt)
    t = 0.0
    delta = Inf
    steps = 0

    while steps < max_steps && delta >= 1e-12
        delta = apply_constitutive_step!(psi, grad, lambda, dt, n_sub)
        steps += 1
        t += dt
    end

    return (C=conformation_from_psi(psi), n_sub=n_sub, steps=steps, time=t, delta=delta)
end

function run_relaxation_decay(; lambda=1.0, dt=0.005, t_final=5.0)
    psi = psi_state(xx=log(2.0))
    grad = velocity_gradient()
    estimate = Kraken.logfv_oldroydb_subcycle_estimate_3d(0.0, lambda, dt)
    n_sub = estimate.recommended
    n_steps = ceil(Int, t_final / dt)
    max_error = 0.0
    t = 0.0

    for _ in 1:n_steps
        apply_constitutive_step!(psi, grad, lambda, dt, n_sub)
        t += dt
        cxx = conformation_from_psi(psi)[1]
        expected = 1.0 + exp(-t / lambda)
        max_error = max(max_error, abs(cxx - expected))
    end

    return (max_error=max_error, n_sub=n_sub, final_time=t)
end

@testset "FVFD 3D log-conformation constitutive step" begin
    @testset "V2 uniform-shear exact fixed point" begin
        for Wi in (0.5, 2.0)
            result = run_uniform_shear_fixed_point(Wi)
            cxx, cxy, cxz, cyy, cyz, czz = result.C
            target_cxy = Wi
            target_cxx = 1.0 + 2.0 * Wi^2
            err_cxy = abs(cxy - target_cxy) / target_cxy
            err_cxx = abs(cxx - target_cxx) / target_cxx

            println(
                "V2 Wi=$(Wi) n_sub=$(result.n_sub) steps=$(result.steps) ",
                "err_Cxy=$(err_cxy) err_Cxx=$(err_cxx) ",
                "abs_Cyy=$(abs(cyy - 1.0)) abs_Czz=$(abs(czz - 1.0)) ",
                "abs_Cxz=$(abs(cxz)) abs_Cyz=$(abs(cyz))",
            )

            @test err_cxy <= 1e-6
            @test err_cxx <= 1e-6
            @test abs(cyy - 1.0) <= 1e-9
            @test abs(czz - 1.0) <= 1e-9
            @test abs(cxz) <= 1e-9
            @test abs(cyz) <= 1e-9
        end
    end

    @testset "V2b uniform-shear dt independence" begin
        coarse = run_uniform_shear_fixed_point(0.5; dt=0.05)
        fine = run_uniform_shear_fixed_point(0.5; dt=0.0125)
        cxy_coarse = coarse.C[2]
        cxy_fine = fine.C[2]
        diff = abs(cxy_coarse - cxy_fine)
        println(
            "V2b Cxy(dt=0.05)=$(cxy_coarse) ",
            "Cxy(dt=0.0125)=$(cxy_fine) diff=$(diff)",
        )
        @test diff <= 1e-6
        @test abs(cxy_coarse - 0.5) / 0.5 <= 1e-6
        @test abs(cxy_fine - 0.5) / 0.5 <= 1e-6
    end

    @testset "V3 relaxation decay" begin
        result = run_relaxation_decay()
        println(
            "V3 relaxation max_decay_error=$(result.max_error) ",
            "n_sub=$(result.n_sub) final_time=$(result.final_time)",
        )
        @test result.max_error <= 1e-3
    end
end

println("EXIT=0")
