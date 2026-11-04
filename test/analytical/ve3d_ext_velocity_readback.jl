# Bug report: run_viscoelastic_fvfd_extensional_3d reads the velocity back from
# the POST-collision populations (issue #68).
#
# src/drivers/viscoelastic_fvfd_extensional_3d.jl:268 (and :293 after the loop)
# calls compute_macroscopic_forced_field_3d! on f_out, i.e. after the fused
# TRT-Guo collision, and adds F/2. The fused step has already written the Guo
# velocity (sum_q c_q f_pre + F/2)/rho of the populations it collided; line 268
# overwrites it with a value that is off by (sum_q c_q (f_out - f_pre))/rho, which
# is g*F/rho with the current brick and would still be F/rho with a corrected one.
# The overwritten velocity is the one the next step's constitutive update uses
# and the one the driver returns.
#
# Case: E2 parameters on a small box (12 x 12 x 4, eps_dot = 0.005, nu_s = nu_p =
# 0.05, lambda = 50), coupled mode, 2 and 3 steps. The polymer force is zero to
# round-off at step 1 (max 1.1e-19, tau_p is uniform) and up to 1.6e-5 on the
# compared cells at step 2. The test replays the two fused steps from the
# driver's initial state with the force recomputed from the returned tau_p, and
# compares on the cells i, j in 3:N-2 (never touched by the Zou-He face
# rebuilds, and whose step-2 pull reads only cells the step-1 fused kernel
# wrote):
#   d2    = returned ux, uy after 2 steps  vs  the step-2 collision velocity;
#   dloop = duxdx, duydy the driver computes at the start of step 3 from the
#           velocity step 2 left in the loop (driver :204)  vs  the same
#           gradient of the step-2 collision velocity, on i, j in 4:N-3 (the
#           centred stencil there reads only cells 3:N-2).
# d2 alone would also flip for a partial fix that returns a saved collision
# velocity but keeps the in-loop read-back; dloop checks the velocity the
# polymer update actually receives.
#
# The replay copies the driver's recipe: initial state (rho = 1 equilibrium of
# the analytical straining field, as in viscoelastic_fvfd_extensional_3d.jl:85-98),
# the force call and its keyword arguments, nu_s and periodic_z=true in the
# fused step. If a fix changes that recipe, the replay must follow it, or this
# test stays Broken after a correct fix.
#
# Frozen gate: max |difference| <= 1e-12 for d2 and dloop.
# Origin: in each pair the two sides are the same quantity by definition, so the
# gate is round-off. Floor measured at step 1, where F = 0 (CPU, Float64,
# 2026-09-28, Apple M3 Max, Julia 1.12.5): 4.2e-17. Measured with the current
# driver: d2 = 8.23e-6 (max over ux, uy; cell by cell the difference equals
# g*F/rho with g = 0.516484 to 3.6e-17) and dloop = 3.73e-6.
#
# Controls (plain @test, must pass before and after the fix):
#  - d1 <= gate: returned velocity after 1 step vs the step-1 collision
#    velocity. This checks the replayed initial state and streaming; it relies
#    on F being zero at step 1 (uniform tau_p). If a change makes the step-1
#    force non-zero on these cells, d1 becomes g*F1/rho and this control must
#    be revisited, not loosened.
#  - drho2 <= gate: returned density after 2 steps vs the density of the
#    replayed step-2 collision. The Guo source has no mass moment, so none of
#    the three known defects changes it; it checks that the replay reproduces
#    the step-1 collision and the step-2 streaming. Measured: 4.4e-16.
#  - max |F| (x and y) on those cells at step 2 >= 1e-9, so that the defect
#    (g*F/rho with g = 0.516 here, F/rho once the brick is fixed, rho ~ 1)
#    stays at least 500 times above the gate. The step-2 force comes partly
#    from the boundary defect of issue #69: with the k = 1 and
#    k = Nz rebuilds added in a scratch copy it drops from 1.6e-5 to 2.0e-6
#    (measured, same machine and date), hence a threshold far below both.
#
# CPU Float64 only: the defect is algebraic (which populations the velocity is
# read from) and does not depend on backend or precision; the GPU runs the same
# kernels. A Float32 variant would need gates above Float32 round-off and would
# add no information.

using Test
using Kraken

@testset "Extensional 3D driver returns the collision velocity" begin
    Nx, Ny, Nz = 12, 12, 4
    eps_dot = 0.005; nu_s = 0.05; nu_p = 0.05; lam = 50.0
    gate = 1e-12
    fmin = 1e-9

    run_n(n) = Kraken.run_viscoelastic_fvfd_extensional_3d(; Nx, Ny, Nz,
        epsilon_dot=eps_dot, ν_s=nu_s, ν_p=nu_p, lambda=lam, max_steps=n,
        velocity_mode=:coupled)

    function force_from(res)
        Fx = zeros(Nx, Ny, Nz); Fy = zeros(Nx, Ny, Nz); Fz = zeros(Nx, Ny, Nz)
        Kraken.compute_polymeric_force_3d!(Fx, Fy, Fz,
            res.tau_p_xx, res.tau_p_xy, res.tau_p_xz,
            res.tau_p_yy, res.tau_p_yz, res.tau_p_zz;
            periodic_x=false, periodic_z=true)
        return Fx, Fy, Fz
    end

    is_solid = zeros(Bool, Nx, Ny, Nz)
    q_wall = zeros(Nx, Ny, Nz, 19); uw = zeros(Nx, Ny, Nz, 19)
    function fused_step(f_in, F)
        f_out = zeros(Nx, Ny, Nz, 19)
        rho = ones(Nx, Ny, Nz); ux = zeros(Nx, Ny, Nz)
        uy = zeros(Nx, Ny, Nz); uz = zeros(Nx, Ny, Nz)
        Kraken.fused_trt_libb_v2_guo_field_step_3d!(f_out, f_in, rho, ux, uy, uz,
            is_solid, q_wall, uw, uw, uw, F[1], F[2], F[3], Nx, Ny, Nz, nu_s;
            periodic_z=true)
        return f_out, rho, ux, uy, uz
    end

    # Same operator and arguments as the driver's gradient call (driver :204).
    function grad_xx_yy(ux, uy, uz)
        g = [zeros(Nx, Ny, Nz) for _ in 1:9]
        Kraken.fvfd_velocity_gradient_3d!(g..., ux, uy, uz, is_solid;
            dx=1.0, dy=1.0, dz=1.0, x_bc=:open, y_bc=:open, z_bc=:periodic,
            sync=true)
        return g[1], g[5]   # duxdx, duydy
    end

    # Driver initial state, as in viscoelastic_fvfd_extensional_3d.jl:85-98.
    vel = Kraken.fvfd_planar_extensional_velocity_field_host_3d(Nx, Ny, Nz, eps_dot)
    f0 = zeros(Nx, Ny, Nz, 19)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx, q in 1:19
        f0[i, j, k, q] = Kraken.equilibrium(D3Q19(), 1.0,
            vel.ux[i, j, k], vel.uy[i, j, k], 0.0, q)
    end

    r1 = run_n(1)
    r2 = run_n(2)
    r3 = run_n(3)
    F1 = force_from(r1)
    F2 = force_from(r2)
    f1, _, ux1, uy1, _ = fused_step(f0, F1)
    _, rho2, ux2, uy2, uz2 = fused_step(f1, F2)

    ic = 3:Nx-2; jc = 3:Ny-2
    d1 = max(maximum(abs, r1.ux[ic, jc, :] .- ux1[ic, jc, :]),
             maximum(abs, r1.uy[ic, jc, :] .- uy1[ic, jc, :]))
    d2 = max(maximum(abs, r2.ux[ic, jc, :] .- ux2[ic, jc, :]),
             maximum(abs, r2.uy[ic, jc, :] .- uy2[ic, jc, :]))
    drho2 = maximum(abs, r2.ρ[ic, jc, :] .- rho2[ic, jc, :])
    fmax = max(maximum(abs, F2[1][ic, jc, :]), maximum(abs, F2[2][ic, jc, :]))

    gx2, gy2 = grad_xx_yy(ux2, uy2, uz2)
    ig = 4:Nx-3; jg = 4:Ny-3
    dloop = max(maximum(abs, r3.duxdx[ig, jg, :] .- gx2[ig, jg, :]),
                maximum(abs, r3.duydy[ig, jg, :] .- gy2[ig, jg, :]))
    println("readback: step1 diff=$(d1) (F=0)  step2 diff=$(d2)  step2 rho diff=$(drho2)  ",
            "in-loop gradient diff=$(dloop)  max|Fx|,|Fy| step2=$(fmax)")

    # Controls: the replay reproduces the driver (step 1 velocity, step 2
    # density), and the step-2 force is large enough for the defect to show far
    # above the gate.
    @test d1 <= gate
    @test drho2 <= gate
    @test fmax >= fmin
    # Defect: returned velocity, and velocity fed to the next polymer update.
    @test_broken d2 <= gate
    @test_broken dloop <= gate
end
