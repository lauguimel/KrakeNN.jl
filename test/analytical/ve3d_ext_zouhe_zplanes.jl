# Bug report: the open-face Zou-He rebuilds of run_viscoelastic_fvfd_extensional_3d
# skip the k = 1 and k = Nz planes although z is periodic (issue #69).
#
# The west kernel (src/bc/rebuild_3d.jl:2, launched at :95-96) and the east,
# south and north kernels (src/fvfd/operators_3d_openbc.jl, launched at :338-343)
# use k = km1 + 1 on ndrange (N - 2, Nz - 2), so they rebuild k = 2..Nz-1 only,
# without a periodic wrap. The face cells at k = 1 and k = Nz keep the
# PullHalfwayBB_3D{true} x/y bounce-back with zero wall velocity: they act as
# stationary no-slip walls, and the flow stops being invariant in z.
#
# Case: extensional driver, E2 parameters on a small box (12 x 12 x 4,
# eps_dot = 0.005, nu_s = nu_p = 0.05, lambda = 50), coupled mode, 20 steps.
# The initial state (analytical straining field, uz = 0, C = I), the face data
# and every other operator of the step are z-invariant and z-periodic, so the
# exact discrete solution is z-invariant.
#
# Frozen gate: max over (i, j, k) of |u(i, j, k) - u(i, j, 2)| <= 1e-12 for ux and
# uy, and max |uz| <= 1e-12.
# Origin: symmetry (the discrete problem is invariant under z-translation), so
# the gate is round-off. Measured with the current driver (CPU, Float64,
# 2026-09-28, Apple M3 Max, Julia 1.12.5), after 20 steps: spread 2.46e-2 in ux,
# 2.57e-2 in uy, max |uz| = 4.0e-3; east-face ux at mid-height is 0.0072 on
# k = 1 and k = 4 against 0.0300 on k = 2 and k = 3.
#
# What this test does not protect. It only shows that the missing planes break
# the z symmetry; it does not validate a candidate fix.
#  - The west kernel is shared (through apply_bc_rebuild_3d!) with drivers whose
#    k = 1 and k = Nz planes are no-slip z walls: viscoelastic_3d.jl:245,
#    obstacle_3d.jl:154, li_bb_3d_v2.jl:238 and :348. A fix that rebuilds every
#    k plane for all callers also turns this test into an Unexpected Pass while
#    it rebuilds the wall planes of those drivers. The k-range extension and
#    the periodic wrap must apply only when z is periodic.
#  - With a z-invariant state, clamping the k-1 / k+1 reads gives the same
#    values as a periodic wrap (deduced from the symmetry, not measured), so a
#    clamp would also pass here although it is wrong for flows that vary in z.
#
# CPU Float64 only: the defect is which planes a kernel covers, which does not
# depend on backend or precision; the GPU runs the same kernels. A Float32
# variant would need a gate above Float32 round-off and would add no
# information.

using Test
using Kraken

@testset "Extensional 3D driver keeps a z-invariant flow z-invariant" begin
    Nx, Ny, Nz = 12, 12, 4
    gate = 1e-12
    res = Kraken.run_viscoelastic_fvfd_extensional_3d(; Nx, Ny, Nz,
        epsilon_dot=0.005, ν_s=0.05, ν_p=0.05, lambda=50.0, max_steps=20,
        velocity_mode=:coupled)
    spread_x = maximum(abs, res.ux .- res.ux[:, :, 2:2])
    spread_y = maximum(abs, res.uy .- res.uy[:, :, 2:2])
    uz_max = maximum(abs, res.uz)
    # Face values on the east face at mid-height, per plane (target eps*(Nx-xc)).
    east = [res.ux[Nx, Ny ÷ 2, k] for k in 1:Nz]
    println("z-planes: spread ux=$(spread_x) uy=$(spread_y) max|uz|=$(uz_max) ",
            "east-face ux per k=$(east) target=$(0.005 * (Nx - (Nx + 1) / 2))")

    @test res.completed_steps == 20
    @test all(isfinite, res.ux) && all(isfinite, res.uy) && all(isfinite, res.uz)
    # Defect.
    @test_broken spread_x <= gate && spread_y <= gate && uz_max <= gate
end
