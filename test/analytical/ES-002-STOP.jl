# Issue #23: potential convergence does not imply field-moment convergence.
# The capacitor fixtures run the production adaptive solve
# (`Kraken.ehd_phi_ddf_solve!`) at the drivers' check cadences; since the #23
# repair its stopping rule also checks the field, and the cold starts meet the
# field gate. The same fixtures with `field_tol = Inf` (the pre-#23 rule) are the
# negative control. The public-driver check at default settings remains broken
# for a different reason (see its comment).
# CPU Float64 regression; no GPU, charge-transport or onset qualification.
# Run: julia --project test/analytical/ES-002-STOP.jl
module ES002FieldStoppingRegression

using Test
using Kraken
using KernelAbstractions

const CPU_BACKEND = KernelAbstractions.CPU()
const H = 16
const FIELD_GATE = 1e-6
const WEIGHTS = (4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36)
const CY = (0, 0, 1, 0, -1, 1, 1, -1, -1)

# The drivers' check cadences: every 8 iterations in the EC box (:neumann),
# every iteration in the hydrostatic driver (:periodic).
check_every(xbc) = xbc === :neumann ? Kraken.EHD_EC_PHI_CHECK_EVERY :
                                      Kraken.EHD_HYDROSTATIC_PHI_CHECK_EVERY

function capacitor(xbc, tau, initialization; field_tol=1e-10)
    nx = xbc === :neumann ? 11 : 10
    ny = H + 1
    reference = [1.0-(j-1)/H for i in 1:nx, j in 1:ny]
    f = zeros(nx, ny, 9)
    # Use the real driver initializer for the cold populations.
    Kraken._fill_phi_populations!(f, vec(reference[1, :]), nx, ny, Float64)
    if initialization === :consistent
        for k in 1:9
            f[:, :, k] .+= WEIGHTS[k] * tau * CY[k] / H
        end
    end
    g = copy(f)
    phi = copy(reference)
    q = zeros(nx, ny)
    ex, ey = zeros(nx, ny), zeros(nx, ny)
    # The production adaptive solve (the one both drivers call), with this
    # fixture's tolerance 1e-10 on both moments and its 128-iteration cap.
    p = (; eps=0.7, omega_U=inv(tau), nu_U=(tau-0.5)/3, tau_U=tau)
    ws = Kraken.ehd_phi_ddf_workspace(phi)
    f, g, st = Kraken.ehd_phi_ddf_solve!(f, g, phi, q, p, xbc, ws;
                                         phi_tol=1e-10, field_tol=field_tol, max_iter=128,
                                         check_every=check_every(xbc),
                                         phi_bottom=1.0, phi_top=0.0)
    KernelAbstractions.synchronize(CPU_BACKEND)
    accepted = st.converged
    n = st.iters
    relative_change = st.phi_rel
    # Independent reconstruction of E from the returned populations.
    Kraken.compute_electric_field_2d!(ex, ey, f, tau)
    KernelAbstractions.synchronize(CPU_BACKEND)
    ex .*= H
    ey .*= H
    field_error = max(maximum(abs, ex), maximum(abs.(ey .- 1.0)))
    # Independent recurrence: a_(n+1)=(1-1/tau)*a_n+1/H, a_0=0.
    prediction = initialization === :cold ? 1-(1-1/tau)^n : 1.0
    prediction_error = max(maximum(abs, ex), maximum(abs.(ey .- prediction)))
    return (; accepted, n, relative_change, field_error, prediction_error,
            phi_error=maximum(abs.(phi .- reference)),
            finite=all(isfinite, f) && all(isfinite, ex) && all(isfinite, ey))
end

field_difference(a, b) = H * max(maximum(abs.(a.Ex .- b.Ex)),
                               maximum(abs.(a.Ey .- b.Ey)))

@testset "ES-002-STOP: field-aware potential stopping (#23)" begin
    @testset "$xbc tau=$tau $initialization" for xbc in (:neumann, :periodic),
            tau in (0.8, 1.4, 2.0), initialization in (:cold, :consistent)
        result = capacitor(xbc, tau, initialization)
        @test result.accepted
        @test result.finite
        @test result.relative_change <= 1e-10
        # Exact capacitor phi*=1-y/H, E*=(0,1), ALL nodes including corners.
        # Budgets frozen in ES-002-STOP on 2026-09-15, not fitted to a repair.
        @test result.phi_error <= 1e-10
        @test result.prediction_error <= 1e-11
        if initialization === :cold
            # https://github.com/lauguimel/Kraken.jl/issues/23 (repaired: the
            # stopping rule also checks the field moment).
            @test result.field_error <= FIELD_GATE
        else
            @test result.field_error <= FIELD_GATE
            @test result.field_error <= 1e-10
        end
        @info "Capacitor stopping" xbc tau initialization result
    end

    @testset "negative control: pre-#23 rule, $xbc tau=$tau" for xbc in (:neumann, :periodic),
            tau in (0.8, 1.4, 2.0)
        # field_tol = Inf is the phi-only rule, run through the same production
        # solve: the cold starts must still fail the field gate, which shows
        # the promoted assertions above test the field check and nothing else.
        result = capacitor(xbc, tau, :cold; field_tol=Inf)
        @test result.accepted
        @test result.field_error > FIELD_GATE
        @info "Capacitor stopping, pre-#23 rule" xbc tau result
    end

    @testset "Public EC driver: first-cycle field convergence" begin
        # The driver cannot accept q=0/custom potential populations. This
        # additional small charged case exercises its ACTUAL adaptive loop,
        # without copying a stop predicate or changing production code.
        # C=0.01, tau_U=2 and one cycle isolate cold-start potential iteration;
        # q is identical until that first inner solve finishes. Returned E
        # still belongs to that solve, before the subsequent charge update.
        parameters = (; Nx=11, Ny=H+1, C=0.01, M=10.0, T=175.0,
                      Ma_E=0.01, alpha=1e-4, gamma=0.5, max_cycles=1,
                      perturb_amplitude=0.0, phi_scheme=:lbm,
                      backend=CPU_BACKEND, FT=Float64)
        adaptive = Kraken.run_electroconvection_2d(; parameters..., phi_tol=1e-4)
        fixed = Kraken.run_electroconvection_2d(; parameters..., phi_substeps=1024)
        reference = Kraken.run_electroconvection_2d(; parameters..., phi_substeps=2048)
        # Fixed sweeps are a test-only iterative reference, NOT the proposed
        # production repair. Doubling must leave E* unchanged to 1e-8, two
        # orders tighter than the 1e-6 stopping-accuracy gate (set 2026-09-16).
        @test all(isfinite, adaptive.Ex) && all(isfinite, adaptive.Ey)
        @test all(isfinite, reference.Ex) && all(isfinite, reference.Ey)
        @test adaptive.steps == fixed.steps == reference.steps == 1
        @test field_difference(fixed, reference) <= 1e-8
        # Still broken after the #23 repair, for another reason: the remaining
        # default-settings error (~3.5e-5) comes from the slow diffusive mode of
        # the pseudo-time Poisson iteration. phi_tol and field_tol bound an
        # increment between checks, not the error. Tracked in issue #63.
        # test/analytical/ehd_phi_ddf_solve_2d.jl runs this case with a tighter
        # field_tol and meets the gate.
        @test_broken field_difference(adaptive, reference) <= FIELD_GATE
        @info "Public driver field stopping" iterations=adaptive.phi_iters_last error=field_difference(adaptive, reference) reference_change=field_difference(fixed, reference)
        # Code-to-code iterative convergence is not physical validation.
    end
end

end # module
