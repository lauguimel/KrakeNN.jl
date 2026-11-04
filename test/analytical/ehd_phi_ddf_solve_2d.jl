# Issue #23: the adaptive DDF potential solve (`Kraken.ehd_phi_ddf_solve!`) stops on
# the field as well as on the potential. Complements ES-002-STOP.jl with
#   1. the #23 mechanism on the public EC driver, with a tight field_tol;
#   2. wiring guards: the drivers' defaults switch the field check on;
#   3. the function contract and the drivers' entry validation;
#   4. cold capacitors at production tolerances (CPU Float64 and Float32) against a
#      bound derived before any run;
#   5. an identically zero field (E_ref = 0): no division by zero, the rule accepts;
#   6. red/green: the Float32 gate of 4. rejects the pre-#23 rule's field.
# CPU only: GPU acceptance waits for the corner-race issue (separate ticket).
# Every gate below was written, with its rationale, before the file was first run.
# Run: julia --project test/analytical/ehd_phi_ddf_solve_2d.jl
module EHDPhiDDFSolveTests

using Test
using Kraken
using KernelAbstractions

const CPU_BACKEND = KernelAbstractions.CPU()
const H = 16
# The stopping-accuracy gate of issue #23 (ES-002-STOP.jl), on E* = H * E.
const FIELD_GATE = 1e-6

# Exact capacitor, no charge: phi* = 1 - y/H, E* = (0, 1). Cold start with the
# drivers' initialiser, f_i = w_i * phi: the first moment, hence E, is zero on entry.
function cold_capacitor(xbc, tau, FT)
    nx = xbc === :neumann ? 11 : 10
    ny = H + 1
    phi_profile = [one(FT) - FT(j - 1) / FT(H) for j in 1:ny]
    f = Kraken._fill_phi_populations!(zeros(FT, nx, ny, 9), phi_profile, nx, ny, FT)
    g = copy(f)
    phi = zeros(FT, nx, ny)
    Kraken.compute_ehd_scalar_2d!(phi, f)
    p = (; eps=FT(0.7), omega_U=inv(FT(tau)), nu_U=(FT(tau) - FT(0.5)) / FT(3),
         tau_U=FT(tau))
    return (; f, g, phi, q=zeros(FT, nx, ny), p, ws=Kraken.ehd_phi_ddf_workspace(phi))
end

function solve(c, xbc; kwargs...)
    FT = eltype(c.phi)
    return Kraken.ehd_phi_ddf_solve!(c.f, c.g, c.phi, c.q, c.p, xbc, c.ws; kwargs...,
                                     phi_bottom=one(FT), phi_top=zero(FT))
end

# Zero applied potential difference, no charge: both plates at 0, f = 0. The drivers
# fix the plates at 1 and 0, so this state is reachable only through the solve itself.
function zero_capacitor(xbc, tau, FT)
    c = cold_capacitor(xbc, tau, FT)
    fill!(c.f, zero(FT))
    fill!(c.g, zero(FT))
    fill!(c.phi, zero(FT))
    return c
end

function field(f, tau_U)
    ex, ey = similar(f, size(f, 1), size(f, 2)), similar(f, size(f, 1), size(f, 2))
    Kraken.compute_electric_field_2d!(ex, ey, f, tau_U)
    return ex, ey
end

# Error of E* = H * E against the exact (0, 1), all nodes, evaluated in Float64.
function capacitor_field_error(f, tau_U)
    ex, ey = field(f, tau_U)
    return max(maximum(abs, H .* Float64.(ex)), maximum(abs, H .* Float64.(ey) .- 1))
end

# A-priori bound on that error when the solve accepts at production tolerances.
# The cold capacitor field is a single mode: E* = (0, a_n) at every node, with
# a_n = 1 - r^n and r = 1 - 1/tau_U (ES-002-STOP.jl checks this recurrence to
# 1e-11). The error is e = |r|^n. A check on cadence m compares a_n with a_(n-m):
# |a_n - a_(n-m)| = |r|^(n-m) |1 - r^m|, and field_rel is that change divided by
# max(max|E*|, E_ref* = 1). With rho = |r|^m, acceptance (field_rel <= field_tol)
# gives e <= field_tol * rho / (1 - rho): max|E*| = a_n < 1 whenever r^m > 0, and
# when r^m < 0 (r < 0, m odd) the factor |1 - r^m| = 1 + rho outweighs a_n <= 1 + e.
# For m = 1 this bound is weak (1e-4 at tau_U = 2), which is why it is asserted
# instead of FIELD_GATE.
# Valid for the tested mode only: exact phi from the start, with a single relaxing
# first-moment mode contracting by r per iteration. It is not a bound for a general
# start, where the slow diffusive mode of the iteration (contraction about
# 1 - nu_U * pi^2 / H^2 per iteration) dominates and the error is about
# kappa * field_tol with kappa >> 1; that general case is issue #63.
stopping_bound(tau, m, field_tol) = (rho = abs(1 - 1 / tau)^m; field_tol * rho / (1 - rho))
# Roundoff floor: E* is H times a first moment of O(1) populations carrying unit
# roundoff eps(FT), so its noise is a few eps(FT) * H; 16 is the margin. For Float32
# at cadence 8 this floor dominates; at cadence 1 the stopping bound can exceed it
# (1e-4 against 3.1e-5 at tau_U = 2), so the two terms are added.
roundoff_floor(FT) = 16 * eps(FT) * H

check_every(xbc) = xbc === :neumann ? Kraken.EHD_EC_PHI_CHECK_EVERY :
                                      Kraken.EHD_HYDROSTATIC_PHI_CHECK_EVERY

field_difference(a, b) = H * max(maximum(abs.(a.Ex .- b.Ex)), maximum(abs.(a.Ey .- b.Ey)))

function thrown(f)
    try
        f()
    catch err
        return err
    end
    return nothing
end

# ES-002-STOP.jl's public-driver case: C = 0.01, tau_U = 2 (gamma = 0.5), one cycle.
const PUBLIC = (; Nx=11, Ny=H + 1, C=0.01, M=10.0, T=175.0, Ma_E=0.01, alpha=1e-4,
                gamma=0.5, max_cycles=1, perturb_amplitude=0.0, phi_scheme=:lbm,
                backend=CPU_BACKEND, FT=Float64)

@testset "EHD DDF potential solve: field-aware stopping (#23)" begin
    @testset "public EC driver, tight field_tol" begin
        # The #23 mechanism through run_electroconvection_2d. The slow diffusive
        # mode of the pseudo-time iteration turns a relative field change
        # field_rel into an error of about kappa * field_rel, kappa ~ H^2/(m gamma pi^2)
        # ~ 6.5 here (H = 16, m = 8, gamma = 0.5): field_tol = 1e-8 leaves ~1e-7,
        # under FIELD_GATE. The 2048-substep reference is itself converged to 1e-8
        # (ES-002-STOP.jl, 1024 against 2048 substeps).
        tight = run_electroconvection_2d(; PUBLIC..., phi_tol=1e-4, field_tol=1e-8)
        reference = run_electroconvection_2d(; PUBLIC..., phi_substeps=2048)
        @test tight.steps == reference.steps == 1
        @test all(isfinite, tight.Ex) && all(isfinite, tight.Ey)
        @test field_difference(tight, reference) <= FIELD_GATE
        @info "Public driver, field_tol=1e-8" iterations=tight.phi_iters_last error=field_difference(tight, reference)
    end

    @testset "wiring guards: default settings check the field" begin
        # Hydrostatic default run, one iteration per solve: the potential meets
        # phi_tol after one iteration from the analytic start, the field (zero on
        # entry) cannot, so the error must name the field.
        err = thrown(() -> Kraken.run_ehd_hydrostatic_2d(; phi_max_iter=1))
        @test err isa ErrorException
        @test err !== nothing && occursin("field", err.msg)
        # EC, public-driver case: the pre-#23 rule accepted at the first check
        # (iteration 8). The field has moved from zero there, so with the default
        # field_tol that check cannot accept and 8 iterations are not enough.
        err_ec = thrown(() -> run_electroconvection_2d(; PUBLIC..., phi_tol=1e-4,
                                                        phi_max_iter=8))
        @test err_ec isa ErrorException
        @test err_ec !== nothing && occursin("field", err_ec.msg)
        # Positive control: the same case under the pre-#23 rule completes.
        @test run_electroconvection_2d(; PUBLIC..., phi_tol=1e-4, phi_max_iter=8,
                                       field_tol=Inf).steps == 1
    end

    @testset "function contract (cold :neumann capacitor)" begin
        tau = 2.0
        m = Kraken.EHD_EC_PHI_CHECK_EVERY

        # Not converged: stats, no exception; the returned f_in carries ws.Ex/ws.Ey.
        c = cold_capacitor(:neumann, tau, Float64)
        f, g, st = solve(c, :neumann; phi_tol=1e-10, field_tol=1e-10, max_iter=m,
                         check_every=m)
        @test st.converged == false
        @test st.iters == m
        @test st.iters isa Int
        ex, ey = field(f, tau)
        @test ex == c.ws.Ex && ey == c.ws.Ey

        # Converged: same pair contract.
        c = cold_capacitor(:neumann, tau, Float64)
        f, g, st = solve(c, :neumann; phi_tol=1e-10, field_tol=1e-10, max_iter=128,
                         check_every=m)
        @test st.converged
        @test st.iters % m == 0
        ex, ey = field(f, tau)
        @test ex == c.ws.Ex && ey == c.ws.Ey
        phi_check = similar(c.phi)
        Kraken.compute_ehd_scalar_2d!(phi_check, f)
        @test phi_check == c.phi

        # field_tol = Inf: the phi-only rule accepts at the first check (the cold
        # start has the exact potential already).
        c = cold_capacitor(:neumann, tau, Float64)
        _, _, st = solve(c, :neumann; phi_tol=1e-10, field_tol=Inf, max_iter=128,
                         check_every=m)
        @test st.converged
        @test st.iters == Kraken.EHD_EC_PHI_CHECK_EVERY

        # Off-cadence final check: tau = 0.8 (r = -0.25), max_iter = 12, m = 8. At
        # iteration 8 the field has moved by ~1 from zero; at 12 it moves by
        # |r|^8 |1 - r^4| ~ 1.5e-5 < 1e-4, but over 4 iterations only, so that final
        # check cannot accept. The phi-only rule accepts at iteration 8.
        c = cold_capacitor(:neumann, 0.8, Float64)
        _, _, st = solve(c, :neumann; phi_tol=1e-4, field_tol=1e-4, max_iter=12,
                         check_every=m)
        @test st.converged == false
        @test st.iters == 12
        @test st.phi_rel <= 1e-4 && st.field_rel <= 1e-4
        c = cold_capacitor(:neumann, 0.8, Float64)
        _, _, st = solve(c, :neumann; phi_tol=1e-4, field_tol=Inf, max_iter=12,
                         check_every=m)
        @test st.converged && st.iters == m

        # Argument errors.
        c = cold_capacitor(:neumann, tau, Float64)
        for bad in ((; xbc=:bogus), (; field_tol=-1.0), (; field_tol=NaN), (; max_iter=0),
                    (; max_iter=8.0), (; check_every=0))
            args = merge((; xbc=:neumann, phi_tol=1e-10, field_tol=1e-10, max_iter=8,
                          check_every=m), bad)
            @test_throws ArgumentError solve(c, args.xbc; phi_tol=args.phi_tol,
                                             field_tol=args.field_tol,
                                             max_iter=args.max_iter,
                                             check_every=args.check_every)
        end
        @test_throws ArgumentError Kraken.ehd_phi_ddf_step!(c.f, c.g, c.phi, c.q, c.p, :bogus;
                                                            phi_bottom=1.0, phi_top=0.0)
    end

    @testset "driver entry validation" begin
        small = (; Nx=11, Ny=H + 1)
        for bad in ((; field_tol=-1e-4), (; field_tol=NaN), (; field_tol=-Inf), (; phi_max_iter=0),
                    (; phi_max_iter=2.5), (; phi_max_iter=-3))
            @test_throws ArgumentError init_state(ECState; small..., bad...)
            @test_throws ArgumentError run_electroconvection_2d(; small..., bad...)
            @test_throws ArgumentError Kraken.run_ehd_hydrostatic_2d(; small..., bad...)
        end
        # An integral phi_max_iter is stored as an Int; field_tol as given.
        s = init_state(ECState; small..., phi_max_iter=10000.0)
        @test s.config.phi_max_iter === 10000
        @test s.config.field_tol === 1e-4
        # phi_max_iter only governs the adaptive DDF solve.
        @test init_state(ECState; small..., phi_substeps=4, phi_max_iter=0) isa ECState
        @test init_state(ECState; small..., phi_scheme=:direct, phi_max_iter=0) isa ECState
        # field_tol = Inf is accepted: the pre-#23 rule.
        @test init_state(ECState; small..., field_tol=Inf).config.field_tol == Inf
        # Accepted range is [0, Inf], both ends included (same as the EC checkpoint).
        @test init_state(ECState; small..., field_tol=0.0).config.field_tol === 0.0
    end

    @testset "cold capacitor, production tolerances, $FT $xbc tau=$tau" for FT in (Float64, Float32),
            xbc in (:neumann, :periodic), tau in (0.8, 1.4, 2.0)
        m = check_every(xbc)
        field_tol = 1e-4  # the drivers' default
        c = cold_capacitor(xbc, tau, FT)
        f, _, st = solve(c, xbc; phi_tol=1e-4, field_tol=field_tol, max_iter=10000,
                         check_every=m)
        error_star = capacitor_field_error(f, tau)
        gate = stopping_bound(tau, m, field_tol) + roundoff_floor(FT)
        @test st.converged
        @test all(isfinite, f)
        @test error_star <= gate
        @info "Cold capacitor, production tolerances" FT xbc tau iterations=st.iters phi_rel=st.phi_rel field_rel=st.field_rel error_star gate
    end

    @testset "identically zero field (E_ref = 0), $FT $xbc" for FT in (Float64, Float32),
            xbc in (:neumann, :periodic)
        # Both plates at 0, q = 0, f = 0. One step maps this state to itself exactly:
        # the collision relaxes towards feq(0) = 0 with a zero source, streaming
        # permutes zeros, and the plate rebuild adds feq(0) = 0 to a zero
        # non-equilibrium part. So phi = 0 and E* = 0 at every iteration, and
        # E_ref = |0 - 0| / H = 0: both relative changes are 0 / 0 but for the floatmin
        # term of their denominators. With it they are exactly 0, both moments are
        # finite, and the rule accepts at the first check (iteration m), even with
        # zero tolerances. Were the ratios NaN, NaN <= tol would be false and the
        # solve would run to max_iter unconverged.
        m = check_every(xbc)
        tau = 2.0
        for field_tol in (0.0, 1e-4, Inf)
            c = zero_capacitor(xbc, tau, FT)
            f, _, st = Kraken.ehd_phi_ddf_solve!(c.f, c.g, c.phi, c.q, c.p, xbc, c.ws;
                                                 phi_tol=0.0, field_tol=field_tol,
                                                 max_iter=4m, check_every=m,
                                                 phi_bottom=zero(FT), phi_top=zero(FT))
            @test st.converged
            @test st.iters == m
            @test st.phi_rel === zero(FT) && st.field_rel === zero(FT)
            # Zero implies finite: no NaN or Inf in the populations or the moments.
            @test all(iszero, f) && all(iszero, c.phi)
            @test all(iszero, c.ws.Ex) && all(iszero, c.ws.Ey)
        end
        # The two reductions directly, on zero inputs with E_ref = 0; the NaN fill
        # shows that every slot is written.
        nx, ny = size(zero_capacitor(xbc, tau, FT).phi)
        z = zeros(FT, nx, ny)
        diag = fill(FT(NaN), 4)
        Kraken.ehd_rel_change_2d!(diag, z, z, nx, ny)
        Kraken.ehd_field_change_2d!(diag, z, z, z, z, 0.0, nx, ny, 2)
        @test diag == FT[0, 1, 0, 1]
    end

    @testset "red/green: Float32 gate against the pre-#23 rule, $xbc tau=2.0" for xbc in (:periodic, :neumann)
        # The gate is that of the production-tolerance testset: stopping_bound, the
        # contraction tail e <= field_tol * rho / (1 - rho) with rho = |1 - 1/tau_U|^m,
        # plus roundoff_floor, 16 eps(Float32) H. Float32, tau_U = 2, at each driver's
        # cadence (m = 1 hydrostatic, m = 8 EC): 1.3e-4 and 3.1e-5. Under field_tol = Inf
        # (pre-#23 rule) the phi-only rule accepts at the first check, iteration m,
        # since the cold start has the exact potential; the field error there is
        # |r|^m = 2^-m, 0.5 and 3.9e-3, above the gate. The default field_tol must
        # bring it under. Not every case of the production testset discriminates:
        # Float32 :neumann tau_U = 0.8 (|r|^8 = 0.25^8 = 1.5e-5): the old rule leaves
        # an error of 1.72e-5 (measured 2026-10-08, CPU), below the gate 3.05e-5, which
        # is the roundoff floor there (the stopping-bound term is 1.5e-9). The roundoff
        # floor dominates, so the gate cannot separate the rules in that case, and the
        # production testset above passes it under either rule. This is a limit of the
        # Float32 test at that tau_U, not of the stopping rule.
        FT = Float32
        tau = 2.0
        m = check_every(xbc)
        gate = stopping_bound(tau, m, 1e-4) + roundoff_floor(FT)
        c_old = cold_capacitor(xbc, tau, FT)
        f_old, _, st_old = solve(c_old, xbc; phi_tol=1e-4, field_tol=Inf, max_iter=10000,
                                 check_every=m)
        c_new = cold_capacitor(xbc, tau, FT)
        f_new, _, st_new = solve(c_new, xbc; phi_tol=1e-4, field_tol=1e-4, max_iter=10000,
                                 check_every=m)
        err_old = capacitor_field_error(f_old, tau)
        err_new = capacitor_field_error(f_new, tau)
        @test st_old.converged && st_old.iters == m
        @test st_new.converged
        @test err_old > gate
        @test err_new <= gate
        @info "Red/green, Float32 tau=2" xbc iters_old=st_old.iters iters_new=st_new.iters err_old err_new gate
    end
end

end # module
