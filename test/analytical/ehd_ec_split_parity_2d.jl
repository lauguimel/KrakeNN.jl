using Test
using Kraken

# Issue #26: `run_electroconvection_2d` is now init_state + advance! + solution on
# the platform state contract. This file proves the split changed no number:
#   1. the wrapper against a frozen verbatim copy of the former monolithic function,
#   2. any split of a run into segments against the continuous run,
#   3. negative controls showing that the exact comparisons of (2) can fail.
# Every comparison is exact (`isequal`), never a tolerance: after 20 cycles the
# Coulomb force is ~1e-10, far below anything a tolerance would notice.
# The frozen copy predates the repair of issue #23: its adaptive potential solve
# stops on the potential alone. In (1) the new code therefore runs with
# `field_tol = Inf`, which reproduces that rule bit for bit; the default `field_tol`
# also checks the field and takes more potential iterations. Parts (2) and (3)
# compare the new code with itself and keep the defaults, and so does the error-path
# parity: its `phi_max_iter = 1` case fails on the potential under both rules, with
# the same message.

if !isdefined(Kraken, :_legacy_run_electroconvection_2d)
    Base.include(Kraken, joinpath(@__DIR__, "..", "reference", "ehd_ec_legacy.jl"))
end

const EC_SPLIT_TIMING_KEYS = (:loop_ms_per_step,)

# Exact comparison of two result NamedTuples: same keys in the same order, and for
# every key but the wall-clock one, same type and same value (arrays element-wise).
function ec_results_identical(a::NamedTuple, b::NamedTuple)
    keys(a) == keys(b) || return false
    for k in keys(a)
        k in EC_SPLIT_TIMING_KEYS && continue
        typeof(a[k]) == typeof(b[k]) || return false
        isequal(a[k], b[k]) || return false
    end
    return true
end

const EC_DYNAMIC_FIELDS = (:phi_f_in, :q_f_in, :f_in, :Fx_prev, :Fy_prev, :phi, :qfield,
                           :umax_history, :cycle_history,
                           :phi_iters_last, :phi_rel_last, :q_rel_last, :cycle)

# Names of the dynamic fields of two states that differ (empty = identical).
function ec_state_differences(a::ECState, b::ECState)
    return [name for name in EC_DYNAMIC_FIELDS
            if !isequal(Array_or_value(getfield(a, name)), Array_or_value(getfield(b, name)))]
end
Array_or_value(x::AbstractArray) = Array(x)
Array_or_value(x) = x

function ec_thrown(f)
    try
        f()
    catch err
        return err
    end
    return nothing
end

@testset "EHD electroconvection: state-contract split parity" begin
    grid = (Nx=16, Ny=24)
    phi_paths = (lbm_iterative=(phi_scheme=:lbm,),
                 lbm_substep=(phi_scheme=:lbm, phi_substeps=1),
                 direct=(phi_scheme=:direct,))

    @testset "wrapper == frozen legacy copy" begin
        cases = Pair{String,NamedTuple}[]
        for (path_name, path) in pairs(phi_paths), ns_scheme in (:bgk, :mrt)
            push!(cases, "$path_name $ns_scheme" =>
                  (; grid..., path..., ns_scheme, max_cycles=30, history_interval=5))
        end
        push!(cases, "substep srt" =>
              (; grid..., phi_substeps=1, charge_scheme=:srt, max_cycles=30))
        push!(cases, "iterative srt mrt projection xy" =>
              (; grid..., charge_scheme=:srt, ns_scheme=:mrt, force_projection=:xy, max_cycles=24))
        push!(cases, "substep projection y" =>
              (; grid..., phi_substeps=1, force_projection=:y, max_cycles=30, history_interval=4))
        push!(cases, "direct projection y" =>
              (; grid..., phi_scheme=:direct, force_projection=:y, max_cycles=30))
        push!(cases, "substep target_t_star" =>
              (; grid..., phi_substeps=1, max_cycles=500, target_t_star=2e-3, history_interval=6))
        push!(cases, "direct target_t_star above max_cycles" =>
              (; grid..., phi_scheme=:direct, max_cycles=21, target_t_star=10.0, history_interval=8))
        push!(cases, "iterative interval not dividing" =>
              (; grid..., max_cycles=23, history_interval=7))
        push!(cases, "substep 3 interval not dividing" =>
              (; grid..., phi_substeps=3, max_cycles=31, history_interval=7, perturb_mode=2,
                 perturb_amplitude=1e-2, T=400.0))
        push!(cases, "zero cycles" => (; grid..., phi_substeps=1, max_cycles=0))
        push!(cases, "other grid and physics" =>
              (; Nx=12, Ny=20, C=5.0, M=8.0, T=300.0, phi_substeps=1, max_cycles=40,
                 history_interval=9))

        for FT in (Float64, Float32), (name, kwargs) in cases
            legacy = Kraken._legacy_run_electroconvection_2d(; kwargs..., FT=FT)
            new = run_electroconvection_2d(; kwargs..., field_tol=Inf, FT=FT)
            @testset "$name $FT" begin
                @test ec_results_identical(legacy, new)
                @test eltype(new.ux) == FT
            end
        end

        # The horizon cases did what their name says.
        r = run_electroconvection_2d(; grid..., phi_substeps=1, max_cycles=500,
                                     target_t_star=2e-3, history_interval=6)
        @test 0 < r.steps < 500
        @test r.cycle_history[end] == r.steps
        @test r.steps % 6 != 0
        r = run_electroconvection_2d(; grid..., max_cycles=23, history_interval=7)
        @test r.cycle_history == [7, 14, 21, 23]
    end

    @testset "segment independence ($path_name, $FT)" for (path_name, path) in pairs(phi_paths),
                                                          FT in (Float64, Float32)
        kwargs = (; grid..., path..., history_interval=5, FT=FT)
        continuous = init_state(ECState; kwargs...)
        @test Kraken.at_boundary(continuous)
        advance!(continuous, 20; sample_final=true)
        @test Kraken.at_boundary(continuous)
        @test continuous.cycle == 20
        @test continuous.cycle_history == [5, 10, 15, 20]
        reference = solution(continuous)
        @test reference isa ECSolution
        @test reference isa Kraken.AbstractSolution

        for (a, b) in ((7, 13), (8, 12))
            s = init_state(ECState; kwargs...)
            advance!(s, a)
            advance!(s, 0)                       # no-op
            @test s.cycle == a
            advance!(s, b; sample_final=true)
            @test isempty(ec_state_differences(s, continuous))
            @test s.cycle_history == [5, 10, 15, 20]
            @test ec_results_identical(solution(s).result, reference.result)
        end

        # `solution` in the middle of a run (twice) must not disturb what follows.
        s = init_state(ECState; kwargs...)
        solution(s)
        advance!(s, 7)
        mid = solution(s).result
        @test ec_results_identical(solution(s).result, mid)
        @test mid.steps == 7
        advance!(s, 13; sample_final=true)
        @test isempty(ec_state_differences(s, continuous))
        @test ec_results_identical(solution(s).result, reference.result)

        # The wrapper is this very sequence.
        @test ec_results_identical(run_electroconvection_2d(; kwargs..., max_cycles=20),
                                   reference.result)
    end

    @testset "negative controls ($path_name, $FT)" for (path_name, path) in pairs(phi_paths),
                                                       FT in (Float64, Float32)
        kwargs = (; grid..., path..., history_interval=5, FT=FT)
        continuous = advance!(init_state(ECState; kwargs...), 20; sample_final=true)

        # Force history dropped at the segment boundary.
        s = advance!(init_state(ECState; kwargs...), 7)
        fill!(s.Fy_prev, zero(FT))
        advance!(s, 13; sample_final=true)
        @test !isempty(ec_state_differences(s, continuous))
        @test !ec_results_identical(solution(s).result, solution(continuous).result)

        # Issue #26: distinguish comparator sensitivity from propagation.
        # Upstream CI 35523915211 loses the nine one-ulp changes after 13
        # cycles on direct/Float64. Finite-precision evolution is not injective:
        # exact comparison must catch a stored change, but cannot guarantee that
        # a one-ulp change survives subsequent moment sums/collision/rounding.
        # Keep that injection and log its cycle-level fate without prescribing
        # a hardware-dependent disappearance cycle.
        pristine = advance!(init_state(ECState; kwargs...), 7)
        s = advance!(init_state(ECState; kwargs...), 7)
        @test isempty(ec_state_differences(s, pristine))
        q_pop = Array(s.q_f_in)
        for qdir in 1:9
            q_pop[8, 12, qdir] = nextfloat(q_pop[8, 12, qdir])
        end
        copyto!(s.q_f_in, q_pop)
        @test !isequal(Array(s.q_f_in), Array(pristine.q_f_in))
        @test :q_f_in in ec_state_differences(s, pristine)
        charge_difference_cycles = Int[]
        state_difference_cycles = Int[]
        for step in 1:13
            advance!(pristine, 1; sample_final=(step == 13))
            advance!(s, 1; sample_final=(step == 13))
            differences = ec_state_differences(s, pristine)
            :q_f_in in differences && push!(charge_difference_cycles, s.cycle)
            !isempty(differences) && push!(state_difference_cycles, s.cycle)
        end
        @test isempty(ec_state_differences(pristine, continuous))
        @info "EC one-ulp propagation diagnostic" path_name FT charge_difference_cycles state_difference_cycles

        # The continuation negative control must change a retained moment by a
        # finite amount, not only the last bit. Scale all populations of one
        # interior node by 65/64: a positive 1.5625% local charge corruption.
        # This binary-exact factor is fixed before testing, not fitted to a
        # measured error or used as a numerical acceptance tolerance. Normal
        # wrapper/segment comparisons elsewhere remain strictly isequal.
        pristine = advance!(init_state(ECState; kwargs...), 7)
        s = advance!(init_state(ECState; kwargs...), 7)
        q_pop = Array(s.q_f_in)
        charge_before = sum(@view q_pop[8, 12, :])
        q_pop[8, 12, :] .*= FT(65) / FT(64)
        charge_after = sum(@view q_pop[8, 12, :])
        @test isfinite(charge_before) && charge_before > zero(FT)
        @test isfinite(charge_after) && charge_after > charge_before
        copyto!(s.q_f_in, q_pop)
        @test :q_f_in in ec_state_differences(s, pristine)
        advance!(s, 13; sample_final=true)
        @test all(isfinite, s.q_f_in) && all(isfinite, s.f_in)
        @test :q_f_in in ec_state_differences(s, continuous)
        @test !ec_results_identical(solution(s).result, solution(continuous).result)
    end

    @testset "at_boundary after an interrupted advance!" begin
        s = init_state(ECState; grid..., phi_substeps=1, velocity_stop=1e-30)
        err = ec_thrown(() -> advance!(s, 5; sample_final=true))
        @test err isa ErrorException
        @test occursin("Flow field became unstable at cycle 1", err.msg)
        @test !Kraken.at_boundary(s)
        @test s.cycle == 0

        s = init_state(ECState; grid..., phi_max_iter=1)
        @test_throws ErrorException advance!(s, 5)
        @test !Kraken.at_boundary(s)

        # Cycles completed before the failure stay counted; the broken one is not.
        s = init_state(ECState; grid..., phi_substeps=1, velocity_stop=1e-30, history_interval=4)
        @test_throws ErrorException advance!(s, 10)
        @test s.cycle == 3
        @test !Kraken.at_boundary(s)

        @test_throws ArgumentError advance!(init_state(ECState; grid...), -1)
    end

    @testset "advance! and solution refuse a state left mid-cycle" begin
        # Review of #34: the velocity stop fires on a sampled cycle after the
        # populations were advanced but before Fx_prev/Fy_prev and the counter
        # moved. A second advance! must not run on that half-advanced state.
        s = init_state(ECState; grid..., phi_substeps=1, history_interval=5, velocity_stop=1e-30)
        @test_throws ErrorException advance!(s, 1; sample_final=true)
        @test !Kraken.at_boundary(s)
        @test s.cycle == 0
        frozen = (f_in=Array(s.f_in), q_f_in=Array(s.q_f_in), phi_f_in=Array(s.phi_f_in),
                  Fx_prev=Array(s.Fx_prev), Fy_prev=Array(s.Fy_prev),
                  umax_history=copy(s.umax_history), cycle_history=copy(s.cycle_history))
        for call in (() -> advance!(s, 1), () -> advance!(s, 0),
                     () -> advance!(s, 4; sample_final=true), () -> solution(s))
            err = ec_thrown(call)
            @test err isa ArgumentError
            @test occursin("not on a completed cycle boundary", err.msg)
            @test !Kraken.at_boundary(s)
            @test s.cycle == 0
        end
        @test isequal(Array(s.f_in), frozen.f_in)
        @test isequal(Array(s.q_f_in), frozen.q_f_in)
        @test isequal(Array(s.phi_f_in), frozen.phi_f_in)
        @test isequal(Array(s.Fx_prev), frozen.Fx_prev)
        @test isequal(Array(s.Fy_prev), frozen.Fy_prev)
        @test s.umax_history == frozen.umax_history
        @test s.cycle_history == frozen.cycle_history
        # The refusal names the verb; a fresh state is the recovery path.
        @test startswith(ec_thrown(() -> advance!(s, 1)).msg, "advance!:")
        @test startswith(ec_thrown(() -> solution(s)).msg, "solution:")
        fresh = init_state(ECState; grid..., phi_substeps=1, history_interval=5)
        @test advance!(fresh, 1) === fresh && Kraken.at_boundary(fresh)
        @test solution(fresh) isa ECSolution
    end

    @testset "error-path parity" begin
        failing = ((; grid..., phi_max_iter=1, max_cycles=5),
                   (; grid..., phi_substeps=1, velocity_stop=1e-30, max_cycles=9, history_interval=4),
                   (; grid..., Nx=3),
                   (; grid..., Ny=7),
                   (; grid..., charge_scheme=:bogus),
                   (; grid..., ns_scheme=:bogus),
                   (; grid..., force_projection=:bogus),
                   (; grid..., phi_scheme=:bogus),
                   (; grid..., history_interval=0),
                   (; grid..., max_cycles=2.5),
                   # several faults at once: the first check must win in both
                   (; grid..., Nx=3, ns_scheme=:bogus, max_cycles=2.5))
        for kwargs in failing
            legacy_err = ec_thrown(() -> Kraken._legacy_run_electroconvection_2d(; kwargs...))
            new_err = ec_thrown(() -> run_electroconvection_2d(; kwargs...))
            @test legacy_err !== nothing
            @test typeof(new_err) == typeof(legacy_err)
            @test sprint(showerror, new_err) == sprint(showerror, legacy_err)
        end
    end
end
