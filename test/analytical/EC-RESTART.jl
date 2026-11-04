using Test, Kraken
using KernelAbstractions

if !isdefined(@__MODULE__, :run_state_contract_suite)
    include(joinpath(@__DIR__, "..", "platform", "state_contract_suite.jl"))
end

# Prospective gates: same-backend CPU restarts and zero-step snapshots are exact,
# not tolerance-based. These short interface tests do not qualify paper Fig. 4.
# `adaptive=true` selects the adaptive DDF potential solve (`phi_scheme = :lbm`,
# `phi_substeps = nothing`), the only path on which `field_tol` acts (#23).
function ec_restart_fixture(FT; phi_scheme=:direct, ns_scheme=:bgk,
                            sidewall_bc=:free_slip, backend=CPU(),
                            field_tol=1e-4, adaptive=false)
    return (; Nx=10, Ny=16, C=10.0, M=10.0, T=190.0, Ma_E=0.01,
        alpha=1e-4, delta_U=1.0, gamma=0.3, phi_tol=1e-4, field_tol,
        phi_max_iter=10000,
        phi_substeps=phi_scheme === :lbm && !adaptive ? 2 : nothing,
        phi_scheme, charge_scheme=:regularized, ns_scheme, sidewall_bc,
        perturb_amplitude=1e-4, perturb_mode=1, force_projection=:none,
        velocity_stop=0.2, history_interval=3, backend, FT)
end
function ec_restart_kwargs(cfg)
    return (; (k => v for (k, v) in pairs(cfg)
               if !(k in (:T, :perturb_amplitude, :perturb_mode)))...)
end
function ec_snapshot_updated(mid, name, value; backend=CPU())
    s = restore_state(ECState, mid; backend)
    update_parameter!(s, name, value)
    return export_state(s)
end
function ec_result_equal(a, b)
    # Timing is observational, not deterministic state.
    return all(k -> isequal(getproperty(a, k), getproperty(b, k)),
               filter(!=(:loop_ms_per_step), keys(a)))
end

@testset "EC-RESTART CPU contract" begin
    for FT in (Float64, Float32), scheme in (:direct, :lbm)
        cfg = ec_restart_fixture(FT; phi_scheme=scheme)
        @testset "$FT $scheme" begin
            mktempdir() do dir
                # The generic corner perturbation can be removed by EC boundary
                # reconstruction; direct phi is output time-level state, not next-
                # cycle input. Use explicit per-field zero-step controls below,
                # plus an interior all-nine-population control after advancement.
                run_state_contract_suite(ECState; make_state=() -> init_state(ECState; cfg...),
                    restore_kwargs=ec_restart_kwargs(cfg),
                    interrupt! = s -> (s.at_boundary = false), tmpdir=dir,
                    negative_controls=false, parameter_snapshot=ec_snapshot_updated)
                s = advance!(init_state(ECState; cfg...), 8)
                mid = export_state(s)
                path = joinpath(dir, "even.h5")
                save_checkpoint(path, s)
                r = load_checkpoint(ECState, path; ec_restart_kwargs(cfg)...)
                @test ec_result_equal(solution(s).result, solution(r).result)
                advance!(r, 12)
                ref = advance!(init_state(ECState; cfg...), 20)
                @test isempty(snapshot_differences(export_state(r), export_state(ref)))
                for field in keys(mid.fields)
                    bad = tampered(mid)
                    # A zeroed field must change the restored image even when a
                    # later collision or boundary would hide the missing state.
                    fill!(bad.fields[field], 0)
                    if isequal(bad.fields[field], mid.fields[field])
                        bad.fields[field][4, 5] = one(FT)
                    end
                    @test "fields/$field" in snapshot_differences(
                        export_state(restore_state(ECState, bad)), mid)
                    wrong = tampered(mid)
                    delete!(wrong.fields, field)
                    @test_throws CheckpointError restore_state(ECState, wrong)
                end
                bad = tampered(mid)
                bad.fields["q_f_in"][4, 5, :] .*= FT(1.01)
                perturbed = advance!(restore_state(ECState, bad), 1)
                reference = advance!(restore_state(ECState, mid), 1)
                @test !isempty(snapshot_differences(export_state(perturbed), export_state(reference)))
                bad = tampered(mid); bad.fields["f_in"] = zeros(FT, 2, 2, 9)
                @test_throws CheckpointError restore_state(ECState, bad)
                bad = tampered(mid); bad.fields["f_in"][1] = FT(NaN)
                @test_throws CheckpointError restore_state(ECState, bad)
                bad = tampered(mid); bad.derived["nu"] *= FT(2)
                @test_throws CheckpointError restore_state(ECState, bad)
                bad = tampered(mid); bad.parameters["T"] = 180.0
                @test_throws CheckpointError restore_state(ECState, bad)
                @test_throws CheckpointError restore_state(ECState, mid; FT=FT === Float64 ? Float32 : Float64)
                initial = init_state(ECState; cfg...)
                @test ec_result_equal(solution(initial).result,
                    solution(restore_state(ECState, export_state(initial))).result)
            end
        end
    end
end

@testset "EC-RESTART sidewall identity" begin
    # PR60 review: mandatory schema-1 wall identity, with no default/migration.
    for FT in (Float64, Float32), scheme in (:direct, :lbm), wall in (:free_slip, :no_slip)
        @testset "$FT $scheme $wall" begin
            cfg = ec_restart_fixture(FT; phi_scheme=scheme, sidewall_bc=wall)
            s = advance!(init_state(ECState; cfg...), 7)
            mid = export_state(s)
            @test mid.schema_version == 2
            @test mid.identity["sidewall_bc"] == String(wall)
            other = wall === :free_slip ? :no_slip : :free_slip
            mktempdir() do dir
                path = joinpath(dir, "wall.h5")
                save_checkpoint(path, s)
                r = load_checkpoint(ECState, path; sidewall_bc=wall)
                @test r.config.sidewall_bc === wall
                @test isempty(snapshot_differences(export_state(r), mid))
                @test_throws CheckpointError load_checkpoint(ECState, path; sidewall_bc=other)
                missing = tampered(mid)
                delete!(missing.identity, "sidewall_bc")
                @test_throws CheckpointError restore_state(ECState, missing)
                @test_throws CheckpointError restore_state(ECState, missing; sidewall_bc=wall)
                missing_path = joinpath(dir, "missing_wall.h5")
                write_checkpoint(missing_path, missing)
                @test_throws CheckpointError load_checkpoint(ECState, missing_path)
                invalid = tampered(mid)
                invalid.identity["sidewall_bc"] = "periodic"
                @test_throws CheckpointError restore_state(ECState, invalid)
                for state in (s, r)
                    update_parameter!(state, :T, 180.0)
                    @test state.config.sidewall_bc === wall
                    @test export_state(state).identity["sidewall_bc"] == String(wall)
                    @test state.parameter_cycles == [0, 7]
                    @test state.parameter_values == [190.0, 180.0]
                    advance!(state, 13)
                    @test state.config.sidewall_bc === wall
                end
                @test isempty(snapshot_differences(export_state(s), export_state(r)))
                save_checkpoint(path, r)
                @test load_checkpoint(ECState, path).config.sidewall_bc === wall
            end
        end
    end
end

@testset "EC-RESTART T continuation" begin
    for FT in (Float64, Float32), ns in (:bgk, :mrt)
        cfg = ec_restart_fixture(FT; ns_scheme=ns)
        s = advance!(init_state(ECState; cfg...), 7)
        before = export_state(s)
        old = s.p
        mktempdir() do dir
            path = joinpath(dir, "parameter.h5")
            save_checkpoint(path, s)
            r = load_checkpoint(ECState, path)
            for state in (s, r)
                @test update_parameter!(state, :T, 180.0) === state
                @test isempty(snapshot_differences(before, export_state(state);
                    classes=(:fields, :scalars, :identity)))
                @test state.p.dt_star === old.dt_star
                @test state.p.nu ≈ old.nu * FT(190 / 180) rtol=4eps(FT)
                @test state.parameter_cycles == [0, 7]
                @test state.parameter_values == [190.0, 180.0]
                frozen = export_state(state)
                for value in (0.0, -1.0, Inf, NaN, floatmax(Float64))
                    @test_throws ArgumentError update_parameter!(state, :T, value)
                    @test isempty(snapshot_differences(export_state(state), frozen))
                end
                @test_throws ArgumentError update_parameter!(state, :C, 1.0)
                advance!(state, 13)
            end
            @test isempty(snapshot_differences(export_state(s), export_state(r)))
            save_checkpoint(path, s)
            @test isempty(snapshot_differences(export_state(load_checkpoint(ECState, path)), export_state(s)))
        end
    end
end

@testset "EC-RESTART field_tol identity" begin
    # #23 x #60: field_tol decides where each adaptive potential solve stops, hence
    # the trajectory. It is mandatory schema-1 identity, with no default/migration.
    for FT in (Float64, Float32), wall in (:free_slip, :no_slip)
        @testset "$FT $wall" begin
            cfg = ec_restart_fixture(FT; phi_scheme=:lbm, adaptive=true, sidewall_bc=wall)
            cfg_inf = merge(cfg, (; field_tol=Inf))
            @test cfg.phi_substeps === nothing && cfg.field_tol === 1e-4
            s = advance!(init_state(ECState; cfg...), 8)
            s_inf = advance!(init_state(ECState; cfg_inf...), 8)
            mid, mid_inf = export_state(s), export_state(s_inf)
            @test mid.schema_version == 2
            @test mid.identity["field_tol"] === 1e-4
            @test mid_inf.identity["field_tol"] === Inf
            # The fixture exercises the field check: over 8 cycles, the 1e-4 and Inf
            # rules stop the potential solves at different iterations.
            @test !isempty(snapshot_differences(mid, mid_inf; classes=(:fields,)))
            mktempdir() do dir
                path, path_inf = joinpath(dir, "tol.h5"), joinpath(dir, "inf.h5")
                save_checkpoint(path, s)
                save_checkpoint(path_inf, s_inf)
                # (a) same field_tol accepted, image unchanged.
                r = load_checkpoint(ECState, path; field_tol=1e-4)
                @test r.config.field_tol === 1e-4
                @test isempty(snapshot_differences(export_state(r), mid))
                # (e) Inf round-trips through HDF5 exactly, with and without the check.
                r_inf = load_checkpoint(ECState, path_inf)
                @test r_inf.config.field_tol === Inf
                @test export_state(r_inf).identity["field_tol"] === Inf
                @test load_checkpoint(ECState, path_inf; field_tol=Inf).config.field_tol === Inf
                @test isempty(snapshot_differences(export_state(r_inf), mid_inf))
                # (b) mismatch refused both ways, from memory and from disk.
                @test_throws CheckpointError load_checkpoint(ECState, path; field_tol=Inf)
                @test_throws CheckpointError load_checkpoint(ECState, path_inf; field_tol=1e-4)
                @test_throws CheckpointError restore_state(ECState, mid; field_tol=Inf)
                @test_throws CheckpointError restore_state(ECState, mid_inf; field_tol=1e-4)
                @test_throws CheckpointError restore_state(ECState, mid; field_tol=1e-6)
                # (c) a snapshot lacking field_tol is refused, never defaulted.
                missing = tampered(mid)
                delete!(missing.identity, "field_tol")
                @test_throws CheckpointError restore_state(ECState, missing)
                @test_throws CheckpointError restore_state(ECState, missing; field_tol=1e-4)
                missing_path = joinpath(dir, "missing_tol.h5")
                write_checkpoint(missing_path, missing)
                @test_throws CheckpointError load_checkpoint(ECState, missing_path)
                # Stored range is [0, Inf], as at driver entry.
                for value in (NaN, -1e-4, -Inf, "1e-4")
                    bad = tampered(mid)
                    bad.identity["field_tol"] = value
                    @test_throws CheckpointError restore_state(ECState, bad)
                end
                zero_tol = tampered(mid)
                zero_tol.identity["field_tol"] = 0.0
                @test validate_snapshot(ECState, zero_tol) === nothing
                # (d) adaptive-path restart is bit-identical to an uninterrupted run,
                # for both rules.
                advance!(r, 12)
                advance!(r_inf, 12)
                ref = advance!(init_state(ECState; cfg...), 20)
                ref_inf = advance!(init_state(ECState; cfg_inf...), 20)
                @test isempty(snapshot_differences(export_state(r), export_state(ref)))
                @test isempty(snapshot_differences(export_state(r_inf), export_state(ref_inf)))
                @test ec_result_equal(solution(r).result, solution(ref).result)
                @test ec_result_equal(solution(r_inf).result, solution(ref_inf).result)
                @test r.config.field_tol === 1e-4 && r_inf.config.field_tol === Inf
            end
        end
    end
end
