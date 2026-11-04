using Test, Kraken, KernelAbstractions

# #73: CPU invariants and exact same-backend restart. Full ordered-corner,
# analytic ladders and CUDA controls are attached separately with the PR SHA.
@testset "2D moment injector and checkpoint identity" begin
    qb=0.4; vx=.002; vy=.006
    f=reshape([.02+.001*sin(k) for k=1:9],1,1,9)
    g=f[1,1,5]+f[1,1,8]+f[1,1,9]; d=f[1,1,9]-f[1,1,8]
    Kraken._ehd_moment_wall!(f,1,1,qb,vx,vy,false)
    @test sum(f) ≈ qb atol=2e-15
    @test f[1,1,5]+f[1,1,8]+f[1,1,9] ≈ g atol=2e-15
    @test f[1,1,9]-f[1,1,8] ≈ d atol=2e-15
    cfg=(Nx=8,Ny=12,C=10.,M=10.,T=190.,phi_scheme=:direct,
        charge_scheme=:regularized,ns_scheme=:mrt,history_interval=1)
    replace_snapshot(s; kw...)=Kraken.StateSnapshot(;
        merge((; (k=>getfield(s,k) for k in fieldnames(typeof(s)))...), (;kw...))...)
    electrodes = Dict{Symbol,Array{Float64,3}}()
    for closure in (:nee,:moment)
        s=Kraken.init_state(Kraken.ECState;cfg...,injector_closure=closure)
        Kraken.advance!(s,4)
        # Keep injector interior nodes: exclude sidewall/corner scheduling effects.
        electrodes[closure] = copy(s.q_f_in[2:end-1,1:1,:])
        snap=Kraken.export_state(s)
        @test snap.schema_version==2
        @test snap.identity["injector_closure"]==String(closure)
        r=Kraken.restore_state(Kraken.ECState,snap)
        Kraken.advance!(s,4); Kraken.advance!(r,4)
        for k in (:q_f_in,:f_in,:phi,:Fx_prev,:Fy_prev,:qfield,:umax_history)
            @test isequal(getproperty(s,k),getproperty(r,k))
        end
        other=closure===:nee ? :moment : :nee
        @test_throws Kraken.CheckpointError Kraken.restore_state(Kraken.ECState,snap;injector_closure=other)
        for invalid in ("static_moment","unknown")
            bad=deepcopy(snap); bad.identity["injector_closure"]=invalid
            @test_throws Kraken.CheckpointError Kraken.restore_state(Kraken.ECState,bad)
        end
        bad=deepcopy(snap); delete!(bad.identity,"injector_closure")
        @test_throws Kraken.CheckpointError Kraken.restore_state(Kraken.ECState,bad)
        if closure===:nee
            old=replace_snapshot(bad;schema_version=1)
            rr=Kraken.restore_state(Kraken.ECState,old)
            @test rr.config.injector_closure===:nee
            @test !haskey(old.identity,"injector_closure")
            @test isequal(rr.q_f_in,snap.fields["q_f_in"])
            Kraken.advance!(rr,4)
            @test isequal(rr.q_f_in,s.q_f_in)
            mktempdir() do dir
                # Write an actual schema-1 container, without the new identity key.
                path = joinpath(dir, "schema1.h5")
                Kraken.write_checkpoint(path, old)
                disk = Kraken.read_checkpoint(path)
                @test disk.schema_version == 1
                @test !haskey(disk.identity, "injector_closure")
                loaded = Kraken.load_checkpoint(Kraken.ECState, path)
                @test loaded.config.injector_closure === :nee
                @test Kraken.export_state(loaded).schema_version == 2
                for key in keys(snap.fields)
                    @test isequal(Kraken.export_state(loaded).fields[key], snap.fields[key])
                end
                Kraken.advance!(loaded,4)
                @test isequal(loaded.q_f_in,s.q_f_in)
                @test isequal(loaded.f_in,s.f_in)
                @test isequal(loaded.umax_history,s.umax_history)
            end
            delete!(old.identity,"sidewall_bc")
            @test_throws Kraken.CheckpointError Kraken.restore_state(Kraken.ECState,old)
        end
    end
    # Dispatch regression, not an accuracy gate: silently routing :moment to
    # production NEE makes these arrays identical. A scale-relative roundoff
    # margin rejects incidental last-bit differences, without fitting to a result.
    delta = maximum(abs.(electrodes[:moment] .- electrodes[:nee]))
    scale = max(maximum(abs, electrodes[:moment]), maximum(abs, electrodes[:nee]))
    @test all(isfinite, electrodes[:moment]) && all(isfinite, electrodes[:nee])
    @test delta > 100eps(Float64)*scale
    @info "Transient injector dispatch regression" electrode_difference=delta roundoff_margin=100eps(Float64)*scale
    for overrides in ((phi_scheme=:lbm,), (charge_scheme=:srt,), (FT=Float32,), (delta_U=2.,))
        @test_throws ArgumentError Kraken.init_state(Kraken.ECState;
            merge(cfg,(injector_closure=:moment,),overrides)...)
    end
end
