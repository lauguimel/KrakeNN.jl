include("common.jl")
sealed = split(strip(read("raw/tolerances.sha256",String)))[1]
@assert bytes2hex(sha256(read("raw/tolerances.toml"))) == sealed
tol = TOML.parsefile("raw/tolerances.toml")
@assert tol["source"] == strip(read("SOURCE_SHA",String))
checks = Any[]; comparisons = Any[]; stages = Any[]
function check(name, passed; details="")
    push!(checks,Dict("name"=>name,"pass"=>Bool(passed),"details"=>details))
    save_toml("raw/checks.toml",Dict("checks"=>checks))
    passed || println("FAILED ",name," ",details)
    return passed
end
function disk_exact(s,r,tag)
    a,b = Kraken.export_state(s),Kraken.export_state(r)
    for k in keys(a.fields)
        check("$tag/disk/$k",isequal(a.fields[k],b.fields[k]))
    end
    for k in (:cycle,:identity,:series,:scalars,:parameters,:derived,:run_control)
        check("$tag/disk/$k",isequal(getproperty(a,k),getproperty(b,k)))
    end
    check("$tag/closure",s.config.injector_closure===r.config.injector_closure)
end
function compare(a,b,closure,tag,cycles; exact=false)
    for q in keys(a[1])
        errs = [distance(a[i][q],b[i][q]) for i in eachindex(a)]
        err = maximum(errs)
        gate = exact ? 0.0 : tol[string(closure)][q]["tolerance_abs"]
        scale = maximum(maxabs(t[q]) for t in a)
        bitexact = all(isequal(a[i][q],b[i][q]) for i in eachindex(a))
        differing = findfirst(i -> !isequal(a[i][q],b[i][q]),eachindex(a))
        push!(comparisons,Dict("tag"=>tag,"quantity"=>q,"max_abs"=>err,
            "tolerance_abs"=>gate,"relative_to_signal"=>err/max(scale,floatmin(Float64)),
            "bit_exact"=>bitexact,"first_different_cycle"=>isnothing(differing) ? -1 : cycles[differing],
            "cycle_abs_errors"=>errs))
        check("$tag/$q",exact ? bitexact : err<=gate)
    end
    save_toml("raw/comparisons.toml",Dict("comparisons"=>comparisons))
end
for closure in (:nee,:moment), gpu in (false,true)
    backend = gpu ? CUDA.CUDABackend() : CPU()
    for trial in 1:(gpu ? 3 : 1)
        tag = string(closure,"_",gpu ? "gpu" : "cpu","_",trial)
        s = fresh(closure,backend)
        prefix = trajectory(s,7)
        path = "raw/restart_$tag.h5"
        Kraken.save_checkpoint(path,s)
        r = Kraken.load_checkpoint(Kraken.ECState,path;backend,injector_closure=closure)
        gpu && check("$tag/restored_backend",r.poisson_setup isa Kraken.EhdPoissonSetupGPU)
        disk_exact(s,r,tag)
        tail = trajectory(s,13); restarted = trajectory(r,13)
        save_bin("raw/continuous_$tag.bin",vcat(prefix,tail[2:end]))
        save_bin("raw/resumed_$tag.bin",restarted)
        compare(tail[2:end],restarted[2:end],closure,"$tag/restart",collect(8:20);exact=!gpu)
        for key in (:cycle_history,:parameter_cycles,:parameter_values,:config)
            check("$tag/final/$key",isequal(getproperty(s,key),getproperty(r,key)))
        end
        if gpu
            cold = fresh(closure,backend)
            repeated = trajectory(cold,20)
            save_bin("raw/heldout_$tag.bin",repeated)
            compare(vcat(prefix,tail[2:end]),repeated,closure,"$tag/cold",collect(0:20))
        end
        println("VALIDATION ",tag); flush(stdout)
        GC.gc(); gpu && CUDA.reclaim()
    end
end
# Diagnostic clones only: production prefix calls, never resumed after the prefix.
for closure in (:nee,:moment), trial in 1:3
    s = fresh(closure); Kraken.advance!(s,7)
    tag = "$(closure)_$(trial)"
    path = "raw/diagnostic_$tag.h5"; Kraken.save_checkpoint(path,s)
    r = Kraken.load_checkpoint(Kraken.ECState,path;backend=CUDA.CUDABackend(),injector_closure=closure)
    disk_exact(s,r,"diagnostic/$tag")
    before_a,before_b = stage_sample(s),stage_sample(r)
    # Ex/Ey are recomputed on restore and are not Poisson inputs.
    pre = stage_diff(before_a,before_b)
    Kraken.ehd_poisson_solve!(s.phi,s.poisson_setup,s.qfield)
    Kraken.ehd_poisson_solve!(r.phi,r.poisson_setup,r.qfield)
    after_a,after_b = stage_sample(s),stage_sample(r)
    post = stage_diff(after_a,after_b)
    for state in (s,r)
        Kraken.compute_electric_field_fd_2d!(state.Ex,state.Ey,state.phi,:neumann,10,16)
    end
    field_a,field_b = stage_sample(s),stage_sample(r)
    fd = stage_diff(field_a,field_b)
    inputs_exact = all(isequal(before_a[k],before_b[k]) for k in
        ("f_in","q_f_in","Fx_prev","Fy_prev","qfield","phi"))
    first = !inputs_exact ? "PRE_POISSON_INPUT" : post["phi"]>0 ? "POISSON" :
        (fd["Ex"]>0 || fd["Ey"]>0) ? "FIELD_RECONSTRUCTION" : "NOT_OBSERVED"
    push!(stages,Dict("tag"=>tag,"input_exact"=>inputs_exact,"first_difference"=>first,
        "before"=>pre,"after_poisson"=>post,"after_field"=>fd))
    save_bin("raw/stages_$tag.bin",(before_a,before_b,after_a,after_b,field_a,field_b))
    save_toml("raw/stages.toml",Dict("stages"=>stages))
    println("STAGE ",tag," ",first); flush(stdout)
    GC.gc(); CUDA.reclaim()
end
check("no_cudss_fallback",!Kraken._EHD_CUDSS_FALLBACK_WARNED[])
check("frozen_tolerances_unchanged",bytes2hex(sha256(read("raw/tolerances.toml")))==sealed)
passed = count(x->x["pass"],checks); failed=length(checks)-passed
save_toml("raw/status.toml",Dict("status"=>failed==0 ? "PASS_SCOPED_RESTART" : "FAIL",
    "passed"=>passed,"failed"=>failed,"tolerance_sha256"=>sealed,
    "physical_validation"=>"NOT_ASSESSED_BY_THIS_TEST","utc"=>string(now(UTC))))
println("RESTART_RESULT passed=",passed," failed=",failed)
failed==0 || error("Restart verification failed; preserve frozen gates and diagnostics")
