# V5: all stages (a) to (e) in one Julia process (one compilation; meant for
# the CPU smoke run, also usable on a GPU). Stops at the first stage whose
# run fails (exit code 1); the science gate of stage (d) is printed, it does
# not stop the chain. FV5_STAGES (default "abcde") selects a subset.
#
#   FV3D_BACKEND=cpu FV5_N=12 FV5_OUT=<dir> julia --project=<env> benchmarks/fv3d/v5_all.jl
if lowercase(get(ENV, "FV3D_BACKEND", "cuda")) == "cuda"
    using CUDA
    CUDA.functional() || error("CUDA not functional")
    using CUDSS
end
include(joinpath(@__DIR__, "v5_common.jl"))
for f in ("v5a_truth_lbm.jl", "v5b_piv.jl", "v5c_model_floor.jl", "v5d_fit_carreau.jl", "v5e_fit_mlp.jl")
    include(joinpath(@__DIR__, f))
end

function v5_all_main()
    stages = Dict('a' => v5a_main, 'b' => v5b_main, 'c' => v5c_main, 'd' => v5d_main, 'e' => v5e_main)
    t0 = time()
    for ch in get(ENV, "FV5_STAGES", "abcde")
        haskey(stages, ch) || error("unknown stage '$ch' in FV5_STAGES")
        t = @elapsed code = stages[ch]()
        @printf("STAGE %c exit %d  %.0f s (total %.0f s)\n", ch, code, t, time() - t0)
        flush(stdout)
        code == 0 || return code
    end
    return 0
end

exit(v5_all_main())
