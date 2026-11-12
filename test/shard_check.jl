# Static check of the test sharding (test/shards.jl), run by CI with a bare
# Julia, no package loaded, in a few seconds:
#
#   julia --startup-file=no test/shard_check.jl
#
# It fails when
#   - runtests.jl includes a suite file without going through `test_file`
#     (that file would run in every shard, or a later edit could skip it);
#   - a file listed in runtests.jl does not exist;
#   - for some N, the shards do not partition the file list (a file in no
#     shard, in two shards, or in a shard outside 1..N);
#   - the shard count in .github/workflows/CI.yml disagrees with its matrix.
# It warns, without failing, about files missing from the cost table (they
# still run, in a hashed shard) and stale table entries.
using Test

include(joinpath(@__DIR__, "shards.jl"))
using .KrakenTestShards

const TESTDIR = @__DIR__
const RUNTESTS = read(joinpath(TESTDIR, "runtests.jl"), String)

# Raw `include` calls allowed in runtests.jl: the selector itself, its own
# body, and the single-file entry points that exit before the suite starts.
const ALLOWED_RAW = Set([
    "include(\"shards.jl\")",
    "include(path)",
    "include(only)",
    "include(\"ad/test_ad_sensitivity.jl\")",   # KRAKEN_AD_ONLY
])

strip_comments(s) = join((replace(l, r"#.*$" => "") for l in split(s, '\n')), '\n')

const CODE = strip_comments(RUNTESTS)
const SUITE = [m.captures[1] for m in eachmatch(r"\btest_file\(\"([^\"]+)\"\)", CODE)]
const RAW = [m.match for m in eachmatch(r"(?<![\w.!])include\([^)]*\)", CODE)]

@testset "test sharding" begin
    @testset "every suite include goes through test_file" begin
        bad = filter(r -> !(r in ALLOWED_RAW), RAW)
        isempty(bad) || @error "runtests.jl: use test_file(...) instead of include(...)" bad
        @test isempty(bad)
        @test length(SUITE) > 100
    end

    @testset "listed files exist, none twice" begin
        missing_files = filter(f -> !isfile(joinpath(TESTDIR, f)), SUITE)
        isempty(missing_files) || @error "runtests.jl lists missing files" missing_files
        @test isempty(missing_files)
        dups = unique(filter(f -> count(==(f), SUITE) > 1, SUITE))
        isempty(dups) || @error "runtests.jl lists files twice" dups
        @test isempty(dups)
    end

    @testset "shards partition the suite, N = $n" for n in 1:12
        asg = assignment(n)
        shards = [Set(f for f in SUITE if shard_of(f, n, asg) == k) for k in 1:n]
        @test all(f -> 1 <= shard_of(f, n, asg) <= n, SUITE)
        @test sum(length, shards) == length(unique(SUITE))       # disjoint
        @test reduce(union, shards) == Set(SUITE)                # complete
    end

    @testset "parse_shard" begin
        @test parse_shard("") === nothing
        @test parse_shard("2/6") == (2, 6)
        @test parse_shard(" 6 / 6 ") == (6, 6)
        @test_throws ErrorException parse_shard("0/6")
        @test_throws ErrorException parse_shard("7/6")
        @test_throws ErrorException parse_shard("2")
        @test_throws ErrorException parse_shard("a/b")
    end

    @testset "CI.yml shard count matches its matrix" begin
        ci_path = joinpath(TESTDIR, "..", ".github", "workflows", "CI.yml")
        if isfile(ci_path)
            ci = read(ci_path, String)
            mlist = match(r"shard:\s*\[([0-9,\s]+)\]", ci)
            mvar = match(r"KRAKEN_TEST_SHARD:\s*\"?\$\{\{\s*matrix\.shard\s*\}\}/(\d+)", ci)
            @test mlist !== nothing && mvar !== nothing
            if mlist !== nothing && mvar !== nothing
                ks = parse.(Int, split(strip(mlist.captures[1]), r"\s*,\s*"))
                @test ks == collect(1:parse(Int, mvar.captures[1]))
            end
        end
    end
end

# Advisory output: table coverage and the predicted load per shard.
let suite = Set(SUITE), table = Set(keys(KrakenTestShards.COST_SECONDS))
    untimed = sort!(collect(setdiff(suite, table)))
    stale = sort!(collect(setdiff(table, suite)))
    # The AD, 3D-viscoelastic, heavy and experimental files are expected here:
    # the pull-request tier skips them, so the table has no time for them.
    isempty(untimed) || @warn "Files without a measured cost (placed by hash; add PR-tier ones to COST_SECONDS in test/shards.jl)" untimed
    isempty(stale) || @warn "COST_SECONDS entries no longer in runtests.jl" stale
    for n in (4, 6, 8)
        asg = assignment(n)
        load = [sum((KrakenTestShards.COST_SECONDS[f] for f in table if asg[f] == k); init = 0) for k in 1:n]
        println("N = $n: predicted minutes per shard (Julia 1.12 table) ", round.(load ./ 60; digits = 1))
    end
end
