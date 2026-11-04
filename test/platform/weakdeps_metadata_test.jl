# Offline metadata regression test (#61): every [weakdeps] UUID in Project.toml
# must equal the UUID registered in General. A wrong UUID makes Pkg treat the
# package as a different one, so the extension silently never activates.
#
# The UUIDs below were read from the General registry (Registry.toml in
# ~/.julia/registries/General.tar.gz) on 2026-10-08. When adding a weakdep,
# look it up there and add it to this table.

using Test
using TOML

const REGISTERED_WEAKDEP_UUIDS = Dict(
    "CUDSS"       => "45b445bb-4962-46a0-9369-b4df9d0f772e",
    "CairoMakie"  => "13f3f980-e62b-5c42-98c6-ff1f3baf88f0",
    "Enzyme"      => "7da242da-08ed-463a-9acd-ee780be4f1d9",
    "LinearSolve" => "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae",
    "Optim"       => "429524aa-4258-5aef-a3af-852621145aeb",
)

@testset "Project.toml weakdeps match General registry UUIDs" begin
    proj = TOML.parsefile(joinpath(pkgdir(Kraken), "Project.toml"))
    weak = proj["weakdeps"]
    for (name, uuid) in weak
        @test haskey(REGISTERED_WEAKDEP_UUIDS, name)  # new weakdep: add it to the table
        haskey(REGISTERED_WEAKDEP_UUIDS, name) &&
            @test uuid == REGISTERED_WEAKDEP_UUIDS[name]
    end
    # Every extension trigger must be a declared weakdep or dep.
    alldeps = merge(get(proj, "deps", Dict()), weak)
    for (ext, trig) in get(proj, "extensions", Dict())
        for t in (trig isa AbstractString ? [trig] : trig)
            @test haskey(alldeps, t)
        end
    end
end
