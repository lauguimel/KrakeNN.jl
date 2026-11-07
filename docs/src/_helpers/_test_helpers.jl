# # Helper self-test
#
# Proof-of-concept page for the Phase 4.1A doc infrastructure helpers.
# It exercises `source_extract`, `krk_download`, and `api_extract`
# end-to-end during the Documenter build. Not listed in the public nav.

# Load the helpers into this Literate sandbox module.
helpers_dir = @__DIR__
include(joinpath(helpers_dir, "source_extract.jl"))
include(joinpath(helpers_dir, "krk_download.jl"))
include(joinpath(helpers_dir, "api_extract.jl"))

# ## Source extraction
#
# Embed the real source of [`run_cavity_2d`](@ref) from `src/drivers/basic.jl`.

#md # ```@raw html
#md # <!-- begin embedded source -->
#md # ```

src_md = include_function(
    joinpath(@__DIR__, "..", "..", "..", "src", "drivers", "basic.jl"),
    :run_cavity_2d,
)
println(src_md)                                                                 #src

# ## KRK download badge

badge = krk_download(
    joinpath(@__DIR__, "..", "..", "..", "examples", "cavity.krk");
    build_dir = joinpath(@__DIR__, "..", "..", "build", "assets"),
)
println(badge)                                                                  #src

# ## API extraction

exports = extract_exports(
    joinpath(@__DIR__, "..", "..", "..", "src", "KrakeNN.jl"),
)
println("Exported symbols: ", length(exports))                                  #src

data = api_page_data(
    joinpath(@__DIR__, "..", "..", "..", "src", "KrakeNN.jl"),
)
println("In-scope categories: ", sort(collect(keys(data))))                     #src
