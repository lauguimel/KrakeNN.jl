using Documenter
using DocumenterCitations
using DocumenterVitepress
using Literate
using NodeJS_20_jll
using PlutoStaticHTML
using KrakeNN

const DOCS_LINKCHECK = lowercase(get(ENV, "DOCUMENTER_LINKCHECK", "false")) in ("1", "true", "yes")

function lint_implication_maps()
    agent_dir = joinpath(@__DIR__, "agent")
    linter = joinpath(@__DIR__, "..", "scripts", "lint-implication-map.sh")
    isfile(linter) || error("Track-C linter not found: $(linter)")
    isdir(agent_dir) || error("Track-C agent docs directory not found: $(agent_dir)")

    maps = sort(filter(name -> endswith(name, "-implication.md"), readdir(agent_dir; join = true)))
    isempty(maps) && error("No Track-C implication maps found under $(agent_dir)")

    for map in maps
        @info "Linting Track-C implication map" map
        try
            run(`bash $linter $map`)
        catch err
            error("Track-C implication map lint failed: $(map)\n$(sprint(showerror, err))")
        end
    end
end

lint_implication_maps()

# --- Living-documentation helpers (Phase 4.1A) ---
# Loaded into Main so Literate.jl preprocessing and @example blocks
# can call extract_function / krk_download / api_page_data directly.
include(joinpath(@__DIR__, "src", "_helpers", "source_extract.jl"))
include(joinpath(@__DIR__, "src", "_helpers", "krk_download.jl"))
include(joinpath(@__DIR__, "src", "_helpers", "api_extract.jl"))

# --- Process Literate.jl files ---

const DOCS_SRC = joinpath(@__DIR__, "src")

const LITERATE_DIRS = [
    "theory",
    "examples",
    "benchmarks",
    "tutorials",
]

# --- Living-doc preprocessing: expand @@EXTRACT path symbol@@ markers ---
# Markers in Literate sources are replaced at build-time with a fenced
# Julia code block extracted from the real source file via
# `extract_function`. This keeps theory pages in sync with the code.
const _SRC_ROOT = joinpath(@__DIR__, "..")

# The `@@EXTRACT path symbol@@` marker sits inside a Literate `# ...`
# comment block. It is expanded at build time into a fenced julia code
# block that still lives in the Literate comment region (each emitted
# line is prefixed with `# `), so Literate renders it as raw markdown.
function literate_preprocess(content::AbstractString)
    re = r"(?m)^#\s*@@EXTRACT\s+(\S+)\s+(\S+?)@@\s*$"
    return replace(content, re => function (m)
        mm = match(re, m)
        relpath = String(mm.captures[1])
        symname = Symbol(String(mm.captures[2]))
        filepath = joinpath(_SRC_ROOT, relpath)
        try
            r = extract_function(filepath, symname)
            # Emit as a Literate markdown block: each line prefixed with "# "
            lines = split(r.full_text, '\n')
            return join(("# " * l for l in lines), '\n')
        catch err
            @warn "EXTRACT marker expansion failed" relpath symname err
            return "# `extract failed: $(relpath) :$(symname)`"
        end
    end)
end

for dir in LITERATE_DIRS
    src_dir = joinpath(DOCS_SRC, dir)
    out_dir = joinpath(DOCS_SRC, dir)
    for file in sort(readdir(src_dir))
        endswith(file, ".jl") || continue
        Literate.markdown(
            joinpath(src_dir, file), out_dir;
            documenter = true,
            credit = false,
            preprocess = literate_preprocess,
            # Non-executable code blocks: show code without running simulations
            # To enable execution (for CI with GPU), change to:
            #   codefence = nothing  (default, generates @example blocks)
            codefence = "```julia" => "```",
        )
    end
end

# --- Phase 4.1A proof-of-concept: build _helpers/_test_helpers.jl ---
# Only the `_test_helpers.jl` file in docs/src/_helpers/ is a Literate page;
# the other .jl files there are plain Julia helper modules loaded above.
let helpers_dir = joinpath(DOCS_SRC, "_helpers"),
    test_file  = joinpath(helpers_dir, "_test_helpers.jl")
    if isfile(test_file)
        Literate.markdown(
            test_file, helpers_dir;
            documenter = true,
            credit = false,
            execute = true,
        )
    end
end

# --- Process Pluto notebooks (interactive tutorials with WGLMakie) ---

const TUTORIAL_DIR = joinpath(DOCS_SRC, "tutorials")

# Pluto notebooks (if any): only process files that start with Pluto header
if isdir(TUTORIAL_DIR)
    pluto_files = filter(readdir(TUTORIAL_DIR)) do f
        endswith(f, ".jl") || return false
        first_line = readline(joinpath(TUTORIAL_DIR, f))
        return startswith(first_line, "### A Pluto.jl notebook ###")
    end
    if !isempty(pluto_files)
        @info "Building Pluto notebooks → tutorials/" pluto_files
        bopts = BuildOptions(TUTORIAL_DIR;
            output_format = documenter_output,
            use_distributed = false,
        )
        build_notebooks(bopts, sort(pluto_files))
    end
end

# --- Bibliography ---

bib = CitationBibliography(joinpath(@__DIR__, "refs.bib"); style=:numeric)

# --- Build documentation ---

# Note: on Julia 1.12+ with Metal.jl, makedocs may segfault due to libgit2.
# Workaround: run from a temp directory with remotes=nothing, or use CI.

makedocs(;
    sitename = "KrakeNN",
    modules = [KrakeNN],
    plugins = [bib],
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/lauguimel/KrakeNN.jl",
        devurl = "dev",
        devbranch = "release/v0.5",
        build_vitepress = false,
        keep = :patch,
    ),
    pages = [
        "Home" => "index.md",
        "Guide" => [
            "Getting started" => "getting_started.md",
            "Installation" => "installation.md",
            "Concepts" => "concepts_index.md",
            "Capabilities" => "capabilities.md",
            "Architecture" => "architecture.md",
            "KRK reference" => "users/krk-reference.md",
            "Incompressible Navier–Stokes (FVFD/SIMPLE)" => "users/incompressible-navier-stokes.md",
            "Thermal coupling (steady FV)" => "users/fv-thermal.md",
            "Simulation state and checkpoints" => "users/simulation-state-checkpoints.md",
        ],
        "Examples" => [
            "Newtonian" => [
                "Poiseuille (2D)" => "examples/01_poiseuille_2d.md",
                "Couette (2D)" => "examples/02_couette_2d.md",
                "Taylor–Green (2D)" => "examples/03_taylor_green_2d.md",
                "Lid-driven cavity (2D & 3D)" => "examples/04_cavity_2d.md",
                "Cylinder (2D)" => "examples/06_cylinder_2d.md",
                "Hagen–Poiseuille" => "examples/09_hagen_poiseuille.md",
            ],
            "Thermal" => [
                "Heat conduction" => "examples/07_heat_conduction.md",
                "Rayleigh–Bénard" => "examples/08_rayleigh_benard.md",
            ],
            "Non-Newtonian" => [
                "Viscoelastic cylinder" => "users/tutorials/viscoelastic-cylinder.md",
            ],
            "Geometry / STL" => [
                "Sphere drag 3D" => "users/tutorials/sphere-drag-3d.md",
            ],
            "Configuration (.krk)" => [
                "KRK config" => "examples/10_krk_config.md",
            ],
        ],
        "Benchmarks" => [
            "Validation matrix" => "users/benchmarks/validation-matrix.md",
            "Cartesian cavity" => "users/benchmarks/cartesian-cavity.md",
            "Thermal natural convection" => "users/benchmarks/thermal-natural-convection.md",
            "Generalised Newtonian natural convection" => "users/benchmarks/gnf-natural-convection.md",
            "Sphere drag 3D" => "users/benchmarks/sphere-drag-3d.md",
            "Viscoelastic cylinder (Oldroyd-B)" => "users/benchmarks/viscoelastic-cylinder.md",
            "Viscoelastic constitutive models" => "users/benchmarks/ve-constitutive-models.md",
            "Viscoelastic Poiseuille (FVFD 3D)" => "users/benchmarks/ve3d-poiseuille-convergence.md",
            "Generalised-Newtonian channel (3D)" => "users/benchmarks/gnf3d-channel-convergence.md",
            "Steady shape sensitivity (AD)" => "users/benchmarks/ad-shape-sensitivity.md",
            "Steady shape sensitivity — viscoelastic (AD)" => "users/benchmarks/ad-shape-sensitivity-viscoelastic.md",
            "GNF adjoint gradient" => "users/benchmarks/gnf-adjoint-gradient.md",
            "GPU certification" => "users/benchmarks/gpu-certification.md",
            "Electroconvection (EHD)" => "users/benchmarks/electroconvection.md",
            "Performance" => "benchmarks/performance.md",
            "Accuracy" => "benchmarks/accuracy.md",
            "External comparisons" => "benchmarks/external.md",
            "Hardware" => "benchmarks/hardware.md",
        ],
        "Reference" => [
            ".krk DSL" => [
                "krk/overview.md",
                "krk/directives.md",
                "krk/bc_types.md",
                "krk/modules.md",
                "krk/presets.md",
                "krk/helpers.md",
                "krk/expressions.md",
                "krk/sanity.md",
                "krk/errors.md",
                "krk/aliases.md",
            ],
            "API" => [
                "Units" => "api/units.md",
                "Geometry" => "api/geometry.md",
                "LBM" => "api/lbm.md",
                "Physics: Newtonian" => "api/physics-newtonian.md",
                "Physics: Viscoelastic" => "api/physics-viscoelastic.md",
                "Physics: Thermal" => "api/physics-thermal.md",
                "Boundary conditions" => "api/bc.md",
                "Backend" => "api/backend.md",
                "KRK I/O" => "api/io-krk.md",
                "Platform state & checkpoints" => "api/platform.md",
            ],
            "Julia API" => [
                "api/lattice.md",
                "api/collision.md",
                "api/streaming.md",
                "api/boundary.md",
                "api/macroscopic.md",
                "api/drivers.md",
                "api/io.md",
                "api/postprocess.md",
                "api/config.md",
            ],
            # v0.2.0 scope: single-phase LBM (2D/3D), thermal,
            # spatial BCs, .krk DSL. Out-of-scope pages (phasefield, VOF/PLIC,
            # rheology, viscoelastic, Shan-Chen, species) are excluded here.
            "Theory" => [
                "theory/01_lbm_fundamentals.md",
                "theory/02_d2q9_lattice.md",
                "theory/03_bgk_collision.md",
                "theory/04_streaming.md",
                "theory/05_boundary_conditions.md",
                "theory/06_from_2d_to_3d.md",
                "theory/07_body_forces.md",
                "theory/08_thermal_ddf.md",
                "theory/09_axisymmetric.md",
                "theory/10_limitations.md",
                "theory/12_mrt.md",
                "theory/14_rheology.md",
                "theory/15_viscoelastic.md",
                "theory/19_spatial_bcs.md",
            ],
            # The one page carrying a @bibliography block. Without it every
            # citation in the docs has no destination and DocumenterCitations
            # degrades it to plain unlinked text (#46).
            "References" => "references.md",
        ],
    ],
    remotes = nothing,
    linkcheck = DOCS_LINKCHECK,
    warnonly = DOCS_LINKCHECK ? Documenter.except(:linkcheck) : true,
    checkdocs = :none,
)

# --- Prune orphan pages, then invoke Vitepress build manually ---
# DocumenterVitepress copies all of `docs/src/` into `build/.documenter/`.
# Keep the v0.2 navigation strict while preserving hidden pages that visible
# pages link to, so Vitepress dead-link checks stay useful.
const VITEPRESS_KEEP_HIDDEN = Set{String}([
    "benchmarks/mlups_cpu_gpu.md",
])

function generated_sidebar_pages(config_path::AbstractString)
    config = read(config_path, String)
    pages = Set{String}()
    for m in eachmatch(r"link: '/([^'#?]*)'", config)
        link = String(m.captures[1])
        isempty(link) && continue
        push!(pages, link * ".md")
    end
    return pages
end

function prune_vitepress_markdown!(vp_input::AbstractString)
    keep = union(
        generated_sidebar_pages(joinpath(vp_input, ".vitepress", "config.mts")),
        VITEPRESS_KEEP_HIDDEN,
    )
    for (root, _, files) in walkdir(vp_input)
        for file in files
            endswith(file, ".md") || continue
            path = joinpath(root, file)
            rel = replace(relpath(path, vp_input), '\\' => '/')
            rel in keep || rm(path; force = true)
        end
    end
end

function prune_vitepress_bases!(build_dir::AbstractString)
    bases_file = joinpath(build_dir, "bases.txt")
    bases = isfile(bases_file) ? readlines(bases_file) : String[]
    patch_base = r"^v\d+\.\d+\.\d+(?:[-+].*)?$"
    if any(base -> occursin(patch_base, base), bases)
        bases = filter(base -> base == "stable" || occursin(patch_base, base), bases)
        open(bases_file, "w") do io
            foreach(base -> println(io, base), bases)
        end
    end
    return bases
end

function vitepress_base_url(deploy_abspath::AbstractString, base::AbstractString)
    deploy_relpath = isempty(base) ? "" : "$(base)/"
    return deploy_abspath == "/" ? "/$(deploy_relpath)" : "$(deploy_abspath)/$(deploy_relpath)"
end

function vitepress_current_version(bases)
    patch_base = r"^v\d+\.\d+\.\d+(?:[-+].*)?$"
    patch = findfirst(base -> occursin(patch_base, base), bases)
    patch !== nothing && return bases[patch]
    nonempty = filter(!isempty, bases)
    return isempty(nonempty) ? "" : first(nonempty)
end

function npm_executable()
    for candidate in ("/opt/homebrew/bin/npm", Sys.which("npm"))
        candidate === nothing && continue
        isfile(candidate) && return candidate
    end
    error("npm not found; install Node 20 or run through the docs CI setup-node step")
end

function build_vitepress_outputs!(vp_input::AbstractString)
    build_dir = joinpath(@__DIR__, "build")
    config_path = joinpath(vp_input, ".vitepress", "config.mts")
    bases = prune_vitepress_bases!(build_dir)
    isempty(bases) && return

    template_config = read(config_path, String)
    deploy_abspath_match = match(
        r"__DEPLOY_ABSPATH__:\s*JSON\.stringify\('([^']*)'\)",
        template_config,
    )
    deploy_abspath = deploy_abspath_match === nothing ? "/" : deploy_abspath_match.captures[1]
    current_version = vitepress_current_version(bases)

    cd(@__DIR__) do
        # docs/package.json and docs/package-lock.json are tracked so the npm
        # tree is reproducible: the DocumenterVitepress template uses caret
        # ranges, and on 2026-10-13 `@nolebase/*` 2.21 started requiring
        # vitepress 2, which broke `npm install` (ERESOLVE). docs/package.json
        # pins the plugin to ~2.18.2 and `overrides` its `@nolebase/ui`
        # dependency to ~2.18.2 too (the plugin asks for ^2.18.2, which would
        # pull ui 2.21 with an unmet vitepress-2 peer). Drop both pins only
        # when moving to vitepress 2. The template copy below is only a
        # fallback for a checkout without docs/package.json.
        tmpl_pkg = joinpath(dirname(pathof(DocumenterVitepress)), "..", "template", "package.json")
        pkg_json = joinpath(@__DIR__, "package.json")
        lock_json = joinpath(@__DIR__, "package-lock.json")
        cleanup_pkg = !isfile(pkg_json)
        cleanup_pkg && cp(tmpl_pkg, pkg_json)
        try
            npm_bin = npm_executable()
            if !cleanup_pkg && isfile(lock_json)
                # Install exactly the locked tree; fails if package.json and
                # the lockfile disagree instead of silently re-resolving.
                run(`$(npm_bin) ci --no-audit --no-fund`)
            else
                run(`$(npm_bin) install --no-audit --no-fund`)
            end
            for (i, base) in enumerate(bases)
                base_url = vitepress_base_url(deploy_abspath, base)
                config = replace(
                    template_config,
                    r"base: '[^']*'" => "base: '$(base_url)'",
                    r"outDir: '../[^']*'" => "outDir: '../$(i)'",
                )
                write(config_path, config)
                rm(joinpath(build_dir, string(i)); recursive = true, force = true)
                run(`$(npm_bin) run env -- vitepress build $(vp_input)`)
                open(joinpath(build_dir, string(i), "siteinfo.js"), "w") do io
                    println(io, """var DOCUMENTER_CURRENT_VERSION = "$(current_version)";""")
                end
            end
        finally
            if cleanup_pkg
                rm(pkg_json; force = true)
                rm(lock_json; force = true)
            end
        end
    end
end

let vp_input = joinpath(@__DIR__, "build", ".documenter")
    rm(joinpath(vp_input, "_helpers"); recursive = true, force = true)
    prune_vitepress_markdown!(vp_input)

    # DocumenterVitepress rewrites `../assets/...` links inside `examples/`
    # to `assets/...`. Mirror downloadable .krk files at that rewritten path
    # so VitePress dead-link checks stay strict.
    krk_src = joinpath(vp_input, "assets", "krk")
    krk_dst = joinpath(vp_input, "examples", "assets", "krk")
    if isdir(krk_src)
        mkpath(krk_dst)
        for file in readdir(krk_src; join = true)
            endswith(file, ".krk") && cp(file, joinpath(krk_dst, basename(file)); force = true)
        end
    end

    build_vitepress_outputs!(vp_input)
end

if startswith(get(ENV, "GITHUB_REF", ""), "refs/tags/v")
    DocumenterVitepress.deploydocs(;
        repo = "github.com/lauguimel/KrakeNN.jl.git",
        target = joinpath(@__DIR__, "build"),
        devbranch = "release/v0.5",
        branch = "gh-pages",
        push_preview = true,
    )
else
    @info "Skipping docs deploy; deployment is restricted to v* tags"
end
