# ============================================================================
# live_viewer.jl — Live VTR field viewer for KrakeNN simulations
# ============================================================================
#
# Usage:
#   julia --project=scripts scripts/live_viewer.jl <output_directory>
#
# Required packages (see scripts/Project.toml):
#   WGLMakie or GLMakie, ReadVTK
#
# Environment variables:
#   KRAKEN_MAKIE_BACKEND — "WGLMakie" (default) or "GLMakie"
#
# The viewer polls <output_directory> every 1 s for new .vtr files,
# reads them with ReadVTK, and displays a 2×2 heatmap dashboard:
#   C  (volume fraction)  — :blues  [0, 1]
#   |u| (velocity mag.)   — :viridis
#   φ  (level-set / field)— :RdBu
#   κ  (curvature)        — :turbo
# ============================================================================

# ── Backend selection ────────────────────────────────────────────────────────
const BACKEND = get(ENV, "KRAKEN_MAKIE_BACKEND", "WGLMakie")

if BACKEND == "GLMakie"
    @eval using GLMakie
elseif BACKEND == "WGLMakie"
    @eval using WGLMakie
else
    error("Unknown Makie backend: $BACKEND. Use \"WGLMakie\" or \"GLMakie\".")
end

using ReadVTK

# ── CLI argument ─────────────────────────────────────────────────────────────
if isempty(ARGS)
    println(stderr, "Usage: julia --project=scripts scripts/live_viewer.jl <output_directory>")
    exit(1)
end

const OUTDIR = ARGS[1]

if !isdir(OUTDIR)
    error("Directory not found: $OUTDIR")
end

# ── Helpers ──────────────────────────────────────────────────────────────────

"""
    extract_field(vtk, name, nx, ny) -> Matrix{Float64}

Extract a named field from a VTK rectilinear dataset and reshape to (nx, ny).
Returns a zero matrix if the field is absent.
"""
function extract_field(vtk, name::String, nx::Int, ny::Int)
    pd = get_point_data(vtk)
    if haskey(pd, name)
        raw = get_data(pd[name])
        # Flatten to Float64 vector, then reshape
        return reshape(Float64.(vec(raw)), nx, ny)
    else
        return zeros(Float64, nx, ny)
    end
end

"""
    velocity_magnitude(vtk, nx, ny) -> Matrix{Float64}

Compute |u| = sqrt(ux² + uy²) from the velocity field "u".
Falls back to zeros if "u" is absent.
"""
function velocity_magnitude(vtk, nx::Int, ny::Int)
    pd = get_point_data(vtk)
    if haskey(pd, "u")
        raw = get_data(pd["u"])  # (ndim, npoints)
        ux = Float64.(raw[1, :])
        uy = Float64.(raw[2, :])
        mag = sqrt.(ux .^ 2 .+ uy .^ 2)
        return reshape(mag, nx, ny)
    else
        return zeros(Float64, nx, ny)
    end
end

"""
    list_vtr_files(dir) -> Vector{String}

Return sorted list of .vtr file paths in `dir`.
"""
function list_vtr_files(dir::String)
    files = filter(f -> endswith(f, ".vtr"), readdir(dir))
    sort!(files)
    return [joinpath(dir, f) for f in files]
end

# ── Build figure with Observables ────────────────────────────────────────────
fig = Figure(size = (1200, 900))

# Initial dummy data
dummy = zeros(Float64, 2, 2)

obs_C   = Observable(dummy)
obs_umag = Observable(dummy)
obs_phi = Observable(dummy)
obs_kappa = Observable(dummy)
obs_title = Observable("Waiting for VTR files…")

Label(fig[0, 1:2], obs_title; fontsize = 18, halign = :center)

# C — volume fraction
ax1 = Axis(fig[1, 1]; title = "C (volume fraction)", aspect = DataAspect())
hm1 = heatmap!(ax1, obs_C; colormap = :blues, colorrange = (0.0, 1.0))
Colorbar(fig[1, 1][1, 2], hm1)

# |u| — velocity magnitude
ax2 = Axis(fig[1, 2]; title = "|u| (velocity)", aspect = DataAspect())
hm2 = heatmap!(ax2, obs_umag; colormap = :viridis)
Colorbar(fig[1, 2][1, 2], hm2)

# φ — level-set / signed distance
ax3 = Axis(fig[2, 1]; title = "φ (level-set)", aspect = DataAspect())
hm3 = heatmap!(ax3, obs_phi; colormap = :RdBu)
Colorbar(fig[2, 1][1, 2], hm3)

# κ — curvature
ax4 = Axis(fig[2, 2]; title = "κ (curvature)", aspect = DataAspect())
hm4 = heatmap!(ax4, obs_kappa; colormap = :turbo)
Colorbar(fig[2, 2][1, 2], hm4)

display(fig)

# ── Polling loop ─────────────────────────────────────────────────────────────
seen = Set{String}()

println("Live viewer started — watching $OUTDIR (poll every 1 s)")
println("Press Ctrl-C to stop.")

try
    while true
        files = list_vtr_files(OUTDIR)

        for fpath in files
            fname = basename(fpath)
            fname in seen && continue
            push!(seen, fname)

            println("  Loading $fname …")
            vtk = VTKFile(fpath)

            # Determine grid dimensions from the rectilinear grid coordinates
            coords = get_coordinates(vtk)
            nx = length(coords[1])
            ny = length(coords[2])

            # Update Observables
            obs_C[]    = extract_field(vtk, "C", nx, ny)
            obs_umag[] = velocity_magnitude(vtk, nx, ny)
            obs_phi[]  = extract_field(vtk, "phi", nx, ny)
            obs_kappa[] = extract_field(vtk, "kappa", nx, ny)
            obs_title[] = fname
        end

        sleep(1.0)
    end
catch e
    if e isa InterruptException
        println("\nViewer stopped.")
    else
        rethrow(e)
    end
end
