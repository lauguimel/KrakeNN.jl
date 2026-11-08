module KrakeNNMakieExt

using KrakeNN
using CairoMakie

"""
    save_snapshot_png(path, field, field_name; colormap=:viridis, size=(800,600))

Save a 2D field as a heatmap PNG.
"""
function save_snapshot_png(path::String, field::Matrix, field_name::String;
                           colormap=:viridis, figsize=(800, 600))
    fig = Figure(; size=figsize)
    ax = Axis(fig[1, 1]; title=field_name, aspect=DataAspect())
    heatmap!(ax, field'; colormap=colormap)
    Colorbar(fig[1, 2]; colormap=colormap,
             limits=(minimum(field), maximum(field)))
    save(path, fig; px_per_unit=2)
    return nothing
end

"""
    save_animation_gif(path, frames, field_name; fps=10, colormap=:viridis, size=(800,600))

Assemble a vector of 2D arrays into an animated GIF.
"""
function save_animation_gif(path::String, frames::Vector{<:Matrix},
                            field_name::String;
                            fps::Int=10, colormap=:viridis, figsize=(800, 600))
    isempty(frames) && return nothing

    # Compute global color limits
    vmin = minimum(minimum.(frames))
    vmax = maximum(maximum.(frames))
    if vmin ≈ vmax
        vmax = vmin + one(vmin)
    end

    fig = Figure(; size=figsize)
    ax = Axis(fig[1, 1]; title=field_name, aspect=DataAspect())
    obs = Observable(frames[1]')
    heatmap!(ax, obs; colormap=colormap, colorrange=(vmin, vmax))
    Colorbar(fig[1, 2]; colormap=colormap, limits=(vmin, vmax))

    record(fig, path, 1:length(frames); framerate=fps) do i
        obs[] = frames[i]'
    end
    return nothing
end

# Register hooks on module load
function __init__()
    KrakeNN._png_saver[] = save_snapshot_png
    KrakeNN._gif_saver[] = save_animation_gif
end

end # module KrakeNNMakieExt
