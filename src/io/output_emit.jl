
"""Write VTK output."""
function _write_output(ρ, ux, uy, setup::SimulationSetup, out::OutputSetup,
                       pvd, output_dir, dx, step;
                       extra_fields=Dict{String,Any}())
    fields_dict = Dict{String, Matrix{Float64}}()
    field_set = Set(out.fields)

    if :rho in field_set
        fields_dict["rho"] = Array(ρ)
    end
    if :ux in field_set
        fields_dict["ux"] = Array(ux)
    end
    if :uy in field_set
        fields_dict["uy"] = Array(uy)
    end

    # Merge extra fields (C, phi, kappa, etc.)
    for (k, v) in extra_fields
        if Symbol(k) in field_set
            fields_dict[k] = v isa AbstractArray ? Array(v) : v
        end
    end

    Nx, Ny = setup.domain.Nx, setup.domain.Ny
    fname = joinpath(output_dir, "$(setup.name)_$(lpad(step, 8, '0'))")
    write_vtk_to_pvd(pvd, fname, Nx, Ny, dx, fields_dict, Float64(step))
end

# --- PNG/GIF output helpers ---

"""Compute a requested field from macroscopic arrays."""
function _compute_field(field::Symbol, ρ, ux, uy)
    if field == Symbol("|u|")
        return sqrt.(Array(ux).^2 .+ Array(uy).^2)
    elseif field == :rho
        return Array(ρ)
    elseif field == :ux
        return Array(ux)
    elseif field == :uy
        return Array(uy)
    else
        return Array(ρ)  # fallback
    end
end

"""Emit a warning if png/gif output is requested but CairoMakie is not loaded."""
function _check_image_backend(png_out, gif_out)
    need = png_out !== nothing || gif_out !== nothing
    if need && _png_saver[] === nothing
        @warn "Output png/gif requested but CairoMakie is not loaded. " *
              "Add `using CairoMakie` before `using KrakeNN` to enable PNG/GIF output."
    end
end

"""Initialize GIF frame storage."""
function _init_gif_frames(gif_out)
    gif_out === nothing && return Dict{Symbol, Vector{Matrix{Float64}}}()
    frames = Dict{Symbol, Vector{Matrix{Float64}}}()
    for f in gif_out.fields
        frames[f] = Matrix{Float64}[]
    end
    return frames
end

"""Save a PNG snapshot if it's time and the backend is loaded."""
function _maybe_save_png(png_out, ρ, ux, uy, setup, output_dir, step)
    png_out === nothing && return
    _png_saver[] === nothing && return
    step % png_out.interval != 0 && return

    dir = isempty(output_dir) ? setup_output_dir(png_out.directory) : output_dir
    for field_name in png_out.fields
        data = _compute_field(field_name, ρ, ux, uy)
        fname = joinpath(dir, "$(setup.name)_$(field_name)_$(lpad(step, 8, '0')).png")
        _png_saver[](fname, data, string(field_name))
    end
end

"""Collect a GIF frame if it's time."""
function _maybe_collect_gif(gif_out, gif_frames, ρ, ux, uy, step)
    gif_out === nothing && return
    _gif_saver[] === nothing && return
    step % gif_out.interval != 0 && return

    for field_name in gif_out.fields
        data = _compute_field(field_name, ρ, ux, uy)
        push!(gif_frames[field_name], copy(data))
    end
end

"""Assemble and save GIF after simulation completes."""
function _maybe_save_gif(gif_out, gif_frames, setup, output_dir)
    gif_out === nothing && return
    _gif_saver[] === nothing && return

    dir = isempty(output_dir) ? setup_output_dir(gif_out.directory) : output_dir
    for field_name in gif_out.fields
        frames = gif_frames[field_name]
        isempty(frames) && continue
        fname = joinpath(dir, "$(setup.name)_$(field_name).gif")
        _gif_saver[](fname, frames, string(field_name); fps=gif_out.fps)
    end
end

"""Write 3D VTK output using the existing VTK writer infrastructure."""
function _write_output_3d(ρ, ux, uy, uz, setup, vtk_out, pvd, output_dir, dx, step)
    Nx, Ny, Nz = size(ρ)
    fname = joinpath(output_dir, "$(setup.name)_$(lpad(step, 8, '0'))")
    fields = Dict{String, Array{Float64, 3}}(
        "rho" => Array(ρ),
        "ux"  => Array(ux),
        "uy"  => Array(uy),
        "uz"  => Array(uz),
    )
    write_vtk_to_pvd(pvd, fname, Nx, Ny, Nz, dx, fields, Float64(step))
end
