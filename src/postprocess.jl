# --- Post-processing helpers for run_simulation results ---

"""
    extract_line(result, field, axis; at=0.5)

Extract a 1D profile from a 2D simulation result.

- `field`: field name (`:ux`, `:uy`, `:rho`)
- `axis`: axis along which to extract (`:x` or `:y`)
- `at`: position on the perpendicular axis, in physical units or as a
  fraction of the domain if `0 < at ≤ 1` and `at` is given without units.

Returns a NamedTuple `(coord, values)`.

# Example
```julia
result = run_simulation("couette.krk")
prof = extract_line(result, :ux, :y; at=0.5)
lines(prof.values, prof.coord)  # plot profile
```
"""
function extract_line(result::NamedTuple, field::Symbol, axis::Symbol; at::Real=0.5)
    data = getfield(result, field)
    setup = result.setup
    Nx, Ny = setup.domain.Nx, setup.domain.Ny
    dx = setup.domain.Lx / Nx
    dy = setup.domain.Ly / Ny

    if axis == :y
        # Profile along y at fixed x
        x_pos = _resolve_position(at, setup.domain.Lx)
        i = clamp(round(Int, x_pos / dx + 0.5), 1, Nx)
        coord = [(j - 0.5) * dy for j in 1:Ny]
        values = data[i, :]
        return (coord=coord, values=values, axis=:y, at_x=x_pos, index=i)
    elseif axis == :x
        # Profile along x at fixed y
        y_pos = _resolve_position(at, setup.domain.Ly)
        j = clamp(round(Int, y_pos / dy + 0.5), 1, Ny)
        coord = [(i - 0.5) * dx for i in 1:Nx]
        values = data[:, j]
        return (coord=coord, values=values, axis=:x, at_y=y_pos, index=j)
    else
        throw(ArgumentError("axis must be :x or :y, got :$axis"))
    end
end

"""
    field_error(result, field, analytical; norm=:L2)

Compute the error between a simulation field and an analytical expression.

- `field`: field name (`:ux`, `:uy`, `:rho`)
- `analytical`: expression string using `x`, `y`, and any `Define` variables
  from the `.krk` file (e.g. `"u_wall * y / H"`)
- `norm`: `:L2`, `:Linf`, or `:L1`

Returns a NamedTuple `(error, norm, analytical_field)`.

# Example
```julia
result = run_simulation("couette.krk")
err = field_error(result, :ux, "u_wall * (y - 0.5*dy) / (Ny*dy - dy)")
err.error  # L2 relative error
```
"""
function field_error(result::NamedTuple, field::Symbol, analytical::AbstractString;
                     norm::Symbol=:L2)
    data = getfield(result, field)
    setup = result.setup
    Nx, Ny = setup.domain.Nx, setup.domain.Ny
    dx = setup.domain.Lx / Nx
    dy = setup.domain.Ly / Ny

    # Parse analytical expression with user variables
    expr = parse_kraken_expr(analytical, setup.user_vars)

    # Evaluate on the grid
    ana = zeros(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        x = (i - 0.5) * dx
        y = (j - 0.5) * dy
        ana[i, j] = evaluate(expr; x=x, y=y,
                             Lx=setup.domain.Lx, Ly=setup.domain.Ly,
                             Nx=Float64(Nx), Ny=Float64(Ny),
                             dx=dx, dy=dy)
    end

    diff = data .- ana
    err = _compute_norm(diff, ana, norm)

    return (error=err, norm=norm, analytical_field=ana)
end

"""
    field_error(result, field, analytical_fn::Function; norm=:L2)

Compute error using a Julia function `(x, y) -> value`.
"""
function field_error(result::NamedTuple, field::Symbol, analytical_fn::Function;
                     norm::Symbol=:L2)
    data = getfield(result, field)
    setup = result.setup
    Nx, Ny = setup.domain.Nx, setup.domain.Ny
    dx = setup.domain.Lx / Nx
    dy = setup.domain.Ly / Ny

    ana = zeros(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        ana[i, j] = analytical_fn((i - 0.5) * dx, (j - 0.5) * dy)
    end

    diff = data .- ana
    err = _compute_norm(diff, ana, norm)

    return (error=err, norm=norm, analytical_field=ana)
end

"""
    probe(result, field, x, y)

Sample a field at a physical location `(x, y)` using nearest-node interpolation.

# Example
```julia
result = run_simulation("cavity.krk")
probe(result, :ux, 0.5, 0.5)  # velocity at domain center
```
"""
function probe(result::NamedTuple, field::Symbol, x::Real, y::Real)
    data = getfield(result, field)
    setup = result.setup
    dx = setup.domain.Lx / setup.domain.Nx
    dy = setup.domain.Ly / setup.domain.Ny
    i = clamp(round(Int, x / dx + 0.5), 1, setup.domain.Nx)
    j = clamp(round(Int, y / dy + 0.5), 1, setup.domain.Ny)
    return data[i, j]
end

"""
    domain_stats(result)

Compute global statistics of the simulation result.

Returns a NamedTuple with `max_u`, `mean_rho`, `mass_error`, `max_ux`, `max_uy`.
"""
function domain_stats(result::NamedTuple)
    Nx, Ny = result.setup.domain.Nx, result.setup.domain.Ny
    umag = @. sqrt(result.ux^2 + result.uy^2)
    mean_rho = sum(result.ρ) / (Nx * Ny)
    return (
        max_u      = maximum(umag),
        max_ux     = maximum(abs.(result.ux)),
        max_uy     = maximum(abs.(result.uy)),
        mean_rho   = mean_rho,
        mass_error = abs(mean_rho - 1.0),
    )
end

# --- Basilisk interface data loader ---

"""
    load_basilisk_interfaces(filepath::String)

Load a Basilisk interface file (gnuplot segment format).
Returns a vector of segments, each segment being a pair of (z, r) points.

File format: pairs of `z r` lines separated by blank lines.
"""
function load_basilisk_interfaces(filepath::String)
    segments = Vector{NTuple{2, NTuple{2, Float64}}}()
    pts = NTuple{2, Float64}[]

    for line in eachline(filepath)
        stripped = strip(line)
        if isempty(stripped)
            if length(pts) == 2
                push!(segments, (pts[1], pts[2]))
            end
            empty!(pts)
        else
            vals = split(stripped)
            if length(vals) >= 2
                z = parse(Float64, vals[1])
                r = parse(Float64, vals[2])
                push!(pts, (z, r))
            end
        end
    end
    # Handle last segment (no trailing blank line)
    if length(pts) == 2
        push!(segments, (pts[1], pts[2]))
    end
    return segments
end

"""
    load_basilisk_interface_contour(filepath::String)

Load a Basilisk interface file and return a sorted (z, r) contour.
Extracts midpoints of each segment and sorts by z coordinate.
"""
function load_basilisk_interface_contour(filepath::String)
    segments = load_basilisk_interfaces(filepath)
    contour = NTuple{2, Float64}[]
    for ((z1, r1), (z2, r2)) in segments
        push!(contour, ((z1 + z2) / 2, (r1 + r2) / 2))
    end
    sort!(contour, by=first)
    return contour
end

"""
    find_basilisk_snapshot(data_dir, Re, Amp, t_phys; tol=0.1)

Find the Basilisk interface file closest to physical time `t_phys`.
Returns the filepath or `nothing` if not found.

`data_dir` should point to the ds_num/ directory.
"""
function find_basilisk_snapshot(data_dir::String, Re::Real, Amp::Real, t_phys::Real; tol=0.5)
    folder = joinpath(data_dir, "$(Int(Re))_$(lpad(string(Amp), 5, '0'))")
    if !isdir(folder)
        # Try with different formatting
        for d in readdir(data_dir)
            parts = split(d, "_")
            if length(parts) == 2
                re_str, amp_str = parts
                re_val = tryparse(Int, re_str)
                amp_val = tryparse(Float64, amp_str)
                if re_val == Int(Re) && amp_val !== nothing && abs(amp_val - Amp) < 1e-6
                    folder = joinpath(data_dir, d)
                    break
                end
            end
        end
    end
    !isdir(folder) && return nothing

    best_file = nothing
    best_dt = Inf
    for f in readdir(folder)
        startswith(f, "interfaces_") && endswith(f, ".dat") || continue
        # Parse time from filename: interfaces_Re_Amp_Time.dat
        m = match(r"interfaces_[\d.]+_[\d.]+_([\d.]+)\.dat", f)
        m === nothing && continue
        t = parse(Float64, m.captures[1])
        dt = abs(t - t_phys)
        if dt < best_dt
            best_dt = dt
            best_file = joinpath(folder, f)
        end
    end
    best_dt > tol ? nothing : best_file
end

"""
    compare_interfaces(kraken_pts, basilisk_contour, R0_lb)

Compare KrakeNN interface points with Basilisk contour.
KrakeNN points are in lattice units; Basilisk in physical units (R₀=1).

Returns (z_krak, r_krak, z_bas, r_bas) all in physical units (normalized by R₀).
"""
function compare_interfaces(kraken_pts::Vector{<:NTuple{2}}, basilisk_contour, R0_lb::Real)
    z_krak = [p[1] / R0_lb for p in kraken_pts]
    r_krak = [p[2] / R0_lb for p in kraken_pts]
    z_bas = [p[1] for p in basilisk_contour]
    r_bas = [p[2] for p in basilisk_contour]
    return (z_krak=z_krak, r_krak=r_krak, z_bas=z_bas, r_bas=r_bas)
end

# --- Internal helpers ---

function _resolve_position(at::Real, L::Real)
    # If at ∈ (0, 1] and L > 1, treat as fraction of domain
    if 0 < at <= 1.0 && L > 1.0
        return at * L
    end
    return Float64(at)
end

function _compute_norm(diff::AbstractArray, ref::AbstractArray, norm::Symbol)
    if norm == :L2
        ref_norm = sqrt(sum(ref .^ 2))
        ref_norm < eps() && return sqrt(sum(diff .^ 2))
        return sqrt(sum(diff .^ 2)) / ref_norm
    elseif norm == :Linf
        ref_max = maximum(abs.(ref))
        ref_max < eps() && return maximum(abs.(diff))
        return maximum(abs.(diff)) / ref_max
    elseif norm == :L1
        ref_sum = sum(abs.(ref))
        ref_sum < eps() && return sum(abs.(diff))
        return sum(abs.(diff)) / ref_sum
    else
        throw(ArgumentError("Unknown norm :$norm. Use :L2, :Linf, or :L1"))
    end
end
