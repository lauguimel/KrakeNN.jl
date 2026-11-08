# Post-processing

Post-processing helpers operate on finished runs: line and point
sampling of macroscopic fields, L1/L2/L∞ error against an
analytical reference, and interface-comparison utilities for
Basilisk cross-validation. They are pure Julia functions — no
state, no I/O side effects — so they plug directly into scripts and
test suites.


## Quick reference

| Symbol | Purpose |
|--------|---------|
| `extract_line` | Sample a field along an axis-aligned line |
| `field_error` | L1/L2/L∞ error against a reference field |
| `probe` | Interpolate a field at one point |
| `domain_stats` | min/max/mean/std of a field |
| `load_basilisk_interfaces` | Load Basilisk interface dumps |
| `load_basilisk_interface_contour` | Load a single Basilisk contour |
| `find_basilisk_snapshot` | Locate a Basilisk snapshot by time |
| `compare_interfaces` | Compare two interface curves (Hausdorff, RMS) |

## Details

### `extract_line`

**Source:** `src/postprocess.jl`

```julia
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
```


### `probe`

**Source:** `src/postprocess.jl`

```julia
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
```


### `field_error`

**Source:** `src/postprocess.jl`

```julia
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
```


### `domain_stats`

**Source:** `src/postprocess.jl`

```julia
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
```


### `compare_interfaces`

**Source:** `src/postprocess.jl`

```julia
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
```


