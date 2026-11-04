# I/O and diagnostics

I/O utilities write simulation state to ParaView-compatible VTK
files (rectilinear grid in 2D/3D, multi-block)
and to `.pvd` time-series collections. The `DiagnosticsLogger`
writes one CSV row per probe step (residuals, drag, Nusselt, …);
`setup_output_dir` centralises run-directory creation and
optional cleanup.


## Quick reference

| Symbol | Purpose |
|--------|---------|
| `write_vtk` | Write a 2D/3D VTK rectilinear-grid file |
| `create_pvd` | Open a ParaView .pvd time-series collection |
| `write_vtk_to_pvd` | Append a VTK snapshot to a .pvd collection |
| `setup_output_dir` | Create (or clean) a run output directory |
| `write_snapshot_2d!` | Write one 2D macroscopic snapshot (VTK) |
| `write_snapshot_3d!` | Write one 3D macroscopic snapshot (VTK) |
| `write_vtk_multiblock` | Write a multi-block VTK (used by the refinement code, which is under development and not validated) |
| `write_snapshot_refined_2d!` | Write a refined-domain snapshot (grid refinement is under development and not validated) |
| `DiagnosticsLogger` | CSV diagnostics log descriptor |
| `open_diagnostics` | Open a `DiagnosticsLogger` |
| `log_diagnostics!` | Append one row (step, residuals, etc.) |
| `close_diagnostics!` | Close the underlying CSV stream |

## Details

### `write_vtk`

**Source:** `src/io/vtk_writer.jl`

```julia
"""
    write_vtk(filename, Nx, Ny, dx, fields::Dict{String, <:AbstractMatrix})

Write 2D field data on a rectilinear grid to a VTK `.vtr` file.
"""
function write_vtk(filename::String, Nx::Int, Ny::Int, dx::Float64,
                   fields::Dict{String, <:AbstractMatrix})
    xs = collect(range(0.0, step=dx, length=Nx + 1))
    ys = collect(range(0.0, step=dx, length=Ny + 1))

    vtk_grid(filename, xs, ys) do vtk
        for (name, data) in fields
            vtk[name] = Array(data)
        end
    end
end
```


### `setup_output_dir`

**Source:** `src/io/vtk_writer.jl`

```julia
"""
    setup_output_dir(path::String) -> String

Create the output directory (and parents) if it does not exist.
Returns the absolute path.
"""
function setup_output_dir(path::String)
    mkpath(path)
    return abspath(path)
end
```


### `write_snapshot_2d!`

**Source:** `src/io/vtk_writer.jl`

```julia
"""
    write_snapshot_2d!(output_dir, step, Nx, Ny, dx, fields; pvd=nothing, time=0.0)

Write a 2D VTK snapshot with a zero-padded filename `snapshot_NNNNNNN`.
If `pvd` is provided, the snapshot is also registered in the PVD collection.
"""
function write_snapshot_2d!(output_dir::String, step::Int, Nx::Int, Ny::Int, dx,
                            fields::Dict{String, <:AbstractMatrix};
                            pvd=nothing, time::Float64=0.0)
    tag = lpad(step, 7, '0')
    filename = joinpath(output_dir, "snapshot_$tag")
    dx_f = Float64(dx)
    if pvd !== nothing
        write_vtk_to_pvd(pvd, filename, Nx, Ny, dx_f, fields, Float64(time))
    else
        write_vtk(filename, Nx, Ny, dx_f, fields)
    end
end
```


### `DiagnosticsLogger`

**Source:** `src/io/diagnostics.jl`

```julia
"""
    DiagnosticsLogger

Mutable struct holding an IO handle, file path, and column names for CSV diagnostics output.
"""
mutable struct DiagnosticsLogger
    io::IO
    filepath::String
    columns::Vector{String}
end
```


