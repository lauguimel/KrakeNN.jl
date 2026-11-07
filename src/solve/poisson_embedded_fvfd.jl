if !isdefined(@__MODULE__, :assemble_poisson_embedded)
    include(joinpath(@__DIR__, "poisson_embedded.jl"))
end

const _FVFD_POISSON_FRACTION_TOL = sqrt(eps(Float64))

_fvfd_poisson_zero_dirichlet(x, y) = 0.0

function _fvfd_poisson_check_size(name::Symbol, values, expected::Tuple{Int,Int})
    size(values) == expected ||
        throw(DimensionMismatch("$(name) must have size $(expected), got $(size(values))"))
    return nothing
end

function _fvfd_poisson_fraction(value, name::Symbol, i::Integer, j::Integer;
                                tol::Float64=_FVFD_POISSON_FRACTION_TOL)
    α = Float64(value)
    if α < -tol || α > 1.0 + tol
        throw(ArgumentError("$(name)[$i,$j] must lie in [0, 1], got $α"))
    end
    return min(1.0, max(0.0, α))
end

function _fvfd_poisson_check_face_match(face_a, face_b, name_a::Symbol, name_b::Symbol,
                                        ia::Integer, ja::Integer, ib::Integer, jb::Integer;
                                        tol::Float64=_FVFD_POISSON_FRACTION_TOL)
    α = Float64(face_a)
    β = Float64(face_b)
    if abs(α - β) > tol
        throw(ArgumentError(
            "$(name_a)[$ia,$ja] must match $(name_b)[$ib,$jb] on the shared face; " *
            "got $α and $β (diff $(abs(α - β)), tol $tol)",
        ))
    end
    return nothing
end

"""
    fractions_from_fvfd(eb) -> (vol_frac, face_frac_x, face_frac_y)

Convert a duck-typed FVFD embedded boundary object with `cell_fraction`,
`west_fraction`, `east_fraction`, `south_fraction`, and `north_fraction`
fields into the open-fraction arrays consumed by `assemble_poisson_embedded`.
"""
function fractions_from_fvfd(eb)
    Nx, Ny = size(eb.cell_fraction)
    Nx > 0 && Ny > 0 ||
        throw(ArgumentError("cell_fraction must have positive dimensions, got ($Nx, $Ny)"))
    _fvfd_poisson_check_size(:west_fraction, eb.west_fraction, (Nx, Ny))
    _fvfd_poisson_check_size(:east_fraction, eb.east_fraction, (Nx, Ny))
    _fvfd_poisson_check_size(:south_fraction, eb.south_fraction, (Nx, Ny))
    _fvfd_poisson_check_size(:north_fraction, eb.north_fraction, (Nx, Ny))

    vol_frac = Matrix{Float64}(undef, Nx, Ny)
    face_frac_x = Matrix{Float64}(undef, Nx + 1, Ny)
    face_frac_y = Matrix{Float64}(undef, Nx, Ny + 1)

    @inbounds for j in 1:Ny, i in 1:Nx
        vol_frac[i, j] = _fvfd_poisson_fraction(eb.cell_fraction[i, j], :cell_fraction, i, j)
    end

    @inbounds for j in 1:Ny
        face_frac_x[1, j] = _fvfd_poisson_fraction(eb.west_fraction[1, j], :west_fraction, 1, j)
        for i in 1:Nx
            face_frac_x[i + 1, j] =
                _fvfd_poisson_fraction(eb.east_fraction[i, j], :east_fraction, i, j)
        end
        for i in 1:(Nx - 1)
            _fvfd_poisson_check_face_match(
                eb.east_fraction[i, j], eb.west_fraction[i + 1, j],
                :east_fraction, :west_fraction, i, j, i + 1, j,
            )
        end
    end

    @inbounds for i in 1:Nx
        face_frac_y[i, 1] =
            _fvfd_poisson_fraction(eb.south_fraction[i, 1], :south_fraction, i, 1)
        for j in 1:Ny
            face_frac_y[i, j + 1] =
                _fvfd_poisson_fraction(eb.north_fraction[i, j], :north_fraction, i, j)
        end
        for j in 1:(Ny - 1)
            _fvfd_poisson_check_face_match(
                eb.north_fraction[i, j], eb.south_fraction[i, j + 1],
                :north_fraction, :south_fraction, i, j, i, j + 1,
            )
        end
    end

    return vol_frac, face_frac_x, face_frac_y
end

"""
    assemble_poisson_embedded_from_fvfd(eb, f; outer_bc=:neumann,
                                        embedded_bc=:neumann,
                                        outer_dirichlet=g, embedded_dirichlet=g)
        -> (A, b)

Bridge from a duck-typed FVFD embedded-boundary object `eb` (fields
`cell_fraction`, `west_fraction`, `east_fraction`, `south_fraction`,
`north_fraction` — the per-cell convention of the FVFD geometry layer) to the
embedded Poisson assembly: converts to the `(vol_frac, face_frac_x, face_frac_y)`
aperture arrays via [`fractions_from_fvfd`](@ref) — clamping fractions into
`[0, 1]` within `sqrt(eps)` tolerance and validating that shared interior faces
agree (`east_fraction[i,j] == west_fraction[i+1,j]`, etc.) — then calls
`assemble_poisson_embedded`. Requires a square grid (`Nx == Ny`).

Same singularity caveat as [`solve_poisson_embedded`](@ref): with all-Neumann BCs
pin a fluid DOF before solving. Receipt:
`test/analytical/poisson_embedded_fvfd_mms.jl` (FVFD-built fractions reproduce
the hand-built tilted half-plane assembly and converge ~2nd order). Registered
in `src/KrakeNN.jl` (exported by `using KrakeNN`).
"""
function assemble_poisson_embedded_from_fvfd(eb, f;
                                             outer_bc::Symbol=:neumann,
                                             embedded_bc::Symbol=:neumann,
                                             outer_dirichlet::Function=_fvfd_poisson_zero_dirichlet,
                                             embedded_dirichlet::Function=_fvfd_poisson_zero_dirichlet)
    vol_frac, face_frac_x, face_frac_y = fractions_from_fvfd(eb)
    Nx, Ny = size(vol_frac)
    Nx == Ny ||
        throw(ArgumentError("assemble_poisson_embedded_from_fvfd requires a square grid, got ($Nx, $Ny)"))
    return assemble_poisson_embedded(
        Nx, face_frac_x, face_frac_y, vol_frac, f;
        outer_bc, embedded_bc, outer_dirichlet, embedded_dirichlet,
    )
end
