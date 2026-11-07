using LinearAlgebra
using SparseArrays

if !isdefined(@__MODULE__, :linear_index)
    include(joinpath(@__DIR__, "poisson.jl"))
end

# Fraction-array convention:
#   face_frac_x[i,j], i in 1:N+1, j in 1:N, is the aperture fraction of the
#   vertical face at x=(i-1)/N.  Interior face i is shared by cells (i-1,j)
#   and (i,j); i=1 and i=N+1 are the left/right box faces.
#   face_frac_y[i,j], i in 1:N, j in 1:N+1, is the aperture fraction of the
#   horizontal face at y=(j-1)/N.  Interior face j is shared by cells (i,j-1)
#   and (i,j); j=1 and j=N+1 are the bottom/top box faces.
# Fully solid cells have vol_frac[i,j] == 0 and are kept as isolated identity
# rows with b=0 so the global N^2 indexing remains compatible with rung 1.

const EMBEDDED_POISSON_BCS = (:neumann, :dirichlet)

_zero_dirichlet_value(x, y) = 0.0

function _check_bc_symbol(bc::Symbol, name::AbstractString)
    bc in EMBEDDED_POISSON_BCS ||
        throw(ArgumentError("$name must be :neumann or :dirichlet"))
    return bc
end

function _check_fraction_value(value::Real, name::AbstractString, i::Integer, j::Integer)
    α = Float64(value)
    0.0 <= α <= 1.0 ||
        throw(ArgumentError("$name[$i,$j] must lie in [0, 1], got $α"))
    return α
end

function _check_fraction_arrays(N::Integer, face_frac_x, face_frac_y, vol_frac)
    size(face_frac_x) == (N + 1, N) ||
        throw(ArgumentError("face_frac_x must have size (N+1, N)"))
    size(face_frac_y) == (N, N + 1) ||
        throw(ArgumentError("face_frac_y must have size (N, N+1)"))
    size(vol_frac) == (N, N) ||
        throw(ArgumentError("vol_frac must have size (N, N)"))
    return nothing
end

_is_fluid(vol::Real) = Float64(vol) > 0.0

function _add_dirichlet_face!(diag::Vector{Float64}, b::Vector{Float64},
                              k::Integer, coeff::Float64, value::Float64)
    diag[k] += coeff
    b[k] += coeff * value
    return nothing
end

function _add_symmetric_face!(I::Vector{Int}, J::Vector{Int}, V::Vector{Float64},
                              diag::Vector{Float64}, k1::Integer, k2::Integer,
                              coeff::Float64)
    diag[k1] += coeff
    diag[k2] += coeff
    _push_entry!(I, J, V, k1, k2, -coeff)
    _push_entry!(I, J, V, k2, k1, -coeff)
    return nothing
end

function _add_outer_dirichlet_faces!(diag::Vector{Float64}, b::Vector{Float64},
                                     N::Integer, face_frac_x, face_frac_y, vol_frac,
                                     g::Function, invh2::Float64)
    for j in 1:N
        y = cell_center(j, N)

        if _is_fluid(vol_frac[1, j])
            α = _check_fraction_value(face_frac_x[1, j], "face_frac_x", 1, j)
            coeff = 2.0 * α * invh2
            _add_dirichlet_face!(diag, b, linear_index(1, j, N), coeff, Float64(g(0.0, y)))
        end

        if _is_fluid(vol_frac[N, j])
            α = _check_fraction_value(face_frac_x[N + 1, j], "face_frac_x", N + 1, j)
            coeff = 2.0 * α * invh2
            _add_dirichlet_face!(diag, b, linear_index(N, j, N), coeff, Float64(g(1.0, y)))
        end
    end

    for i in 1:N
        x = cell_center(i, N)

        if _is_fluid(vol_frac[i, 1])
            α = _check_fraction_value(face_frac_y[i, 1], "face_frac_y", i, 1)
            coeff = 2.0 * α * invh2
            _add_dirichlet_face!(diag, b, linear_index(i, 1, N), coeff, Float64(g(x, 0.0)))
        end

        if _is_fluid(vol_frac[i, N])
            α = _check_fraction_value(face_frac_y[i, N + 1], "face_frac_y", i, N + 1)
            coeff = 2.0 * α * invh2
            _add_dirichlet_face!(diag, b, linear_index(i, N, N), coeff, Float64(g(x, 1.0)))
        end
    end

    return nothing
end

function _clip_polygon_half_plane(poly::Vector{NTuple{2, Float64}},
                                  inside::Function, intersection::Function)
    isempty(poly) && return NTuple{2, Float64}[]

    clipped = NTuple{2, Float64}[]
    previous = poly[end]
    previous_inside = inside(previous)

    for current in poly
        current_inside = inside(current)
        if current_inside
            if !previous_inside
                push!(clipped, intersection(previous, current))
            end
            push!(clipped, current)
        elseif previous_inside
            push!(clipped, intersection(previous, current))
        end
        previous = current
        previous_inside = current_inside
    end

    return clipped
end

function _polygon_area(poly::Vector{NTuple{2, Float64}})
    length(poly) >= 3 || return 0.0

    area2 = 0.0
    previous = poly[end]
    for current in poly
        area2 += previous[1] * current[2] - current[1] * previous[2]
        previous = current
    end

    return 0.5 * abs(area2)
end

function _add_embedded_dirichlet_faces!(diag::Vector{Float64}, b::Vector{Float64},
                                        N::Integer, face_frac_x, face_frac_y, vol_frac,
                                        g::Function, invh2::Float64)
    for j in 1:N, i in 1:N
        vol = _check_fraction_value(vol_frac[i, j], "vol_frac", i, j)
        0.0 < vol < 1.0 || continue

        α_left = _check_fraction_value(face_frac_x[i, j], "face_frac_x", i, j)
        α_right = _check_fraction_value(face_frac_x[i + 1, j], "face_frac_x", i + 1, j)
        α_bottom = _check_fraction_value(face_frac_y[i, j], "face_frac_y", i, j)
        α_top = _check_fraction_value(face_frac_y[i, j + 1], "face_frac_y", i, j + 1)

        # In embedded-Dirichlet mode, the blocked part β=1-α of each cut-cell
        # Cartesian face is a Dirichlet wall.  The open part α still contributes
        # only through the symmetric fluid-fluid conductance assembled above.
        h = 1.0 / Float64(N)
        x, y = cell_coordinates(i, j, N)
        k = linear_index(i, j, N)
        face_data = (
            (1.0 - α_left, Float64(i - 1) * h, y),
            (1.0 - α_right, Float64(i) * h, y),
            (1.0 - α_bottom, x, Float64(j - 1) * h),
            (1.0 - α_top, x, Float64(j) * h),
        )

        for (β, x_face, y_face) in face_data
            β > 0.0 || continue
            coeff = 2.0 * β * invh2
            _add_dirichlet_face!(diag, b, k, coeff, Float64(g(x_face, y_face)))
        end
    end

    return nothing
end

function assemble_poisson_embedded(N::Integer, face_frac_x, face_frac_y, vol_frac,
                                   f::Function;
                                   outer_bc::Symbol=:neumann,
                                   embedded_bc::Symbol=:neumann,
                                   outer_dirichlet::Function=_zero_dirichlet_value,
                                   embedded_dirichlet::Function=_zero_dirichlet_value)
    N = _check_grid_size(N)
    _check_bc_symbol(outer_bc, "outer_bc")
    _check_bc_symbol(embedded_bc, "embedded_bc")
    _check_fraction_arrays(N, face_frac_x, face_frac_y, vol_frac)

    n = N * N
    h = 1.0 / Float64(N)
    invh2 = 1.0 / (h * h)

    I = Int[]
    J = Int[]
    V = Float64[]
    sizehint!(I, 6n)
    sizehint!(J, 6n)
    sizehint!(V, 6n)

    diag = zeros(Float64, n)
    b = zeros(Float64, n)

    for j in 1:N, i in 1:N
        k = linear_index(i, j, N)
        vol = _check_fraction_value(vol_frac[i, j], "vol_frac", i, j)
        if _is_fluid(vol)
            x, y = cell_coordinates(i, j, N)
            b[k] = vol * Float64(f(x, y))
        else
            diag[k] = 1.0
        end
    end

    for j in 1:N, i in 2:N
        left = linear_index(i - 1, j, N)
        right = linear_index(i, j, N)
        if _is_fluid(vol_frac[i - 1, j]) && _is_fluid(vol_frac[i, j])
            α = _check_fraction_value(face_frac_x[i, j], "face_frac_x", i, j)
            α > 0.0 && _add_symmetric_face!(I, J, V, diag, left, right, α * invh2)
        end
    end

    for j in 2:N, i in 1:N
        bottom = linear_index(i, j - 1, N)
        top = linear_index(i, j, N)
        if _is_fluid(vol_frac[i, j - 1]) && _is_fluid(vol_frac[i, j])
            α = _check_fraction_value(face_frac_y[i, j], "face_frac_y", i, j)
            α > 0.0 && _add_symmetric_face!(I, J, V, diag, bottom, top, α * invh2)
        end
    end

    if outer_bc == :dirichlet
        _add_outer_dirichlet_faces!(
            diag, b, N, face_frac_x, face_frac_y, vol_frac, outer_dirichlet, invh2,
        )
    end

    if embedded_bc == :dirichlet
        _add_embedded_dirichlet_faces!(
            diag, b, N, face_frac_x, face_frac_y, vol_frac, embedded_dirichlet, invh2,
        )
    end

    for k in 1:n
        _push_entry!(I, J, V, k, k, diag[k])
    end

    return sparse(I, J, V, n, n), b
end

"""
    solve_poisson_embedded(N, face_frac_x, face_frac_y, vol_frac, f;
                           outer_bc=:neumann, embedded_bc=:neumann,
                           outer_dirichlet=g, embedded_dirichlet=g) -> Matrix{Float64}

Solve the cut-cell (embedded-boundary) Poisson problem `-∇²u = f` on the unit
square: assemble via `assemble_poisson_embedded`, then solve through
[`solve_poisson`](@ref) (factorize-once CPU CHOLMOD seam).

Aperture-fraction convention (see file header): `face_frac_x :: (N+1, N)` and
`face_frac_y :: (N, N+1)` are OPEN fractions of the x-/y-normal faces (shared
interior faces, plus the box faces at indices 1 and N+1); `vol_frac :: (N, N)` is
the open volume fraction. Fully solid cells (`vol_frac == 0`) are kept as
isolated identity rows with `b = 0`, so the global `N²` indexing stays compatible
with the regular-grid solver. The fluid-fluid conductance of an interior face is
the symmetric `α/h²` pair; with `embedded_bc=:dirichlet` the BLOCKED part
`β = 1-α` of each cut-cell face becomes a Dirichlet wall contributing `2β/h²`
(half-spacing) to the diagonal and `2β/h² · g` to the RHS.

With the default all-Neumann BCs the assembled operator is singular: do NOT call
this directly — assemble, pin a fluid DOF (`first_fluid_dof` +
`pin_reference_dof`), and call `solve_poisson`, as the MMS tests do. Direct calls
are valid whenever a Dirichlet face makes the operator non-singular.

Receipt: `test/analytical/poisson_embedded_mms.jl` (all-ones fractions reproduce
the regular solver; tilted half-plane geometry `tilted_half_plane_fractions`
converges ~2nd order in the fluid L2 norm). Registered in `src/KrakeNN.jl`
(exported by `using KrakeNN`; still standalone-include-able via the guards).
"""
function solve_poisson_embedded(N::Integer, face_frac_x, face_frac_y, vol_frac,
                                f::Function; kwargs...)
    A, b = assemble_poisson_embedded(N, face_frac_x, face_frac_y, vol_frac, f; kwargs...)
    return solve_poisson(A, b, N)
end

function first_fluid_dof(vol_frac, N::Integer)
    N = _check_grid_size(N)
    size(vol_frac) == (N, N) || throw(ArgumentError("vol_frac must have size (N, N)"))
    for j in 1:N, i in 1:N
        _is_fluid(vol_frac[i, j]) && return linear_index(i, j, N)
    end
    throw(ArgumentError("vol_frac contains no fluid cells"))
end

function fluid_l2_error(u::AbstractMatrix{<:Real}, u_exact::Function,
                        N::Integer, vol_frac)
    N = _check_grid_size(N)
    size(u) == (N, N) || throw(ArgumentError("u must have size (N, N)"))
    size(vol_frac) == (N, N) || throw(ArgumentError("vol_frac must have size (N, N)"))

    h = 1.0 / Float64(N)
    err2 = 0.0
    for j in 1:N, i in 1:N
        vol = _check_fraction_value(vol_frac[i, j], "vol_frac", i, j)
        vol > 0.0 || continue
        x, y = cell_coordinates(i, j, N)
        diff = Float64(u[i, j]) - Float64(u_exact(x, y))
        err2 += vol * diff * diff
    end

    return sqrt(h * h * err2)
end

function fluid_row_sum_max(A::SparseMatrixCSC{Float64, Int}, N::Integer, vol_frac)
    N = _check_grid_size(N)
    size(A) == (N * N, N * N) || throw(ArgumentError("A size must match N^2"))
    size(vol_frac) == (N, N) || throw(ArgumentError("vol_frac must have size (N, N)"))

    row_sums = vec(sum(A; dims=2))
    max_row_sum = 0.0
    for j in 1:N, i in 1:N
        if _is_fluid(vol_frac[i, j])
            max_row_sum = max(max_row_sum, abs(row_sums[linear_index(i, j, N)]))
        end
    end

    return max_row_sum
end

function fluid_constant_deviation(u::AbstractMatrix{<:Real}, value::Real,
                                  N::Integer, vol_frac)
    N = _check_grid_size(N)
    size(u) == (N, N) || throw(ArgumentError("u must have size (N, N)"))
    size(vol_frac) == (N, N) || throw(ArgumentError("vol_frac must have size (N, N)"))

    max_deviation = 0.0
    reference = Float64(value)
    for j in 1:N, i in 1:N
        if _is_fluid(vol_frac[i, j])
            max_deviation = max(max_deviation, abs(Float64(u[i, j]) - reference))
        end
    end

    return max_deviation
end

function _segment_fraction_in_half_plane(p0::NTuple{2, Float64},
                                         p1::NTuple{2, Float64},
                                         normal::NTuple{2, Float64},
                                         point::NTuple{2, Float64})
    s0 = normal[1] * (p0[1] - point[1]) + normal[2] * (p0[2] - point[2])
    s1 = normal[1] * (p1[1] - point[1]) + normal[2] * (p1[2] - point[2])

    if s0 >= 0.0 && s1 >= 0.0
        return 1.0
    elseif s0 <= 0.0 && s1 <= 0.0
        return 0.0
    end

    t = s0 / (s0 - s1)
    return s0 >= 0.0 ? t : 1.0 - t
end

function _cell_volume_fraction_half_plane(i::Integer, j::Integer, N::Integer,
                                          normal::NTuple{2, Float64},
                                          point::NTuple{2, Float64})
    h = 1.0 / Float64(N)
    x0 = Float64(i - 1) * h
    x1 = Float64(i) * h
    y0 = Float64(j - 1) * h
    y1 = Float64(j) * h
    square = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    inside(p) = normal[1] * (p[1] - point[1]) + normal[2] * (p[2] - point[2]) >= 0.0
    function intersect_edge(p0, p1)
        s0 = normal[1] * (p0[1] - point[1]) + normal[2] * (p0[2] - point[2])
        s1 = normal[1] * (p1[1] - point[1]) + normal[2] * (p1[2] - point[2])
        t = s0 / (s0 - s1)
        return (p0[1] + t * (p1[1] - p0[1]), p0[2] + t * (p1[2] - p0[2]))
    end

    return _polygon_area(_clip_polygon_half_plane(square, inside, intersect_edge)) / (h * h)
end

function tilted_half_plane_fractions(N::Integer;
                                     normal::NTuple{2, Float64}=(cos(pi / 6), sin(pi / 6)),
                                     point::NTuple{2, Float64}=(0.43, 0.52))
    N = _check_grid_size(N)
    norm_normal = hypot(normal[1], normal[2])
    norm_normal > 0.0 || throw(ArgumentError("normal must be nonzero"))
    n̂ = (normal[1] / norm_normal, normal[2] / norm_normal)

    h = 1.0 / Float64(N)
    face_frac_x = Matrix{Float64}(undef, N + 1, N)
    face_frac_y = Matrix{Float64}(undef, N, N + 1)
    vol_frac = Matrix{Float64}(undef, N, N)

    for j in 1:N, i in 1:(N + 1)
        x = Float64(i - 1) * h
        y0 = Float64(j - 1) * h
        y1 = Float64(j) * h
        face_frac_x[i, j] = _segment_fraction_in_half_plane((x, y0), (x, y1), n̂, point)
    end

    for j in 1:(N + 1), i in 1:N
        x0 = Float64(i - 1) * h
        x1 = Float64(i) * h
        y = Float64(j - 1) * h
        face_frac_y[i, j] = _segment_fraction_in_half_plane((x0, y), (x1, y), n̂, point)
    end

    for j in 1:N, i in 1:N
        vol_frac[i, j] = _cell_volume_fraction_half_plane(i, j, N, n̂, point)
    end

    return face_frac_x, face_frac_y, vol_frac
end
