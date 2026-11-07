# ===========================================================================
# 3D curvilinear mesh infrastructure for SLBM on D3Q19.
#
# CurvilinearMesh3D: Nξ × Nη × Nζ logically structured grid with a mapping
# (ξ, η, ζ) → (X, Y, Z). The 3×3 Jacobian is precomputed via ForwardDiff.
# ===========================================================================

"""
    CurvilinearMesh3D{T, AT}

Logically structured `Nξ × Nη × Nζ` mesh in computational space with a
user-supplied mapping to physical 3D space. All 9 metric derivatives and
the Jacobian determinant are precomputed.

# Fields
- `Nξ, Nη, Nζ`: logical grid extents
- `periodic_ξ, periodic_η, periodic_ζ`: wrap-around flags
- `X, Y, Z`: physical node coordinates `[Nξ, Nη, Nζ]`
- `dXdξ, dXdη, dXdζ, dYdξ, ..., dZdζ`: 9 metric derivatives
- `J`: Jacobian determinant
- `dx_ref`: physical distance per lattice unit
"""
struct CurvilinearMesh3D{T<:AbstractFloat, AT<:AbstractArray{T, 3}}
    type::Symbol
    Nξ::Int
    Nη::Int
    Nζ::Int
    periodic_ξ::Bool
    periodic_η::Bool
    periodic_ζ::Bool
    X::AT
    Y::AT
    Z::AT
    dXdξ::AT; dXdη::AT; dXdζ::AT
    dYdξ::AT; dYdη::AT; dYdζ::AT
    dZdξ::AT; dZdη::AT; dZdζ::AT
    J::AT
    dx_ref::T
end

"""
    compute_metric_3d(mapping, ξ, η, ζ) -> (dX/dξ, dX/dη, dX/dζ, ..., J)

Forward-mode AD evaluation of the 3×3 Jacobian at a point.
"""
function compute_metric_3d(mapping, ξ::T, η::T, ζ::T) where {T<:AbstractFloat}
    f = p -> begin
        xyz = mapping(p[1], p[2], p[3])
        return [xyz[1], xyz[2], xyz[3]]
    end
    Jmat = ForwardDiff.jacobian(f, T[ξ, η, ζ])
    dXdξ = Jmat[1,1]; dXdη = Jmat[1,2]; dXdζ = Jmat[1,3]
    dYdξ = Jmat[2,1]; dYdη = Jmat[2,2]; dYdζ = Jmat[2,3]
    dZdξ = Jmat[3,1]; dZdη = Jmat[3,2]; dZdζ = Jmat[3,3]
    # detJ for 3×3 matrix
    detJ = dXdξ*(dYdη*dZdζ - dYdζ*dZdη) -
           dXdη*(dYdξ*dZdζ - dYdζ*dZdξ) +
           dXdζ*(dYdξ*dZdη - dYdη*dZdξ)
    return dXdξ, dXdη, dXdζ, dYdξ, dYdη, dYdζ, dZdξ, dZdη, dZdζ, detJ
end

"""
    build_mesh_3d(mapping; Nξ, Nη, Nζ, ...) -> CurvilinearMesh3D

Sample the 3D mapping on the logical grid and compute all 9 metric
derivatives + Jacobian determinant via ForwardDiff.
"""
function build_mesh_3d(mapping;
                       Nξ::Int, Nη::Int, Nζ::Int,
                       periodic_ξ::Bool=false, periodic_η::Bool=false,
                       periodic_ζ::Bool=false,
                       type::Symbol=:custom,
                       dx_ref::Union{Real, Nothing}=nothing,
                       FT::Type{<:AbstractFloat}=Float64)
    X = zeros(FT, Nξ, Nη, Nζ); Y = similar(X); Z = similar(X)
    dXdξ = similar(X); dXdη = similar(X); dXdζ = similar(X)
    dYdξ = similar(X); dYdη = similar(X); dYdζ = similar(X)
    dZdξ = similar(X); dZdη = similar(X); dZdζ = similar(X)
    J = similar(X)

    denom_ξ = FT(periodic_ξ ? Nξ : Nξ - 1)
    denom_η = FT(periodic_η ? Nη : Nη - 1)
    denom_ζ = FT(periodic_ζ ? Nζ : Nζ - 1)

    for k in 1:Nζ, j in 1:Nη, i in 1:Nξ
        ξ = FT(i - 1) / denom_ξ
        η = FT(j - 1) / denom_η
        ζ = FT(k - 1) / denom_ζ
        xyz = mapping(ξ, η, ζ)
        X[i,j,k] = xyz[1]; Y[i,j,k] = xyz[2]; Z[i,j,k] = xyz[3]
        dX_ξ, dX_η, dX_ζ, dY_ξ, dY_η, dY_ζ, dZ_ξ, dZ_η, dZ_ζ, detJ =
            compute_metric_3d(mapping, ξ, η, ζ)
        dXdξ[i,j,k] = dX_ξ; dXdη[i,j,k] = dX_η; dXdζ[i,j,k] = dX_ζ
        dYdξ[i,j,k] = dY_ξ; dYdη[i,j,k] = dY_η; dYdζ[i,j,k] = dY_ζ
        dZdξ[i,j,k] = dZ_ξ; dZdη[i,j,k] = dZ_η; dZdζ[i,j,k] = dZ_ζ
        J[i,j,k] = detJ
    end

    dxr = dx_ref === nothing ?
          _default_dx_ref_3d(dXdξ, dXdη, dXdζ, dYdξ, dYdη, dYdζ,
                              dZdξ, dZdη, dZdζ, Nξ, Nη, Nζ,
                              periodic_ξ, periodic_η, periodic_ζ, FT) :
          FT(dx_ref)

    mesh = CurvilinearMesh3D{FT, Array{FT,3}}(type, Nξ, Nη, Nζ,
        periodic_ξ, periodic_η, periodic_ζ, X, Y, Z,
        dXdξ, dXdη, dXdζ, dYdξ, dYdη, dYdζ, dZdξ, dZdη, dZdζ, J, dxr)
    validate_mesh_3d(mesh)
    return mesh
end

function _default_dx_ref_3d(dXdξ, dXdη, dXdζ, dYdξ, dYdη, dYdζ,
                              dZdξ, dZdη, dZdζ, Nξ, Nη, Nζ,
                              periodic_ξ, periodic_η, periodic_ζ, ::Type{T}) where {T}
    denom_ξ = T(periodic_ξ ? Nξ : Nξ - 1)
    denom_η = T(periodic_η ? Nη : Nη - 1)
    denom_ζ = T(periodic_ζ ? Nζ : Nζ - 1)
    Δξ = one(T) / denom_ξ
    Δη = one(T) / denom_η
    Δζ = one(T) / denom_ζ
    min_edge = T(Inf)
    @inbounds for k in 1:Nζ, j in 1:Nη, i in 1:Nξ
        lξ = sqrt(dXdξ[i,j,k]^2 + dYdξ[i,j,k]^2 + dZdξ[i,j,k]^2) * Δξ
        lη = sqrt(dXdη[i,j,k]^2 + dYdη[i,j,k]^2 + dZdη[i,j,k]^2) * Δη
        lζ = sqrt(dXdζ[i,j,k]^2 + dYdζ[i,j,k]^2 + dZdζ[i,j,k]^2) * Δζ
        min_edge = min(min_edge, lξ, lη, lζ)
    end
    return min_edge
end

"""
    validate_mesh_3d(mesh::CurvilinearMesh3D{T}) where {T}

Public function in the curvilinear mesh and SLBM API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.validate_mesh_3d)
```
"""
function validate_mesh_3d(mesh::CurvilinearMesh3D{T}) where {T}
    Jmin = minimum(mesh.J); Jmax = maximum(mesh.J)
    Jabs_min = min(abs(Jmin), abs(Jmax))
    if Jabs_min == zero(T) || (Jmin < zero(T) && Jmax > zero(T))
        error("CurvilinearMesh3D: degenerate Jacobian (sign change or zero): " *
              "Jmin=$Jmin, Jmax=$Jmax")
    end
    return nothing
end

# ===========================================================================
# Stretched box 3D mesh generator.
# ===========================================================================

"""
    stretched_box_mesh_3d(; x_min, x_max, y_min, y_max, z_min, z_max,
                           Nx, Ny, Nz,
                           x_stretch=0, y_stretch=0, z_stretch=0,
                           x_stretch_dir=:none, y_stretch_dir=:none, z_stretch_dir=:none,
                           FT=Float64) -> CurvilinearMesh3D

Rectangular 3D mesh with optional tanh stretching per axis. Same
stretching directions as the 2D version.
"""
function stretched_box_mesh_3d(; x_min::Real, x_max::Real,
                                 y_min::Real, y_max::Real,
                                 z_min::Real, z_max::Real,
                                 Nx::Int, Ny::Int, Nz::Int,
                                 x_stretch::Real=0.0, y_stretch::Real=0.0, z_stretch::Real=0.0,
                                 x_stretch_dir::Symbol=:none,
                                 y_stretch_dir::Symbol=:none,
                                 z_stretch_dir::Symbol=:none,
                                 FT::Type{<:AbstractFloat}=Float64)
    xminT, xmaxT = FT(x_min), FT(x_max)
    yminT, ymaxT = FT(y_min), FT(y_max)
    zminT, zmaxT = FT(z_min), FT(z_max)
    sxT, syT, szT = FT(x_stretch), FT(y_stretch), FT(z_stretch)

    function mapping(ξ, η, ζ)
        tx = _stretch(ξ, sxT, x_stretch_dir)
        ty = _stretch(η, syT, y_stretch_dir)
        tz = _stretch(ζ, szT, z_stretch_dir)
        return (xminT + (xmaxT - xminT) * tx,
                yminT + (ymaxT - yminT) * ty,
                zminT + (zmaxT - zminT) * tz)
    end

    return build_mesh_3d(mapping; Nξ=Nx, Nη=Ny, Nζ=Nz,
                         periodic_ξ=false, periodic_η=false, periodic_ζ=false,
                         type=:stretched_box_3d, FT=FT)
end

"""
    cartesian_mesh_3d(; x_min, x_max, y_min, y_max, z_min, z_max,
                        Nx, Ny, Nz, FT=Float64)

Convenience: uniform Cartesian 3D mesh.
"""
function cartesian_mesh_3d(; x_min::Real, x_max::Real,
                             y_min::Real, y_max::Real,
                             z_min::Real, z_max::Real,
                             Nx::Int, Ny::Int, Nz::Int,
                             FT::Type{<:AbstractFloat}=Float64)
    return stretched_box_mesh_3d(; x_min=x_min, x_max=x_max,
                                   y_min=y_min, y_max=y_max,
                                   z_min=z_min, z_max=z_max,
                                   Nx=Nx, Ny=Ny, Nz=Nz, FT=FT)
end
