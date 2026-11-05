# Modular axis-aligned step/channel geometry for LI-BB validation cases.
#
# This mirrors the SLBM pattern: build one immutable geometry/spec object once,
# transfer it to the backend, then pass the object to drivers/kernels. Concrete
# cases such as a 4:1 contraction or a backward-facing step are constructors of
# the same geometry type, not separate solver pipelines.

"""
    StepChannelGeometry2D

Precomputed axis-aligned step/channel geometry for Cartesian LI-BB runs.

Fields carry both the solid/cut-link geometry (`is_solid`, `q_wall`) and the
masked inlet/outlet faces needed by step cases with partial openings.
"""
struct StepChannelGeometry2D{T<:AbstractFloat,A3<:AbstractArray{T,3},
                             A2<:AbstractArray{Bool,2},V<:AbstractArray{Bool,1}}
    name::Symbol
    Nx::Int
    Ny::Int
    i_step::Int
    inlet_open::UnitRange{Int}
    outlet_open::UnitRange{Int}
    H_ref::Int
    H_in::Int
    H_out::Int
    q_wall::A3
    is_solid::A2
    west_hydro_mask::V
    east_hydro_mask::V
    west_conformation_mask::V
    east_conformation_mask::V
end

@inline _interior_open_range(open::UnitRange{Int}, Ny::Int) =
    max(first(open) + 1, 2):min(last(open) - 1, Ny - 1)

function _mask_from_range(Ny::Int, open::UnitRange{Int}; interior_only::Bool)
    mask = fill(false, Ny)
    rows = interior_only ? _interior_open_range(open, Ny) : open
    for j in rows
        1 <= j <= Ny && (mask[j] = true)
    end
    return mask
end

function _q_wall_from_solid_mask_2d(is_solid::AbstractMatrix{Bool};
                                    include_y_domain_walls::Bool=true,
                                    FT::Type{<:AbstractFloat}=Float64)
    Nx, Ny = size(is_solid)
    q_wall = zeros(FT, Nx, Ny, 9)
    cxs = velocities_x(D2Q9())
    cys = velocities_y(D2Q9())

    @inbounds for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        for q in 2:9
            ni = i + cxs[q]
            nj = j + cys[q]
            inside = 1 <= ni <= Nx && 1 <= nj <= Ny
            if !inside
                if include_y_domain_walls && cys[q] != 0 && (1 <= ni <= Nx)
                    q_wall[i, j, q] = FT(0.5)
                end
            elseif is_solid[ni, nj]
                q_wall[i, j, q] = FT(0.5)
            end
        end
    end
    return q_wall
end

function step_channel_geometry_2d(name::Symbol;
                                  is_solid::AbstractMatrix{Bool},
                                  i_step::Int,
                                  inlet_open::UnitRange{Int},
                                  outlet_open::UnitRange{Int},
                                  H_ref::Int,
                                  H_in::Int=length(inlet_open),
                                  H_out::Int=length(outlet_open),
                                  include_y_domain_walls::Bool=true,
                                  FT::Type{<:AbstractFloat}=Float64)
    Nx, Ny = size(is_solid)
    1 <= i_step <= Nx || error("i_step out of [1, Nx]")
    1 <= first(inlet_open) <= last(inlet_open) <= Ny ||
        error("inlet_open must be inside [1, Ny]")
    1 <= first(outlet_open) <= last(outlet_open) <= Ny ||
        error("outlet_open must be inside [1, Ny]")

    q_wall = _q_wall_from_solid_mask_2d(is_solid;
        include_y_domain_walls=include_y_domain_walls, FT=FT)
    west_hydro_mask = _mask_from_range(Ny, inlet_open; interior_only=true)
    east_hydro_mask = _mask_from_range(Ny, outlet_open; interior_only=true)
    west_conformation_mask = _mask_from_range(Ny, inlet_open; interior_only=false)
    east_conformation_mask = _mask_from_range(Ny, outlet_open; interior_only=false)

    return StepChannelGeometry2D{FT,Array{FT,3},Matrix{Bool},Vector{Bool}}(
        name, Nx, Ny, i_step, inlet_open, outlet_open, H_ref, H_in, H_out,
        q_wall, Matrix{Bool}(is_solid),
        west_hydro_mask, east_hydro_mask,
        west_conformation_mask, east_conformation_mask)
end

"""
    contraction_step_geometry_2d(; H_out=20, β_c=4, L_up=20, L_down=50, FT=Float64)

Build a symmetric planar sudden contraction geometry. `H_out` is the
downstream opening height and `β_c = H_in/H_out`.
"""
function contraction_step_geometry_2d(; H_out::Int=20, β_c::Int=4,
                                      L_up::Int=20, L_down::Int=50,
                                      FT::Type{<:AbstractFloat}=Float64)
    Nx = (L_up + L_down) * H_out
    Ny = β_c * H_out
    i_step = L_up * H_out + 1
    j_low = (Ny - H_out) ÷ 2 + 1
    j_high = j_low + H_out - 1
    isodd(Ny - H_out) && @warn "(Ny − H_out) odd → 1-cell asymmetric outlet" Ny H_out j_low j_high

    is_solid = zeros(Bool, Nx, Ny)
    @inbounds for j in 1:Ny, i in i_step:Nx
        if j < j_low || j > j_high
            is_solid[i, j] = true
        end
    end

    return step_channel_geometry_2d(:contraction;
        is_solid, i_step, inlet_open=1:Ny, outlet_open=j_low:j_high,
        H_ref=H_out, H_in=Ny, H_out=H_out, FT=FT)
end

"""
    backward_facing_step_geometry_2d(; H_in=16, expansion_ratio=2,
                                      L_up=4, L_down=12, FT=Float64)

Build a top-aligned backward-facing step: the top wall is horizontal, the
upstream inlet occupies the upper `H_in` rows, and the bottom block ends at
`i_step`. No dedicated BFS solver is needed; this is another
`StepChannelGeometry2D` spec.
"""
function backward_facing_step_geometry_2d(; H_in::Int=16, expansion_ratio::Int=2,
                                          L_up::Int=4, L_down::Int=12,
                                          FT::Type{<:AbstractFloat}=Float64)
    H_out = expansion_ratio * H_in
    Nx = (L_up + L_down) * H_in
    Ny = H_out
    i_step = L_up * H_in + 1
    j_low = Ny - H_in + 1
    inlet_open = j_low:Ny
    outlet_open = 1:Ny

    is_solid = zeros(Bool, Nx, Ny)
    @inbounds for j in 1:(j_low - 1), i in 1:(i_step - 1)
        is_solid[i, j] = true
    end

    return step_channel_geometry_2d(:backward_facing_step;
        is_solid, i_step, inlet_open, outlet_open,
        H_ref=H_in, H_in=H_in, H_out=H_out, FT=FT)
end

"""
    square_obstacle_channel_geometry_2d(; H=32, side=8, L_up=4, L_down=8, FT=Float64)

Build a straight channel with a centered axis-aligned square obstacle. This is
the intermediate Cartesian-obstacle case between step geometries and curved
cylinders: same `StepChannelGeometry2D` driver, no curved cut-links.
"""
function square_obstacle_channel_geometry_2d(; H::Int=32, side::Int=8,
                                             L_up::Int=4, L_down::Int=8,
                                             FT::Type{<:AbstractFloat}=Float64)
    H >= side + 4 || error("H must leave at least two fluid rows on each side")
    side >= 2 || error("side must be at least 2")
    Nx = (L_up + L_down) * side + side
    Ny = H
    i_step = L_up * side + 1
    i_last = i_step + side - 1
    j_low = (Ny - side) ÷ 2 + 1
    j_high = j_low + side - 1
    isodd(Ny - side) && @warn "(Ny − side) odd → 1-cell asymmetric square obstacle" Ny side j_low j_high

    is_solid = zeros(Bool, Nx, Ny)
    @inbounds for j in j_low:j_high, i in i_step:i_last
        is_solid[i, j] = true
    end

    return step_channel_geometry_2d(:square_obstacle;
        is_solid, i_step, inlet_open=1:Ny, outlet_open=1:Ny,
        H_ref=side, H_in=Ny, H_out=Ny, FT=FT)
end

function transfer_step_geometry_2d(geom::StepChannelGeometry2D{T}, backend) where {T}
    q_wall = KernelAbstractions.allocate(backend, T, geom.Nx, geom.Ny, 9)
    is_solid = KernelAbstractions.allocate(backend, Bool, geom.Nx, geom.Ny)
    west_hydro_mask = KernelAbstractions.allocate(backend, Bool, geom.Ny)
    east_hydro_mask = KernelAbstractions.allocate(backend, Bool, geom.Ny)
    west_conformation_mask = KernelAbstractions.allocate(backend, Bool, geom.Ny)
    east_conformation_mask = KernelAbstractions.allocate(backend, Bool, geom.Ny)

    copyto!(q_wall, geom.q_wall)
    copyto!(is_solid, geom.is_solid)
    copyto!(west_hydro_mask, geom.west_hydro_mask)
    copyto!(east_hydro_mask, geom.east_hydro_mask)
    copyto!(west_conformation_mask, geom.west_conformation_mask)
    copyto!(east_conformation_mask, geom.east_conformation_mask)

    return StepChannelGeometry2D{T,typeof(q_wall),typeof(is_solid),typeof(west_hydro_mask)}(
        geom.name, geom.Nx, geom.Ny, geom.i_step, geom.inlet_open,
        geom.outlet_open, geom.H_ref, geom.H_in, geom.H_out,
        q_wall, is_solid, west_hydro_mask, east_hydro_mask,
        west_conformation_mask, east_conformation_mask)
end

function parabolic_face_profile_2d(geom::StepChannelGeometry2D;
                                   face::Symbol=:west,
                                   mean_velocity,
                                   FT::Type{<:AbstractFloat}=Float64)
    open = face === :west ? geom.inlet_open :
           face === :east ? geom.outlet_open :
           error("face must be :west or :east")
    H = length(open)
    H >= 2 || error("open face must contain at least two rows")
    u_max = FT(1.5) * FT(mean_velocity)
    profile = zeros(FT, geom.Ny)
    denom = FT(H - 1)^2
    @inbounds for (k, j) in enumerate(open)
        y_node = FT(k - 1)
        profile[j] = FT(4) * u_max * y_node * FT(H - k) / denom
    end
    return profile
end

function _log_sym2x2_components(cxx, cxy, cyy, ::Type{FT}) where {FT<:AbstractFloat}
    tr = cxx + cyy
    diff = cxx - cyy
    disc = sqrt(diff * diff + FT(4) * cxy * cxy)
    μ1 = FT(0.5) * (tr + disc)
    μ2 = FT(0.5) * (tr - disc)
    l1 = log(max(μ1, FT(1e-30)))
    l2 = log(max(μ2, FT(1e-30)))
    θ = FT(0.5) * atan(FT(2) * cxy, diff)
    c = cos(θ)
    s = sin(θ)
    return (c * c * l1 + s * s * l2,
            c * s * (l1 - l2),
            s * s * l1 + c * c * l2)
end

function oldroydb_inlet_conformation_profile_2d(geom::StepChannelGeometry2D;
                                                face::Symbol=:west,
                                                mean_velocity,
                                                λ,
                                                log_formulation::Bool=false,
                                                FT::Type{<:AbstractFloat}=Float64)
    open = face === :west ? geom.inlet_open :
           face === :east ? geom.outlet_open :
           error("face must be :west or :east")
    H = length(open)
    H >= 2 || error("open face must contain at least two rows")
    u_max = FT(1.5) * FT(mean_velocity)
    C_xx = ones(FT, geom.Ny)
    C_xy = zeros(FT, geom.Ny)
    C_yy = ones(FT, geom.Ny)
    H_chan = FT(H)
    @inbounds for (k, j) in enumerate(open)
        y = FT(k) - FT(0.5)
        dudy = u_max * FT(4) * (H_chan - FT(2) * y) / (H_chan * H_chan)
        C_xy[j] = FT(λ) * dudy
        C_xx[j] = FT(1) + FT(2) * (FT(λ) * dudy)^2
        if log_formulation
            C_xx[j], C_xy[j], C_yy[j] =
                _log_sym2x2_components(C_xx[j], C_xy[j], C_yy[j], FT)
        end
    end
    return C_xx, C_xy, C_yy
end

function default_step_bcspec_2d(geom::StepChannelGeometry2D, u_profile, ρ_out)
    if geom.inlet_open == 1:geom.Ny && geom.outlet_open == 1:geom.Ny
        return BCSpec2D(;
            west = ZouHeVelocity(u_profile),
            east = NeqExtrapolationPressure(ρ_out))
    end
    throw(ArgumentError(
        "default_step_bcspec_2d requires masked Zou-He BC support for partial openings"))
end
