@kernel function _bc_west_zh_velocity_2d!(f_out, f_in, profile, s_p, s_m,
                                           j_shift::Int=1)
    jm1 = @index(Global); j = jm1 + j_shift
    T = eltype(f_out)
    @inbounds begin
        fp1 = f_in[1, j,   1]
        fp3 = f_in[1, j-1, 3]
        fp4 = f_in[2, j,   4]
        fp5 = f_in[1, j+1, 5]
        fp7 = f_in[2, j-1, 7]
        fp8 = f_in[2, j+1, 8]
        u_in = profile[j]
        ρ_w  = (fp1 + fp3 + fp5 + T(2)*(fp4 + fp7 + fp8)) / (one(T) - u_in)
        fp2  = fp4 + T(2/3) * ρ_w * u_in
        fp6  = fp8 - T(0.5)*(fp3 - fp5) + T(1/6) * ρ_w * u_in
        fp9  = fp7 + T(0.5)*(fp3 - fp5) + T(1/6) * ρ_w * u_in
        F1,F2,F3,F4,F5,F6,F7,F8,F9 = _trt_collide_local(
            fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, s_p, s_m)
        f_out[1, j, 1] = F1; f_out[1, j, 2] = F2; f_out[1, j, 3] = F3
        f_out[1, j, 4] = F4; f_out[1, j, 5] = F5; f_out[1, j, 6] = F6
        f_out[1, j, 7] = F7; f_out[1, j, 8] = F8; f_out[1, j, 9] = F9
    end
end

@kernel function _bc_east_zh_pressure_2d!(f_out, f_in, Nx, ρ_out, s_p, s_m,
                                           j_shift::Int=1)
    jm1 = @index(Global); j = jm1 + j_shift
    T = eltype(f_out)
    @inbounds begin
        fp1 = f_in[Nx,   j,   1]
        fp2 = f_in[Nx-1, j,   2]
        fp3 = f_in[Nx,   j-1, 3]
        fp5 = f_in[Nx,   j+1, 5]
        fp6 = f_in[Nx-1, j-1, 6]
        fp9 = f_in[Nx-1, j+1, 9]
        u_x = -one(T) + (fp1 + fp3 + fp5 + T(2)*(fp2 + fp6 + fp9)) / ρ_out
        fp4 = fp2 - T(2/3) * ρ_out * u_x
        fp7 = fp9 - T(0.5)*(fp3 - fp5) - T(1/6) * ρ_out * u_x
        fp8 = fp6 + T(0.5)*(fp3 - fp5) - T(1/6) * ρ_out * u_x
        F1,F2,F3,F4,F5,F6,F7,F8,F9 = _trt_collide_local(
            fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, s_p, s_m)
        f_out[Nx, j, 1] = F1; f_out[Nx, j, 2] = F2; f_out[Nx, j, 3] = F3
        f_out[Nx, j, 4] = F4; f_out[Nx, j, 5] = F5; f_out[Nx, j, 6] = F6
        f_out[Nx, j, 7] = F7; f_out[Nx, j, 8] = F8; f_out[Nx, j, 9] = F9
    end
end

# East-face velocity BC, symmetric to west-velocity. Prescribes u_x at
# i=Nx; unknown populations after pull-stream are q=4,7,8. This enables
# multi-block topologies (e.g. O-grid rings) where inlet/outlet migrate
# from west/east after a transpose reorient.
@kernel function _bc_east_zh_velocity_2d!(f_out, f_in, Nx, profile, s_p, s_m,
                                           j_shift::Int=1)
    jm1 = @index(Global); j = jm1 + j_shift
    T = eltype(f_out)
    @inbounds begin
        fp1 = f_in[Nx,   j,   1]
        fp2 = f_in[Nx-1, j,   2]
        fp3 = f_in[Nx,   j-1, 3]
        fp5 = f_in[Nx,   j+1, 5]
        fp6 = f_in[Nx-1, j-1, 6]
        fp9 = f_in[Nx-1, j+1, 9]
        u_x = profile[j]
        ρ_w = (fp1 + fp3 + fp5 + T(2)*(fp2 + fp6 + fp9)) / (one(T) + u_x)
        fp4 = fp2 - T(2/3) * ρ_w * u_x
        fp7 = fp9 - T(0.5)*(fp3 - fp5) - T(1/6) * ρ_w * u_x
        fp8 = fp6 + T(0.5)*(fp3 - fp5) - T(1/6) * ρ_w * u_x
        F1,F2,F3,F4,F5,F6,F7,F8,F9 = _trt_collide_local(
            fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, s_p, s_m)
        f_out[Nx, j, 1] = F1; f_out[Nx, j, 2] = F2; f_out[Nx, j, 3] = F3
        f_out[Nx, j, 4] = F4; f_out[Nx, j, 5] = F5; f_out[Nx, j, 6] = F6
        f_out[Nx, j, 7] = F7; f_out[Nx, j, 8] = F8; f_out[Nx, j, 9] = F9
    end
end

# West-face pressure BC, symmetric to east-pressure, for the mirror case.
@kernel function _bc_west_zh_pressure_2d!(f_out, f_in, ρ_in, s_p, s_m,
                                           j_shift::Int=1)
    jm1 = @index(Global); j = jm1 + j_shift
    T = eltype(f_out)
    @inbounds begin
        fp1 = f_in[1, j,   1]
        fp3 = f_in[1, j-1, 3]
        fp4 = f_in[2, j,   4]
        fp5 = f_in[1, j+1, 5]
        fp7 = f_in[2, j-1, 7]
        fp8 = f_in[2, j+1, 8]
        u_x = one(T) - (fp1 + fp3 + fp5 + T(2)*(fp4 + fp7 + fp8)) / ρ_in
        fp2 = fp4 + T(2/3) * ρ_in * u_x
        fp6 = fp8 - T(0.5)*(fp3 - fp5) + T(1/6) * ρ_in * u_x
        fp9 = fp7 + T(0.5)*(fp3 - fp5) + T(1/6) * ρ_in * u_x
        F1,F2,F3,F4,F5,F6,F7,F8,F9 = _trt_collide_local(
            fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, s_p, s_m)
        f_out[1, j, 1] = F1; f_out[1, j, 2] = F2; f_out[1, j, 3] = F3
        f_out[1, j, 4] = F4; f_out[1, j, 5] = F5; f_out[1, j, 6] = F6
        f_out[1, j, 7] = F7; f_out[1, j, 8] = F8; f_out[1, j, 9] = F9
    end
end

# East-face kernel with WEST-direction populations (physical normal ≈ -x).
# Used when autoreorient maps a physical inlet (west-pointing) to logical east.
# Known after pull: q=4,7,8 (cx<0 → depart towards interior). Unknown: q=2,6,9.
@kernel function _bc_east_zh_velocity_westpops_2d!(f_out, f_in, Nx, profile, s_p, s_m,
                                                     j_shift::Int=1)
    jm1 = @index(Global); j = jm1 + j_shift
    T = eltype(f_out)
    @inbounds begin
        fp1 = f_in[Nx,     j,   1]
        fp3 = f_in[Nx,     j-1, 3]
        fp4 = f_in[Nx-1,   j,   4]
        fp5 = f_in[Nx,     j+1, 5]
        fp7 = f_in[Nx-1,   j-1, 7]
        fp8 = f_in[Nx-1,   j+1, 8]
        u_in = profile[j]
        ρ_w  = (fp1 + fp3 + fp5 + T(2)*(fp4 + fp7 + fp8)) / (one(T) - u_in)
        fp2  = fp4 + T(2/3) * ρ_w * u_in
        fp6  = fp8 - T(0.5)*(fp3 - fp5) + T(1/6) * ρ_w * u_in
        fp9  = fp7 + T(0.5)*(fp3 - fp5) + T(1/6) * ρ_w * u_in
        F1,F2,F3,F4,F5,F6,F7,F8,F9 = _trt_collide_local(
            fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, s_p, s_m)
        f_out[Nx, j, 1] = F1; f_out[Nx, j, 2] = F2; f_out[Nx, j, 3] = F3
        f_out[Nx, j, 4] = F4; f_out[Nx, j, 5] = F5; f_out[Nx, j, 6] = F6
        f_out[Nx, j, 7] = F7; f_out[Nx, j, 8] = F8; f_out[Nx, j, 9] = F9
    end
end

@kernel function _bc_east_zh_pressure_westpops_2d!(f_out, f_in, Nx, ρ_in, s_p, s_m,
                                                     j_shift::Int=1)
    jm1 = @index(Global); j = jm1 + j_shift
    T = eltype(f_out)
    @inbounds begin
        fp1 = f_in[Nx,     j,   1]
        fp3 = f_in[Nx,     j-1, 3]
        fp4 = f_in[Nx-1,   j,   4]
        fp5 = f_in[Nx,     j+1, 5]
        fp7 = f_in[Nx-1,   j-1, 7]
        fp8 = f_in[Nx-1,   j+1, 8]
        u_x = one(T) - (fp1 + fp3 + fp5 + T(2)*(fp4 + fp7 + fp8)) / ρ_in
        fp2 = fp4 + T(2/3) * ρ_in * u_x
        fp6 = fp8 - T(0.5)*(fp3 - fp5) + T(1/6) * ρ_in * u_x
        fp9 = fp7 + T(0.5)*(fp3 - fp5) + T(1/6) * ρ_in * u_x
        F1,F2,F3,F4,F5,F6,F7,F8,F9 = _trt_collide_local(
            fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, s_p, s_m)
        f_out[Nx, j, 1] = F1; f_out[Nx, j, 2] = F2; f_out[Nx, j, 3] = F3
        f_out[Nx, j, 4] = F4; f_out[Nx, j, 5] = F5; f_out[Nx, j, 6] = F6
        f_out[Nx, j, 7] = F7; f_out[Nx, j, 8] = F8; f_out[Nx, j, 9] = F9
    end
end

# East pressure outlet by non-equilibrium extrapolation (`NeqExtrapolationPressure`).
# Reads the post-collision populations of column Nx-1 from f_out, which the
# bulk step wrote in a previous launch; this kernel writes column Nx only, so
# the read is race-free for Nx ≥ 3. Rebuilds
#   f_q(Nx, j) = f_q(Nx-1, j) + f_q^eq(ρ_out - ρ, u),
# i.e. f^eq(ρ_out, u) + (f(Nx-1) - f^eq(ρ, u)) with (ρ, u) the raw moments of
# f(Nx-1, j) (f^eq is linear in ρ). No collision at Nx, so no rates.
@kernel function _bc_east_neq_extrap_2d!(f_out, Nx, ρ_out, j_shift::Int=1)
    jm1 = @index(Global); j = jm1 + j_shift
    @inbounds begin
        g1 = f_out[Nx-1, j, 1]; g2 = f_out[Nx-1, j, 2]; g3 = f_out[Nx-1, j, 3]
        g4 = f_out[Nx-1, j, 4]; g5 = f_out[Nx-1, j, 5]; g6 = f_out[Nx-1, j, 6]
        g7 = f_out[Nx-1, j, 7]; g8 = f_out[Nx-1, j, 8]; g9 = f_out[Nx-1, j, 9]
        ρ, ux, uy = moments_2d(g1, g2, g3, g4, g5, g6, g7, g8, g9)
        usq = ux * ux + uy * uy
        dρ = ρ_out - ρ
        f_out[Nx, j, 1] = g1 + feq_2d(Val(1), dρ, ux, uy, usq)
        f_out[Nx, j, 2] = g2 + feq_2d(Val(2), dρ, ux, uy, usq)
        f_out[Nx, j, 3] = g3 + feq_2d(Val(3), dρ, ux, uy, usq)
        f_out[Nx, j, 4] = g4 + feq_2d(Val(4), dρ, ux, uy, usq)
        f_out[Nx, j, 5] = g5 + feq_2d(Val(5), dρ, ux, uy, usq)
        f_out[Nx, j, 6] = g6 + feq_2d(Val(6), dρ, ux, uy, usq)
        f_out[Nx, j, 7] = g7 + feq_2d(Val(7), dρ, ux, uy, usq)
        f_out[Nx, j, 8] = g8 + feq_2d(Val(8), dρ, ux, uy, usq)
        f_out[Nx, j, 9] = g9 + feq_2d(Val(9), dρ, ux, uy, usq)
    end
end

# Rows 2:Ny-1 only; the corner rows keep the state written by the bulk step.
# In particular, wall corners keep half-way bounce-back. Shared by the
# uniform-rate and per-node-rate rebuilds: the rule uses no rate.
function _apply_bc_2d_east_neq_extrap!(backend, f_out, bc::NeqExtrapolationPressure,
                                       Nx, Ny, south_bc, north_bc)
    Nx ≥ 3 || throw(ArgumentError(
        "NeqExtrapolationPressure needs Nx ≥ 3 (reads column Nx-1), got Nx = $Nx"))
    count = Ny - 2
    count ≤ 0 && return nothing
    _bc_east_neq_extrap_2d!(backend)(f_out, Nx, eltype(f_out)(bc.ρ_out), 1;
                                     ndrange=(count,))
    return nothing
end

function _check_neq_extrap_faces_2d(bcspec)
    for (name, bc) in ((:west, bcspec.west), (:south, bcspec.south), (:north, bcspec.north))
        bc isa NeqExtrapolationPressure && throw(ArgumentError(
            "NeqExtrapolationPressure is implemented on the east face only (got it on :$name)"))
    end
    return nothing
end

# 2D dispatch per face. HalfwayBB is a no-op; other BCs call their kernel.
# south_bc / north_bc: when the adjacent face is InterfaceBC, extend the
# range to include the corner that would otherwise be skipped.
@inline function _apply_bc_2d_west!(backend, f_out, f_in, ::HalfwayBB,
                                     s_p, s_m, Nx, Ny; south_bc=nothing, north_bc=nothing) end
@inline function _apply_bc_2d_west!(backend, f_out, f_in, ::InterfaceBC,
                                     s_p, s_m, Nx, Ny; south_bc=nothing, north_bc=nothing) end
@inline function _apply_bc_2d_west!(backend, f_out, f_in, bc::ZouHeVelocity,
                                     s_p, s_m, Nx, Ny; south_bc=nothing, north_bc=nothing)
    j_lo = (south_bc isa InterfaceBC) ? 1 : 2
    j_hi = (north_bc isa InterfaceBC) ? Ny : Ny - 1
    count = j_hi - j_lo + 1
    count ≤ 0 && return nothing
    _bc_west_zh_velocity_2d!(backend)(f_out, f_in, bc.profile, s_p, s_m,
                                       j_lo - 1; ndrange=(count,))
end
@inline function _apply_bc_2d_west!(backend, f_out, f_in, bc::ZouHePressure,
                                     s_p, s_m, Nx, Ny; south_bc=nothing, north_bc=nothing)
    j_lo = (south_bc isa InterfaceBC) ? 1 : 2
    j_hi = (north_bc isa InterfaceBC) ? Ny : Ny - 1
    count = j_hi - j_lo + 1
    count ≤ 0 && return nothing
    _bc_west_zh_pressure_2d!(backend)(f_out, f_in, eltype(f_out)(bc.ρ_out),
                                        s_p, s_m, j_lo - 1; ndrange=(count,))
end

@inline function _apply_bc_2d_east!(backend, f_out, f_in, ::HalfwayBB,
                                     s_p, s_m, Nx, Ny; south_bc=nothing, north_bc=nothing) end
@inline function _apply_bc_2d_east!(backend, f_out, f_in, ::InterfaceBC,
                                     s_p, s_m, Nx, Ny; south_bc=nothing, north_bc=nothing) end
@inline function _apply_bc_2d_east!(backend, f_out, f_in, bc::ZouHePressure,
                                     s_p, s_m, Nx, Ny; south_bc=nothing, north_bc=nothing)
    j_lo = (south_bc isa InterfaceBC) ? 1 : 2
    j_hi = (north_bc isa InterfaceBC) ? Ny : Ny - 1
    count = j_hi - j_lo + 1
    count ≤ 0 && return nothing
    if bc.physical_dir === :west
        _bc_east_zh_pressure_westpops_2d!(backend)(f_out, f_in, Nx, eltype(f_out)(bc.ρ_out),
                                                     s_p, s_m, j_lo - 1; ndrange=(count,))
    else
        _bc_east_zh_pressure_2d!(backend)(f_out, f_in, Nx, eltype(f_out)(bc.ρ_out),
                                            s_p, s_m, j_lo - 1; ndrange=(count,))
    end
end
@inline function _apply_bc_2d_east!(backend, f_out, f_in, bc::NeqExtrapolationPressure,
                                     s_p, s_m, Nx, Ny; south_bc=nothing, north_bc=nothing)
    _apply_bc_2d_east_neq_extrap!(backend, f_out, bc, Nx, Ny, south_bc, north_bc)
end
@inline function _apply_bc_2d_east!(backend, f_out, f_in, bc::ZouHeVelocity,
                                     s_p, s_m, Nx, Ny; south_bc=nothing, north_bc=nothing)
    j_lo = (south_bc isa InterfaceBC) ? 1 : 2
    j_hi = (north_bc isa InterfaceBC) ? Ny : Ny - 1
    count = j_hi - j_lo + 1
    count ≤ 0 && return nothing
    if bc.physical_dir === :west
        _bc_east_zh_velocity_westpops_2d!(backend)(f_out, f_in, Nx, bc.profile,
                                                     s_p, s_m, j_lo - 1; ndrange=(count,))
    else
        _bc_east_zh_velocity_2d!(backend)(f_out, f_in, Nx, bc.profile,
                                            s_p, s_m, j_lo - 1; ndrange=(count,))
    end
end

# South / North wall bounce-back kernels.
# Overwrites the wall-crossing populations at j=1 (south) and j=Ny (north)
# with standard halfway BB: f_out[i,j,q̄] = f_in[i,j,q]. No collision
# needed — these are on the wall row itself if is_solid, or on the first
# fluid row if the streaming handles the solid row separately.
# For PullSLBM (which clamps at boundaries), these kernels fix the
# populations that the streaming couldn't bounce.

@kernel function _bc_south_halfwaybb_2d!(f_out, @Const(f_in), Ny, i_shift::Int)
    im1 = @index(Global); i = im1 + i_shift
    @inbounds begin
        # j=1: bounce populations heading south back north
        f_out[i, 1, 3] = f_in[i, 1, 5]   # 5→3
        f_out[i, 1, 6] = f_in[i, 1, 8]   # 8→6
        f_out[i, 1, 7] = f_in[i, 1, 9]   # 9→7
    end
end

@kernel function _bc_north_halfwaybb_2d!(f_out, @Const(f_in), Ny, i_shift::Int)
    im1 = @index(Global); i = im1 + i_shift
    @inbounds begin
        # j=Ny: bounce populations heading north back south
        f_out[i, Ny, 5] = f_in[i, Ny, 3]   # 3→5
        f_out[i, Ny, 8] = f_in[i, Ny, 6]   # 6→8
        f_out[i, Ny, 9] = f_in[i, Ny, 7]   # 7→9
    end
end

# South/North ZouHe kernels (uniform tau).
# South (j=1): unknown q=3,6,7 (cy>0). Known from interior (j=2): q=5,8,9.
# Tangential: q=2 from (i-1,1), q=4 from (i+1,1).
@kernel function _bc_south_zh_velocity_2d!(f_out, f_in, profile, s_p, s_m, i_shift::Int=0)
    im1 = @index(Global); i = im1 + i_shift
    T = eltype(f_out)
    @inbounds begin
        fp1 = f_in[i,   1, 1]; fp2 = f_in[i-1, 1, 2]; fp4 = f_in[i+1, 1, 4]
        fp5 = f_in[i,   2, 5]; fp8 = f_in[i+1, 2, 8]; fp9 = f_in[i-1, 2, 9]
        u_in = profile[i]
        ρ_w = (fp1 + fp2 + fp4 + T(2)*(fp5 + fp8 + fp9)) / (one(T) - u_in)
        fp3 = fp5 + T(2/3) * ρ_w * u_in
        fp6 = fp9 - T(0.5)*(fp4 - fp2) + T(1/6) * ρ_w * u_in
        fp7 = fp8 + T(0.5)*(fp4 - fp2) + T(1/6) * ρ_w * u_in
        F1,F2,F3,F4,F5,F6,F7,F8,F9 = _trt_collide_local(
            fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, s_p, s_m)
        f_out[i,1,1]=F1; f_out[i,1,2]=F2; f_out[i,1,3]=F3; f_out[i,1,4]=F4; f_out[i,1,5]=F5
        f_out[i,1,6]=F6; f_out[i,1,7]=F7; f_out[i,1,8]=F8; f_out[i,1,9]=F9
    end
end

@kernel function _bc_south_zh_pressure_2d!(f_out, f_in, ρ_out, s_p, s_m, i_shift::Int=0)
    im1 = @index(Global); i = im1 + i_shift
    T = eltype(f_out)
    @inbounds begin
        fp1 = f_in[i,   1, 1]; fp2 = f_in[i-1, 1, 2]; fp4 = f_in[i+1, 1, 4]
        fp5 = f_in[i,   2, 5]; fp8 = f_in[i+1, 2, 8]; fp9 = f_in[i-1, 2, 9]
        u_y = -one(T) + (fp1 + fp2 + fp4 + T(2)*(fp5 + fp8 + fp9)) / ρ_out
        fp3 = fp5 + T(2/3) * ρ_out * u_y
        fp6 = fp9 - T(0.5)*(fp4 - fp2) + T(1/6) * ρ_out * u_y
        fp7 = fp8 + T(0.5)*(fp4 - fp2) + T(1/6) * ρ_out * u_y
        F1,F2,F3,F4,F5,F6,F7,F8,F9 = _trt_collide_local(
            fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, s_p, s_m)
        f_out[i,1,1]=F1; f_out[i,1,2]=F2; f_out[i,1,3]=F3; f_out[i,1,4]=F4; f_out[i,1,5]=F5
        f_out[i,1,6]=F6; f_out[i,1,7]=F7; f_out[i,1,8]=F8; f_out[i,1,9]=F9
    end
end

# North (j=Ny): unknown q=5,8,9 (cy<0). Known from interior (j=Ny-1): q=3,6,7.
@kernel function _bc_north_zh_velocity_2d!(f_out, f_in, Ny, profile, s_p, s_m, i_shift::Int=0)
    im1 = @index(Global); i = im1 + i_shift
    T = eltype(f_out)
    @inbounds begin
        fp1 = f_in[i,   Ny, 1]; fp2 = f_in[i-1, Ny, 2]; fp4 = f_in[i+1, Ny, 4]
        fp3 = f_in[i, Ny-1, 3]; fp6 = f_in[i-1, Ny-1, 6]; fp7 = f_in[i+1, Ny-1, 7]
        u_in = profile[i]
        ρ_w = (fp1 + fp2 + fp4 + T(2)*(fp3 + fp6 + fp7)) / (one(T) + u_in)
        fp5 = fp3 - T(2/3) * ρ_w * u_in
        fp8 = fp6 + T(0.5)*(fp4 - fp2) - T(1/6) * ρ_w * u_in
        fp9 = fp7 - T(0.5)*(fp4 - fp2) - T(1/6) * ρ_w * u_in
        F1,F2,F3,F4,F5,F6,F7,F8,F9 = _trt_collide_local(
            fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, s_p, s_m)
        f_out[i,Ny,1]=F1; f_out[i,Ny,2]=F2; f_out[i,Ny,3]=F3; f_out[i,Ny,4]=F4; f_out[i,Ny,5]=F5
        f_out[i,Ny,6]=F6; f_out[i,Ny,7]=F7; f_out[i,Ny,8]=F8; f_out[i,Ny,9]=F9
    end
end

@kernel function _bc_north_zh_pressure_2d!(f_out, f_in, Ny, ρ_out, s_p, s_m, i_shift::Int=0)
    im1 = @index(Global); i = im1 + i_shift
    T = eltype(f_out)
    @inbounds begin
        fp1 = f_in[i,   Ny, 1]; fp2 = f_in[i-1, Ny, 2]; fp4 = f_in[i+1, Ny, 4]
        fp3 = f_in[i, Ny-1, 3]; fp6 = f_in[i-1, Ny-1, 6]; fp7 = f_in[i+1, Ny-1, 7]
        u_y = one(T) - (fp1 + fp2 + fp4 + T(2)*(fp3 + fp6 + fp7)) / ρ_out
        fp5 = fp3 - T(2/3) * ρ_out * u_y
        fp8 = fp6 + T(0.5)*(fp4 - fp2) - T(1/6) * ρ_out * u_y
        fp9 = fp7 - T(0.5)*(fp4 - fp2) - T(1/6) * ρ_out * u_y
        F1,F2,F3,F4,F5,F6,F7,F8,F9 = _trt_collide_local(
            fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, s_p, s_m)
        f_out[i,Ny,1]=F1; f_out[i,Ny,2]=F2; f_out[i,Ny,3]=F3; f_out[i,Ny,4]=F4; f_out[i,Ny,5]=F5
        f_out[i,Ny,6]=F6; f_out[i,Ny,7]=F7; f_out[i,Ny,8]=F8; f_out[i,Ny,9]=F9
    end
end

# Local-tau variants: read sp/sm from 2D arrays at the face index
@kernel function _bc_west_zh_velocity_local_2d!(f_out, f_in, profile, sp_field, sm_field,
                                                  j_shift::Int=1)
    jm1 = @index(Global); j = jm1 + j_shift
    T = eltype(f_out)
    @inbounds begin
        fp1 = f_in[1, j,   1]
        fp3 = f_in[1, j-1, 3]
        fp4 = f_in[2, j,   4]
        fp5 = f_in[1, j+1, 5]
        fp7 = f_in[2, j-1, 7]
        fp8 = f_in[2, j+1, 8]
        u_in = profile[j]
        ρ_w  = (fp1 + fp3 + fp5 + T(2)*(fp4 + fp7 + fp8)) / (one(T) - u_in)
        fp2  = fp4 + T(2/3) * ρ_w * u_in
        fp6  = fp8 - T(0.5)*(fp3 - fp5) + T(1/6) * ρ_w * u_in
        fp9  = fp7 + T(0.5)*(fp3 - fp5) + T(1/6) * ρ_w * u_in
        s_p = sp_field[1, j]; s_m = sm_field[1, j]
        F1,F2,F3,F4,F5,F6,F7,F8,F9 = _trt_collide_local(
            fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, s_p, s_m)
        f_out[1, j, 1] = F1; f_out[1, j, 2] = F2; f_out[1, j, 3] = F3
        f_out[1, j, 4] = F4; f_out[1, j, 5] = F5; f_out[1, j, 6] = F6
        f_out[1, j, 7] = F7; f_out[1, j, 8] = F8; f_out[1, j, 9] = F9
    end
end

@kernel function _bc_east_zh_pressure_local_2d!(f_out, f_in, Nx, ρ_out, sp_field, sm_field,
                                                  j_shift::Int=1)
    jm1 = @index(Global); j = jm1 + j_shift
    T = eltype(f_out)
    @inbounds begin
        fp1 = f_in[Nx,   j,   1]
        fp2 = f_in[Nx-1, j,   2]
        fp3 = f_in[Nx,   j-1, 3]
        fp5 = f_in[Nx,   j+1, 5]
        fp6 = f_in[Nx-1, j-1, 6]
        fp9 = f_in[Nx-1, j+1, 9]
        u_x = -one(T) + (fp1 + fp3 + fp5 + T(2)*(fp2 + fp6 + fp9)) / ρ_out
        fp4 = fp2 - T(2/3) * ρ_out * u_x
        fp7 = fp9 - T(0.5)*(fp3 - fp5) - T(1/6) * ρ_out * u_x
        fp8 = fp6 + T(0.5)*(fp3 - fp5) - T(1/6) * ρ_out * u_x
        s_p = sp_field[Nx, j]; s_m = sm_field[Nx, j]
        F1,F2,F3,F4,F5,F6,F7,F8,F9 = _trt_collide_local(
            fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, s_p, s_m)
        f_out[Nx, j, 1] = F1; f_out[Nx, j, 2] = F2; f_out[Nx, j, 3] = F3
        f_out[Nx, j, 4] = F4; f_out[Nx, j, 5] = F5; f_out[Nx, j, 6] = F6
        f_out[Nx, j, 7] = F7; f_out[Nx, j, 8] = F8; f_out[Nx, j, 9] = F9
    end
end

@kernel function _bc_east_zh_velocity_local_2d!(f_out, f_in, Nx, profile,
                                                    sp_field, sm_field,
                                                    j_shift::Int=1)
    jm1 = @index(Global); j = jm1 + j_shift
    T = eltype(f_out)
    @inbounds begin
        fp1 = f_in[Nx,   j,   1]
        fp2 = f_in[Nx-1, j,   2]
        fp3 = f_in[Nx,   j-1, 3]
        fp5 = f_in[Nx,   j+1, 5]
        fp6 = f_in[Nx-1, j-1, 6]
        fp9 = f_in[Nx-1, j+1, 9]
        u_x = profile[j]
        ρ_w = (fp1 + fp3 + fp5 + T(2)*(fp2 + fp6 + fp9)) / (one(T) + u_x)
        fp4 = fp2 - T(2/3) * ρ_w * u_x
        fp7 = fp9 - T(0.5)*(fp3 - fp5) - T(1/6) * ρ_w * u_x
        fp8 = fp6 + T(0.5)*(fp3 - fp5) - T(1/6) * ρ_w * u_x
        s_p = sp_field[Nx, j]; s_m = sm_field[Nx, j]
        F1,F2,F3,F4,F5,F6,F7,F8,F9 = _trt_collide_local(
            fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, s_p, s_m)
        f_out[Nx, j, 1] = F1; f_out[Nx, j, 2] = F2; f_out[Nx, j, 3] = F3
        f_out[Nx, j, 4] = F4; f_out[Nx, j, 5] = F5; f_out[Nx, j, 6] = F6
        f_out[Nx, j, 7] = F7; f_out[Nx, j, 8] = F8; f_out[Nx, j, 9] = F9
    end
end

@kernel function _bc_west_zh_pressure_local_2d!(f_out, f_in, ρ_in,
                                                   sp_field, sm_field,
                                                   j_shift::Int=1)
    jm1 = @index(Global); j = jm1 + j_shift
    T = eltype(f_out)
    @inbounds begin
        fp1 = f_in[1, j,   1]
        fp3 = f_in[1, j-1, 3]
        fp4 = f_in[2, j,   4]
        fp5 = f_in[1, j+1, 5]
        fp7 = f_in[2, j-1, 7]
        fp8 = f_in[2, j+1, 8]
        u_x = one(T) - (fp1 + fp3 + fp5 + T(2)*(fp4 + fp7 + fp8)) / ρ_in
        fp2 = fp4 + T(2/3) * ρ_in * u_x
        fp6 = fp8 - T(0.5)*(fp3 - fp5) + T(1/6) * ρ_in * u_x
        fp9 = fp7 + T(0.5)*(fp3 - fp5) + T(1/6) * ρ_in * u_x
        s_p = sp_field[1, j]; s_m = sm_field[1, j]
        F1,F2,F3,F4,F5,F6,F7,F8,F9 = _trt_collide_local(
            fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, s_p, s_m)
        f_out[1, j, 1] = F1; f_out[1, j, 2] = F2; f_out[1, j, 3] = F3
        f_out[1, j, 4] = F4; f_out[1, j, 5] = F5; f_out[1, j, 6] = F6
        f_out[1, j, 7] = F7; f_out[1, j, 8] = F8; f_out[1, j, 9] = F9
    end
end

@kernel function _bc_east_zh_velocity_westpops_local_2d!(f_out, f_in, Nx, profile,
                                                              sp_field, sm_field,
                                                              j_shift::Int=1)
    jm1 = @index(Global); j = jm1 + j_shift
    T = eltype(f_out)
    @inbounds begin
        fp1 = f_in[Nx,     j,   1]
        fp3 = f_in[Nx,     j-1, 3]
        fp4 = f_in[Nx-1,   j,   4]
        fp5 = f_in[Nx,     j+1, 5]
        fp7 = f_in[Nx-1,   j-1, 7]
        fp8 = f_in[Nx-1,   j+1, 8]
        u_in = profile[j]
        ρ_w  = (fp1 + fp3 + fp5 + T(2)*(fp4 + fp7 + fp8)) / (one(T) - u_in)
        fp2  = fp4 + T(2/3) * ρ_w * u_in
        fp6  = fp8 - T(0.5)*(fp3 - fp5) + T(1/6) * ρ_w * u_in
        fp9  = fp7 + T(0.5)*(fp3 - fp5) + T(1/6) * ρ_w * u_in
        s_p = sp_field[Nx, j]; s_m = sm_field[Nx, j]
        F1,F2,F3,F4,F5,F6,F7,F8,F9 = _trt_collide_local(
            fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, s_p, s_m)
        f_out[Nx, j, 1] = F1; f_out[Nx, j, 2] = F2; f_out[Nx, j, 3] = F3
        f_out[Nx, j, 4] = F4; f_out[Nx, j, 5] = F5; f_out[Nx, j, 6] = F6
        f_out[Nx, j, 7] = F7; f_out[Nx, j, 8] = F8; f_out[Nx, j, 9] = F9
    end
end

@kernel function _bc_east_zh_pressure_westpops_local_2d!(f_out, f_in, Nx, ρ_in,
                                                              sp_field, sm_field,
                                                              j_shift::Int=1)
    jm1 = @index(Global); j = jm1 + j_shift
    T = eltype(f_out)
    @inbounds begin
        fp1 = f_in[Nx,     j,   1]
        fp3 = f_in[Nx,     j-1, 3]
        fp4 = f_in[Nx-1,   j,   4]
        fp5 = f_in[Nx,     j+1, 5]
        fp7 = f_in[Nx-1,   j-1, 7]
        fp8 = f_in[Nx-1,   j+1, 8]
        u_x = one(T) - (fp1 + fp3 + fp5 + T(2)*(fp4 + fp7 + fp8)) / ρ_in
        fp2 = fp4 + T(2/3) * ρ_in * u_x
        fp6 = fp8 - T(0.5)*(fp3 - fp5) + T(1/6) * ρ_in * u_x
        fp9 = fp7 + T(0.5)*(fp3 - fp5) + T(1/6) * ρ_in * u_x
        s_p = sp_field[Nx, j]; s_m = sm_field[Nx, j]
        F1,F2,F3,F4,F5,F6,F7,F8,F9 = _trt_collide_local(
            fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, s_p, s_m)
        f_out[Nx, j, 1] = F1; f_out[Nx, j, 2] = F2; f_out[Nx, j, 3] = F3
        f_out[Nx, j, 4] = F4; f_out[Nx, j, 5] = F5; f_out[Nx, j, 6] = F6
        f_out[Nx, j, 7] = F7; f_out[Nx, j, 8] = F8; f_out[Nx, j, 9] = F9
    end
end

@inline function _apply_bc_2d_west_local!(backend, f_out, f_in, ::HalfwayBB,
                                           sp_field, sm_field, Nx, Ny;
                                           south_bc=nothing, north_bc=nothing) end
@inline function _apply_bc_2d_west_local!(backend, f_out, f_in, ::InterfaceBC,
                                           sp_field, sm_field, Nx, Ny;
                                           south_bc=nothing, north_bc=nothing) end
@inline function _apply_bc_2d_west_local!(backend, f_out, f_in, bc::ZouHeVelocity,
                                           sp_field, sm_field, Nx, Ny;
                                           south_bc=nothing, north_bc=nothing)
    j_lo = (south_bc isa InterfaceBC) ? 1 : 2
    j_hi = (north_bc isa InterfaceBC) ? Ny : Ny - 1
    count = j_hi - j_lo + 1
    count ≤ 0 && return nothing
    _bc_west_zh_velocity_local_2d!(backend)(f_out, f_in, bc.profile, sp_field, sm_field,
                                             j_lo - 1; ndrange=(count,))
end
@inline function _apply_bc_2d_west_local!(backend, f_out, f_in, bc::ZouHePressure,
                                           sp_field, sm_field, Nx, Ny;
                                           south_bc=nothing, north_bc=nothing)
    j_lo = (south_bc isa InterfaceBC) ? 1 : 2
    j_hi = (north_bc isa InterfaceBC) ? Ny : Ny - 1
    count = j_hi - j_lo + 1
    count ≤ 0 && return nothing
    _bc_west_zh_pressure_local_2d!(backend)(f_out, f_in, eltype(f_out)(bc.ρ_out),
                                              sp_field, sm_field, j_lo - 1; ndrange=(count,))
end
@inline function _apply_bc_2d_east_local!(backend, f_out, f_in, ::HalfwayBB,
                                           sp_field, sm_field, Nx, Ny;
                                           south_bc=nothing, north_bc=nothing) end
@inline function _apply_bc_2d_east_local!(backend, f_out, f_in, ::InterfaceBC,
                                           sp_field, sm_field, Nx, Ny;
                                           south_bc=nothing, north_bc=nothing) end
@inline function _apply_bc_2d_east_local!(backend, f_out, f_in, bc::NeqExtrapolationPressure,
                                           sp_field, sm_field, Nx, Ny;
                                           south_bc=nothing, north_bc=nothing)
    _apply_bc_2d_east_neq_extrap!(backend, f_out, bc, Nx, Ny, south_bc, north_bc)
end
@inline function _apply_bc_2d_east_local!(backend, f_out, f_in, bc::ZouHeVelocity,
                                           sp_field, sm_field, Nx, Ny;
                                           south_bc=nothing, north_bc=nothing)
    j_lo = (south_bc isa InterfaceBC) ? 1 : 2
    j_hi = (north_bc isa InterfaceBC) ? Ny : Ny - 1
    count = j_hi - j_lo + 1
    count ≤ 0 && return nothing
    if bc.physical_dir === :west
        _bc_east_zh_velocity_westpops_local_2d!(backend)(f_out, f_in, Nx, bc.profile,
                                                           sp_field, sm_field, j_lo - 1; ndrange=(count,))
    else
        _bc_east_zh_velocity_local_2d!(backend)(f_out, f_in, Nx, bc.profile,
                                                  sp_field, sm_field, j_lo - 1; ndrange=(count,))
    end
end
@inline function _apply_bc_2d_east_local!(backend, f_out, f_in, bc::ZouHePressure,
                                           sp_field, sm_field, Nx, Ny;
                                           south_bc=nothing, north_bc=nothing)
    j_lo = (south_bc isa InterfaceBC) ? 1 : 2
    j_hi = (north_bc isa InterfaceBC) ? Ny : Ny - 1
    count = j_hi - j_lo + 1
    count ≤ 0 && return nothing
    if bc.physical_dir === :west
        _bc_east_zh_pressure_westpops_local_2d!(backend)(f_out, f_in, Nx, eltype(f_out)(bc.ρ_out),
                                                           sp_field, sm_field, j_lo - 1; ndrange=(count,))
    else
        _bc_east_zh_pressure_local_2d!(backend)(f_out, f_in, Nx, eltype(f_out)(bc.ρ_out),
                                                  sp_field, sm_field, j_lo - 1; ndrange=(count,))
    end
end

@inline function _apply_bc_2d_south!(backend, f_out, f_in, ::InterfaceBC,
                                      s_p, s_m, Nx, Ny;
                                      west_bc=nothing, east_bc=nothing) end
@inline function _apply_bc_2d_south!(backend, f_out, f_in, ::HalfwayBB,
                                      s_p, s_m, Nx, Ny;
                                      west_bc=nothing, east_bc=nothing) end
@inline function _apply_bc_2d_south!(backend, f_out, f_in, bc::ZouHeVelocity,
                                      s_p, s_m, Nx, Ny;
                                      west_bc=nothing, east_bc=nothing)
    i_lo = (west_bc isa InterfaceBC) ? 1 : 2
    i_hi = (east_bc isa InterfaceBC) ? Nx : Nx - 1
    count = i_hi - i_lo + 1; count ≤ 0 && return nothing
    _bc_south_zh_velocity_2d!(backend)(f_out, f_in, bc.profile, s_p, s_m,
                                        i_lo - 1; ndrange=(count,))
end
@inline function _apply_bc_2d_south!(backend, f_out, f_in, bc::ZouHePressure,
                                      s_p, s_m, Nx, Ny;
                                      west_bc=nothing, east_bc=nothing)
    i_lo = (west_bc isa InterfaceBC) ? 1 : 2
    i_hi = (east_bc isa InterfaceBC) ? Nx : Nx - 1
    count = i_hi - i_lo + 1; count ≤ 0 && return nothing
    _bc_south_zh_pressure_2d!(backend)(f_out, f_in, eltype(f_out)(bc.ρ_out), s_p, s_m,
                                        i_lo - 1; ndrange=(count,))
end
@inline function _apply_bc_2d_north!(backend, f_out, f_in, ::InterfaceBC,
                                      s_p, s_m, Nx, Ny;
                                      west_bc=nothing, east_bc=nothing) end
@inline function _apply_bc_2d_north!(backend, f_out, f_in, ::HalfwayBB,
                                      s_p, s_m, Nx, Ny;
                                      west_bc=nothing, east_bc=nothing) end
@inline function _apply_bc_2d_north!(backend, f_out, f_in, bc::ZouHeVelocity,
                                      s_p, s_m, Nx, Ny;
                                      west_bc=nothing, east_bc=nothing)
    i_lo = (west_bc isa InterfaceBC) ? 1 : 2
    i_hi = (east_bc isa InterfaceBC) ? Nx : Nx - 1
    count = i_hi - i_lo + 1; count ≤ 0 && return nothing
    _bc_north_zh_velocity_2d!(backend)(f_out, f_in, Ny, bc.profile, s_p, s_m,
                                        i_lo - 1; ndrange=(count,))
end
@inline function _apply_bc_2d_north!(backend, f_out, f_in, bc::ZouHePressure,
                                      s_p, s_m, Nx, Ny;
                                      west_bc=nothing, east_bc=nothing)
    i_lo = (west_bc isa InterfaceBC) ? 1 : 2
    i_hi = (east_bc isa InterfaceBC) ? Nx : Nx - 1
    count = i_hi - i_lo + 1; count ≤ 0 && return nothing
    _bc_north_zh_pressure_2d!(backend)(f_out, f_in, Ny, eltype(f_out)(bc.ρ_out), s_p, s_m,
                                        i_lo - 1; ndrange=(count,))
end

"""
    apply_bc_rebuild_2d!(f_out, f_in, bcspec, ν, Nx, Ny;
                          sp_field=nothing, sm_field=nothing)

Apply the per-face BCs in `bcspec::BCSpec2D` to `f_out` after the bulk step.
Zou-He faces read pre-step values from `f_in` (streamed from interior),
close the populations, and collide locally with TRT Λ=3/16 at viscosity `ν`.
The east `NeqExtrapolationPressure` face reads the completed post-collision
column `Nx-1` of `f_out` and rebuilds rows `2:Ny-1` without collision.

If `sp_field` and `sm_field` (2D arrays) are provided, per-cell local
rates are used at the west/east Zou-He faces instead of uniform ν-derived rates.
The extrapolation outlet is independent of these rates.
This is needed for SLBM on non-uniform meshes where τ varies per cell.
"""
function apply_bc_rebuild_2d!(f_out, f_in, bcspec::BCSpec2D, ν::Real,
                                Nx::Int, Ny::Int;
                                sp_field=nothing, sm_field=nothing,
                                ρ_out=nothing, ux_out=nothing, uy_out=nothing)
    backend = KernelAbstractions.get_backend(f_out)
    T = eltype(f_out)
    _check_neq_extrap_faces_2d(bcspec)
    s_p_r, s_m_r = trt_rates(ν; Λ=3/16)
    s_p_uni = T(s_p_r); s_m_uni = T(s_m_r)

    if isnothing(sp_field)
        _apply_bc_2d_west!(backend, f_out, f_in, bcspec.west, s_p_uni, s_m_uni, Nx, Ny;
                             south_bc=bcspec.south, north_bc=bcspec.north)
        _apply_bc_2d_east!(backend, f_out, f_in, bcspec.east, s_p_uni, s_m_uni, Nx, Ny;
                             south_bc=bcspec.south, north_bc=bcspec.north)
    else
        _apply_bc_2d_west_local!(backend, f_out, f_in, bcspec.west, sp_field, sm_field, Nx, Ny;
                                   south_bc=bcspec.south, north_bc=bcspec.north)
        _apply_bc_2d_east_local!(backend, f_out, f_in, bcspec.east, sp_field, sm_field, Nx, Ny;
                                   south_bc=bcspec.south, north_bc=bcspec.north)
    end
    _apply_bc_2d_south!(backend, f_out, f_in, bcspec.south, s_p_uni, s_m_uni, Nx, Ny;
                          west_bc=bcspec.west, east_bc=bcspec.east)
    _apply_bc_2d_north!(backend, f_out, f_in, bcspec.north, s_p_uni, s_m_uni, Nx, Ny;
                          west_bc=bcspec.west, east_bc=bcspec.east)

    if !isnothing(ρ_out)
        _update_bc_moments_2d!(f_out, ρ_out, ux_out, uy_out, bcspec, Nx, Ny)
    end
    return nothing
end

