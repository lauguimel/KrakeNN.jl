using KernelAbstractions

# --- Zou-He velocity boundary condition (2D) ---

"""
    zou_he_velocity_north_2d!(f, ρ, u_wall_x, Nx)

Apply Zou-He velocity BC on the north boundary (j = Ny).
Imposes velocity (u_wall_x, 0) on the top row.

For the lid-driven cavity: u_wall_x = u_lid, applied at j = Ny.
"""
@kernel function zou_he_velocity_north_2d_kernel!(f, u_wall_x, Ny)
    i = @index(Global)
    j = Ny

    @inbounds begin
        T = eltype(f)

        # Known populations (coming from interior or along boundary):
        # f1 (rest), f2 (+x), f4 (-x), f6 (+x,+y), f7 (-x,+y), f3 (+y)
        # Unknown: f5 (0,-1), f8 (-x,-y), f9 (+x,-1)

        f1 = f[i, j, 1]
        f2 = f[i, j, 2]
        f3 = f[i, j, 3]
        f4 = f[i, j, 4]
        f6 = f[i, j, 6]
        f7 = f[i, j, 7]

        # Compute density from equilibrium assumption
        ρ_wall = (f1 + f2 + f4 + T(2) * (f3 + f6 + f7)) / (one(T) + u_wall_x * T(0))
        # For uy_wall = 0: ρ = (f1 + f2 + f4 + 2*(f3 + f6 + f7))

        # Unknown populations (Zou-He)
        f[i, j, 5] = f3 # - T(2.0/3.0) * ρ_wall * uy_wall  (uy_wall = 0)

        f[i, j, 9] = f7 - T(0.5) * (f2 - f4) + T(0.5) * ρ_wall * u_wall_x

        f[i, j, 8] = f6 + T(0.5) * (f2 - f4) - T(0.5) * ρ_wall * u_wall_x
    end
end

"""
    apply_zou_he_north_2d!(f, u_wall_x, Nx, Ny)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.apply_zou_he_north_2d!)
```
"""
function apply_zou_he_north_2d!(f, u_wall_x, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = zou_he_velocity_north_2d_kernel!(backend)
    kernel!(f, eltype(f)(u_wall_x), Ny; ndrange=(Nx,))
end

# --- Bounce-back on all 4 walls (2D cavity, no-slip) ---

@kernel function bounce_back_walls_2d_kernel!(f, Nx, Ny, wall_id)
    idx = @index(Global)
    @inbounds begin
        if wall_id == 1  # South (j=1)
            i = idx
            # Populations pointing inward (north): 3,6,7 are reflected
            f[i, 1, 3] = f[i, 1, 5]   # (0,+1) <- (0,-1)
            f[i, 1, 6] = f[i, 1, 8]   # (+1,+1) <- (-1,-1)
            f[i, 1, 7] = f[i, 1, 9]   # (-1,+1) <- (+1,-1)
        elseif wall_id == 2  # West (i=1)
            j = idx
            f[1, j, 2] = f[1, j, 4]   # (+1,0) <- (-1,0)
            f[1, j, 6] = f[1, j, 8]   # (+1,+1) <- (-1,-1)
            f[1, j, 9] = f[1, j, 7]   # (+1,-1) <- (-1,+1)
        elseif wall_id == 3  # East (i=Nx)
            j = idx
            f[Nx, j, 4] = f[Nx, j, 2]
            f[Nx, j, 7] = f[Nx, j, 9]
            f[Nx, j, 8] = f[Nx, j, 6]
        elseif wall_id == 4  # North (j=Ny)
            i = idx
            f[i, Ny, 5] = f[i, Ny, 3]
            f[i, Ny, 8] = f[i, Ny, 6]
            f[i, Ny, 9] = f[i, Ny, 7]
        end
    end
end

"""
    apply_bounce_back_walls_2d!(f, Nx, Ny)

Apply simple bounce-back on south, west, and east walls of a 2D cavity.
North wall is handled by Zou-He (lid velocity).
"""
function apply_bounce_back_walls_2d!(f, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = bounce_back_walls_2d_kernel!(backend)

    # South wall (j=1)
    kernel!(f, Nx, Ny, Int32(1); ndrange=(Nx,))
    # West wall (i=1)
    kernel!(f, Nx, Ny, Int32(2); ndrange=(Ny,))
    # East wall (i=Nx)
    kernel!(f, Nx, Ny, Int32(3); ndrange=(Ny,))

end

"""
    apply_bounce_back_wall_2d!(f, Nx, Ny, side)

Apply bounce-back on a single wall. `side ∈ (:south, :west, :east, :north)`.
Used by refinement patches that touch only some domain walls.
"""
function apply_bounce_back_wall_2d!(f, Nx, Ny, side::Symbol)
    backend = KernelAbstractions.get_backend(f)
    kernel! = bounce_back_walls_2d_kernel!(backend)
    if side === :south
        kernel!(f, Nx, Ny, Int32(1); ndrange=(Nx,))
    elseif side === :west
        kernel!(f, Nx, Ny, Int32(2); ndrange=(Ny,))
    elseif side === :east
        kernel!(f, Nx, Ny, Int32(3); ndrange=(Ny,))
    elseif side === :north
        kernel!(f, Nx, Ny, Int32(4); ndrange=(Nx,))
    else
        error("apply_bounce_back_wall_2d!: unknown side $(side)")
    end
end

# --- Zou-He velocity BC on south wall (j=1) — for Couette ---

@kernel function zou_he_velocity_south_2d_kernel!(f, u_wall_x)
    i = @index(Global)
    j = 1

    @inbounds begin
        T = eltype(f)
        f1 = f[i,j,1]; f2 = f[i,j,2]; f4 = f[i,j,4]
        f5 = f[i,j,5]; f8 = f[i,j,8]; f9 = f[i,j,9]

        ρ_wall = f1 + f2 + f4 + T(2) * (f5 + f8 + f9)
        f[i,j,3] = f5
        f[i,j,6] = f8 - T(0.5) * (f2 - f4) + T(0.5) * ρ_wall * u_wall_x
        f[i,j,7] = f9 + T(0.5) * (f2 - f4) - T(0.5) * ρ_wall * u_wall_x
    end
end

"""
    apply_zou_he_south_2d!(f, u_wall_x, Nx)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.apply_zou_he_south_2d!)
```
"""
function apply_zou_he_south_2d!(f, u_wall_x, Nx)
    backend = KernelAbstractions.get_backend(f)
    kernel! = zou_he_velocity_south_2d_kernel!(backend)
    kernel!(f, eltype(f)(u_wall_x); ndrange=(Nx,))
end

# --- Zou-He velocity inlet on west wall (i=1) — for cylinder ---

@kernel function zou_he_velocity_west_2d_kernel!(f, u_in, Ny)
    j = @index(Global)

    @inbounds begin
        T = eltype(f)
        f1 = f[1,j,1]; f3 = f[1,j,3]; f4 = f[1,j,4]
        f5 = f[1,j,5]; f7 = f[1,j,7]; f8 = f[1,j,8]

        ρ_wall = (f1 + f3 + f5 + T(2) * (f4 + f7 + f8)) / (one(T) - u_in)
        f[1,j,2] = f4 + T(2.0/3.0) * ρ_wall * u_in
        f[1,j,6] = f8 - T(0.5) * (f3 - f5) + T(1.0/6.0) * ρ_wall * u_in
        f[1,j,9] = f7 + T(0.5) * (f3 - f5) + T(1.0/6.0) * ρ_wall * u_in
    end
end

"""
    apply_zou_he_west_2d!(f, u_in, Nx, Ny)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.apply_zou_he_west_2d!)
```
"""
function apply_zou_he_west_2d!(f, u_in, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = zou_he_velocity_west_2d_kernel!(backend)
    kernel!(f, eltype(f)(u_in), Ny; ndrange=(Ny,))
end

# --- Zou-He pressure outlet on east wall (i=Nx) ---

@kernel function zou_he_pressure_east_2d_kernel!(f, Nx, ρ_out)
    j = @index(Global)

    @inbounds begin
        T = eltype(f)
        # Known: f1, f3(+y), f5(-y) parallel; f2(+x), f6(+x,+y), f9(+x,-y) from interior
        f1 = f[Nx,j,1]; f2 = f[Nx,j,2]; f3 = f[Nx,j,3]
        f5 = f[Nx,j,5]; f6 = f[Nx,j,6]; f9 = f[Nx,j,9]

        # Compute ux from fixed ρ_out
        ux = -one(T) + (f1 + f3 + f5 + T(2) * (f2 + f6 + f9)) / ρ_out

        # Unknown: f4(-x), f7(-x,+y), f8(-x,-y)
        f[Nx,j,4] = f2 - T(2.0/3.0) * ρ_out * ux
        f[Nx,j,7] = f9 - T(0.5) * (f3 - f5) - T(1.0/6.0) * ρ_out * ux
        f[Nx,j,8] = f6 + T(0.5) * (f3 - f5) - T(1.0/6.0) * ρ_out * ux
    end
end

"""
    apply_zou_he_pressure_east_2d!(f, Nx, Ny; ρ_out=1.0)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.apply_zou_he_pressure_east_2d!)
```
"""
function apply_zou_he_pressure_east_2d!(f, Nx, Ny; ρ_out=1.0)
    backend = KernelAbstractions.get_backend(f)
    kernel! = zou_he_pressure_east_2d_kernel!(backend)
    kernel!(f, Nx, eltype(f)(ρ_out); ndrange=(Ny,))
end

# --- Zou-He pressure inlet on west wall (i=1) ---
#
# D2Q9 direction convention used throughout this file (1-based index → (cx, cy)):
#   1:(0,0) 2:(+1,0) 3:(0,+1) 4:(-1,0) 5:(0,-1)
#   6:(+1,+1) 7:(-1,+1) 8:(-1,-1) 9:(+1,-1)
#
# At i = 1 the populations with cx = +1 (q = 2, 6, 9) have no upstream
# neighbour, so they are the unknowns; the mirror image of the east outlet,
# where the unknowns are the cx = -1 set (q = 4, 7, 8).
#
# With ρ prescribed and u_y = 0 the Zou-He closure gives, from
#   ρ      = (f1+f3+f5) + (f4+f7+f8) + (f2+f6+f9)
#   ρ·u_x  = (f2+f6+f9) - (f4+f7+f8)
# the normal velocity
#   u_x = 1 - (f1+f3+f5 + 2(f4+f7+f8)) / ρ_in
# (sign opposite to the east face, where the known set streams in from -x),
# then bounce-back of the non-equilibrium normal part, f2 - f2^eq = f4 - f4^eq
# with f2^eq - f4^eq = 2·w2·ρ·c_x·u_x/c_s² = (2/3)·ρ·u_x, and the two diagonal
# closures fixed by the transverse momentum balance f3+f6+f7 = f5+f8+f9
# (u_y = 0).

@kernel function zou_he_pressure_west_2d_kernel!(f, ρ_in)
    j = @index(Global)

    @inbounds begin
        T = eltype(f)
        # Known: f1, f3(+y), f5(-y) parallel; f4(-x), f7(-x,+y), f8(-x,-y) from interior
        f1 = f[1,j,1]; f3 = f[1,j,3]; f4 = f[1,j,4]
        f5 = f[1,j,5]; f7 = f[1,j,7]; f8 = f[1,j,8]

        # Compute ux from fixed ρ_in
        ux = one(T) - (f1 + f3 + f5 + T(2) * (f4 + f7 + f8)) / ρ_in

        # Unknown: f2(+x), f6(+x,+y), f9(+x,-y)
        f[1,j,2] = f4 + T(2.0/3.0) * ρ_in * ux
        f[1,j,6] = f8 - T(0.5) * (f3 - f5) + T(1.0/6.0) * ρ_in * ux
        f[1,j,9] = f7 + T(0.5) * (f3 - f5) + T(1.0/6.0) * ρ_in * ux
    end
end

"""
    apply_zou_he_pressure_west_2d!(f, Nx, Ny; ρ_in=1.0)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.apply_zou_he_pressure_west_2d!)
```
"""
function apply_zou_he_pressure_west_2d!(f, Nx, Ny; ρ_in=1.0)
    backend = KernelAbstractions.get_backend(f)
    kernel! = zou_he_pressure_west_2d_kernel!(backend)
    kernel!(f, eltype(f)(ρ_in); ndrange=(Ny,))
end

# --- Zero-gradient (Neumann) outflow on east wall ---

@kernel function extrapolate_east_2d_kernel!(f, Nx)
    j = @index(Global)

    @inbounds begin
        for q in 1:9
            f[Nx, j, q] = f[Nx-1, j, q]
        end
    end
end

"""
    apply_extrapolate_east_2d!(f, Nx, Ny)

Zero-gradient (Neumann) outflow BC on east wall: copy distributions from i=Nx-1 to i=Nx.
Suitable for two-phase flows where Zou-He pressure outlet creates density artifacts.
"""
function apply_extrapolate_east_2d!(f, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = extrapolate_east_2d_kernel!(backend)
    kernel!(f, Nx; ndrange=(Ny,))
end
