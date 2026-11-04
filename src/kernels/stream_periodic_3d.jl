using KernelAbstractions

# --- 3D periodic-x/z streaming with bounce-back walls at y boundaries ---
#
# Pull-scheme D3Q19 streamer for confined shear/channel flows whose flow and
# neutral directions (x and z) are periodic while the two walls live on the
# y-faces (j = 1 and j = Ny). It is the 3D analogue of
# `stream_periodic_x_wall_y_2d!` and is reusable for:
#   - planar Couette  (y-walls overwritten by moving-wall Zou-He afterwards),
#   - Poiseuille      (y-walls are the no-slip bounce-back, body force in x).
#
# The bounce-back at the y-faces is the standard no-slip half-way BB; for
# Couette it is harmless because the moving-wall Zou-He rebuild overwrites the
# wall-incident unknowns post-stream. x and z are fully periodic (wrap), so the
# solvent f is genuinely homogeneous in the flow/neutral plane — unlike the
# plain `stream_3d!` which bounce-backs all six faces.
#
# D3Q19 indexing (matches src/lattice/d3q19.jl):
#   1:(0,0,0) 2:(+x) 3:(-x) 4:(+y) 5:(-y) 6:(+z) 7:(-z)
#   8:(+x,+y) 9:(-x,+y) 10:(+x,-y) 11:(-x,-y)
#  12:(+x,+z) 13:(-x,+z) 14:(+x,-z) 15:(-x,-z)
#  16:(+y,+z) 17:(-y,+z) 18:(+y,-z) 19:(-y,-z)

@kernel function stream_periodic_xz_wall_y_3d_kernel!(f_out, @Const(f_in),
                                                       Nx, Ny, Nz)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Periodic wrap in x and z (pull from node − c_q).
        im = ifelse(i > 1,  i - 1, Nx)   # source for +x populations
        ip = ifelse(i < Nx, i + 1, 1)    # source for -x populations
        km = ifelse(k > 1,  k - 1, Nz)   # source for +z populations
        kp = ifelse(k < Nz, k + 1, 1)    # source for -z populations

        # y has physical walls: pulling from j-1 (for +y pops) is invalid at
        # j=1 → bounce-back; pulling from j+1 (for -y pops) invalid at j=Ny.
        at_lo = j == 1
        at_hi = j == Ny
        # Wall-normal neighbour indices clamped in range: `ifelse` is a plain
        # function, so BOTH branches are evaluated and the unselected one must
        # never index at j = 0 or j = Ny + 1 (out of bounds under @inbounds).
        jm1 = ifelse(j > 1,  j - 1, j)
        jp1 = ifelse(j < Ny, j + 1, j)

        f_out[i,j,k,1] = f_in[i, j, k, 1]   # rest

        # Axis x (periodic)
        f_out[i,j,k,2] = f_in[im, j, k, 2]
        f_out[i,j,k,3] = f_in[ip, j, k, 3]
        # Axis y (wall bounce-back)
        f_out[i,j,k,4] = ifelse(at_lo, f_in[i, j, k, 5], f_in[i, jm1, k, 4])
        f_out[i,j,k,5] = ifelse(at_hi, f_in[i, j, k, 4], f_in[i, jp1, k, 5])
        # Axis z (periodic)
        f_out[i,j,k,6] = f_in[i, j, km, 6]
        f_out[i,j,k,7] = f_in[i, j, kp, 7]

        # xy edges
        f_out[i,j,k,8]  = ifelse(at_lo, f_in[i, j, k, 11], f_in[im, jm1, k, 8])
        f_out[i,j,k,9]  = ifelse(at_lo, f_in[i, j, k, 10], f_in[ip, jm1, k, 9])
        f_out[i,j,k,10] = ifelse(at_hi, f_in[i, j, k, 9],  f_in[im, jp1, k, 10])
        f_out[i,j,k,11] = ifelse(at_hi, f_in[i, j, k, 8],  f_in[ip, jp1, k, 11])

        # xz edges (both periodic — no wall crossing)
        f_out[i,j,k,12] = f_in[im, j, km, 12]
        f_out[i,j,k,13] = f_in[ip, j, km, 13]
        f_out[i,j,k,14] = f_in[im, j, kp, 14]
        f_out[i,j,k,15] = f_in[ip, j, kp, 15]

        # yz edges (z periodic, y wall)
        f_out[i,j,k,16] = ifelse(at_lo, f_in[i, j, k, 19], f_in[i, jm1, km, 16])
        f_out[i,j,k,17] = ifelse(at_hi, f_in[i, j, k, 18], f_in[i, jp1, km, 17])
        f_out[i,j,k,18] = ifelse(at_lo, f_in[i, j, k, 17], f_in[i, jm1, kp, 18])
        f_out[i,j,k,19] = ifelse(at_hi, f_in[i, j, k, 16], f_in[i, jp1, kp, 19])
    end
end

"""
    stream_periodic_xz_wall_y_3d!(f_out, f_in, Nx, Ny, Nz)

3D D3Q19 pull-stream with **periodic x and z** and **no-slip bounce-back walls
at the y-faces** (j = 1 and j = Ny). The 3D analogue of
`stream_periodic_x_wall_y_2d!`. Use for confined shear / channel flows where
the flow direction (x) and the neutral direction (z) are homogeneous; the two
walls live on the y-faces. For moving-wall Couette, follow this stream with the
y-face Zou-He rebuild (which overwrites the wall unknowns); for Poiseuille the
y bounce-back is the no-slip wall and the drive is a body force in x.
"""
function stream_periodic_xz_wall_y_3d!(f_out, f_in, Nx, Ny, Nz)
    backend = KernelAbstractions.get_backend(f_in)
    kernel! = stream_periodic_xz_wall_y_3d_kernel!(backend)
    kernel!(f_out, f_in, Nx, Ny, Nz; ndrange=(Nx, Ny, Nz))
    KernelAbstractions.synchronize(backend)
end

# --- Moving-wall (Ladd) variant: periodic x/z, tangentially moving y-walls ---
#
# Same pull-stream as above but the y-face bounce-back carries the standard Ladd
# momentum correction for a wall moving tangentially in x:
#   f_q̄ = f_q − 2 w_q ρ_w (c_q · u_w) / cs²   (cs² = 1/3 ⇒ −6 w_q ρ_w c_qx u_w).
# The wall is the half-way no-slip plane offset by ½ cell; this imposes the wall
# velocity cleanly without the Zou-He node-velocity overshoot, giving a uniform
# shear rate γ̇ = (U_top − U_bot) / Ny across the gap. `Ub`/`Ut` are the bottom
# (j=1) and top (j=Ny) wall x-velocities. `rho_w` is the reference wall density
# (use 1 for the incompressible reference).

@kernel function stream_periodic_xz_movingwall_y_3d_kernel!(f_out, @Const(f_in),
                                                             Ub, Ut, rho_w,
                                                             Nx, Ny, Nz)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        T = eltype(f_out)
        im = ifelse(i > 1,  i - 1, Nx)
        ip = ifelse(i < Nx, i + 1, 1)
        km = ifelse(k > 1,  k - 1, Nz)
        kp = ifelse(k < Nz, k + 1, 1)

        at_lo = j == 1
        at_hi = j == Ny
        # Wall-normal neighbour indices clamped in range: `ifelse` is a plain
        # function, so BOTH branches are evaluated and the unselected one must
        # never index at j = 0 or j = Ny + 1 (out of bounds under @inbounds).
        jm1 = ifelse(j > 1,  j - 1, j)
        jp1 = ifelse(j < Ny, j + 1, j)

        # Ladd correction magnitude for an edge population (w_e = 1/36):
        #   2 w_e / cs² · ρ_w · u_w = 6 w_e ρ_w u_w = (1/6) ρ_w u_w.
        cbot = T(1/6) * rho_w * Ub
        ctop = T(1/6) * rho_w * Ut

        f_out[i,j,k,1] = f_in[i, j, k, 1]

        # Axis x (periodic)
        f_out[i,j,k,2] = f_in[im, j, k, 2]
        f_out[i,j,k,3] = f_in[ip, j, k, 3]
        # Axis y (wall bounce-back; axial pops have c_qx=0 → no Ladd term)
        f_out[i,j,k,4] = ifelse(at_lo, f_in[i, j, k, 5], f_in[i, jm1, k, 4])
        f_out[i,j,k,5] = ifelse(at_hi, f_in[i, j, k, 4], f_in[i, jp1, k, 5])
        # Axis z (periodic)
        f_out[i,j,k,6] = f_in[i, j, km, 6]
        f_out[i,j,k,7] = f_in[i, j, kp, 7]

        # xy edges — these carry the tangential (x) Ladd momentum at the walls.
        # Bottom wall (j=1): reflect +y unknowns 8 (+x,+y) and 9 (−x,+y).
        #   The reflected population gains +2 w_q ρ_w (c_q·u_w)/cs²; with the
        #   half-way reflection the sign that yields a co-moving fluid is:
        #   q8 (c_qx=+1): + ; q9 (c_qx=−1): − .
        f_out[i,j,k,8]  = ifelse(at_lo, f_in[i, j, k, 11] + cbot, f_in[im, jm1, k, 8])
        f_out[i,j,k,9]  = ifelse(at_lo, f_in[i, j, k, 10] - cbot, f_in[ip, jm1, k, 9])
        # Top wall (j=Ny): reflect −y unknowns 10 (+x,−y) and 11 (−x,−y).
        f_out[i,j,k,10] = ifelse(at_hi, f_in[i, j, k, 9]  + ctop, f_in[im, jp1, k, 10])
        f_out[i,j,k,11] = ifelse(at_hi, f_in[i, j, k, 8]  - ctop, f_in[ip, jp1, k, 11])

        # xz edges (both periodic — no wall crossing)
        f_out[i,j,k,12] = f_in[im, j, km, 12]
        f_out[i,j,k,13] = f_in[ip, j, km, 13]
        f_out[i,j,k,14] = f_in[im, j, kp, 14]
        f_out[i,j,k,15] = f_in[ip, j, kp, 15]

        # yz edges (z periodic, y wall; c_qx=0 → no tangential-x Ladd term)
        f_out[i,j,k,16] = ifelse(at_lo, f_in[i, j, k, 19], f_in[i, jm1, km, 16])
        f_out[i,j,k,17] = ifelse(at_hi, f_in[i, j, k, 18], f_in[i, jp1, km, 17])
        f_out[i,j,k,18] = ifelse(at_lo, f_in[i, j, k, 17], f_in[i, jm1, kp, 18])
        f_out[i,j,k,19] = ifelse(at_hi, f_in[i, j, k, 16], f_in[i, jp1, kp, 19])
    end
end

"""
    stream_periodic_xz_movingwall_y_3d!(f_out, f_in, Ub, Ut, Nx, Ny, Nz; rho_w=1)

3D D3Q19 pull-stream, periodic in x and z, with **tangentially moving no-slip
walls** on the y-faces (bottom j=1 velocity `(Ub,0,0)`, top j=Ny velocity
`(Ut,0,0)`). Uses the half-way bounce-back with the Ladd momentum correction so
the wall x-velocity is imposed cleanly — yielding a uniform shear rate
γ̇ = (Ut − Ub) / Ny without the Zou-He node-velocity overshoot. The moving-wall
analogue of `stream_periodic_xz_wall_y_3d!`; pairs with planar-Couette drivers.
"""
function stream_periodic_xz_movingwall_y_3d!(f_out, f_in, Ub, Ut, Nx, Ny, Nz;
                                              rho_w=1)
    backend = KernelAbstractions.get_backend(f_in)
    T = eltype(f_in)
    kernel! = stream_periodic_xz_movingwall_y_3d_kernel!(backend)
    kernel!(f_out, f_in, T(Ub), T(Ut), T(rho_w), Nx, Ny, Nz;
            ndrange=(Nx, Ny, Nz))
    KernelAbstractions.synchronize(backend)
end
