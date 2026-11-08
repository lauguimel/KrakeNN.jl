# Diagnose where exactly the Jacobian folds in extended ring_0 mesh
# with different curved_edges settings. Print ext coordinates around
# every ≤0-Jacobian cell.

using KrakeNN, KernelAbstractions, Gmsh

include(joinpath(@__DIR__, "gen_ogrid_rect_8block.jl"))

const T = Float64
const Lx, Ly = 1.0, 0.5
const cx_p, cy_p = 0.5, 0.245
const R_p = 0.025
const N_arc = 7; const N_radial = 20

function extend_manual(blk, n_ghost, curved_edges)
    # Minimal reimplementation of extend_mesh_2d (2-pass) that returns
    # only the X, Y extended arrays (no spline refit) so we can inspect
    # Jacobian without hitting validate_mesh.
    Nx = blk.mesh.Nξ; Ny = blk.mesh.Nη
    Nx_ext = Nx + 2n_ghost; Ny_ext = Ny + 2n_ghost
    X = zeros(T, Nx_ext, Ny_ext); Y = zeros(T, Nx_ext, Ny_ext)
    X[(n_ghost+1):(n_ghost+Nx), (n_ghost+1):(n_ghost+Ny)] .= blk.mesh.X
    Y[(n_ghost+1):(n_ghost+Nx), (n_ghost+1):(n_ghost+Ny)] .= blk.mesh.Y
    cx_c = cx_p; cy_c = cy_p
    polar_extrap(x1, y1, x2, y2, k) = begin
        r1 = sqrt((x1-cx_c)^2 + (y1-cy_c)^2); r2 = sqrt((x2-cx_c)^2 + (y2-cy_c)^2)
        θ1 = atan(y1-cy_c, x1-cx_c); θ2 = atan(y2-cy_c, x2-cx_c)
        Δθ = θ2 - θ1; Δθ > π && (Δθ -= 2π); Δθ < -π && (Δθ += 2π)
        r_g = r1 - k*(r2-r1); θ_g = θ1 - k*Δθ
        (cx_c + r_g*cos(θ_g), cy_c + r_g*sin(θ_g))
    end
    # Pass 1: W/E
    for j in 1:Ny
        j_ext = j + n_ghost
        if :west in curved_edges
            for k in 1:n_ghost
                xg,yg = polar_extrap(blk.mesh.X[1,j], blk.mesh.Y[1,j], blk.mesh.X[2,j], blk.mesh.Y[2,j], k)
                X[n_ghost-k+1, j_ext] = xg; Y[n_ghost-k+1, j_ext] = yg
            end
        else
            dX = blk.mesh.X[2,j] - blk.mesh.X[1,j]; dY = blk.mesh.Y[2,j] - blk.mesh.Y[1,j]
            for k in 1:n_ghost
                X[n_ghost-k+1, j_ext] = blk.mesh.X[1,j] - k*dX; Y[n_ghost-k+1, j_ext] = blk.mesh.Y[1,j] - k*dY
            end
        end
        if :east in curved_edges
            for k in 1:n_ghost
                xg,yg = polar_extrap(blk.mesh.X[Nx,j], blk.mesh.Y[Nx,j], blk.mesh.X[Nx-1,j], blk.mesh.Y[Nx-1,j], k)
                X[Nx+n_ghost+k, j_ext] = xg; Y[Nx+n_ghost+k, j_ext] = yg
            end
        else
            dX = blk.mesh.X[Nx,j] - blk.mesh.X[Nx-1,j]; dY = blk.mesh.Y[Nx,j] - blk.mesh.Y[Nx-1,j]
            for k in 1:n_ghost
                X[Nx+n_ghost+k, j_ext] = blk.mesh.X[Nx,j] + k*dX; Y[Nx+n_ghost+k, j_ext] = blk.mesh.Y[Nx,j] + k*dY
            end
        end
    end
    # Pass 2: S/N over full i range
    for i_ext in 1:Nx_ext
        j1 = n_ghost+1; j2 = n_ghost+2
        if :south in curved_edges
            for k in 1:n_ghost
                xg,yg = polar_extrap(X[i_ext,j1], Y[i_ext,j1], X[i_ext,j2], Y[i_ext,j2], k)
                X[i_ext, n_ghost-k+1] = xg; Y[i_ext, n_ghost-k+1] = yg
            end
        else
            dX = X[i_ext,j2] - X[i_ext,j1]; dY = Y[i_ext,j2] - Y[i_ext,j1]
            for k in 1:n_ghost
                X[i_ext, n_ghost-k+1] = X[i_ext,j1] - k*dX; Y[i_ext, n_ghost-k+1] = Y[i_ext,j1] - k*dY
            end
        end
        jN = Ny+n_ghost; jNm = jN-1
        if :north in curved_edges
            for k in 1:n_ghost
                xg,yg = polar_extrap(X[i_ext,jN], Y[i_ext,jN], X[i_ext,jNm], Y[i_ext,jNm], k)
                X[i_ext, jN+k] = xg; Y[i_ext, jN+k] = yg
            end
        else
            dX = X[i_ext,jN] - X[i_ext,jNm]; dY = Y[i_ext,jN] - Y[i_ext,jNm]
            for k in 1:n_ghost
                X[i_ext, jN+k] = X[i_ext,jN] + k*dX; Y[i_ext, jN+k] = Y[i_ext,jN] + k*dY
            end
        end
    end
    return X, Y
end

function count_neg_J(X, Y)
    Nx_ext, Ny_ext = size(X)
    neg = 0; first_loc = (0, 0, 0.0)
    for j in 2:Ny_ext-1, i in 2:Nx_ext-1
        dXdξ = (X[i+1,j] - X[i-1,j])/2; dYdξ = (Y[i+1,j] - Y[i-1,j])/2
        dXdη = (X[i,j+1] - X[i,j-1])/2; dYdη = (Y[i,j+1] - Y[i,j-1])/2
        J = dXdξ*dYdη - dXdη*dYdξ
        if J ≤ 0
            neg += 1
            neg == 1 && (first_loc = (i, j, J))
        end
    end
    return neg, first_loc
end

mktempdir() do dir
    geo_path = joinpath(dir, "ogrid.geo")
    write_ogrid_rect_8block_geo(geo_path; Lx=Lx, Ly=Ly, cx_p=cx_p, cy_p=cy_p, R_in=R_p,
                                  N_arc=N_arc, N_radial=N_radial)
    mbm_raw, _ = load_gmsh_multiblock_2d(geo_path; FT=T, layout=:topological)
    mbm = autoreorient_blocks(mbm_raw; verbose=false)
    blk = mbm.blocks[1]  # ring_0

    # (autoreorient already ensures root has J>0)
    println("Block :ring_0 after autoreorient: Nξ=$(blk.mesh.Nξ) Nη=$(blk.mesh.Nη)  corners:")
    Nx, Ny = blk.mesh.Nξ, blk.mesh.Nη
    for (nm, ix, iy) in [("SW", 1, 1), ("SE", Nx, 1), ("NW", 1, Ny), ("NE", Nx, Ny)]
        println("  $nm (i=$ix, j=$iy): ($(round(blk.mesh.X[ix, iy], digits=4)), $(round(blk.mesh.Y[ix, iy], digits=4)))")
    end
    for curved in [(), (:south,), (:south, :west, :east, :north)]
        X, Y = extend_manual(blk, 1, curved)
        neg, loc = count_neg_J(X, Y)
        i, j, J = loc
        print("curved=$curved  → neg J cells: $neg")
        if neg > 0
            println("  first at (i=$i, j=$j) J=$(round(J, sigdigits=3))")
            println("    ext[i=$i, j=$j] = ($(round(X[i,j],digits=4)), $(round(Y[i,j],digits=4)))")
            println("    ext[i=$(i-1), j=$j] = ($(round(X[i-1,j],digits=4)), $(round(Y[i-1,j],digits=4)))")
            println("    ext[i=$(i+1), j=$j] = ($(round(X[i+1,j],digits=4)), $(round(Y[i+1,j],digits=4)))")
            println("    ext[i=$i, j=$(j-1)] = ($(round(X[i,j-1],digits=4)), $(round(Y[i,j-1],digits=4)))")
            println("    ext[i=$i, j=$(j+1)] = ($(round(X[i,j+1],digits=4)), $(round(Y[i,j+1],digits=4)))")
        else
            println("  ✅")
        end
    end
end
