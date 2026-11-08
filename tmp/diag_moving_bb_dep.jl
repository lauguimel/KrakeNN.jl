# Moving wall departure-aware BB — ghost strategy comparison
# Goal: achieve machine precision for Couette on oblique mesh
using KrakeNN, LinearAlgebra, Printf

const cx_ = [0,1,0,-1,0,1,-1,-1,1]
const cy_ = [0,0,1,0,-1,1,1,-1,-1]
const w_  = [4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]
const opp_ = [1,4,5,2,3,8,9,6,7]

_fmt(x) = @sprintf("%.3e", Float64(x))

function feq_q(q, ρ, ux, uy)
    cu = cx_[q]*ux + cy_[q]*uy
    w_[q] * ρ * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*(ux*ux + uy*uy))
end

function feq_q_linear(q, ρ, ux, uy)
    cu = cx_[q]*ux + cy_[q]*uy
    w_[q] * ρ * (1.0 + 3.0*cu)
end

_feq_q(equilibrium::Symbol, q, ρ, ux, uy) =
    equilibrium == :linear ? feq_q_linear(q, ρ, ux, uy) : feq_q(q, ρ, ux, uy)

function ghost_fill_f_linear!(f, Nxe, Nye, ng, Nx, Ny)
    for g in 1:ng
        for i in 1:Nxe, q in 1:9
            f[i, ng+1-g, q] = f[i, ng+1, q] + g*(f[i, ng+1, q] - f[i, ng+2, q])
            f[i, ng+Ny+g, q] = f[i, ng+Ny, q] + g*(f[i, ng+Ny, q] - f[i, ng+Ny-1, q])
        end
        for j in 1:Nye, q in 1:9
            f[ng+1-g, j, q] = f[ng+1, j, q] + g*(f[ng+1, j, q] - f[ng+2, j, q])
            f[ng+Nx+g, j, q] = f[ng+Nx, j, q] + g*(f[ng+Nx, j, q] - f[ng+Nx-1, j, q])
        end
    end
end

function _moments(f, i, j)
    ρ = sum(f[i,j,q] for q in 1:9)
    ux = (f[i,j,2]-f[i,j,4]+f[i,j,6]-f[i,j,7]-f[i,j,8]+f[i,j,9]) / ρ
    uy = (f[i,j,3]-f[i,j,5]+f[i,j,6]+f[i,j,7]-f[i,j,8]-f[i,j,9]) / ρ
    return ρ, ux, uy
end

function _fill_ghost_edge!(f, ib, ib2, ig, Nother, dim, fneq_mode)
    for idx in 1:Nother
        i1, j1 = dim == :y ? (idx, ib) : (ib, idx)
        i2, j2 = dim == :y ? (idx, ib2) : (ib2, idx)
        ig_, jg_ = dim == :y ? (idx, ig) : (ig, idx)
        ρ1, ux1, uy1 = _moments(f, i1, j1)
        ρ2, ux2, uy2 = _moments(f, i2, j2)
        # feq part: extrapolate moments linearly, compute feq
        g_dist = abs(ig - ib)  # ghost layer index (1, 2, ...)
        ρg  = ρ1  + g_dist*(ρ1  - ρ2)
        uxg = ux1 + g_dist*(ux1 - ux2)
        uyg = uy1 + g_dist*(uy1 - uy2)
        # fneq part from boundary
        for q in 1:9
            feq1 = feq_q(q, ρ1, ux1, uy1)
            fneq1 = f[i1,j1,q] - feq1
            if fneq_mode == :zeroth
                fneq_g = fneq1
            else  # :linear
                feq2 = feq_q(q, ρ2, ux2, uy2)
                fneq2 = f[i2,j2,q] - feq2
                fneq_g = fneq1 + g_dist*(fneq1 - fneq2)
            end
            f[ig_,jg_,q] = feq_q(q, ρg, uxg, uyg) + fneq_g
        end
    end
end

function ghost_fill_hybrid!(f, Nxe, Nye, ng, Nx, Ny; fneq_mode=:zeroth)
    for g in 1:ng
        _fill_ghost_edge!(f, ng+1, ng+2, ng+1-g, Nxe, :y, fneq_mode)        # south
        _fill_ghost_edge!(f, ng+Ny, ng+Ny-1, ng+Ny+g, Nxe, :y, fneq_mode)   # north
        _fill_ghost_edge!(f, ng+1, ng+2, ng+1-g, Nye, :x, fneq_mode)         # west
        _fill_ghost_edge!(f, ng+Nx, ng+Nx-1, ng+Nx+g, Nye, :x, fneq_mode)    # east
    end
end

function _zou_he_rho(fp, bb_set, non_bb_set, ux_prescribed, uy_prescribed)
    # Compute self-consistent ρ from non-BB populations + prescribed velocity
    # For the standard sets (all BB have same cy or same cx), Zou-He gives exact ρ.
    # General case: use the uy=0 constraint to eliminate one unknown.
    cy_bb = [cy_[q] for q in bb_set]
    if all(c == cy_bb[1] for c in cy_bb)
        # Degenerate case: all BB have same cy (e.g., {S, SW, SE} all cy=-1)
        # Zou-He: ρ = (tangential pops + 2*outgoing pops) / (1 + uy_prescribed)
        cy_val = cy_bb[1]
        outgoing = [q for q in non_bb_set if cy_[q] == -cy_val]
        tangential = [q for q in non_bb_set if cy_[q] == 0]
        ρ = (sum(fp[q] for q in tangential; init=0.0) +
             2.0 * sum(fp[q] for q in outgoing; init=0.0)) / (1.0 + uy_prescribed)
        return ρ
    end
    cx_bb = [cx_[q] for q in bb_set]
    if all(c == cx_bb[1] for c in cx_bb)
        cx_val = cx_bb[1]
        outgoing = [q for q in non_bb_set if cx_[q] == -cx_val]
        tangential = [q for q in non_bb_set if cx_[q] == 0]
        ρ = (sum(fp[q] for q in tangential; init=0.0) +
             2.0 * sum(fp[q] for q in outgoing; init=0.0)) / (1.0 + ux_prescribed)
        return ρ
    end
    # Non-degenerate: solve 3×3 system. Use total ρ from all pops as approximation.
    return sum(fp)
end

function ghost_fill_hybrid_reg!(f, Nxe, Nye, ng, Nx, Ny; omega=1.0)
    tau = 1.0 / omega
    for g in 1:ng
        # South wall
        for i in 1:Nxe
            jb = ng+1; jb2 = ng+2; jg = ng+1-g
            ρ1, ux1, uy1 = _moments(f, i, jb)
            ρ2, ux2, uy2 = _moments(f, i, jb2)
            ρg = ρ1 + g*(ρ1 - ρ2)
            uxg = ux1 + g*(ux1 - ux2)
            uyg = uy1 + g*(uy1 - uy2)
            S_xy = (ux1 - ux2)  # strain rate from boundary gradient
            for q in 1:9
                Q_xy = cx_[q]*cy_[q]
                fneq_reg = -(tau - 0.5) * w_[q] * 2.0 * Q_xy * S_xy * 9.0
                f[i,jg,q] = feq_q(q, ρg, uxg, uyg) + fneq_reg
            end
        end
        # North wall
        for i in 1:Nxe
            jb = ng+Ny; jb2 = ng+Ny-1; jg = ng+Ny+g
            ρ1, ux1, uy1 = _moments(f, i, jb)
            ρ2, ux2, uy2 = _moments(f, i, jb2)
            ρg = ρ1 + g*(ρ1 - ρ2)
            uxg = ux1 + g*(ux1 - ux2)
            uyg = uy1 + g*(uy1 - uy2)
            S_xy = (ux1 - ux2)
            for q in 1:9
                Q_xy = cx_[q]*cy_[q]
                fneq_reg = -(tau - 0.5) * w_[q] * 2.0 * Q_xy * S_xy * 9.0
                f[i,jg,q] = feq_q(q, ρg, uxg, uyg) + fneq_reg
            end
        end
        # West/East — zeroth-order for simplicity
        for j in 1:Nye
            ig_w = ng+1-g; ig_e = ng+Nx+g
            for q in 1:9
                f[ig_w,j,q] = f[ng+1,j,q]
                f[ig_e,j,q] = f[ng+Nx,j,q]
            end
        end
    end
end

function ghost_fill_feq_extrap!(f, Nxe, Nye, ng, Nx, Ny)
    ghost_fill_hybrid!(f, Nxe, Nye, ng, Nx, Ny; fneq_mode=:zeroth)
    # override: set fneq=0 in ghost (pure feq)
    for g in 1:ng
        for i in 1:Nxe
            jg_s = ng+1-g; jg_n = ng+Ny+g
            ρs, uxs, uys = _moments(f, i, jg_s)
            ρn, uxn, uyn = _moments(f, i, jg_n)
            for q in 1:9
                f[i, jg_s, q] = feq_q(q, ρs, uxs, uys)
                f[i, jg_n, q] = feq_q(q, ρn, uxn, uyn)
            end
        end
        for j in 1:Nye
            ig_w = ng+1-g; ig_e = ng+Nx+g
            ρw, uxw, uyw = _moments(f, ig_w, j)
            ρe, uxe, uye = _moments(f, ig_e, j)
            for q in 1:9
                f[ig_w, j, q] = feq_q(q, ρw, uxw, uyw)
                f[ig_e, j, q] = feq_q(q, ρe, uxe, uye)
            end
        end
    end
end

function test_moving_wall(angle; ghost_mode=:f_linear, steps=200, interp=:biquadratic,
                          Ny=5, u_w=0.01, omega=1.0, bb_mode=:ladd)
    Nx = Ny; ng = 2; ω = omega
    Nxe = Nx + 2*ng; Nye = Ny + 2*ng
    u_couette(j) = u_w * (j - 1) / (Ny - 1)
    dirs = ["rest","E","N","W","S","NE","NW","SW","SE"]

    X = zeros(Nx, Ny); Y = zeros(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        X[i,j] = Float64(i-1)
        Y[i,j] = Float64(j-1) + Float64(i-1)*tand(angle)
    end
    mesh = CurvilinearMesh(X, Y; periodic_ξ=false, periodic_η=false,
                           type=:curvilinear, dx_ref=1.0, skip_validate=true, FT=Float64)
    mesh_ext = KrakeNN.extend_mesh_2d(mesh; n_ghost=ng)
    geom = build_slbm_geometry(mesh_ext; local_cfl=false, dx_ref=1.0)
    is_sol = zeros(Bool, Nxe, Nye)
    j_s = ng + 1; j_n = ng + Ny; ic = ng + div(Nx,2) + 1

    # Departure-aware BB sets
    bb_south = Int[]; bb_north = Int[]
    for q in 2:9
        if geom.j_dep[ic, j_s, q] < j_s - 0.5; push!(bb_south, q); end
        if geom.j_dep[ic, j_n, q] > j_n + 0.5; push!(bb_north, q); end
    end

    # Init with exact Couette feq
    f = zeros(Nxe, Nye, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        f[ng+i, ng+j, q] = feq_q(q, 1.0, u_couette(j), 0.0)
    end

    gfill! = if ghost_mode == :feq_extrap
        ghost_fill_feq_extrap!
    elseif ghost_mode == :hybrid_reg
        (f_, args...) -> ghost_fill_hybrid_reg!(f_, args...; omega=ω)
    elseif ghost_mode == :hybrid_z
        (f_, args...) -> ghost_fill_hybrid!(f_, args...; fneq_mode=:zeroth)
    elseif ghost_mode == :hybrid_l
        (f_, args...) -> ghost_fill_hybrid!(f_, args...; fneq_mode=:linear)
    else
        ghost_fill_f_linear!
    end
    gfill!(f, Nxe, Nye, ng, Nx, Ny)

    interp_fn = interp == :biquadratic ?
        (f_, id, jd, q_) -> KrakeNN.biquadratic_f(f_, is_sol, id, jd, q_, Nxe, Nye, false, false) :
        (f_, id, jd, q_) -> KrakeNN.bilinear_f(f_, id, jd, q_, Nxe, Nye, false, false)

    nan_step = 0
    for it in 1:steps
        gfill!(f, Nxe, Nye, ng, Nx, Ny)

        fo = zeros(Nxe, Nye, 9)

        if bb_mode == :prescribe
            # ── Prescribed-moment collision at walls ──
            # Interior: standard SLBM + collision
            for je in 1:Nye, ie in 1:Nxe
                fp = [interp_fn(f, geom.i_dep[ie,je,q], geom.j_dep[ie,je,q], q) for q in 1:9]
                if je == j_s
                    # South wall: prescribe u = 0
                    # Zou-He ρ from non-BB pops (bb_south has cy > 0 component)
                    non_bb_s = [q for q in 1:9 if !(q in bb_south)]
                    ρw = _zou_he_rho(fp, bb_south, non_bb_s, 0.0, 0.0)
                    for q in 1:9; fo[ie,je,q] = feq_q(q, ρw, 0.0, 0.0); end
                elseif je == j_n
                    # North wall: prescribe u = (u_w, 0)
                    non_bb_n = [q for q in 1:9 if !(q in bb_north)]
                    ρw = _zou_he_rho(fp, bb_north, non_bb_n, u_w, 0.0)
                    for q in 1:9; fo[ie,je,q] = feq_q(q, ρw, u_w, 0.0); end
                else
                    ρ = sum(fp)
                    ux = (fp[2]-fp[4]+fp[6]-fp[7]-fp[8]+fp[9]) / ρ
                    uy = (fp[3]-fp[5]+fp[6]+fp[7]-fp[8]-fp[9]) / ρ
                    for q in 1:9
                        fo[ie,je,q] = fp[q] - ω * (fp[q] - feq_q(q, ρ, ux, uy))
                    end
                end
            end
            # BB at south (stationary)
            for i in 1:Nxe
                for q in bb_south; fo[i, j_s, q] = fo[i, j_s, opp_[q]]; end
            end
            # BB at north (moving) with Ladd
            for i in 1:Nxe
                ρw = sum(fo[i, j_n, q] for q in 1:9)
                for q in bb_north
                    qo = opp_[q]
                    fo[i, j_n, q] = fo[i, j_n, qo] - 2.0*w_[qo]*ρw*(cx_[qo]*u_w)*3.0
                end
            end
        else
            # ── Standard collision everywhere ──
            for je in 1:Nye, ie in 1:Nxe
                fp = [interp_fn(f, geom.i_dep[ie,je,q], geom.j_dep[ie,je,q], q) for q in 1:9]
                ρ = sum(fp)
                ux = (fp[2]-fp[4]+fp[6]-fp[7]-fp[8]+fp[9]) / ρ
                uy = (fp[3]-fp[5]+fp[6]+fp[7]-fp[8]-fp[9]) / ρ
                for q in 1:9
                    fo[ie,je,q] = fp[q] - ω * (fp[q] - feq_q(q, ρ, ux, uy))
                end
            end

            # South wall: stationary — departure-aware BB
            for i in 1:Nxe
                for q in bb_south; fo[i, j_s, q] = fo[i, j_s, opp_[q]]; end
            end
        end # bb_mode dispatch for collision

        # North wall: moving — departure-aware BB (for non-prescribe modes)
        if bb_mode == :wall_override
            # Override ALL wall pops: non-BB get feq(u_w), BB get Ladd from feq(u_w)
            for i in 1:Nxe
                # Set all pops to exact equilibrium at wall velocity
                for q in 1:9; fo[i, j_n, q] = feq_q(q, 1.0, u_w, 0.0); end
                # BB pops: Ladd correction on top of feq_opp
                for q in bb_north
                    qo = opp_[q]
                    fo[i, j_n, q] = fo[i, j_n, qo] - 2.0*w_[qo]*1.0*(cx_[qo]*u_w)*3.0
                end
            end
        elseif bb_mode == :ladd
            for i in 1:Nxe
                ρw = sum(fo[i, j_n, q] for q in 1:9)
                for q in bb_north
                    qo = opp_[q]
                    fo[i, j_n, q] = fo[i, j_n, qo] - 2.0*w_[qo]*ρw*(cx_[qo]*u_w)*3.0
                end
            end
        elseif bb_mode == :ladd_reg
            # Standard Ladd BB followed by regularization at wall
            for i in 1:Nxe
                ρw = sum(fo[i, j_n, q] for q in 1:9)
                for q in bb_north
                    qo = opp_[q]
                    fo[i, j_n, q] = fo[i, j_n, qo] - 2.0*w_[qo]*ρw*(cx_[qo]*u_w)*3.0
                end
            end
            # Regularize: replace ALL wall pops with feq + fneq_reg(strain rate)
            for i in 1:Nxe
                ρw = sum(fo[i, j_n, q] for q in 1:9)
                uxw = (fo[i,j_n,2]-fo[i,j_n,4]+fo[i,j_n,6]-fo[i,j_n,7]-fo[i,j_n,8]+fo[i,j_n,9]) / ρw
                uyw = (fo[i,j_n,3]-fo[i,j_n,5]+fo[i,j_n,6]+fo[i,j_n,7]-fo[i,j_n,8]-fo[i,j_n,9]) / ρw
                # Strain rate from interior: S_xy = (ux_wall - ux_interior) / Δη
                ρi = sum(fo[i, j_n-1, q] for q in 1:9)
                uxi = (fo[i,j_n-1,2]-fo[i,j_n-1,4]+fo[i,j_n-1,6]-fo[i,j_n-1,7]-fo[i,j_n-1,8]+fo[i,j_n-1,9]) / ρi
                uyi = (fo[i,j_n-1,3]-fo[i,j_n-1,5]+fo[i,j_n-1,6]+fo[i,j_n-1,7]-fo[i,j_n-1,8]-fo[i,j_n-1,9]) / ρi
                S_xy = (uxw - uxi)  # Δη = 1 in comp space
                S_xx = 0.0; S_yy = 0.0  # Couette: no normal strain
                tau = 1.0 / ω
                for q in 1:9
                    # fneq_reg = -(τ - 0.5) * w_q * Q_αβ * S_αβ / cs^4
                    # Q_αβ = c_qα c_qβ - cs² δ_αβ
                    # For D2Q9, cs² = 1/3, cs^4 = 1/9
                    Q_xx = cx_[q]^2 - 1/3; Q_yy = cy_[q]^2 - 1/3; Q_xy = cx_[q]*cy_[q]
                    Pi_neq = Q_xx*S_xx + Q_yy*S_yy + 2*Q_xy*S_xy
                    fneq_reg = -(tau - 0.5) * w_[q] * Pi_neq * 9.0  # 9 = 1/cs^4
                    fo[i, j_n, q] = feq_q(q, ρw, uxw, uyw) + fneq_reg
                end
            end
        elseif bb_mode in (:da_grad, :da_qw, :da_full)
            # Departure-aware moving wall BB with geometric corrections
            for i in 1:Nxe
                ρw = sum(fo[i, j_n, q] for q in 1:9)
                for q in bb_north
                    qo = opp_[q]
                    # Departure distance into ghost
                    Δη = geom.j_dep[i, j_n, q] - j_n  # > 0.5 for bb_north
                    q_w = 0.5 / Δη
                    # Mirror offset: δj = j_mirror - j_n = 1 - Δη
                    δj = 1.0 - Δη

                    # 1. Gradient correction on reflected population
                    if bb_mode in (:da_grad, :da_full)
                        grad_opp = fo[i, j_n, qo] - fo[i, j_n - 1, qo]
                        f_refl = fo[i, j_n, qo] + δj * grad_opp
                    else
                        f_refl = fo[i, j_n, qo]
                    end

                    # 2. Momentum correction scaled by q_w
                    if bb_mode in (:da_qw, :da_full)
                        corr_factor = q_w <= 0.5 ? 2.0 : 1.0 / q_w
                    else
                        corr_factor = 2.0
                    end
                    momentum = corr_factor * w_[qo] * ρw * (cx_[qo] * u_w) * 3.0

                    fo[i, j_n, q] = f_refl - momentum
                end
            end
        elseif bb_mode == :zou_he_da
            # Enforce (u_w, 0) at north wall using departure-aware unknown set
            # Build coefficient matrix once
            A_mat = zeros(3, length(bb_north))
            for (col, q) in enumerate(bb_north)
                A_mat[1, col] = 1.0
                A_mat[2, col] = Float64(cx_[q])
                A_mat[3, col] = Float64(cy_[q])
            end
            is_singular = abs(det(A_mat)) < 1e-10
            known_set = [q for q in 1:9 if !(q in bb_north)]

            for i in 1:Nxe
                f_known_ρ  = sum(fo[i, j_n, q] for q in known_set)
                f_known_jx = sum(cx_[q]*fo[i, j_n, q] for q in known_set)
                f_known_jy = sum(cy_[q]*fo[i, j_n, q] for q in known_set)

                if is_singular
                    # Standard Zou-He for north wall: all BB pops have same cy
                    # Self-consistent ρ from outgoing + tangential pops
                    cy_bb = cy_[bb_north[1]]  # all same
                    # outgoing = known pops with cy opposite to bb direction
                    outgoing = [q for q in known_set if cy_[q] == -cy_bb]
                    tangential = [q for q in known_set if cy_[q] == 0]
                    ρw = (sum(fo[i,j_n,q] for q in tangential) +
                          2*sum(fo[i,j_n,q] for q in outgoing)) / (1.0 + 0.0)
                    # ZH decomposition: f_q = f_opp(q) + correction
                    for q in bb_north
                        qo = opp_[q]
                        fo[i, j_n, q] = fo[i, j_n, qo]
                    end
                    # Add velocity correction via non-eq bounce-back
                    # x-momentum deficit: need ρw*u_w, currently have f_known_jx + Σ_bb cx[q]*f_opp(q)
                    jx_from_bb = sum(cx_[q]*fo[i,j_n,q] for q in bb_north)
                    deficit_x = ρw * u_w - f_known_jx - jx_from_bb
                    # Distribute deficit among BB pops proportional to cx[q]*w[q]
                    norm = sum(cx_[q]^2 * w_[q] for q in bb_north)
                    if abs(norm) > 1e-15
                        for q in bb_north
                            fo[i, j_n, q] += cx_[q] * w_[q] / norm * deficit_x
                        end
                    end
                else
                    # Non-singular: solve 3×3 directly
                    # Self-consistent ρ: use known pops symmetry
                    # For non-degenerate set, can compute ρ from velocity + known
                    ρw = 1.0  # assume incompressible
                    # Try to get ρ more accurately
                    # If the system includes pops with different cy, ρ comes from solving
                    rhs = [ρw - f_known_ρ,
                           ρw * u_w - f_known_jx,
                           0.0 - f_known_jy]
                    sol = A_mat \ rhs
                    for (col, q) in enumerate(bb_north)
                        fo[i, j_n, q] = sol[col]
                    end
                end
            end
        elseif bb_mode == :mirror
            # Mirror BB: interpolate opp(q) at the mirror of the departure across the wall
            # Then add Ladd correction for wall velocity
            j_wall = j_n + 0.5
            for i in 1:Nxe
                for q in bb_north
                    qo = opp_[q]
                    # Departure of q at (i, j_n)
                    i_dep_q = geom.i_dep[i, j_n, q]
                    j_dep_q = geom.j_dep[i, j_n, q]
                    # Mirror across wall: same i, reflected j
                    i_mirror = i_dep_q
                    j_mirror = 2*j_wall - j_dep_q
                    # Interpolate ALL 9 pops at mirror point → compute moments → feq
                    fp_mirror = [interp_fn(f, i_mirror, j_mirror, qq) for qq in 1:9]
                    ρm = sum(fp_mirror)
                    uxm = (fp_mirror[2]-fp_mirror[4]+fp_mirror[6]-fp_mirror[7]-fp_mirror[8]+fp_mirror[9]) / ρm
                    uym = (fp_mirror[3]-fp_mirror[5]+fp_mirror[6]+fp_mirror[7]-fp_mirror[8]-fp_mirror[9]) / ρm
                    # Post-collision value of opp(q) at mirror (ω=1 → feq)
                    f_refl = feq_q(qo, ρm, uxm, uym)
                    # BB with Ladd correction
                    fo[i, j_n, q] = f_refl - 2.0*w_[qo]*ρm*(cx_[qo]*u_w)*3.0
                end
            end
        elseif bb_mode == :mirror_gen
            # Generalized mirror: works for any ω (interpolate + local collision at mirror)
            j_wall = j_n + 0.5
            for i in 1:Nxe
                for q in bb_north
                    qo = opp_[q]
                    i_dep_q = geom.i_dep[i, j_n, q]
                    j_dep_q = geom.j_dep[i, j_n, q]
                    i_mirror = i_dep_q
                    j_mirror = 2*j_wall - j_dep_q
                    # Interpolate opp pop at mirror
                    fp_mirror = [interp_fn(f, i_mirror, j_mirror, qq) for qq in 1:9]
                    ρm = sum(fp_mirror)
                    uxm = (fp_mirror[2]-fp_mirror[4]+fp_mirror[6]-fp_mirror[7]-fp_mirror[8]+fp_mirror[9]) / ρm
                    uym = (fp_mirror[3]-fp_mirror[5]+fp_mirror[6]+fp_mirror[7]-fp_mirror[8]-fp_mirror[9]) / ρm
                    # Apply collision at mirror point
                    f_opp_coll = fp_mirror[qo] - ω * (fp_mirror[qo] - feq_q(qo, ρm, uxm, uym))
                    # BB with Ladd correction
                    fo[i, j_n, q] = f_opp_coll - 2.0*w_[qo]*ρm*(cx_[qo]*u_w)*3.0
                end
            end
        elseif bb_mode == :ibb_da
            # Interpolated BB using actual q_w from departure distance
            j_wall = j_n + 0.5  # wall at halfway to ghost
            for i in 1:Nxe
                ρw = sum(fo[i, j_n, q] for q in 1:9)
                for q in bb_north
                    qo = opp_[q]
                    Δη_q = -(geom.j_dep[i, j_n, q] - j_n)  # departure displacement (positive = into ghost)
                    q_w = 0.5 / abs(Δη_q)  # fraction from node to wall
                    # Bouzidi IBB with moving wall
                    if q_w <= 0.5
                        # Use neighbor value for opp
                        j_nb = j_n - 1  # interior neighbor
                        fo[i, j_n, q] = 2*q_w*fo[i, j_n, qo] + (1-2*q_w)*fo[i, j_nb, qo] +
                                        2*w_[q]*ρw*(cx_[q]*u_w)*3.0
                    else
                        # Standard branch
                        fo[i, j_n, q] = (1/(2*q_w))*fo[i, j_n, qo] + (1-1/(2*q_w))*fo[i, j_n, q] +
                                        (1/q_w)*w_[q]*ρw*(cx_[q]*u_w)*3.0
                    end
                end
            end
        end

        f .= fo
        if any(isnan, f); nan_step = it; break; end
    end

    # Measure error
    max_err = 0.0
    if nan_step == 0
        for j in 1:Ny
            ie = ic; je = ng + j
            ρn = sum(f[ie,je,q] for q in 1:9)
            uxn = (f[ie,je,2]-f[ie,je,4]+f[ie,je,6]-f[ie,je,7]-f[ie,je,8]+f[ie,je,9]) / ρn
            max_err = max(max_err, abs(uxn - u_couette(j)))
        end
    end

    bb_n_str = join([dirs[q] for q in bb_north], ",")
    bb_s_str = join([dirs[q] for q in bb_south], ",")

    if nan_step > 0
        println("  θ=$(lpad(Int(angle),3))°  ghost=$(rpad(ghost_mode,12))  bb=$(rpad(bb_mode,12))  interp=$(rpad(interp,12))  BB_N={$(bb_n_str)}  → NaN at step $nan_step")
    else
        println("  θ=$(lpad(Int(angle),3))°  ghost=$(rpad(ghost_mode,12))  bb=$(rpad(bb_mode,12))  interp=$(rpad(interp,12))  BB_N={$(bb_n_str)}  → max|Δux|=$(round(max_err, sigdigits=3))")
    end
    return (nan_step == 0 ? max_err : NaN)
end

function _make_case(angle; Ny=5, ng=2, u_w=0.01, omega=1.0,
                    interp=:biquadratic, equilibrium=:quadratic)
    Nx = Ny
    Nxe = Nx + 2*ng
    Nye = Ny + 2*ng
    X = zeros(Nx, Ny)
    Y = zeros(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        X[i,j] = Float64(i-1)
        Y[i,j] = Float64(j-1) + Float64(i-1)*tand(angle)
    end
    mesh = CurvilinearMesh(X, Y; periodic_ξ=false, periodic_η=false,
                           type=:curvilinear, dx_ref=1.0, skip_validate=true, FT=Float64)
    mesh_ext = KrakeNN.extend_mesh_2d(mesh; n_ghost=ng)
    geom = build_slbm_geometry(mesh_ext; local_cfl=false, dx_ref=1.0)
    is_sol = zeros(Bool, Nxe, Nye)
    ic = ng + div(Nx, 2) + 1
    j_s = ng + 1
    j_n = ng + Ny
    return (; angle, Nx, Ny, ng, u_w, omega, interp, equilibrium,
            Nxe, Nye, geom, is_sol, ic, j_s, j_n)
end

_u_at(c, jext) = c.u_w * ((jext - c.ng) - 1.0) / (c.Ny - 1.0)
_exact_f(c, q, jext) = _feq_q(c.equilibrium, q, 1.0, _u_at(c, jext), 0.0)

function _fill_exact_feq!(f, c)
    fill!(f, 0.0)
    for j in 1:c.Ny, i in 1:c.Nx, q in 1:9
        f[c.ng+i, c.ng+j, q] = _feq_q(c.equilibrium, q, 1.0,
                                      c.u_w * (j - 1) / (c.Ny - 1), 0.0)
    end
    return f
end

function _fill_ghost_edge_case!(f, c, ib, ib2, ig, Nother, dim, fneq_mode)
    for idx in 1:Nother
        i1, j1 = dim == :y ? (idx, ib) : (ib, idx)
        i2, j2 = dim == :y ? (idx, ib2) : (ib2, idx)
        ig_, jg_ = dim == :y ? (idx, ig) : (ig, idx)
        ρ1, ux1, uy1 = _moments(f, i1, j1)
        ρ2, ux2, uy2 = _moments(f, i2, j2)
        g_dist = abs(ig - ib)
        ρg  = ρ1  + g_dist*(ρ1  - ρ2)
        uxg = ux1 + g_dist*(ux1 - ux2)
        uyg = uy1 + g_dist*(uy1 - uy2)
        for q in 1:9
            feq1 = _feq_q(c.equilibrium, q, ρ1, ux1, uy1)
            fneq1 = f[i1,j1,q] - feq1
            if fneq_mode == :zeroth
                fneq_g = fneq1
            else
                feq2 = _feq_q(c.equilibrium, q, ρ2, ux2, uy2)
                fneq2 = f[i2,j2,q] - feq2
                fneq_g = fneq1 + g_dist*(fneq1 - fneq2)
            end
            f[ig_,jg_,q] = _feq_q(c.equilibrium, q, ρg, uxg, uyg) + fneq_g
        end
    end
end

function _ghost_fill_hybrid_case!(f, c; fneq_mode=:zeroth)
    for g in 1:c.ng
        _fill_ghost_edge_case!(f, c, c.ng+1, c.ng+2, c.ng+1-g, c.Nxe, :y, fneq_mode)
        _fill_ghost_edge_case!(f, c, c.ng+c.Ny, c.ng+c.Ny-1, c.ng+c.Ny+g, c.Nxe, :y, fneq_mode)
        _fill_ghost_edge_case!(f, c, c.ng+1, c.ng+2, c.ng+1-g, c.Nye, :x, fneq_mode)
        _fill_ghost_edge_case!(f, c, c.ng+c.Nx, c.ng+c.Nx-1, c.ng+c.Nx+g, c.Nye, :x, fneq_mode)
    end
    return f
end

function _ghost_fill_feq_extrap_case!(f, c)
    _ghost_fill_hybrid_case!(f, c; fneq_mode=:zeroth)
    for g in 1:c.ng
        for i in 1:c.Nxe
            for jg in (c.ng+1-g, c.ng+c.Ny+g)
                ρ, ux, uy = _moments(f, i, jg)
                for q in 1:9
                    f[i, jg, q] = _feq_q(c.equilibrium, q, ρ, ux, uy)
                end
            end
        end
        for j in 1:c.Nye
            for ig in (c.ng+1-g, c.ng+c.Nx+g)
                ρ, ux, uy = _moments(f, ig, j)
                for q in 1:9
                    f[ig, j, q] = _feq_q(c.equilibrium, q, ρ, ux, uy)
                end
            end
        end
    end
    return f
end

function _ghost_fill!(f, c, ghost_mode)
    if ghost_mode == :feq_extrap
        _ghost_fill_feq_extrap_case!(f, c)
    elseif ghost_mode == :hybrid_z
        _ghost_fill_hybrid_case!(f, c; fneq_mode=:zeroth)
    elseif ghost_mode == :hybrid_l
        _ghost_fill_hybrid_case!(f, c; fneq_mode=:linear)
    else
        ghost_fill_f_linear!(f, c.Nxe, c.Nye, c.ng, c.Nx, c.Ny)
    end
    return f
end

function _interp_f(f, c, i, j, q)
    if c.interp == :biquadratic
        return KrakeNN.biquadratic_f(f, c.is_sol, c.geom.i_dep[i,j,q], c.geom.j_dep[i,j,q],
                                    q, c.Nxe, c.Nye, false, false)
    else
        return KrakeNN.bilinear_f(f, c.geom.i_dep[i,j,q], c.geom.j_dep[i,j,q],
                                 q, c.Nxe, c.Nye, false, false)
    end
end

function _bb_sets(c)
    bb_south = Int[]
    bb_north = Int[]
    for q in 2:9
        c.geom.j_dep[c.ic, c.j_s, q] < c.j_s - 0.5 && push!(bb_south, q)
        c.geom.j_dep[c.ic, c.j_n, q] > c.j_n + 0.5 && push!(bb_north, q)
    end
    return bb_south, bb_north
end

function _max_ghost_analytic_error(f, c)
    best = (err=0.0, i=0, j=0, q=0)
    for j in vcat(1:c.ng, c.j_n+1:c.j_n+c.ng), i in 1:c.Nxe, q in 1:9
        err = abs(f[i,j,q] - _exact_f(c, q, j))
        err > best.err && (best = (err=err, i=i, j=j, q=q))
    end
    return best
end

function _max_interp_analytic_error(f, c; rows)
    best = (err=0.0, i=0, j=0, q=0, depj=0.0)
    for j in rows, q in 1:9
        val = _interp_f(f, c, c.ic, j, q)
        jd = c.geom.j_dep[c.ic, j, q]
        err = abs(val - _exact_f(c, q, jd))
        err > best.err && (best = (err=err, i=c.ic, j=j, q=q, depj=jd))
    end
    return best
end

function _streamed_moments(f, c, i, j)
    fp = [_interp_f(f, c, i, j, q) for q in 1:9]
    ρ = sum(fp)
    ux = (fp[2]-fp[4]+fp[6]-fp[7]-fp[8]+fp[9]) / ρ
    uy = (fp[3]-fp[5]+fp[6]+fp[7]-fp[8]-fp[9]) / ρ
    return ρ, ux, uy, fp
end

function _max_streamed_moment_error(f, c; rows)
    best = (err=0.0, j=0, ux=0.0, ref=0.0, ρ=0.0, uy=0.0)
    for j in rows
        ρ, ux, uy, _ = _streamed_moments(f, c, c.ic, j)
        ref = _u_at(c, j)
        err = abs(ux - ref)
        err > best.err && (best = (err=err, j=j, ux=ux, ref=ref, ρ=ρ, uy=uy))
    end
    return best
end

function _print_streamed_population_table(f, c, j)
    dirs = ["rest","E","N","W","S","NE","NW","SW","SE"]
    _, _, _, fp = _streamed_moments(f, c, c.ic, j)
    uref = _u_at(c, j)
    println("  streamed populations at arrival j=$(j), exact ux=$(uref)")
    for q in 1:9
        local_exact = _feq_q(c.equilibrium, q, 1.0, uref, 0.0)
        dep_exact = _exact_f(c, q, c.geom.j_dep[c.ic,j,q])
        println("    q=$(q) $(rpad(dirs[q],4)) depj=$(round(c.geom.j_dep[c.ic,j,q],digits=6)) fp-local_feq=$(round(fp[q]-local_exact,sigdigits=6)) fp-dep_exact=$(round(fp[q]-dep_exact,sigdigits=3))")
    end
end

function _max_interp_diff(fa, fb, c; rows)
    best = (err=0.0, i=0, j=0, q=0, depj=0.0)
    for j in rows, q in 1:9
        err = abs(_interp_f(fa, c, c.ic, j, q) - _interp_f(fb, c, c.ic, j, q))
        err > best.err && (best = (err=err, i=c.ic, j=j, q=q, depj=c.geom.j_dep[c.ic,j,q]))
    end
    return best
end

function _profile_error(f, c)
    best = (err=0.0, j=0, ux=0.0, ref=0.0)
    for jp in 1:c.Ny
        je = c.ng + jp
        ρ = sum(f[c.ic,je,q] for q in 1:9)
        ux = (f[c.ic,je,2]-f[c.ic,je,4]+f[c.ic,je,6]-f[c.ic,je,7]-f[c.ic,je,8]+f[c.ic,je,9]) / ρ
        ref = c.u_w * (jp - 1) / (c.Ny - 1)
        err = abs(ux - ref)
        err > best.err && (best = (err=err, j=je, ux=ux, ref=ref))
    end
    return best
end

function _wall_feq_error(f, c)
    best = (err=0.0, i=0, j=0, q=0)
    for i in 1:c.Nxe, q in 1:9
        err = abs(f[i,c.j_n,q] - _feq_q(c.equilibrium, q, 1.0, c.u_w, 0.0))
        err > best.err && (best = (err=err, i=i, j=c.j_n, q=q))
    end
    return best
end

function _clean_walls_to_exact_feq!(f, c)
    for i in 1:c.Nxe, q in 1:9
        f[i,c.j_s,q] = _feq_q(c.equilibrium, q, 1.0, 0.0, 0.0)
        f[i,c.j_n,q] = _feq_q(c.equilibrium, q, 1.0, c.u_w, 0.0)
    end
    return f
end

function _step_once(f, c; ghost_mode=:hybrid_z, bb_mode=:ladd)
    _ghost_fill!(f, c, ghost_mode)
    bb_south, bb_north = _bb_sets(c)
    fo = zeros(c.Nxe, c.Nye, 9)
    for je in 1:c.Nye, ie in 1:c.Nxe
        fp = [_interp_f(f, c, ie, je, q) for q in 1:9]
        if bb_mode == :prescribe && je == c.j_s
            non_bb = [q for q in 1:9 if !(q in bb_south)]
            ρw = _zou_he_rho(fp, bb_south, non_bb, 0.0, 0.0)
            for q in 1:9
                fo[ie,je,q] = _feq_q(c.equilibrium, q, ρw, 0.0, 0.0)
            end
        elseif bb_mode == :prescribe && je == c.j_n
            non_bb = [q for q in 1:9 if !(q in bb_north)]
            ρw = _zou_he_rho(fp, bb_north, non_bb, c.u_w, 0.0)
            for q in 1:9
                fo[ie,je,q] = _feq_q(c.equilibrium, q, ρw, c.u_w, 0.0)
            end
        else
            ρ = sum(fp)
            ux = (fp[2]-fp[4]+fp[6]-fp[7]-fp[8]+fp[9]) / ρ
            uy = (fp[3]-fp[5]+fp[6]+fp[7]-fp[8]-fp[9]) / ρ
            for q in 1:9
                fo[ie,je,q] = fp[q] - c.omega * (fp[q] - _feq_q(c.equilibrium, q, ρ, ux, uy))
            end
        end
    end
    for i in 1:c.Nxe
        for q in bb_south
            fo[i,c.j_s,q] = fo[i,c.j_s,opp_[q]]
        end
        ρw = sum(fo[i,c.j_n,q] for q in 1:9)
        for q in bb_north
            qo = opp_[q]
            fo[i,c.j_n,q] = fo[i,c.j_n,qo] - 2.0*w_[qo]*ρw*(cx_[qo]*c.u_w)*3.0
        end
    end
    return fo
end

function _stepwise_metrics(c; ghost_mode=:hybrid_z, bb_mode=:ladd, print_populations=false)
    dirs = ["rest","E","N","W","S","NE","NW","SW","SE"]
    f0 = zeros(c.Nxe, c.Nye, 9)
    _fill_exact_feq!(f0, c)
    _ghost_fill!(f0, c, ghost_mode)

    ge0 = _max_ghost_analytic_error(f0, c)
    iw0 = _max_interp_analytic_error(f0, c; rows=[c.j_n-1, c.j_n])
    sm0 = _max_streamed_moment_error(f0, c; rows=collect(c.j_s:c.j_n))
    if print_populations
        println("  max ghost-vs-analytic error = $(ge0.err) at i=$(ge0.i) j=$(ge0.j) q=$(ge0.q) $(dirs[ge0.q])")
        println("  max interp-vs-analytic error at rows j_n-1,j_n = $(iw0.err) at j=$(iw0.j) q=$(iw0.q) $(dirs[iw0.q]) depj=$(round(iw0.depj,digits=6))")
        println("  max streamed-moment ux error before collision/BB = $(sm0.err) at j=$(sm0.j) ux=$(sm0.ux) ref=$(sm0.ref) rho=$(sm0.ρ) uy=$(sm0.uy)")
        _print_streamed_population_table(f0, c, sm0.j)
    end

    f1 = _step_once(copy(f0), c; ghost_mode=ghost_mode, bb_mode=bb_mode)
    pe1 = _profile_error(f1, c)
    we1 = _wall_feq_error(f1, c)

    dirty = copy(f1)
    clean = copy(f1)
    _clean_walls_to_exact_feq!(clean, c)
    _ghost_fill!(dirty, c, ghost_mode)
    _ghost_fill!(clean, c, ghost_mode)
    gd = _max_interp_diff(dirty, clean, c; rows=[c.j_n-1, c.j_n])

    f = copy(f0)
    for _ in 1:200
        f = _step_once(f, c; ghost_mode=ghost_mode, bb_mode=bb_mode)
    end
    pe200 = _profile_error(f, c)
    return (; ghost=ge0, interp=iw0, streamed=sm0, one_step=pe1,
            wall=we1, dirty_clean=gd, steps200=pe200)
end

function run_stepwise_debug(; angle=45.0, Ny=5, u_w=0.01,
                            ghost_mode=:hybrid_z, bb_mode=:ladd,
                            interp=:biquadratic, equilibrium=:both)
    equilibria = equilibrium === :both ? (:quadratic, :linear) : (equilibrium,)
    c0 = _make_case(angle; Ny=Ny, u_w=u_w, interp=interp, equilibrium=first(equilibria))
    dirs = ["rest","E","N","W","S","NE","NW","SW","SE"]
    bb_south, bb_north = _bb_sets(c0)
    println("="^112)
    println("STEPWISE SLBM moving wall debug")
    println("theta=$(angle) Ny=$(Ny) u_w=$(u_w) ghost=$(ghost_mode) bb=$(bb_mode) interp=$(interp)")
    println("BB_S={$(join(dirs[bb_south], ","))}  BB_N={$(join(dirs[bb_north], ","))}")
    println("-"^112)
    println("North-wall departures at i=$(c0.ic), j_n=$(c0.j_n)")
    for q in 2:9
        jd = c0.geom.j_dep[c0.ic,c0.j_n,q]
        flag = jd > c0.j_n + 0.5 ? "BB" : "fluid"
        println("  q=$(q) $(rpad(dirs[q], 2))  j_dep=$(round(jd,digits=6))  $(flag)")
    end

    results = Dict{Symbol, Any}()
    for eq in equilibria
        c = _make_case(angle; Ny=Ny, u_w=u_w, interp=interp, equilibrium=eq)
        println("-"^112)
        println("Equilibrium = $(eq)")
        results[eq] = _stepwise_metrics(c; ghost_mode=ghost_mode, bb_mode=bb_mode,
                                        print_populations=true)
        m = results[eq]
        println("  after 1 step profile error = $(m.one_step.err) at j=$(m.one_step.j)")
        println("  north-wall pop diff from exact feq = $(m.wall.err) at q=$(m.wall.q) $(dirs[m.wall.q])")
        println("  next-step dirty-vs-clean interp diff = $(m.dirty_clean.err) at j=$(m.dirty_clean.j) q=$(m.dirty_clean.q) $(dirs[m.dirty_clean.q])")
        println("  after 200 steps profile error = $(m.steps200.err) at j=$(m.steps200.j)")
    end

    println("-"^112)
    println("Side-by-side summary")
    println(rpad("equilibrium", 13), rpad("interp_err", 15),
            rpad("streamed_ux_err", 18), rpad("step1_err", 15),
            rpad("step200_err", 15), "dirty_clean")
    for eq in equilibria
        m = results[eq]
        println(rpad(String(eq), 13),
                rpad(_fmt(m.interp.err), 15),
                rpad(_fmt(m.streamed.err), 18),
                rpad(_fmt(m.one_step.err), 15),
                rpad(_fmt(m.steps200.err), 15),
                _fmt(m.dirty_clean.err))
    end
    return nothing
end

function run_ma2_equilibrium_sweep(; angles=(0.0, 20.0, 45.0),
                                   uws=(0.01, 0.001, 0.0001),
                                   Ny=5, ghost_mode=:hybrid_z,
                                   bb_mode=:ladd, interp=:biquadratic)
    println("="^112)
    println("MA2 equilibrium sweep: quadratic vs linear")
    println("Ny=$(Ny) ghost=$(ghost_mode) bb=$(bb_mode) interp=$(interp)")
    println(rpad("theta", 8), rpad("u_w", 12), rpad("equilibrium", 13),
            rpad("streamed_ux_err", 18), rpad("err/u_w^2", 14),
            rpad("step1_err", 15), "step200_err")
    for angle in angles, uw in uws, eq in (:quadratic, :linear)
        c = _make_case(angle; Ny=Ny, u_w=uw, interp=interp, equilibrium=eq)
        m = _stepwise_metrics(c; ghost_mode=ghost_mode, bb_mode=bb_mode)
        ratio = uw == 0 ? NaN : m.streamed.err / (uw^2)
        println(rpad(string(angle), 8),
                rpad(string(uw), 12),
                rpad(String(eq), 13),
                rpad(_fmt(m.streamed.err), 18),
                rpad(_fmt(ratio), 14),
                rpad(_fmt(m.one_step.err), 15),
                _fmt(m.steps200.err))
    end
    return nothing
end

function _post_collision_wall_populations(fp, c)
    ρ = sum(fp)
    ux = (fp[2]-fp[4]+fp[6]-fp[7]-fp[8]+fp[9]) / ρ
    uy = (fp[3]-fp[5]+fp[6]+fp[7]-fp[8]-fp[9]) / ρ
    fo = similar(fp)
    for q in 1:9
        fo[q] = fp[q] - c.omega * (fp[q] - _feq_q(c.equilibrium, q, ρ, ux, uy))
    end
    return fo, ρ, ux, uy
end

function _apply_north_bb_to_wall_populations(fo_prebb, c, bb_north)
    fo = copy(fo_prebb)
    ρw = sum(fo_prebb)
    for q in bb_north
        qo = opp_[q]
        fo[q] = fo[qo] - 2.0*w_[qo]*ρw*(cx_[qo]*c.u_w)*3.0
    end
    return fo, ρw
end

function _population_moments(pop)
    ρ = sum(pop)
    ux = (pop[2]-pop[4]+pop[6]-pop[7]-pop[8]+pop[9]) / ρ
    uy = (pop[3]-pop[5]+pop[6]+pop[7]-pop[8]-pop[9]) / ρ
    return ρ, ux, uy
end

function _argmax_abs(vals)
    imax = 1
    vmax = abs(vals[1])
    for i in 2:length(vals)
        v = abs(vals[i])
        if v > vmax
            imax = i
            vmax = v
        end
    end
    return imax, vmax
end

function run_wall_pipeline_trace(; angle=20.0, Ny=5, u_w=0.01,
                                 ghost_mode=:hybrid_z, bb_mode=:ladd,
                                 interp=:biquadratic, equilibrium=:quadratic)
    c = _make_case(angle; Ny=Ny, u_w=u_w, interp=interp, equilibrium=equilibrium)
    dirs = ["rest","E","N","W","S","NE","NW","SW","SE"]
    _, bb_north = _bb_sets(c)
    f = zeros(c.Nxe, c.Nye, 9)
    _fill_exact_feq!(f, c)
    _ghost_fill!(f, c, ghost_mode)

    i = c.ic
    j = c.j_n
    fp = [_interp_f(f, c, i, j, q) for q in 1:9]
    fp_m = _population_moments(fp)
    fo_prebb, ρc, uxc, uyc = _post_collision_wall_populations(fp, c)
    fo_postbb, ρw = _apply_north_bb_to_wall_populations(fo_prebb, c, bb_north)
    post_m = _population_moments(fo_postbb)

    println("="^112)
    println("WALL PIPELINE TRACE")
    println("theta=$(angle) Ny=$(Ny) u_w=$(u_w) ghost=$(ghost_mode) bb=$(bb_mode) interp=$(interp) equilibrium=$(equilibrium)")
    println("arrival i=$(i) j_n=$(j)  BB_N={$(join(dirs[bb_north], ","))}")
    println("-"^112)
    println("Moments")
    println("  after interpolation: rho=$(_fmt(fp_m[1])) ux=$(_fmt(fp_m[2])) uy=$(_fmt(fp_m[3]))  target rho=1 ux=$(_fmt(c.u_w))")
    println("  after collision:     rho=$(_fmt(ρc)) ux=$(_fmt(uxc)) uy=$(_fmt(uyc))")
    println("  BB density used:     rho_w=$(_fmt(ρw))")
    println("  after post-BB:       rho=$(_fmt(post_m[1])) ux=$(_fmt(post_m[2])) uy=$(_fmt(post_m[3]))")
    println("-"^112)
    println(rpad("q", 4), rpad("dir", 7), rpad("depj", 10), rpad("BB", 5),
            rpad("fp-dep", 12), rpad("fp-wall", 12),
            rpad("coll-wall", 12), rpad("postBB-wall", 13), "mechanism")

    fp_dep_err = zeros(9)
    fp_wall_err = zeros(9)
    coll_wall_err = zeros(9)
    post_wall_err = zeros(9)
    for q in 1:9
        dep_exact = _exact_f(c, q, c.geom.j_dep[i,j,q])
        wall_exact = _feq_q(c.equilibrium, q, 1.0, c.u_w, 0.0)
        fp_dep_err[q] = fp[q] - dep_exact
        fp_wall_err[q] = fp[q] - wall_exact
        coll_wall_err[q] = fo_prebb[q] - wall_exact
        post_wall_err[q] = fo_postbb[q] - wall_exact
        isbb = q in bb_north
        mechanism = if abs(fp_dep_err[q]) > 1e-12
            "interp/ghost"
        elseif abs(coll_wall_err[q]) > 1e-12 && !isbb
            "mixed-moment collision"
        elseif abs(post_wall_err[q]) > 1e-12 && isbb
            "BB correction/rho"
        elseif abs(post_wall_err[q]) > 1e-12
            "unchanged non-BB"
        else
            "ok"
        end
        println(rpad(string(q), 4),
                rpad(dirs[q], 7),
                rpad(_fmt(c.geom.j_dep[i,j,q]), 10),
                rpad(isbb ? "yes" : "no", 5),
                rpad(_fmt(fp_dep_err[q]), 12),
                rpad(_fmt(fp_wall_err[q]), 12),
                rpad(_fmt(coll_wall_err[q]), 12),
                rpad(_fmt(post_wall_err[q]), 13),
                mechanism)
    end

    q_fp, e_fp = _argmax_abs(fp_dep_err)
    q_coll, e_coll = _argmax_abs(coll_wall_err)
    q_post, e_post = _argmax_abs(post_wall_err)
    println("-"^112)
    println("Max errors")
    println("  interpolation vs departure exact: q=$(q_fp) $(dirs[q_fp]) err=$(_fmt(e_fp))")
    println("  post-collision vs wall exact:     q=$(q_coll) $(dirs[q_coll]) err=$(_fmt(e_coll))")
    println("  post-BB vs wall exact:            q=$(q_post) $(dirs[q_post]) err=$(_fmt(e_post))")
    return nothing
end

function _tangent_velocity_at(c, jext)
    s = ((jext - c.ng) - 1.0) / (c.Ny - 1.0)
    return c.u_w * s, c.u_w * tand(c.angle) * s
end

function _fill_exact_tangent_feq!(f, c)
    fill!(f, 0.0)
    for j in 1:c.Ny, i in 1:c.Nx, q in 1:9
        ux, uy = _tangent_velocity_at(c, c.ng + j)
        f[c.ng+i, c.ng+j, q] = _feq_q(c.equilibrium, q, 1.0, ux, uy)
    end
    return f
end

function _profile_error_tangent(f, c)
    best = (err=0.0, j=0, ux=0.0, uy=0.0, uref=0.0, vref=0.0, ρ=0.0)
    for jp in 1:c.Ny
        je = c.ng + jp
        ρ = sum(f[c.ic,je,q] for q in 1:9)
        ux = (f[c.ic,je,2]-f[c.ic,je,4]+f[c.ic,je,6]-f[c.ic,je,7]-f[c.ic,je,8]+f[c.ic,je,9]) / ρ
        uy = (f[c.ic,je,3]-f[c.ic,je,5]+f[c.ic,je,6]+f[c.ic,je,7]-f[c.ic,je,8]-f[c.ic,je,9]) / ρ
        uref, vref = _tangent_velocity_at(c, je)
        err = max(abs(ux - uref), abs(uy - vref))
        err > best.err && (best = (err=err, j=je, ux=ux, uy=uy, uref=uref, vref=vref, ρ=ρ))
    end
    return best
end

function _max_streamed_moment_error_tangent(f, c; rows)
    best = (err=0.0, j=0, ux=0.0, uy=0.0, uref=0.0, vref=0.0, ρ=0.0)
    for j in rows
        ρ, ux, uy, _ = _streamed_moments(f, c, c.ic, j)
        uref, vref = _tangent_velocity_at(c, j)
        err = max(abs(ux - uref), abs(uy - vref))
        err > best.err && (best = (err=err, j=j, ux=ux, uy=uy, uref=uref, vref=vref, ρ=ρ))
    end
    return best
end

function _step_once_tangent(f, c; ghost_mode=:hybrid_z)
    _ghost_fill!(f, c, ghost_mode)
    bb_south, bb_north = _bb_sets(c)
    fo = zeros(c.Nxe, c.Nye, 9)
    for je in 1:c.Nye, ie in 1:c.Nxe
        fp = [_interp_f(f, c, ie, je, q) for q in 1:9]
        ρ, ux, uy = _population_moments(fp)
        for q in 1:9
            fo[ie,je,q] = fp[q] - c.omega * (fp[q] - _feq_q(c.equilibrium, q, ρ, ux, uy))
        end
    end
    uxw, uyw = c.u_w, c.u_w * tand(c.angle)
    for i in 1:c.Nxe
        for q in bb_south
            fo[i,c.j_s,q] = fo[i,c.j_s,opp_[q]]
        end
        ρw = sum(fo[i,c.j_n,q] for q in 1:9)
        for q in bb_north
            qo = opp_[q]
            fo[i,c.j_n,q] = fo[i,c.j_n,qo] -
                             2.0*w_[qo]*ρw*(cx_[qo]*uxw + cy_[qo]*uyw)*3.0
        end
    end
    return fo
end

function run_tangent_wall_check(; angles=(0.0, 20.0, 45.0), Ny=5,
                                u_w=0.01, steps=200,
                                ghost_mode=:hybrid_z, interp=:biquadratic,
                                equilibrium=:quadratic)
    println("="^112)
    println("TANGENTIAL moving-wall check")
    println("u_wall = (u_w, u_w*tan(theta)); reference u(j) is parallel to the oblique wall")
    println("Ny=$(Ny) u_w_x=$(u_w) steps=$(steps) ghost=$(ghost_mode) interp=$(interp) equilibrium=$(equilibrium)")
    println(rpad("theta", 8), rpad("streamed_err", 15), rpad("step$(steps)_err", 15),
            rpad("ux", 12), rpad("ux_ref", 12), rpad("uy", 12), "uy_ref")
    for angle in angles
        c = _make_case(angle; Ny=Ny, u_w=u_w, interp=interp, equilibrium=equilibrium)
        f0 = zeros(c.Nxe, c.Nye, 9)
        _fill_exact_tangent_feq!(f0, c)
        _ghost_fill!(f0, c, ghost_mode)
        sm = _max_streamed_moment_error_tangent(f0, c; rows=collect(c.j_s:c.j_n))

        f = copy(f0)
        for _ in 1:steps
            f = _step_once_tangent(f, c; ghost_mode=ghost_mode)
        end
        pe = _profile_error_tangent(f, c)
        println(rpad(string(angle), 8),
                rpad(_fmt(sm.err), 15),
                rpad(_fmt(pe.err), 15),
                rpad(_fmt(pe.ux), 12),
                rpad(_fmt(pe.uref), 12),
                rpad(_fmt(pe.uy), 12),
                _fmt(pe.vref))
    end
    return nothing
end

# === PRESCRIBE vs LADD ===
if abspath(PROGRAM_FILE) == abspath(@__FILE__) && get(ENV, "KRK_SLBM_SWEEP", "0") == "1"
    run_ma2_equilibrium_sweep()
elseif abspath(PROGRAM_FILE) == abspath(@__FILE__) && get(ENV, "KRK_SLBM_FULL_SWEEP", "0") == "1"
    angles = [0.0, 10.0, 20.0, 30.0, 45.0]

    println("="^115)
    println("PRESCRIBE vs LADD: hybrid_z ghost + biquad, Ny=5, omega=1")
    println("="^115)
    for bbm in [:ladd, :prescribe]
        println("\n--- bb_mode = $bbm ---")
        for a in angles
            test_moving_wall(a; ghost_mode=:hybrid_z, interp=:biquadratic, bb_mode=bbm)
        end
    end

    println("\n" * "="^115)
    println("PRESCRIBE: varying Ny (theta=20)")
    println("="^115)
    for Ny in [5, 10, 20, 40]
        steps = max(200, 10 * Ny^2)
        test_moving_wall(20.0; ghost_mode=:hybrid_z, interp=:biquadratic, bb_mode=:prescribe, Ny=Ny, steps=steps)
    end

    println("\n" * "="^115)
    println("PRESCRIBE: varying u_w (theta=20, Ny=10)")
    println("="^115)
    for uw in [0.01, 0.001, 0.0001]
        test_moving_wall(20.0; ghost_mode=:hybrid_z, interp=:biquadratic, bb_mode=:prescribe, Ny=10, steps=500, u_w=uw)
    end

    println("\n" * "="^115)
    println("PRESCRIBE: all angles (Ny=10)")
    println("="^115)
    for a in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
        test_moving_wall(a; ghost_mode=:hybrid_z, interp=:biquadratic, bb_mode=:prescribe, Ny=10, steps=500)
    end
elseif abspath(PROGRAM_FILE) == abspath(@__FILE__)
    run_wall_pipeline_trace()
    println()
    run_stepwise_debug()
    println()
    run_tangent_wall_check()
end
