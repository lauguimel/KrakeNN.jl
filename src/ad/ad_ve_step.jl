# BIT-MIRROR of the production M8 viscoelastic coupled step
# (drivers/viscoelastic_logfv_coupled_step_2d.jl). The tapeable CPU-Float64
# coupled one-step operator `ad_ve_coupled_step!` is a step-for-step mirror of
# the production VE coupled step to FP floor; if the production LBM/LI-BB/TRT/
# ZouHe/FVFD-advection/constitutive algebra changes, update this file too.
#
# This is the VE analogue of `ad_thermal_step.jl` (coupled f+T) — here the
# coupled fields are f (D2Q9 distributions) and psi (log-conformation tensor),
# stacked into one flat state w = (f, psi). Plain-Julia, Enzyme-tapeable: NO
# @kernel, NO GPU, NO `using Enzyme` (Enzyme stays a weakdep; the reverse seams
# live in ext/KrakenADExt.jl).
#
# The operator applies the THREE fixes the forensic mission isolated to match
# production exactly:
#   (1) is_solid mask: psi is advected / differentiated on the LBM NODE mask
#       (`g.is_solid` here IS the LBM mask, built via `ad_ve_build_matched_geom`),
#       NOT the FVFD cell-fraction mask. FVFD fractions / wall geometry are kept
#       verbatim from the circle builder.
#   (2) psi-advection edge BC = production Dirichlet (west_phi=0, east_phi=psi[Nx,:],
#       south/north wall mirror) + the WEST embedded face velocity = inlet profile.
#   (3) fused ZouHe rebuild on f cols i=1 (west velocity profile) and i=Nx (east
#       pressure rho_out=1), rows j=2..Ny-1, reading the PRE-stream f, via the
#       regularized TRT collide — fused into the same pass to match the production
#       kernel order (fused step -> apply_bc_rebuild).
#
# Ported verbatim (preserving every formula / ordering / float op) from the
# validated scratch chain (bench/scratch/ve_c0_matched.jl + ve_ad_c0.jl +
# ve_ad_bc_attribution.jl + ve_ad_spike{,_embedded}.jl), namespaced `ad_ve_*`.
#
# The D2Q9 LBM helpers, the log-conformation constitutive math, the cut-cell
# circle geometry (`ADVEEmbeddedGeom` + builders) and the embedded FVFD operators
# this step uses live in the companion `ad_ve_ops.jl` (included first in Kraken.jl).

# ----------------------------------------------------------------------------
# Stacked-state coupled params (scratch VECoupledParams). Flat layout:
#   w[1:9n]      = f[i,j,q]    (popidx layout)
#   w[9n+1:12n]  = [psixx; psixy; psiyy]  (3 sym components, 3n total)
struct ADVECoupledParams
    Nx::Int
    Ny::Int
    lambda::Float64
    dt::Float64           # constitutive substep dt
    n_substeps::Int
    prefactor::Float64    # nu_p/lambda for tau_p = prefactor*(C-I)
    nu_lbm::Float64       # solvent (+bsd) lattice viscosity -> TRT rates
    Fx_body::Float64      # frozen body force x
    s_plus::Float64
    s_minus::Float64
end

@inline ad_ve_n(p::ADVECoupledParams) = p.Nx * p.Ny

"""
    ad_ve_compute_faces(w, g, p) -> (ux_face, uy_face)

Compute the live embedded face velocities from a state `w` (Convention-I raw
momentum). Used to build the frozen-switch sign-source faces at the base point;
the matched operator does not consume them (it uses production hard-Dirichlet
advection), so this is a no-op switch but kept for chain-signature parity.
"""
function ad_ve_compute_faces(w, g::ADVEEmbeddedGeom, p::ADVECoupledParams)
    Nx, Ny = p.Nx, p.Ny
    is_solid = g.is_solid
    ux = zeros(Float64, Nx, Ny); uy = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        f1 = w[ad_ve_fpop(i, j, 1, Nx, Ny)]; f2 = w[ad_ve_fpop(i, j, 2, Nx, Ny)]
        f3 = w[ad_ve_fpop(i, j, 3, Nx, Ny)]; f4 = w[ad_ve_fpop(i, j, 4, Nx, Ny)]
        f5 = w[ad_ve_fpop(i, j, 5, Nx, Ny)]; f6 = w[ad_ve_fpop(i, j, 6, Nx, Ny)]
        f7 = w[ad_ve_fpop(i, j, 7, Nx, Ny)]; f8 = w[ad_ve_fpop(i, j, 8, Nx, Ny)]
        f9 = w[ad_ve_fpop(i, j, 9, Nx, Ny)]
        inv_rho = 1.0 / (f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9)
        ux[i, j] = (f2 - f4 + f6 - f7 - f8 + f9) * inv_rho
        uy[i, j] = (f3 - f5 + f6 + f7 - f8 - f9) * inv_rho
    end
    ux_face = zeros(Float64, Nx + 1, Ny); uy_face = zeros(Float64, Nx, Ny + 1)
    # west face = local zero-gradient ux[1,J] here (sign-source only; unused by
    # the matched operator). Mirror scratch cell_velocity_to_faces_embedded!.
    @inbounds for J in 1:Ny, I in 1:(Nx + 1)
        if I == 1
            ux_face[I, J] = is_solid[1, J] ? 0.0 : g.west_fraction[1, J] * ux[1, J]
        elseif I == Nx + 1
            ux_face[I, J] = is_solid[Nx, J] ? 0.0 : g.east_fraction[Nx, J] * ux[Nx, J]
        else
            frac = ad_ve_xface_frac(is_solid, g.west_fraction, g.east_fraction, I - 1, I, J)
            ux_face[I, J] = frac * ad_ve_xface_avg0(ux, is_solid, I - 1, I, J)
        end
    end
    @inbounds for J in 1:(Ny + 1), I in 1:Nx
        if J == 1 || J == Ny + 1
            uy_face[I, J] = 0.0
        else
            frac = ad_ve_yface_frac(is_solid, g.south_fraction, g.north_fraction, I, J - 1, J)
            uy_face[I, J] = frac * ad_ve_yface_avg0(uy, is_solid, I, J - 1, J)
        end
    end
    return (ux_face, uy_face)
end

# ----------------------------------------------------------------------------
"""
    ad_ve_coupled_step!(w_out, w_in, g, q_wall, p, u_profile, rho_out=1.0, faces0=nothing)

The tapeable coupled one-step VE operator G(w) -> w'. BIT-MIRROR of the
production M8 VE coupled step. Stacked flat layout:
  w[1:9n]      = f[i,j,q]
  w[9n+1:12n]  = [psixx; psixy; psiyy]

Arguments (`g`, `q_wall`, `p`, `u_profile`, `rho_out`, `faces0` are Enzyme Const):
  g          : `ADVEEmbeddedGeom` whose `is_solid` IS the LBM mask (matched geom)
  q_wall     : f-side cut-link q_wall (production frame, Array{Float64,3})
  p          : `ADVECoupledParams`
  u_profile  : inlet parabolic west-face / ZouHe velocity profile (length Ny)
  rho_out    : east-pressure ZouHe outlet density (1.0 for production)
  faces0     : frozen-switch sign-source faces (unused here; the matched advect
               uses production hard-Dirichlet BC). Kept for signature parity.

Enzyme-tapeable: all arrays heap-allocated locally, plain control flow.
"""
function ad_ve_coupled_step!(w_out, w_in, g::ADVEEmbeddedGeom, q_wall::Array{Float64,3},
                             p::ADVECoupledParams, u_profile::Vector{Float64},
                             rho_out::Float64=1.0, faces0=nothing)
    Nx, Ny = p.Nx, p.Ny
    n = Nx * Ny
    inv_dx = 1.0; inv_dy = 1.0
    inv_2dx = 0.5; inv_2dy = 0.5
    wb, eb, sb, nb = AD_VE_WB, AD_VE_EB, AD_VE_SB, AD_VE_NB
    is_solid = g.is_solid
    sp = p.s_plus; sm = p.s_minus
    foff = 0; poff = 9n

    # ---- unpack psi ----
    psixx_in = zeros(Float64, Nx, Ny)
    psixy_in = zeros(Float64, Nx, Ny)
    psiyy_in = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        k = ad_ve_lin(i, j, Nx)
        psixx_in[i, j] = w_in[poff + k]
        psixy_in[i, j] = w_in[poff + n + k]
        psiyy_in[i, j] = w_in[poff + 2n + k]
    end

    # ---- recover macro velocity u from INPUT f (Convention I, raw momentum) ----
    ux = zeros(Float64, Nx, Ny)
    uy = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        f1 = w_in[foff + ad_ve_fpop(i, j, 1, Nx, Ny)]
        f2 = w_in[foff + ad_ve_fpop(i, j, 2, Nx, Ny)]
        f3 = w_in[foff + ad_ve_fpop(i, j, 3, Nx, Ny)]
        f4 = w_in[foff + ad_ve_fpop(i, j, 4, Nx, Ny)]
        f5 = w_in[foff + ad_ve_fpop(i, j, 5, Nx, Ny)]
        f6 = w_in[foff + ad_ve_fpop(i, j, 6, Nx, Ny)]
        f7 = w_in[foff + ad_ve_fpop(i, j, 7, Nx, Ny)]
        f8 = w_in[foff + ad_ve_fpop(i, j, 8, Nx, Ny)]
        f9 = w_in[foff + ad_ve_fpop(i, j, 9, Nx, Ny)]
        inv_rho = 1.0 / (f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9)
        ux[i, j] = (f2 - f4 + f6 - f7 - f8 + f9) * inv_rho
        uy[i, j] = (f3 - f5 + f6 + f7 - f8 - f9) * inv_rho
    end

    # 1. embedded cell -> face velocity, WEST face = inlet profile (fix #2b)
    ux_face = zeros(Float64, Nx + 1, Ny)
    uy_face = zeros(Float64, Nx, Ny + 1)
    ad_ve_cell_velocity_to_faces_westprofile!(ux_face, uy_face, ux, uy, g, Nx, Ny, u_profile)

    # 2. advect psi with production Dirichlet edge BC (fix #2a):
    #    west_phi=0, east_phi=psi[Nx,:], south/north = wall mirror.
    east_xx = psixx_in[Nx, :]; east_xy = psixy_in[Nx, :]; east_yy = psiyy_in[Nx, :]
    psixx_adv = ad_ve_advect_prodbc(psixx_in, ux_face, uy_face, is_solid, Nx, Ny, east_xx)
    psixy_adv = ad_ve_advect_prodbc(psixy_in, ux_face, uy_face, is_solid, Nx, Ny, east_xy)
    psiyy_adv = ad_ve_advect_prodbc(psiyy_in, ux_face, uy_face, is_solid, Nx, Ny, east_yy)

    # 3. embedded velocity gradient (wall-grad correction)
    dudx = zeros(Float64, Nx, Ny); dudy = zeros(Float64, Nx, Ny)
    dvdx = zeros(Float64, Nx, Ny); dvdy = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        gxx = ad_ve_deriv_x_2d(ux, is_solid, i, j, Nx, inv_dx, inv_2dx, wb, eb)
        gxy = ad_ve_deriv_y_2d(ux, is_solid, i, j, Ny, inv_dy, inv_2dy, sb, nb)
        gyx = ad_ve_deriv_x_2d(uy, is_solid, i, j, Nx, inv_dx, inv_2dx, wb, eb)
        gyy = ad_ve_deriv_y_2d(uy, is_solid, i, j, Ny, inv_dy, inv_2dy, sb, nb)
        gxx, gxy = ad_ve_apply_embedded_wall_gradient(gxx, gxy, ux, g.wall_nx, g.wall_ny,
                                                      g.wall_inv_distance_to_center, i, j)
        gyx, gyy = ad_ve_apply_embedded_wall_gradient(gyx, gyy, uy, g.wall_nx, g.wall_ny,
                                                      g.wall_inv_distance_to_center, i, j)
        dudx[i, j] = gxx; dudy[i, j] = gxy; dvdx[i, j] = gyx; dvdy[i, j] = gyy
    end

    # 4. constitutive substeps psi_adv -> psi'
    psixx_p = zeros(Float64, Nx, Ny); psixy_p = zeros(Float64, Nx, Ny)
    psiyy_p = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        px = psixx_adv[i, j]; py = psixy_adv[i, j]; pyy = psiyy_adv[i, j]
        for _ in 1:p.n_substeps
            px, py, pyy = ad_ve_constitutive_step_log_2d(
                px, py, pyy, dudx[i, j], dudy[i, j], dvdx[i, j], dvdy[i, j],
                p.lambda, p.dt)
        end
        psixx_p[i, j] = px; psixy_p[i, j] = py; psiyy_p[i, j] = pyy
    end

    # 5. stress tau_p = prefactor*(C - I)
    tauxx = zeros(Float64, Nx, Ny); tauxy = zeros(Float64, Nx, Ny)
    tauyy = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        txx, txy, tyy = ad_ve_stress_from_log_2d(psixx_p[i, j], psixy_p[i, j],
                                                 psiyy_p[i, j], p.prefactor)
        tauxx[i, j] = txx; tauxy[i, j] = txy; tauyy[i, j] = tyy
    end

    # 6. embedded F_poly = div . tau_p + Option-A cell-fraction rescale
    fx_poly = zeros(Float64, Nx, Ny); fy_poly = zeros(Float64, Nx, Ny)
    ad_ve_tensor_divergence_embedded!(fx_poly, fy_poly, tauxx, tauxy, tauyy, g, Nx, Ny,
                                      inv_dx, inv_dy)
    @inbounds for j in 1:Ny, i in 1:Nx
        if !is_solid[i, j]
            c = g.cell_fraction[i, j]
            fx_poly[i, j] *= c
            fy_poly[i, j] *= c
        end
    end

    # 7. total force = F_poly + constant body force
    fx_total = zeros(Float64, Nx, Ny); fy_total = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        if !is_solid[i, j]
            fx_total[i, j] = fx_poly[i, j] + p.Fx_body
            fy_total[i, j] = fy_poly[i, j]
        end
    end

    # ============================================================
    # f-side fused step: pull-stream (halfway-BB at domain edges) + LI-BB cut-links
    # + TRT-Guo collide (mirror fused_trt_libb_v2_guo_field_step!).
    # ============================================================
    @inbounds for j in 1:Ny, i in 1:Nx
        if is_solid[i, j]
            w_out[foff + ad_ve_fpop(i, j, 1, Nx, Ny)] = 4.0 / 9.0
            w_out[foff + ad_ve_fpop(i, j, 2, Nx, Ny)] = 1.0 / 9.0
            w_out[foff + ad_ve_fpop(i, j, 3, Nx, Ny)] = 1.0 / 9.0
            w_out[foff + ad_ve_fpop(i, j, 4, Nx, Ny)] = 1.0 / 9.0
            w_out[foff + ad_ve_fpop(i, j, 5, Nx, Ny)] = 1.0 / 9.0
            w_out[foff + ad_ve_fpop(i, j, 6, Nx, Ny)] = 1.0 / 36.0
            w_out[foff + ad_ve_fpop(i, j, 7, Nx, Ny)] = 1.0 / 36.0
            w_out[foff + ad_ve_fpop(i, j, 8, Nx, Ny)] = 1.0 / 36.0
            w_out[foff + ad_ve_fpop(i, j, 9, Nx, Ny)] = 1.0 / 36.0
            continue
        end

        fp1 = w_in[foff + ad_ve_fpop(i, j, 1, Nx, Ny)]
        fp2 = i > 1  ? w_in[foff + ad_ve_fpop(i-1, j, 2, Nx, Ny)] : w_in[foff + ad_ve_fpop(i, j, 4, Nx, Ny)]
        fp3 = j > 1  ? w_in[foff + ad_ve_fpop(i, j-1, 3, Nx, Ny)] : w_in[foff + ad_ve_fpop(i, j, 5, Nx, Ny)]
        fp4 = i < Nx ? w_in[foff + ad_ve_fpop(i+1, j, 4, Nx, Ny)] : w_in[foff + ad_ve_fpop(i, j, 2, Nx, Ny)]
        fp5 = j < Ny ? w_in[foff + ad_ve_fpop(i, j+1, 5, Nx, Ny)] : w_in[foff + ad_ve_fpop(i, j, 3, Nx, Ny)]
        fp6 = (i > 1 && j > 1)   ? w_in[foff + ad_ve_fpop(i-1, j-1, 6, Nx, Ny)] : w_in[foff + ad_ve_fpop(i, j, 8, Nx, Ny)]
        fp7 = (i < Nx && j > 1)  ? w_in[foff + ad_ve_fpop(i+1, j-1, 7, Nx, Ny)] : w_in[foff + ad_ve_fpop(i, j, 9, Nx, Ny)]
        fp8 = (i < Nx && j < Ny) ? w_in[foff + ad_ve_fpop(i+1, j+1, 8, Nx, Ny)] : w_in[foff + ad_ve_fpop(i, j, 6, Nx, Ny)]
        fp9 = (i > 1 && j < Ny)  ? w_in[foff + ad_ve_fpop(i-1, j+1, 9, Nx, Ny)] : w_in[foff + ad_ve_fpop(i, j, 7, Nx, Ny)]

        h1 = w_in[foff + ad_ve_fpop(i, j, 1, Nx, Ny)]
        h2 = w_in[foff + ad_ve_fpop(i, j, 2, Nx, Ny)]
        h3 = w_in[foff + ad_ve_fpop(i, j, 3, Nx, Ny)]
        h4 = w_in[foff + ad_ve_fpop(i, j, 4, Nx, Ny)]
        h5 = w_in[foff + ad_ve_fpop(i, j, 5, Nx, Ny)]
        h6 = w_in[foff + ad_ve_fpop(i, j, 6, Nx, Ny)]
        h7 = w_in[foff + ad_ve_fpop(i, j, 7, Nx, Ny)]
        h8 = w_in[foff + ad_ve_fpop(i, j, 8, Nx, Ny)]
        h9 = w_in[foff + ad_ve_fpop(i, j, 9, Nx, Ny)]
        qw2 = q_wall[i, j, 2]; (qw2 > 0.0) && (fp4 = ad_ve_libb_branch(qw2, h2, fp2, h4))
        qw4 = q_wall[i, j, 4]; (qw4 > 0.0) && (fp2 = ad_ve_libb_branch(qw4, h4, fp4, h2))
        qw3 = q_wall[i, j, 3]; (qw3 > 0.0) && (fp5 = ad_ve_libb_branch(qw3, h3, fp3, h5))
        qw5 = q_wall[i, j, 5]; (qw5 > 0.0) && (fp3 = ad_ve_libb_branch(qw5, h5, fp5, h3))
        qw6 = q_wall[i, j, 6]; (qw6 > 0.0) && (fp8 = ad_ve_libb_branch(qw6, h6, fp6, h8))
        qw8 = q_wall[i, j, 8]; (qw8 > 0.0) && (fp6 = ad_ve_libb_branch(qw8, h8, fp8, h6))
        qw7 = q_wall[i, j, 7]; (qw7 > 0.0) && (fp9 = ad_ve_libb_branch(qw7, h7, fp7, h9))
        qw9 = q_wall[i, j, 9]; (qw9 > 0.0) && (fp7 = ad_ve_libb_branch(qw9, h9, fp9, h7))

        rho = fp1 + fp2 + fp3 + fp4 + fp5 + fp6 + fp7 + fp8 + fp9
        inv_rho = 1.0 / rho
        ux_raw = (fp2 - fp4 + fp6 - fp7 - fp8 + fp9) * inv_rho
        uy_raw = (fp3 - fp5 + fp6 + fp7 - fp8 - fp9) * inv_rho

        fx = fx_total[i, j]; fy = fy_total[i, j]
        ux_c = ux_raw; uy_c = uy_raw
        if fx != 0.0 || fy != 0.0
            ux_c = (rho * ux_raw + fx / 2.0) * inv_rho
            uy_c = (rho * uy_raw + fy / 2.0) * inv_rho
        end
        usq = ux_c * ux_c + uy_c * uy_c
        feq1 = ad_ve_feq(1, rho, ux_c, uy_c, usq)
        feq2 = ad_ve_feq(2, rho, ux_c, uy_c, usq)
        feq3 = ad_ve_feq(3, rho, ux_c, uy_c, usq)
        feq4 = ad_ve_feq(4, rho, ux_c, uy_c, usq)
        feq5 = ad_ve_feq(5, rho, ux_c, uy_c, usq)
        feq6 = ad_ve_feq(6, rho, ux_c, uy_c, usq)
        feq7 = ad_ve_feq(7, rho, ux_c, uy_c, usq)
        feq8 = ad_ve_feq(8, rho, ux_c, uy_c, usq)
        feq9 = ad_ve_feq(9, rho, ux_c, uy_c, usq)
        a = 0.5 * (sp + sm)
        b = 0.5 * (sp - sm)
        # Split Guo source, bit mirror of CollideTRTDirectGuoField (bricks.jl):
        # even part S⁺ at (1 - sp/2), odd part S⁻ = 3 w c·F at (1 - sm/2)
        # (Silva 2020, doi:10.1016/j.compfluid.2020.104537). Checked against the
        # brick in test/analytical/trt_guo_field_momentum.jl.
        guo_even = 1.0 - sp / 2.0
        guo_odd = 1.0 - sm / 2.0
        uF = ux_c * fx + uy_c * fy

        Sp1 = (4.0 / 9.0) * (-uF) * 3.0
        Sp24 = (1.0 / 9.0) * (ux_c * fx * 9.0 - uF * 3.0)
        Sm24 = (1.0 / 9.0) * fx * 3.0
        Sp35 = (1.0 / 9.0) * (uy_c * fy * 9.0 - uF * 3.0)
        Sm35 = (1.0 / 9.0) * fy * 3.0
        Sp68 = (1.0 / 36.0) * ((ux_c + uy_c) * (fx + fy) * 9.0 - uF * 3.0)
        Sm68 = (1.0 / 36.0) * (fx + fy) * 3.0
        Sp79 = (1.0 / 36.0) * ((-ux_c + uy_c) * (-fx + fy) * 9.0 - uF * 3.0)
        Sm79 = (1.0 / 36.0) * (-fx + fy) * 3.0

        w_out[foff + ad_ve_fpop(i, j, 1, Nx, Ny)] = fp1 - sp*(fp1-feq1) + guo_even*Sp1
        w_out[foff + ad_ve_fpop(i, j, 2, Nx, Ny)] = fp2 - a*(fp2-feq2) - b*(fp4-feq4) + guo_even*Sp24 + guo_odd*Sm24
        w_out[foff + ad_ve_fpop(i, j, 4, Nx, Ny)] = fp4 - a*(fp4-feq4) - b*(fp2-feq2) + guo_even*Sp24 - guo_odd*Sm24
        w_out[foff + ad_ve_fpop(i, j, 3, Nx, Ny)] = fp3 - a*(fp3-feq3) - b*(fp5-feq5) + guo_even*Sp35 + guo_odd*Sm35
        w_out[foff + ad_ve_fpop(i, j, 5, Nx, Ny)] = fp5 - a*(fp5-feq5) - b*(fp3-feq3) + guo_even*Sp35 - guo_odd*Sm35
        w_out[foff + ad_ve_fpop(i, j, 6, Nx, Ny)] = fp6 - a*(fp6-feq6) - b*(fp8-feq8) + guo_even*Sp68 + guo_odd*Sm68
        w_out[foff + ad_ve_fpop(i, j, 8, Nx, Ny)] = fp8 - a*(fp8-feq8) - b*(fp6-feq6) + guo_even*Sp68 - guo_odd*Sm68
        w_out[foff + ad_ve_fpop(i, j, 7, Nx, Ny)] = fp7 - a*(fp7-feq7) - b*(fp9-feq9) + guo_even*Sp79 + guo_odd*Sm79
        w_out[foff + ad_ve_fpop(i, j, 9, Nx, Ny)] = fp9 - a*(fp9-feq9) - b*(fp7-feq7) + guo_even*Sp79 - guo_odd*Sm79
    end

    # ============================================================
    # (3) FUSED ZouHe rebuild on f cols i=1 (west velocity) and i=Nx (east pressure),
    # rows j=2..Ny-1, reading PRE-stream f (w_in). Mirrors apply_bc_rebuild_2d!.
    # ============================================================
    @inbounds for j in 2:(Ny-1)
        # WEST velocity (i=1): unknown q=2,6,9 after pull
        fp1 = w_in[foff + ad_ve_fpop(1, j,   1, Nx, Ny)]
        fp3 = w_in[foff + ad_ve_fpop(1, j-1, 3, Nx, Ny)]
        fp4 = w_in[foff + ad_ve_fpop(2, j,   4, Nx, Ny)]
        fp5 = w_in[foff + ad_ve_fpop(1, j+1, 5, Nx, Ny)]
        fp7 = w_in[foff + ad_ve_fpop(2, j-1, 7, Nx, Ny)]
        fp8 = w_in[foff + ad_ve_fpop(2, j+1, 8, Nx, Ny)]
        u_in = u_profile[j]
        rho_w = (fp1 + fp3 + fp5 + 2.0*(fp4 + fp7 + fp8)) / (1.0 - u_in)
        fp2 = fp4 + (2.0/3.0)*rho_w*u_in
        fp6 = fp8 - 0.5*(fp3 - fp5) + (1.0/6.0)*rho_w*u_in
        fp9 = fp7 + 0.5*(fp3 - fp5) + (1.0/6.0)*rho_w*u_in
        F = ad_ve_trt_collide_local(fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, sp, sm)
        w_out[foff + ad_ve_fpop(1, j, 1, Nx, Ny)] = F[1]
        w_out[foff + ad_ve_fpop(1, j, 2, Nx, Ny)] = F[2]
        w_out[foff + ad_ve_fpop(1, j, 3, Nx, Ny)] = F[3]
        w_out[foff + ad_ve_fpop(1, j, 4, Nx, Ny)] = F[4]
        w_out[foff + ad_ve_fpop(1, j, 5, Nx, Ny)] = F[5]
        w_out[foff + ad_ve_fpop(1, j, 6, Nx, Ny)] = F[6]
        w_out[foff + ad_ve_fpop(1, j, 7, Nx, Ny)] = F[7]
        w_out[foff + ad_ve_fpop(1, j, 8, Nx, Ny)] = F[8]
        w_out[foff + ad_ve_fpop(1, j, 9, Nx, Ny)] = F[9]
        # EAST pressure (i=Nx): unknown q=4,7,8 after pull
        gp1 = w_in[foff + ad_ve_fpop(Nx,   j,   1, Nx, Ny)]
        gp2 = w_in[foff + ad_ve_fpop(Nx-1, j,   2, Nx, Ny)]
        gp3 = w_in[foff + ad_ve_fpop(Nx,   j-1, 3, Nx, Ny)]
        gp5 = w_in[foff + ad_ve_fpop(Nx,   j+1, 5, Nx, Ny)]
        gp6 = w_in[foff + ad_ve_fpop(Nx-1, j-1, 6, Nx, Ny)]
        gp9 = w_in[foff + ad_ve_fpop(Nx-1, j+1, 9, Nx, Ny)]
        u_x = -1.0 + (gp1 + gp3 + gp5 + 2.0*(gp2 + gp6 + gp9)) / rho_out
        gp4 = gp2 - (2.0/3.0)*rho_out*u_x
        gp7 = gp9 - 0.5*(gp3 - gp5) - (1.0/6.0)*rho_out*u_x
        gp8 = gp6 + 0.5*(gp3 - gp5) - (1.0/6.0)*rho_out*u_x
        G = ad_ve_trt_collide_local(gp1, gp2, gp3, gp4, gp5, gp6, gp7, gp8, gp9, sp, sm)
        w_out[foff + ad_ve_fpop(Nx, j, 1, Nx, Ny)] = G[1]
        w_out[foff + ad_ve_fpop(Nx, j, 2, Nx, Ny)] = G[2]
        w_out[foff + ad_ve_fpop(Nx, j, 3, Nx, Ny)] = G[3]
        w_out[foff + ad_ve_fpop(Nx, j, 4, Nx, Ny)] = G[4]
        w_out[foff + ad_ve_fpop(Nx, j, 5, Nx, Ny)] = G[5]
        w_out[foff + ad_ve_fpop(Nx, j, 6, Nx, Ny)] = G[6]
        w_out[foff + ad_ve_fpop(Nx, j, 7, Nx, Ny)] = G[7]
        w_out[foff + ad_ve_fpop(Nx, j, 8, Nx, Ny)] = G[8]
        w_out[foff + ad_ve_fpop(Nx, j, 9, Nx, Ny)] = G[9]
    end

    # ---- pack psi' ----
    @inbounds for j in 1:Ny, i in 1:Nx
        k = ad_ve_lin(i, j, Nx)
        w_out[poff + k]      = psixx_p[i, j]
        w_out[poff + n + k]  = psixy_p[i, j]
        w_out[poff + 2n + k] = psiyy_p[i, j]
    end
    return nothing
end
