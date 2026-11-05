struct ADNatconvParams
    Nx::Int
    Ny::Int
    omega_f::Float64
    omega_T::Float64
    beta_g::Float64
    T_ref::Float64
    T_hot::Float64
    T_cold::Float64
    Ra::Float64
    Pr::Float64
end

function ad_natconv_params(; N::Int, Ra::Real, Pr::Real,
                           T_hot::Real=1.0, T_cold::Real=0.0)
    nu = 0.05
    alpha = nu / Float64(Pr)
    delta_T = Float64(T_hot) - Float64(T_cold)
    H = Float64(N)
    beta_g = Float64(Ra) * nu * alpha / (delta_T * H^3)
    omega_f = 1.0 / (3.0 * nu + 0.5)
    omega_T = 1.0 / (3.0 * alpha + 0.5)
    T_ref = 0.5 * (Float64(T_hot) + Float64(T_cold))
    return ADNatconvParams(N, N, omega_f, omega_T, beta_g, T_ref,
                           Float64(T_hot), Float64(T_cold),
                           Float64(Ra), Float64(Pr))
end

@inline ad_thermal_nlat(p::ADNatconvParams) = p.Nx * p.Ny * 9
@inline function ad_thermal_popidx(i::Int, j::Int, q::Int, Nx::Int, Ny::Int)
    return i + (j - 1) * Nx + (q - 1) * Nx * Ny
end

@inline ad_thermal_readpop(w, offset::Int, i::Int, j::Int, q::Int,
                           Nx::Int, Ny::Int) =
    w[offset + ad_thermal_popidx(i, j, q, Nx, Ny)]

@inline function ad_thermal_writepops!(w, offset::Int, i::Int, j::Int,
                                       Nx::Int, Ny::Int,
                                       p1, p2, p3, p4, p5, p6, p7, p8, p9)
    w[offset + ad_thermal_popidx(i, j, 1, Nx, Ny)] = p1
    w[offset + ad_thermal_popidx(i, j, 2, Nx, Ny)] = p2
    w[offset + ad_thermal_popidx(i, j, 3, Nx, Ny)] = p3
    w[offset + ad_thermal_popidx(i, j, 4, Nx, Ny)] = p4
    w[offset + ad_thermal_popidx(i, j, 5, Nx, Ny)] = p5
    w[offset + ad_thermal_popidx(i, j, 6, Nx, Ny)] = p6
    w[offset + ad_thermal_popidx(i, j, 7, Nx, Ny)] = p7
    w[offset + ad_thermal_popidx(i, j, 8, Nx, Ny)] = p8
    w[offset + ad_thermal_popidx(i, j, 9, Nx, Ny)] = p9
    return nothing
end

@inline function ad_thermal_stream_pull_flat(w, offset::Int, i::Int, j::Int,
                                             Nx::Int, Ny::Int)
    fp1 = ad_thermal_readpop(w, offset, i, j, 1, Nx, Ny)
    fp2 = i > 1 ? ad_thermal_readpop(w, offset, i - 1, j, 2, Nx, Ny) :
          ad_thermal_readpop(w, offset, i, j, 4, Nx, Ny)
    fp3 = j > 1 ? ad_thermal_readpop(w, offset, i, j - 1, 3, Nx, Ny) :
          ad_thermal_readpop(w, offset, i, j, 5, Nx, Ny)
    fp4 = i < Nx ? ad_thermal_readpop(w, offset, i + 1, j, 4, Nx, Ny) :
          ad_thermal_readpop(w, offset, i, j, 2, Nx, Ny)
    fp5 = j < Ny ? ad_thermal_readpop(w, offset, i, j + 1, 5, Nx, Ny) :
          ad_thermal_readpop(w, offset, i, j, 3, Nx, Ny)
    fp6 = (i > 1 && j > 1) ?
          ad_thermal_readpop(w, offset, i - 1, j - 1, 6, Nx, Ny) :
          ad_thermal_readpop(w, offset, i, j, 8, Nx, Ny)
    fp7 = (i < Nx && j > 1) ?
          ad_thermal_readpop(w, offset, i + 1, j - 1, 7, Nx, Ny) :
          ad_thermal_readpop(w, offset, i, j, 9, Nx, Ny)
    fp8 = (i < Nx && j < Ny) ?
          ad_thermal_readpop(w, offset, i + 1, j + 1, 8, Nx, Ny) :
          ad_thermal_readpop(w, offset, i, j, 6, Nx, Ny)
    fp9 = (i > 1 && j < Ny) ?
          ad_thermal_readpop(w, offset, i - 1, j + 1, 9, Nx, Ny) :
          ad_thermal_readpop(w, offset, i, j, 7, Nx, Ny)
    return fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9
end

# Thermal pull of `stream_pull_thermal_node` (src/kernels/fused_thermal_2d.jl):
# the adiabatic top and bottom walls reflect the diagonal populations
# specularly (zero normal heat flux, tangential flux kept), the slots whose
# source lies across the west or east wall keep the bounced value for the
# Dirichlet rule.
@inline function ad_thermal_g_stream_pull_flat(w, offset::Int, i::Int, j::Int,
                                               Nx::Int, Ny::Int)
    jm = max(j - 1, 1)
    jp = min(j + 1, Ny)
    gp1 = ad_thermal_readpop(w, offset, i, j, 1, Nx, Ny)
    gp2 = i > 1 ? ad_thermal_readpop(w, offset, i - 1, j, 2, Nx, Ny) :
          ad_thermal_readpop(w, offset, i, j, 4, Nx, Ny)
    gp3 = j > 1 ? ad_thermal_readpop(w, offset, i, j - 1, 3, Nx, Ny) :
          ad_thermal_readpop(w, offset, i, j, 5, Nx, Ny)
    gp4 = i < Nx ? ad_thermal_readpop(w, offset, i + 1, j, 4, Nx, Ny) :
          ad_thermal_readpop(w, offset, i, j, 2, Nx, Ny)
    gp5 = j < Ny ? ad_thermal_readpop(w, offset, i, j + 1, 5, Nx, Ny) :
          ad_thermal_readpop(w, offset, i, j, 3, Nx, Ny)
    gp6 = i > 1 ?
          ad_thermal_readpop(w, offset, i - 1, jm, j > 1 ? 6 : 9, Nx, Ny) :
          ad_thermal_readpop(w, offset, i, j, 8, Nx, Ny)
    gp7 = i < Nx ?
          ad_thermal_readpop(w, offset, i + 1, jm, j > 1 ? 7 : 8, Nx, Ny) :
          ad_thermal_readpop(w, offset, i, j, 9, Nx, Ny)
    gp8 = i < Nx ?
          ad_thermal_readpop(w, offset, i + 1, jp, j < Ny ? 8 : 7, Nx, Ny) :
          ad_thermal_readpop(w, offset, i, j, 6, Nx, Ny)
    gp9 = i > 1 ?
          ad_thermal_readpop(w, offset, i - 1, jp, j < Ny ? 9 : 6, Nx, Ny) :
          ad_thermal_readpop(w, offset, i, j, 7, Nx, Ny)
    return gp1, gp2, gp3, gp4, gp5, gp6, gp7, gp8, gp9
end

@inline function ad_dirichlet_cutlink_ghost(temp_here, temp_wall, q_wall_link)
    T = typeof(temp_here)
    q = T(q_wall_link)
    return (T(temp_wall) - (one(T) - q) * temp_here) / q
end

# Cut-link anti-bounce-back for a Dirichlet wall at fraction `q_wall_link` of
# the link from the boundary node. `g_bounced` is the post-collision population
# that left the boundary node towards the wall, which the bounce-back pull has
# already placed in the unknown slot. The wall term interpolates the
# temperature linearly to the link midpoint, T_mid = (temp_here + ghost) / 2,
# so q = 0.5 reduces to the half-way rule of the fused kernel,
# g_q = -g*_opp + 2 w_q T_wall.
@inline function ad_dirichlet_cutlink_pop(g_bounced, temp_here, temp_wall,
                                          q_wall_link, weight)
    T = typeof(g_bounced)
    ghost = ad_dirichlet_cutlink_ghost(temp_here, temp_wall, q_wall_link)
    return -g_bounced + T(weight) * (temp_here + ghost)
end

function ad_initial_thermal_w(p::ADNatconvParams, L::Real, q_hot::Real)
    Nx = p.Nx
    Ny = p.Ny
    goff = ad_thermal_nlat(p)
    w = zeros(Float64, 2 * goff)
    delta_T = p.T_hot - p.T_cold
    @inbounds for j in 1:Ny, i in 1:Nx
        for q in 1:9
            w[ad_thermal_popidx(i, j, q, Nx, Ny)] =
                ad_feq(Val(q), 1.0, 0.0, 0.0, 0.0)
        end
        x = Float64(q_hot) + Float64(i - 1)
        T_init = p.T_hot - delta_T * x / Float64(L)
        T_init += 0.01 * delta_T * sin(2 * pi * i / Nx) * sin(pi * j / Ny)
        for q in 1:9
            w[goff + ad_thermal_popidx(i, j, q, Nx, Ny)] =
                ad_qwgt(Val(q)) * T_init
        end
    end
    return w
end

@inline function ad_apply_flow_cutlinks_flat(w, offset::Int, i::Int, j::Int,
                                             Nx::Int, Ny::Int, q_flow,
                                             fp1, fp2, fp3, fp4, fp5,
                                             fp6, fp7, fp8, fp9)
    qw2 = q_flow[i, j, 2]
    if qw2 > 0.0
        fp4 = ad_libb_branch(qw2,
                             ad_thermal_readpop(w, offset, i, j, 2, Nx, Ny),
                             fp2,
                             ad_thermal_readpop(w, offset, i, j, 4, Nx, Ny))
    end
    qw4 = q_flow[i, j, 4]
    if qw4 > 0.0
        fp2 = ad_libb_branch(qw4,
                             ad_thermal_readpop(w, offset, i, j, 4, Nx, Ny),
                             fp4,
                             ad_thermal_readpop(w, offset, i, j, 2, Nx, Ny))
    end
    qw6 = q_flow[i, j, 6]
    if qw6 > 0.0
        fp8 = ad_libb_branch(qw6,
                             ad_thermal_readpop(w, offset, i, j, 6, Nx, Ny),
                             fp6,
                             ad_thermal_readpop(w, offset, i, j, 8, Nx, Ny))
    end
    qw8 = q_flow[i, j, 8]
    if qw8 > 0.0
        fp6 = ad_libb_branch(qw8,
                             ad_thermal_readpop(w, offset, i, j, 8, Nx, Ny),
                             fp8,
                             ad_thermal_readpop(w, offset, i, j, 6, Nx, Ny))
    end
    qw7 = q_flow[i, j, 7]
    if qw7 > 0.0
        fp9 = ad_libb_branch(qw7,
                             ad_thermal_readpop(w, offset, i, j, 7, Nx, Ny),
                             fp7,
                             ad_thermal_readpop(w, offset, i, j, 9, Nx, Ny))
    end
    qw9 = q_flow[i, j, 9]
    if qw9 > 0.0
        fp7 = ad_libb_branch(qw9,
                             ad_thermal_readpop(w, offset, i, j, 9, Nx, Ny),
                             fp9,
                             ad_thermal_readpop(w, offset, i, j, 7, Nx, Ny))
    end
    return fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9
end

@inline function ad_thermal_bc_cut_cavity(i::Int, j::Int, Nx::Int, q_therm,
                                          T_hot, T_cold,
                                          g1, g2, g3, g4, g5, g6, g7, g8, g9)
    temp_here = g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 + g9
    # After the thermal pull, each unknown slot (2, 6, 9 at the west wall;
    # 4, 7, 8 at the east wall) holds the bounced post-collision value of its
    # opposite direction, which is the population the cut-link rule reflects.
    # q_therm is indexed by the outgoing (wall-facing) direction.
    if i == 1
        g2 = ad_dirichlet_cutlink_pop(g2, temp_here, T_hot,
                                      q_therm[1, j, 4], 1.0 / 9.0)
        g6 = ad_dirichlet_cutlink_pop(g6, temp_here, T_hot,
                                      q_therm[1, j, 8], 1.0 / 36.0)
        g9 = ad_dirichlet_cutlink_pop(g9, temp_here, T_hot,
                                      q_therm[1, j, 7], 1.0 / 36.0)
    end
    if i == Nx
        g4 = ad_dirichlet_cutlink_pop(g4, temp_here, T_cold,
                                      q_therm[Nx, j, 2], 1.0 / 9.0)
        g7 = ad_dirichlet_cutlink_pop(g7, temp_here, T_cold,
                                      q_therm[Nx, j, 9], 1.0 / 36.0)
        g8 = ad_dirichlet_cutlink_pop(g8, temp_here, T_cold,
                                      q_therm[Nx, j, 6], 1.0 / 36.0)
    end
    return g1, g2, g3, g4, g5, g6, g7, g8, g9
end

@inline function ad_macroscopic_boussinesq(T_local, beta_g, T_ref,
                                           f1, f2, f3, f4, f5,
                                           f6, f7, f8, f9)
    T = typeof(f1)
    fy = T(beta_g) * (T_local - T(T_ref))
    rho = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
    inv_rho = one(T) / rho
    ux = (f2 - f4 + f6 - f7 - f8 + f9) * inv_rho
    uy = ((f3 - f5 + f6 + f7 - f8 - f9) + fy / T(2)) * inv_rho
    return rho, ux, uy, fy
end

@inline function ad_collide_thermal_node(g1, g2, g3, g4, g5, g6, g7, g8, g9,
                                         ux, uy, omega_T)
    T = typeof(g1)
    omega = T(omega_T)
    temp = g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 + g9
    t3 = T(3)
    o1 = g1 - omega * (g1 - T(4.0 / 9.0) * temp)
    o2 = g2 - omega * (g2 - T(1.0 / 9.0) * temp * (one(T) + t3 * ux))
    o3 = g3 - omega * (g3 - T(1.0 / 9.0) * temp * (one(T) + t3 * uy))
    o4 = g4 - omega * (g4 - T(1.0 / 9.0) * temp * (one(T) - t3 * ux))
    o5 = g5 - omega * (g5 - T(1.0 / 9.0) * temp * (one(T) - t3 * uy))
    o6 = g6 - omega * (g6 - T(1.0 / 36.0) * temp * (one(T) + t3 * (ux + uy)))
    o7 = g7 - omega * (g7 - T(1.0 / 36.0) * temp * (one(T) + t3 * (-ux + uy)))
    o8 = g8 - omega * (g8 - T(1.0 / 36.0) * temp * (one(T) + t3 * (-ux - uy)))
    o9 = g9 - omega * (g9 - T(1.0 / 36.0) * temp * (one(T) + t3 * (ux - uy)))
    return o1, o2, o3, o4, o5, o6, o7, o8, o9
end

@inline function ad_guo_source(::Val{Q}, omega::T, fx::T, fy::T,
                               ux::T, uy::T) where {Q,T}
    cx = T(ad_qcx(Val(Q)))
    cy = T(ad_qcy(Val(Q)))
    wq = T(ad_qwgt(Val(Q)))
    cu_force = (cx - ux) * fx + (cy - uy) * fy
    cuf = (cx * ux + cy * uy) * (cx * fx + cy * fy)
    return (one(T) - omega / T(2)) * wq * (T(3) * cu_force + T(9) * cuf)
end

@inline function ad_collide_boussinesq_node(f1, f2, f3, f4, f5, f6, f7, f8, f9,
                                            rho, ux, uy, fy, omega_f)
    T = typeof(f1)
    omega = T(omega_f)
    fx = zero(T)
    usq = ux * ux + uy * uy
    o1 = f1 - omega * (f1 - ad_feq(Val(1), rho, ux, uy, usq)) +
         ad_guo_source(Val(1), omega, fx, fy, ux, uy)
    o2 = f2 - omega * (f2 - ad_feq(Val(2), rho, ux, uy, usq)) +
         ad_guo_source(Val(2), omega, fx, fy, ux, uy)
    o3 = f3 - omega * (f3 - ad_feq(Val(3), rho, ux, uy, usq)) +
         ad_guo_source(Val(3), omega, fx, fy, ux, uy)
    o4 = f4 - omega * (f4 - ad_feq(Val(4), rho, ux, uy, usq)) +
         ad_guo_source(Val(4), omega, fx, fy, ux, uy)
    o5 = f5 - omega * (f5 - ad_feq(Val(5), rho, ux, uy, usq)) +
         ad_guo_source(Val(5), omega, fx, fy, ux, uy)
    o6 = f6 - omega * (f6 - ad_feq(Val(6), rho, ux, uy, usq)) +
         ad_guo_source(Val(6), omega, fx, fy, ux, uy)
    o7 = f7 - omega * (f7 - ad_feq(Val(7), rho, ux, uy, usq)) +
         ad_guo_source(Val(7), omega, fx, fy, ux, uy)
    o8 = f8 - omega * (f8 - ad_feq(Val(8), rho, ux, uy, usq)) +
         ad_guo_source(Val(8), omega, fx, fy, ux, uy)
    o9 = f9 - omega * (f9 - ad_feq(Val(9), rho, ux, uy, usq)) +
         ad_guo_source(Val(9), omega, fx, fy, ux, uy)
    return o1, o2, o3, o4, o5, o6, o7, o8, o9
end

function ad_thermal_cut_step!(w_out, w, q_flow, q_therm, p::ADNatconvParams)
    Nx = p.Nx
    Ny = p.Ny
    goff = ad_thermal_nlat(p)
    @inbounds for j in 1:Ny, i in 1:Nx
        f1, f2, f3, f4, f5, f6, f7, f8, f9 =
            ad_thermal_stream_pull_flat(w, 0, i, j, Nx, Ny)
        f1, f2, f3, f4, f5, f6, f7, f8, f9 =
            ad_apply_flow_cutlinks_flat(w, 0, i, j, Nx, Ny, q_flow,
                                        f1, f2, f3, f4, f5, f6, f7, f8, f9)

        g1, g2, g3, g4, g5, g6, g7, g8, g9 =
            ad_thermal_g_stream_pull_flat(w, goff, i, j, Nx, Ny)
        g1, g2, g3, g4, g5, g6, g7, g8, g9 =
            ad_thermal_bc_cut_cavity(i, j, Nx, q_therm, p.T_hot, p.T_cold,
                                     g1, g2, g3, g4, g5, g6, g7, g8, g9)

        T_local = g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 + g9
        rho, ux, uy, fy = ad_macroscopic_boussinesq(T_local, p.beta_g, p.T_ref,
                                                     f1, f2, f3, f4, f5,
                                                     f6, f7, f8, f9)

        o_g1, o_g2, o_g3, o_g4, o_g5, o_g6, o_g7, o_g8, o_g9 =
            ad_collide_thermal_node(g1, g2, g3, g4, g5, g6, g7, g8, g9,
                                    ux, uy, p.omega_T)
        ad_thermal_writepops!(w_out, goff, i, j, Nx, Ny,
                              o_g1, o_g2, o_g3, o_g4, o_g5,
                              o_g6, o_g7, o_g8, o_g9)

        o_f1, o_f2, o_f3, o_f4, o_f5, o_f6, o_f7, o_f8, o_f9 =
            ad_collide_boussinesq_node(f1, f2, f3, f4, f5, f6, f7, f8, f9,
                                       rho, ux, uy, fy, p.omega_f)
        ad_thermal_writepops!(w_out, 0, i, j, Nx, Ny,
                              o_f1, o_f2, o_f3, o_f4, o_f5,
                              o_f6, o_f7, o_f8, o_f9)
    end
    return nothing
end

function ad_thermal_mass_gradient(p::ADNatconvParams)
    nf = ad_thermal_nlat(p)
    m = zeros(Float64, 2 * nf)
    @inbounds for idx in 1:nf
        m[idx] = 1.0
    end
    return m
end
