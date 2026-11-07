# Reference oracle for EHD electroconvection parity tests.
#
# This is a deliberate verbatim transcription of the reference MATLAB
# electroconvection LBM loop used by Luo et al., Phys. Rev. E 93, 023309:
#   standard_LBM_EC/run_standard_LBM_electroconvection.m
#   standard_LBM_EC/Lattice.m
#
# It is kept independent of KrakeNN internals ON PURPOSE. Its value as an
# oracle depends on it NOT sharing code with src/.
#
# Do NOT refactor this file to reuse KrakeNN kernels/helpers, even for
# "obvious" deduplication, without deliberately re-deriving parity from the
# MATLAB source again.

using LinearAlgebra

const EHD_CX = (0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0)
const EHD_CY = (0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0)
const EHD_CXI = (0, 1, 0, -1, 0, 1, -1, -1, 1)
const EHD_CYI = (0, 0, 1, 0, -1, 1, 1, -1, -1)
const EHD_W = (4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36)
const EHD_OPP = (1, 4, 5, 2, 3, 8, 9, 6, 7)
const EHD_TMRT = [
     1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0;
    -4.0 -1.0 -1.0 -1.0 -1.0  2.0  2.0  2.0  2.0;
     4.0 -2.0 -2.0 -2.0 -2.0  1.0  1.0  1.0  1.0;
     0.0  1.0  0.0 -1.0  0.0  1.0 -1.0 -1.0  1.0;
     0.0 -2.0  0.0  2.0  0.0  1.0 -1.0 -1.0  1.0;
     0.0  0.0  1.0  0.0 -1.0  1.0  1.0 -1.0 -1.0;
     0.0  0.0 -2.0  0.0  2.0  1.0  1.0 -1.0 -1.0;
     0.0  1.0 -1.0  1.0 -1.0  0.0  0.0  0.0  0.0;
     0.0  0.0  0.0  0.0  0.0  1.0 -1.0  1.0 -1.0
]

mutable struct EHDTwinState{T}
    Nx::Int
    Ny::Int
    p::NamedTuple
    wall::Matrix{Bool}
    bb_links::Vector{Tuple{Int,Int,Int,Int}}
    phi_f::Array{T,3}
    phi_tmp::Array{T,3}
    phi::Matrix{T}
    Ex::Matrix{T}
    Ey::Matrix{T}
    q_f::Array{T,3}
    q_tmp::Array{T,3}
    q::Matrix{T}
    rho::Matrix{T}
    ux::Matrix{T}
    uy::Matrix{T}
    ns_f::Array{T,3}
    ns_tmp::Array{T,3}
    Fx_slots::Array{T,3}
    Fy_slots::Array{T,3}
    mrt_M::Matrix{T}
    mrt_N::Matrix{T}
end

function ehd_ec_lattice_params(Ny, C, M, T_ehd, Ma_E, alpha, delta_U, gamma; FT=Float64)
    H = FT(Ny - 1)
    cs = inv(sqrt(FT(3)))
    K = FT(Ma_E) * H * cs / FT(delta_U)
    nu = FT(M)^2 * K * FT(delta_U) / FT(T_ehd)
    tau = FT(0.5) + FT(3) * nu
    eps_e = (FT(M) * K)^2
    q_inj = FT(C) * eps_e * FT(delta_U) / H^2
    D = FT(alpha) * K * FT(delta_U)
    tau_U = FT(0.5) + FT(3) * FT(gamma)
    tau_q = FT(0.5) + FT(3) * D
    dt_star = K * FT(delta_U) / H^2
    return (H=H, cs=cs, K=K, nu=nu, tau=tau, omega=inv(tau),
            eps=eps_e, q_inj=q_inj, D=D, tau_U=tau_U, nu_U=FT(gamma),
            omega_U=inv(tau_U), tau_q=tau_q, omega_q=inv(tau_q),
            dt_star=dt_star, u_ref=K * FT(delta_U) / H)
end

function hydro_b_bisect(C; FT=Float64, tol=FT(1e-14))
    CT = FT(C)
    f(b) = (FT(4) * CT / FT(3)) * sqrt(b) * ((one(FT) + b)^FT(1.5) - b^FT(1.5)) - one(FT)
    lo = FT(1e-12)
    hi = one(FT)
    flo = f(lo)
    fhi = f(hi)
    while sign(flo) == sign(fhi) && hi < FT(1e12)
        hi *= FT(2)
        fhi = f(hi)
    end
    sign(flo) == sign(fhi) && error("Unable to bracket hydrostatic root.")
    while hi - lo > tol
        mid = (lo + hi) / FT(2)
        fm = f(mid)
        if sign(flo) == sign(fm)
            lo = mid
            flo = fm
        else
            hi = mid
        end
    end
    return (lo + hi) / FT(2)
end

function hydrostatic_profiles(C, Ny; FT=Float64)
    b = hydro_b_bisect(C; FT=FT)
    a = FT(2) * FT(C) * sqrt(b)
    y = [FT(j - 1) / FT(Ny - 1) for j in 1:Ny]
    q_star = [a / (FT(2) * FT(C) * sqrt(yj + b)) for yj in y]
    E_star = [a * sqrt(yj + b) for yj in y]
    phi = [one(FT) - (FT(2) * a / FT(3)) * ((yj + b)^FT(1.5) - b^FT(1.5)) for yj in y]
    return (y=y, b=b, a=a, q_star=q_star, E_star=E_star, phi=phi)
end

@inline function feq_pop(rho, ux, uy, qdir)
    cu = EHD_CX[qdir] * ux + EHD_CY[qdir] * uy
    usq = ux * ux + uy * uy
    return rho * EHD_W[qdir] * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * usq)
end

@inline charge_feq_pop(q, ux, uy, qdir) = feq_pop(q, ux, uy, qdir)

function build_bb_links(wall)
    Nx, Ny = size(wall)
    links = Tuple{Int,Int,Int,Int}[]
    for j in 1:Ny, i in 1:Nx, q in 1:9
        inb = mod1(i + EHD_CXI[q], Nx)
        jnb = mod1(j + EHD_CYI[q], Ny)
        wall[inb, jnb] && push!(links, (i, j, q, EHD_OPP[q]))
    end
    return links
end

function matlab_stream!(dst, src; bb_links=nothing)
    Nx, Ny, _ = size(src)
    @inbounds for q in 1:9, j in 1:Ny, i in 1:Nx
        dst[mod1(i + EHD_CXI[q], Nx), mod1(j + EHD_CYI[q], Ny), q] = src[i, j, q]
    end
    if bb_links !== nothing
        @inbounds for (i, j, q, qopp) in bb_links
            dst[i, j, qopp] = src[i, j, q]
        end
    end
    return dst
end

function scalar!(out, f)
    Nx, Ny, _ = size(f)
    @inbounds for j in 1:Ny, i in 1:Nx
        s = zero(eltype(f))
        for q in 1:9
            s += f[i, j, q]
        end
        out[i, j] = s
    end
    return out
end

function calculate_E!(Ex, Ey, f, tau_U)
    Nx, Ny, _ = size(f)
    invden = inv(tau_U * (one(eltype(f)) / 3))
    @inbounds for j in 1:Ny, i in 1:Nx
        Ex[i, j] = (f[i,j,2] - f[i,j,4] + f[i,j,6] - f[i,j,7] - f[i,j,8] + f[i,j,9]) * invden
        Ey[i, j] = (f[i,j,3] - f[i,j,5] + f[i,j,6] + f[i,j,7] - f[i,j,8] - f[i,j,9]) * invden
    end
    return Ex, Ey
end

function collide_phi!(f, qfield, eps_e, omega_U, nu_U)
    Nx, Ny, _ = size(f)
    @inbounds for j in 1:Ny, i in 1:Nx
        phi = zero(eltype(f))
        for q in 1:9
            phi += f[i, j, q]
        end
        src = nu_U * qfield[i, j] / eps_e
        for q in 1:9
            f[i, j, q] = f[i, j, q] - omega_U * (f[i, j, q] - EHD_W[q] * phi) + EHD_W[q] * src
        end
    end
    return f
end

function apply_phi_bc!(f, phi; phi_bottom=1.0, phi_top=0.0)
    Nx, Ny, _ = size(f)
    @inbounds begin
        for i in 2:Nx-1, q in 1:9
            f[i, 1, q] = f[i, 2, q] + EHD_W[q] * (phi_bottom - phi[i, 2])
            f[i, Ny, q] = f[i, Ny - 1, q] + EHD_W[q] * (phi_top - phi[i, Ny - 1])
        end
        for j in 1:Ny, q in 1:9
            f[1, j, q] = f[2, j, q]
            f[Nx, j, q] = f[Nx - 1, j, q]
        end
    end
    return f
end

function collide_charge_regularized!(f, ux, uy, Ex, Ey, tau_q, K)
    Nx, Ny, _ = size(f)
    pref = 1 - inv(tau_q)
    @inbounds for j in 1:Ny, i in 1:Nx
        qrho = zero(eltype(f))
        for q in 1:9
            qrho += f[i, j, q]
        end
        ueqx = ux[i, j] + K * Ex[i, j]
        ueqy = uy[i, j] + K * Ey[i, j]
        feq = ntuple(q -> charge_feq_pop(qrho, ueqx, ueqy, q), 9)
        jx = (f[i,j,2] - feq[2]) - (f[i,j,4] - feq[4]) + (f[i,j,6] - feq[6]) -
             (f[i,j,7] - feq[7]) - (f[i,j,8] - feq[8]) + (f[i,j,9] - feq[9])
        jy = (f[i,j,3] - feq[3]) - (f[i,j,5] - feq[5]) + (f[i,j,6] - feq[6]) +
             (f[i,j,7] - feq[7]) - (f[i,j,8] - feq[8]) - (f[i,j,9] - feq[9])
        for qdir in 1:9
            f[i, j, qdir] = feq[qdir] + pref * EHD_W[qdir] * 3 * (EHD_CX[qdir] * jx + EHD_CY[qdir] * jy)
        end
    end
    return f
end

function apply_charge_bc!(f, qfield, ux, uy, Ex, Ey, q_bottom, K)
    Nx, Ny, _ = size(f)
    @inline function set_node!(ib, jb, inb, jnb, qb)
        uxb = ux[ib, jb] + K * Ex[ib, jb]
        uyb = uy[ib, jb] + K * Ey[ib, jb]
        uxn = ux[inb, jnb] + K * Ex[inb, jnb]
        uyn = uy[inb, jnb] + K * Ey[inb, jnb]
        qn = qfield[inb, jnb]
        for qdir in 1:9
            f[ib, jb, qdir] = f[inb, jnb, qdir] +
                              charge_feq_pop(qb, uxb, uyb, qdir) -
                              charge_feq_pop(qn, uxn, uyn, qdir)
        end
    end
    @inbounds begin
        for i in 2:Nx-1
            set_node!(i, 1, i, 2, q_bottom)
            set_node!(i, Ny, i, Ny - 1, qfield[i, Ny - 1])
        end
        for j in 1:Ny
            set_node!(1, j, 2, j, qfield[2, j])
            set_node!(Nx, j, Nx - 1, j, qfield[Nx - 1, j])
        end
    end
    return f
end

function enforce_free_side_macros!(ux, uy)
    Nx, Ny = size(ux)
    @inbounds for j in 2:Ny-1
        ux[1, j] = 0
        ux[Nx, j] = 0
        uy[1, j] = uy[2, j]
        uy[Nx, j] = uy[Nx - 1, j]
    end
    return ux, uy
end

function macros_from_force!(rho, ux, uy, f, Fx_accel, Fy_accel, wall; enforce_free=true)
    Nx, Ny, _ = size(f)
    @inbounds for j in 1:Ny, i in 1:Nx
        r = zero(eltype(f))
        mx = zero(eltype(f))
        my = zero(eltype(f))
        for q in 1:9
            fq = f[i, j, q]
            r += fq
            mx += EHD_CX[q] * fq
            my += EHD_CY[q] * fq
        end
        rho[i, j] = r
        ux[i, j] = mx / r + 0.5 * Fx_accel[i, j]
        uy[i, j] = my / r + 0.5 * Fy_accel[i, j]
        if wall[i, j]
            ux[i, j] = 0
            uy[i, j] = 0
        end
    end
    enforce_free && enforce_free_side_macros!(ux, uy)
    return rho, ux, uy
end

function apply_ef!(Fx_slots, Fy_slots, Ex, Ey, qfield, rho, wall; projection_mode=:xy, force_ramp=1.0)
    Nx, Ny = size(qfield)
    Fx = @view Fx_slots[:, :, 2]
    Fy = @view Fy_slots[:, :, 2]
    @inbounds for j in 1:Ny, i in 1:Nx
        Fx[i, j] = force_ramp * Ex[i, j] * qfield[i, j] / rho[i, j]
        Fy[i, j] = force_ramp * Ey[i, j] * qfield[i, j] / rho[i, j]
    end
    if projection_mode != :none
        @inbounds for j in 1:Ny
            sx = zero(eltype(qfield))
            sy = zero(eltype(qfield))
            n = 0
            for i in 1:Nx
                if !wall[i, j]
                    sx += Fx[i, j]
                    sy += Fy[i, j]
                    n += 1
                end
            end
            if n > 0
                mx = sx / n
                my = sy / n
                for i in 1:Nx
                    if !wall[i, j]
                        projection_mode == :xy && (Fx[i, j] -= mx)
                        (projection_mode == :xy || projection_mode == :y) && (Fy[i, j] -= my)
                    end
                end
            end
        end
    end
    @inbounds for j in 1:Ny, i in 1:Nx
        if wall[i, j]
            Fx[i, j] = 0
            Fy[i, j] = 0
        end
        Fx_slots[i, j, 1] = 0
        Fy_slots[i, j, 1] = 0
    end
    return Fx_slots, Fy_slots
end

function collide_sp_mrt!(f, rho, ux, uy, Fx_slots, Fy_slots, wall, Mmat, Nmat)
    # MATLAB Lattice.m collide_SP branches on obj.M:
    # this transcribes the MRT-with-force branch used after d_NS.setMRT(tau).
    Nx, Ny, _ = size(f)
    cs2 = one(eltype(f)) / 3
    fvec = zeros(eltype(f), 9)
    feq = similar(fvec)
    df = similar(fvec)
    src = similar(fvec)
    @inbounds for j in 1:Ny, i in 1:Nx
        r = zero(eltype(f))
        mx = zero(eltype(f))
        my = zero(eltype(f))
        for q in 1:9
            fq = f[i, j, q]
            fvec[q] = fq
            r += fq
            mx += EHD_CX[q] * fq
            my += EHD_CY[q] * fq
        end
        fx = Fx_slots[i, j, 1] + Fx_slots[i, j, 2]
        fy = Fy_slots[i, j, 1] + Fy_slots[i, j, 2]
        uxi = mx / r + 0.5 * fx
        uyi = my / r + 0.5 * fy
        if wall[i, j]
            uxi = 0
            uyi = 0
        end
        rho[i, j] = r
        ux[i, j] = uxi
        uy[i, j] = uyi
        for q in 1:9
            feq[q] = feq_pop(r, uxi, uyi, q)
            df[q] = fvec[q] - feq[q]
            src[q] = (fx * (EHD_CX[q] - uxi) + fy * (EHD_CY[q] - uyi)) * feq[q]
        end
        out = fvec - Mmat * df + (Nmat * src) / cs2
        for q in 1:9
            f[i, j, q] = out[q]
        end
    end
    return f
end

function apply_free_slip_sidewalls!(f)
    Nx, Ny, _ = size(f)
    @inbounds for j in 2:Ny-1
        f[1, j, 2] = f[1, j, 4]
        f[1, j, 6] = f[1, j, 7]
        f[1, j, 9] = f[1, j, 8]
        f[Nx, j, 4] = f[Nx, j, 2]
        f[Nx, j, 7] = f[Nx, j, 6]
        f[Nx, j, 8] = f[Nx, j, 9]
    end
    return f
end

function init_twin_state(; Nx=59, Ny=96, C=10.0, M=10.0, T=300.0, Ma_E=0.01,
                         alpha=1e-4, delta_U=1.0, gamma=0.3,
                         perturb_amplitude=1e-4, perturb_mode=1, FT=Float64)
    p = ehd_ec_lattice_params(Ny, C, M, T, Ma_E, alpha, delta_U, gamma; FT=FT)
    analytic = hydrostatic_profiles(C, Ny; FT=FT)
    A = FT(Nx - 1) / FT(Ny - 1)
    phi_f = zeros(FT, Nx, Ny, 9)
    q_f = zeros(FT, Nx, Ny, 9)
    ns_f = zeros(FT, Nx, Ny, 9)
    phi = zeros(FT, Nx, Ny)
    qfield = zeros(FT, Nx, Ny)
    Ey0 = zeros(FT, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        xstar = A * FT(i - 1) / FT(Nx - 1)
        ystar = FT(j - 1) / FT(Ny - 1)
        qval = p.q_inj * analytic.q_star[j] +
               FT(perturb_amplitude) * p.q_inj * sin(FT(pi) * ystar) *
               cos(FT(pi) * FT(perturb_mode) * xstar / A)
        j == 1 && (qval = p.q_inj)
        phi[i, j] = analytic.phi[j]
        qfield[i, j] = qval
        Ey0[i, j] = analytic.E_star[j] / p.H
        for qdir in 1:9
            phi_f[i, j, qdir] = EHD_W[qdir] * phi[i, j]
            q_f[i, j, qdir] = charge_feq_pop(qval, zero(FT), p.K * Ey0[i, j], qdir)
            ns_f[i, j, qdir] = EHD_W[qdir]
        end
    end
    wall = zeros(Bool, Nx, Ny)
    wall[:, 1] .= true
    wall[:, Ny] .= true
    Ak = Diagonal(FT[1, 1.64, 1.54, 1, 1, 1, 1, inv(p.tau), inv(p.tau)])
    Tmat = FT.(EHD_TMRT)
    Mmat = Tmat \ (Ak * Tmat)
    Nmat = Tmat \ ((I - Ak / 2) * Tmat)
    return EHDTwinState(Nx, Ny, p, wall, build_bb_links(wall),
                        phi_f, similar(phi_f), phi, zeros(FT, Nx, Ny), Ey0,
                        q_f, similar(q_f), qfield, ones(FT, Nx, Ny),
                        zeros(FT, Nx, Ny), zeros(FT, Nx, Ny),
                        ns_f, similar(ns_f), zeros(FT, Nx, Ny, 2),
                        zeros(FT, Nx, Ny, 2), Mmat, Nmat)
end

sum_slots(a) = @views a[:, :, 1] .+ a[:, :, 2]

force_density_from_slots(st::EHDTwinState) =
    (st.rho .* sum_slots(st.Fx_slots), st.rho .* sum_slots(st.Fy_slots))

function twin_step!(st::EHDTwinState; phi_tol=1e-4, phi_max_iter=10000,
                    force_projection=:xy, record=false)
    checkpoints = Dict{String,Any}()
    Up = similar(st.phi)
    Un = similar(st.phi)
    phi_rel = Inf
    phi_iter = 0
    while true
        phi_iter += 1
        copyto!(Up, st.phi)
        collide_phi!(st.phi_f, st.q, st.p.eps, st.p.omega_U, st.p.nu_U)
        matlab_stream!(st.phi_tmp, st.phi_f)
        st.phi_f, st.phi_tmp = st.phi_tmp, st.phi_f
        scalar!(st.phi, st.phi_f)
        apply_phi_bc!(st.phi_f, st.phi)
        scalar!(Un, st.phi_f)
        phi_rel = maximum(abs.(Un .- Up)) / max(maximum(abs.(Un)), floatmin(eltype(Un)))
        copyto!(st.phi, Un)
        phi_rel <= phi_tol && break
        phi_iter >= phi_max_iter && error("Twin phi solve failed; rel=$(phi_rel)")
    end
    calculate_E!(st.Ex, st.Ey, st.phi_f, st.p.tau_U)
    record && (checkpoints["phi_after_solve"] = copy(st.phi);
               checkpoints["Ex"] = copy(st.Ex);
               checkpoints["Ey"] = copy(st.Ey))

    Fx_prev = sum_slots(st.Fx_slots)
    Fy_prev = sum_slots(st.Fy_slots)
    macros_from_force!(st.rho, st.ux, st.uy, st.ns_f, Fx_prev, Fy_prev, st.wall)
    record && (checkpoints["macros_pre_charge.rho"] = copy(st.rho);
               checkpoints["macros_pre_charge.ux"] = copy(st.ux);
               checkpoints["macros_pre_charge.uy"] = copy(st.uy))

    collide_charge_regularized!(st.q_f, st.ux, st.uy, st.Ex, st.Ey, st.p.tau_q, st.p.K)
    record && (checkpoints["q_after_collide.f"] = copy(st.q_f))
    matlab_stream!(st.q_tmp, st.q_f)
    st.q_f, st.q_tmp = st.q_tmp, st.q_f
    scalar!(st.q, st.q_f)
    apply_charge_bc!(st.q_f, st.q, st.ux, st.uy, st.Ex, st.Ey, st.p.q_inj, st.p.K)
    scalar!(st.q, st.q_f)
    record && (checkpoints["q_after_stream_bc.f"] = copy(st.q_f);
               checkpoints["q_after_stream_bc.q"] = copy(st.q))

    apply_ef!(st.Fx_slots, st.Fy_slots, st.Ex, st.Ey, st.q, st.rho, st.wall;
              projection_mode=force_projection, force_ramp=1.0)
    Fx_dens, Fy_dens = force_density_from_slots(st)
    record && (checkpoints["Fx"] = copy(Fx_dens); checkpoints["Fy"] = copy(Fy_dens))
    collide_sp_mrt!(st.ns_f, st.rho, st.ux, st.uy, st.Fx_slots, st.Fy_slots, st.wall,
                    st.mrt_M, st.mrt_N)
    record && (checkpoints["f_after_ns_collide.f"] = copy(st.ns_f))
    matlab_stream!(st.ns_tmp, st.ns_f; bb_links=st.bb_links)
    st.ns_f, st.ns_tmp = st.ns_tmp, st.ns_f
    apply_free_slip_sidewalls!(st.ns_f)
    record && (checkpoints["f_after_ns_stream_bc.f"] = copy(st.ns_f))
    macros_from_force!(st.rho, st.ux, st.uy, st.ns_f, sum_slots(st.Fx_slots),
                       sum_slots(st.Fy_slots), st.wall)
    umax = maximum(sqrt.(st.ux .* st.ux .+ st.uy .* st.uy))
    record && (checkpoints["macros_post.rho"] = copy(st.rho);
               checkpoints["macros_post.ux"] = copy(st.ux);
               checkpoints["macros_post.uy"] = copy(st.uy);
               checkpoints["macros_post.umax"] = [umax])
    return (checkpoints=checkpoints, phi_iter=phi_iter, phi_rel=phi_rel, umax=umax)
end

function twin_run(; Nx=59, Ny=96, C=10.0, M=10.0, T=300.0, Ma_E=0.01,
                  alpha=1e-4, delta_U=1.0, gamma=0.3, perturb_amplitude=1e-4,
                  perturb_mode=1, steps=10, record_steps=10, phi_tol=1e-4,
                  phi_max_iter=10000, force_projection=:xy, FT=Float64,
                  sample_interval=0)
    st = init_twin_state(; Nx=Nx, Ny=Ny, C=C, M=M, T=T, Ma_E=Ma_E,
                         alpha=alpha, delta_U=delta_U, gamma=gamma,
                         perturb_amplitude=perturb_amplitude,
                         perturb_mode=perturb_mode, FT=FT)
    records = Vector{Dict{String,Any}}()
    samples = Tuple{Int,FT}[]
    for step in 1:steps
        out = twin_step!(st; phi_tol=phi_tol, phi_max_iter=phi_max_iter,
                         force_projection=force_projection, record=step <= record_steps)
        step <= record_steps && push!(records, out.checkpoints)
        if sample_interval > 0 && step % sample_interval == 0
            push!(samples, (step, out.umax / st.p.u_ref))
        end
    end
    return (state=st, records=records, samples=samples, params=st.p)
end
