# --- 3D FVFD log-conformation planar-extensional driver ---
#
# Mirrors `run_viscoelastic_fvfd_poiseuille_3d` for the ψ-space
# log-conformation pipeline. The solvent step uses lateral open-face
# LI-BB + Zou-He rebuilds, with no constant body force.

export run_viscoelastic_fvfd_extensional_3d

"""
    run_viscoelastic_fvfd_extensional_3d(; Nx, Ny, Nz, epsilon_dot,
        ν_s, ν_p, lambda, velocity_mode=:coupled, ...)

3D Oldroyd-B planar-extension FVFD canary. The polymer state is advanced as
`ψ=log(C)` with open x/y and periodic z FVFD boundary semantics.

`velocity_mode=:coupled` runs the open lateral-face solvent step
self-consistently.
`velocity_mode=:imposed` keeps that coupled solvent/force step live, then
overwrites `ux,uy,uz` with `u=(epsilon_dot*x,-epsilon_dot*y,0)` for the next
FVFD ψ update. The latter is the documented YELLOW fallback for the analytical
fixed-point canary.
"""
function run_viscoelastic_fvfd_extensional_3d(;
        Nx::Int=24, Ny::Int=24, Nz::Int=6,
        epsilon_dot::Real=0.005,
        ν_s::Real=0.05, ν_p::Union{Nothing,Real}=0.05,
        lambda::Real=50.0,
        polymer_model::Union{Nothing,AbstractPolymerModel}=nothing,
        max_steps::Int=1_500,
        ρ_out::Real=1.0,
        backend=KernelAbstractions.CPU(),
        FT::Type{<:AbstractFloat}=Float64,
        advection_scheme::Symbol=:muscl_superbee,
        max_polymer_substeps::Int=64,
        velocity_mode::Symbol=:coupled)

    mode = velocity_mode === :imposed_velocity ? :imposed : velocity_mode
    mode in (:coupled, :imposed) ||
        throw(ArgumentError("velocity_mode must be :coupled or :imposed"))

    if polymer_model === nothing
        isnothing(ν_p) && error("supply either `polymer_model` or (`ν_p`, `lambda`).")
        polymer_model = LogConfOldroydB(G=FT(ν_p / lambda), λ=FT(lambda))
    end
    lambda_p = polymer_relaxation_time(polymer_model)
    nu_p_eff = polymer_modulus(polymer_model) * lambda_p
    # Finite for LogConfFENEP (= L²), Inf for Oldroyd-B → OB constitutive path.
    L2_fene = polymer_max_extensibility(polymer_model)
    wi_ext = Float64(lambda_p) * Float64(epsilon_dot)
    2 * wi_ext < 1 ||
        throw(ArgumentError("planar-extension fixed point requires 2*lambda*epsilon_dot < 1"))

    nu_total = Float64(ν_s) + Float64(nu_p_eff)
    beta = Float64(ν_s) / nu_total
    dx = one(FT)
    dy = one(FT)
    dz = one(FT)
    x_center = FT((Nx + 1) / 2)
    y_center = FT((Ny + 1) / 2)

    velocity_h = fvfd_planar_extensional_velocity_field_host_3d(
        Nx, Ny, Nz, epsilon_dot; dx, dy, x_center, y_center, FT,
    )
    u_max = maximum(
        sqrt(velocity_h.ux[i, j, k]^2 + velocity_h.uy[i, j, k]^2)
        for k in 1:Nz for j in 1:Ny for i in 1:Nx
    )
    Re = Float64(u_max) * Float64(Ny) / max(Float64(ν_s), eps(Float64))

    @info "FVFD log-conf planar extension (3D)" Nx Ny Nz epsilon_dot wi_ext beta lambda_p Re mode polymer_model=typeof(polymer_model)

    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny, Nz)
    fill!(is_solid, false)
    q_wall = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
    uw_x = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
    uw_y = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
    uw_z = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)

    f_in = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz, 19)
    f_out = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz, 19)
    ρ = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz)
    ux = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz)
    uy = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz)
    uz = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz)
    fill!(ρ, one(FT))
    copyto!(ux, velocity_h.ux)
    copyto!(uy, velocity_h.uy)
    copyto!(uz, velocity_h.uz)

    f_in_h = zeros(FT, Nx, Ny, Nz, 19)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        u0 = velocity_h.ux[i, j, k]
        v0 = velocity_h.uy[i, j, k]
        for q in 1:19
            f_in_h[i, j, k, q] = Kraken.equilibrium(D3Q19(), one(FT), u0, v0, zero(FT), q)
        end
    end
    copyto!(f_in, f_in_h)
    fill!(f_out, zero(FT))

    psixx = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    psixy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    psixz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    psiyy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    psiyz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    psizz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)

    psixx_adv = similar(psixx); psixy_adv = similar(psixy); psixz_adv = similar(psixz)
    psiyy_adv = similar(psiyy); psiyz_adv = similar(psiyz); psizz_adv = similar(psizz)
    psixx_next = similar(psixx); psixy_next = similar(psixy); psixz_next = similar(psixz)
    psiyy_next = similar(psiyy); psiyz_next = similar(psiyz); psizz_next = similar(psizz)

    C_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz); fill!(C_xx, one(FT))
    C_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    C_xz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    C_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz); fill!(C_yy, one(FT))
    C_yz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    C_zz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz); fill!(C_zz, one(FT))

    tau_p_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    tau_p_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    tau_p_xz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    tau_p_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    tau_p_yz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    tau_p_zz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)

    Fx_poly = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    Fy_poly = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    Fz_poly = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    Fx_tot = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    Fy_tot = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    Fz_tot = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)

    ux_face = KernelAbstractions.zeros(backend, FT, Nx + 1, Ny, Nz)
    uy_face = KernelAbstractions.zeros(backend, FT, Nx, Ny + 1, Nz)
    uz_face = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz + 1)

    duxdx = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    duxdy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    duxdz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    duydx = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    duydy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    duydz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    duzdx = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    duzdy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    duzdz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)

    ux_west = KernelAbstractions.zeros(backend, FT, Ny, Nz)
    ux_east = KernelAbstractions.zeros(backend, FT, Ny, Nz)
    uy_south = KernelAbstractions.zeros(backend, FT, Nx, Nz)
    uy_north = KernelAbstractions.zeros(backend, FT, Nx, Nz)
    uz_back = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    uz_front = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    u_profile_west = KernelAbstractions.zeros(backend, FT, Ny, Nz)

    psi_west = KernelAbstractions.zeros(backend, FT, Ny, Nz)
    psi_east = KernelAbstractions.zeros(backend, FT, Ny, Nz)
    psi_south = KernelAbstractions.zeros(backend, FT, Nx, Nz)
    psi_north = KernelAbstractions.zeros(backend, FT, Nx, Nz)
    psi_back = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    psi_front = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    psi_bc = (psi_west, psi_east, psi_south, psi_north, psi_back, psi_front)

    fvfd_fill_planar_extensional_openxy_bcs_3d!(
        ux_west, ux_east, u_profile_west,
        uy_south, uy_north,
        psi_west, psi_east, psi_south, psi_north,
        Nx, Ny, epsilon_dot; dx, dy, x_center, y_center, sync=true,
    )
    bcspec = BCSpec3D(; west=ZouHeVelocity(u_profile_west))

    compute_macroscopic_forced_field_3d!(ρ, ux, uy, uz, f_in, Fx_tot, Fy_tot, Fz_tot)
    if mode === :imposed
        fvfd_impose_planar_extensional_velocity_3d!(
            ux, uy, uz, epsilon_dot; dx, dy, x_center, y_center, sync=false,
        )
    end

    max_substeps_observed = 1
    last_n_sub = 1
    last_grad_norm = 0.0
    completed_steps = 0

    for step in 1:max_steps
        completed_steps = step

        fvfd_cell_velocity_to_faces_3d!(
            ux_face, uy_face, uz_face, ux, uy, uz, is_solid,
            ux_west, ux_east, uy_south, uy_north, uz_back, uz_front,
            :open, :open, :open, :open, :periodic, :periodic;
            sync=false,
        )
        fvfd_sym3_advect_upwind_3d!(
            psixx_adv, psixy_adv, psixz_adv, psiyy_adv, psiyz_adv, psizz_adv,
            psixx, psixy, psixz, psiyy, psiyz, psizz,
            psi_bc, psi_bc, psi_bc, psi_bc, psi_bc, psi_bc,
            ux_face, uy_face, uz_face, is_solid,
            dx, dy, dz,
            :open, :open, :open, :open, :periodic, :periodic,
            one(FT);
            sync=false,
            advection_scheme,
        )
        fvfd_velocity_gradient_3d!(
            duxdx, duxdy, duxdz,
            duydx, duydy, duydz,
            duzdx, duzdy, duzdz,
            ux, uy, uz, is_solid;
            dx, dy, dz, x_bc=:open, y_bc=:open, z_bc=:periodic,
            sync=false,
        )

        KernelAbstractions.synchronize(backend)
        last_grad_norm = logfv_max_grad_norm_3d(
            duxdx, duxdy, duxdz,
            duydx, duydy, duydz,
            duzdx, duzdy, duzdz,
        )
        last_n_sub = logfv_recommended_oldroydb_substeps_3d(
            last_grad_norm, Float64(lambda_p), 1.0; max_substeps=max_polymer_substeps,
        )
        max_substeps_observed = max(max_substeps_observed, last_n_sub)
        # Shared per-model constitutive-step dispatch (DRY): OB / FENE-P /
        # Giesekus / PTT all route through one helper. OB → dedicated OB
        # kernel; FENE-P → FENE-P kernel with L²=polymer_max_extensibility
        # (bit-identical to the prior isfinite(L2) branch); Giesekus(α=0) /
        # PTT(ε=0) reproduce the OB trajectory bit-for-bit.
        logfv_constitutive_step_dispatch_3d!(
            polymer_model,
            psixx_next, psixy_next, psixz_next, psiyy_next, psiyz_next, psizz_next,
            psixx_adv, psixy_adv, psixz_adv, psiyy_adv, psiyz_adv, psizz_adv,
            duxdx, duxdy, duxdz,
            duydx, duydy, duydz,
            duzdx, duzdy, duzdz,
            FT(lambda_p), one(FT), last_n_sub;
            sync=true,
        )

        psi_to_C_3d!(
            C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
            psixx_next, psixy_next, psixz_next, psiyy_next, psiyz_next, psizz_next,
        )
        update_polymer_stress_3d!(
            tau_p_xx, tau_p_xy, tau_p_xz, tau_p_yy, tau_p_yz, tau_p_zz,
            C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
            polymer_model,
        )
        compute_polymeric_force_3d!(
            Fx_poly, Fy_poly, Fz_poly,
            tau_p_xx, tau_p_xy, tau_p_xz, tau_p_yy, tau_p_yz, tau_p_zz;
            periodic_x=false, periodic_z=true,
        )
        Fx_tot .= Fx_poly
        Fy_tot .= Fy_poly
        Fz_tot .= Fz_poly

        fused_trt_libb_v2_guo_field_step_3d!(
            f_out, f_in, ρ, ux, uy, uz, is_solid,
            q_wall, uw_x, uw_y, uw_z,
            Fx_tot, Fy_tot, Fz_tot,
            Nx, Ny, Nz, FT(ν_s);
            periodic_z=true,
        )
        apply_bc_rebuild_3d!(f_out, f_in, bcspec, FT(ν_s), Nx, Ny, Nz)
        fvfd_apply_extensional_straining_bc_3d!(
            f_out, f_in, ux_east, uy_south, uy_north, FT(ν_s), Nx, Ny, Nz,
        )
        compute_macroscopic_forced_field_3d!(ρ, ux, uy, uz, f_out, Fx_tot, Fy_tot, Fz_tot)
        if mode === :imposed
            fvfd_impose_planar_extensional_velocity_3d!(
                ux, uy, uz, epsilon_dot; dx, dy, x_center, y_center, sync=false,
            )
        end

        f_in, f_out = f_out, f_in
        psixx, psixx_next = psixx_next, psixx
        psixy, psixy_next = psixy_next, psixy
        psixz, psixz_next = psixz_next, psixz
        psiyy, psiyy_next = psiyy_next, psiyy
        psiyz, psiyz_next = psiyz_next, psiyz
        psizz, psizz_next = psizz_next, psizz
    end
    KernelAbstractions.synchronize(backend)

    compute_polymeric_force_3d!(
        Fx_poly, Fy_poly, Fz_poly,
        tau_p_xx, tau_p_xy, tau_p_xz, tau_p_yy, tau_p_yz, tau_p_zz;
        periodic_x=false, periodic_z=true,
    )
    Fx_tot .= Fx_poly
    Fy_tot .= Fy_poly
    Fz_tot .= Fz_poly
    compute_macroscopic_forced_field_3d!(ρ, ux, uy, uz, f_in, Fx_tot, Fy_tot, Fz_tot)
    if mode === :imposed
        fvfd_impose_planar_extensional_velocity_3d!(
            ux, uy, uz, epsilon_dot; dx, dy, x_center, y_center, sync=true,
        )
    else
        KernelAbstractions.synchronize(backend)
    end

    ux_h = Array(ux); uy_h = Array(uy); uz_h = Array(uz)
    Cxx_h = Array(C_xx); Cxy_h = Array(C_xy); Cxz_h = Array(C_xz)
    Cyy_h = Array(C_yy); Cyz_h = Array(C_yz); Czz_h = Array(C_zz)
    psixx_h = Array(psixx); psixy_h = Array(psixy); psixz_h = Array(psixz)
    psiyy_h = Array(psiyy); psiyz_h = Array(psiyz); psizz_h = Array(psizz)
    txx_h = Array(tau_p_xx); txy_h = Array(tau_p_xy); txz_h = Array(tau_p_xz)
    tyy_h = Array(tau_p_yy); tyz_h = Array(tau_p_yz); tzz_h = Array(tau_p_zz)
    duxdx_h = Array(duxdx); duydy_h = Array(duydy)

    i1 = max(1, Nx ÷ 2)
    i2 = min(Nx, Nx ÷ 2 + 1)
    j1 = max(1, Ny ÷ 2)
    j2 = min(Ny, Ny ÷ 2 + 1)
    center_mean(A) = sum(@view A[i1:i2, j1:j2, :]) / ((i2 - i1 + 1) * (j2 - j1 + 1) * Nz)

    return (;
        ux=ux_h, uy=uy_h, uz=uz_h, ρ=Array(ρ),
        psi_xx=psixx_h, psi_xy=psixy_h, psi_xz=psixz_h,
        psi_yy=psiyy_h, psi_yz=psiyz_h, psi_zz=psizz_h,
        C_xx=Cxx_h, C_xy=Cxy_h, C_xz=Cxz_h,
        C_yy=Cyy_h, C_yz=Cyz_h, C_zz=Czz_h,
        tau_p_xx=txx_h, tau_p_xy=txy_h, tau_p_xz=txz_h,
        tau_p_yy=tyy_h, tau_p_yz=tyz_h, tau_p_zz=tzz_h,
        duxdx=duxdx_h, duydy=duydy_h,
        center_Cxx=center_mean(Cxx_h), center_Cyy=center_mean(Cyy_h),
        center_Czz=center_mean(Czz_h), center_Cxy=center_mean(Cxy_h),
        center_duxdx=center_mean(duxdx_h), center_duydy=center_mean(duydy_h),
        epsilon_dot=Float64(epsilon_dot), lambda=Float64(lambda_p), Wi_ext=wi_ext,
        beta=beta, Re=Re, eta_s=Float64(ν_s), eta_p=Float64(nu_p_eff),
        eta_total=nu_total, L2_fene=L2_fene,
        velocity_mode=mode, open_x_gradient_supported=true,
        open_y_gradient_supported=true, bc_config=:openxy_zh_velocity,
        completed_steps=completed_steps, last_n_sub=last_n_sub,
        max_substeps_observed=max_substeps_observed,
        last_grad_norm=last_grad_norm,
    )
end
