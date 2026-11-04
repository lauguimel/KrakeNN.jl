# --- 3D FVFD log-conformation viscoelastic planar-Poiseuille driver ---
#
# Mirrors `run_conformation_poiseuille_libb_3d` for the D3Q19 solvent and Guo
# coupling, but transports the polymer state as ψ = log(C) with the FVFD M1/M2/M3a
# operators instead of the diffusive conformation LBM-CDE.

export run_viscoelastic_fvfd_poiseuille_3d

"""
    run_viscoelastic_fvfd_poiseuille_3d(; Nx, Ny, Nz, Fx, ν_s, ν_p, lambda,
                                         polymer_model=nothing,
                                         max_steps, backend, FT)

3D Oldroyd-B planar Poiseuille with periodic x/z, no-slip half-way bounce-back
y walls, D3Q19 solvent at ν_s, and polymer feedback only through Guo
F_poly = ∇·τ_p. The conformation state is advanced in symmetric log form ψ.
"""
function run_viscoelastic_fvfd_poiseuille_3d(;
        Nx::Int=6, Ny::Int=32, Nz::Int=6,
        Fx::Real=1e-5, ν_s::Real=0.05, ν_p::Union{Nothing,Real}=0.05,
        lambda::Real=10.0,
        polymer_model::Union{Nothing,AbstractPolymerModel}=nothing,
        max_steps::Int=20_000,
        backend=KernelAbstractions.CPU(),
        FT::Type{<:AbstractFloat}=Float64,
        advection_scheme::Symbol=:muscl_superbee,
        max_polymer_substeps::Int=64)

    # Step order:
    # 1. cell u -> faces: fvfd_cell_velocity_to_faces_3d!
    # 2. advect ψ (6 comps): fvfd_sym3_advect_upwind_3d!
    # 3. ∇u: fvfd_velocity_gradient_3d!
    # 4. constitutive subcycles: logfv_constitutive_step_log_3d!
    # 5. ψ -> C: psi_to_C_3d!
    # 6. τ_p from C: update_polymer_stress_3d!
    # 7. F_poly = ∇·τ_p: compute_polymeric_force_3d!
    # 8. add constant Poiseuille body force
    # 9. D3Q19 solvent: collide_guo_field_3d! + stream_periodic_xz_wall_y_3d!
    # 10. macro: compute_macroscopic_forced_field_3d!
    # 11. swap f and ψ buffers
    if polymer_model === nothing
        isnothing(ν_p) && error("supply either `polymer_model` or (`ν_p`, `lambda`).")
        G_ = FT(ν_p / lambda)
        polymer_model = LogConfOldroydB(G=G_, λ=FT(lambda))
    end
    λ_p = polymer_relaxation_time(polymer_model)
    ν_p_eff = polymer_modulus(polymer_model) * λ_p
    # Finite for LogConfFENEP (= L²), Inf for Oldroyd-B → OB constitutive path.
    L2_fene = polymer_max_extensibility(polymer_model)

    ν_total = Float64(ν_s) + Float64(ν_p_eff)
    beta = Float64(ν_s) / ν_total
    ω_s = 1.0 / (3.0 * Float64(ν_s) + 0.5)
    Fx_d = Float64(Fx)
    dx = one(FT)
    dy = one(FT)
    dz = one(FT)

    u_analytical = [Fx_d / (2 * ν_total) * (j - 0.5) * (Ny + 0.5 - j) for j in 1:Ny]
    u_max = maximum(u_analytical)
    gamma_dot_an = [abs(Fx_d / (2 * ν_total) * (Ny + 1 - 2j)) for j in 1:Ny]
    gamma_dot_wall = maximum(gamma_dot_an)
    Wi_wall = Float64(λ_p) * gamma_dot_wall
    Re = u_max * Float64(Ny) / ν_total

    @info "FVFD log-conf Poiseuille (3D)" Nx Ny Nz Fx u_max gamma_dot_wall Wi_wall beta λ_p Re L2_fene polymer_model=typeof(polymer_model)

    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny, Nz)
    fill!(is_solid, false)

    f_in = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz, 19)
    f_out = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz, 19)
    ρ = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz)
    ux = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz)
    uy = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz)
    uz = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz)
    fill!(ρ, one(FT)); fill!(ux, zero(FT)); fill!(uy, zero(FT)); fill!(uz, zero(FT))

    f_in_h = zeros(FT, Nx, Ny, Nz, 19)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        u0 = FT(u_analytical[j])
        for q in 1:19
            f_in_h[i, j, k, q] = Kraken.equilibrium(D3Q19(), one(FT), u0,
                                                     zero(FT), zero(FT), q)
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

    psi_west = KernelAbstractions.zeros(backend, FT, Ny, Nz)
    psi_east = KernelAbstractions.zeros(backend, FT, Ny, Nz)
    psi_south = KernelAbstractions.zeros(backend, FT, Nx, Nz)
    psi_north = KernelAbstractions.zeros(backend, FT, Nx, Nz)
    psi_back = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    psi_front = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    psi_bc = (psi_west, psi_east, psi_south, psi_north, psi_back, psi_front)

    Fx_tot .= FT(Fx_d)
    fill!(Fy_tot, zero(FT))
    fill!(Fz_tot, zero(FT))
    compute_macroscopic_forced_field_3d!(ρ, ux, uy, uz, f_in, Fx_tot, Fy_tot, Fz_tot)

    max_substeps_observed = 1
    last_n_sub = 1
    last_grad_norm = 0.0
    completed_steps = 0

    for step in 1:max_steps
        completed_steps = step

        fvfd_cell_velocity_to_faces_3d!(
            ux_face, uy_face, uz_face, ux, uy, uz, is_solid,
            ux_west, ux_east, uy_south, uy_north, uz_back, uz_front,
            :periodic, :periodic, :wall, :wall, :periodic, :periodic;
            sync=false,
        )
        fvfd_sym3_advect_upwind_3d!(
            psixx_adv, psixy_adv, psixz_adv, psiyy_adv, psiyz_adv, psizz_adv,
            psixx, psixy, psixz, psiyy, psiyz, psizz,
            psi_bc, psi_bc, psi_bc, psi_bc, psi_bc, psi_bc,
            ux_face, uy_face, uz_face, is_solid,
            dx, dy, dz,
            :periodic, :periodic, :wall, :wall, :periodic, :periodic,
            one(FT);
            sync=false,
            advection_scheme,
        )
        fvfd_velocity_gradient_3d!(
            duxdx, duxdy, duxdz,
            duydx, duydy, duydz,
            duzdx, duzdy, duzdz,
            ux, uy, uz, is_solid;
            dx, dy, dz, x_bc=:periodic, y_bc=:wall, z_bc=:periodic,
            sync=false,
        )

        KernelAbstractions.synchronize(backend)
        last_grad_norm = logfv_max_grad_norm_3d(
            duxdx, duxdy, duxdz,
            duydx, duydy, duydz,
            duzdx, duzdy, duzdz,
        )
        last_n_sub = logfv_recommended_oldroydb_substeps_3d(
            last_grad_norm, Float64(λ_p), 1.0; max_substeps=max_polymer_substeps,
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
            FT(λ_p), one(FT), last_n_sub;
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
            periodic_x=true, periodic_z=true,
        )
        Fx_tot .= Fx_poly .+ FT(Fx_d)
        Fy_tot .= Fy_poly
        Fz_tot .= Fz_poly

        collide_guo_field_3d!(f_in, is_solid, Fx_tot, Fy_tot, Fz_tot, FT(ω_s))
        stream_periodic_xz_wall_y_3d!(f_out, f_in, Nx, Ny, Nz)
        compute_macroscopic_forced_field_3d!(ρ, ux, uy, uz, f_out, Fx_tot, Fy_tot, Fz_tot)

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
        periodic_x=true, periodic_z=true,
    )
    Fx_tot .= Fx_poly .+ FT(Fx_d)
    Fy_tot .= Fy_poly
    Fz_tot .= Fz_poly
    compute_macroscopic_forced_field_3d!(ρ, ux, uy, uz, f_in, Fx_tot, Fy_tot, Fz_tot)

    ux_h = Array(ux)
    Cxx_h = Array(C_xx); Cxy_h = Array(C_xy); Cxz_h = Array(C_xz)
    Cyy_h = Array(C_yy); Cyz_h = Array(C_yz); Czz_h = Array(C_zz)
    psixx_h = Array(psixx); psixy_h = Array(psixy); psixz_h = Array(psixz)
    psiyy_h = Array(psiyy); psiyz_h = Array(psiyz); psizz_h = Array(psizz)
    txx_h = Array(tau_p_xx); txy_h = Array(tau_p_xy); txz_h = Array(tau_p_xz)
    tyy_h = Array(tau_p_yy); tyz_h = Array(tau_p_yz); tzz_h = Array(tau_p_zz)

    planeavg(A, j) = sum(@view A[:, j, :]) / (Nx * Nz)
    profile = [planeavg(ux_h, j) for j in 1:Ny]
    Cxy_prof = [planeavg(Cxy_h, j) for j in 1:Ny]
    Cxx_prof = [planeavg(Cxx_h, j) for j in 1:Ny]
    Cyy_prof = [planeavg(Cyy_h, j) for j in 1:Ny]
    Czz_prof = [planeavg(Czz_h, j) for j in 1:Ny]
    Cxz_prof = [planeavg(Cxz_h, j) for j in 1:Ny]
    Cyz_prof = [planeavg(Cyz_h, j) for j in 1:Ny]
    txx_prof = [planeavg(txx_h, j) for j in 1:Ny]
    txy_prof = [planeavg(txy_h, j) for j in 1:Ny]
    tyy_prof = [planeavg(tyy_h, j) for j in 1:Ny]
    tzz_prof = [planeavg(tzz_h, j) for j in 1:Ny]

    eta_s = Float64(ν_s)
    eta_p = Float64(ν_p_eff)
    eta_total = ν_total

    gamma_dot_meas_prof = similar(profile)
    for j in 1:Ny
        if j == 1
            gamma_dot_meas_prof[j] = abs(profile[2] - profile[1])
        elseif j == Ny
            gamma_dot_meas_prof[j] = abs(profile[Ny] - profile[Ny - 1])
        else
            gamma_dot_meas_prof[j] = abs((profile[j + 1] - profile[j - 1]) / 2)
        end
    end
    tau_xy_prof = [eta_s * gamma_dot_meas_prof[j] + txy_prof[j] for j in 1:Ny]
    N1_prof = [txx_prof[j] - tyy_prof[j] for j in 1:Ny]
    N2_prof = [tyy_prof[j] - tzz_prof[j] for j in 1:Ny]

    return (ux=ux_h, ρ=Array(ρ),
            psi_xx=psixx_h, psi_xy=psixy_h, psi_xz=psixz_h,
            psi_yy=psiyy_h, psi_yz=psiyz_h, psi_zz=psizz_h,
            C_xx=Cxx_h, C_xy=Cxy_h, C_xz=Cxz_h,
            C_yy=Cyy_h, C_yz=Cyz_h, C_zz=Czz_h,
            tau_p_xx=txx_h, tau_p_xy=txy_h, tau_p_xz=txz_h,
            tau_p_yy=tyy_h, tau_p_yz=tyz_h, tau_p_zz=tzz_h,
            profile=profile, u_analytical=u_analytical, u_max=u_max,
            gamma_dot_an=gamma_dot_an, gamma_dot_meas_prof=gamma_dot_meas_prof,
            Cxy_prof=Cxy_prof, Cxx_prof=Cxx_prof, Cyy_prof=Cyy_prof,
            Czz_prof=Czz_prof, Cxz_prof=Cxz_prof, Cyz_prof=Cyz_prof,
            tau_xy_prof=tau_xy_prof, N1_prof=N1_prof, N2_prof=N2_prof,
            gamma_dot_wall=gamma_dot_wall, Wi_wall=Wi_wall, lambda=λ_p,
            beta=beta, Re=Re, eta_s=eta_s, eta_p=eta_p, eta_total=eta_total,
            L2_fene=L2_fene,
            completed_steps=completed_steps, last_n_sub=last_n_sub,
            max_substeps_observed=max_substeps_observed,
            last_grad_norm=last_grad_norm)
end
