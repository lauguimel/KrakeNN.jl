# --- 3D viscoelastic planar-Poiseuille driver (D3Q19 + 6-component conformation) ---
#
# Second canonical analytical 3D viscoelastic canary: steady pressure/body-force
# driven channel flow. Flow in x driven by a constant Guo body force; no-slip
# half-way bounce-back walls on the y-faces (j=1, j=Ny); periodic x AND z. Unlike
# Couette (constant γ̇), Poiseuille has a γ̇(y) that varies LINEARLY across the
# channel — zero at the centre-line, maximal at the walls — so this exercises the
# velocity-gradient stencil of the conformation transport across y via the C(y)
# and N1(y) PROFILES (not single bulk scalars).
#
# Reuses the geometry-agnostic 3D constitutive stack and the reusable periodic-xz
# / no-slip-y streamer introduced for the Couette canary:
#   - `compute_polymeric_force_3d!`        polymer body force F_poly = ∇·τ_p
#   - `collide_guo_field_3d!`              BGK solvent collision + fused Guo force
#                                          (constant Fx + ∇·τ_p), force consumed once
#   - `stream_periodic_xz_wall_y_3d!`      periodic-xz / no-slip y-wall streamer
#   - `compute_macroscopic_forced_field_3d!` force-corrected velocity (u = Σf·c + F/2)
#   - `collide_conformation_3d!` ×6        TRT conformation transport (∇u central diff)
#   - `apply_cnebb_conformation_y_walls_3d!`  conformation wall BC (y-faces)
#   - `update_polymer_stress_3d!`          Oldroyd-B τ_p = G·(C − I)
#
# Closed-form steady Oldroyd-B targets (OB has CONSTANT shear viscosity, so the
# velocity is the Newtonian parabola). With half-way bounce-back the walls sit ½
# cell outside the first/last fluid row (y = 0.5 and y = Ny+0.5), fluid row j at
# height (j − 0.5):
#   u_x(j) = Fx/(2 ν_total)·(j − 0.5)·(Ny + 0.5 − j)        (parabola)
#   γ̇(j)  = |du_x/dy| , du_x/dy = Fx/(2 ν_total)·(Ny + 1 − 2j)  (linear in y)
#   C_xy(j) = λ·γ̇(j),  C_xx(j) = 1 + 2·(λγ̇(j))²,  C_yy = C_zz = 1,  C_xz = C_yz = 0
#   N1(j) = 2·η_p·λ·γ̇(j)²  (parabolic-in-γ̇: 0 at centre, max near walls);  N2 = 0.

"""
    run_conformation_poiseuille_libb_3d(; Nx, Ny, Nz, Fx, ν_s, ν_p, lambda,
                                          polymer_model=nothing,
                                          tau_plus=1.0, max_steps, backend, FT)

3D viscoelastic planar Poiseuille (Oldroyd-B). Channel flow in x driven by a
constant body force `Fx` (Guo 2002), no-slip half-way bounce-back walls on the
y-faces (j=1, j=Ny), periodic x and z. The solvent uses BGK + the periodic-xz /
no-slip-y streamer; the polymer uses the 6-component conformation TRT-LBM coupled
through the Hermite stress source. Because the shear rate γ̇(y) varies across y,
the conformation and first-normal-stress are PROFILES C(y), N1(y) (validated at
several y-stations: centre, mid, near-wall).

Returns a NamedTuple with the full fields plus the x,z-averaged y-profiles
(`profile`, `gamma_dot_meas_prof`, `Cxy_prof`, `Cxx_prof`, `Cyy_prof`, `Czz_prof`,
`Cxz_prof`, `Cyz_prof`, `N1_prof`, `N2_prof`, `tau_xy_prof`) used for the
analytical assertions, together with `gamma_dot_wall`, `Wi_wall`, `beta`,
`eta_s`, `eta_p`, `eta_total`, and the analytical parabola `u_analytical`.
"""
function run_conformation_poiseuille_libb_3d(;
        Nx::Int=6, Ny::Int=32, Nz::Int=6,
        Fx::Real=1e-5, ν_s::Real=0.05, ν_p::Union{Nothing,Real}=0.05,
        lambda::Real=10.0,
        polymer_model::Union{Nothing,AbstractPolymerModel}=nothing,
        tau_plus::Real=1.0,
        max_steps::Int=20_000,
        backend=KernelAbstractions.CPU(),
        FT::Type{<:AbstractFloat}=Float64)

    # Resolve polymer model from (ν_p, lambda) if not supplied.
    if polymer_model === nothing
        isnothing(ν_p) && error("supply either `polymer_model` or (`ν_p`, `lambda`).")
        G_ = FT(ν_p / lambda)
        polymer_model = OldroydBSpec(G=G_, λ=FT(lambda))
    end
    if uses_log_conformation(polymer_model)
        error("3D log-conformation is not yet wired into the Poiseuille driver; pass an OldroydBSpec model.")
    end
    λ_p     = polymer_relaxation_time(polymer_model)
    ν_p_eff = polymer_modulus(polymer_model) * λ_p

    ν_total = Float64(ν_s) + ν_p_eff
    beta    = Float64(ν_s) / ν_total
    ω_s     = 1.0 / (3.0 * Float64(ν_s) + 0.5)  # solvent rate; polymer via ∇·τ_p Guo force
    Fx_d    = Float64(Fx)

    # Analytical Newtonian parabola (OB shear viscosity is constant). Half-way BB
    # walls at y=0.5 and y=Ny+0.5; fluid row j at height (j − 0.5).
    u_analytical = [Fx_d / (2 * ν_total) * (j - 0.5) * (Ny + 0.5 - j) for j in 1:Ny]
    u_max = maximum(u_analytical)
    # Analytical local shear rate γ̇(j) = |du/dy|, du/dy = Fx/(2ν)·(Ny+1−2j).
    gamma_dot_an = [abs(Fx_d / (2 * ν_total) * (Ny + 1 - 2j)) for j in 1:Ny]
    gamma_dot_wall = maximum(gamma_dot_an)
    Wi_wall = λ_p * gamma_dot_wall
    Re = u_max * Float64(Ny) / ν_total

    @info "Conformation Poiseuille (3D)" Nx Ny Nz Fx u_max gamma_dot_wall Wi_wall beta λ_p Re tau_plus polymer_model=typeof(polymer_model)

    # No obstacle: every cell is fluid.
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny, Nz)
    fill!(is_solid, false)

    # --- Solvent allocations (init to rest; the body force ramps up the flow) ---
    f_in  = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz, 19)
    f_out = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz, 19)
    ρ  = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz)
    ux = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz)
    uy = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz)
    uz = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz)
    fill!(ρ, one(FT)); fill!(ux, zero(FT)); fill!(uy, zero(FT)); fill!(uz, zero(FT))

    # Warm start f at equilibrium on the analytical parabola → faster convergence.
    f_in_h = zeros(FT, Nx, Ny, Nz, 19)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        u0 = FT(u_analytical[j])
        for q in 1:19
            f_in_h[i, j, k, q] = Kraken.equilibrium(D3Q19(), one(FT), u0,
                                                     zero(FT), zero(FT), q)
        end
    end
    copyto!(f_in, f_in_h); fill!(f_out, zero(FT))

    # --- Conformation fields (6 components) ---
    C_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz); fill!(C_xx, FT(1))
    C_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    C_xz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    C_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz); fill!(C_yy, FT(1))
    C_yz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    C_zz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz); fill!(C_zz, FT(1))

    g_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
    g_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
    g_xz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
    g_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
    g_yz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
    g_zz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
    init_conformation_field_3d!(g_xx, C_xx, ux, uy, uz)
    init_conformation_field_3d!(g_xy, C_xy, ux, uy, uz)
    init_conformation_field_3d!(g_xz, C_xz, ux, uy, uz)
    init_conformation_field_3d!(g_yy, C_yy, ux, uy, uz)
    init_conformation_field_3d!(g_yz, C_yz, ux, uy, uz)
    init_conformation_field_3d!(g_zz, C_zz, ux, uy, uz)

    g_xx_buf = similar(g_xx); g_xy_buf = similar(g_xy); g_xz_buf = similar(g_xz)
    g_yy_buf = similar(g_yy); g_yz_buf = similar(g_yz); g_zz_buf = similar(g_zz)

    tau_p_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    tau_p_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    tau_p_xz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    tau_p_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    tau_p_yz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    tau_p_zz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)

    # --- Polymer body force F_poly = ∇·τ_p + the constant driving force ---
    # VALIDATED 2D Guo coupling ported to 3D: the polymer enters the momentum
    # equation EXACTLY ONCE as a first-moment Guo body force at the solvent rate
    # ω_s (lattice viscosity = ν_s), NOT via a standalone re-relaxed Hermite
    # source. Total force field = ∇·τ_p (per-cell) + Fx_d (constant in x).
    Fx_poly = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    Fy_poly = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    Fz_poly = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    Fx_tot  = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    Fy_tot  = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    Fz_tot  = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)

    for step in 1:max_steps
        # --- Polymer body force: F_poly = ∇·τ_p (periodic x AND z) ---
        compute_polymeric_force_3d!(Fx_poly, Fy_poly, Fz_poly,
                                      tau_p_xx, tau_p_xy, tau_p_xz,
                                      tau_p_yy, tau_p_yz, tau_p_zz;
                                      periodic_x=true, periodic_z=true)
        # Total Guo force = polymer divergence + constant x-driving force.
        Fx_tot .= Fx_poly .+ FT(Fx_d)
        Fy_tot .= Fy_poly
        Fz_tot .= Fz_poly

        # --- Solvent: BGK+Guo-field collide (force fused once) → y-wall stream ---
        collide_guo_field_3d!(f_in, is_solid, Fx_tot, Fy_tot, Fz_tot, FT(ω_s))
        # Periodic x/z, no-slip half-way bounce-back walls on the y-faces.
        stream_periodic_xz_wall_y_3d!(f_out, f_in, Nx, Ny, Nz)

        compute_macroscopic_forced_field_3d!(ρ, ux, uy, uz, f_out, Fx_tot, Fy_tot, Fz_tot)

        # --- Conformation TRT (6 components), periodic-xz + y-wall CNEBB ---
        stream_periodic_xz_wall_y_3d!(g_xx_buf, g_xx, Nx, Ny, Nz)
        stream_periodic_xz_wall_y_3d!(g_xy_buf, g_xy, Nx, Ny, Nz)
        stream_periodic_xz_wall_y_3d!(g_xz_buf, g_xz, Nx, Ny, Nz)
        stream_periodic_xz_wall_y_3d!(g_yy_buf, g_yy, Nx, Ny, Nz)
        stream_periodic_xz_wall_y_3d!(g_yz_buf, g_yz, Nx, Ny, Nz)
        stream_periodic_xz_wall_y_3d!(g_zz_buf, g_zz, Nx, Ny, Nz)

        apply_cnebb_conformation_y_walls_3d!(g_xx_buf, g_xx, C_xx)
        apply_cnebb_conformation_y_walls_3d!(g_xy_buf, g_xy, C_xy)
        apply_cnebb_conformation_y_walls_3d!(g_xz_buf, g_xz, C_xz)
        apply_cnebb_conformation_y_walls_3d!(g_yy_buf, g_yy, C_yy)
        apply_cnebb_conformation_y_walls_3d!(g_yz_buf, g_yz, C_yz)
        apply_cnebb_conformation_y_walls_3d!(g_zz_buf, g_zz, C_zz)

        g_xx, g_xx_buf = g_xx_buf, g_xx
        g_xy, g_xy_buf = g_xy_buf, g_xy
        g_xz, g_xz_buf = g_xz_buf, g_xz
        g_yy, g_yy_buf = g_yy_buf, g_yy
        g_yz, g_yz_buf = g_yz_buf, g_yz
        g_zz, g_zz_buf = g_zz_buf, g_zz

        compute_conformation_macro_3d!(C_xx, g_xx)
        compute_conformation_macro_3d!(C_xy, g_xy)
        compute_conformation_macro_3d!(C_xz, g_xz)
        compute_conformation_macro_3d!(C_yy, g_yy)
        compute_conformation_macro_3d!(C_yz, g_yz)
        compute_conformation_macro_3d!(C_zz, g_zz)

        collide_conformation_3d!(g_xx, C_xx, ux, uy, uz,
                                   C_xx, C_xy, C_xz, C_yy, C_yz, C_zz, is_solid,
                                   tau_plus, λ_p; component=1)
        collide_conformation_3d!(g_xy, C_xy, ux, uy, uz,
                                   C_xx, C_xy, C_xz, C_yy, C_yz, C_zz, is_solid,
                                   tau_plus, λ_p; component=2)
        collide_conformation_3d!(g_xz, C_xz, ux, uy, uz,
                                   C_xx, C_xy, C_xz, C_yy, C_yz, C_zz, is_solid,
                                   tau_plus, λ_p; component=3)
        collide_conformation_3d!(g_yy, C_yy, ux, uy, uz,
                                   C_xx, C_xy, C_xz, C_yy, C_yz, C_zz, is_solid,
                                   tau_plus, λ_p; component=4)
        collide_conformation_3d!(g_yz, C_yz, ux, uy, uz,
                                   C_xx, C_xy, C_xz, C_yy, C_yz, C_zz, is_solid,
                                   tau_plus, λ_p; component=5)
        collide_conformation_3d!(g_zz, C_zz, ux, uy, uz,
                                   C_xx, C_xy, C_xz, C_yy, C_yz, C_zz, is_solid,
                                   tau_plus, λ_p; component=6)

        update_polymer_stress_3d!(tau_p_xx, tau_p_xy, tau_p_xz,
                                    tau_p_yy, tau_p_yz, tau_p_zz,
                                    C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
                                    polymer_model)

        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    # Final readout: recompute the total Guo force from the converged τ_p so the
    # +F/2 velocity correction matches the last collision.
    compute_polymeric_force_3d!(Fx_poly, Fy_poly, Fz_poly,
                                  tau_p_xx, tau_p_xy, tau_p_xz,
                                  tau_p_yy, tau_p_yz, tau_p_zz;
                                  periodic_x=true, periodic_z=true)
    Fx_tot .= Fx_poly .+ FT(Fx_d)
    Fy_tot .= Fy_poly
    Fz_tot .= Fz_poly
    compute_macroscopic_forced_field_3d!(ρ, ux, uy, uz, f_in, Fx_tot, Fy_tot, Fz_tot)

    ux_h  = Array(ux)
    Cxx_h = Array(C_xx); Cxy_h = Array(C_xy); Cxz_h = Array(C_xz)
    Cyy_h = Array(C_yy); Cyz_h = Array(C_yz); Czz_h = Array(C_zz)
    txx_h = Array(tau_p_xx); txy_h = Array(tau_p_xy)
    tyy_h = Array(tau_p_yy); tzz_h = Array(tau_p_zz)

    # x,z-averaged y-profiles (the channel is homogeneous in x and z).
    planeavg(A, j) = sum(@view A[:, j, :]) / (Nx * Nz)
    profile        = [planeavg(ux_h, j)  for j in 1:Ny]
    Cxy_prof       = [planeavg(Cxy_h, j) for j in 1:Ny]
    Cxx_prof       = [planeavg(Cxx_h, j) for j in 1:Ny]
    Cyy_prof       = [planeavg(Cyy_h, j) for j in 1:Ny]
    Czz_prof       = [planeavg(Czz_h, j) for j in 1:Ny]
    Cxz_prof       = [planeavg(Cxz_h, j) for j in 1:Ny]
    Cyz_prof       = [planeavg(Cyz_h, j) for j in 1:Ny]
    txx_prof       = [planeavg(txx_h, j) for j in 1:Ny]
    txy_prof       = [planeavg(txy_h, j) for j in 1:Ny]
    tyy_prof       = [planeavg(tyy_h, j) for j in 1:Ny]
    tzz_prof       = [planeavg(tzz_h, j) for j in 1:Ny]

    eta_s = Float64(ν_s)                 # ρ₀ = 1
    eta_p = ν_p_eff
    eta_total = ν_total

    # Realized local shear rate from the central difference of the measured
    # x,z-averaged velocity profile (what the conformation kernel actually
    # advects). Interior central diff; one-sided at the wall rows.
    gamma_dot_meas_prof = similar(profile)
    for j in 1:Ny
        if j == 1
            gamma_dot_meas_prof[j] = abs(profile[2] - profile[1])
        elseif j == Ny
            gamma_dot_meas_prof[j] = abs(profile[Ny] - profile[Ny-1])
        else
            gamma_dot_meas_prof[j] = abs((profile[j+1] - profile[j-1]) / 2)
        end
    end

    # Profiles of the stress diagnostics.
    tau_xy_prof = [eta_s * gamma_dot_meas_prof[j] + txy_prof[j] for j in 1:Ny]
    N1_prof     = [txx_prof[j] - tyy_prof[j] for j in 1:Ny]
    N2_prof     = [tyy_prof[j] - tzz_prof[j] for j in 1:Ny]

    return (ux=ux_h, ρ=Array(ρ),
            C_xx=Cxx_h, C_xy=Cxy_h, C_xz=Cxz_h,
            C_yy=Cyy_h, C_yz=Cyz_h, C_zz=Czz_h,
            tau_p_xx=txx_h, tau_p_xy=txy_h, tau_p_yy=tyy_h, tau_p_zz=tzz_h,
            profile=profile, u_analytical=u_analytical, u_max=u_max,
            gamma_dot_an=gamma_dot_an, gamma_dot_meas_prof=gamma_dot_meas_prof,
            Cxy_prof=Cxy_prof, Cxx_prof=Cxx_prof, Cyy_prof=Cyy_prof,
            Czz_prof=Czz_prof, Cxz_prof=Cxz_prof, Cyz_prof=Cyz_prof,
            tau_xy_prof=tau_xy_prof, N1_prof=N1_prof, N2_prof=N2_prof,
            gamma_dot_wall=gamma_dot_wall, Wi_wall=Wi_wall, lambda=λ_p,
            beta=beta, Re=Re, eta_s=eta_s, eta_p=eta_p, eta_total=eta_total)
end
