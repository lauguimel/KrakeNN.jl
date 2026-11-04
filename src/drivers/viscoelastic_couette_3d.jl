# --- 3D viscoelastic planar-Couette driver (D3Q19 + 6-component conformation) ---
#
# Closest-to-analytical 3D viscoelastic canary: steady simple shear in a small
# box, shear in y, flow in x, neutral z. Periodic in x AND z (true periodicity
# via `stream_periodic_xz_wall_y_3d!`); moving no-slip walls on the y-faces via
# Zou-He (bottom u_x = 0, top u_x = U) → uniform shear rate γ̇ = U / (Ny − 1).
#
# Reuses the geometry-agnostic 3D constitutive stack from the sphere driver:
#   - `compute_polymeric_force_3d!`      polymer body force F_poly = ∇·τ_p
#   - `collide_guo_field_3d!`            BGK solvent collision + fused Guo force
#   - `stream_periodic_xz_wall_y_3d!`    periodic-xz / y-wall streamer (NEW)
#   - `apply_zou_he_south_3d!/north_3d!` moving walls in y
#   - `collide_conformation_3d!` ×6      TRT conformation transport
#   - `apply_cnebb_conformation_y_walls_3d!`  conformation wall BC (y-faces)
#   - `update_polymer_stress_3d!`        Oldroyd-B τ_p = G·(C − I)
#
# Closed-form steady Oldroyd-B targets (Wi ≡ λγ̇), bulk (away from y-walls):
#   C_xy = Wi,  C_xx = 1 + 2 Wi²,  C_yy = C_zz = 1,  C_xz = C_yz = 0
#   u_x(y) = γ̇·(y − y_mid)
#   τ_xy = η_total·γ̇,  N1 = 2 η_p λ γ̇²,  N2 = τ_yy − τ_zz = 0.

"""
    run_conformation_couette_libb_3d(; Nx, Ny, Nz, U, ν_s, ν_p, lambda,
                                       polymer_model=nothing,
                                       tau_plus=1.0, max_steps, backend, FT)

3D viscoelastic planar Couette (Oldroyd-B). Flow in x, shear in y (walls at
j=1, j=Ny), neutral and periodic z; periodic x. Bottom wall u_x = 0, top wall
u_x = `U` → uniform γ̇ = U / (Ny − 1). The solvent uses BGK + the periodic-xz /
y-wall streamer; the polymer uses the 6-component conformation TRT-LBM coupled
through the Hermite stress source.

Returns a NamedTuple with the full fields plus convenience bulk scalars
(`Cxy, Cxx, Cyy, Czz, Cxz, Cyz, tau_xy, N1, N2`) averaged over the central
y-slab (and the full x,z planes) used for the analytical assertions, together
with `gamma_dot`, `Wi`, `beta`, `eta_s`, `eta_p`, `eta_total`.
"""
function run_conformation_couette_libb_3d(;
        Nx::Int=8, Ny::Int=33, Nz::Int=8,
        U::Real=0.02, ν_s::Real=0.05, ν_p::Union{Nothing,Real}=0.05,
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
        error("3D log-conformation is not yet wired into the Couette driver; pass an OldroydBSpec model.")
    end
    λ_p     = polymer_relaxation_time(polymer_model)
    ν_p_eff = polymer_modulus(polymer_model) * λ_p

    # Half-way bounce-back places the walls ½ cell outside the first/last fluid
    # row, so the effective wall-to-wall gap is Ny (bottom wall at j=0.5, top at
    # j=Ny+0.5) and γ̇ = U / Ny is exactly uniform across the fluid rows.
    H = Float64(Ny)                   # wall-to-wall gap in lattice units
    γ̇ = Float64(U) / H                # uniform shear rate
    Wi = λ_p * γ̇
    ν_total = Float64(ν_s) + ν_p_eff
    beta    = Float64(ν_s) / ν_total
    Re = Float64(U) * H / ν_total
    ω_s = 1.0 / (3.0 * Float64(ν_s) + 0.5)  # solvent rate; polymer via ∇·τ_p Guo force

    @info "Conformation Couette (3D)" Nx Ny Nz U γ̇ Wi beta λ_p Re tau_plus polymer_model=typeof(polymer_model)

    # No obstacle: every cell is fluid.
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny, Nz)
    fill!(is_solid, false)

    # --- Solvent allocations ---
    f_in  = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz, 19)
    f_out = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz, 19)
    ρ  = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz)
    ux = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz)
    uy = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz)
    uz = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz)
    fill!(ρ, one(FT)); fill!(ux, zero(FT)); fill!(uy, zero(FT)); fill!(uz, zero(FT))

    # Initialize f to equilibrium at the analytical linear Couette profile.
    # With half-way bounce-back walls the fluid row j sits at height (j − 0.5),
    # so u_x(j) = γ̇·(j − 0.5). Warm start accelerates convergence; the moving
    # walls hold it.
    f_in_h = zeros(FT, Nx, Ny, Nz, 19)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        u0 = FT(γ̇ * (j - 0.5))
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

    # --- Polymer body force F_poly = ∇·τ_p (validated 2D Guo coupling) ---
    # The polymer enters the momentum equation EXACTLY ONCE as a first-moment
    # Guo body force at the solvent rate ω_s (lattice viscosity = ν_s), not via
    # a standalone re-relaxed Hermite source. In the uniform-shear Couette bulk
    # ∇·τ_p ≈ 0, so the moving walls set γ̇ and the constitutive checks hold; the
    # force only matters in the near-wall stress gradient.
    Fx_poly = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    Fy_poly = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    Fz_poly = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)

    Uw = FT(U)

    for step in 1:max_steps
        # --- Polymer body force: F_poly = ∇·τ_p (periodic x AND z) ---
        compute_polymeric_force_3d!(Fx_poly, Fy_poly, Fz_poly,
                                      tau_p_xx, tau_p_xy, tau_p_xz,
                                      tau_p_yy, tau_p_yz, tau_p_zz;
                                      periodic_x=true, periodic_z=true)
        # --- Solvent: BGK+Guo-field collide (force fused once) → moving walls ---
        collide_guo_field_3d!(f_in, is_solid, Fx_poly, Fy_poly, Fz_poly, FT(ω_s))
        # Moving walls in y via half-way bounce-back + Ladd correction:
        # bottom (j=1) u=0, top (j=Ny) u=(U,0,0). Clean γ̇ = U/Ny, no Zou-He
        # node-velocity overshoot.
        stream_periodic_xz_movingwall_y_3d!(f_out, f_in, zero(FT), Uw,
                                             Nx, Ny, Nz; rho_w=one(FT))

        compute_macroscopic_forced_field_3d!(ρ, ux, uy, uz, f_out, Fx_poly, Fy_poly, Fz_poly)

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

    compute_polymeric_force_3d!(Fx_poly, Fy_poly, Fz_poly,
                                  tau_p_xx, tau_p_xy, tau_p_xz,
                                  tau_p_yy, tau_p_yz, tau_p_zz;
                                  periodic_x=true, periodic_z=true)
    compute_macroscopic_forced_field_3d!(ρ, ux, uy, uz, f_in, Fx_poly, Fy_poly, Fz_poly)

    Cxx_h = Array(C_xx); Cxy_h = Array(C_xy); Cxz_h = Array(C_xz)
    Cyy_h = Array(C_yy); Cyz_h = Array(C_yz); Czz_h = Array(C_zz)
    txx_h = Array(tau_p_xx); txy_h = Array(tau_p_xy)
    tyy_h = Array(tau_p_yy); tzz_h = Array(tau_p_zz)

    # Bulk = central y-slab (away from the two y-walls), full x,z planes.
    jlo = max(2, (Ny ÷ 2) - max(1, Ny ÷ 8))
    jhi = min(Ny - 1, (Ny ÷ 2) + max(1, Ny ÷ 8))
    bulk(A) = sum(@view A[:, jlo:jhi, :]) / (Nx * (jhi - jlo + 1) * Nz)

    Cxy_b = bulk(Cxy_h); Cxx_b = bulk(Cxx_h); Cyy_b = bulk(Cyy_h)
    Czz_b = bulk(Czz_h); Cxz_b = bulk(Cxz_h); Cyz_b = bulk(Cyz_h)

    eta_s = Float64(ν_s)              # ρ₀ = 1
    eta_p = ν_p_eff
    eta_total = ν_total

    # Derived stresses from the polymer-stress field + the solvent shear.
    # τ_xy_total = η_s·γ̇ (solvent) + τ_p,xy (polymer); N1 = τ_xx − τ_yy
    # (only the polymer contributes a normal-stress difference for OB).
    tau_xy_b = eta_s * γ̇ + bulk(txy_h)
    N1_b = bulk(txx_h) - bulk(tyy_h)
    N2_b = bulk(tyy_h) - bulk(tzz_h)

    # Measured (realized) shear rate at the centre-line: central difference of
    # the x,z-averaged velocity profile at the mid-y cell. This is what the
    # conformation kernel actually advects, so it gives the self-consistent
    # local Weissenberg number Wi_local = λ·γ̇_meas — the rigorous constitutive
    # check (independent of the small profile curvature induced by the finite-β
    # momentum coupling, which makes γ̇_meas drift a few % from the imposed U/Ny).
    ux_h = Array(ux)
    prof = [sum(@view ux_h[:, j, :]) / (Nx * Nz) for j in 1:Ny]
    jc = Ny ÷ 2 + 1
    gamma_dot_meas = (prof[jc + 1] - prof[jc - 1]) / 2
    Wi_local = λ_p * gamma_dot_meas

    # Centre-cell (x,z-averaged) conformation & polymer stress for the
    # self-consistent constitutive assertions.
    cell(A) = sum(@view A[:, jc, :]) / (Nx * Nz)
    Cxy_c = cell(Cxy_h); Cxx_c = cell(Cxx_h)
    Cyy_c = cell(Cyy_h); Czz_c = cell(Czz_h)
    Cxz_c = cell(Cxz_h); Cyz_c = cell(Cyz_h)
    tau_xy_c = eta_s * gamma_dot_meas + cell(txy_h)
    N1_c = cell(txx_h) - cell(tyy_h)
    N2_c = cell(tyy_h) - cell(tzz_h)

    return (ux=ux_h, uy=Array(uy), uz=Array(uz), ρ=Array(ρ),
            C_xx=Cxx_h, C_xy=Cxy_h, C_xz=Cxz_h,
            C_yy=Cyy_h, C_yz=Cyz_h, C_zz=Czz_h,
            tau_p_xx=txx_h, tau_p_xy=txy_h,
            tau_p_yy=tyy_h, tau_p_zz=tzz_h,
            Cxy=Cxy_b, Cxx=Cxx_b, Cyy=Cyy_b, Czz=Czz_b, Cxz=Cxz_b, Cyz=Cyz_b,
            tau_xy=tau_xy_b, N1=N1_b, N2=N2_b,
            Cxy_c=Cxy_c, Cxx_c=Cxx_c, Cyy_c=Cyy_c, Czz_c=Czz_c,
            Cxz_c=Cxz_c, Cyz_c=Cyz_c,
            tau_xy_c=tau_xy_c, N1_c=N1_c, N2_c=N2_c,
            gamma_dot=γ̇, gamma_dot_meas=gamma_dot_meas,
            Wi=Wi, Wi_local=Wi_local, lambda=λ_p,
            beta=beta, Re=Re,
            eta_s=eta_s, eta_p=eta_p, eta_total=eta_total,
            profile=prof, bulk_j=(jlo, jhi))
end
