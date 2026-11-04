# --- 3D viscoelastic drivers (D3Q19 + 6-component conformation tensor) ---
#
# Mirrors `drivers/viscoelastic.jl::run_conformation_cylinder_libb_2d`
# in 3D. Uses:
#   - `compute_polymeric_force_3d!`  polymer body force F_poly = ∇·τ_p
#   - `fused_trt_libb_v2_guo_field_step_3d!`  solvent flow + fused Guo polymer
#                                    force (the validated 2D ∇·τ_p coupling;
#                                    polymer momentum consumed EXACTLY ONCE at ω_s)
#   - `apply_bc_rebuild_3d!` (BCSpec3D) for inlet/outlet
#   - `collide_conformation_3d!` ×6  for each conformation component
#   - `apply_polymer_wall_bc!` dispatching to `apply_cnebb_conformation_3d!`
#   - `update_polymer_stress_3d!`    Oldroyd-B / log-conf stress closure
#
# Drag: standard `compute_drag_libb_3d` post-collision on f_out. With the Guo
# ∇·τ_p coupling the polymer momentum enters as a VOLUME body force (not a
# post-collision f modification), so the standard halfway-BB MEA on the
# post-collision cut-link f captures the solvent traction; the polymer
# contribution is carried by the body force. (Replaces the prior Hermite-source
# rationale — `apply_hermite_source_3d!` is no longer in the coupling path.)
#
# Log-conformation 3D has low-level Ψ kernels, but this sphere driver has
# not yet wired the log-field inlet/outlet/wall treatment. For now
# `polymer_model` must be `OldroydBSpec`. The driver refuses to silently run
# with `LogConfOldroydB` in 3D.

"""
    run_conformation_sphere_libb_3d(; Nx, Ny, Nz, radius, cx, cy, cz,
                                       u_in=0.04, ν_s=0.04, ν_p=0.0, lambda=10.0,
                                       polymer_model=nothing,
                                       polymer_bc=CNEBB(),
                                       bcspec=nothing,
                                       inlet=:parabolic_y, ρ_out=1.0,
                                       tau_plus=1.0,
                                       max_steps=20_000, avg_window=5_000,
                                       backend=CPU(), FT=Float64)

3D viscoelastic flow past a sphere — Oldroyd-B (or any 3D-compatible
`AbstractPolymerModel`). The solvent uses the validated TRT + LI-BB V2
pipeline; the polymer uses 6 D3Q19 conformation distributions
(C_xx, C_xy, C_xz, C_yy, C_yz, C_zz) advected with TRT and coupled to f
via the Hermite source. CNEBB is applied at sphere-adjacent fluid cells.

`u_in` is the inlet centreline velocity for `:parabolic_y` (default —
y-only parabolic profile uniform in z; analog of the 2D channel),
`:parabolic` for fully 3D doubly parabolic, or `:uniform` for plug flow.

Returns NamedTuple `(ux, uy, uz, ρ, C_xx, C_xy, C_xz, C_yy, C_yz, C_zz,
                     tau_p_*, q_wall, is_solid, Cd, Fx, Fy, Fz,
                     Re, Wi, beta, u_ref, D)`.
"""
function run_conformation_sphere_libb_3d(;
        Nx::Int=120, Ny::Int=60, Nz::Int=60,
        radius::Real=8,
        cx::Union{Nothing,Real}=nothing,
        cy::Union{Nothing,Real}=nothing,
        cz::Union{Nothing,Real}=nothing,
        u_in::Real=0.04, ν_s::Real=0.04, ν_p::Union{Nothing,Real}=nothing,
        lambda::Real=10.0,
        polymer_model::Union{Nothing,AbstractPolymerModel}=nothing,
        polymer_bc::AbstractPolymerWallBC=CNEBB(),
        bcspec::Union{Nothing,BCSpec3D}=nothing,
        inlet::Symbol=:parabolic_y, ρ_out::Real=1.0,
        tau_plus::Real=1.0,
        max_steps::Int=20_000, avg_window::Int=5_000,
        backend=KernelAbstractions.CPU(),
        FT::Type{<:AbstractFloat}=Float64)

    # Resolve polymer model
    if polymer_model === nothing
        isnothing(ν_p) && error("supply either `polymer_model` or (`ν_p`, `lambda`).")
        G_ = FT(ν_p / lambda)
        polymer_model = OldroydBSpec(G=G_, λ=FT(lambda))
    end
    if uses_log_conformation(polymer_model)
        error("3D log-conformation is not yet wired into the sphere driver; pass an OldroydBSpec model.")
    end
    λ_p     = polymer_relaxation_time(polymer_model)
    ν_p_eff = polymer_modulus(polymer_model) * λ_p

    cx = isnothing(cx) ? Nx ÷ 4 : Float64(cx)
    cy = isnothing(cy) ? Ny ÷ 2 : Float64(cy)
    cz = isnothing(cz) ? Nz ÷ 2 : Float64(cz)

    D = 2 * Float64(radius)
    ν_total = ν_s + ν_p_eff
    beta    = ν_s / ν_total
    # ω_s = 1/(3ν_s+0.5) is applied inside `fused_trt_libb_v2_guo_field_step_3d!`
    # via trt_rates(ν_s); the polymer enters as the ∇·τ_p Guo force at that rate.

    # Reference velocity for Cd / Wi (matches `run_sphere_libb_3d`)
    u_ref = inlet === :parabolic   ? (4/9) * Float64(u_in) :
            inlet === :parabolic_y ? (2/3) * Float64(u_in) :
                                       Float64(u_in)
    Re = u_ref * D / ν_total
    Wi = λ_p * u_ref / radius

    @info "Conformation sphere (LI-BB V2, 3D)" Nx Ny Nz radius Re Wi beta λ_p tau_plus inlet polymer_bc=typeof(polymer_bc) polymer_model=typeof(polymer_model)

    # --- Sphere geometry (analytic q_wall) ---
    q_wall_h, is_solid_h = precompute_q_wall_sphere_3d(Nx, Ny, Nz, cx, cy, cz,
                                                        Float64(radius); FT=FT)

    # --- Inlet velocity + conformation profiles ---
    u_profile_h = zeros(FT, Ny, Nz)
    Hy = FT(Ny - 1); Hz = FT(Nz - 1)
    if inlet === :uniform
        fill!(u_profile_h, FT(u_in))
    elseif inlet === :parabolic
        for k in 1:Nz, j in 1:Ny
            yy = FT(j - 1); zz = FT(k - 1)
            u_profile_h[j, k] = FT(16) * FT(u_in) *
                                yy * (Hy - yy) * zz * (Hz - zz) / (Hy^2 * Hz^2)
        end
    elseif inlet === :parabolic_y
        for j in 1:Ny
            yy = FT(j - 1)
            val = FT(4) * FT(u_in) * yy * (Hy - yy) / Hy^2
            for k in 1:Nz
                u_profile_h[j, k] = val
            end
        end
    else
        error("unknown inlet $(inlet); expected :uniform|:parabolic|:parabolic_y")
    end

    # Analytic Oldroyd-B inlet conformation. For y-only parabolic shear we
    # have ∂u/∂y as the only non-zero gradient → C_xy = λ·∂u/∂y,
    # C_xx = 1 + 2·(λ·∂u/∂y)², other components = identity. For doubly-
    # parabolic and uniform inlets we approximate with the y-shear only
    # (z-shear contributes a C_xz that is symmetric and integrates to ~0
    # at the centerline). This is the same simplification as 2D.
    C_xx_inlet_h = ones(FT, Ny, Nz);  C_xy_inlet_h = zeros(FT, Ny, Nz)
    C_xz_inlet_h = zeros(FT, Ny, Nz); C_yy_inlet_h = ones(FT, Ny, Nz)
    C_yz_inlet_h = zeros(FT, Ny, Nz); C_zz_inlet_h = ones(FT, Ny, Nz)
    for k in 1:Nz, j in 1:Ny
        y = FT(j) - FT(0.5)
        if inlet === :parabolic_y
            dudy = FT(u_in) * FT(4) * (Hy + one(FT) - FT(2)*y) / ((Hy + one(FT))^2)
            C_xy_inlet_h[j, k] = FT(λ_p) * dudy
            C_xx_inlet_h[j, k] = FT(1) + FT(2) * (FT(λ_p) * dudy)^2
        end
    end

    # --- Device allocations ---
    q_wall   = KernelAbstractions.allocate(backend, FT,   Nx, Ny, Nz, 19)
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny, Nz)
    uw_x     = KernelAbstractions.allocate(backend, FT,   Nx, Ny, Nz, 19)
    uw_y     = KernelAbstractions.allocate(backend, FT,   Nx, Ny, Nz, 19)
    uw_z     = KernelAbstractions.allocate(backend, FT,   Nx, Ny, Nz, 19)
    f_in     = KernelAbstractions.allocate(backend, FT,   Nx, Ny, Nz, 19)
    f_out    = KernelAbstractions.allocate(backend, FT,   Nx, Ny, Nz, 19)
    ρ        = KernelAbstractions.allocate(backend, FT,   Nx, Ny, Nz)
    ux       = KernelAbstractions.allocate(backend, FT,   Nx, Ny, Nz)
    uy       = KernelAbstractions.allocate(backend, FT,   Nx, Ny, Nz)
    uz       = KernelAbstractions.allocate(backend, FT,   Nx, Ny, Nz)
    u_profile  = KernelAbstractions.allocate(backend, FT, Ny, Nz)
    C_xx_inlet = KernelAbstractions.allocate(backend, FT, Ny, Nz)
    C_xy_inlet = KernelAbstractions.allocate(backend, FT, Ny, Nz)
    C_xz_inlet = KernelAbstractions.allocate(backend, FT, Ny, Nz)
    C_yy_inlet = KernelAbstractions.allocate(backend, FT, Ny, Nz)
    C_yz_inlet = KernelAbstractions.allocate(backend, FT, Ny, Nz)
    C_zz_inlet = KernelAbstractions.allocate(backend, FT, Ny, Nz)

    copyto!(q_wall, q_wall_h);   copyto!(is_solid, is_solid_h)
    fill!(uw_x, zero(FT)); fill!(uw_y, zero(FT)); fill!(uw_z, zero(FT))
    fill!(ρ, one(FT));      fill!(ux, zero(FT))
    fill!(uy, zero(FT));    fill!(uz, zero(FT))
    copyto!(u_profile, u_profile_h)
    copyto!(C_xx_inlet, C_xx_inlet_h); copyto!(C_xy_inlet, C_xy_inlet_h)
    copyto!(C_xz_inlet, C_xz_inlet_h); copyto!(C_yy_inlet, C_yy_inlet_h)
    copyto!(C_yz_inlet, C_yz_inlet_h); copyto!(C_zz_inlet, C_zz_inlet_h)

    # Default BCSpec3D: ZouHe velocity inlet, ZouHe pressure outlet
    if bcspec === nothing
        bcspec = BCSpec3D(; west = ZouHeVelocity(u_profile),
                            east = ZouHePressure(FT(ρ_out)))
    end

    # Initialize f to equilibrium at the inlet profile velocity (same
    # warm-up trick as the 2D cylinder driver)
    f_in_h = zeros(FT, Nx, Ny, Nz, 19)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx, q in 1:19
        u0 = is_solid_h[i, j, k] ? zero(FT) : u_profile_h[j, k]
        f_in_h[i, j, k, q] = Kraken.equilibrium(D3Q19(), one(FT), u0,
                                                 zero(FT), zero(FT), q)
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
    # Guo body force fused into the TRT+LI-BB collision at the solvent rate ω_s
    # (lattice viscosity = ν_s), replacing the standalone re-relaxed Hermite
    # source. Non-periodic x (inlet/outlet) and z (walls), so the divergence
    # stencil clamps on both. 3D port of the validated 2D cylinder coupling.
    Fx_poly = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    Fy_poly = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    Fz_poly = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)

    Fx_sum = 0.0; Fy_sum = 0.0; Fz_sum = 0.0; n_avg = 0

    for step in 1:max_steps
        # --- Polymer body force: F_poly = ∇·τ_p (clamped x and z) ---
        compute_polymeric_force_3d!(Fx_poly, Fy_poly, Fz_poly,
                                      tau_p_xx, tau_p_xy, tau_p_xz,
                                      tau_p_yy, tau_p_yz, tau_p_zz;
                                      periodic_x=false, periodic_z=false)

        # --- Solvent TRT + LI-BB V2 with fused Guo polymer force ---
        fused_trt_libb_v2_guo_field_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                               q_wall, uw_x, uw_y, uw_z,
                                               Fx_poly, Fy_poly, Fz_poly,
                                               Nx, Ny, Nz, FT(ν_s))
        # Pre-collision Zou-He rebuild at inlet/outlet
        apply_bc_rebuild_3d!(f_out, f_in, bcspec, ν_s, Nx, Ny, Nz)

        # Drag (standard halfway-BB MEA on post-collision f) — note:
        # `compute_drag_libb_3d` reads f_out only and integrates over
        # cut links via q_wall; with q_w on a symmetric sphere this gives
        # axisymmetric drag in x.
        if step > max_steps - avg_window
            drag = compute_drag_libb_3d(f_out, q_wall, Nx, Ny, Nz)
            Fx_sum += drag.Fx; Fy_sum += drag.Fy; Fz_sum += drag.Fz
            n_avg += 1
        end

        # --- Conformation TRT (6 components) ---
        stream_3d!(g_xx_buf, g_xx, Nx, Ny, Nz)
        stream_3d!(g_xy_buf, g_xy, Nx, Ny, Nz)
        stream_3d!(g_xz_buf, g_xz, Nx, Ny, Nz)
        stream_3d!(g_yy_buf, g_yy, Nx, Ny, Nz)
        stream_3d!(g_yz_buf, g_yz, Nx, Ny, Nz)
        stream_3d!(g_zz_buf, g_zz, Nx, Ny, Nz)

        apply_polymer_wall_bc!(g_xx_buf, g_xx, is_solid, C_xx, polymer_bc)
        apply_polymer_wall_bc!(g_xy_buf, g_xy, is_solid, C_xy, polymer_bc)
        apply_polymer_wall_bc!(g_xz_buf, g_xz, is_solid, C_xz, polymer_bc)
        apply_polymer_wall_bc!(g_yy_buf, g_yy, is_solid, C_yy, polymer_bc)
        apply_polymer_wall_bc!(g_yz_buf, g_yz, is_solid, C_yz, polymer_bc)
        apply_polymer_wall_bc!(g_zz_buf, g_zz, is_solid, C_zz, polymer_bc)

        reset_conformation_inlet_3d!(g_xx_buf, C_xx_inlet, u_profile, Ny, Nz)
        reset_conformation_inlet_3d!(g_xy_buf, C_xy_inlet, u_profile, Ny, Nz)
        reset_conformation_inlet_3d!(g_xz_buf, C_xz_inlet, u_profile, Ny, Nz)
        reset_conformation_inlet_3d!(g_yy_buf, C_yy_inlet, u_profile, Ny, Nz)
        reset_conformation_inlet_3d!(g_yz_buf, C_yz_inlet, u_profile, Ny, Nz)
        reset_conformation_inlet_3d!(g_zz_buf, C_zz_inlet, u_profile, Ny, Nz)
        reset_conformation_outlet_3d!(g_xx_buf, Nx, Ny, Nz)
        reset_conformation_outlet_3d!(g_xy_buf, Nx, Ny, Nz)
        reset_conformation_outlet_3d!(g_xz_buf, Nx, Ny, Nz)
        reset_conformation_outlet_3d!(g_yy_buf, Nx, Ny, Nz)
        reset_conformation_outlet_3d!(g_yz_buf, Nx, Ny, Nz)
        reset_conformation_outlet_3d!(g_zz_buf, Nx, Ny, Nz)

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

    Fx = n_avg > 0 ? Fx_sum / n_avg : 0.0
    Fy = n_avg > 0 ? Fy_sum / n_avg : 0.0
    Fz = n_avg > 0 ? Fz_sum / n_avg : 0.0
    A  = π * Float64(radius)^2
    Cd = 2.0 * Fx / (u_ref^2 * A)

    @info "Conformation sphere (LI-BB V2, 3D) result" Cd Re Wi

    return (ux=Array(ux), uy=Array(uy), uz=Array(uz), ρ=Array(ρ),
            C_xx=Array(C_xx), C_xy=Array(C_xy), C_xz=Array(C_xz),
            C_yy=Array(C_yy), C_yz=Array(C_yz), C_zz=Array(C_zz),
            tau_p_xx=Array(tau_p_xx), tau_p_xy=Array(tau_p_xy), tau_p_xz=Array(tau_p_xz),
            tau_p_yy=Array(tau_p_yy), tau_p_yz=Array(tau_p_yz), tau_p_zz=Array(tau_p_zz),
            q_wall=Array(q_wall), is_solid=Array(is_solid),
            Cd=Cd, Fx=Fx, Fy=Fy, Fz=Fz,
            Re=Re, Wi=Wi, beta=beta, u_ref=u_ref, D=D)
end
