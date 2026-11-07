# Residual-converged steady forward solve for the AD cylinder path.
# CPU Float64 only; this iterates the inline AD step, not the fused kernel.

# Same profile as `run_cylinder_libb_2d`: zero on the half-way channel walls,
# row sum exactly u_ref * Ny with u_ref = (2/3) u_in.
function ad_inlet_profile(Ny::Int, u_in::Real, inlet::Symbol)
    if inlet === :parabolic
        return _halfway_poiseuille_profile(Float64, Ny, (2 / 3) * Float64(u_in))
    elseif inlet === :uniform
        return fill(Float64(u_in), Ny)
    end
    throw(ArgumentError("unknown inlet $(inlet); expected :parabolic or :uniform"))
end

function ad_initial_cylinder_equilibrium(Nx::Int, Ny::Int, u_profile)
    f = zeros(Float64, Nx, Ny, 9)
    @inbounds for j in 1:Ny, i in 1:Nx
        rho = 1.0
        ux = u_profile[j]
        uy = 0.0
        usq = ux * ux
        f[i, j, 1] = ad_feq(Val(1), rho, ux, uy, usq)
        f[i, j, 2] = ad_feq(Val(2), rho, ux, uy, usq)
        f[i, j, 3] = ad_feq(Val(3), rho, ux, uy, usq)
        f[i, j, 4] = ad_feq(Val(4), rho, ux, uy, usq)
        f[i, j, 5] = ad_feq(Val(5), rho, ux, uy, usq)
        f[i, j, 6] = ad_feq(Val(6), rho, ux, uy, usq)
        f[i, j, 7] = ad_feq(Val(7), rho, ux, uy, usq)
        f[i, j, 8] = ad_feq(Val(8), rho, ux, uy, usq)
        f[i, j, 9] = ad_feq(Val(9), rho, ux, uy, usq)
    end
    return f
end

function ad_relative_step_residual(f_out, f_in)
    num = 0.0
    den = 0.0
    @inbounds for idx in eachindex(f_in)
        d = f_out[idx] - f_in[idx]
        num += d * d
        den += f_in[idx] * f_in[idx]
    end
    return sqrt(num) / sqrt(den)
end

function ad_geometry_cylinder(Nx::Int, Ny::Int, cx::Real, cy::Real, radius::Real)
    q_wall, is_solid =
        precompute_q_wall_cylinder(Nx, Ny, Float64(cx), Float64(cy),
                                   Float64(radius); FT=Float64)
    return (; q_wall=q_wall, is_solid=is_solid)
end

function ad_forward_solve(; Nx::Int, Ny::Int,
                            cx::Union{Nothing,Real}=nothing,
                            cy::Union{Nothing,Real}=nothing,
                            radius::Real,
                            u_in::Real,
                            nu::Real,
                            inlet::Symbol=:parabolic,
                            rho_out::Real=1.0,
                            tol::Real=1e-12,
                            max_steps::Int=120_000,
                            f_init=nothing)
    cx_f = Float64(isnothing(cx) ? Nx ÷ 4 : cx)
    cy_f = Float64(isnothing(cy) ? Ny ÷ 2 : cy)
    radius_f = Float64(radius)
    nu_f = Float64(nu)
    rho_out_f = Float64(rho_out)

    geom = ad_geometry_cylinder(Nx, Ny, cx_f, cy_f, radius_f)
    u_profile = ad_inlet_profile(Ny, u_in, inlet)
    u_ref = inlet === :parabolic ? (2.0 / 3.0) * Float64(u_in) : Float64(u_in)
    D = 2.0 * radius_f
    s_plus, s_minus = ad_trt_rates_inline(nu_f)

    f_in = f_init === nothing ?
        ad_initial_cylinder_equilibrium(Nx, Ny, u_profile) : copy(f_init)
    f_out = similar(f_in)
    residual = Inf
    n_iter = 0
    converged = false

    for step in 1:max_steps
        ad_step!(f_out, f_in, geom.q_wall, geom.is_solid, u_profile,
                 rho_out_f, s_plus, s_minus, Nx, Ny)
        residual = ad_relative_step_residual(f_out, f_in)
        n_iter = step
        if residual < Float64(tol)
            converged = true
            break
        end
        f_in, f_out = f_out, f_in
    end

    f_star = copy(f_out)
    cd = cd_pure(f_star, geom.q_wall, u_ref, D, Nx, Ny)

    return (;
        f_star=f_star,
        q_wall=geom.q_wall,
        is_solid=geom.is_solid,
        u_profile=u_profile,
        rho_out=rho_out_f,
        nu=nu_f,
        s_plus=s_plus,
        s_minus=s_minus,
        Nx=Nx,
        Ny=Ny,
        cx=cx_f,
        cy=cy_f,
        radius=radius_f,
        u_in=Float64(u_in),
        u_ref=u_ref,
        D=D,
        Cd=cd,
        n_iter=n_iter,
        residual=Float64(residual),
        converged=converged,
        inlet=inlet,
    )
end

function ad_thermal_forward_solve(; N::Int,
                                    Ra::Real,
                                    Pr::Real,
                                    L::Union{Nothing,Real}=nothing,
                                    q_hot::Real=0.5,
                                    q_cold::Real=0.7,
                                    T_hot::Real=1.0,
                                    T_cold::Real=0.0,
                                    tol::Real=1e-11,
                                    max_steps::Int=450_000,
                                    w_init=nothing)
    p = ad_natconv_params(; N=N, Ra=Ra, Pr=Pr,
                          T_hot=T_hot, T_cold=T_cold)
    L_f = Float64(isnothing(L) ? Float64(N - 1) + Float64(q_hot) + Float64(q_cold) : L)
    geom = ad_cavity_wall_geometry(p.Nx, p.Ny, L_f; q_hot=q_hot)
    w_in = w_init === nothing ?
           ad_initial_thermal_w(p, geom.x_cold, geom.q_hot) : copy(w_init)
    w_out = similar(w_in)
    residual = Inf
    n_iter = 0
    converged = false

    for step in 1:max_steps
        ad_thermal_cut_step!(w_out, w_in, geom.q_wall, geom.q_wall, p)
        residual = ad_relative_step_residual(w_out, w_in)
        n_iter = step
        if residual < Float64(tol)
            converged = true
            break
        end
        w_in, w_out = w_out, w_in
    end

    w_star = copy(w_out)
    nusselt = nu_pure(w_star, geom.q_wall, p)
    return (;
        w_star=w_star,
        q_wall=geom.q_wall,
        dq_dL=geom.dq_dL,
        params=p,
        N=N,
        Nx=p.Nx,
        Ny=p.Ny,
        Ra=Float64(Ra),
        Pr=Float64(Pr),
        L=L_f,
        q_hot=geom.q_hot,
        q_cold=geom.q_cold,
        Nu=nusselt,
        n_iter=n_iter,
        residual=Float64(residual),
        converged=converged,
    )
end

"""
    ad_forward_solve_nufield(; Nx, Ny, cx, cy, radius, u_in, nu_field, inlet, rho_out,
                               tol, max_steps, f_init) -> NamedTuple

Forward solve for the per-row ν(y) field variant. Iterates `ad_step_nufield!` to
convergence using the same residual criterion as `ad_forward_solve`.
INTERNAL — access via `KrakeNN.ad_forward_solve_nufield`; not exported.
"""
function ad_forward_solve_nufield(; Nx::Int, Ny::Int,
                                  cx::Union{Nothing,Real}=nothing,
                                  cy::Union{Nothing,Real}=nothing,
                                  radius::Real,
                                  u_in::Real,
                                  nu_field::Vector{Float64},
                                  inlet::Symbol=:parabolic,
                                  rho_out::Real=1.0,
                                  tol::Real=1e-12,
                                  max_steps::Int=120_000,
                                  f_init=nothing)
    length(nu_field) == Ny ||
        throw(ArgumentError("nu_field length $(length(nu_field)) must equal Ny=$Ny"))
    cx_f = Float64(isnothing(cx) ? Nx ÷ 4 : cx)
    cy_f = Float64(isnothing(cy) ? Ny ÷ 2 : cy)
    radius_f = Float64(radius)
    rho_out_f = Float64(rho_out)

    geom = ad_geometry_cylinder(Nx, Ny, cx_f, cy_f, radius_f)
    u_profile = ad_inlet_profile(Ny, u_in, inlet)

    f_in = f_init === nothing ?
        ad_initial_cylinder_equilibrium(Nx, Ny, u_profile) : copy(f_init)
    f_out = similar(f_in)
    residual_val = Inf
    n_iter = 0
    converged = false

    for step in 1:max_steps
        ad_step_nufield!(f_out, f_in, geom.q_wall, geom.is_solid, u_profile,
                         rho_out_f, nu_field, Nx, Ny)
        residual_val = ad_relative_step_residual(f_out, f_in)
        n_iter = step
        if residual_val < Float64(tol)
            converged = true
            break
        end
        f_in, f_out = f_out, f_in
    end

    f_star = copy(f_out)
    return (;
        f_star=f_star,
        q_wall=geom.q_wall,
        is_solid=geom.is_solid,
        u_profile=u_profile,
        rho_out=rho_out_f,
        nu_field=copy(nu_field),
        Nx=Nx,
        Ny=Ny,
        cx=cx_f,
        cy=cy_f,
        radius=radius_f,
        u_in=Float64(u_in),
        n_iter=n_iter,
        residual=Float64(residual_val),
        converged=converged,
        inlet=inlet,
    )
end
