"""
    gnf_reference_scales(; Ra, Pr, nu0, H, T_hot, T_cold)

Compute cavity scales using the reference kinematic viscosity `nu0`:
`alpha = nu0/Pr`, `beta_g = Ra*nu0*alpha/((T_hot-T_cold)*H^3)`.
These reference-viscosity Ra/Pr definitions are independent of the local
constitutive viscosity and are isolated here for future physical definitions.
"""
function gnf_reference_scales(; Ra, Pr, nu0, H, T_hot, T_cold)
    nu0 > 0 && Pr > 0 && H > 0 || throw(ArgumentError("nu0, Pr and H must be positive"))
    delta_T = T_hot - T_cold
    iszero(delta_T) && throw(ArgumentError("T_hot and T_cold must differ"))
    alpha = nu0 / Pr
    return (; alpha, beta_g=Ra * nu0 * alpha / (delta_T * H^3),
            T_ref=(T_hot + T_cold) / 2, delta_T)
end

"""
    cavity_powerlaw(n; nu0=0.05, alpha, H, nu_min=1e-4, nu_max=1.0)

Construct a power law with `K = nu0*(alpha/H^2)^(1-n)`. The diffusive shear
scale `alpha/H^2` is a placeholder pending a published power-law Ra/Pr
definition; this helper does not implement a quantitative literature mapping.
The floating-point type of `nu0` determines the model parameter type.
"""
function cavity_powerlaw(n; nu0=0.05, alpha, H, nu_min=1e-4, nu_max=1.0)
    T = typeof(float(nu0))
    nu0 > 0 && alpha > 0 && H > 0 && 0 < nu_min <= nu_max ||
        throw(ArgumentError("positive scales and ordered positive viscosity bounds required"))
    K = T(nu0) * (T(alpha) / T(H)^2)^(one(T) - T(n))
    return PowerLaw(K, T(n); nu_min=T(nu_min), nu_max=T(nu_max))
end

# Exterior-link bounce-back belongs to stream_pull_node. Every stored node is
# fluid, including the wall-adjacent nodes: no second solid-node bounce-back.
@kernel function _natconv_gnf_transport!(f_out, @Const(f_in), g_out, @Const(g_in),
                                        Temp, rho, ux_field, uy_field, Fx, Fy,
                                        N, omega_T, beta_g, T_ref, T_hot, T_cold)
    i, j = @index(Global, NTuple)
    @inbounds begin
        fp = stream_pull_node(f_in, i, j, N, N)
        gp = apply_thermal_bc_cavity(stream_pull_node(g_in, i, j, N, N),
                                     i, j, N, N, T_hot, T_cold)
        temperature = gp[1]+gp[2]+gp[3]+gp[4]+gp[5]+gp[6]+gp[7]+gp[8]+gp[9]
        density, ux, uy, fy = macroscopic_boussinesq(fp, temperature, beta_g, T_ref)
        Temp[i,j] = temperature
        rho[i,j] = density
        ux_field[i,j] = ux
        uy_field[i,j] = uy
        Fx[i,j] = zero(temperature)
        Fy[i,j] = fy
        gp_out, _ = collide_thermal_node(gp, ux, uy, omega_T)
        write_populations!(g_out, i, j, gp_out)
        write_populations!(f_out, i, j, fp)
    end
end

function _gnf_hot_wall_nusselt(Temp, H, delta_T)
    T = eltype(Temp)
    total = zero(T)
    for j in 2:size(Temp, 2)-1
        total += -H * (-3Temp[1,j] + 4Temp[2,j] - Temp[3,j]) / T(2) / delta_T
    end
    return total / (size(Temp, 2) - 2)
end

_gnf_clamped_fraction(::GeneralizedNewtonian, tau) = 0.0
function _gnf_clamped_fraction(model::PowerLaw, tau)
    T = eltype(tau)
    lo = one(T) / (one(T) / (T(3) * model.nu_min + T(0.5)))
    hi = one(T) / (one(T) / (T(3) * model.nu_max + T(0.5)))
    return count(t -> t == lo || t == hi, tau) / length(tau)
end

"""
    run_natural_convection_gnf_2d(; N=64, Ra=1e4, Pr=0.71, rheology,
        nu0=0.05, T_hot=1.0, T_cold=0.0, max_steps=30000, backend=CPU(),
        FT=Float64, check_every=0, convergence_rtol=1e-6, min_steps=0)

Run a differentially heated D2Q9 cavity with generalized Newtonian viscosity.
Ra and Pr use the reference viscosity `nu0`, through `gnf_reference_scales`.
West/east temperatures are fixed; top/bottom are adiabatic; all walls are
no-slip. Thermal advection uses the pre-collision Guo velocity `j/rho+F/(2rho)`.

Args:
    rheology: GeneralizedNewtonian model whose floating parameters, including
        those in its thermal coupling, have type `FT`.
    check_every: Positive interval enables Nu-drift stopping after three
        consecutive relative changes below `convergence_rtol`, after `min_steps`.
        Zero disables convergence checks and executes exactly `max_steps`.

Returns:
    NamedTuple of CPU arrays. `ux`, `uy` and `rho` are the final step's
    pre-collision fields (initial rest fields if zero steps). `Temp` is the
    final thermal moment. `ν_field=(tau_field-0.5)/3` (also `nu_field`) is the viscosity used
    by the last collision. `clamped_fraction` counts the final PowerLaw nodes
    at either cutoff divided by all N*N fluid nodes, not time-accumulated;
    models without explicit clamps report zero. `converged` concerns Nu only.
"""
function run_natural_convection_gnf_2d(; N=64, Ra=1e4, Pr=0.71,
        rheology::GeneralizedNewtonian, nu0=0.05, T_hot=1.0, T_cold=0.0,
        max_steps=30000, backend=KernelAbstractions.CPU(), FT=Float64,
        check_every=0, convergence_rtol=1e-6, min_steps=0)
    N >= 3 || throw(ArgumentError("N must be at least 3"))
    max_steps >= 0 && check_every >= 0 && min_steps >= 0 && convergence_rtol > 0 ||
        throw(ArgumentError("invalid step count or convergence controls"))
    for object in (rheology, rheology.thermal), name in fieldnames(typeof(object))
        value = getfield(object, name)
        if value isa AbstractFloat && !(value isa FT)
            throw(ArgumentError("rheology and thermal coupling parameter types must match FT"))
        end
    end
    scales = gnf_reference_scales(; Ra=FT(Ra), Pr=FT(Pr), nu0=FT(nu0),
                                  H=FT(N), T_hot=FT(T_hot), T_cold=FT(T_cold))
    omega0 = one(FT) / (FT(3) * FT(nu0) + FT(0.5))
    omega_T = one(FT) / (FT(3) * scales.alpha + FT(0.5))
    config = LBMConfig(D2Q9(); Nx=N, Ny=N, ν=Float64(nu0), u_lid=0.0, max_steps)
    state = initialize_2d(config, FT; backend)
    f_in, f_out = state.f_in, state.f_out
    g_in = KernelAbstractions.zeros(backend, FT, N, N, 9)
    g_out = similar(g_in)
    Temp = KernelAbstractions.zeros(backend, FT, N, N)
    Fx, Fy, tau_field = similar(Temp), similar(Temp), similar(Temp)
    fill!(tau_field, one(FT) / omega0)
    w = weights(D2Q9())
    g_cpu = zeros(FT, N, N, 9)
    for j in 1:N, i in 1:N
        temperature = FT(T_hot - (T_hot - T_cold) * (i - 1) / (N - 1))
        temperature += FT(0.01 * (T_hot - T_cold)) * sin(FT(2π * i / N)) * sin(FT(π * j / N))
        for q in 1:9
            g_cpu[i,j,q] = FT(w[q]) * temperature
        end
    end
    copyto!(g_in, g_cpu)
    transport! = _natconv_gnf_transport!(backend)
    previous_Nu = FT(NaN)
    Nu_drift = FT(Inf)
    stable_checks = 0
    steps = 0
    converged = false
    for step in 1:max_steps
        transport!(f_out, f_in, g_out, g_in, Temp, state.ρ, state.ux, state.uy,
                   Fx, Fy, N, omega_T, scales.beta_g, scales.T_ref,
                   FT(T_hot), FT(T_cold); ndrange=(N, N))
        collide_rheology_thermal_2d!(f_out, state.is_solid, rheology, tau_field, Fx, Fy, Temp)
        f_in, f_out = f_out, f_in
        g_in, g_out = g_out, g_in
        steps = step
        if check_every > 0 && step % check_every == 0
            KernelAbstractions.synchronize(backend)
            current_Nu = _gnf_hot_wall_nusselt(Array(Temp), FT(N), scales.delta_T)
            Nu_drift = abs(current_Nu - previous_Nu) / max(abs(current_Nu), eps(FT))
            stable_checks = step >= min_steps && Nu_drift < convergence_rtol ? stable_checks + 1 : 0
            previous_Nu = current_Nu
            if stable_checks >= 3
                converged = true
                break
            end
        end
    end
    compute_temperature_2d!(Temp, g_in)
    KernelAbstractions.synchronize(backend)
    temperature = Array(Temp)
    tau = Array(tau_field)
    nu_field = (tau .- FT(0.5)) ./ FT(3)
    Nu = _gnf_hot_wall_nusselt(temperature, FT(N), scales.delta_T)
    return (; Nu, Temp=temperature, ux=Array(state.ux), uy=Array(state.uy),
            rho=Array(state.ρ), ν_field=nu_field, nu_field, tau_field=tau, Ra, Pr, nu0=FT(nu0), rheology,
            alpha=scales.alpha, clamped_fraction=_gnf_clamped_fraction(rheology, tau),
            steps, converged, Nu_drift)
end
