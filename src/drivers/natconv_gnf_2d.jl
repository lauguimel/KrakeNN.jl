"""
    gnf_reference_scales(; Ra, Pr, nu0, H, T_hot, T_cold)

Compute the thermal scales of a differentially heated cavity from the
Rayleigh and Prandtl numbers, both defined with the reference kinematic
viscosity `nu0` (not with a shear rate):

```math
\\mathrm{Pr} = \\frac{\\nu_0}{\\alpha}, \\qquad
\\mathrm{Ra} = \\frac{\\beta g\\,\\Delta T\\,H^3}{\\nu_0\\,\\alpha}
```

so that `alpha = nu0/Pr` and `beta_g = Ra*nu0*alpha/((T_hot-T_cold)*H^3)`,
where `beta_g` is the product ``\\beta g`` in lattice units. These
definitions do not depend on the local constitutive viscosity. For a power-law
fluid built with [`cavity_powerlaw`](@ref), `nu0` is the nominal viscosity of
Turan et al. (2011, doi:10.1016/j.jnnfm.2011.06.003, Eq. 5), and Ra and Pr are
their Eq. 6 definitions.

Args:
    Ra: Rayleigh number based on `nu0`.
    Pr: Prandtl number based on `nu0`.
    nu0: Reference kinematic viscosity (lattice units), positive.
    H: Cavity height (lattice units), positive.
    T_hot: Hot wall temperature.
    T_cold: Cold wall temperature; must differ from `T_hot`.

Returns:
    NamedTuple `(alpha, beta_g, T_ref, delta_T)` with
    `T_ref = (T_hot + T_cold)/2` and `delta_T = T_hot - T_cold`.

Throws `ArgumentError` if `nu0`, `Pr` or `H` is not positive, or if
`T_hot == T_cold`.

Examples:
```julia
scales = gnf_reference_scales(; Ra=1e4, Pr=0.71, nu0=0.05, H=32.0,
                              T_hot=1.0, T_cold=0.0)
scales.alpha   # nu0/Pr
```
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

Construct a [`PowerLaw`](@ref) model whose viscosity equals `nu0` at the
diffusive shear rate ``\\alpha/H^2``:

```math
K = \\nu_0\\left(\\frac{\\alpha}{H^2}\\right)^{1-n}, \\qquad
\\nu(\\dot\\gamma) = K\\,\\dot\\gamma^{\\,n-1}
```

``\\alpha/H^2`` is the characteristic shear rate of Turan, Sachdeva,
Chakraborty and Poole (2011, doi:10.1016/j.jnnfm.2011.06.003, Eq. 5), so
`nu0` is their nominal viscosity and, with ``\\mathrm{Pr} = \\nu_0/\\alpha`` and
``\\mathrm{Ra} = \\beta g\\,\\Delta T\\,H^3/(\\nu_0\\alpha)``,

```math
\\mathrm{Pr} = K\\,\\alpha^{n-2} H^{2-2n}, \\qquad
\\mathrm{Ra} = \\frac{\\beta g\\,\\Delta T\\,H^{2n+1}}{K\\,\\alpha^{n}},
```

which are their Eq. 6 definitions (`K` here is the kinematic consistency, their
``K/\\rho``). Unlike the paper, `PowerLaw` clips the viscosity to
`[nu_min, nu_max]`. Compared with the paper on CPU up to N = 96.
The floating-point type of `nu0` determines the model parameter type.

Args:
    n: Power-law index (`n < 1` shear-thinning, `n > 1` shear-thickening).
    nu0: Viscosity at shear rate `alpha/H^2` (lattice units), positive.
    alpha: Thermal diffusivity (lattice units), positive; typically
        `gnf_reference_scales(...).alpha`.
    H: Cavity height (lattice units), positive.
    nu_min: Lower viscosity cutoff.
    nu_max: Upper viscosity cutoff; `0 < nu_min <= nu_max` is required.

Returns:
    A `PowerLaw(K, n; nu_min, nu_max)`.

Examples:
```julia
scales = gnf_reference_scales(; Ra=1e4, Pr=0.71, nu0=0.05, H=32.0,
                              T_hot=1.0, T_cold=0.0)
model = cavity_powerlaw(0.8; nu0=0.05, alpha=scales.alpha, H=32)
```
"""
function cavity_powerlaw(n; nu0=0.05, alpha, H, nu_min=1e-4, nu_max=1.0)
    T = typeof(float(nu0))
    nu0 > 0 && alpha > 0 && H > 0 && 0 < nu_min <= nu_max ||
        throw(ArgumentError("positive scales and ordered positive viscosity bounds required"))
    K = T(nu0) * (T(alpha) / T(H)^2)^(one(T) - T(n))
    return PowerLaw(K, T(n); nu_min=T(nu_min), nu_max=T(nu_max))
end

# Exterior-link bounce-back of the flow belongs to stream_pull_node. Every
# stored node is fluid, including the wall-adjacent nodes: no second solid-node
# bounce-back. The thermal populations stream as in fused_natconv_step!:
# stream_pull_thermal_node reflects them specularly at the adiabatic top and
# bottom walls (zero normal heat flux, tangential flux kept), and
# apply_thermal_bc_cavity is the half-way anti-bounce-back of the Dirichlet
# walls at x = 0.5 and N + 0.5, applied once per step after streaming.
@kernel function _natconv_gnf_transport!(f_out, @Const(f_in), g_out, @Const(g_in),
                                        Temp, rho, ux_field, uy_field, Fx, Fy,
                                        N, omega_T, beta_g, T_ref, T_hot, T_cold)
    i, j = @index(Global, NTuple)
    @inbounds begin
        fp = stream_pull_node(f_in, i, j, N, N)
        gp = apply_thermal_bc_cavity(stream_pull_thermal_node(g_in, i, j, N, N),
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

# Hot wall at x = 0.5 (half-way): same Nu as run_natural_convection_2d, the
# second-order one-sided gradient averaged over all rows 1:N.
_gnf_hot_wall_nusselt(Temp, H, T_hot, delta_T) =
    _hot_wall_nusselt_2d(Temp, H, T_hot, delta_T)

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
Ra and Pr use the reference viscosity `nu0`, through [`gnf_reference_scales`](@ref);
the local viscosity of `rheology` changes the flow but not the definition of
Ra and Pr. West/east temperatures are fixed; top/bottom are adiabatic (thermal
populations reflected specularly, as in [`run_natural_convection_2d`](@ref));
all walls are no-slip. Thermal advection uses the pre-collision Guo velocity
`j/rho+F/(2rho)`. Each step streams both populations, computes the
temperature from the thermal populations, applies the Boussinesq force, then
calls `collide_rheology_thermal_2d!` for the viscosity-dependent collision.

The fixed-temperature walls are the half-way anti-bounce-back of
[`run_natural_convection_2d`](@ref), at the same place as the no-slip walls, so
Ra, Pr and Nu use the cavity width `H = N`; with `n = 1` both drivers give the
same Nu to round-off. Validated on CPU in Float64 only (`test/test_natconv_gnf.jl`),
including two power-law points against Turan et al. (2011) at Pr = 10. GPU
(one A100, CUDA Float64, N = 64 to 256) results are archived in
`benchmarks/validation/natconv_gnf/results/`; the test suite itself runs on CPU only.

Args:
    N: Number of nodes per side of the square cavity (at least 3).
    Ra: Rayleigh number based on `nu0`.
    Pr: Prandtl number based on `nu0`.
    rheology: GeneralizedNewtonian model whose floating parameters, including
        those in its thermal coupling, have type `FT`.
    nu0: Reference kinematic viscosity (lattice units), also the initial
        viscosity of every node.
    T_hot: West wall temperature.
    T_cold: East wall temperature.
    max_steps: Maximum number of time steps.
    backend: KernelAbstractions backend.
    FT: Floating-point type of all fields.
    check_every: Positive interval enables Nu-drift stopping after three
        consecutive relative changes below `convergence_rtol`, after `min_steps`.
        Zero disables convergence checks and executes exactly `max_steps`.
    convergence_rtol: Relative Nu change threshold for the stopping rule.
    min_steps: Step count before which convergence is not accepted.

Returns:
    NamedTuple of CPU arrays. `ux`, `uy` and `rho` are the final step's
    pre-collision fields (initial rest fields if zero steps). `Temp` is the
    final thermal moment. `ν_field=(tau_field-0.5)/3` (also `nu_field`) is the viscosity used
    by the last collision. `clamped_fraction` counts the final PowerLaw nodes
    at either cutoff divided by all N*N fluid nodes, not time-accumulated;
    models without explicit clamps report zero. `converged` concerns Nu only.
    `Nu` is the hot-wall Nusselt number, with the definition of
    `run_natural_convection_2d` (second-order one-sided wall gradient,
    averaged over all rows `1:N`).

Examples:
```julia
using Kraken
scales = gnf_reference_scales(; Ra=1e4, Pr=0.71, nu0=0.05, H=32.0,
                              T_hot=1.0, T_cold=0.0)
model = cavity_powerlaw(0.8; nu0=0.05, alpha=scales.alpha, H=32)
result = run_natural_convection_gnf_2d(; N=32, Ra=1e4, Pr=0.71, nu0=0.05,
    rheology=model, max_steps=8000, check_every=1000,
    convergence_rtol=1e-5, min_steps=5000)
result.Nu, result.converged
```
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
    # Linear profile between the half-way walls at x = 0.5 (T_hot) and
    # x = N + 0.5 (T_cold), node i at x = i, as in run_natural_convection_2d.
    for j in 1:N, i in 1:N
        temperature = FT(T_hot - (T_hot - T_cold) * (i - 0.5) / N)
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
            current_Nu = _gnf_hot_wall_nusselt(Array(Temp), FT(N), FT(T_hot), scales.delta_T)
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
    Nu = _gnf_hot_wall_nusselt(temperature, FT(N), FT(T_hot), scales.delta_T)
    return (; Nu, Temp=temperature, ux=Array(state.ux), uy=Array(state.uy),
            rho=Array(state.ρ), ν_field=nu_field, nu_field, tau_field=tau, Ra, Pr, nu0=FT(nu0), rheology,
            alpha=scales.alpha, clamped_fraction=_gnf_clamped_fraction(rheology, tau),
            steps, converged, Nu_drift)
end
