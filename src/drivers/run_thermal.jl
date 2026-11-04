
"""Dispatch thermal cases (non-refined) to the appropriate thermal driver."""
function _run_thermal(setup::SimulationSetup;
                      backend=KernelAbstractions.CPU(), T=Float64)
    name   = lowercase(setup.name)
    dom    = setup.domain
    params = setup.physics.params
    Ra = Float64(get(params, :Ra, 1e4))
    Pr = Float64(get(params, :Pr, 0.71))

    if occursin("rayleigh_benard", name) || occursin("rayleigh-benard", name)
        result = run_rayleigh_benard_2d(; Nx=dom.Nx, Ny=dom.Ny, Ra=Ra, Pr=Pr,
                                         max_steps=setup.max_steps,
                                         backend=backend, FT=T)
        return merge(result, (setup=setup,))
    elseif occursin("natural_convection", name)
        entry = findfirst(rs -> rs.phase in (:default, :liquid), setup.rheology)
        if entry !== nothing && setup.rheology[entry].model !== :newtonian
            rheology = build_rheology_model(setup.rheology[entry]; FT=T)
            result = run_natural_convection_gnf_2d(; N=dom.Nx, Ra, Pr, rheology,
                nu0=get(params, :nu, 0.05), max_steps=setup.max_steps, backend, FT=T)
        else
            result = run_natural_convection_2d(; N=dom.Nx, Ra=Ra, Pr=Pr,
                                            max_steps=setup.max_steps,
                                            backend=backend, FT=T)
        end
        return merge(result, (setup=setup,))
    elseif occursin("heat_conduction", name) || occursin("conduction", name)
        # No dedicated conduction driver: run the Boussinesq driver with Ra≈0
        # so buoyancy is negligible and diffusion dominates. Documented
        # as a pragmatic fallback — the resulting temperature field
        # matches a 1D diffusive profile once steady state is reached.
        # The supplied material parameters and thermal faces are forwarded;
        # a configuration the fallback cannot represent raises instead of
        # being silently replaced by the driver defaults.
        orientation, T_hot, T_cold = _thermal_conduction_orientation(setup)
        nu = get(params, :nu, nothing)
        alpha = get(params, :alpha, nothing)
        result = run_rayleigh_benard_2d(; Nx=dom.Nx, Ny=dom.Ny,
                                         Ra=1e-8, Pr=Pr,
                                         T_hot=T_hot, T_cold=T_cold,
                                         nu=nu, alpha=alpha,
                                         orientation=orientation,
                                         max_steps=setup.max_steps,
                                         backend=backend, FT=T)
        return merge(result, (setup=setup,))
    else
        throw(ArgumentError(
            "thermal dispatch: unrecognized case name '$(setup.name)'. " *
            "Known cases: rayleigh_benard, natural_convection, heat_conduction."))
    end
end

"""
    _thermal_face_temperatures(setup) -> Dict{Symbol,Float64}

Collect the `T = ...` values attached to the boundary faces of a `.krk` setup.
Faces without a temperature are absent from the dictionary.
"""
function _thermal_face_temperatures(setup::SimulationSetup)
    temps = Dict{Symbol, Float64}()
    for bc in setup.boundaries
        haskey(bc.values, :T) || continue
        expr = bc.values[:T]
        is_spatial(expr) && throw(ArgumentError(
            "thermal dispatch: spatially varying wall temperature on face " *
            ":$(bc.face) is not supported by the conduction fallback."))
        is_time_dependent(expr) && throw(ArgumentError(
            "thermal dispatch: time-dependent wall temperature on face " *
            ":$(bc.face) is not supported by the conduction fallback."))
        temps[bc.face] = Float64(evaluate(expr))
    end
    return temps
end

"""
    _thermal_conduction_orientation(setup) -> (orientation, T_hot, T_cold)

Map the thermal faces of a conduction `.krk` case onto what the Boussinesq
fallback driver can actually run: one opposing pair of fixed-temperature
faces, either west/east (`:horizontal`) or south/north (`:vertical`).

Anything else — a single heated face, two adjacent faces, both pairs at once,
or an equal pair — is rejected with an explicit message. The fallback has no
way to represent those, and substituting a default would run a different
problem than the one described in the file.
"""
function _thermal_conduction_orientation(setup::SimulationSetup)
    temps = _thermal_face_temperatures(setup)

    has_we = haskey(temps, :west) || haskey(temps, :east)
    has_sn = haskey(temps, :south) || haskey(temps, :north)

    if !has_we && !has_sn
        # No thermal face given: keep the historical defaults.
        return (:vertical, 1.0, 0.0)
    end

    if has_we && has_sn
        throw(ArgumentError(
            "thermal dispatch: conduction case '$(setup.name)' fixes a temperature " *
            "on both the west/east and the south/north pair " *
            "($(join(sort!(collect(String.(keys(temps)))), ", "))). " *
            "The conduction fallback imposes exactly one opposing pair; " *
            "leave the other pair without `T = ...` (adiabatic)."))
    end

    pair = has_we ? (:west, :east) : (:south, :north)
    orientation = has_we ? :horizontal : :vertical
    hot_face, cold_face = pair

    for face in pair
        haskey(temps, face) || throw(ArgumentError(
            "thermal dispatch: conduction case '$(setup.name)' fixes a temperature " *
            "on :$(face === hot_face ? cold_face : hot_face) but not on :$(face). " *
            "The conduction fallback needs both faces of one opposing pair; " *
            "a single heated face with an adiabatic opposite is not implemented."))
    end

    T_hot = temps[hot_face]
    T_cold = temps[cold_face]
    T_hot == T_cold && throw(ArgumentError(
        "thermal dispatch: conduction case '$(setup.name)' imposes the same " *
        "temperature $(T_hot) on :$(hot_face) and :$(cold_face). " *
        "The Boussinesq fallback scales buoyancy by the temperature difference, " *
        "which is then undefined."))

    return (orientation, T_hot, T_cold)
end
