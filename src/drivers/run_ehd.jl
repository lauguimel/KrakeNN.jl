"""Dispatch non-refined EHD `.krk` cases to the appropriate EHD driver."""
function _run_ehd(setup::SimulationSetup;
                  backend=KernelAbstractions.CPU(), T=Float64)
    name = lowercase(setup.name)
    dom = setup.domain
    params = setup.physics.params

    C = Float64(get(params, :C, 10.0))
    M = Float64(get(params, :M, 10.0))
    Ma_E = Float64(get(params, :Ma_E, 1e-2))
    alpha = Float64(get(params, :alpha, 1e-4))
    delta_U = Float64(get(params, :delta_U, 1.0))
    phi_scheme = _ehd_symbol_param(params, :phi_scheme, :lbm)
    injector_closure = _ehd_symbol_param(params, :injector_closure, :nee)

    if name == "ehd_hydrostatic" || startswith(name, "ehd_hydrostatic_") ||
       name == "hydrostatic" || startswith(name, "hydrostatic_")
        charge_scheme = _ehd_symbol_param(params, :charge_scheme, :srt)
        result = run_ehd_hydrostatic_2d(; Nx=dom.Nx, Ny=dom.Ny,
                                         C=C, M=M, Ma_E=Ma_E, alpha=alpha,
                                         delta_U=delta_U,
                                         charge_scheme=charge_scheme,
                                         phi_scheme=phi_scheme,
                                         injector_closure=injector_closure,
                                         max_steps=setup.max_steps,
                                         backend=backend, FT=T)
        return merge(result, (setup=setup,))
    elseif name == "electroconvection" || startswith(name, "electroconvection_")
        sidewall_bc = _ehd_ec_sidewall_bc(setup.boundaries)
        T_ehd = Float64(get(params, :T, 175.0))
        charge_scheme = _ehd_symbol_param(params, :charge_scheme, :regularized)
        ns_scheme = _ehd_symbol_param(params, :ns_scheme, :bgk)
        force_projection = _ehd_symbol_param(params, :force_projection, :none)
        result = run_electroconvection_2d(; Nx=dom.Nx, Ny=dom.Ny,
                                           C=C, M=M, T=T_ehd, Ma_E=Ma_E,
                                           alpha=alpha, delta_U=delta_U,
                                           charge_scheme=charge_scheme,
                                           phi_scheme=phi_scheme,
                                           injector_closure=injector_closure,
                                           ns_scheme=ns_scheme,
                                           sidewall_bc=sidewall_bc,
                                           force_projection=force_projection,
                                           max_cycles=setup.max_steps,
                                           backend=backend, FT=T)
        return merge(result, (setup=setup,))
    else
        throw(ArgumentError(
            "EHD dispatch: unrecognized case name '$(setup.name)'. " *
            "Known cases: ehd_hydrostatic, electroconvection."))
    end
end

"""Lower existing lateral flow declarations; absent declarations keep free slip.

Both `west` and `east` must be declared together, as either `wall` (stationary
no slip) or `symmetry` (impermeable free slip), without parameters. Explicit
declarations on other faces are rejected: EC electrode/plate conditions are
built into the driver and cannot be selected through `Boundary` declarations.
Asymmetric, repeated, parameterised or
unsupported lateral declarations are rejected instead of being ignored.
"""
function _ehd_ec_sidewall_bc(boundaries)
    all(b -> b.face in (:west, :east), boundaries) ||
        throw(ArgumentError("Electroconvection Boundary declarations support only west/east; electrode/plate conditions are built in."))
    sides = filter(b -> b.face in (:west, :east), boundaries)
    isempty(sides) && return :free_slip
    length(sides) == 2 && count(b -> b.face === :west, sides) == 1 &&
        count(b -> b.face === :east, sides) == 1 ||
        throw(ArgumentError("Electroconvection requires one west and one east boundary, or neither."))
    all(b -> isempty(b.values), sides) ||
        throw(ArgumentError("Electroconvection sidewalls do not support boundary parameters."))
    all(b -> b.type === :wall, sides) && return :no_slip
    all(b -> b.type === :symmetry, sides) && return :free_slip
    throw(ArgumentError("Electroconvection sidewalls must both be wall or both be symmetry."))
end

function _ehd_symbol_param(params::Dict{Symbol,Any}, key::Symbol, default::Symbol)
    value = get(params, key, default)
    value isa Symbol && return value
    return Symbol(value)
end
