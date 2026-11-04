# --- Generic simulation runner for .krk config files ---

# `backend` defaults to this sentinel (never a real KernelAbstractions
# backend) so `run_simulation` can tell "caller passed backend=..." apart
# from "caller left it unset". Only in the latter case does a `.krk`
# `Backend <name>` directive get to pick the backend — an explicit Julia
# kwarg ALWAYS wins over the file (backward compatibility: existing calls
# that pass `backend=...` keep behaving exactly as before).
struct _UnspecifiedBackend end
const _KRK_BACKEND_UNSPECIFIED = _UnspecifiedBackend()

"""
    run_simulation(filename::String; backend=<unspecified>, T=Float64,
                   max_steps=nothing, kwargs...) -> NamedTuple

Run an LBM simulation defined by a `.krk` configuration file.
Keyword arguments override `Define` defaults for parametric studies.
Pass `max_steps` to override the `Run N steps` directive (useful for tests).

## Backend selection
If `backend` is not passed, the backend is taken from the file's `Backend
<cpu|cuda|metal>` directive (see `docs/src/krk/directives.md`); if the file
has no `Backend` directive either, it defaults to `KernelAbstractions.CPU()`.
Passing `backend=...` explicitly always overrides the file. Requesting
`cuda`/`metal` when the corresponding package is not loaded/functional
raises a clear `ArgumentError` naming the backend — it is never silently
downgraded to CPU.

Returns a NamedTuple with final fields on CPU: `(ρ, ux, uy, setup)`.
If the setup contains `Sensitivity { ... }`, returns the
`steady_shape_sensitivity` API result instead.

## Dispatch rules
The runner selects a backend driver based on `setup.modules`, `setup.lattice`,
`setup.refinements`, and the `setup.name` (case name):

1. `setup.sensitivity !== nothing` → steady AD shape sensitivity
2. `:advection_only in modules`  → pure VOF advection (no LBM solve)
3. `:twophase_vof   in modules`  → two-phase LBM with surface tension
4. `:axisymmetric   in modules`  → `run_hagen_poiseuille_2d` if the case
   name contains `hagen_poiseuille`, otherwise an informative error
5. `!isempty(setup.refinements)` → refined-grid drivers. Only refined
   natural convection is currently supported via the .krk runner;
   other refined cases raise an informative error (run them via the
   Julia API — see `create_refined_domain` / `create_thermal_patch_arrays`).
6. `:thermal in modules`         → `run_rayleigh_benard_2d`,
   `run_natural_convection_2d`, or a thermal-conduction fallback
   depending on the case name
7. `setup.lattice === :D3Q19`    → `run_cavity_3d` if the case name
   contains `cavity_3d`, otherwise an informative error
8. Default (`:D2Q9`, no modules) → generic single-phase LBM loop
   (existing behavior, compatible with all cavity/Poiseuille/Couette/
   Taylor-Green/cylinder examples).

# Example
```julia
result = run_simulation("examples/cavity.krk")
result = run_simulation("examples/cavity.krk"; Re=400, N=256)
result = run_simulation("examples/rayleigh_benard.krk"; max_steps=100)
```
"""
function run_simulation(filename::String;
                        backend=_KRK_BACKEND_UNSPECIFIED, T=Float64,
                        callback::Union{Nothing,Function}=nothing,
                        callback_every::Int=100,
                        max_steps::Union{Nothing,Int}=nothing, kwargs...)
    setup = load_kraken(filename; kwargs...)
    if max_steps !== nothing
        setup = _override_max_steps(setup, max_steps)
    end
    return run_simulation(setup; backend=backend, T=T,
                          callback=callback, callback_every=callback_every)
end

"""Find the first output spec with format `fmt`, or nothing."""
function _find_output(setup::SimulationSetup, fmt::Symbol)
    idx = findfirst(o -> o.format == fmt, setup.outputs)
    return idx === nothing ? nothing : setup.outputs[idx]
end

"""Return a copy of `setup` with `max_steps` overridden."""
function _override_max_steps(setup::SimulationSetup, new_max::Int)
    return SimulationSetup(
        setup.name, setup.lattice, setup.domain, setup.physics,
        setup.user_vars, setup.regions, setup.boundaries, setup.initial,
        setup.modules, new_max,
        setup.outputs, setup.diagnostics, setup.refinements,
        setup.velocity_field, setup.rheology, setup.mesh, setup.units,
        setup.collision, setup.wall_bc, setup.sensitivity, setup.backend)
end

_krk_selector_symbol(sym::Symbol) = Symbol(lowercase(String(sym)))

"""
    _backend_from_krk_symbol(sym::Symbol) -> KernelAbstractions.Backend

Resolve a `.krk` `Backend <name>` directive (`sym ∈ (:cpu, :cuda, :metal)`,
already validated by `_parse_backend` at parse time) to a concrete
KernelAbstractions backend object.

CUDA/Metal are declared package dependencies of Kraken but are deliberately
NOT `using`'d eagerly at module load (that stays a separate question — see
the mission notes on the CUDA/Metal hard-dependency). Instead this function
lazily loads the requested package and checks `<Package>.functional()`
on demand: if the package fails to load, or reports no usable device, it
raises a clear `ArgumentError` naming the backend. It NEVER falls back to
CPU silently — a silent fallback is exactly the failure mode this exists
to remove.
"""
function _backend_from_krk_symbol(sym::Symbol)
    sym === :cpu && return KernelAbstractions.CPU()
    sym === :cuda && return _require_functional_gpu_backend(
        :cuda, Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"),
        :CUDABackend)
    sym === :metal && return _require_functional_gpu_backend(
        :metal, Base.PkgId(Base.UUID("dde4c033-4e86-420c-a63e-0dd931031962"), "Metal"),
        :MetalBackend)
    throw(ArgumentError(
        "Unknown backend '$sym' requested via .krk 'Backend' directive. " *
        "Supported values: $(join(_KRK_KNOWN_BACKENDS, ", "))."))
end

function _require_functional_gpu_backend(name::Symbol, pkgid::Base.PkgId, backend_fn::Symbol)
    local mod
    try
        mod = Base.require(pkgid)
    catch err
        throw(ArgumentError(
            "Backend '$name' requested via .krk 'Backend' directive, but the " *
            "$(pkgid.name) package could not be loaded ($(sprint(showerror, err))). " *
            "$(pkgid.name) is already a Kraken dependency, but its runtime/driver " *
            "must be available to actually use it. Use `Backend cpu` (or pass " *
            "backend=KernelAbstractions.CPU() explicitly) instead."))
    end
    is_functional = try
        Base.invokelatest(getfield(mod, :functional))
    catch err
        throw(ArgumentError(
            "Backend '$name' requested via .krk 'Backend' directive, but calling " *
            "$(pkgid.name).functional() failed ($(sprint(showerror, err))). This " *
            "usually means no compatible $(pkgid.name) driver/runtime was found. " *
            "Use `Backend cpu` instead."))
    end
    is_functional || throw(ArgumentError(
        "Backend '$name' requested via .krk 'Backend' directive, but " *
        "$(pkgid.name).functional() returned false — no usable device was " *
        "detected on this machine. Use `Backend cpu` instead."))
    return Base.invokelatest(getfield(mod, backend_fn))
end

"""Resolve the effective backend for a run: an explicit `backend=...` kwarg
(anything but the `_KRK_BACKEND_UNSPECIFIED` sentinel) always wins; otherwise
fall back to the `.krk` file's `Backend` directive (`setup.backend`), and
finally to `KernelAbstractions.CPU()` (the historical default) if neither is
present."""
function _resolve_run_backend(setup::SimulationSetup, backend)
    backend !== _KRK_BACKEND_UNSPECIFIED && return backend
    setup.backend === nothing && return KernelAbstractions.CPU()
    return _backend_from_krk_symbol(setup.backend)
end

function _normalize_collision_selector(sym::Symbol)
    mode = _krk_selector_symbol(sym)
    mode in (:bgk, :trt) || throw(ArgumentError(
        "Unsupported Physics collision='$sym'. Supported generic 2D values: bgk, trt."))
    return mode
end

function _normalize_wall_bc_selector(sym::Symbol)
    mode = _krk_selector_symbol(sym)
    mode in (:halfwaybb, :halfway_bb, :halfway) && return :halfwaybb
    mode === :libb && return :libb
    throw(ArgumentError(
        "Unsupported Physics wall_bc='$sym'. Supported generic 2D values: halfwaybb, libb."))
end

function _setup_with_global_stl_libb(setup::SimulationSetup)
    regions = GeometryRegion[]
    sizehint!(regions, length(setup.regions))
    found_stl_obstacle = false
    for region in setup.regions
        if region.kind === :obstacle && region.stl !== nothing
            found_stl_obstacle = true
            push!(regions, GeometryRegion(region.name, region.kind, region.condition,
                                          region.stl, :libb, region.bc_values))
        else
            push!(regions, region)
        end
    end
    found_stl_obstacle || throw(ArgumentError(
        "Physics wall_bc=libb in the generic 2D path currently requires an STL Obstacle. " *
        "Analytic-obstacle LI-BB dispatch is deferred."))
    # TODO(K1-followup): add LI-BB q-wall dispatch for analytic obstacles.
    return SimulationSetup(setup.name, setup.lattice, setup.domain, setup.physics,
                           setup.user_vars, regions, setup.boundaries,
                           setup.initial, setup.modules, setup.max_steps,
                           setup.outputs, setup.diagnostics, setup.refinements,
                           setup.velocity_field, setup.rheology, setup.mesh,
                           setup.units, setup.collision, :libb,
                           setup.sensitivity, setup.backend)
end

function _ensure_generic_trt_supported!(setup::SimulationSetup, has_body_force::Bool,
                                        has_rheology::Bool)
    if has_body_force || has_rheology
        throw(ArgumentError(
            "Physics collision=trt in the generic 2D path currently supports only " *
            "Newtonian flow without body force."))
    end
    bad = findfirst(bc -> bc.type !== :wall, setup.boundaries)
    bad === nothing && return nothing
    bc = setup.boundaries[bad]
    throw(ArgumentError(
        "Physics collision=trt in the generic 2D path currently supports only " *
        "static wall boundaries; got $(bc.face) $(bc.type)."))
end

"""
    run_simulation(setup::SimulationSetup; backend=<unspecified>, T=Float64,
                   callback=nothing, callback_every=100)

Run an LBM simulation from a parsed `SimulationSetup`.

## Backend selection
If `backend` is not passed, the backend is taken from `setup.backend` (set by
a `.krk` `Backend <cpu|cuda|metal>` directive); if that is also unset, it
defaults to `KernelAbstractions.CPU()`. Passing `backend=...` explicitly
always overrides `setup.backend`. See `_backend_from_krk_symbol` for the
clear-error-on-non-functional-backend behavior.

If `callback` is provided, it is called every `callback_every` steps as
`callback(step, state)` where `state` is a NamedTuple `(; rho, ux, uy)` of
CPU arrays. Useful for live monitoring or custom post-processing.
"""
function run_simulation(setup::SimulationSetup;
                        backend=_KRK_BACKEND_UNSPECIFIED, T=Float64,
                        callback::Union{Nothing,Function}=nothing,
                        callback_every::Int=100)
    backend = _resolve_run_backend(setup, backend)

    # --- Sanity checks (tau, Mach, CFL) ---
    sanity_check(setup)

    # --- Dispatch to specialized runners based on modules ---
    if setup.sensitivity !== nothing
        T === Float64 || throw(ArgumentError(
            ".krk Sensitivity dispatch supports T=Float64 only."))
        # Function barrier (#39): once Enzyme is loaded, inferring any
        # run_simulation call would otherwise expand every shape-sensitivity
        # thunk, one of which crashes Enzyme's LLVM pipeline on x64 Linux.
        return Base.invokelatest(run_krk_sensitivity, setup)
    elseif setup.mesh !== nothing
        if :slbm_drag in setup.modules
            return _run_gmsh_slbm_drag(setup; backend=backend, T=T,
                                       callback=callback,
                                       callback_every=callback_every)
        end
        error("Mesh directive is present, but no mesh-capable runner was selected. " *
              "For Gmsh cylinder drag use `Module slbm_drag`.")
    elseif :advection_only in setup.modules
        return _run_advection_only(setup; backend=backend, T=T)
    elseif :twophase_vof in setup.modules
        return _run_twophase_vof(setup; backend=backend, T=T)
    elseif :axisymmetric in setup.modules
        return _run_axisymmetric(setup; backend=backend, T=T)
    elseif !isempty(setup.refinements)
        return _run_refined(setup; backend=backend, T=T,
                            callback=callback, callback_every=callback_every)
    elseif :thermal in setup.modules
        return _run_thermal(setup; backend=backend, T=T)
    elseif :ehd in setup.modules
        return _run_ehd(setup; backend=backend, T=T)
    elseif :viscoelastic in setup.modules
        return _run_viscoelastic(setup; backend=backend, T=T)
    elseif setup.lattice === :D3Q19
        return _run_d3q19(setup; backend=backend, T=T)
    end

    # --- Default: single-phase LBM ---
    dom = setup.domain
    Nx, Ny = dom.Nx, dom.Ny
    dx = dom.Lx / Nx
    dy = dom.Ly / Ny
    ν = setup.physics.params[:nu]
    ω = T(1.0 / (3.0 * ν + 0.5))
    collision = _normalize_collision_selector(setup.collision)
    wall_bc = _normalize_wall_bc_selector(setup.wall_bc)

    # --- Initialize state ---
    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=ν, u_lid=0.0,
                       max_steps=setup.max_steps, output_interval=1000)
    state = initialize_2d(config, T; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid

    # --- Apply geometry ---
    _apply_geometry!(is_solid, setup, dx, dy)
    libb_setup = wall_bc === :libb ? _setup_with_global_stl_libb(setup) : setup
    stl_libb = _has_stl_libb_obstacle(libb_setup)

    # --- Apply initial conditions ---
    if setup.initial !== nothing
        _apply_initial_conditions!(f_in, f_out, setup, dx, dy, T)
    end

    # --- Select streaming kernel ---
    stream_fn! = _select_streaming_kernel(setup)

    # --- Select collision kernel ---
    has_body_force = !isempty(setup.physics.body_force)
    Fx_val = T(0)
    Fy_val = T(0)
    if has_body_force
        Fx_val = haskey(setup.physics.body_force, :Fx) ?
            T(evaluate(setup.physics.body_force[:Fx])) : T(0)
        Fy_val = haskey(setup.physics.body_force, :Fy) ?
            T(evaluate(setup.physics.body_force[:Fy])) : T(0)
    end

    # --- Non-Newtonian rheology ---
    has_rheology = !isempty(setup.rheology)
    rheology_model = nothing
    tau_field = nothing
    Fx_field = nothing
    Fy_field = nothing
    if has_rheology
        rs = first(r for r in setup.rheology if r.phase in (:default, :liquid))
        rheology_model = build_rheology_model(rs; FT=T)
        tau_field = KernelAbstractions.ones(backend, T, Nx, Ny)  # initial tau = 1
        if has_body_force
            Fx_field = KernelAbstractions.zeros(backend, T, Nx, Ny)
            Fy_field = KernelAbstractions.zeros(backend, T, Nx, Ny)
            Fx_field .= Fx_val
            Fy_field .= Fy_val
        end
    end
    if stl_libb && (has_body_force || has_rheology)
        error("STL wall=libb in the generic .krk runner supports only Newtonian flow without body force")
    end
    use_trt_step = !stl_libb && collision === :trt
    use_trt_step && _ensure_generic_trt_supported!(setup, has_body_force, has_rheology)

    # --- Build boundary handlers ---
    bc_handlers = _build_boundary_handlers(setup, dx, dy, Nx, Ny, T, backend)
    libb_q_wall = nothing
    libb_uw_x = nothing
    libb_uw_y = nothing
    libb_bcspec = nothing
    if stl_libb
        q_wall_cpu = _precompute_stl_libb_q_wall_2d(Array(is_solid), libb_setup, dx, dy, T)
        libb_q_wall = _copy_to_backend(backend, T, q_wall_cpu)
        libb_uw_x = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
        libb_uw_y = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
        libb_bcspec = _build_libb_bc_rebuild_spec_2d(libb_setup, dx, dy, Nx, Ny, T, backend)
    end

    # --- Setup output ---
    vtk_out = _find_output(setup, :vtk)
    pvd = nothing
    output_dir = ""
    if vtk_out !== nothing
        output_dir = setup_output_dir(vtk_out.directory)
        pvd = create_pvd(joinpath(output_dir, setup.name))
    end

    # PNG/GIF output setup
    png_out = _find_output(setup, :png)
    gif_out = _find_output(setup, :gif)
    gif_frames = _init_gif_frames(gif_out)
    _check_image_backend(png_out, gif_out)

    # --- Time loop ---
    for step in 1:setup.max_steps
        if stl_libb
            # Fused TRT LI-BB performs pull-streaming, wall treatment, collision,
            # and moment writes. Face BCs are rebuilt from f_in afterwards.
            fused_trt_libb_v2_step!(f_out, f_in, ρ, ux, uy, is_solid,
                                     libb_q_wall, libb_uw_x, libb_uw_y,
                                     Nx, Ny, T(ν))
            apply_bc_rebuild_2d!(f_out, f_in, libb_bcspec, ν, Nx, Ny)
        elseif use_trt_step
            fused_trt_step!(f_out, f_in, ρ, ux, uy, is_solid, Nx, Ny, T(ν))
        else
            # 1. Stream
            stream_fn!(f_out, f_in, Nx, Ny)

            # 2. Apply boundary conditions
            _apply_boundary_conditions!(f_out, bc_handlers, step, Nx, Ny, dx, dy, dom, T)

            # 3. Collide
            if has_rheology && has_body_force
                collide_rheology_guo_2d!(f_out, is_solid, rheology_model, tau_field,
                                          Fx_field, Fy_field)
            elseif has_rheology
                collide_rheology_2d!(f_out, is_solid, rheology_model, tau_field)
            elseif has_body_force
                collide_guo_2d!(f_out, is_solid, ω, Fx_val, Fy_val)
            else
                collide_2d!(f_out, is_solid, ω)
            end

            # 4. Macroscopic quantities
            if has_body_force
                compute_macroscopic_forced_2d!(ρ, ux, uy, f_out, Fx_val, Fy_val)
            else
                compute_macroscopic_2d!(ρ, ux, uy, f_out)
            end
        end

        # 5. Swap
        f_in, f_out = f_out, f_in

        # 6. VTK Output
        if vtk_out !== nothing && step % vtk_out.interval == 0
            _write_output(ρ, ux, uy, setup, vtk_out, pvd, output_dir, dx, step)
        end

        # 6b. PNG/GIF snapshots
        _maybe_save_png(png_out, ρ, ux, uy, setup, output_dir, step)
        _maybe_collect_gif(gif_out, gif_frames, ρ, ux, uy, step)

        # 7. Callback (live visualization / probes)
        if callback !== nothing && step % callback_every == 0
            callback(step, (; rho=Array(ρ), ux=Array(ux), uy=Array(uy)))
        end
    end

    # Finalize PVD
    if pvd !== nothing
        vtk_save(pvd)
    end

    # Finalize GIF
    _maybe_save_gif(gif_out, gif_frames, setup, output_dir)

    # Return on CPU
    result = (ρ=Array(ρ), ux=Array(ux), uy=Array(uy), setup=setup)
    if has_rheology
        result = merge(result, (tau_field=Array(tau_field),))
    end
    return result
end

function _ve_numeric_param(setup::SimulationSetup, key::Symbol, default=nothing)
    if haskey(setup.physics.params, key)
        return Float64(setup.physics.params[key])
    elseif haskey(setup.user_vars, key)
        return Float64(setup.user_vars[key])
    elseif default !== nothing
        return Float64(default)
    end
    throw(ArgumentError(
        "viscoelastic dispatch: missing parameter `$key`. " *
        "Provide it as `Define $key = ...` or `Physics $key = ...`."))
end

function _ve_symbol_param(setup::SimulationSetup, key::Symbol, default::Symbol)
    if haskey(setup.physics.params, key) || haskey(setup.user_vars, key)
        throw(ArgumentError(
            "viscoelastic dispatch: `$key` is symbolic, but `.krk` Define/Physics " *
            "values are numeric in the current parser. Omit `$key` to use `:$default` " *
            "or call the Julia driver directly."))
    end
    return default
end

function _ve_oldroydb_rheology(setup::SimulationSetup)
    idx = findfirst(rs -> rs.model === :oldroyd_b, setup.rheology)
    idx !== nothing && return setup.rheology[idx]
    throw(ArgumentError(
        "viscoelastic dispatch: cylinder requires " *
        "`Rheology oldroyd_b { nu_s = ..., nu_p = ..., lambda = ... }`."))
end

function _ve_rheology_param(rs::RheologySetup, key::Symbol)
    haskey(rs.params, key) && return Float64(rs.params[key])
    throw(ArgumentError(
        "viscoelastic dispatch: Rheology $(rs.model) is missing `$key`."))
end

"""Locate a viscoelastic Rheology block — `oldroyd_b`, `fene_p`,
`giesekus`, or `ptt`."""
function _ve_polymer_rheology(setup::SimulationSetup)
    idx = findfirst(rs -> rs.model === :oldroyd_b || rs.model === :fene_p ||
                          rs.model === :giesekus  || rs.model === :ptt,
                    setup.rheology)
    idx !== nothing && return setup.rheology[idx]
    throw(ArgumentError(
        "viscoelastic dispatch: requires `Rheology oldroyd_b { ... }`, " *
        "`Rheology fene_p { ... Lmax2 = ... }`, " *
        "`Rheology giesekus { ... alpha = ... }`, or " *
        "`Rheology ptt { ... epsilon = ... }`."))
end

"""Build a log-conformation polymer model from a VE Rheology block.
`oldroyd_b` → `LogConfOldroydB`; `fene_p` → `LogConfFENEP` (needs `Lmax2`/`L2`);
`giesekus` → `LogConfGiesekus` (needs `alpha`/`α`); `ptt` → `LogConfPTT`
(needs `epsilon`/`eps`/`ε`, optional `variant` = 0 linear (default) / 1
exponential, since `.krk` Rheology params are numeric-only). G = nu_p / lambda
in every case."""
function _ve_build_polymer_model(rs::RheologySetup; FT=Float64)
    nu_p   = _ve_rheology_param(rs, :nu_p)
    lambda = _ve_rheology_param(rs, :lambda)
    G = FT(nu_p / lambda)
    if rs.model === :fene_p
        L2 = haskey(rs.params, :Lmax2) ? Float64(rs.params[:Lmax2]) :
             haskey(rs.params, :L2)    ? Float64(rs.params[:L2])    :
             throw(ArgumentError(
                 "viscoelastic dispatch: Rheology fene_p needs `Lmax2` (alias `L2`)."))
        return LogConfFENEP(G=G, λ=FT(lambda), Lmax2=FT(L2))
    elseif rs.model === :giesekus
        α = haskey(rs.params, :alpha) ? Float64(rs.params[:alpha]) :
            haskey(rs.params, :α)     ? Float64(rs.params[:α])     :
            throw(ArgumentError(
                "viscoelastic dispatch: Rheology giesekus needs `alpha` (alias `α`)."))
        return LogConfGiesekus(G=G, λ=FT(lambda), α=FT(α))
    elseif rs.model === :ptt
        ε = haskey(rs.params, :epsilon) ? Float64(rs.params[:epsilon]) :
            haskey(rs.params, :eps)     ? Float64(rs.params[:eps])     :
            haskey(rs.params, :ε)       ? Float64(rs.params[:ε])       :
            throw(ArgumentError(
                "viscoelastic dispatch: Rheology ptt needs `epsilon` (alias `eps`/`ε`)."))
        # `.krk` Rheology params are numeric-only, so the variant is encoded
        # as 0 = :linear (default) / 1 = :exponential.
        variant_code = haskey(rs.params, :variant) ? Float64(rs.params[:variant]) : 0.0
        variant = variant_code == 1.0 ? :exponential : :linear
        return LogConfPTT(G=G, λ=FT(lambda), ε=FT(ε), variant=variant)
    end
    return LogConfOldroydB(G=G, λ=FT(lambda))
end

function _ve_cylinder_obstacle_radius(setup::SimulationSetup)
    for region in setup.regions
        region_name = lowercase(region.name)
        (occursin("cylinder", region_name) || occursin("circle", region_name)) || continue
        for key in (:radius, :R)
            if haskey(region.bc_values, key)
                return Float64(evaluate(region.bc_values[key]))
            end
        end
    end
    return nothing
end

function _ve_sphere_obstacle_radius(setup::SimulationSetup)
    for region in setup.regions
        region_name = lowercase(region.name)
        (occursin("sphere", region_name) || occursin("sph", region_name) ||
         occursin("ball", region_name)) || continue
        for key in (:radius, :R)
            if haskey(region.bc_values, key)
                return Float64(evaluate(region.bc_values[key]))
            end
        end
    end
    return nothing
end

"""Dispatch viscoelastic cases to the existing production log-FV driver."""
function _run_viscoelastic(setup::SimulationSetup;
                           backend=KernelAbstractions.CPU(), T=Float64)
    name = lowercase(setup.name)
    if occursin("poiseuille", name) || occursin("channel", name)
        return _run_viscoelastic_fvfd_poiseuille_3d(setup; backend=backend, T=T)
    end
    if occursin("extension", name) || occursin("extensional", name)
        return _run_viscoelastic_extensional_3d(setup; backend=backend, T=T)
    end
    if setup.lattice === :D3Q19 || occursin("sphere", name)
        return _run_viscoelastic_sphere_3d(setup; backend=backend, T=T)
    end
    if !occursin("cylinder", name)
        throw(ArgumentError(
            "viscoelastic dispatch: unrecognized case name '$(setup.name)'. " *
            "Known cases: cylinder (2D), sphere (3D)."))
    end

    rs = _ve_oldroydb_rheology(setup)
    radius_from_obstacle = _ve_cylinder_obstacle_radius(setup)
    R = radius_from_obstacle === nothing ?
        _ve_numeric_param(setup, :R) : Float64(radius_from_obstacle)
    nu_s = _ve_rheology_param(rs, :nu_s)
    nu_p = _ve_rheology_param(rs, :nu_p)
    lambda = _ve_rheology_param(rs, :lambda)
    Re = _ve_numeric_param(setup, :Re, 1.0)
    nu_total = nu_s + nu_p
    u_mean = _ve_numeric_param(setup, :u_mean, nu_total * Re / R)
    L_up = _ve_numeric_param(setup, :L_up, 15.0)
    L_down = _ve_numeric_param(setup, :L_down, 15.0)
    bsd_fraction = _ve_numeric_param(setup, :bsd_fraction, 1.0)
    wall_bc = _ve_symbol_param(setup, :wall_bc, :halfwayBB)
    advection_scheme = _ve_symbol_param(setup, :advection_scheme, :muscl_superbee)
    avg_window = round(Int, 0.2 * setup.max_steps)

    result = run_viscoelastic_logfv_cylinder_coupled_2d(;
        radius=R,
        L_up=L_up,
        L_down=L_down,
        nu_s=nu_s,
        nu_p=nu_p,
        lambda=lambda,
        u_mean=u_mean,
        bsd_fraction=bsd_fraction,
        polymer_model=:oldroydb,
        wall_bc=wall_bc,
        advection_scheme=advection_scheme,
        max_steps=setup.max_steps,
        avg_window=avg_window,
        backend=backend,
        T=T,
    )
    return merge(result, (setup=setup,))
end

"""Inlet centreline/plug velocity for the VE sphere: explicit `Define u_in`
(or `u_mean`), else the west velocity BC `ux`, else error."""
function _ve_sphere_inlet_velocity(setup::SimulationSetup)
    for key in (:u_in, :u_mean)
        if haskey(setup.physics.params, key) || haskey(setup.user_vars, key)
            return _ve_numeric_param(setup, key)
        end
    end
    for b in setup.boundaries
        if b.type === :velocity && haskey(b.values, :ux)
            return Float64(evaluate(b.values[:ux]))
        end
    end
    throw(ArgumentError(
        "viscoelastic sphere dispatch: missing inlet velocity. Provide " *
        "`Define u_in = ...` or a `Boundary west velocity(ux = ...)`."))
end

"""Dispatch a 3D viscoelastic Oldroyd-B confined-sphere `.krk` case to the
`run_conformation_sphere_libb_3d` driver. Geometry is rebuilt by the driver
from `Nx/Ny/Nz`, `radius`, and (optional) `cx/cy/cz`; the `.krk` Obstacle
supplies the radius, the Domain the grid, and the Rheology block the
solvent/polymer viscosities and relaxation time."""
function _run_viscoelastic_sphere_3d(setup::SimulationSetup;
                                     backend=KernelAbstractions.CPU(), T=Float64)
    rs = _ve_oldroydb_rheology(setup)
    nu_s   = _ve_rheology_param(rs, :nu_s)
    nu_p   = _ve_rheology_param(rs, :nu_p)
    lambda = _ve_rheology_param(rs, :lambda)

    radius_from_obstacle = _ve_sphere_obstacle_radius(setup)
    R = radius_from_obstacle === nothing ?
        _ve_numeric_param(setup, :R) : Float64(radius_from_obstacle)

    dom = setup.domain
    u_in = _ve_sphere_inlet_velocity(setup)

    cx = haskey(setup.physics.params, :cx) || haskey(setup.user_vars, :cx) ?
         _ve_numeric_param(setup, :cx) : nothing
    cy = haskey(setup.physics.params, :cy) || haskey(setup.user_vars, :cy) ?
         _ve_numeric_param(setup, :cy) : nothing
    cz = haskey(setup.physics.params, :cz) || haskey(setup.user_vars, :cz) ?
         _ve_numeric_param(setup, :cz) : nothing

    tau_plus = _ve_numeric_param(setup, :tau_plus, 1.0)
    inlet    = _ve_symbol_param(setup, :inlet, :parabolic_y)
    avg_window = round(Int, 0.2 * setup.max_steps)

    result = run_conformation_sphere_libb_3d(;
        Nx=dom.Nx, Ny=dom.Ny, Nz=dom.Nz,
        radius=R, cx=cx, cy=cy, cz=cz,
        u_in=u_in, ν_s=nu_s, ν_p=nu_p, lambda=lambda,
        inlet=inlet, tau_plus=tau_plus,
        max_steps=setup.max_steps, avg_window=avg_window,
        backend=backend, FT=T,
    )
    return merge(result, (setup=setup,))
end

"""Dispatch a 3D viscoelastic planar-extension `.krk` canary to the
`run_viscoelastic_fvfd_extensional_3d` driver in `velocity_mode=:imposed`. Accepts
both `Rheology oldroyd_b { nu_s nu_p lambda }` (Oldroyd-B) and
`Rheology fene_p { nu_s nu_p lambda Lmax2 }` (FENE-P, finite extensibility). The
driver imposes `u = (epsilon_dot*x, -epsilon_dot*y, 0)` analytically (no
obstacle/inflow geometry); the `.krk` Domain supplies `Nx/Ny/Nz`, the Rheology
block the solvent/polymer viscosities and relaxation time, and `epsilon_dot`
comes from a `Define`/`Physics` entry. At the Oldroyd-B fixed point
`C_xx = 1/(1 - 2λε̇)`, `C_yy = 1/(1 + 2λε̇)`; FENE-P caps trC below `L²`."""
function _run_viscoelastic_extensional_3d(setup::SimulationSetup;
                                          backend=KernelAbstractions.CPU(), T=Float64)
    rs = _ve_polymer_rheology(setup)
    nu_s = _ve_rheology_param(rs, :nu_s)
    polymer_model = _ve_build_polymer_model(rs; FT=T)

    epsilon_dot = _ve_numeric_param(setup, :epsilon_dot)

    dom = setup.domain
    advection_scheme = _ve_symbol_param(setup, :advection_scheme, :muscl_superbee)
    velocity_mode = _ve_symbol_param(setup, :velocity_mode, :imposed)

    result = run_viscoelastic_fvfd_extensional_3d(;
        Nx=dom.Nx, Ny=dom.Ny, Nz=dom.Nz,
        epsilon_dot=epsilon_dot,
        ν_s=nu_s, ν_p=nothing, lambda=polymer_relaxation_time(polymer_model),
        polymer_model=polymer_model,
        advection_scheme=advection_scheme,
        velocity_mode=velocity_mode,
        max_steps=setup.max_steps,
        backend=backend, FT=T,
    )
    return merge(result, (setup=setup,))
end

"""Dispatch a 3D viscoelastic planar-Poiseuille `.krk` case to the FVFD
log-conformation driver `run_viscoelastic_fvfd_poiseuille_3d`. Accepts both
`Rheology oldroyd_b { nu_s nu_p lambda }` (Oldroyd-B) and
`Rheology fene_p { nu_s nu_p lambda Lmax2 }` (FENE-P, finite extensibility).
The Domain supplies `Nx/Ny/Nz`; the constant body force `Fx` comes from a
`Define`/`Physics` entry (default 1e-5)."""
function _run_viscoelastic_fvfd_poiseuille_3d(setup::SimulationSetup;
                                              backend=KernelAbstractions.CPU(), T=Float64)
    rs = _ve_polymer_rheology(setup)
    nu_s = _ve_rheology_param(rs, :nu_s)
    polymer_model = _ve_build_polymer_model(rs; FT=T)

    dom = setup.domain
    Fx = _ve_numeric_param(setup, :Fx, 1e-5)
    advection_scheme = _ve_symbol_param(setup, :advection_scheme, :muscl_superbee)

    result = run_viscoelastic_fvfd_poiseuille_3d(;
        Nx=dom.Nx, Ny=dom.Ny, Nz=dom.Nz,
        Fx=Fx, ν_s=nu_s, ν_p=nothing, lambda=polymer_relaxation_time(polymer_model),
        polymer_model=polymer_model,
        advection_scheme=advection_scheme,
        max_steps=setup.max_steps,
        backend=backend, FT=T,
    )
    return merge(result, (setup=setup,))
end

# --- Gmsh multi-block SLBM drag runner ---

function _setup_number(setup::SimulationSetup, keys, default)
    key_tuple = keys isa Tuple ? keys : (keys,)
    for key in key_tuple
        haskey(setup.physics.params, key) && return setup.physics.params[key]
        haskey(setup.user_vars, key) && return setup.user_vars[key]
    end
    return default
end

function _copy_to_backend(backend, ::Type{T}, host::AbstractArray) where T
    dev = KernelAbstractions.allocate(backend, T, size(host)...)
    copyto!(dev, T.(host))
    return dev
end

function _copy_bool_to_backend(backend, host::AbstractArray{Bool})
    dev = KernelAbstractions.allocate(backend, Bool, size(host)...)
    copyto!(dev, host)
    return dev
end

function _allocate_block_state_as(block::Block, ::Type{T}, backend, ng::Int) where T
    nx = block.mesh.Nξ + 2 * ng
    ny = block.mesh.Nη + 2 * ng
    f = KernelAbstractions.allocate(backend, T, nx, ny, 9); fill!(f, zero(T))
    rho = KernelAbstractions.allocate(backend, T, nx, ny); fill!(rho, one(T))
    ux = KernelAbstractions.allocate(backend, T, nx, ny); fill!(ux, zero(T))
    uy = KernelAbstractions.allocate(backend, T, nx, ny); fill!(uy, zero(T))
    return BlockState2D{T, typeof(f), typeof(rho)}(f, rho, ux, uy,
                                                    block.mesh.Nξ, block.mesh.Nη, ng)
end

@inline function _edge_node(block::Block, edge::Symbol, r::Int)
    edge === :west  && return 1, r
    edge === :east  && return block.mesh.Nξ, r
    edge === :south && return r, 1
    edge === :north && return r, block.mesh.Nη
    error("unknown edge $edge")
end

function _parabolic_channel_u(y, Ly, u_max)
    yy = clamp(Float64(y), 0.0, Float64(Ly))
    return 4.0 * Float64(u_max) * yy * (Float64(Ly) - yy) / Float64(Ly)^2
end


"""Apply initial conditions from expressions."""
function _apply_initial_conditions!(f_in, f_out, setup::SimulationSetup,
                                    dx, dy, ::Type{T}) where T
    Nx, Ny = setup.domain.Nx, setup.domain.Ny
    Lx, Ly = setup.domain.Lx, setup.domain.Ly

    ic = setup.initial
    w = weights(D2Q9())
    f_cpu = zeros(T, Nx, Ny, 9)

    for j in 1:Ny, i in 1:Nx
        x = (i - 0.5) * dx
        y = (j - 0.5) * dy
        kw = (; x=x, y=y, Lx=Lx, Ly=Ly, Nx=Float64(Nx), Ny=Float64(Ny), dx=dx, dy=dy)

        ρ_val = haskey(ic.fields, :rho) ? T(evaluate(ic.fields[:rho]; kw...)) : one(T)
        ux_val = haskey(ic.fields, :ux) ? T(evaluate(ic.fields[:ux]; kw...)) : zero(T)
        uy_val = haskey(ic.fields, :uy) ? T(evaluate(ic.fields[:uy]; kw...)) : zero(T)

        for q in 1:9
            f_cpu[i, j, q] = equilibrium(D2Q9(), ρ_val, ux_val, uy_val, q)
        end
    end

    copyto!(f_in, f_cpu)
    copyto!(f_out, f_cpu)
end
