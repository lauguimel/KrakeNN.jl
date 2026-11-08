"""
    api_extract.jl

JuliaSyntax-based extraction of exported symbols from `src/KrakeNN.jl`,
plus rule-based categorization for the per-category API pages built in
Phase 4.6.
"""

using JuliaSyntax

# ---------- export parsing ----------

"""
    extract_exports(module_file::String) -> Vector{Symbol}

Parse a Julia module file and return every symbol that appears in an
`export ...` statement. Order follows source order; duplicates removed.
"""
function extract_exports(module_file::String)
    src = read(module_file, String)
    tree = parseall(SyntaxNode, src; filename=basename(module_file))
    out = Symbol[]
    seen = Set{Symbol}()
    function walk(n)
        if kind(n) == K"export"
            ch = JuliaSyntax.children(n)
            if ch !== nothing
                for c in ch
                    s = Symbol(strip(string(c)))
                    if !(s in seen)
                        push!(out, s); push!(seen, s)
                    end
                end
            end
            return
        end
        ch = JuliaSyntax.children(n)
        ch === nothing && return
        for c in ch
            walk(c)
        end
    end
    walk(tree)
    return out
end

# ---------- categorization rules ----------

"Default category rules: name-pattern → category symbol."
const DEFAULT_RULES = [
    # Out-of-scope families (still emitted so callers can exclude)
    (s -> startswith(string(s), "run_spinodal") ||
          startswith(string(s), "compute_psi") ||
          startswith(string(s), "compute_sc_") ||
          startswith(string(s), "collide_sc_") ||
          startswith(string(s), "run_static_droplet") ||
          startswith(string(s), "run_plateau") ||
          startswith(string(s), "run_rp_") ||
          startswith(string(s), "run_cij_") ||
          startswith(string(s), "compute_vof") ||
          startswith(string(s), "advect_vof") ||
          startswith(string(s), "compute_hf_curvature") ||
          startswith(string(s), "compute_surface_tension") ||
          startswith(string(s), "collide_twophase") ||
          startswith(string(s), "set_vof") ||
          startswith(string(s), "apply_density_correction") ||
          startswith(string(s), "collide_pressure_vof") ||
          startswith(string(s), "init_pressure_vof") ||
          startswith(string(s), "smooth_vof") ||
          startswith(string(s), "correct_mass") ||
          startswith(string(s), "add_axisym_viscous") ||
          startswith(string(s), "extrapolate_velocity_ghost") ||
          startswith(string(s), "reset_feq_ghost") ||
          startswith(string(s), "phasefield_") ||
          startswith(string(s), "compute_phi_") ||
          startswith(string(s), "compute_chemical_potential") ||
          startswith(string(s), "add_azimuthal") ||
          startswith(string(s), "compute_phasefield") ||
          startswith(string(s), "compute_vof_from_phi") ||
          startswith(string(s), "compute_antidiffusion") ||
          startswith(string(s), "collide_allen_cahn") ||
          startswith(string(s), "collide_pressure_phasefield") ||
          startswith(string(s), "compute_macroscopic_phasefield") ||
          startswith(string(s), "set_phasefield") ||
          startswith(string(s), "extrapolate_phasefield") ||
          startswith(string(s), "init_phasefield") ||
          startswith(string(s), "init_pressure_equilibrium") ||
          startswith(string(s), "prolongate_bilinear") ||
          startswith(string(s), "restrict_average") ||
          startswith(string(s), "compute_hf_curvature_dx") ||
          startswith(string(s), "compute_surface_tension_dx") ||
          startswith(string(s), "clamp_field") ||
          startswith(string(s), "fill_velocity_field") ||
          startswith(string(s), "init_vof_field") ||
          startswith(string(s), "run_advection") ||
          s === :add_azimuthal_curvature_2d! ||
          s === :TwophaseRefinedArrays || s === :create_twophase_patch_arrays ||
          s === :advance_twophase_refined_step!) => :multiphase,

    (s -> s in (:AbstractRheology, :GeneralizedNewtonian, :Viscoelastic,
                :AbstractThermalCoupling, :IsothermalCoupling, :ArrheniusCoupling,
                :WLFCoupling, :Newtonian, :PowerLaw, :CarreauYasuda, :Cross,
                :Bingham, :HerschelBulkley, :OldroydB, :FENEP, :Saramito,
                :StressFormulation, :LogConfFormulation, :effective_viscosity,
                :effective_viscosity_thermal, :thermal_shift_factor,
                :strain_rate_magnitude_2d, :strain_rate_magnitude_3d,
                :collide_rheology_2d!, :collide_rheology_guo_2d!,
                :collide_rheology_thermal_2d!, :collide_twophase_rheology_2d!)) => :rheology,

    (s -> s in (:eigen_sym2x2, :mat_exp_sym2x2, :mat_log_sym2x2,
                :decompose_velocity_gradient, :compute_polymeric_force_2d!,
                :evolve_stress_2d!, :evolve_logconf_2d!,
                :compute_stress_from_conf_2d!, :compute_stress_from_logconf_2d!,
                :run_viscoelastic_cylinder_2d)) => :viscoelastic,

    (s -> startswith(string(s), "collide_species") ||
          startswith(string(s), "compute_concentration") ||
          startswith(string(s), "apply_fixed_conc")) => :species,

    # In-scope
    (s -> startswith(string(s), "collide_")) => :collision,
    (s -> startswith(string(s), "stream_")) => :streaming,
    (s -> (startswith(string(s), "apply_") &&
           (endswith(string(s), "_2d!") || endswith(string(s), "_3d!")))) => :boundary,
    (s -> startswith(string(s), "compute_macroscopic")) => :macroscopic,
    (s -> s in (:RefinementPatch, :RefinedDomain, :create_patch,
                :create_refined_domain, :rescaled_omega,
                :rescaling_factor_c2f, :rescaling_factor_f2c,
                :prolongate_f_rescaled_2d!, :restrict_f_rescaled_2d!,
                :temporal_interpolate_2d!, :copy_macroscopic_overlap_2d!,
                :advance_refined_step!, :ThermalPatchArrays,
                :create_thermal_patch_arrays, :advance_thermal_refined_step!)) => :refinement,
    (s -> startswith(string(s), "write_vtk") ||
          startswith(string(s), "write_snapshot") ||
          s in (:create_pvd, :write_vtk_to_pvd, :setup_output_dir,
                :DiagnosticsLogger, :open_diagnostics, :log_diagnostics!,
                :close_diagnostics!)) => :io,
    (s -> s in (:extract_line, :probe, :field_error, :domain_stats,
                :load_basilisk_interfaces, :load_basilisk_interface_contour,
                :find_basilisk_snapshot, :compare_interfaces)) => :postprocess,
    (s -> startswith(string(s), "run_") ||
          startswith(string(s), "initialize_taylor_green") ||
          startswith(string(s), "initialize_cylinder") ||
          s in (:compute_drag_mea_2d, :fused_natconv_step!,
                :fused_natconv_vt_step!, :fused_bgk_step!,
                :aa_even_step!, :aa_odd_step!,
                :persistent_fused_bgk!, :persistent_aa_bgk!,
                :benchmark_mlups)) => :drivers,
    (s -> s in (:AbstractLattice, :D2Q9, :D3Q19, :lattice_dim, :lattice_q,
                :weights, :velocities_x, :velocities_y, :velocities_z,
                :opposite, :cs2, :equilibrium)) => :lattice,
    (s -> s in (:LBMConfig, :omega, :reynolds, :initialize_2d, :initialize_3d)) => :config,
    (s -> s in (:KrakenExpr, :parse_kraken_expr, :evaluate, :has_variable,
                :is_time_dependent, :is_spatial, :SimulationSetup, :DomainSetup,
                :PhysicsSetup, :GeometryRegion, :BoundarySetup, :RheologySetup,
                :InitialSetup, :OutputSetup, :DiagnosticsSetup, :STLSource,
                :RefineSetup, :load_kraken, :parse_kraken, :build_rheology_model,
                :parse_kraken_sweep, :load_kraken_sweep, :sanity_check,
                :run_simulation, :STLTriangle, :STLMesh, :read_stl,
                :transform_mesh, :voxelize_2d, :voxelize_3d)) => :krk_dsl,
]

"""
    categorize_exports(exports; rules=DEFAULT_RULES) -> Dict{Symbol, Vector{Symbol}}

Group exported symbols into categories using a list of
`(predicate => category)` pairs. The first matching rule wins; symbols
that match no rule go into `:other`.
"""
function categorize_exports(exports::Vector{Symbol}; rules=DEFAULT_RULES)
    out = Dict{Symbol, Vector{Symbol}}()
    for s in exports
        cat = :other
        for (pred, c) in rules
            if pred(s)
                cat = c
                break
            end
        end
        push!(get!(out, cat, Symbol[]), s)
    end
    return out
end

# ---------- source index across src/ ----------

"""Build a `Dict{Symbol, (filepath, docstring, body, signature)}` for every
top-level function/struct discovered under `src_dir`. Used to locate the
source file of an exported symbol."""
function _build_source_index(src_dir::String)
    idx = Dict{Symbol, NamedTuple}()
    for (root, _, files) in walkdir(src_dir)
        for f in files
            endswith(f, ".jl") || continue
            path = joinpath(root, f)
            local src, tree
            try
                src = read(path, String)
                tree = parseall(SyntaxNode, src; filename=f)
            catch
                continue
            end
            _each_toplevel(tree) do name, node, docstr
                if !haskey(idx, name)
                    body = _byte_slice(src, node)
                    ds = docstr === nothing ? "" : _byte_slice(src, docstr)
                    sig = first(split(body, '\n'))
                    idx[name] = (; filepath=path, docstring=ds, body=body, signature=sig)
                end
            end
        end
    end
    return idx
end

"""
    api_page_data(module_file; exclude_categories=[:multiphase,:rheology,:viscoelastic,:species])
        -> Dict{Symbol, Vector{NamedTuple}}

For each in-scope category, return a vector of NamedTuples
`(name, signature, docstring, source_excerpt)` suitable for rendering a
per-category API reference page.

`source_excerpt` is a markdown fenced code block (may be empty when the
symbol comes from a macro/const or cannot be located on disk).
"""
function api_page_data(module_file::String;
                       exclude_categories::Vector{Symbol}=[:multiphase, :rheology, :viscoelastic, :species])
    exports = extract_exports(module_file)
    cats = categorize_exports(exports)
    src_dir = joinpath(dirname(module_file))
    index = _build_source_index(src_dir)
    out = Dict{Symbol, Vector{NamedTuple}}()
    for (cat, syms) in cats
        cat in exclude_categories && continue
        entries = NamedTuple[]
        for s in syms
            if haskey(index, s)
                e = index[s]
                excerpt = "```julia\n" * (isempty(e.docstring) ? e.body : e.docstring * "\n" * e.body) * "\n```"
                push!(entries, (; name=s, signature=e.signature, docstring=e.docstring, source_excerpt=excerpt))
            else
                push!(entries, (; name=s, signature="", docstring="", source_excerpt=""))
            end
        end
        out[cat] = entries
    end
    return out
end
