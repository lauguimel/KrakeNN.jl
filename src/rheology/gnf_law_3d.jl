# --- Scalar constitutive laws (ClassicalLaw, DenseNet, ...) in the 3D LBM GNF step ---
#
# The D3Q19 GNF collision (`CollideTRTGNFGuoField_3D`, used by
# `fused_trt_gnf_libb_guo_field_step_3d!` and `run_gnf_channel_3d`) evaluates
# `effective_viscosity(rheology, γ̇)` with γ̇ = √(2 S:S) (`strain_rate_from_pi_3d`),
# whereas `LBMLawViscosity` is defined on the invariant I1 = S:S of the 2D step
# (`gnf_viscosity(closure, I1)`). The method below converts with I1 = γ̇²/2, so
# the 3D and 2D steps evaluate the same function of S:S:
#
#   ν_lat = nu_scale * law(√(2 (gamma_scale² γ̇²/2 + eps²/2))) = nu_scale * law(√(gamma_scale² γ̇² + eps²)).
#
# With eps = 0 and unit scales, √(2 · (γ̇²/2)) == γ̇ in IEEE arithmetic (the
# halving and doubling are exact, and √(fl(γ̇²)) == |γ̇| under round to
# nearest), so a `ClassicalLaw(CarreauYasuda(...))` gives bit for bit the
# viscosity of the `CarreauYasuda` closure in the 3D step as well.

"""
    effective_viscosity(c::LBMLawViscosity, gamma_dot)

Lattice kinematic viscosity of a law closure at the shear rate
`gamma_dot = √(2 S:S)` (lattice units) of the 3D GNF step:
`gnf_viscosity(c, gamma_dot² / 2)`, i.e. the 2D convention of
[`LBMLawViscosity`](@ref) on the invariant I1 = S:S.
"""
@inline effective_viscosity(c::LBMLawViscosity, gamma_dot) =
    gnf_viscosity(c, gamma_dot * gamma_dot / 2)

"""
    _gnf3d_rheology(T, c::LBMLawViscosity)

Law closure in the population element type `T` (`gnf_convert_closure`). The
parameter vector keeps its backend; it is adapted at kernel launch
(`Adapt.adapt_structure` of [`LBMLawViscosity`](@ref)).
"""
_gnf3d_rheology(::Type{T}, c::LBMLawViscosity) where {T} = gnf_convert_closure(T, c)

"""
    lbm_law_by_value(c::LBMLawViscosity; max_params=256) -> LBMLawViscosity

The same law closure with its parameter vector held by value (an `SVector`
copied to the host once) instead of a device array (step N1 of
`.orchestrator/3d-kernel-plan.md`). Passed to the 3D GNF step, the parameters
then travel with the kernel arguments (CUDA parameter space) instead of being
read from global memory at every cell; the law evaluates the same expression
on the same values. Opt-in, never applied silently: build the closure with it
once, before the time loop. `max_params` bounds the kernel argument size
(256 Float64 parameters = 2 KB, under the 4 KB launch-parameter limit of older
CUDA setups); a larger vector throws an `ArgumentError`.
"""
function lbm_law_by_value(c::LBMLawViscosity; max_params::Integer=256)
    n = length(c.theta)
    n <= max_params ||
        throw(ArgumentError("lbm_law_by_value: $n parameters > max_params = $max_params"))
    theta = SVector{n}(Array(c.theta))
    return LBMLawViscosity(c.law, theta, c.eps, c.nu_scale, c.gamma_scale)
end
