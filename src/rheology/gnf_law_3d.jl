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
