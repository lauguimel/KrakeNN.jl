# Production-facing adaptation helpers for conservative-tree D2Q9 AMR.
#
# This layer keeps adaptation decisions separate from transport. It turns
# indicators or DSL Refine blocks into bounded patch plans, then leaves the
# actual conservative population transfer to the existing direct regrid path.

"""
    ConservativeTreeAdaptationPolicy2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreeAdaptationPolicy2D
```
"""
struct ConservativeTreeAdaptationPolicy2D
    pad_parent::Int
    min_i_cells::Int
    min_j_cells::Int
    max_growth::Int
    shrink_margin::Int
end

function ConservativeTreeAdaptationPolicy2D(;
        pad_parent::Integer=0,
        min_i_cells::Integer=1,
        min_j_cells::Integer=1,
        max_growth::Integer=typemax(Int),
        shrink_margin::Integer=1)
    pad = Int(pad_parent)
    min_i = Int(min_i_cells)
    min_j = Int(min_j_cells)
    growth = Int(max_growth)
    shrink = Int(shrink_margin)
    pad >= 0 || throw(ArgumentError("pad_parent must be nonnegative"))
    min_i > 0 || throw(ArgumentError("min_i_cells must be positive"))
    min_j > 0 || throw(ArgumentError("min_j_cells must be positive"))
    growth >= 0 || throw(ArgumentError("max_growth must be nonnegative"))
    shrink >= 0 || throw(ArgumentError("shrink_margin must be nonnegative"))
    return ConservativeTreeAdaptationPolicy2D(pad, min_i, min_j, growth, shrink)
end

"""
    ConservativeTreePatchProposal2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreePatchProposal2D
```
"""
struct ConservativeTreePatchProposal2D
    name::String
    i_range::UnitRange{Int}
    j_range::UnitRange{Int}
    reason::Symbol
end

function ConservativeTreePatchProposal2D(i_range::AbstractUnitRange{<:Integer},
                                         j_range::AbstractUnitRange{<:Integer};
                                         name::AbstractString="",
                                         reason::Symbol=:manual)
    i = _normalize_parent_range_2d(i_range, "i_range")
    j = _normalize_parent_range_2d(j_range, "j_range")
    return ConservativeTreePatchProposal2D(String(name), i, j, reason)
end

"""
    ConservativeTreeAdaptationPlan2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreeAdaptationPlan2D
```
"""
struct ConservativeTreeAdaptationPlan2D
    current_i_range::UnitRange{Int}
    current_j_range::UnitRange{Int}
    requested_i_range::UnitRange{Int}
    requested_j_range::UnitRange{Int}
    i_range::UnitRange{Int}
    j_range::UnitRange{Int}
    reason::Symbol
    changed::Bool
end

function _normalize_parent_range_2d(range::AbstractUnitRange{<:Integer},
                                    name::AbstractString)
    isempty(range) && throw(ArgumentError("$name must be nonempty"))
    return Int(first(range)):Int(last(range))
end

function _check_range_inside_domain_2d(range::UnitRange{Int},
                                       upper::Int,
                                       name::AbstractString)
    first(range) >= 1 && last(range) <= upper ||
        throw(ArgumentError("$name must be inside 1:$upper"))
    return nothing
end

function _pad_and_clamp_range_2d(range::UnitRange{Int}, pad::Int, upper::Int)
    lo = max(1, first(range) - pad)
    hi = min(upper, last(range) + pad)
    lo <= hi || throw(ArgumentError("range does not overlap the domain"))
    return lo:hi
end

function _expand_range_to_min_cells_2d(range::UnitRange{Int},
                                       min_cells::Int,
                                       upper::Int)
    length(range) >= min_cells && return range
    target_width = min(min_cells, upper)
    deficit = target_width - length(range)
    left = deficit ÷ 2
    right = deficit - left
    lo = first(range) - left
    hi = last(range) + right
    if lo < 1
        hi += 1 - lo
        lo = 1
    end
    if hi > upper
        lo -= hi - upper
        hi = upper
    end
    lo = max(1, lo)
    return lo:hi
end

function _limit_growth_range_2d(current::UnitRange{Int},
                                requested::UnitRange{Int},
                                max_growth::Int)
    max_growth == typemax(Int) && return requested
    lo = first(requested)
    hi = last(requested)
    if lo < first(current)
        lo = max(lo, first(current) - max_growth)
    end
    if hi > last(current)
        hi = min(hi, last(current) + max_growth)
    end
    lo <= hi || throw(ArgumentError("max_growth produced an empty patch range"))
    return lo:hi
end

"""
    conservative_tree_adaptation_plan_2d(Nx, Ny, current_i, current_j,
                                         requested_i, requested_j; policy, reason)

Build a bounded adaptation plan for a single ratio-2 conservative-tree patch.
The requested range is padded, clamped to the domain, expanded to minimum patch
size, growth-limited per side, then passed through range hysteresis.
"""
function conservative_tree_adaptation_plan_2d(
        Nx::Integer,
        Ny::Integer,
        current_i_range::AbstractUnitRange{<:Integer},
        current_j_range::AbstractUnitRange{<:Integer},
        requested_i_range::AbstractUnitRange{<:Integer},
        requested_j_range::AbstractUnitRange{<:Integer};
        policy::ConservativeTreeAdaptationPolicy2D=ConservativeTreeAdaptationPolicy2D(),
        reason::Symbol=:manual)
    nx = Int(Nx)
    ny = Int(Ny)
    _check_multipatch_domain_2d(nx, ny)
    current_i = _normalize_parent_range_2d(current_i_range, "current_i_range")
    current_j = _normalize_parent_range_2d(current_j_range, "current_j_range")
    requested_i = _normalize_parent_range_2d(requested_i_range, "requested_i_range")
    requested_j = _normalize_parent_range_2d(requested_j_range, "requested_j_range")
    _check_range_inside_domain_2d(current_i, nx, "current_i_range")
    _check_range_inside_domain_2d(current_j, ny, "current_j_range")

    i_range = _pad_and_clamp_range_2d(requested_i, policy.pad_parent, nx)
    j_range = _pad_and_clamp_range_2d(requested_j, policy.pad_parent, ny)
    i_range = _expand_range_to_min_cells_2d(i_range, policy.min_i_cells, nx)
    j_range = _expand_range_to_min_cells_2d(j_range, policy.min_j_cells, ny)
    i_range = _limit_growth_range_2d(current_i, i_range, policy.max_growth)
    j_range = _limit_growth_range_2d(current_j, j_range, policy.max_growth)

    selected = conservative_tree_hysteresis_patch_range_2d(
        current_i, current_j, i_range, j_range;
        shrink_margin=policy.shrink_margin)
    changed = selected.i_range != current_i || selected.j_range != current_j
    return ConservativeTreeAdaptationPlan2D(
        current_i, current_j, requested_i, requested_j,
        selected.i_range, selected.j_range, reason, changed)
end

"""
    conservative_tree_adaptation_plan_from_proposal_2d(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_adaptation_plan_from_proposal_2d)
```
"""
function conservative_tree_adaptation_plan_from_proposal_2d(
        Nx::Integer,
        Ny::Integer,
        current_i_range::AbstractUnitRange{<:Integer},
        current_j_range::AbstractUnitRange{<:Integer},
        proposal::ConservativeTreePatchProposal2D;
        policy::ConservativeTreeAdaptationPolicy2D=ConservativeTreeAdaptationPolicy2D())
    return conservative_tree_adaptation_plan_2d(
        Nx, Ny, current_i_range, current_j_range,
        proposal.i_range, proposal.j_range;
        policy=policy, reason=proposal.reason)
end

"""
    conservative_tree_indicator_adaptation_plan_2d(indicator, current_i,
                                                   current_j; threshold, policy)

Select a parent-grid patch from a scalar indicator and turn it into a bounded
adaptation plan. Padding and hysteresis are controlled only by `policy`.
"""
function conservative_tree_indicator_adaptation_plan_2d(
        indicator::AbstractArray{<:Real,2},
        current_i_range::AbstractUnitRange{<:Integer},
        current_j_range::AbstractUnitRange{<:Integer};
        threshold::Real,
        policy::ConservativeTreeAdaptationPolicy2D=ConservativeTreeAdaptationPolicy2D(),
        reason::Symbol=:indicator)
    requested = conservative_tree_indicator_patch_range_2d(
        indicator; threshold=threshold, pad=0)
    return conservative_tree_adaptation_plan_2d(
        size(indicator, 1), size(indicator, 2),
        current_i_range, current_j_range,
        requested.i_range, requested.j_range;
        policy=policy, reason=reason)
end

"""
    conservative_tree_patch_proposals_from_krk_2d(setup)

Convert parsed `.krk` `Refine` blocks to named parent-grid patch proposals.
The same conservative-tree guardrails as the patch-set helper apply: D2Q9,
2D base-grid `Refine`, and `ratio = 2`.
"""
function conservative_tree_patch_proposals_from_krk_2d(setup)
    getproperty(setup, :lattice) == :D2Q9 ||
        throw(ArgumentError("conservative-tree 2D patch proposals require D2Q9"))
    domain = getproperty(setup, :domain)
    refinements = getproperty(setup, :refinements)
    ranges = conservative_tree_patch_ranges_from_krk_refines_2d(domain, refinements)
    proposals = ConservativeTreePatchProposal2D[]
    for (ref, range_pair) in zip(refinements, ranges)
        push!(proposals, ConservativeTreePatchProposal2D(
            range_pair[1], range_pair[2];
            name=getproperty(ref, :name), reason=:krk_refine))
    end
    return proposals
end

"""
    conservative_tree_adaptation_policy_from_krk_refine_2d(refine)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_adaptation_policy_from_krk_refine_2d)
```
"""
function conservative_tree_adaptation_policy_from_krk_refine_2d(refine)
    criterion = getproperty(refine, :criterion)
    criterion === nothing && return ConservativeTreeAdaptationPolicy2D()
    return ConservativeTreeAdaptationPolicy2D(
        pad_parent=getproperty(criterion, :pad),
        max_growth=getproperty(criterion, :max_growth),
        shrink_margin=getproperty(criterion, :shrink_margin))
end

"""
    conservative_tree_indicator_adaptation_plan_from_krk_2d(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_indicator_adaptation_plan_from_krk_2d)
```
"""
function conservative_tree_indicator_adaptation_plan_from_krk_2d(
        indicator::AbstractArray{<:Real,2},
        current_i_range::AbstractUnitRange{<:Integer},
        current_j_range::AbstractUnitRange{<:Integer},
        refine)
    criterion = getproperty(refine, :criterion)
    criterion !== nothing ||
        throw(ArgumentError("Refine block has no adaptive criterion"))
    getproperty(criterion, :indicator) == :gradient ||
        throw(ArgumentError("Only gradient Refine criteria are supported"))
    return conservative_tree_indicator_adaptation_plan_2d(
        indicator, current_i_range, current_j_range;
        threshold=getproperty(criterion, :threshold),
        policy=conservative_tree_adaptation_policy_from_krk_refine_2d(refine),
        reason=:krk_refine_criterion)
end

"""
    adapt_conservative_tree_patch_with_plan_2d(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.adapt_conservative_tree_patch_with_plan_2d)
```
"""
function adapt_conservative_tree_patch_with_plan_2d(
        coarse_F::AbstractArray{T,3},
        patch::ConservativeTreePatch2D{T},
        plan::ConservativeTreeAdaptationPlan2D) where T
    _check_composite_coarse_layout(coarse_F, patch)
    patch.parent_i_range == plan.current_i_range &&
        patch.parent_j_range == plan.current_j_range ||
        throw(ArgumentError("plan current range does not match patch"))

    if !plan.changed
        return (coarse_F=coarse_F, patch=patch, changed=false)
    end

    patch_out = create_conservative_tree_patch_2d(
        plan.i_range, plan.j_range; T=T)
    coarse_out = similar(coarse_F)
    regrid_conservative_tree_patch_direct_F_2d!(
        coarse_out, patch_out, coarse_F, patch)
    return (coarse_F=coarse_out, patch=patch_out, changed=true)
end
