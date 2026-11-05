# CPU inference guard for the actual KernelAbstractions inner functions.
# This catches boxed closure captures and unresolved calls before a GPU job;
# it is not a replacement for CUDA compilation/parity/allocation tests.
using Test
using Kraken
using KernelAbstractions
using ForwardDiff

_fvc4_concrete(t::Core.Const) = true
_fvc4_concrete(t::Core.PartialStruct) = _fvc4_concrete(t.typ)
_fvc4_concrete(t) = t === Union{} || (t isa Type &&
    (isconcretetype(t) || (t isa Union && all(_fvc4_concrete, Base.uniontypes(t)))))

function _fvc4_has_box(x)
    x === Core.Box && return true
    x isa Core.Const && return _fvc4_has_box(x.val)
    x isa Core.PartialStruct && return _fvc4_has_box(x.typ) || any(_fvc4_has_box, x.fields)
    x isa DataType && return x === Core.Box || any(_fvc4_has_box, x.parameters)
    x isa Union && return any(_fvc4_has_box, Base.uniontypes(x))
    x isa GlobalRef && return x.mod === Core && x.name === :Box
    x isa Expr && return any(_fvc4_has_box, x.args)
    x isa QuoteNode && return _fvc4_has_box(x.value)
    return false
end

function _fvc4_inference_issues(ci, rt)
    issues = String[]
    rt === Nothing || push!(issues, "kernel return is $rt")
    for t in ci.slottypes
        _fvc4_has_box(t) && push!(issues, "boxed slot: $t")
    end
    for (i, ex) in enumerate(ci.code)
        t = ci.ssavaluetypes[i]
        _fvc4_has_box(ex) || _fvc4_has_box(t) ? push!(issues, "boxed SSA $i") : nothing
        # Meta/loop annotations carry Any without producing runtime values.
        computes = ex isa Core.PhiNode || ex isa Core.PiNode ||
                   (ex isa Expr && ex.head in (:call, :invoke, :new, :splatnew, :foreigncall))
        computes && !_fvc4_concrete(t) && push!(issues, "abstract SSA $i: $t")
        if ex isa Expr && ex.head === :call
            f = ex.args[1]
            callee = f isa GlobalRef ? getfield(f.mod, f.name) : f isa QuoteNode ? f.value : f
            # Fully inferred Julia calls use :invoke or inline to primitives.
            # An SSA/slot callee or a remaining generic call is runtime dispatch.
            callee isa Core.Builtin || callee isa Core.IntrinsicFunction ||
                push!(issues, "runtime call at SSA $i: $f")
        elseif ex isa Expr && ex.head === :foreigncall
            push!(issues, "foreign call at SSA $i")
        end
    end
    return unique(issues)
end

function _fvc4_kernel_issues(factory, args, ndrange)
    kernel = factory(KernelAbstractions.CPU(), 64)
    ci, rt = only(KernelAbstractions.ka_code_typed(kernel, typeof(args); ndrange))
    return _fvc4_inference_issues(ci, rt)
end

@testset "FV adjoint kernel inference (CPU, not GPU validation)" begin
    p = fv_steady_problem(3, 2)
    n = fv_nunknowns(p)
    U = zeros(n)
    colors = Int32[1, 2, 3, 1, 2, 3]
    covered = Set{Symbol}()
    for chunk in (1, 4, 8)
        D = ForwardDiff.Dual{Kraken.FVParamTag,Float64,chunk}
        Rd = zeros(D, n)
        nud = zeros(D, size(p.nu))
        cases = (
            (:_fv_lambda_dot_kernel!, (zeros(chunk, 2), U, Rd, n, 10, Val(chunk)), 2),
            (:_fv_nu_seed_kernel!, (nud, p.nu, colors, 0, Val(chunk)), length(p.nu)),
            (:_fv_nu_direction_kernel!, (nud, p.nu, ones(size(p.nu)), Val(chunk)), length(p.nu)),
            (:_fv_partial1_kernel!, (similar(U), Rd), n),
            (:_fv_nu_gather_kernel!, (zeros(size(p.nu)), U, Rd, p.tag, colors, p.grid, 0, Val(chunk)), size(p.nu)),
        )
        for (name, args, ndrange) in cases
            push!(covered, name)
            @testset "$name chunk=$chunk" begin
                @test isempty(_fvc4_kernel_issues(getfield(Kraken, name), args, ndrange))
            end
        end
    end
    # KA emits all method locations as macros.jl. Parse the declarations only
    # for inventory; safety above is checked on inferred executable code.
    source = read(joinpath(@__DIR__, "..", "..", "src", "methods", "inc_ns", "fv_adjoint.jl"), String)
    actual = Set{Symbol}()
    for ex in Meta.parseall(source).args
        ex isa Expr && ex.head === :macrocall && ex.args[1] === Symbol("@kernel") || continue
        definition = ex.args[end]
        signature = definition.args[1]
        signature.head === :where && (signature = signature.args[1])
        push!(actual, signature.args[1])
    end
    @test covered == actual
end
