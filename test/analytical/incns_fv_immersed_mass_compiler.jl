# CPU inference guards for the M3 kernels. No CUDA execution is implied.
using Test, Kraken, ForwardDiff, KernelAbstractions

if !isdefined(@__MODULE__, :_fvc4_kernel_issues)
    include(joinpath(@__DIR__, "incns_fv_adjoint_compiler.jl"))
end

@testset "M3 pair residual and direct Jacobian kernel inference" begin
    covered = Set{Symbol}()
    for closure in (:linear, :quadratic)
        ib = fv_immersed_circle(12, 12, 6.13, 5.87, 2.21; h=1.0, closure, mass=:M3)
        p = fv_steady_problem(12, 12; h=1.0, immersed=ib, wall_flux=0.0)
        b, g = p.immersed, p.grid
        m = b.mass
        n = fv_nunknowns(p)
        for T in (Float32, Float64), chunk in (1, 4)
            D = ForwardDiff.Dual{Kraken.FVParamTag,T,chunk}
            V = fill(T(0.1), n)
            cases = (
                ("state", D.(V), T.(b.qcut), T.(b.wvel), T.(m.Q), T.(m.A)),
                ("cut", V, D.(b.qcut), T.(b.wvel), T.(m.Q), T.(m.A)),
                ("wall", V, T.(b.qcut), D.(b.wvel), T.(m.Q), T.(m.A)),
                ("target and area", V, T.(b.qcut), T.(b.wvel), D.(m.Q), D.(m.A)),
            )
            for (payload, U, q, w, Q, A) in cases
                @testset "$closure $T chunk=$chunk $payload" begin
                    args = (zeros(promote_type(D, typeof(g.h)), n), U, p.bvel,
                        g, q, w, b.closure, m.faceptr, m.faces, Q, A)
                    @test isempty(_fvc4_kernel_issues(Kraken._fv_mass_residual_kernel!,
                        args, length(m.Q)))
                end
            end
        end
        push!(covered, :_fv_mass_residual_kernel!)
        s = fv_newton_solver(p)
        args = (s.nzval, s.asm.nzrow, s.asm.nzcol, fv_nunknowns(g),
                m.rowptr, m.rowcol, m.rowval)
        @test isempty(_fvc4_kernel_issues(Kraken._fv_mass_jacobian_kernel!, args, s.asm.nnz))
        push!(covered, :_fv_mass_jacobian_kernel!)
        # Decompression must skip the global pair rows even when local
        # columns share a color there; the direct kernel supplies those rows.
        chunk = Kraken.fv_jacobian_chunk(s.asm)
        args = (s.nzval, s.asm.nzrow, s.asm.nzcol, s.asm.colors,
                s.asm.Rd, 0, Val(chunk), fv_nunknowns(g))
        @test isempty(_fvc4_kernel_issues(Kraken._fv_decompress_kernel!, args, s.asm.nnz))
    end
    source = read(joinpath(@__DIR__, "..", "..", "src", "methods", "inc_ns",
        "fv_immersed_mass.jl"), String)
    declared = Set{Symbol}()
    for ex in Meta.parseall(source).args
        ex isa Expr && ex.head === :macrocall && ex.args[1] === Symbol("@kernel") || continue
        signature = ex.args[end].args[1]
        signature.head === :where && (signature = signature.args[1])
        push!(declared, signature.args[1])
    end
    @test covered == declared
end

@testset "M3 augmented residual norm and state checksum inference" begin
    for T in (Float32, Float64), nextra in (1, 3, 5)
        ncell, per = 257, 256
        nt = cld(ncell+nextra, per)
        args = (zeros(4, nt), zeros(T, 3ncell+nextra), ncell, per)
        @test isempty(_fvc4_kernel_issues(Kraken._fv_norm_partials_kernel!, args, nt))
        @test isempty(_fvc4_kernel_issues(Kraken._fv_state_sig_kernel!, args, nt))
    end
end

@testset "M3 corrected cell flux kernel inference" begin
    D = ForwardDiff.Dual{Kraken.FVJacobianTag,Float64,4}
    law = CarreauYasuda(0.1, 0.01, 2.0, 2.0, 0.6)
    for closure in (:linear, :quadratic), rheology in (nothing, law)
        ib = fv_immersed_circle(12, 12, 6.13, 5.87, 2.21; h=1.0, closure, mass=:M3)
        p = fv_steady_problem(12, 12; h=1.0, immersed=ib, wall_flux=0.0,
            rheology, gnf_eps=1e-2, conv=true)
        n = fv_nunknowns(p)
        for T in (Float64, D)
            args = (zeros(T, n), fill(T(0.1), n), p.tag, p.nu, p.bvel,
                    p.src, p.grid, p.visc, Kraken._fv_immersed_args(p.immersed)...,
                    p.immersed.mass.face_pair)
            @test isempty(_fvc4_kernel_issues(Kraken._fv_residual_kernel!, args, (12, 12)))
        end
    end
end
