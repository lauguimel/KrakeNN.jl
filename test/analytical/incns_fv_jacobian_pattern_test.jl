# T0b: structural completeness and actual coloured FV Jacobian, CPU Float64.
# Reproduce: sh tmp/thermal_julia.sh --project=. -O0 test/analytical/incns_fv_jacobian_pattern_test.jl
# Measured session: ~10 s wall time, CPU Julia 1.12.5 on macOS arm64.
# Frozen gates from tmp/thermal_plan_final.md, T0b/R1/R5 (2026-11-08 mandate):
# significant reference entry > 1e-13*maxabs(Jref); every coloured entry agrees
# within 1e-12*maxabs(Jref). These are derivative-consistency gates, not a
# continuum-accuracy claim. The reference seeds every state column separately,
# never consulting the declared sparsity or colours, including M3 pair rows.
using Test, KrakeNN, ForwardDiff, Random
using KrakeNN.SparseArrays

function _fvt0b_dense_jacobian(p, U)
    # Reuse its dual type to avoid compiling a second copy of the same kernel.
    # Only lane 1 is seeded, in exactly one column; other lanes stay zero.
    # Neither the seed nor the dense extraction consults pattern or colouring.
    D = ForwardDiff.Dual{KrakeNN.FVJacobianTag,Float64,4}
    Ud = D.(U)
    Rd = similar(Ud)
    J = zeros(length(U), length(U))
    for col in eachindex(U)
        Ud[col] = D(U[col], ForwardDiff.Partials((1.0, 0.0, 0.0, 0.0)))
        fv_residual!(Rd, Ud, p)
        for row in eachindex(U)
            J[row, col] = ForwardDiff.partials(Rd[row])[1]
        end
        Ud[col] = D(U[col])
    end
    return J
end

function _fvt0b_state(p)
    rng = MersenneTwister(70819)
    U = 0.3 .* randn(rng, fv_nunknowns(p))
    g = p.grid
    for comp in 1:2
        ids = [fv_index(g, i, j, comp) for j in 1:g.Ny for i in 1:g.Nx
               if p.tag[i, j] != FV_CELL_SOLID]
        mean_value = sum(U[ids]) / length(ids)
        U[ids] .+= (comp == 1 ? 0.12 : 0.07) - mean_value
    end
    U[3:3:fv_nunknowns(g)] .*= 0.17
    return U
end

function _fvt0b_fluxes(p, U)
    # Diagnostic evaluation of the actual interior Rhie--Chow formula in
    # fv_cell_residual, not a separate residual or differentiation reference.
    g = p.grid
    q, w, closure = KrakeNN._fv_immersed_args(p.immersed)
    flux = Float64[]; mean_normal = Float64[]; xflux = Float64[]
    cutflux = Float64[]
    for j in 1:g.Ny, i in 1:g.Nx
        p.tag[i, j] == FV_CELL_SOLID && continue
        gp = KrakeNN._fv_cell_grad(U, p.tag, p.bvel, g, i, j, q, w, closure)
        for d in (1, 3) # Unique east/north orientation, including periodic wrap.
            kind, ii, jj = KrakeNN._fv_face_kind(g, p.tag, i, j, d)
            kind == KrakeNN._FV_F_INT || continue
            gn = KrakeNN._fv_cell_grad(U, p.tag, p.bvel, g, ii, jj, q, w, closure)
            up, vp, pp = ntuple(c -> U[fv_index(g, i, j, c)], 3)
            un, vn, pn = ntuple(c -> U[fv_index(g, ii, jj, c)], 3)
            gradient = d == 1 ? ((un-up)/g.h, (gp[2]+gn[2])/2,
                                 (vn-vp)/g.h, (gp[4]+gn[4])/2) :
                                ((gp[1]+gn[1])/2, (un-up)/g.h,
                                 (gp[3]+gn[3])/2, (vn-vp)/g.h)
            nu = KrakeNN._fv_face_nu(p.visc, p.nu[i,j], p.nu[ii,jj],
                                    gradient..., i, j, ii, jj)
            ub = d == 1 ? (up+un)/2 : (vp+vn)/2
            df = g.h^2 / (4nu + g.conv*abs(ub)*g.h)
            gpnormal = d == 1 ? (gp[5]+gn[5])/2 : (gp[6]+gn[6])/2
            f = g.h * (ub - df*((pn-pp)/g.h-gpnormal))
            push!(flux, f); push!(mean_normal, ub)
            d == 1 && push!(xflux, f)
        end
        if p.immersed !== nothing
            for d in 1:4
                q[d, i, j] > 0 || continue
                parts = KrakeNN._fv_wall_flux_components(U, p.tag, p.nu, p.bvel,
                    g, i, j, d, p.visc, q, w, closure, gp,
                    KrakeNN._fv_face_pair(p.immersed))
                push!(cutflux, last(parts))
            end
        end
    end
    return (; flux, mean_normal, xflux, cutflux)
end

function _fvt0b_check(name, p)
    s = fv_newton_solver(p)
    pattern = Matrix(fv_jacobian_pattern(p))
    U = _fvt0b_state(p)
    forward_flux = _fvt0b_fluxes(p, U)
    @testset "$name" begin
        @test eltype(U) === Float64
        @test p.grid.conv == 1
        if p.immersed !== nothing
            @test p.immersed.closure.ghost == 2
            @test p.immersed.closure.mass == 3
            @test fv_nunknowns(p) > fv_nunknowns(p.grid)
        end
        for direction in (1, -1)
            @testset "mean direction $direction" begin
                X = direction .* U
                for comp in 1:2
                    ids = [fv_index(p.grid, i, j, comp) for j in 1:p.grid.Ny
                           for i in 1:p.grid.Nx if p.tag[i,j] != FV_CELL_SOLID]
                    @test minimum(X[ids]) < 0 < maximum(X[ids])
                    @test direction * sum(X[ids]) > 0
                end
                f = _fvt0b_fluxes(p, X)
                @test minimum(f.flux) < 0 < maximum(f.flux)
                @test minimum(abs, f.flux) > 0
                @test minimum(abs, f.mean_normal) > 0
                @test direction * sum(f.xflux) > 0
                @test f.flux ≈ direction .* forward_flux.flux rtol=1e-14 atol=0
                if !isempty(f.cutflux)
                    @test minimum(abs, f.cutflux) > 0
                    @test f.cutflux ≈ direction .* forward_flux.cutflux rtol=1e-14 atol=0
                end
                R = similar(X)
                fv_residual!(R, X, p)
                @test maximum(abs, R) > 0 # Intentionally unconverged states.
                Jref = _fvt0b_dense_jacobian(p, X)
                Jcol = Matrix(fv_jacobian!(s, X))
                @test all(isfinite, Jref)
                @test all(isfinite, Jcol)
                scale = maximum(abs, Jref)
                @test scale > 0
                significant = abs.(Jref) .> 1e-13 * scale
                missing = findall(significant .& .!pattern)
                difference = abs.(Jcol .- Jref)
                bad = findall(difference .> 1e-12 * scale)
                @info("T0b Jacobian", case=name, direction,
                    pattern_nnz=count(pattern), reference_nnz=count(significant),
                    numerical_zeros=count(pattern .& .!significant),
                    missing=length(missing), unequal=length(bad),
                    max_difference=maximum(difference), reference_max=scale,
                    min_flux=minimum(abs, f.flux),
                    positive_fluxes=count(>(0), f.flux), negative_fluxes=count(<(0), f.flux),
                    mean_x_flux=sum(f.xflux)/length(f.xflux),
                    min_cut_flux=isempty(f.cutflux) ? nothing : minimum(abs, f.cutflux))
                if !isempty(missing) || !isempty(bad)
                    @info "T0b defect entries" case=name direction missing=Tuple.(missing) unequal=Tuple.(bad)
                end
                for entry in findall(significant)
                    @test pattern[entry]
                end
                for entry in CartesianIndices(Jref)
                    @test difference[entry] <= 1e-12 * scale
                end
            end
        end
    end
end

@testset "FV Jacobian pattern completeness (T0b, CPU Float64)" begin
    nx, ny = 8, 6
    bc = (W=:periodic, E=:periodic, S=:wall, N=:wall)
    common = (; h=1.0, bc, conv=true, source=(x,y)->(0.013, -0.007, 0.0))
    cy = CarreauYasuda(0.3, 0.05, 1.2, 2.0, 0.7)
    field = [0.13i + 0.09j + 0.03sin(i*j) for i in 1:nx, j in 1:ny]
    law = LearnedViscosity(TinyMLP{2,2,1}(); nu_ref=0.3, monotone=false)
    # Public TinyMLP layout: input weights, hidden biases, output weights/bias.
    theta = [0.3, -0.2, 0.4, -0.3, 0.1, -0.15, 0.2, -0.15, 0.04]
    visc = FVLawViscosity(law, theta, (field,); eps=0.02)
    @test law_value(law, (0.4, 0.7), theta) != law_value(law, (0.8, 0.7), theta)
    @test law_value(law, (0.4, 0.7), theta) != law_value(law, (0.4, 1.1), theta)
    solid = falses(nx, ny)
    solid[4:5, 3:4] .= true
    solid[[nx, 1], 2:3] .= true # One connected block straddles the periodic seam.
    _fvt0b_check("Newtonian periodic channel 9x7", fv_steady_problem(9, 7; nu=0.3, common...))
    _fvt0b_check("Carreau-Yasuda periodic channel", fv_steady_problem(nx, ny; rheology=cy, gnf_eps=0.02, common...))
    _fvt0b_check("Learned law prescribed field", fv_steady_problem(nx, ny; viscosity=visc, common...))
    _fvt0b_check("Carreau-Yasuda two voxel blocks", fv_steady_problem(nx, ny; rheology=cy, gnf_eps=0.02, solid, common...))
    _fvt0b_check("Learned law two voxel blocks", fv_steady_problem(nx, ny; viscosity=visc, solid, common...))
    # Smallest checked circle: radius >=2h and a >=3h gap on all domain sides
    # require 10x10. Keep the production geometry checks and one M3 lambda.
    ib = fv_immersed_circle(10, 10, 5.0, 5.0, 2.0; h=1.0,
                            closure=:quadratic, mass=:M3, pressure=:linear)
    _fvt0b_check("Newtonian immersed quadratic M3", fv_steady_problem(10, 10;
        nu=0.3, immersed=ib, wall_flux=0.0, common...))
end
