# T2: ice stream uniform in x (flat bed, surface slope in x, no-slip margins).
# The 2D code (zero-gradient x edges, Dirichlet y edges) must reproduce the 1D
# across-flow balance  d_y(eta H u_y) - tau_b = rho g H s_x.
#  (a) Glen n = 1, linear drag: analytic u = u_p (1 - cosh(k (y-W/2)) / cosh(k W/2)),
#      k^2 = beta / (eta H), u_p = -f / beta; second-order convergence.
#  (b) Glen n = 3, Weertman m = 1/3: no closed form, self-convergence by block
#      averaging onto the coarser grid. The u^(1/3) drag makes a steep margin layer,
#      so the ratio of successive differences approaches 4 from below (2.7, 3.1, ...).
using Test
if !isdefined(@__MODULE__, :ssa_stream_problem)
    include(joinpath(@__DIR__, "ssa_helpers.jl"))
end

function ssa_stream1d_problem(Ny; n, drag, A, H0 = 1000.0, alpha = 0.005, W = 20e3, Nx = 3)
    return ssa_problem(Nx, Ny; Lx = W * Nx / Ny, Ly = W, H = H0, s = (x, y) -> H0 - alpha * x,
                       A, n, drag, dirichlet = (false, false, true, true))
end
ssa_profile(r, p) = vec(sum(r.u; dims = 1)) ./ p.Nx

@testset "SSA ice stream across-flow profile (T2)" begin
    W = 20e3; H0 = 1000.0; alpha = 0.005; Aone = 1e-8; beta = 2.0   # n = 1: eta = 1/(2A)
    @testset "n = 1, linear drag: analytic profile" begin
        eta = 1 / (2Aone); k = sqrt(beta / (eta * H0))
        f = SSA_RHOG * H0 * (-alpha); up = -f / beta
        errs = Float64[]
        for Ny in (8, 16, 32)
            p = ssa_stream1d_problem(Ny; n = 1, drag = SSAWeertman(beta, 1.0), A = Aone, W)
            r = ssa_solve(p; U0 = zeros(ssa_nunknowns(p)))
            @test r.converged
            y = ((1:Ny) .- 0.5) .* (W / Ny)
            exact = up .* (1 .- cosh.(k .* (y .- W / 2)) ./ cosh(k * W / 2))
            push!(errs, maximum(abs, ssa_profile(r, p) .- exact))
            @test maximum(abs, r.v) < 1e-8                      # no cross-flow
            @test maximum(abs, r.u .- r.u[1:1, :]) < 1e-8      # uniform in x
        end
        orders = log2.(errs[1:end-1] ./ errs[2:end])
        println("SSA T2 n=1 analytic: max errors ", errs, " observed orders ", orders, " u_peak ", up)
        @test all(orders .> 1.9)
        @test errs[end] < 1e-3 * up
    end
    @testset "n = 3, Weertman: self-convergence" begin
        sols = Vector{Float64}[]
        for Ny in (16, 32, 64, 128, 256)
            p = ssa_stream1d_problem(Ny; n = 3, drag = SSAWeertman(SSA_C, 1 / 3, 5.0), A = SSA_A, W)
            r = ssa_solve(p)
            @test r.converged
            push!(sols, ssa_profile(r, p))
        end
        avg(v) = (v[1:2:end] .+ v[2:2:end]) ./ 2
        d = [maximum(abs, avg(sols[k+1]) .- sols[k]) for k in 1:4]
        ratios = d[1:end-1] ./ d[2:end]
        println("SSA T2 n=3 self-convergence: diffs ", d, " ratios ", ratios, " u_max ", maximum(sols[end]))
        @test issorted(ratios) && ratios[end] > 3.2        # -> 4 (second order)
        @test maximum(sols[end]) > 1.0
    end
end
