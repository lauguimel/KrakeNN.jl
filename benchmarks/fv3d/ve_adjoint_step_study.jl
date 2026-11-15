# Step study of the centred-FD gate G2 of the 3D FV conformation adjoint (plan
# "discrete viscoelastic adjoints", step F2, R2). It fixes the step range and the
# thresholds VE_G2_STEPS / VE_G2_REL / VE_G2_ABS of
# test/analytical/fv3d_conformation_gate_setup.jl (same slice, states, objective
# and step scales as test/analytical/incns_fv3d_conformation_gate_test.jl).
#
# Directions: Giesekus alpha (De = 1, the stiffest Giesekus row) and FENE-P L2
# (De = 0.5). The learned-law direction of the plan needs the learned relaxation
# law of step F4, which does not exist yet: it is skipped and reported as such.
#
# For each direction, adjoint g_k at θ, then for Newton atol in (1e-14, 1e-15) and
# h in FV3D_STEPS (default 1e-2,...,1e-6), eps = h s_k:
#   D(h)    centred FD of the reconverged objective (warm start from the state at θ)
#   rel     |D(h) - g_k| / |g_k|
#   ratio   Richardson |D(2h) - D(4h)| / |D(h) - D(2h)| (~4 when truncation dominates)
#   tight   |D_1e-14(h) - D_1e-15(h)| / (VE_G2_REL |g_k|), the effect of tightening
#           Newton in units of the target error (should be <= 1e-3)
#   gate    rel error against the G2 tolerance VE_G2_REL |g_k| + VE_G2_ABS |J| / s_k
# Retained range: the h where ratio ~ 4 (3 to 5) and tight <= 1e-3 at both atol.
#
#   FV3D_BACKEND=cpu  julia --project benchmarks/fv3d/ve_adjoint_step_study.jl
#   FV3D_BACKEND=cuda julia --project=<env with CUDA, CUDSS> benchmarks/fv3d/ve_adjoint_step_study.jl
# Optional: FV3D_NS=8 (slice size), FV3D_STEPS=1e-2,1e-3,1e-4,1e-5,1e-6.
if lowercase(get(ENV, "FV3D_BACKEND", "cuda")) == "cuda"
    using CUDA
    CUDA.functional() || error("CUDA not functional")
    using CUDSS
end
include(joinpath(@__DIR__, "fv3d_common.jl"))
include(joinpath(@__DIR__, "..", "..", "test", "analytical", "fv3d_conformation_gate_setup.jl"))

const STEPS = fv3d_floats("FV3D_STEPS", "1e-2,1e-3,1e-4,1e-5,1e-6")
const NSTUDY = first(fv3d_ints("FV3D_NS", "8"))
const ATOLS = (1e-14, 1e-15)

function study(r::VERow, pname::Symbol; N = NSTUDY)
    to = fv3d_to(); linsolve = fv3d_linsolve()
    s, U = ve_state(r; N, to, linsolve)
    Ud = ve_data_state(r, s, U; to, linsolve)
    obj, _ = ve_objective(s.prob, Ud; to)
    m = ve_map(r); θ = ve_theta(r); sk = ve_scales(r)
    k = findfirst(==(pname), ve_names(r))
    res = fv_steady_gradient(s, U, θ, obj; params = m)
    g = res.grad[k]; J = res.J
    tol = VE_G2_REL * abs(g) + VE_G2_ABS * J / sk[k]
    sf = fv_newton_solver(KrakeNN._fv_host_problem(s.prob); to, linsolve)
    hs = sort(unique([STEPS; 2 .* STEPS; 4 .* STEPS]))
    D = Dict{Tuple{Float64,Float64},Float64}()
    R = Dict{Tuple{Float64,Float64},Float64}()
    for atol in ATOLS, h in hs
        D[(atol, h)], R[(atol, h)] = ve_fd(sf, m, θ, k, h, sk[k], U, obj; atol, rtol = 0.0)
    end
    @printf("\n=== %s, parameter %s = %.6g (s_k = %.6g), N = %d, backend %s\n", r.label, pname, θ[k], sk[k], N,
            FV3D_BACKEND)
    @printf("adjoint g_k = %.12e, J = %.6e, G2 tolerance = %.3e (%.2e relative)\n", g, J, tol, tol / abs(g))
    @printf("%8s %8s %20s %10s %8s %10s %10s %5s\n", "atol", "h", "D(h)", "rel", "ratio", "tight", "Newton res", "gate")
    keep = Float64[]
    for h in STEPS
        okh = true
        for atol in ATOLS
            d1 = D[(atol, h)]; d2 = D[(atol, 2h)]; d4 = D[(atol, 4h)]
            rel = abs(d1 - g) / abs(g)
            ratio = abs(d2 - d4) / max(abs(d1 - d2), eps())
            tight = abs(D[(ATOLS[1], h)] - D[(ATOLS[2], h)]) / (VE_G2_REL * abs(g))
            pass = abs(d1 - g) <= tol
            okh &= 3 <= ratio <= 5 && tight <= 1e-3
            @printf("%8.0e %8.0e %20.12e %10.3e %8.2f %10.3e %10.2e %5s\n", atol, h, d1, rel, ratio, tight,
                    R[(atol, h)], pass ? "PASS" : "FAIL")
        end
        okh && push!(keep, h)
    end
    println("retained h (ratio in [3, 5] and tight <= 1e-3 at both atol): ",
            isempty(keep) ? "none" : join(string.(keep), ", "))
    gate = [h for h in VE_G2_STEPS if abs(D[(ATOLS[1], h)] - g) <= tol]
    println("test steps VE_G2_STEPS = $(VE_G2_STEPS): within the G2 tolerance at atol 1e-14 for h = ",
            isempty(gate) ? "none" : join(string.(gate), ", "))
    return (; label = r.label, pname, keep, gate)
end

out = [study(ve_giesekus(1.0), :alpha), study(ve_fenep(0.5), :L2)]
println("\nlearned-law direction: skipped (the learned relaxation law of step F4 is not implemented yet)")
println("\nsummary")
for o in out
    println("  $(o.label) / $(o.pname): retained h = $(o.keep), test steps passing = $(o.gate)")
end
