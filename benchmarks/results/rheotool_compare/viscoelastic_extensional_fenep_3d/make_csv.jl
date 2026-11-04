# Build the FENE-P transient CSV + error-norm CSV from the rheoTestFoam Report.
# Reconstructs the conformation tensor in Kraken's convention (C, equilibrium I)
# from RheoTool's total extra-stress, FAITHFUL to RheoTool's FENE-P closure
# (transport A with equilibrium a*I, varf=1/(1-trA/L2); C = A/a).
#
#   etaS=etaP=0.05, lambda=50, eps_dot=0.005, L2=50, planar L=eps*diag(1,-1,0).
#
# Run from the comparison dir with the Report path as ARGS[1]:
#   julia make_csv.jl ../../../bench/rheotool/extensional_fenep_ve_planar/Report
using Printf

const ETAS=0.05; const ETAP=0.05; const LAM=50.0; const EPS=0.005
const L2=50.0;   const G=ETAP/LAM; const A_EQ=L2/(L2-3.0)
const SOL_XX=ETAS*2*EPS; const SOL_YY=-ETAS*2*EPS   # solvent diag(1,-1,0)*etaS*2eps

# Reconstruct Kraken-convention C from a RheoTool total-stress row (diagonal).
function recon(tot_xx, tot_yy, tot_zz)
    tpx = tot_xx - SOL_XX; tpy = tot_yy - SOL_YY; tpz = tot_zz - 0.0
    Qx = tpx/G + A_EQ; Qy = tpy/G + A_EQ; Qz = tpz/G + A_EQ
    varf = 1.0 + (Qx+Qy+Qz)/L2
    Ax = Qx/varf; Ay = Qy/varf; Az = Qz/varf
    return (Ax/A_EQ, Ay/A_EQ, Az/A_EQ)        # C = A/a
end

report = length(ARGS) >= 1 ? ARGS[1] :
    joinpath(@__DIR__, "..","..","..","bench","rheotool","extensional_fenep_ve_planar","Report")

rows = Tuple{Float64,Float64,Float64,Float64}[]   # t, totxx, totyy, totzz
for ln in eachline(report)
    s = strip(ln)
    (isempty(s) || startswith(s, "*") || startswith(s, "t")) && continue
    p = split(s)
    length(p) < 7 && continue
    t = tryparse(Float64, p[1]); t === nothing && continue
    push!(rows, (t, parse(Float64,p[2]), parse(Float64,p[5]), parse(Float64,p[7])))
end
@printf("read %d data rows from %s\n", length(rows), report)

# Downsample to ~80 rows (keep first/last).
n = length(rows); stride = max(1, n ÷ 78)
keep = unique(vcat(1:stride:n, n))

open(joinpath(@__DIR__, "rheotool_fenep_extensional_transient.csv"), "w") do io
    println(io, "# RheoTool rheoTestFoam FENE-P planar extension, lambda=50 eps_dot=0.005 (2*lambda*eps=0.5), beta=0.5, L2=50")
    println(io, "# C reconstructed in Kraken convention from RheoTool total extra-stress:")
    println(io, "#   tau_p = tauTotal - etaS*(L+L^T); RheoTool A from tau_p=(etaP/lam)(varf*A-a*I), varf=1/(1-trA/L2), a=L2/(L2-3); C=A/a.")
    println(io, "t,extStressXX,extStressYY,extStressZZ,C_xx,C_yy,C_zz")
    for k in keep
        t,txx,tyy,tzz = rows[k]
        cx,cy,cz = recon(txx,tyy,tzz)
        @printf(io, "%g,%.8e,%.8e,%.8e,%.8f,%.8f,%.8f\n", t,txx,tyy,tzz,cx,cy,cz)
    end
end

# Steady reconstruction (last row) + cross-validation table.
tlast = rows[end]
cx,cy,cz = recon(tlast[2],tlast[3],tlast[4])
trC = cx+cy+cz
# Kraken FVFD canary center (test_fvfd_fenep_extensional_3d.jl gate G2):
KR_CXX = 1.949729; KR_TRC = 3.597994
# Kraken steady transcendental (its own closure, f=(L2-3)/(L2-trC), C-eq=I):
let
    w=2*LAM*EPS; fk=1.0; cxk=cyk=czk=trk=0.0
    for _ in 1:400
        cxk=1.0/(fk-w); cyk=1.0/(fk+w); czk=1.0/fk; trk=cxk+cyk+czk
        fk=(L2-3.0)/(L2-trk)
    end
    open(joinpath(@__DIR__,"viscoelastic_extensional_fenep_3d_error_norms.csv"),"w") do io
        println(io,"# FENE-P planar extension: RheoTool (rheoTestFoam) vs Kraken FVFD log-conf. CROSS-VALIDATION (both numerical).")
        println(io,"# Operating point: lambda=50, eps_dot=0.005 (2*lambda*eps_dot=0.5), beta=etaS/(etaS+etaP)=0.5, L2=50.")
        println(io,"# FENE-P has NO closed-form fixed point (transcendental in trC). Conformation reported in Kraken's convention (C-eq=I).")
        println(io,"# RheoTool and Kraken use DIFFERENT FENE-P Peterlin closures (varf=1/(1-trA/L2), A-eq=a*I  vs  f=(L2-3)/(L2-trC), C-eq=I);")
        println(io,"# they coincide only as L2->inf (Oldroyd-B). OB-limit sanity: RheoTool at L2=1e5 gives C_xx=1.99985 (0.007% vs OB=2).")
        println(io,"component,kraken_canary_1000steps,kraken_transcendental,rheotool_fenep_steady,rel_err_rt_vs_kraken_canary")
        @printf(io,"C_xx,%.6f,%.6f,%.6f,%.3e\n", KR_CXX, cxk, cx, abs(cx-KR_CXX)/KR_CXX)
        @printf(io,"C_yy,%s,%.6f,%.6f,%s\n", "NA", cyk, cy, "NA")
        @printf(io,"C_zz,%s,%.6f,%.6f,%s\n", "NA", czk, cz, "NA")
        @printf(io,"trC,%.6f,%.6f,%.6f,%.3e\n", KR_TRC, cxk+cyk+czk, trC, abs(trC-KR_TRC)/KR_TRC)
    end
    @printf("steady: RheoTool C_xx=%.4f trC=%.4f | Kraken canary C_xx=%.3f trC=%.2f | transcendental C_xx=%.4f\n",
            cx, trC, KR_CXX, KR_TRC, cxk)
    @printf("closure gap RheoTool vs Kraken C_xx = %.2f %%\n", 100*abs(cx-cxk)/cxk)
end
println("wrote CSVs")
