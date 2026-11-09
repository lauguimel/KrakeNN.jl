"""Independent continuum BVP reference for the CPU thermal coupling tests.

No KrakeNN implementation, residual, interpolation or reference driver is used.
The first-order ODE uses shear stress and conductive temperature slope, solved
by scipy's fourth-order collocation. Top-temperature continuation makes the
fixed-force turning point accessible without following the FV Newton solver.
Run: conda run -n kraken-v0-3-figures python benchmarks/fv_thermal/nahme_reference.py
"""
from pathlib import Path
import json
import numpy as np
import scipy
from scipy.integrate import solve_bvp
from scipy.optimize import minimize_scalar

OUT = Path(__file__).resolve().parents[2] / "test/reference/fv_thermal_nahme"
NU, ALPHA, BETA, TW, FORCE = 0.3, 0.2, 1.0, 1.0, 0.6


def solution(kind, *, sigma=None, top_rise=None, tol=1e-10, points=129, fixed_mesh=False):
    force, wall = (FORCE, 0.0) if kind == "poiseuille" else (0.0, 1.0)
    x = np.linspace(0, 1, points)
    t = 0.5 if top_rise is None else top_rise
    guess = np.array([wall*x + force*x*(1-x)/(2*NU),
                      NU*wall+force*(0.5-x), TW+t*(2*x-x*x), 2*t*(1-x)])

    def rhs(x, state, parameter=None):
        s = sigma if parameter is None else parameter[0]
        nu = NU*np.exp(-BETA*(state[2]-TW))
        return np.array([state[1]/nu, np.full_like(x, -force),
                         state[3], -s*state[1]**2/(ALPHA*nu)])

    def bc(a, b, parameter=None):
        values = [a[0], b[0]-wall, a[2]-TW, b[3]]
        if parameter is not None:
            values.append(b[2]-TW-top_rise)
        return np.array(values)

    options = {} if top_rise is None else {"p": np.array([0.5])}
    sol = solve_bvp(rhs, bc, x, guess, tol=tol,
                    max_nodes=points if fixed_mesh else 30000, **options)
    if fixed_mesh:
        # Reaching max_nodes ends solve_bvp's outer loop before its bounded
        # Newton loop necessarily exhausts the nonlinear algebraic defect.
        # Restart on exactly the same mesh until that defect is stationary.
        for _ in range(5):
            old = sol.y.copy()
            options = {} if top_rise is None else {"p": sol.p}
            sol = solve_bvp(rhs,bc,x,old,tol=tol,max_nodes=points,**options)
            if np.max(np.abs(sol.y-old)) < 1e-13:
                break
    if not sol.success and not (fixed_mesh and sol.status == 1):
        raise RuntimeError(sol.message)
    return sol


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    y = np.unique(np.concatenate([(np.arange(n)+0.5)/n for n in (16,32,64,128)]))
    metadata = {"date": "2026-11-08", "scipy": scipy.__version__,
                "nu0": NU, "alpha": ALPHA, "beta": BETA, "T_wall": TW,
                "H": 1.0, "force": FORCE, "cases": {}}
    for kind in ("poiseuille", "couette"):
        opts = {"top_rise": 0.7} if kind == "poiseuille" else {"sigma": 0.8}
        sols = [solution(kind, tol=tol, points=n, **opts)
                for tol,n in ((1e-6,33),(1e-8,65),(1e-10,129))]
        fine = sols[-1]
        sigma = float(fine.p[0]) if kind == "poiseuille" else opts["sigma"]
        states = [s.sol(y) for s in sols]
        errors = [float(np.max(np.abs(states[k]-states[-1]))) for k in (0,1)]
        values = fine.sol(y)
        nu = NU*np.exp(-BETA*(values[2]-TW))
        np.savetxt(OUT / (kind+".csv"), np.column_stack([y,values[0],values[2],nu]),
                   delimiter=",", header="y,u,T,nu", comments="", fmt="%.17g")
        mesh = np.linspace(0,1,4097)
        values_mesh = fine.sol(mesh)
        # Independently computed continuum curvatures set a conservative h^2
        # acceptance envelope before any FV run; this is not an FV calibration.
        numesh = NU*np.exp(-(values_mesh[2]-TW))
        force = FORCE if kind == "poiseuille" else 0.0
        upp = (-force+values_mesh[1]*values_mesh[3])/numesh
        tpp = -sigma*values_mesh[1]**2/(ALPHA*numesh)
        nupp = numesh*(values_mesh[3]**2-tpp)
        curvature = float(max(np.max(np.abs(upp)),np.max(np.abs(tpp)),np.max(np.abs(nupp))))
        info = {"sigma":sigma,"top_rise":float(fine.sol(1)[2]-TW),
                "center_nu_change":float(1-np.exp(-(fine.sol(.5)[2]-TW))),
                "nodes":[len(s.x) for s in sols], "tolerances":[1e-6,1e-8,1e-10],
                "coarse_medium_differences_to_fine":errors,
                "reference_error_bound":1e-8,"curvature_envelope":curvature,
                "finest_FV_error_bound":8*curvature/128**2,
                "u_mean":float(np.trapezoid(values_mesh[0],mesh))}
        fixed = [solution(kind,points=n+1,fixed_mesh=True,**opts) for n in (32,64,128,256)]
        fixed_errors = [float(np.max(np.abs(s.sol(mesh)-fine.sol(mesh)))) for s in fixed]
        info["fixed_mesh_intervals"] = [32,64,128,256]
        info["fixed_mesh_errors_to_adaptive_reference"] = fixed_errors
        info["fixed_mesh_orders"] = np.log2(np.array(fixed_errors[:-1])/fixed_errors[1:]).tolist()
        if min(info["fixed_mesh_orders"][:2]) < 3.5:
            raise RuntimeError("Independent collocation refinement did not reach fourth order")
        if errors[1] >= 1e-8 or errors[0] <= 10*errors[1]:
            raise RuntimeError("Independent reference did not demonstrate convergence")
        metadata["cases"][kind]=info
        print(kind, json.dumps(info))

    # Parameterize the entire local branch by top temperature, not sigma.
    # The fold is the independently measured maximum of sigma(top_rise).
    branch=[]
    for rise in np.linspace(.05,4.0,40):
        sol=solution("poiseuille",top_rise=rise,tol=1e-8)
        branch.append((rise,float(sol.p[0])))
    peak=int(np.argmax(np.array(branch)[:,1]))
    def objective(rise,tol):
        return -solution("poiseuille",top_rise=rise,tol=tol).p[0]
    bounds=(branch[peak-1][0],branch[peak+1][0])
    coarse=minimize_scalar(objective,args=(1e-8,),bounds=bounds,method="bounded",
                           options={"xatol":1e-10})
    fine=minimize_scalar(objective,args=(1e-10,),bounds=bounds,method="bounded",
                         options={"xatol":1e-11})
    metadata["fold"]={"sigma":float(-fine.fun),"top_rise":float(fine.x),
                       "coarse_fine_sigma_difference":float(abs(fine.fun-coarse.fun)),
                       "sigma_error_bound":1e-7,
                       "nahme":float(-fine.fun*BETA*FORCE**2/(NU*ALPHA))}
    np.savetxt(OUT/"fold_branch.csv",np.array(branch),delimiter=",",
               header="top_rise,sigma",comments="",fmt="%.17g")
    (OUT/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print("fold",json.dumps(metadata["fold"]))


if __name__ == "__main__":
    main()
