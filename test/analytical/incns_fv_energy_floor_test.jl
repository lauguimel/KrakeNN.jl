# T5d: round-off-aware stopping gate of the FV energy equation (CPU Float64).
# Gate: increment <= 1e-10 DeltaT AND |R_c| <= floor_c in every thermal row, with
# floor_c = C * eps * sum|terms of row c|, C = 64 fixed before any measurement
# (FV_ENERGY_FLOOR_C, never tuned). The scaled tolerance gate is tried first.
# Frozen gates 2026-11-08: shift-equivalence <= 1e-11 K; Newton iterations <= 6
# with unit step lengths (no line-search collapse); a state 1e-6 off the solution
# is >= 1e3 times its floor.
# Run: sh tmp/thermal_julia.sh --project=. test/analytical/incns_fv_energy_floor_test.jl
using Test, KrakeNN

const _FVfl_BC=(W=:periodic,E=:periodic,S=:wall,N=:wall)

# E5-like coupled case with solids; T_ref=0 makes the unknown the absolute
# temperature (468 K), where the scaled tolerance gate sits on the round-off floor.
function _fvfl_e5(;T_ref=nothing,N=24,sigma=.1)
    theta=[2000.,.4,.035,.30,.006,1.5,465.15,2.5]
    law=E5ThermalLaw(theta;T_wall=468.15,nu_ref=.3)
    solid=falses(2N,N);solid[N-2:N+2,N÷2-3:N÷2+3].=true
    visc=FVLawViscosity(law,theta,(FVStateTemperature(),);eps=.02)
    kw=T_ref===nothing ? (;) : (;T_ref)
    p=fv_steady_problem(2N,N;viscosity=visc,bc=_FVfl_BC,conv=true,solid,
        source=(x,y)->(.6,0.,0.),
        energy=FVEnergy(alpha=.2,sigma=sigma,bottom=468.15,top=468.15;kw...))
    return p
end

function _fvfl_floor(p,U)
    Fl=zeros(length(U))
    KrakeNN._fv_energy_roundoff_floor!(Fl,U,p)
    return Fl
end

@testset "FV energy round-off floor gate (T5d, CPU Float64)" begin
    @testset "floor is nonnegative; zero for a zero state with zero source" begin
        N=12
        p0=fv_steady_problem(2N,N;nu=0.3,bc=_FVfl_BC,conv=true,
            energy=FVEnergy(alpha=.2,sigma=0.,bottom=0.,top=0.))
        U0=zeros(fv_nunknowns(p0))
        @test p0.energy.Tref==0.0
        F0=_fvfl_floor(p0,U0)
        @test all(iszero,F0)
        p=_fvfl_e5(;N)
        r=solve_incns_fv_newton(problem=p,maxit=30)
        F=_fvfl_floor(p,Array(r.U))
        fluid=p.tag.!=FV_CELL_SOLID
        @test all(>=(0),F)
        @test all(>(0),F[4:4:end][fluid[:]])        # active rows have a positive floor
        @test all(iszero,F[4:4:end][.!fluid[:]])    # solid storage rows: 0
        # The magnitude sum is the residual's term list: the signed sum cannot exceed it.
        R=fv_residual(Array(r.U),p)
        @test all(abs.(R[4:4:end]) .<= F[4:4:end] .+ 1e-300)
    end

    @testset "high absolute T: tolerance gate cannot pass, floor gate does" begin
        # Same physics, unknown theta (T_ref = wall) and unknown T (T_ref = 0).
        p=_fvfl_e5();     r=solve_incns_fv_newton(problem=p,maxit=30)
        pa=_fvfl_e5(;T_ref=0.0); ra=solve_incns_fv_newton(problem=pa,maxit=30)
        fluid=p.tag.!=FV_CELL_SOLID
        @test r.converged && r.stats.gate==:tolerance
        @test ra.converged && ra.stats.gate==:roundoff_floor
        # the tolerance gate alone is not met by the absolute-T solve
        @test maximum(last(ra.stats.equation_inf))>1e-10
        # bounded iteration count, no line-search collapse
        @test ra.iterations<=6
        @test all(==(1.0),ra.stats.alpha)
        @test ra.stats.failure_reason==:none
        # the solution is the T_ref-shifted reference
        @test maximum(abs,(ra.T.-r.T)[fluid])<=1e-11
        @test maximum(abs,ra.u.-r.u)<=1e-11
        fc=ra.stats.energy_floor
        @test fc.passed && fc.max_ratio<=1
        @info "T5d absolute-T case" gate=ra.stats.gate iterations=(r.iterations,ra.iterations) scaled_inf=maximum(last(ra.stats.equation_inf)) floor_ratio=fc.max_ratio dT=maximum(abs,(ra.T.-r.T)[fluid])
    end

    @testset "floor must not pass when the residual is far above round-off" begin
        p=_fvfl_e5()
        # (a) one Newton step from the far isothermal state: not converged, residual >> floor
        r1=solve_incns_fv_newton(problem=p,maxit=1)
        @test !r1.converged && r1.stats.gate==:none
        @test !r1.stats.energy_floor.passed && r1.stats.energy_floor.max_ratio>1e3
        # (b) the converged state moved by 1e-6 K in one cell: gate :none through the floor test
        r=solve_incns_fv_newton(problem=p,maxit=30)
        U=Array(r.U); k=findfirst(i->p.tag[i]!=FV_CELL_SOLID,eachindex(p.tag))
        U[KrakeNN._fv_cell_index(p.grid,k,4)]+=1e-6
        R=fv_residual(U,p); Fl=_fvfl_floor(p,U)
        nb=KrakeNN._FVNormBuffers(Array,p.grid.Nx*p.grid.Ny;layout=p.grid.layout)
        chk=KrakeNN._fv_energy_floor_check(nb,R,Fl,p.grid.layout)
        @test !chk.passed && chk.max_ratio>1e3
        # (c) increment criterion still required: a restart from the converged state is accepted at once
        s=fv_newton_solver(p); V=copy(Array(r.U))
        st=fv_newton!(s,V;maxit=30)
        @test st.converged && st.iterations==0 && st.gate in (:tolerance,:roundoff_floor)
    end

    @testset "type stability and report plumbing" begin
        p=_fvfl_e5(;N=12)
        r=solve_incns_fv_newton(problem=p,maxit=30)
        U=Array(r.U); Fl=zeros(length(U)); R=fv_residual(U,p)
        nb=KrakeNN._FVNormBuffers(Array,p.grid.Nx*p.grid.Ny;layout=p.grid.layout)
        @inferred KrakeNN._fv_energy_roundoff_floor!(Fl,U,p)
        @inferred KrakeNN._fv_energy_floor_check(nb,R,Fl,p.grid.layout)
        energy,src=KrakeNN._fv_energy_args(p.energy)
        @inferred KrakeNN._fv_energy_cell(U,p.tag,p.nu,p.bvel,p.grid,3,3,p.visc,energy,src,Val(true))
        @inferred KrakeNN._fv_energy_cell(U,p.tag,p.nu,p.bvel,p.grid,3,3,p.visc,energy,src)
        # sum mode is the residual, bit for bit
        @test KrakeNN._fv_energy_cell(U,p.tag,p.nu,p.bvel,p.grid,3,3,p.visc,energy,src)==R[KrakeNN._fv_cell_index(p.grid,3+2*p.grid.Nx,4)]
        # continuation reports the gate of every step
        pc=_fvfl_e5(;N=12)
        c=fv_thermal_continuation(pc)
        @test c.converged
        @test length(c.gates)==length(c.history)
        @test all(g->g in (:tolerance,:roundoff_floor),c.gates)
        @test all(a->a.gate in (:tolerance,:roundoff_floor,:none),c.attempts)
        @test all(h->h.gate==h.stats.gate,c.history)
        # energy-only solve
        u,v,pr,T=fv_fields(p,U)
        es=fv_energy_solve(p,U)
        @test es.converged && es.stats.gate in (:tolerance,:roundoff_floor)
        @test es.stats.energy_floor.passed
    end
end
