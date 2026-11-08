# T5b: the energy unknown is theta = T - T_ref, so absolute temperatures of a
# few hundred kelvin do not set a round-off floor (6e-14 K spacing at 468 K gives
# a scaled residual ~1e-10 on a 24x48 grid) and the convective term carries no
# T_ref * continuity residual. CPU Float64. Gates frozen 2026-11-08, before
# measurement: a +400 K shift changes T - 400 by <= 1e-11 K, the Newton
# iteration count is identical, and the coupled solve at 468 K meets the
# production gate (scaled residual <= 1e-10, increment <= 1e-10 DeltaT).
# Run: sh tmp/thermal_julia.sh --project=. test/analytical/incns_fv_energy_theta_test.jl
using Test, Kraken

struct _FVthExpLaw <: AbstractScalarLaw end
Kraken.law_ninputs(::_FVthExpLaw)=2
Kraken.law_nparams(::_FVthExpLaw)=3
@inline Kraken.law_value(::_FVthExpLaw,x::Tuple,theta::AbstractVector)=
    theta[1]*exp(-theta[2]*(x[2]-theta[3]))

const _FVth_BC=(W=:periodic,E=:periodic,S=:wall,N=:wall)

# Channel with a solid block, convection on, temperature-dependent viscosity.
# `shift` moves every temperature (walls, law reference, initial state).
function _fvθ_exp(shift;N=24,T_ref=nothing)
    solid=falses(2N,N);solid[N-2:N+2,N÷2-3:N÷2+3].=true
    visc=FVLawViscosity(_FVthExpLaw(),[.3,1.,1.0+shift],(FVStateTemperature(),);eps=.02)
    kw=T_ref===nothing ? (;) : (;T_ref=T_ref+shift)
    p=fv_steady_problem(2N,N;viscosity=visc,bc=_FVth_BC,conv=true,solid,
        source=(x,y)->(.6,0.,0.),
        energy=FVEnergy(alpha=.2,sigma=.1,bottom=1.0+shift,top=1.0+shift;kw...))
    U0=fv_state(p,(x,y)->0.,(x,y)->0.,(x,y)->0.,(x,y)->1.0+shift)
    return p,solve_incns_fv_newton(problem=p,U0=U0,maxit=30)
end

# Same, with the E5 thermal law at its physical temperature (468 K).
function _fvθ_e5(;T_ref=nothing,N=24)
    theta=[2000.,.4,.035,.30,.006,1.5,465.15,2.5]
    law=E5ThermalLaw(theta;T_wall=468.15,nu_ref=.3)
    solid=falses(2N,N);solid[N-2:N+2,N÷2-3:N÷2+3].=true
    visc=FVLawViscosity(law,theta,(FVStateTemperature(),);eps=.02)
    kw=T_ref===nothing ? (;) : (;T_ref)
    p=fv_steady_problem(2N,N;viscosity=visc,bc=_FVth_BC,conv=true,solid,
        source=(x,y)->(.6,0.,0.),
        energy=FVEnergy(alpha=.2,sigma=.1,bottom=468.15,top=468.15;kw...))
    return p,solve_incns_fv_newton(problem=p,maxit=30)
end

@testset "FV energy unknown theta = T - T_ref (T5b, CPU Float64)" begin
    @testset "shift invariance: all temperatures +400 K" begin
        p0,r0=_fvθ_exp(0.0);p1,r1=_fvθ_exp(400.0)
        @test r0.converged && r1.converged
        @test r0.iterations==r1.iterations
        fluid=p0.tag.!=FV_CELL_SOLID
        dT=maximum(abs,((r1.T.-400).-r0.T)[fluid])
        du=maximum(abs,r1.u.-r0.u)
        @test dT<=1e-11
        @test du<=1e-11
        # Solid storage rows are returned as stored, unshifted.
        @test all(iszero,r1.T[.!fluid])
        # An explicit T_ref different from the wall temperature changes nothing.
        p2,r2=_fvθ_exp(400.0;T_ref=0.37)
        @test r2.converged && r2.iterations==r1.iterations
        @test maximum(abs,(r2.T.-r1.T)[fluid])<=1e-11
        @test p1.energy.Tref==401.0 && p2.energy.Tref==400.37
        @info "T5b shift invariance" iterations=(r0.iterations,r1.iterations,r2.iterations) max_dT=dT max_du=du scaled_residual=(maximum(last(r0.stats.equation_inf)),maximum(last(r1.stats.equation_inf)))
    end

    @testset "coupled solve with solids at 468 K converges" begin
        p,r=_fvθ_e5()
        @test r.converged
        @test maximum(last(r.stats.equation_inf))<=1e-10
        @test last(r.stats.temperature_increment)<=1e-10*r.stats.thermal_scales.deltaT
        fluid=p.tag.!=FV_CELL_SOLID
        @test all(r.T[fluid].>=468.15-1e-9)   # heating only warms the fluid
        @test maximum(r.T[fluid])>468.15+1e-3
        # The raw state holds theta; the public temperature is absolute.
        @test r.U[4:4:end][fluid[:]][1]≈r.T[fluid][1]-p.energy.Tref atol=1e-12
        # Before: T_ref=0 is the absolute-temperature unknown; it stalls at the
        # 468 K round-off floor (reported, not gated, because the floor sits at
        # the gate itself).
        pa,ra=_fvθ_e5(;T_ref=0.0)
        @info "T5b 468 K coupled solid case" theta_converged=r.converged theta_scaled_residual=maximum(last(r.stats.equation_inf)) absolute_converged=ra.converged absolute_scaled_residual=maximum(last(ra.stats.equation_inf)) iterations=(r.iterations,ra.iterations)
    end
end
