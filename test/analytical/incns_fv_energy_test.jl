# T2 one-way energy contract, CPU Float64. Analytical thresholds are frozen
# before measurement (2026-11-08). The two-cell periodic channel avoids long
# CPU simulations while retaining the coupled four-field Newton equations.
# Run: sh tmp/thermal_julia.sh --project=. test/analytical/incns_fv_energy_test.jl
# Measured 2026-11-08, macOS arm64 / Julia 1.12.5: 430 checks in 15.0 s
# after package loading, including compilation; no optimisation flag needed.
using Test, KrakeNN, ForwardDiff, LinearAlgebra, Random, SparseArrays

const _FVE_BC = (W=:periodic, E=:periodic, S=:wall, N=:wall)
_fve_zero(x, y) = 0.0
_fve_temperature(p, U) = fv_fields(p, U)[4]
_fve_orders(errors) = log2.(errors[1:end-1] ./ errors[2:end])

function _fve_dense_jacobian(p, U)
    # Independent one-column seeds: no declared pattern or colouring is read.
    D = ForwardDiff.Dual{KrakeNN.FVJacobianTag,Float64,4}
    X = D.(U); R = similar(X); J = zeros(length(U), length(U))
    for col in eachindex(U)
        X[col] = D(U[col], ForwardDiff.Partials((1.0, 0.0, 0.0, 0.0)))
        fv_residual!(R, X, p)
        for row in eachindex(U)
            J[row, col] = ForwardDiff.partials(R[row])[1]
        end
        X[col] = D(U[col])
    end
    return J
end

function _fve_fluxes(p,U)
    g=p.grid; flux=Float64[]; axis=Int[]
    for j in 1:g.Ny,i in 1:g.Nx
        p.tag[i,j]==FV_CELL_SOLID && continue
        gp=KrakeNN._fv_cell_grad(U,p.tag,p.bvel,g,i,j)
        for d in (1,3)
            kind,ii,jj=KrakeNN._fv_face_kind(g,p.tag,i,j,d)
            kind==KrakeNN._FV_F_INT || continue
            gn=KrakeNN._fv_cell_grad(U,p.tag,p.bvel,g,ii,jj)
            up,vp,pp=ntuple(c->U[fv_index(g,i,j,c)],3)
            un,vn,pn=ntuple(c->U[fv_index(g,ii,jj,c)],3)
            grad=d==1 ? ((un-up)/g.h,(gp[2]+gn[2])/2,(vn-vp)/g.h,(gp[4]+gn[4])/2) :
                ((gp[1]+gn[1])/2,(un-up)/g.h,(gp[3]+gn[3])/2,(vn-vp)/g.h)
            nu=KrakeNN._fv_face_nu(p.visc,p.nu[i,j],p.nu[ii,jj],grad...,i,j,ii,jj)
            f=KrakeNN._fv_interior_mass_flux(g.h,g.conv,up,vp,pp,un,vn,pn,
                gp,gn,nu,d==1,1)
            push!(flux,f); push!(axis,d)
        end
    end
    return flux,axis
end

function _fve_channel(N; kind=:couette, source=0.0, sigma=0.4)
    nu, alpha, Tw, force = 0.3, 0.2, 1.0, 0.6
    top_speed = kind === :couette ? 1.0 : 0.0
    forcing = kind === :poiseuille ? force : 0.0
    p = fv_steady_problem(2, N; nu, bc=_FVE_BC,
        boundary_velocity=(side,x,y)->(side === :N ? top_speed : 0.0, 0.0),
        source=(x,y)->(forcing,0.0,0.0),
        energy=FVEnergy(; alpha, sigma, bottom=Tw, source))
    q = kind === :couette ? sigma*nu : sigma*force^2/(4nu)
    exact = kind === :couette ?
        ((x,y)->Tw + q/alpha*(y-y^2/2)) :
        ((x,y)->Tw + q/alpha*(y/3-y^2/2+2y^3/3-y^4/3))
    return (; p, nu, alpha, Tw, q, exact)
end

function _fve_heat_balance(p, U)
    g = p.grid; T = _fve_temperature(p, U)
    # Outward conductive heat through the halfway isothermal bottom face.
    # All other boundaries are adiabatic or periodic in these channel cases.
    outward = sum(2p.energy.alpha*(T[i,1]-1.0) for i in 1:g.Nx)
    heating = sum(KrakeNN._fv_energy_heating(U,p.tag,p.nu,p.bvel,g,i,j,
        p.visc,p.energy.sigma) for j in 1:g.Ny, i in 1:g.Nx)
    return abs(outward-heating) / max(abs(outward), abs(heating), eps())
end

# Streamfunction psi=A(x)b(y): u=psi_y, v=-psi_x. It and its gradient
# vanish at the square obstacle's faces and at the channel walls. Thus the
# same continuous solution obeys the voxel geometry exactly on every grid.
function _fve_mms_velocity(x, y)
    a = cospi(2x)^2
    b = cospi(2y)^2*sinpi(y)^2
    db = -2pi*sinpi(4y)*sinpi(y)^2 + pi*cospi(2y)^2*sinpi(2y)
    return a*db, 2pi*sinpi(4x)*b
end
_fve_mms_temperature(x,y) = 1.0 + (1+0.25cospi(4x))*sinpi(2y)^2
function _fve_mms_source(x,y,alpha)
    u,v = _fve_mms_velocity(x,y)
    tx = -pi*sinpi(4x)*sinpi(2y)^2
    ty = 2pi*(1+0.25cospi(4x))*sinpi(4y)
    lap = -4pi^2*cospi(4x)*sinpi(2y)^2 +
          8pi^2*(1+0.25cospi(4x))*cospi(4y)
    return u*tx+v*ty-alpha*lap
end

function _fve_mms(N, scheme; alpha=0.2, masked=true)
    solid = falses(N,N)
    masked && (solid[N÷4+1:3N÷4,N÷4+1:3N÷4] .= true)
    p = fv_steady_problem(N,N; nu=0.3, solid, bc=_FVE_BC,
        energy=FVEnergy(; alpha, bottom=1.0, scheme, T_ref=0.0,
            source=(x,y)->_fve_mms_source(x,y,alpha)))
    U = fv_state(p,(x,y)->_fve_mms_velocity(x,y)[1],
        (x,y)->_fve_mms_velocity(x,y)[2],_fve_zero,(x,y)->1.0)
    # Solid unknowns are storage identities, not conductive temperatures.
    for j in 1:N, i in 1:N
        solid[i,j] || continue
        for c in 1:4
            U[fv_index(p.grid,i,j,c)] = 0.0
        end
    end
    return p, U
end

function _fve_mms_errors(p, U)
    g=p.grid; T=_fve_temperature(p,U)
    exact=[_fve_mms_temperature((i-0.5)*g.h,(j-0.5)*g.h)
           for i in 1:g.Nx,j in 1:g.Ny]
    fluid=p.tag .!= FV_CELL_SOLID
    # Exclude every cell with a missing axial neighbour at radius one or two.
    # Traversal uses the production geometry, so periodic wrap stays included.
    # Full-fluid error is reported alongside this away-from-fallback measure.
    away=copy(fluid)
    for j in 1:g.Ny,i in 1:g.Nx,d in 1:4
        away[i,j] || continue
        kind,ii,jj=KrakeNN._fv_face_kind(g,p.tag,i,j,d)
        away[i,j]=kind==KrakeNN._FV_F_INT &&
            first(KrakeNN._fv_face_kind(g,p.tag,ii,jj,d))==KrakeNN._FV_F_INT
    end
    return (; away=norm((T-exact)[away])/sqrt(count(away)),
        full=norm((T-exact)[fluid])/sqrt(count(fluid)))
end

@testset "FV one-way energy (T2, CPU Float64)" begin
    @testset "uniform source: bottom Dirichlet / top adiabatic" begin
        alpha,q,Tw=0.2,0.12,1.0
        errors=Float64[]
        for N in (16,32,64)
            p=fv_steady_problem(2,N;bc=_FVE_BC,
                energy=FVEnergy(;alpha,bottom=Tw,source=q))
            U=fv_state(p,_fve_zero,_fve_zero,_fve_zero,(x,y)->Tw)
            r=fv_energy_solve(p,U)
            @test r.converged
            exact=[Tw+q/alpha*(y-y^2/2) for i in 1:2,
                y in ((j-0.5)/N for j in 1:N)]
            # Half-cell Dirichlet flux gives the exact quadratic plus the
            # spatially constant q*h^2/(8alpha), not pointwise exactness.
            offset=q/(8alpha*N^2)
            @test maximum(abs,r.T-exact.-offset) <= 2e-12
            push!(errors,maximum(abs,r.T-exact))
        end
        @test minimum(_fve_orders(errors)) >= 1.8
        @info "T2 uniform-source conduction" errors=Tuple(errors) order=Tuple(_fve_orders(errors))
    end

    @testset "coupled Couette and Poiseuille Brinkman heating" begin
        for kind in (:couette,:poiseuille)
            errors=Float64[]; balances=Float64[]
            scaled_residual=Float64[]; increments=Float64[]
            for N in (32,64,128)
                c=_fve_channel(N;kind)
                U0=fv_state(c.p,_fve_zero,_fve_zero,_fve_zero,(x,y)->c.Tw)
                r=solve_incns_fv_newton(;problem=c.p,U0,atol=1e-12,rtol=0.0,maxit=12)
                @test r.converged
                @test maximum(last(r.stats.equation_inf))<=1e-10
                @test last(r.stats.temperature_increment)<=1e-10*r.stats.thermal_scales.deltaT
                @test r.stats.thermal_scales==KrakeNN._fv_energy_scales(c.p)
                push!(scaled_residual,maximum(last(r.stats.equation_inf)))
                push!(increments,last(r.stats.temperature_increment))
                T=_fve_temperature(c.p,r.U)
                exact=[c.exact((i-0.5)/N,(j-0.5)/N) for i in 1:2,j in 1:N]
                push!(errors,maximum(abs,T-exact))
                push!(balances,_fve_heat_balance(c.p,r.U))
                @test balances[end] <= 1e-10
                if N==128
                    # Before measurement: the Couette half-cell error is
                    # q*h^2/(8alpha). A factor 2 allows floating-point error.
                    # For the quartic Poiseuille profile, |T''''|=8q/alpha;
                    # the centred-difference defect and both wall defects
                    # are bounded conservatively by q*h^2/alpha.
                    bound=(kind===:couette ? 0.25 : 1.0)*c.q/(c.alpha*N^2)
                    @test errors[end] <= bound
                end
            end
            @test minimum(_fve_orders(errors)) >= 1.8
            @info "T2 coupled heating" kind errors=Tuple(errors) order=Tuple(_fve_orders(errors)) heat_balance=Tuple(balances) scaled_residual=Tuple(scaled_residual) final_temperature_increment=Tuple(increments)
        end
    end

    @testset "manufactured advection: solved temperature, periodic and masked" begin
        for masked in (false,true), scheme in (:linear_upwind,:upwind)
            errors=Float64[]; full=Float64[]
            for N in (16,32,64)
                p,U=_fve_mms(N,scheme;masked)
                r=fv_energy_solve(p,U)
                @test r.converged
                e=_fve_mms_errors(p,r.U)
                push!(errors,e.away); push!(full,e.full)
                @test r.U[1:4:end] == U[1:4:end]
                @test r.U[2:4:end] == U[2:4:end]
                @test r.U[3:4:end] == U[3:4:end]
            end
            gate=scheme===:linear_upwind ? 1.8 : 0.9
            @test minimum(_fve_orders(errors)) >= gate
            @info "T2 manufactured solved temperature" masked scheme errors=Tuple(errors) full=Tuple(full) order=Tuple(_fve_orders(errors))
        end
    end

    @testset "linear upwind wraps and falls back at solid faces" begin
        solid=falses(5,5); solid[2,3]=true; solid[3,2]=true
        for scheme in (:linear_upwind,:upwind)
            p=fv_steady_problem(5,5;solid,bc=_FVE_BC,
                energy=FVEnergy(;alpha=0.2,bottom=1.0,scheme))
            U=fv_state(p,_fve_zero,_fve_zero,_fve_zero,(x,y)->x+3y+0.2x*y)
            value(i,j)=U[fv_index(p.grid,i,j,4)]
            face(i,j,d,ii,jj)=KrakeNN._fv_energy_face_temperature(
                U,p.tag,p.grid,i,j,d,ii,jj,p.energy.scheme)
            for (i,j,d,ii,jj) in ((3,3,1,4,3),(3,3,3,3,4))
                positive,negative=face(i,j,d,ii,jj)
                @test positive==value(i,j) # upstream cell is solid
                nexti,nextj=d==1 ? (5,3) : (3,5)
                expected=scheme===:upwind ? value(ii,jj) :
                    1.5value(ii,jj)-0.5value(nexti,nextj)
                @test negative≈expected atol=2e-15 rtol=0
                reverse=face(ii,jj,d==1 ? 2 : 4,i,j)
                @test reverse[2]==value(i,j) # reversed inflow has same fallback
            end
            positive,negative=face(5,1,1,1,1)
            @test positive≈(scheme===:upwind ? value(5,1) : 1.5value(5,1)-0.5value(4,1)) atol=2e-15 rtol=0
            @test negative≈(scheme===:upwind ? value(1,1) : 1.5value(1,1)-0.5value(2,1)) atol=2e-15 rtol=0
        end
    end

    @testset "constant temperature equals unpinned mass residual" begin
        rng=MersenneTwister(208)
        for scheme in (:linear_upwind,:upwind)
            p=fv_steady_problem(9,7;bc=_FVE_BC,
                energy=FVEnergy(;alpha=0.2,bottom=2.5,scheme,T_ref=0.0))
            U=randn(rng,fv_nunknowns(p)); U[4:4:end].=2.5
            tags=copy(p.tag); tags[tags.==FV_CELL_PINNED].=FV_CELL_FLUID
            R=fv_residual(U,p); mass=Float64[]; differences=Float64[]
            for j in 1:p.grid.Ny,i in 1:p.grid.Nx
                rc=KrakeNN.fv_cell_residual(U,tags,p.nu,p.bvel,p.src,p.grid,i,j,p.visc)[3]
                push!(mass,rc)
                push!(differences,abs(R[fv_index(p.grid,i,j,4)]-2.5rc))
                @test differences[end] <= 1e-12
            end
            @test maximum(abs,mass)>1e-3 # deliberately not a converged flow
            @info "T2 constant-temperature conservation" scheme max_difference=maximum(differences) continuity_maximum=maximum(abs,mass)
        end
    end

    @testset "thermally isolated fluid pockets are rejected" begin
        solid=falses(7,7); solid[:,4].=true
        for source in (0.0,1.0)
            @test_throws ArgumentError fv_steady_problem(7,7;solid,bc=_FVE_BC,
                energy=FVEnergy(;alpha=0.2,bottom=1.0,source))
        end
    end

    @testset "configuration and frozen thermal scales" begin
        for alpha in (0.0,-1.0,Inf,NaN)
            @test_throws ArgumentError FVEnergy(;alpha,bottom=1.0)
        end
        @test_throws ArgumentError FVEnergy(;alpha=0.2,sigma=-1.0)
        @test_throws ArgumentError FVEnergy(;alpha=0.2,scheme=:central)
        @test_throws ArgumentError FVEnergy(;alpha=0.2,bottom=Inf)
        @test_throws ArgumentError fv_steady_problem(4,4;bc=(W=:inlet,E=:outlet,S=:wall,N=:wall),
            energy=FVEnergy(;alpha=0.2,bottom=1.0))
        @test_throws ArgumentError fv_steady_problem(4,4;bc=_FVE_BC,
            energy=FVEnergy(;alpha=0.2,bottom=1.0,left=1.0))
        @test_throws DimensionMismatch fv_steady_problem(4,4;bc=_FVE_BC,
            energy=FVEnergy(;alpha=0.2,bottom=1.0,source=zeros(3,4)))
        p=fv_steady_problem(2,8;bc=_FVE_BC,
            energy=FVEnergy(;alpha=0.2,bottom=1.0,source=0.1))
        U=fv_state(p,_fve_zero,_fve_zero,_fve_zero,(x,y)->1.0)
        s=fv_energy_solver(p,U); scales=s.scales
        @test all(>(0),scales.equations)
        T=fill(5.0,16); stats=fv_energy_solve!(s,T)
        @test stats.converged
        @test stats.thermal_scales===scales
        @test last(stats.history)<=1e-10
        @test last(stats.temperature_increment)<=1e-10*scales.deltaT
        @test s.scales===scales
        # Constitutive regularisation cannot manufacture physical heating at
        # rest: S:S is exactly zero even when viscosity uses nonzero epsilon.
        regularised=fv_steady_problem(3,4;bc=_FVE_BC,
            rheology=CarreauYasuda(0.3,0.05,1.2,2.0,0.7),gnf_eps=0.2,
            energy=FVEnergy(;alpha=0.2,sigma=3.0,bottom=1.0))
        rest=fv_state(regularised,_fve_zero,_fve_zero,_fve_zero,(x,y)->1.0)
        @test iszero(fv_residual(rest,regularised))
        @test iszero(KrakeNN._fv_energy_heating(rest,regularised.tag,regularised.nu,
            regularised.bvel,regularised.grid,2,2,regularised.visc,3.0))
        # Continuation keeps the final heating scale even at sigma=0. The
        # physical source uses sigma, so nominal heating cannot warm a cell.
        for nominal in (-1.0,0.5,Inf,NaN)
            @test_throws ArgumentError FVEnergy(;alpha=0.2,sigma=1.0,sigma_nom=nominal)
        end
        channel_options=(;nu=0.3,bc=_FVE_BC,
            boundary_velocity=(side,x,y)->(side===:N ? 1.0 : 0.0,0.0))
        initial=fv_steady_problem(2,4;channel_options...,
            energy=FVEnergy(;alpha=0.2,sigma=0.0,sigma_nom=10.0,bottom=1.0))
        nominal=fv_steady_problem(2,4;channel_options...,
            energy=FVEnergy(;alpha=0.2,sigma=10.0,bottom=1.0))
        @test KrakeNN._fv_energy_scales(initial)==KrakeNN._fv_energy_scales(nominal)
        @test KrakeNN._fv_energy_scales(initial).deltaT≈15.0
        @test fv_adapt(Array,initial).energy.sigma_nom==10.0
        shear=fv_state(initial,(x,y)->y,_fve_zero,_fve_zero,(x,y)->1.0)
        @test iszero(fv_residual(shear,initial)[4:4:end])
        @test maximum(abs,fv_residual(shear,nominal)[4:4:end])>0.0
        @info "T2 nominal continuation scales" deltaT=KrakeNN._fv_energy_scales(initial).deltaT zero_sigma_heating=maximum(abs,fv_residual(shear,initial)[4:4:end])

        # Thermal parameter adjoints belong to T4. A caller-provided workspace
        # must not bypass the explicit unsupported-capability guard.
        thermal_solver=fv_newton_solver(initial)
        @test_throws ArgumentError fv_adjoint_workspace(thermal_solver,FVUniformViscosity())
        old_solver=fv_newton_solver(fv_steady_problem(2,4;channel_options...))
        old_workspace=fv_adjoint_workspace(old_solver,FVUniformViscosity())
        @test_throws ArgumentError fv_steady_adjoint(thermal_solver,shear,zeros(length(shear));
            params=FVUniformViscosity(),θ=[0.3],ws=old_workspace)
        @test_throws ArgumentError fv_steady_gradient(thermal_solver,shear,[0.3],
            FVLinearFunctional([1],[1.0]);params=FVUniformViscosity(),ws=old_workspace)
        @test_throws ArgumentError fv_steady_tangent(thermal_solver,shear,[0.3],[1.0];
            params=FVUniformViscosity(),ws=old_workspace)
    end

    @testset "coloured / independent dual Jacobian and fixed-sign FD" begin
        rng=MersenneTwister(11682)
        solid=falses(9,7); solid[4:5,3:4].=true; solid[[9,1],4:5].=true
        law=LearnedViscosity(TinyMLP{2,2,1}();nu_ref=0.3,monotone=false)
        theta=[0.3,-0.2,0.4,-0.3,0.1,-0.15,0.2,-0.15,0.04]
        field=[0.13i+0.09j+0.03sin(i*j) for i in 1:9,j in 1:7]
        models=((:newtonian,(;nu=0.3)),
            (:gnf,(;rheology=CarreauYasuda(0.3,0.05,1.2,2.0,0.7),gnf_eps=0.02)),
            (:law,(;viscosity=FVLawViscosity(law,theta,(field,);eps=0.02))))
        for (model,options) in models, scheme in (:linear_upwind,:upwind)
            p=fv_steady_problem(9,7;solid,bc=_FVE_BC,options...,
                energy=FVEnergy(;alpha=0.2,sigma=0.3,bottom=1.0,scheme))
            base=0.3randn(rng,fv_nunknowns(p)); base[3:4:end].*=0.1
            base[4:4:end].+=2.0
            S=Matrix(fv_jacobian_pattern(p)); solver=fv_newton_solver(p)
            for direction in (1.0,-1.0)
                U=copy(base); U[1:4:end].*=direction; U[2:4:end].*=direction
                U[3:4:end].*=direction
                flux,axis=_fve_fluxes(p,U)
                for d in (1,3)
                    @test minimum(flux[axis.==d])<0<maximum(flux[axis.==d])
                end
                Jref=_fve_dense_jacobian(p,U); J=Matrix(fv_jacobian!(solver,U))
                scale=max(1.0,maximum(abs,Jref))
                missing=findall((.!iszero.(Jref)).&.!S)
                @test isempty(missing)
                @test maximum(abs,J-Jref)<=1e-12*scale
                v=randn(rng,length(U)); v./=norm(v,Inf)
                delta=Float64[]; relative=Float64[]
                for h in (1e-4,1e-5,1e-6,1e-7)
                    @test sign.(first(_fve_fluxes(p,U+h*v)))==sign.(flux)
                    @test sign.(first(_fve_fluxes(p,U-h*v)))==sign.(flux)
                    fd=(fv_residual(U+h*v,p)-fv_residual(U-h*v,p))/(2h)
                    push!(delta,norm(J*v-fd,Inf))
                    push!(relative,delta[end]/max(norm(fd,Inf),eps()))
                end
                # Central FD truncation+roundoff gate, fixed before runs.
                @test any(delta .<= 2e-8 .+ 2e-7*norm(J*v,Inf))
                @info "T2 thermal Jacobian" model scheme direction missing=length(missing) max_dual_difference=maximum(abs,J-Jref) fd_absolute=Tuple(delta) fd_relative=Tuple(relative)
            end
        end
    end

    @testset "cell Peclet 5 and 10: temperature undershoot report" begin
        N=32
        speed=maximum(maximum(abs,_fve_mms_velocity((i-0.5)/N,(j-0.5)/N))
            for i in 1:N,j in 1:N)
        for pe in (5.0,10.0), scheme in (:linear_upwind,:upwind)
            p,U=_fve_mms(N,scheme;alpha=speed/(N*pe),masked=true)
            r=fv_energy_solve(p,U)
            @test r.converged
            fluid=p.tag.!=FV_CELL_SOLID
            @info "T2 cell Peclet undershoot (report, no monotonicity gate)" pe scheme minimum_temperature_above_wall=minimum(r.T[fluid].-1.0)
        end
    end
end
