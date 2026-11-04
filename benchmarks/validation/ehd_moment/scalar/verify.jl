println("IMPORT_BEGIN ",time()); flush(stdout)
using Kraken, CUDA, KernelAbstractions, LinearAlgebra, TOML, DelimitedFiles, Test, Serialization
println("IMPORT_END ",time()); flush(stdout)
include("reference_kernels.jl")
using .ExternalMoment2D
BLAS.set_num_threads(1); CUDA.allowscalar(false)
const SOURCE_SHA=strip(read("SOURCE_SHA",String))
@assert realpath(pathof(Kraken))==realpath("source/src/Kraken.jl")
@assert readchomp(`git -C source rev-parse HEAD`)==SOURCE_SHA
@assert VERSION==v"1.11.9" && Base.pkgversion(CUDA)==v"5.11.3"
@assert CUDA.functional()
println("SOURCE ",pathof(Kraken)," SHA ",SOURCE_SHA); CUDA.versioninfo()
save(path,d)=open(io->TOML.print(io,d),path,"w")
function ek()
    lo=pi/2; hi=pi
    for _=1:60
        k=(lo+hi)/2
        if k*cos(k)+sin(k)>0; lo=k; else; hi=k; end
    end
    (lo+hi)/2
end
function exact(H,t,periodic)
    Nx=periodic ? H : H+1; Ny=H+1
    q=zeros(Nx,Ny); qx=similar(q); qy=similar(q)
    k=periodic ? pi : ek(); kx=periodic ? 2pi : pi
    lam=.05*(kx*kx+k*k+1)
    for j=1:Ny,i=1:Nx
        x=(i-1)/H; y=(j-1)/H; fac=.05*exp(y-lam*t)
        q[i,j]=1+fac*sin(k*y)*cos(kx*x)
        qx[i,j]=-fac*kx*sin(k*y)*sin(kx*x)
        qy[i,j]=fac*(sin(k*y)+k*cos(k*y))*cos(kx*x)
    end
    q,qx,qy
end
function boundary!(f,q,vx,vy,periodic,backend;moment=true)
    Nx,Ny=size(q)
    # Actual proposed src/ implementation. K=0 makes Ex/Ey unused; vx/vy are
    # prescribed total charge velocities in this exact scalar verification.
    Kraken._apply_charge_moment_box_2d!(f,q,vx,vy,vx,vy,1.,0.,Nx,Ny;
        periodic=periodic,moment=moment)
end
function simulate(H,periodic,backend;tau=.8,lowtau=false,moment=true,tend=.2)
    dt=(tau-.5)/3/H^2/.05
    steps=lowtau ? 256 : round(Int,tend/dt)
    q,qx,qy=exact(H,0.,periodic); Nx,Ny=size(q)
    vx=zeros(size(q)); vy=fill(.1*dt*H,size(q))
    if lowtau
        for j=1:Ny,i=1:Nx
            vx[i,j]=.002*sin(pi*(j-1)/H)*cos(pi*(i-1)/H)
            vy[i,j]=.006
        end
    end
    f=zeros(Nx,Ny,9)
    for k=1:9,j=1:Ny,i=1:Nx
        f[i,j,k]=ExternalMoment2D.equilibrium(k,q[i,j],vx[i,j],vy[i,j])-tau*ExternalMoment2D.W[k]*(ExternalMoment2D.CX[k]*qx[i,j]+ExternalMoment2D.CY[k]*qy[i,j])/H
    end
    move(a)=backend isa CPU ? copy(a) : CuArray(a)
    f=move(f); out=similar(f); qd=move(q); xd=move(vx); yd=move(vy); zeroes=move(zeros(size(q)))
    t0=time_ns(); cpu_started=time()
    for step=1:steps
        # Same production collide/stream operators as EC; v is prescribed here.
        Kraken.collide_electric_charge_regularized_2d!(f,xd,yd,zeroes,zeroes,tau,1.)
        if periodic
            Kraken.stream_periodic_x_wall_y_2d!(out,f,Nx,Ny)
        else
            Kraken.stream_wall_x_wall_y_2d!(out,f,Nx,Ny)
        end
        Kraken.compute_ehd_scalar_2d!(qd,out)
        boundary!(out,qd,xd,yd,periodic,backend;moment=moment)
        f,out=out,f
        if step%32==0 || step==steps
            hq=Array(qd)
            @assert all(isfinite,hq) && minimum(hq)>0
            backend isa CPU && time()-cpu_started>30 && error("CPU numerical execution exceeded 30 seconds")
        end
    end
    KernelAbstractions.synchronize(backend)
    elapsed=(time_ns()-t0)*1e-9; q=Array(qd)
    reference,_,_=exact(H,steps*dt,periodic)
    (q=q,f=Array(f),reference=reference,seconds=elapsed,steps=steps,t=steps*dt)
end
function emit(label,H,r,periodic)
    fname="raw/$(label)_H$(H).csv"
    open(fname,"w") do io
        println(io,"i,j,q,reference")
        for j=1:size(r.q,2),i=1:size(r.q,1)
            println(io,"$i,$j,$(r.q[i,j]),$(r.reference[i,j])")
        end
    end
    refnorm=norm(r.reference.-1)
    err=r.q.-r.reference
    maxref=maximum(abs.(r.reference.-1))
    corners=err[[1,end],[1,end]]
    Dict("H"=>H,"label"=>label,"L2"=>norm(err)/refnorm,
         "Linf"=>maximum(abs.(err))/maximum(abs.(r.reference.-1)),
         "corner_Linf"=>maximum(abs.(corners))/maxref,
         "injector_Linf"=>maximum(abs.(err[2:end-1,1]))/maxref,
         "collector_Linf"=>maximum(abs.(err[2:end-1,end]))/maxref,
         "sides_Linf"=>maximum(abs.(err[[1,end],2:end-1]))/maxref,
         "interior_Linf"=>maximum(abs.(err[2:end-1,2:end-1]))/maxref,
         "seconds"=>r.seconds,"steps"=>r.steps,"time"=>r.t,"minimum_q"=>minimum(r.q))
end
function corner_probe(backend,moment)
    Nx,Ny=1025,17
    f0=reshape([.02+.001*sin(k/37) for k=1:Nx*Ny*9],Nx,Ny,9)
    q0=dropdims(sum(f0;dims=3);dims=3)
    vx=[.002*sin(i/51)*sin((j-1)*pi/(Ny-1)) for i=1:Nx,j=1:Ny]
    vy=fill(.006,Nx,Ny)
    move(a)=backend isa CPU ? copy(a) : CuArray(a)
    f,q,x,y=move(f0),move(q0),move(vx),move(vy)
    boundary!(f,q,x,y,false,backend;moment=moment)
    KernelAbstractions.synchronize(backend)
    a=Array(f); charge=Array(q)
    @test maximum(abs.(charge[:,1].-1))<=2e-14
    # Scalar corner reference reads only the completed adjacent electrode column.
    for i in (1,Nx),j in (1,Ny),k in 1:9
        n=i==1 ? 2 : Nx-1
        want=a[n,j,k]+ExternalMoment2D.equilibrium(k,charge[n,j],vx[i,j],vy[i,j])-ExternalMoment2D.equilibrium(k,charge[n,j],vx[n,j],vy[n,j])
        @test abs(a[i,j,k]-want)<=2e-14
    end
    (q=charge,f=a,corners=charge[[1,end],[1,end]])
end
if get(ENV,"CASE_STAGE","")=="single"
    @assert Threads.nthreads()==1
    @testset "Single-thread ordered corner controls" begin
        for moment in (false,true)
            r=corner_probe(CPU(),moment)
            for rep=1:2
                rr=corner_probe(CPU(),moment)
                @test r.q==rr.q && r.f==rr.f
            end
            serialize("raw/single_$(moment).jls",r)
        end
    end
    save("raw/single_status.toml",Dict("status"=>"PASS","threads"=>1,"Nx"=>1025,"Ny"=>17))
    exit()
end
@assert Threads.nthreads()==2
results=Dict{String,Any}()
try
    @testset "Ordered corners: threads and repeated GPU execution" begin
        for moment in (false,true)
            one=deserialize("raw/single_$(moment).jls")
            cpu=corner_probe(CPU(),moment); gpu=corner_probe(CUDA.CUDABackend(),moment)
            for r in (cpu,gpu)
                @test maximum(abs.(r.q-one.q))<=1e-11
                @test maximum(abs.(r.f-one.f))<=1e-11
                @test maximum(abs.(r.corners-one.corners))<=1e-11
            end
            for rep=1:2
                cp=corner_probe(CPU(),moment); gp=corner_probe(CUDA.CUDABackend(),moment)
                @test cp.f==cpu.f && gp.f==gpu.f
            end
            save("raw/corners_$(moment).toml",Dict("status"=>"PASS","Nx"=>1025,"Ny"=>17,
                "gpu_cpu_q_difference"=>maximum(abs.(gpu.q-cpu.q)),
                "gpu_cpu_f_difference"=>maximum(abs.(gpu.f-cpu.f)),
                "corner_values"=>vec(gpu.corners)))
        end
    end
    @testset "Integrated two-moment boundary: production operators CPU/CUDA" begin
        for periodic in (true,false)
            label=periodic ? "periodic_DD" : "box_DN"
            cpu=simulate(16,periodic,CPU())
            for H in (16,32,64)
                gpu=simulate(H,periodic,CUDA.CUDABackend())
                if H==16
                    @test maximum(abs.(cpu.q.-gpu.q))<=1e-11
                    @test maximum(abs.(cpu.f.-gpu.f))<=1e-11
                end
                # Independent NumPy data, retained before Julia execution.
                tab=readdlm("reference/$(label)_H$(H).csv",',',Float64;skipstart=1)
                npq=reshape(tab[:,1],size(gpu.q)); npr=reshape(tab[:,2],size(gpu.q))
                @test maximum(abs.(gpu.q.-npq))<=1e-11
                @test maximum(abs.(gpu.reference.-npr))<=1e-13
                m=emit(label,H,gpu,periodic); results["$(label)_H$H"]=m
                save("raw/$(label)_H$(H).toml",m)
                println(m); flush(stdout)
            end
            r=[results["$(label)_H$H"] for H in (16,32,64)]
            for i in 1:2
                @test log2(r[i]["L2"]/r[i+1]["L2"]) >= (periodic ? 1.7 : .8)
            end
            @test r[3]["L2"] <= (periodic ? .02 : .05)
            @test r[3]["Linf"] <= (periodic ? .03 : .08)
            @test r[3]["corner_Linf"] <= (periodic ? .03 : .08)
        end
        for tau in (.5004434050067377,.5017736200269506)
            cpu=simulate(16,false,CPU();tau=tau,lowtau=true)
            gpu=simulate(16,false,CUDA.CUDABackend();tau=tau,lowtau=true)
            delta=maximum(abs.(cpu.q.-gpu.q))
            @test delta<=1e-11
            @test minimum(gpu.q)>0 && maximum(gpu.q)<2
            @test maximum(abs.(cpu.f.-gpu.f))<=1e-11
            results["lowtau_$tau"]=Dict("tau"=>tau,"steps"=>256,"max_q_difference"=>delta,
                "minimum_q"=>minimum(gpu.q),"maximum_q"=>maximum(gpu.q),
                "scope"=>"short prescribed-shear stability/parity only; no analytic accuracy claim")
        end
    end
    @testset "Same-schedule legacy control and low-tau exact transients" begin
        for tau in (.8,.5004434050067377,.5017736200269506)
            hs=tau==.8 ? (16,32,64) : (8,16,32)
            tf=tau==.8 ? .2 : .02
            pair=Dict{Bool,Any}()
            for moment in (false,true)
                mm=[]
                for H in hs
                    if tau==.8 && moment
                        m=results["box_DN_H$H"]
                    else
                        r=simulate(H,false,CUDA.CUDABackend();tau=tau,moment=moment,tend=tf)
                        label="box_tau$(tau)_moment$(moment)"
                        m=emit(label,H,r,false)
                        save("raw/$(label)_H$H.toml",m)
                        results["$(label)_H$H"]=m
                    end
                    push!(mm,m)
                end
                pair[moment]=mm
            end
            a,b=pair[true],pair[false]
            pnew=[log2(a[i]["L2"]/a[i+1]["L2"]) for i=1:2]
            pold=[log2(b[i]["L2"]/b[i+1]["L2"]) for i=1:2]
            save("raw/orders_tau$(tau).toml",Dict("new"=>pnew,"legacy"=>pold,"tau"=>tau))
            @test all(pnew .>= .8)
            @test all(pnew .>= pold .- .15)
            @test a[3]["L2"]<=.05 && a[3]["Linf"]<=.08 && a[3]["corner_Linf"]<=.08
            for i=1:3, key in ("L2","Linf","corner_Linf")
                @test a[i][key] <= 1.1*b[i][key]+1e-5
            end
            println("TAU ",tau," pnew=",pnew," plegacy=",pold); flush(stdout)
        end
    end
    results["status"]="PASS_INTEGRATED_SCALAR_ONLY"
catch err
    results["status"]="FAIL"; results["error"]=sprint(showerror,err)
    save("raw/result.toml",results); rethrow()
end
results["source_sha"]=SOURCE_SHA
results["scope"]="actual integrated BC and production collision/stream; coupled restart checked separately; not full T190 qualification"
save("raw/result.toml",results)
