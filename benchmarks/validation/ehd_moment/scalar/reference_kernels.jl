module ExternalMoment2D
using KernelAbstractions
const CX=(0,1,0,-1,0,1,-1,-1,1)
const CY=(0,0,1,0,-1,1,1,-1,-1)
const W=(4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36)
@inline function equilibrium(k,q,vx,vy)
    c=CX[k]*vx+CY[k]*vy
    q*W[k]*(1+3c+4.5c*c-1.5*(vx*vx+vy*vy))
end
@inline function moment_wall!(f,i,j,qb,vx,vy,top)
    @inbounds begin
        if top
            G=f[i,j,3]+f[i,j,6]+f[i,j,7]; D=f[i,j,6]-f[i,j,7]
            jy=2*(G-qb*(1+3vy+3vy*vy)/6)
            jx=6D-qb*vx*(1+3vy)
        else
            G=f[i,j,5]+f[i,j,8]+f[i,j,9]; D=f[i,j,9]-f[i,j,8]
            jy=2*(qb*(1-3vy+3vy*vy)/6-G)
            jx=6D-qb*vx*(1-3vy)
        end
        for k in 1:9
            f[i,j,k]=equilibrium(k,qb,vx,vy)+3W[k]*(CX[k]*jx+CY[k]*jy)
        end
    end
end
@kernel function electrodes!(f,@Const(q),@Const(vx),@Const(vy),periodic,moment,Nx,Ny)
    i=@index(Global)
    @inbounds if periodic || (i>1 && i<Nx)
        if moment
            moment_wall!(f,i,1,1.0,vx[i,1],vy[i,1],false)
        else
            for k in 1:9
                f[i,1,k]=f[i,2,k]+equilibrium(k,1.0,vx[i,1],vy[i,1])-equilibrium(k,q[i,2],vx[i,2],vy[i,2])
            end
        end
        if periodic
            if moment
                moment_wall!(f,i,Ny,1.0,vx[i,Ny],vy[i,Ny],true)
            else
                for k in 1:9
                    f[i,Ny,k]=f[i,Ny-1,k]+equilibrium(k,1.0,vx[i,Ny],vy[i,Ny])-equilibrium(k,q[i,Ny-1],vx[i,Ny-1],vy[i,Ny-1])
                end
            end
        else
            qb=q[i,Ny-1]
            for k in 1:9
                f[i,Ny,k]=f[i,Ny-1,k]+equilibrium(k,qb,vx[i,Ny],vy[i,Ny])-equilibrium(k,qb,vx[i,Ny-1],vy[i,Ny-1])
            end
        end
    end
end
@kernel function sides!(f,@Const(q),@Const(vx),@Const(vy),Nx)
    j=@index(Global)
    @inbounds for side in 1:2
        b=side==1 ? 1 : Nx; n=side==1 ? 2 : Nx-1
        for k in 1:9
            f[b,j,k]=f[n,j,k]+equilibrium(k,q[n,j],vx[b,j],vy[b,j])-equilibrium(k,q[n,j],vx[n,j],vy[n,j])
        end
    end
end
end
