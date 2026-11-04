"""Bounded, independent NumPy mirror of the pinned regularized scalar algebra."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import time,json,csv,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent
cx=np.array([0,1,0,-1,0,1,-1,-1,1]); cy=np.array([0,0,1,0,-1,1,1,-1,-1])
w=np.array([4/9]+[1/9]*4+[1/36]*4); opp=[0,3,4,1,2,7,8,5,6]
def eq(q,vx,vy):
    c=vx[...,None]*cx+vy[...,None]*cy
    return q[...,None]*w*(1+3*c+4.5*c*c-1.5*(vx*vx+vy*vy)[...,None])
def wall(a,qb,vx,vy,top=False):
    if top:
        G=a[...,2]+a[...,5]+a[...,6]; D=a[...,5]-a[...,6]
        jy=2*(G-qb*(1+3*vy+3*vy*vy)/6)
        jx=6*D-qb*vx*(1+3*vy)
    else:
        G=a[...,4]+a[...,7]+a[...,8]; D=a[...,8]-a[...,7]
        jy=2*(qb*(1-3*vy+3*vy*vy)/6-G)
        jx=6*D-qb*vx*(1-3*vy)
    return eq(qb,vx,vy)+3*w*(jx[...,None]*cx+jy[...,None]*cy)
def stream(f,periodic):
    n,m,_=f.shape; out=np.empty_like(f)
    for k,(dx,dy) in enumerate(zip(cx,cy)):
        out[:,:,k]=np.roll(f[:,:,k],(dx,dy),(0,1))
        if dy==1: out[:,0,k]=f[:,0,opp[k]]
        if dy==-1: out[:,-1,k]=f[:,-1,opp[k]]
        if not periodic:
            if dx==1: out[0,:,k]=f[0,:,opp[k]]
            if dx==-1: out[-1,:,k]=f[-1,:,opp[k]]
    return out
def boundary(f,vx,vy,periodic,reverse=False):
    q=f.sum(2); sl=slice(None) if periodic else slice(1,-1)
    f[sl,0]=wall(f[sl,0],np.ones_like(q[sl,0]),vx[sl,0],vy[sl,0])
    if periodic:
        f[:,-1]=wall(f[:,-1],np.ones_like(q[:,-1]),vx[:,-1],vy[:,-1],True)
    else:
        # Original zero-gradient collector NEE at electrode interior nodes.
        f[sl,-1]=f[sl,-2]+eq(q[sl,-2],vx[sl,-1],vy[sl,-1])-eq(q[sl,-2],vx[sl,-2],vy[sl,-2])
        q=f.sum(2)
        for b,n in ([( -1,-2),(0,1)] if reverse else [(0,1),(-1,-2)]):
            f[b]=f[n]+eq(q[n],vx[b],vy[b])-eq(q[n],vx[n],vy[n])
    return f
def eigen_k():
    lo,hi=np.pi/2,np.pi
    for _ in range(60):
        k=(lo+hi)/2
        if k*np.cos(k)+np.sin(k)>0: lo=k
        else: hi=k
    return (lo+hi)/2
def exact(H,t,periodic):
    x=np.arange(H if periodic else H+1)/H; y=np.arange(H+1)/H
    X,Y=np.meshgrid(x,y,indexing='ij'); k=np.pi if periodic else eigen_k(); kx=2*np.pi if periodic else np.pi
    lam=.05*(kx*kx+k*k+1)
    z=.05*np.exp(Y)*np.sin(k*Y)*np.cos(kx*X)*np.exp(-lam*t)
    qx=-.05*kx*np.exp(Y)*np.sin(k*Y)*np.sin(kx*X)*np.exp(-lam*t)
    qy=.05*np.exp(Y)*(np.sin(k*Y)+k*np.cos(k*Y))*np.cos(kx*X)*np.exp(-lam*t)
    return 1+z,qx,qy
start=time.perf_counter(); identities=0
for top in (False,True):
    for vx0 in (-.01,0.,.013):
        for vy0 in (0.,.006,.02):
            vx=np.array([vx0]); vy=np.array([vy0]); qb=np.array([1.2])
            f=eq(qb,vx,vy)+.0001*np.sin(np.arange(9))[None,:]
            g=wall(f,qb,vx,vy,top)
            ids=[2,5,6] if top else [4,7,8]; pair=(5,6) if top else (8,7)
            for diff in (g.sum()-qb[0],g[0,ids].sum()-f[0,ids].sum(),g[0,pair[0]]-g[0,pair[1]]-f[0,pair[0]]+f[0,pair[1]]):
                assert abs(diff)<=2e-14; identities+=1
            assert np.max(abs(wall(eq(qb,vx,vy),qb,vx,vy,top)-eq(qb,vx,vy)))<=2e-14
            identities+=1
rows=[]; failures=[]
for periodic in (True,False):
    for H in (16,32,64):
        tau=.8; h=1/H; dt=(tau-.5)/3*h*h/.05; n=round(.2/dt)
        q,qx,qy=exact(H,0,periodic); vx=np.zeros_like(q); vy=np.full_like(q,.1*dt/h)
        f=eq(q,vx,vy)-tau*w*(cx*(h*qx)[...,None]+cy*(h*qy)[...,None])
        minq=float(q.min()); budget=0.; sideorder=0.
        for step in range(n):
            if time.perf_counter()-start>30: raise TimeoutError('Frozen 30-second CPU cap')
            q=f.sum(2); fe=eq(q,vx,vy); jne=f-fe
            jx=jne@cx; jy=jne@cy
            post=fe+(1-1/tau)*3*w*(jx[...,None]*cx+jy[...,None]*cy)
            budget=max(budget,float(np.max(abs(post.sum(2)-q))))
            streamed=stream(post,periodic)
            if not periodic and step==0:
                a=boundary(streamed.copy(),vx,vy,False); b=boundary(streamed.copy(),vx,vy,False,True)
                sideorder=float(np.max(abs(a-b))); assert sideorder<=2e-14
            f=boundary(streamed,vx,vy,periodic)
            minq=min(minq,float(f.sum(2).min()))
            assert np.isfinite(f).all() and minq>0
        q=f.sum(2); ref,_,_=exact(H,n*dt,periodic); err=q-ref
        l2=float(np.linalg.norm(err)/np.linalg.norm(ref-1)); linf=float(np.max(abs(err))/np.max(abs(ref-1)))
        row=dict(layout='periodic_DD' if periodic else 'box_DN',H=H,steps=n,time=n*dt,L2=l2,Linf=linf,minq=minq,collision_mass=budget,side_order_difference=sideorder)
        rows.append(row); print(row,flush=True)
        np.savez_compressed(root/f'{row["layout"]}_H{H}.npz',q=q,reference=ref)
    selected=rows[-3:]; orders=[float(np.log2(selected[i]['L2']/selected[i+1]['L2'])) for i in (0,1)]
    for i,p in enumerate(orders):
        if p<(1.7 if periodic else .8): failures.append(f'{selected[0]["layout"]} order {i}: {p}')
    if selected[-1]['L2']>(.02 if periodic else .05): failures.append(f'{selected[0]["layout"]} L2 gate')
    if selected[-1]['Linf']>(.03 if periodic else .08): failures.append(f'{selected[0]["layout"]} Linf gate')
    print('orders',orders,flush=True)
    if failures: break
with (root/'summary.csv').open('w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=rows[0]); wr.writeheader(); wr.writerows(rows)
result=dict(status='FAIL' if failures else 'PASS',failures=failures,algebraic_checks=identities,seconds=time.perf_counter()-start,rows=rows,scope='NumPy external scalar verification; not production or coupled EC qualification')
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
(root/'input_sha256.txt').write_text(''.join(hashlib.sha256((root/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in ['screen.py','PROTOCOL.md']))
print(json.dumps(result,indent=2)); raise SystemExit(bool(failures))
