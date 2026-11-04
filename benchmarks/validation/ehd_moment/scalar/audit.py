"""Recompute 26985556 scalar errors from immutable node records (no simulation)."""
from pathlib import Path
import csv, hashlib, json, tomllib, sys
import numpy as np

RUN = Path(sys.argv[1]).resolve()
HERE = Path(sys.argv[2]).resolve()
HERE.mkdir(parents=True, exist_ok=True)
rows=[]; hashes=[]; checks=0
for file in sorted((RUN/'raw').glob('*.csv')):
    a=np.genfromtxt(file, delimiter=',', names=True)
    nx,ny=int(a['i'].max()),int(a['j'].max())
    assert len(a)==nx*ny and len(set(zip(a['i'],a['j'])))==len(a)
    assert np.isfinite(a['q']).all() and a['q'].min()>0
    q=a['q'].reshape(ny,nx).T; r=a['reference'].reshape(ny,nx).T
    m=tomllib.loads(file.with_suffix('.toml').read_text())
    h=m['H']; t=m['time']; periodic=m['label']=='periodic_DD'
    lo,hi=np.pi/2,np.pi
    for _ in range(65):
        k=(lo+hi)/2
        if k*np.cos(k)+np.sin(k)>0: lo=k
        else: hi=k
    k=np.pi if periodic else (lo+hi)/2
    kx=2*np.pi if periodic else np.pi
    x=np.arange(nx)[:,None]/h; y=np.arange(ny)[None,:]/h
    exact=1+.05*np.exp(y-.05*(kx*kx+k*k+1)*t)*np.sin(k*y)*np.cos(kx*x)
    assert np.max(abs(exact-r))<1e-13
    e=q-exact; scale=np.max(abs(exact-1))
    actual=dict(L2=np.linalg.norm(e)/np.linalg.norm(exact-1), Linf=np.max(abs(e))/scale,
        corner_Linf=np.max(abs(e[np.ix_([0,nx-1],[0,ny-1])]))/scale,
        injector_Linf=np.max(abs(e[1:-1,0]))/scale,
        collector_Linf=np.max(abs(e[1:-1,-1]))/scale,
        sides_Linf=np.max(abs(e[[0,-1],1:-1]))/scale,
        interior_Linf=np.max(abs(e[1:-1,1:-1]))/scale)
    for key,value in actual.items():
        assert abs(value-m[key])<2e-11,(file.name,key,value,m[key])
        checks+=1
    rows.append(dict(case=file.stem,H=h,time=t,**actual))
    hashes.append(hashlib.sha256(file.read_bytes()).hexdigest()+'  '+str(file.relative_to(RUN)))
lookup={r['case']:r for r in rows}; orders={}
assert len(rows) == 21, 'Incomplete scalar output: expected 21 node tables'
periodic = [lookup[f'periodic_DD_H{h}'] for h in (16,32,64)]
assert all(np.log2(periodic[i]['L2']/periodic[i+1]['L2']) >= 1.7 for i in (0,1))
assert periodic[-1]['L2'] <= .02 and periodic[-1]['Linf'] <= .03 and periodic[-1]['corner_Linf'] <= .03
for tau in (.8,.5004434050067377,.5017736200269506):
    hs=(16,32,64) if tau==.8 else (8,16,32)
    new=[lookup[f'box_DN_H{h}' if tau==.8 else f'box_tau{tau}_momenttrue_H{h}'] for h in hs]
    old=[lookup[f'box_tau{tau}_momentfalse_H{h}'] for h in hs]
    pn=np.log2([new[i]['L2']/new[i+1]['L2'] for i in (0,1)])
    po=np.log2([old[i]['L2']/old[i+1]['L2'] for i in (0,1)])
    assert np.all(pn>=.8) and np.all(pn>=po-.15)
    assert new[-1]['L2']<=.05 and new[-1]['Linf']<=.08 and new[-1]['corner_Linf']<=.08
    for aa,bb in zip(new,old):
        for key in ('L2','Linf','corner_Linf'): assert aa[key]<=1.1*bb[key]+1e-5
    orders[str(tau)]=dict(new=pn.tolist(),legacy=po.tolist(),finest=new[-1])
with (HERE/'summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(HERE/'audit.json').write_text(json.dumps(dict(status='PASS',cases=len(rows),metric_checks=checks,orders=orders),indent=2))
(HERE/'audit_inputs.sha256').write_text('\n'.join(hashes)+'\n')
print(json.dumps(dict(status='PASS',cases=len(rows),metric_checks=checks,orders=orders),indent=2))
