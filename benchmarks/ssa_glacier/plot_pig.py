"""Quick-look figure: observed speed, modelled speed (best Weertman C), difference.
Reads benchmarks/ssa_glacier/out/*.bin written by pig_forward.jl; writes out/pig_quicklook.png.
Run: conda run -n downscalewind python -I benchmarks/ssa_glacier/plot_pig.py"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
h = dict(l.replace(' ', '').split('=') for l in open(os.path.join(out, 'pig_out.txt')) if '=' in l)
Nx, Ny, dx, C = int(h['Nx']), int(h['Ny']), float(h['dx']), float(h['C'])
rd = lambda n: np.fromfile(os.path.join(out, n + '.bin'), dtype='<f8').reshape(Ny, Nx)
obs, mod, mask = rd('obs_speed'), rd('model_speed'), rd('mask')
ext = [0, Nx * dx / 1e3, 0, Ny * dx / 1e3]
fig, ax = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
kw = dict(origin='lower', extent=ext)
for a, f, t in ((ax[0], obs, 'observed speed (m/yr)'), (ax[1], mod, f'SSA, Weertman C = {C:g} (m/yr)')):
    im = a.imshow(np.where(mask > 0, f, np.nan), cmap='viridis', vmin=0, vmax=np.nanpercentile(obs, 99), **kw)
    a.set_title(t); plt.colorbar(im, ax=a, shrink=0.8)
d = np.where(mask > 0, mod - obs, np.nan)
lim = np.nanpercentile(np.abs(d), 98)
im = ax[2].imshow(d, cmap='RdBu_r', vmin=-lim, vmax=lim, **kw)
ax[2].set_title('model - observed (m/yr)'); plt.colorbar(im, ax=ax[2], shrink=0.8)
for a in ax:
    a.contour(np.arange(Nx) * dx / 1e3 + dx / 2e3, np.arange(Ny) * dx / 1e3 + dx / 2e3, mask == 3, levels=[0.5], colors='w', linewidths=0.8)
    a.set_xlabel('x - x0 (km)'); a.set_ylabel('y - y0 (km)')
fig.savefig(os.path.join(out, 'pig_quicklook.png'), dpi=130)
print('saved', os.path.join(out, 'pig_quicklook.png'))
