"""Training figure for the learned basal law on the PIG crop.
Reads benchmarks/ssa_glacier/out/*_c<k>* written by pig_train.jl; writes out/pig_train_c<k>.png.
Run: KRAKEN_PIG_COARSEN=1 python -I benchmarks/ssa_glacier/plot_pig_train.py"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
k = os.environ.get('KRAKEN_PIG_COARSEN', '1'); tag = '_c' + k
h = dict(l.replace(' ', '').split('=') for l in open(os.path.join(out, f'pig_train_meta{tag}.txt')) if '=' in l)
Nx, Ny, dx, C = int(h['Nx']), int(h['Ny']), float(h['dx']), float(h['C'])
rd = lambda n: np.fromfile(os.path.join(out, n + tag + '.bin'), dtype='<f8').reshape(Ny, Nx)
obs, wm, ln, mask = rd('obs_speed'), rd('speed_weertman'), rd('speed_learned'), rd('mask')
ice = mask > 0
ext = [0, Nx * dx / 1e3, 0, Ny * dx / 1e3]
fig = plt.figure(figsize=(16, 10), constrained_layout=True)
gs = fig.add_gridspec(2, 6)
vmax = np.nanpercentile(obs[ice], 99)
panels = [(obs, 'observed speed (m/yr)', 'viridis', (0, vmax)), (wm, f'Weertman C = {C:g} (m/yr)', 'viridis', (0, vmax)),
          (ln, 'learned law (m/yr)', 'viridis', (0, vmax))]
lim = np.nanpercentile(np.abs(np.where(ice, wm - obs, np.nan)), 98)
panels += [(wm - obs, 'Weertman - observed (m/yr)', 'RdBu_r', (-lim, lim)), (ln - obs, 'learned - observed (m/yr)', 'RdBu_r', (-lim, lim))]
for q, (f, t, cm, (lo, hi)) in enumerate(panels):
    a = fig.add_subplot(gs[0, q] if q < 3 else gs[1, q - 3])
    im = a.imshow(np.where(ice, f, np.nan), origin='lower', extent=ext, cmap=cm, vmin=lo, vmax=hi)
    a.contour(np.arange(Nx) * dx / 1e3 + dx / 2e3, np.arange(Ny) * dx / 1e3 + dx / 2e3, mask == 3, levels=[0.5], colors='k', linewidths=0.6)
    a.set_title(t, fontsize=10); plt.colorbar(im, ax=a, shrink=0.8)
# (b) learned F(|u|, N) vs Weertman
cv = np.genfromtxt(os.path.join(out, f'pig_law_curves{tag}.csv'), delimiter=',', names=True)
ax = fig.add_subplot(gs[0, 3:5])
cols = plt.cm.plasma(np.linspace(0.1, 0.85, 4))
for c, Nv in zip(cols, np.unique(cv['N'])):
    s = cv['N'] == Nv
    ax.loglog(cv['speed'][s], cv['F_trained'][s], color=c, label=f'learned, N = {Nv:.0f} kPa')
    ax.loglog(cv['speed'][s], cv['F_pretrained'][s], color=c, ls=':', lw=1)
ax.loglog(cv['speed'][s], cv['F_weertman'][s], 'k--', label=f'Weertman C = {C:g}')
ax.set_xlabel('|u| (m/yr)'); ax.set_ylabel('basal traction F (kPa)'); ax.legend(fontsize=7); ax.set_title('(b) F(|u|, N); dotted = pretrained', fontsize=10)
# (c) misfit vs iteration
hs = np.genfromtxt(os.path.join(out, f'pig_train_history{tag}.csv'), delimiter=',', names=True)
ax = fig.add_subplot(gs[0, 5])
ax.plot(hs['iter'], hs['rms_all'], label='all'); ax.plot(hs['iter'], hs['rms_grounded'], label='grounded'); ax.plot(hs['iter'], hs['rms_floating'], label='floating')
ax.set_xlabel('iteration'); ax.set_ylabel('RMS speed misfit (m/yr)'); ax.legend(fontsize=8); ax.set_title('(c) misfit vs iteration', fontsize=10)
ax2 = fig.add_subplot(gs[1, 3:6])
ax2.semilogy(hs['iter'], hs['L']); ax2.set_xlabel('iteration'); ax2.set_ylabel('L (objective)'); ax2.set_title('objective', fontsize=10)
fig.savefig(os.path.join(out, f'pig_train{tag}.png'), dpi=120)
print('saved', os.path.join(out, f'pig_train{tag}.png'))
