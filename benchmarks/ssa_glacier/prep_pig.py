"""Crop BedMachine Antarctica v4 + MEaSUREs v2 velocity around Pine Island Glacier,
bin both onto one regular grid (default 2 km) and write raw Float64 files for Julia.

Usage: conda run -n downscalewind python -I prep_pig.py [--dx 2000] [--box=x0,x1,y0,y1 (km, EPSG:3031)]
Output (benchmarks/ssa_glacier/data/, gitignored): <field>.bin little-endian Float64,
Nx*Ny values, x fastest, y increasing (Julia column-major Nx x Ny), + pig_header.txt.
Fields: H (m), s surface (m), bed (m), mask (0 ocean, 1 land, 2 grounded, 3 floating),
vx, vy (m/yr, gaps filled), vvalid (1 = at least one observed pixel), verr (m/yr).
"""
import argparse, os
import numpy as np
import xarray as xr

D = '/Users/guillaume/Documents/Recherche/kraken/data/antarctica/'
BM = D + 'NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc'
VEL = D + 'antarctica_ice_velocity_450m_v2.nc'

ap = argparse.ArgumentParser()
ap.add_argument('--dx', type=float, default=2000.0)
ap.add_argument('--box', type=lambda s: [float(t) for t in s.split(',')], default=[-1680, -1520, -310, -130],
                help='x0,x1,y0,y1 in km, e.g. --box=-1680,-1520,-310,-130')
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
ap.add_argument('--stats-only', action='store_true')
a = ap.parse_args()
x0, x1, y0, y1 = [c * 1e3 for c in a.box]
dx = a.dx
Nx = int(round((x1 - x0) / dx)); Ny = int(round((y1 - y0) / dx))


def binned(ds, names):
    """Mean of the pixels whose centre falls in each coarse cell (NaN ignored) + counts."""
    sub = ds.sel(x=slice(x0 - 1000, x1 + 1000), y=slice(y1 + 1000, y0 - 1000))
    X = sub.x.values.astype(float); Y = sub.y.values.astype(float)
    ix = np.floor((X - x0) / dx).astype(int); iy = np.floor((Y - y0) / dx).astype(int)
    okx = (ix >= 0) & (ix < Nx); oky = (iy >= 0) & (iy < Ny)
    ix = np.where(okx, ix, 0); iy = np.where(oky, iy, 0)
    cell = (iy[:, None] * Nx + ix[None, :])
    valid_geo = oky[:, None] & okx[None, :]
    out = {}
    for n in names:
        f = sub[n].values.astype(float)
        if n in ('VX', 'VY', 'ERRX', 'ERRY'):
            f[np.abs(f) > 1e5] = np.nan
        ok = valid_geo & np.isfinite(f)
        cnt = np.bincount(cell[ok], minlength=Nx * Ny).astype(float)
        sm = np.bincount(cell[ok], weights=f[ok], minlength=Nx * Ny)
        out[n] = (np.where(cnt > 0, sm / np.maximum(cnt, 1), np.nan).reshape(Ny, Nx), cnt.reshape(Ny, Nx))
    return out


bm = xr.open_dataset(BM)
sub = bm.sel(x=slice(x0 - 1000, x1 + 1000), y=slice(y1 + 1000, y0 - 1000))
m_pix = sub.mask.values
ice_pix = ((m_pix == 2) | (m_pix == 3)).astype(float)
sub = sub.assign(ice=(('y', 'x'), ice_pix), fl=(('y', 'x'), (m_pix == 3).astype(float)),
                 gr=(('y', 'x'), (m_pix == 2).astype(float)),
                 Hi=(('y', 'x'), sub.thickness.values * ice_pix), si=(('y', 'x'), sub.surface.values * ice_pix),
                 bi=(('y', 'x'), sub.bed.values * ice_pix))
b = binned(sub, ['ice', 'fl', 'gr', 'Hi', 'si', 'bi'])
# 'cnt' is the number of pixels per cell, 'mean' of indicators are fractions
f_ice = b['ice'][0]
nice = np.maximum(f_ice, 1e-12)
H = b['Hi'][0] / nice; s = b['si'][0] / nice; bed = b['bi'][0] / nice
f_fl = b['fl'][0] / nice  # floating fraction among ice pixels
mask = np.where(f_ice < 0.5, 0.0, np.where(f_fl >= 0.5, 3.0, 2.0))

vel = xr.open_dataset(VEL)
v = binned(vel, ['VX', 'VY', 'ERRX', 'ERRY'])
vx, vy = v['VX'][0], v['VY'][0]
vvalid = np.isfinite(vx) & np.isfinite(vy)
verr = np.hypot(v['ERRX'][0], v['ERRY'][0])


def fill(f, valid):
    """Fill invalid cells by iterated 8-neighbour averaging of valid/filled neighbours."""
    f = np.where(valid, f, np.nan).copy()
    for _ in range(max(Nx, Ny)):
        bad = ~np.isfinite(f)
        if not bad.any():
            break
        p = np.pad(f, 1, constant_values=np.nan)
        stack = np.stack([p[1 + di:1 + di + Ny, 1 + dj:1 + dj + Nx] for di in (-1, 0, 1) for dj in (-1, 0, 1)])
        with np.errstate(all='ignore'):
            nb = np.nanmean(stack, axis=0)
        f = np.where(bad & np.isfinite(nb), nb, f)
    return np.where(np.isfinite(f), f, 0.0)


# non-ice cells (rock/ocean inside the box): fill H, s, bed from ice neighbours so the
# solver sees a smooth stand-in; they carry mask 0 and are excluded from the misfit
icem = mask > 0
H = fill(H, icem); s = fill(s, icem); bed = fill(bed, icem)
vxf = fill(vx, vvalid); vyf = fill(vy, vvalid)
sp = np.hypot(vxf, vyf)
ice = mask > 0
print(f'box km x[{a.box[0]},{a.box[1]}] y[{a.box[2]},{a.box[3]}]  grid {Nx} x {Ny}  dx {dx:.0f} m')
print(f'ice cells {ice.mean():.3f}  floating(of all) {(mask == 3).mean():.3f}  grounded {(mask == 2).mean():.3f}'
      f'  non-ice {(mask == 0).sum()} cells')
print(f'velocity missing: {(~vvalid).mean():.3f} of cells, {(~vvalid & ice).sum()} among ice cells')
print(f'obs speed max {sp.max():.0f}  median(ice) {np.median(sp[ice]):.0f} m/yr;  H min/max (ice) {H[ice].min():.0f}/{H[ice].max():.0f}')
for nm, sl in (('W', (slice(None), 0)), ('E', (slice(None), -1)), ('S', (0, slice(None))), ('N', (-1, slice(None)))):
    print(f'  edge {nm}: non-ice {(mask[sl] == 0).sum()}, max speed {sp[sl].max():.0f}')
if a.stats_only:
    raise SystemExit
os.makedirs(a.out, exist_ok=True)
fields = dict(H=H, s=s, bed=np.where(ice, bed, 0.0), mask=mask, vx=vxf, vy=vyf, vvalid=vvalid.astype(float), verr=np.nan_to_num(verr))
for k, arr in fields.items():
    # numpy (Ny, Nx) C order with y ascending == Julia (Nx, Ny) column-major
    # binned arrays are indexed iy from y0 upward already
    arr.astype('<f8').tofile(os.path.join(a.out, k + '.bin'))
with open(os.path.join(a.out, 'pig_header.txt'), 'w') as fh:
    fh.write(f'Nx = {Nx}\nNy = {Ny}\ndx = {dx}\nx0 = {x0}\ny0 = {y0}\nfields = {" ".join(fields)}\n'
             'crs = EPSG:3031\nunits = m, m/yr\n')
print('written to', a.out)
