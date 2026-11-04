"""Independent consistency audit of the frozen short-restart test evidence."""
import hashlib
import json
import sys
import tomllib
from pathlib import Path

run = Path(sys.argv[1])
raw = run / 'raw'
def read(name):
    return tomllib.loads((raw / name).read_text())

tol = read('tolerances.toml')
status = read('status.toml')
checks = read('checks.toml')['checks']
rows = read('comparisons.toml')['comparisons']
stages = read('stages.toml')['stages']
cycles = read('calibration_by_cycle.toml')
assert hashlib.sha256((raw/'tolerances.toml').read_bytes()).hexdigest() == status['tolerance_sha256']
assert len(checks) == 306 and all(c['pass'] for c in checks)
assert status['passed'] == 306 and status['failed'] == 0
assert tol['factor'] == 2 and tol['repeats'] == 8 and tol['cycles'] == 20
assert len(rows) == 84 and len(stages) == 6
summary = {}
for closure in ['nee', 'moment']:
    table = {}
    assert len(list(raw.glob(f'calibration_{closure}_*.bin'))) == 8
    for quantity, limits in tol[closure].items():
        assert len(cycles[f'{closure}_{quantity}']) == 21
        assert limits['envelope_abs'] == max(cycles[f'{closure}_{quantity}'])
        assert limits['tolerance_abs'] == 2 * limits['envelope_abs']
        assert limits['tolerance_abs'] <= 1e-12 * max(1, limits['signal_scale'])
        subset = [r for r in rows if r['quantity']==quantity and r['tag'].startswith(closure+'_')]
        for r in subset:
            assert r['max_abs'] == max(r['cycle_abs_errors'])
            expected_n = 21 if r['tag'].endswith('/cold') else 13
            assert len(r['cycle_abs_errors']) == expected_n
            if '_cpu_' in r['tag']:
                assert r['bit_exact'] and r['max_abs']==0 and r['tolerance_abs']==0
            else:
                assert r['tolerance_abs']==limits['tolerance_abs']
                assert r['max_abs']<=r['tolerance_abs']
        restart = [r['max_abs'] for r in subset if '_gpu_' in r['tag'] and r['tag'].endswith('/restart')]
        cold = [r['max_abs'] for r in subset if r['tag'].endswith('/cold')]
        assert len(restart)==3 and len(cold)==3
        assert max(restart+cold) <= limits['envelope_abs']
        table[quantity] = dict(limits, restart_max=max(restart), cold_max=max(cold),
                              restart_to_envelope=max(restart)/limits['envelope_abs'])
    summary[closure] = table
assert all(s['input_exact'] and all(v==0 for v in s['before'].values()) for s in stages)
assert all(s['first_difference'] in ['POISSON','NOT_OBSERVED'] for s in stages)
summary['stage_counts'] = {k: sum(s['first_difference']==k for s in stages)
                           for k in ['POISSON','NOT_OBSERVED']}
summary['poisson_difference_max'] = max(s['after_poisson']['phi'] for s in stages)
summary['audit'] = 'PASS: 84 quantity comparisons, 306 checks, 12 frozen tolerances and 6 stages'
print(json.dumps(summary, indent=2))
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
