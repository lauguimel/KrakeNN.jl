"""Build the committed tables of the RheoTool viscoelastic-cavity reference (#51).

Reads the per-run CSV files written by ``extract_cavity.py``
(``<tag>_energy.csv``, ``<tag>_profiles.csv``, ``<tag>_vortex.csv``) from one
directory, and writes:

``benchmarks/results/rheotool_compare/viscoelastic_cavity/``
    ``runs.csv``          the run manifest below, one row per run;
    ``energy.csv``        tag, t, kinetic_energy_mean, elastic_energy_mean:
                          the logged series of every run, one row every
                          --energy-spacing time units (every logged row
                          without the option);
    ``vortex.csv``        tag, t, x_centre, y_centre, psi_min,
                          max_abs_column_integral_ux at every written time;
    ``profiles_t10.csv``  tag, t, line, coord, ux, uy, tau_xx, tau_xy, tau_yy,
                          C_xx, C_xy, C_yy at t = 10 for every run, plus
                          t = 30 for ve_re1_n64_t30.

``test/reference/rheotool_cavity_wi1/``
    ``summary_t10.csv``                one row per run: vortex values of the
                                       t = 10 field and the energy row nearest
                                       to t = 10, with that row's time;
    ``ve_re1_n64_profiles_t10.csv``    line profiles of ve_re1_n64 at t = 10;
    ``newt_re1_n64_profiles_t10.csv``  line profiles of newt_re1_n64 at t = 10
                                       (velocity only).

Every value is copied as text from the input files (``%.12g``, as written by
``extract_cavity.py``). Nothing is recomputed or reformatted, so identical
inputs give byte-identical outputs. The run manifest (host, job, times) comes
from the PBS output files of the runs and is written into RUNS below; the
script checks it against the data (N, logging interval, end time).

Usage:
    python build_tables.py <csv_dir> [--results DIR] [--reference DIR]
                           [--energy-spacing DT]

``--energy-spacing DT`` keeps one energy row every DT time units (DT must be a
whole multiple of every run's logging interval, 20 * deltaT, and must keep
each run's t = 10 row and last logged row). It is off by default (every logged row is kept);
the committed energy.csv was built with --energy-spacing 0.008.
"""

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_RESULTS = os.path.join(REPO, "benchmarks", "results", "rheotool_compare", "viscoelastic_cavity")
DEFAULT_REFERENCE = os.path.join(REPO, "test", "reference", "rheotool_cavity_wi1")

# Run manifest. Source: the PBS output file of each job (Job_Name, exec host,
# 'start' line, resources_used.walltime) and each run's own controlDict,
# blockMeshDict and constitutiveProperties. Re = rho, since U = L = 1 and
# etaS + etaP = 1 (Newtonian: eta = 1).
RUN_FIELDS = ("tag", "model", "N", "Re", "deltaT", "endTime", "host", "pbs_job", "start", "wall_time")
RUNS = (
    ("newt_re1_n64", "Newtonian", "64", "1", "0.0002", "10", "cpu1n044", "25840953", "2026-09-24T19:13:34+10:00", "00:04:12"),
    ("newt_re1_n128", "Newtonian", "128", "1", "0.0002", "10", "cpu1n045", "25840955", "2026-09-24T19:13:34+10:00", "00:19:08"),
    ("newt_re1_n256", "Newtonian", "256", "1", "0.0002", "10", "cpu1n045", "25840957", "2026-09-24T19:13:34+10:00", "01:28:38"),
    ("ve_re1_n64", "Oldroyd-BLog", "64", "1", "0.0002", "10", "cpu1n044", "25840952", "2026-09-24T19:13:34+10:00", "00:08:04"),
    ("ve_re1_n128", "Oldroyd-BLog", "128", "1", "0.0002", "10", "cpu1n045", "25840954", "2026-09-24T19:13:34+10:00", "00:33:22"),
    ("ve_re1_n256", "Oldroyd-BLog", "256", "1", "0.0002", "10", "cpu1n045", "25840956", "2026-09-24T19:13:34+10:00", "02:34:04"),
    ("ve_re1_n64_dt1e-4", "Oldroyd-BLog", "64", "1", "0.0001", "10", "cpu1n040", "25841227", "2026-09-24T20:12:44+10:00", "00:16:34"),
    ("ve_re1_n128_dt1e-4", "Oldroyd-BLog", "128", "1", "0.0001", "10", "cpu1n044", "25841228", "2026-09-24T20:12:45+10:00", "01:16:20"),
    ("ve_re1_n256_dt1e-4", "Oldroyd-BLog", "256", "1", "0.0001", "10", "cpu1n047", "25841229", "2026-09-24T20:12:44+10:00", "06:19:23"),
    ("ve_re1_n64_t30", "Oldroyd-BLog", "64", "1", "0.0002", "30", "cpu1n047", "25841230", "2026-09-24T20:12:44+10:00", "00:25:17"),
    ("ve_re001_n128", "Oldroyd-BLog", "128", "0.01", "0.0002", "10", "cpu1n045", "25840958", "2026-09-24T19:13:34+10:00", "00:35:08"),
    ("ve_re001_n127_asshipped", "Oldroyd-BLog", "127", "0.01", "0.0002", "10", "cpu1n045", "25840959", "2026-09-24T19:13:35+10:00", "00:30:35"),
)

LINES = ("x=0.5", "y=0.5", "y=0.75", "y=0.9", "y=0.95")
PROFILE_COLUMNS = ("line", "coord", "ux", "uy", "tau_xx", "tau_xy", "tau_yy", "C_xx", "C_xy", "C_yy")
STEPS_PER_LOG = 20  # the coded function object logs every 20 time steps
TOL = 1e-6          # time matching tolerance; logged times carry ~1e-11 float drift


def read_csv(path):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows, {os.path.getsize(path)} bytes)")


def fail(msg):
    sys.exit(f"build_tables.py: {msg}")


def load_run(csv_dir, run):
    """Read the three CSVs of one run and check them against its manifest row."""
    tag, model, n, _, dt, end = run[:6]
    n, dt, end = int(n), float(dt), float(end)

    eh, energy = read_csv(os.path.join(csv_dir, f"{tag}_energy.csv"))
    vh, vortex = read_csv(os.path.join(csv_dir, f"{tag}_vortex.csv"))
    ph, profiles = read_csv(os.path.join(csv_dir, f"{tag}_profiles.csv"))
    if eh != ["t", "kinetic_energy_mean", "elastic_energy_mean"]:
        fail(f"{tag}: unexpected energy header {eh}")
    if vh != ["t", "x_centre", "y_centre", "psi_min", "max_abs_column_integral_ux"]:
        fail(f"{tag}: unexpected vortex header {vh}")
    if ph != ["t"] + list(PROFILE_COLUMNS):
        fail(f"{tag}: unexpected profiles header {ph}")

    # energy: one row at t = 0, then one every STEPS_PER_LOG steps up to endTime
    log_dt = STEPS_PER_LOG * dt
    expected = int(round(end / log_dt)) + 1
    if len(energy) != expected:
        fail(f"{tag}: {len(energy)} energy rows, expected {expected} for deltaT {dt}, endTime {end}")
    for i, row in enumerate(energy):
        if abs(float(row[0]) - i * log_dt) > TOL:
            fail(f"{tag}: energy row {i} at t = {row[0]}, expected {i * log_dt}")

    # vortex: every written time 1 .. endTime
    times = [float(r[0]) for r in vortex]
    if times != [float(k) for k in range(1, int(end) + 1)]:
        fail(f"{tag}: vortex times {times} do not match endTime {end}")

    # profiles: N points per line at every written time; tau/C present iff viscoelastic
    for t in times:
        rows_t = [r for r in profiles if float(r[0]) == t]
        for line in LINES:
            k = sum(1 for r in rows_t if r[1] == line)
            if k != n:
                fail(f"{tag}: {k} rows on {line} at t = {t}, expected N = {n}")
        has_tau = [all(v != "" for v in r[5:]) for r in rows_t]
        if model == "Newtonian" and any(r[5:] != [""] * 6 for r in rows_t):
            fail(f"{tag}: Newtonian run with stress values")
        if model != "Newtonian" and not all(has_tau):
            fail(f"{tag}: viscoelastic run with missing stress values")
    return energy, vortex, profiles


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv_dir", help="directory holding <tag>_{energy,profiles,vortex}.csv")
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--reference", default=DEFAULT_REFERENCE)
    ap.add_argument("--energy-spacing", type=float, default=None)
    args = ap.parse_args()

    runs_rows, energy_rows, vortex_rows, profile_rows, summary_rows = [], [], [], [], []
    reference_profiles = {}
    for run in RUNS:
        tag, model, n, re, dt, end = run[:6]
        energy, vortex, profiles = load_run(args.csv_dir, run)
        runs_rows.append(list(run))

        stride = 1
        if args.energy_spacing is not None:
            ratio = args.energy_spacing / (STEPS_PER_LOG * float(dt))
            stride = int(round(ratio))
            if stride < 1 or abs(ratio - stride) > 1e-9:
                fail(f"{tag}: --energy-spacing {args.energy_spacing} is not a multiple of "
                     f"the logging interval {STEPS_PER_LOG * float(dt)}")
            if (len(energy) - 1) % stride:
                fail(f"{tag}: --energy-spacing {args.energy_spacing} drops the last logged "
                     f"row (t = {energy[-1][0]})")
            i10 = int(round(10.0 / (STEPS_PER_LOG * float(dt))))
            if i10 % stride:
                fail(f"{tag}: --energy-spacing {args.energy_spacing} drops the t = 10 row "
                     f"(t = {energy[i10][0]})")
        energy_rows += [[tag] + r for i, r in enumerate(energy) if i % stride == 0]
        vortex_rows += [[tag] + r for r in vortex]

        profile_times = [10.0] + ([30.0] if tag == "ve_re1_n64_t30" else [])
        profile_rows += [[tag] + r for r in profiles if float(r[0]) in profile_times]

        # t = 10: vortex row of the field written at t = 10; energy row nearest
        # to t = 10 (the dt 1e-4 runs log it as t = 9.99999999999)
        v10 = next(r for r in vortex if float(r[0]) == 10.0)
        e10 = min(energy, key=lambda r: abs(float(r[0]) - 10.0))
        if abs(float(e10[0]) - 10.0) > TOL:
            fail(f"{tag}: no energy row near t = 10")
        summary_rows.append([tag, model, n, re, dt, v10[0], v10[1], v10[2], v10[3], e10[0], e10[1], e10[2]])

        if tag in ("ve_re1_n64", "newt_re1_n64"):
            ncol = len(PROFILE_COLUMNS) if model != "Newtonian" else 4
            reference_profiles[tag] = (PROFILE_COLUMNS[:ncol],
                                       [r[1:1 + ncol] for r in profiles if float(r[0]) == 10.0])

    os.makedirs(args.results, exist_ok=True)
    os.makedirs(args.reference, exist_ok=True)
    write_csv(os.path.join(args.results, "runs.csv"), RUN_FIELDS, runs_rows)
    write_csv(os.path.join(args.results, "energy.csv"),
              ("tag", "t", "kinetic_energy_mean", "elastic_energy_mean"), energy_rows)
    write_csv(os.path.join(args.results, "vortex.csv"),
              ("tag", "t", "x_centre", "y_centre", "psi_min", "max_abs_column_integral_ux"), vortex_rows)
    write_csv(os.path.join(args.results, "profiles_t10.csv"), ("tag", "t") + PROFILE_COLUMNS, profile_rows)
    write_csv(os.path.join(args.reference, "summary_t10.csv"),
              ("tag", "model", "N", "Re", "deltaT", "t", "x_centre", "y_centre", "psi_min",
               "energy_t", "kinetic_energy_mean", "elastic_energy_mean"), summary_rows)
    for tag, (header, rows) in reference_profiles.items():
        write_csv(os.path.join(args.reference, f"{tag}_profiles_t10.csv"), header, rows)


if __name__ == "__main__":
    main()
