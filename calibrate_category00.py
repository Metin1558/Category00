"""
calibrate_category00.py — First, cheapest check to run against real
tissue for category00. Mirrors calibrate.py (word-sorting) and
calibrate_threat.py (loom-oi), adapted to a task with NO natural
magnitude axis to lean on: X fires electrode 0 once, Y fires electrode
1 once. There is no "weak vs strong" version of this stimulus — it is
already the simplest possible categorical distinction.

WHAT THIS CHECKS
-----------------
In simulation, this exact task returned zero measurable signal across
every tested drive level, connectivity density, and learning rate (see
AMAC_category00.md, and the falsification-ladder history in
organoid-oi's earlier record). This script exists to ask the one
question that matters before spending further session time: does REAL
tissue show ANY separation between X-trials and Y-trials that
simulation did not?

It sweeps membrane resistance (R) — the one parameter repeatedly found
to matter for whether a population sits silent, saturated, or in a
genuinely graded regime elsewhere in this project — and reports raw
X-vs-Y separation at each value. No sigmoid mapping, no electrode-count
tuning: this task has no continuous quantity to map, so the question is
simply "is there a difference at all," not "how do we resolve a
near-threshold case."

USAGE
-----
    python3 calibrate_category00.py --repeats 5
    python3 calibrate_category00.py --backend hardware --api-key ... --culture-id ... --repeats 5

Start with --repeats 5 on real hardware.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
for sub in ["", "core", "sim", "hardware"]:
    sys.path.insert(0, str(ROOT / sub))

import numpy as np
from category00_demo import xy_stimulus, make_config, build_organoid


def measure(organoid, cfg, rng, label, repeats):
    totals = []
    for _ in range(repeats):
        stim = xy_stimulus(label, cfg, rng)
        r = organoid.respond(stim, timestamp=0.0)
        totals.append(sum(len(x) for x in r.spike_times))
    return np.array(totals)


def scan_r(backend, api_key, culture_id, r_values, repeats, seed):
    print(f"  {'R':>6} | {'X mean':>8} {'X std':>7} | {'Y mean':>8} {'Y std':>7} | separated?")
    print("  " + "-" * 62)
    results = []
    for R in r_values:
        cfg = make_config(r_membrane_mohm=R)
        rng = np.random.default_rng(seed)
        organoid = build_organoid(backend, cfg, rng, api_key, culture_id)
        x_vals = measure(organoid, cfg, rng, "X", repeats)
        y_vals = measure(organoid, cfg, rng, "Y", repeats)
        x_ceiling = x_vals.mean() + 2 * x_vals.std()
        y_floor = y_vals.mean() - 2 * y_vals.std()
        y_ceiling = y_vals.mean() + 2 * y_vals.std()
        x_floor = x_vals.mean() - 2 * x_vals.std()
        separated = (x_ceiling < y_floor) or (y_ceiling < x_floor)
        results.append({"R": R, "x_mean": x_vals.mean(), "y_mean": y_vals.mean(), "separated": separated})
        print(f"  {R:>6} | {x_vals.mean():8.1f} {x_vals.std():7.1f} | "
              f"{y_vals.mean():8.1f} {y_vals.std():7.1f} | {'YES' if separated else 'no'}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r-values", type=str, default="10,20,30,40,50,60,80",
                    help="comma-separated membrane resistance values to sweep")
    ap.add_argument("--backend", choices=["synthetic", "hardware"], default="synthetic")
    ap.add_argument("--api-key", type=str, default=None)
    ap.add_argument("--culture-id", type=str, default=None)
    args = ap.parse_args()

    r_values = [float(x) for x in args.r_values.split(",")]

    print("=" * 70)
    print(f"  CALIBRATE-CATEGORY00 — backend: {args.backend}")
    print("=" * 70)
    if args.backend == "synthetic":
        print("  NOTE: this task returned zero signal at every tested R in prior")
        print("  simulation work. A repeat null result here is not a bug — it is")
        print("  the expected replication. A real separation would be the news.")
    print()

    results = scan_r(args.backend, args.api_key, args.culture_id, r_values, args.repeats, args.seed)
    print()

    any_separated = [r for r in results if r["separated"]]
    if any_separated:
        print(f"  RESULT: separation found at R = {[r['R'] for r in any_separated]}.")
        print("  This would be a genuinely new finding relative to prior simulation")
        print("  work — proceed to qualify_seed_category00.py at this R, and treat")
        print("  this as the possible 'H2' outcome described in AMAC_category00.md.")
    else:
        print("  RESULT: no separation at any tested R — consistent with prior")
        print("  simulation findings ('H1' in AMAC_category00.md). Do not proceed")
        print("  to seed qualification without a specific reason to expect a")
        print("  different R would help; the swept range already covers the")
        print("  silent-to-saturated span this project has found relevant elsewhere.")


if __name__ == "__main__":
    main()
