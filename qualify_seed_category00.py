"""
qualify_seed_category00.py — Same staged screening philosophy as
qualify_seed.py (word-sorting) and qualify_seed_threat.py (loom-oi),
adapted to category00's reality: no magnitude axis, no "weak variant"
to generalize to. Three stages here are:

STAGE 1 — Preflight (cheap, untrained): does raw X-response differ from
    raw Y-response at all, before any training?
STAGE 2 — Trained accuracy: after reward/penalty training, does the
    readout call fresh X/Y trials correctly?
STAGE 3 — Repeatability: does a SECOND independent evaluation batch
    (fresh random trials, same trained decoder) give a consistent
    result? This is this task's analog to loom-oi's "generalize to a
    held-out weak signal" stage — there is no weak variant here, so the
    check that matters is whether any apparent success is stable across
    repeated evaluation, not a one-off draw.

USAGE
-----
    python3 qualify_seed_category00.py --seeds 20
    python3 qualify_seed_category00.py --backend hardware --api-key ... --culture-id ... --seeds 5
"""
import argparse
import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

ROOT = Path(__file__).parent.resolve()

try:
    import numpy as _np_probe
    _SITE_PACKAGES = str(Path(_np_probe.__file__).resolve().parent.parent)
except ImportError:
    print("HATA: numpy bulunamadi. Once 'source venv/bin/activate' calistir.")
    sys.exit(1)


def _add_paths(site_packages):
    import sys as _sys
    _sys.path.insert(0, site_packages)
    _sys.path.insert(0, str(ROOT))
    _sys.path.insert(0, str(ROOT / "core"))
    _sys.path.insert(0, str(ROOT / "sim"))
    _sys.path.insert(0, str(ROOT / "hardware"))


def _build_organoid(backend, cfg, rng, api_key, culture_id):
    if backend == "hardware":
        from finalspark_organoid import FinalSparkOrganoid
        return FinalSparkOrganoid(cfg, api_key=api_key, culture_id=culture_id, rng=rng)
    from sim.oi_synth import SyntheticOrganoid
    return SyntheticOrganoid(cfg, rng=rng)


def _stage1(task):
    seed, site_packages, repeats, backend, api_key, culture_id = task
    _add_paths(site_packages)
    import numpy as np
    from category00_demo import xy_stimulus, make_config

    cfg = make_config()
    rng = np.random.default_rng(seed)
    organoid = _build_organoid(backend, cfg, rng, api_key, culture_id)

    def measure(label):
        totals = []
        for _ in range(repeats):
            stim = xy_stimulus(label, cfg, rng)
            r = organoid.respond(stim, timestamp=0.0)
            totals.append(sum(len(x) for x in r.spike_times))
        return np.array(totals)

    x_vals, y_vals = measure("X"), measure("Y")
    x_ceil, x_floor = x_vals.mean() + 2*x_vals.std(), x_vals.mean() - 2*x_vals.std()
    y_ceil, y_floor = y_vals.mean() + 2*y_vals.std(), y_vals.mean() - 2*y_vals.std()
    passed = (x_ceil < y_floor) or (y_ceil < x_floor)
    return {"seed": seed, "stage1_pass": passed, "x_mean": float(x_vals.mean()), "y_mean": float(y_vals.mean())}


def _eval_batch(exp, organoid, cfg, rng, n):
    from category00_demo import xy_stimulus
    correct = 0
    for _ in range(n):
        label = "X" if rng.random() < 0.5 else "Y"
        stim = xy_stimulus(label, cfg, rng)
        response = organoid.respond(stim, timestamp=0.0)
        rates_vec = exp.decoder.firing_rates(response)
        activations = exp.decoder.weights @ rates_vec
        pred = exp.decoder.categories[int(activations.argmax())]
        correct += int(pred == label)
    return correct / n


def _stage2_and_3(task):
    seed, n_trials, site_packages, decoder_lr, stdp_lr, backend, api_key, culture_id = task
    _add_paths(site_packages)
    import numpy as np
    from category00_demo import xy_stimulus, make_config
    from core.oi_loop import ClosedLoopExperiment

    cfg = make_config(learning_rate=stdp_lr, decoder_lr=decoder_lr)
    rng = np.random.default_rng(seed)
    organoid = _build_organoid(backend, cfg, rng, api_key, culture_id)
    exp = ClosedLoopExperiment(organoid, cfg, rng=rng)

    for i in range(n_trials):
        label = "X" if rng.random() < 0.5 else "Y"
        stim = xy_stimulus(label, cfg, rng)
        exp.run_trial(stim, i)

    stage2_acc = _eval_batch(exp, organoid, cfg, rng, 20)
    stage3_acc = _eval_batch(exp, organoid, cfg, rng, 20)  # second, independent batch

    return {
        "seed": seed,
        "stage2_pass": stage2_acc >= 0.75,
        "stage2_acc": stage2_acc,
        "stage3_pass": stage3_acc >= 0.75,
        "stage3_acc": stage3_acc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--decoder-lr", type=float, default=0.05)
    ap.add_argument("--stdp-lr", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=8, help="stage-1 probes per label")
    ap.add_argument("--backend", choices=["synthetic", "hardware"], default="synthetic")
    ap.add_argument("--api-key", type=str, default=None)
    ap.add_argument("--culture-id", type=str, default=None)
    args = ap.parse_args()

    seed_list = list(range(args.seeds))

    if args.backend == "hardware":
        workers = 1
        if args.workers and args.workers != 1:
            print(f"  [backend=hardware] ignoring --workers {args.workers}: forcing 1 (one live session, sequential).")
    else:
        workers = args.workers or cpu_count()

    print("=" * 70)
    print(f"  QUALIFY-SEED-CATEGORY00 — {len(seed_list)} seeds, {workers} workers")
    print("=" * 70)
    if args.backend == "synthetic":
        print("  Prior simulation history for this task: 0 signal at every tested")
        print("  configuration. A null result here (0 qualified seeds) replicates")
        print("  that; ANY qualified seed would be a new, reportable finding.")
    print()

    t0 = time.time()
    s1_tasks = [(s, _SITE_PACKAGES, args.repeats, args.backend, args.api_key, args.culture_id) for s in seed_list]
    with Pool(workers) as pool:
        s1_results = pool.map(_stage1, s1_tasks)
    s1_survivors = [r["seed"] for r in s1_results if r["stage1_pass"]]
    print(f"  STAGE 1: {len(s1_survivors)}/{len(seed_list)} passed: {s1_survivors}")

    if not s1_survivors:
        elapsed = time.time() - t0
        print(f"\n  done in {elapsed:.1f}s. No seed passed stage 1 — consistent with")
        print("  prior simulation findings. See AMAC_category00.md for what this means.")
        return

    s23_tasks = [(s, args.trials, _SITE_PACKAGES, args.decoder_lr, args.stdp_lr,
                 args.backend, args.api_key, args.culture_id) for s in s1_survivors]
    with Pool(workers) as pool:
        s23_results = pool.map(_stage2_and_3, s23_tasks)

    elapsed = time.time() - t0
    qualified = [r for r in s23_results if r["stage2_pass"] and r["stage3_pass"]]

    print(f"\n  done in {elapsed:.1f}s total")
    print()
    print(f"  {'seed':<6}{'stage2':<16}{'stage3 (repeat)':<16}")
    print("  " + "-" * 40)
    for r in sorted(s23_results, key=lambda x: x["seed"]):
        st2 = f"{'PASS' if r['stage2_pass'] else 'fail'} ({r['stage2_acc']:.0%})"
        st3 = f"{'PASS' if r['stage3_pass'] else 'fail'} ({r['stage3_acc']:.0%})"
        print(f"  {r['seed']:<6}{st2:<16}{st3:<16}")

    print()
    print("=" * 70)
    if qualified:
        print(f"  QUALIFIED seeds: {[r['seed'] for r in qualified]}")
        print("  This would be a new finding relative to this project's prior")
        print("  simulation history — see AMAC_category00.md, 'H2' branch.")
    else:
        print("  No seed passed all 3 stages — consistent with prior simulation")
        print("  findings ('H1' in AMAC_category00.md).")
    print("=" * 70)


if __name__ == "__main__":
    main()
