"""
category00_demo.py — The simplest possible categorical task: X fires
electrode 0, Y fires electrode 1, single pulse each. No word length, no
expansion rate — no continuous magnitude to lean on at all. This is the
exact task that returned zero signal in simulation, across every tested
seed, at every tested drive level. Presented here for a real-hardware
re-test — see AMAC_category00.md for the hypothesis and decision rule this
experiment is designed to answer, written BEFORE this code.

STATUS
------
Structurally hardware-ready (same interface as the rest of this
project), NOT YET RUN ON REAL TISSUE. --backend synthetic is default.

This script trains a readout via reward/penalty (starting blind, same
pattern as sentence_demo.py / live_diagnosis.py), then runs a live
diagnosis loop with a panel: watch a fresh X/Y trial, see the organoid's
live call next to it.

USAGE
-----
    python3 category00_demo.py --train-minutes 2 --live-trials 40
    python3 category00_demo.py --backend hardware --api-key ... --culture-id ...
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
for sub in ["", "core", "sim", "hardware"]:
    sys.path.insert(0, str(ROOT / sub))

import numpy as np


def xy_stimulus(label, cfg, rng):
    """
    The simplest possible categorical stimulus. X -> electrode 0 fires
    once. Y -> electrode 1 fires once. No magnitude axis, no shortcut —
    exactly the design that returned zero signal in simulation.
    """
    from core.oi_types import StimulusPattern
    n_el = cfg.n_electrodes
    duration = cfg.stimulus_duration_s
    traces = [np.array([]) for _ in range(n_el)]
    electrode_idx = 0 if label == "X" else 1
    t = duration * 0.4 + rng.normal(0, 0.002)
    traces[electrode_idx] = np.array([max(0.0, min(t, duration - 1e-4))])
    return StimulusPattern(spike_times=traces, n_electrodes=n_el, duration_s=duration, label=label)


def make_config(**kw):
    from core.oi_types import ExperimentConfig
    kw.setdefault("categories", ["X", "Y"])
    kw.setdefault("n_electrodes", 32)
    kw.setdefault("stimulus_duration_s", 0.5)
    kw.setdefault("v_thresh_jitter_mv", 5.0)
    kw.setdefault("decoder_lr", 0.05)
    kw.setdefault("r_membrane_mohm", 30.0)  # STARTING point only, unverified for this task — see calibrate_category00.py
    return ExperimentConfig(**kw)


def build_organoid(backend, cfg, rng, api_key=None, culture_id=None):
    if backend == "hardware":
        from finalspark_organoid import FinalSparkOrganoid
        return FinalSparkOrganoid(cfg, api_key=api_key, culture_id=culture_id, rng=rng)
    from sim.oi_synth import SyntheticOrganoid
    return SyntheticOrganoid(cfg, rng=rng)


def train_readout(cfg, rng, backend, api_key, culture_id, budget_seconds, seconds_per_trial, panel=None):
    from core.oi_loop import ClosedLoopExperiment
    organoid = build_organoid(backend, cfg, rng, api_key, culture_id)
    exp = ClosedLoopExperiment(organoid, cfg, rng=rng)

    n_trials = max(1, int(budget_seconds // seconds_per_trial))
    for i in range(n_trials):
        label = "X" if rng.random() < 0.5 else "Y"
        stim = xy_stimulus(label, cfg, rng)
        exp.run_trial(stim, i)
        if panel and i % 5 == 0:
            panel.push(phase="training", train_progress=(i + 1) / n_trials)
    return exp, organoid


def diagnose_one_trial(exp, organoid, cfg, rng):
    label = "X" if rng.random() < 0.5 else "Y"
    stim = xy_stimulus(label, cfg, rng)
    response = organoid.respond(stim, timestamp=0.0)
    firing = [len(x) for x in response.spike_times]

    decoder = exp.decoder
    rates_vec = decoder.firing_rates(response)
    activations = decoder.weights @ rates_vec
    exp_act = np.exp(activations - activations.max())
    probs = exp_act / (exp_act.sum() + 1e-10)
    best_idx = int(np.argmax(probs))
    diagnosis = decoder.categories[best_idx]
    confidence = float(probs[best_idx])

    return {
        "true_label": label,
        "diagnosis": diagnosis,
        "correct": diagnosis == label,
        "confidence": confidence,
        "total_firing": sum(firing),
        "active_electrodes": sum(1 for f in firing if f > 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-minutes", type=float, default=2.0)
    ap.add_argument("--seconds-per-trial", type=float, default=8.0,
                    help="ESTIMATED real seconds per organoid trial — not a measured hardware value yet.")
    ap.add_argument("--live-trials", type=int, default=40)
    ap.add_argument("--trial-interval", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["synthetic", "hardware"], default="synthetic")
    ap.add_argument("--api-key", type=str, default=None)
    ap.add_argument("--culture-id", type=str, default=None)
    ap.add_argument("--no-panel", action="store_true")
    ap.add_argument("--port", type=int, default=8903)
    args = ap.parse_args()

    panel = None
    if not args.no_panel:
        sys.path.insert(0, str(ROOT / "panel_category00"))
        import server as panel
        panel.start(port=args.port)
        time.sleep(1.0)

    print("=" * 64)
    print("  CATEGORY00 — the simplest possible categorical task, live")
    print(f"  backend: {args.backend}")
    print("=" * 64)

    rng = np.random.default_rng(args.seed)
    cfg = make_config()

    print(f"\n  TRAINING ({args.train_minutes} min budget, readout starts blind)")
    if panel:
        panel.push(phase="training", train_progress=0.0)
    exp, organoid = train_readout(cfg, rng, args.backend, args.api_key, args.culture_id,
                                  args.train_minutes * 60, args.seconds_per_trial, panel=panel)
    print("  training complete.")

    print(f"\n  LIVE DIAGNOSIS — {args.live_trials} trials")
    if panel:
        panel.push(phase="live", live_progress=0.0)

    correct_count = 0
    for i in range(args.live_trials):
        result = diagnose_one_trial(exp, organoid, cfg, rng)
        correct_count += int(result["correct"])
        tag = "correct" if result["correct"] else "WRONG"
        print(f"  trial {i+1:3d}: true={result['true_label']}  diagnosis={result['diagnosis']} "
              f"({result['confidence']:.0%})  [{tag}]")
        if panel:
            panel.push(phase="live", live_progress=(i + 1) / args.live_trials,
                      current_trial=result, running_accuracy=correct_count / (i + 1))
        time.sleep(args.trial_interval)

    print(f"\n  live accuracy: {correct_count}/{args.live_trials} ({correct_count/args.live_trials:.0%})")
    print("  (chance = 50%. See AMAC_category00.md for the decision rule this result feeds into.)")

    if panel:
        panel.push(phase="complete", finished=True)
        print("\n  [panel] recording will auto-save now — leave the browser tab open a moment.")
        time.sleep(3.0)


if __name__ == "__main__":
    main()
