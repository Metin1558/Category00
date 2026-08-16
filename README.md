# category00 — The Simplest Possible Categorical Task, Hardware-Ready

**X fires electrode 0, once. Y fires electrode 1, once. No magnitude axis, no shortcut — the exact design that returned zero signal across every tested configuration in simulation.**

Metin (ORCID: 0009-0006-4635-405X) · August 2026

---

## Why this exists — read AMAC_category00.md first

This experiment is designed to answer one question, written down *before* any of this code was built: was the earlier zero-signal finding a fact about the learning rule (STDP doesn't drive categorical learning at this scale), or an artifact of this project's simplified simulation model (homogeneous thresholds, no real cell-type diversity)? See `AMAC_category00.md` for the full hypothesis and the decision rule this result feeds into — written before the code, not after.

## Status

| Component | Status |
|---|---|
| Live panel + video recording | **Built, verified against real core/sim** |
| Calibration (`calibrate_category00.py`) | **Built, verified** — R-sweep, raw X-vs-Y separation check |
| Seed qualification (`qualify_seed_category00.py`) | **Built, verified** — 3-stage screen (separability, trained accuracy, repeatability) |
| Simulation result (0 seeds qualified, 0/10 at stage 1) | Consistent with this project's prior falsification-ladder history |
| Real-tissue run | **Not yet performed** |

## Run it

```bash
pip install numpy scipy --break-system-packages

python3 calibrate_category00.py --repeats 5       # first, cheapest check
python3 qualify_seed_category00.py --seeds 20      # which seeds (if any) show a real effect
python3 category00_demo.py --train-minutes 2 --live-trials 40   # live panel + video
```

### Against real hardware

```bash
python3 calibrate_category00.py --backend hardware --api-key ... --culture-id ... --repeats 5
```

Run this **first**. `qualify_seed_category00.py --backend hardware` forces sequential execution (one live session, not parallel workers) automatically.

## Architecture

```
category00_demo.py            Stimulus definition, training, live-panel demo
calibrate_category00.py       First real-hardware check — R-sweep, raw separability
qualify_seed_category00.py    3-stage seed screening
panel_category00/             Live panel + automatic video recording
AMAC_category00.md            Hypothesis and decision rule — written before the code
core/, sim/                   Closed-loop simulation engine — shared, task-agnostic
hardware/                     FinalSpark hardware adapter — same interface as sim/
```

Independent of, and built from a clean copy of, this author's other organoid-oi work — shares only the general-purpose closed-loop engine, no task-specific code.

## Data, code

https://github.com/Metin1558/Category00

---

*Not peer reviewed. August 2026.*
