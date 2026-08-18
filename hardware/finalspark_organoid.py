"""
hardware/finalspark_organoid.py — Hardware adapter for FinalSpark's
Neuroplatform, implementing the SAME interface as sim.oi_synth.SyntheticOrganoid.

WHY THIS FILE EXISTS
---------------------
Every simulation-side script in this project's family (sentence_demo.py,
calibrate.py, sample_efficiency.py, live_diagnosis.py, category00_demo.py,
etc.) was built against one contract:

    organoid.respond(stimulus, timestamp, post_stimulus_current=None) -> response

`response` must expose `.spike_times` — a list of arrays, one per
electrode, each array the spike timestamps (in seconds, relative to the
trial start) that electrode recorded.

If this file correctly implements that same contract against FinalSpark's
real API, every script above runs UNCHANGED on real tissue — only the
`--backend hardware` flag changes.

STATUS — READ THIS BEFORE USING (August 2026 update)
-------------------------------------------------------
This version replaces an earlier, more speculative draft. The stimulation
path (`_stimulate`) is now written directly against FinalSpark's real,
published API documentation (finalspark-np.github.io/np-docs/np_core/
np_usage.html) — the class names, method signatures, and StimParam fields
below are taken directly from that page, not guessed.

The reading path (`_read_spikes`) and reward path (`_inject_current`) are
still BEST-EFFORT: the docs confirm a Spike DB exists ("active 24/7...will
record spike events from the electrodes") and reference a separate
"Database tutorial" page this project has not yet been able to fetch the
full contents of, and confirm UV-uncaging exists for dopamine-reward
delivery but not its exact Python method signature. Both are marked
clearly below with TODOs — do not trust their exact method names without
checking the Database and peripheral-control pages in the docs first.

IMPORTANT NEW CONSTRAINT FOUND IN THE DOCS — READ BEFORE BUDGETING SESSION TIME
-----------------------------------------------------------------------------------
Sending stimulation parameters to the Intan software (`intan.send_stimparam()`)
takes 10 SECONDS and "cannot be bypassed" — the docs state this explicitly:
"parameters have to be set before your experiment starts, or during a
ten-second pause in your experiment." If every trial in an experiment uses
a genuinely different set of active electrodes (as sentence_demo.py,
sample_efficiency.py, and category00_demo.py all do — a new word / clip /
label each trial), this 10-second cost is paid EVERY TRIAL, not once.

This directly updates the `--seconds-per-trial` ESTIMATE used throughout
this project's calibration and qualification scripts (previously 5-8s).
Budget for at least ~10-12s/trial until measured directly in a real
session, and re-run calibrate_*.py's timing-sensitivity check
(--seconds-per-trial 8,15) with this in mind before trusting any hour
budget calculated earlier.

WHAT TO DO BEFORE THE FIRST REAL SESSION
------------------------------------------
1. Confirm the Database-tutorial page's exact spike-read method and the
   UV-uncaging trigger's exact method name/signature, and fill in the two
   remaining TODOs.
2. Run calibrate_*.py --backend hardware --repeats 3 FIRST — cheap, short,
   confirms the electrode/session plumbing before spending real budget.
3. You will need a real experiment TOKEN from FinalSpark (see `token`
   argument below) — this is separate from any api_key/culture_id concept
   used elsewhere in this project; kept as aliases below for interface
   compatibility with the rest of this project's scripts.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class _HardwareResponse:
    """Matches sim.oi_synth's OrganoidResponse shape exactly."""
    spike_times: List[np.ndarray]
    n_neurons: int
    duration_s: float
    timestamp: float


class FinalSparkOrganoid:
    """
    Drop-in replacement for sim.oi_synth.SyntheticOrganoid.

    Construct with the same ExperimentConfig used elsewhere in this
    project, plus a FinalSpark experiment token. `respond()` has the
    identical signature and return shape as SyntheticOrganoid.respond().

    NOTE ON ARGUMENT NAMES: the rest of this project's scripts pass
    `api_key` and `culture_id` (a naming convention established before
    this file was written against the real API). FinalSpark's actual
    auth model uses a single `token` string tied to a specific booked
    experiment slot. `api_key` is accepted and used AS the token for
    compatibility; `culture_id` is currently unused by the real API
    (FinalSpark's `Experiment` object exposes which electrodes/organoids
    you have access to via `exp.electrodes`, not a separate culture ID).
    """

    STIM_TRIGGER_KEY = 0  # reserved trigger index for all stimulation in this project

    def __init__(self, config, session=None, api_key: Optional[str] = None,
                culture_id: Optional[str] = None, rng=None):
        self.config = config
        self.rng = rng or np.random.default_rng(0)
        self.token = api_key  # see NOTE above
        self._culture_id = culture_id  # currently unused by the real API; kept for interface compatibility
        self._exp = None
        self._intan = None
        self._trigger_gen = None
        self._started = False

        if session is not None:
            # allow an already-constructed, already-started Experiment to be reused
            self._exp = session
            self._started = True
        elif self.token is None:
            raise ValueError(
                "FinalSparkOrganoid needs either a live `session` (an already-"
                "started neuroplatform.Experiment), or a `api_key` used as the "
                "experiment token. Nothing is connected yet at construction "
                "time — see module docstring."
            )
        else:
            self._connect()

    # ------------------------------------------------------------------
    def _connect(self):
        """
        Opens the real connection. Confirmed against np_usage.html:
            from neuroplatform import StimParam, IntanSofware, Trigger, StimPolarity, Experiment
            exp = Experiment(token)
            exp.start()   # signals the start of an experiment to all users; use try/finally with exp.stop()
        """
        from neuroplatform import Experiment, IntanSofware, Trigger  # noqa: F401 (import kept local — only needed on real hardware)
        self._exp = Experiment(self.token)
        if not self._exp.start():
            raise RuntimeError(
                "FinalSpark Experiment.start() returned False — another experiment "
                "may currently be running on this token/slot. See the docs: "
                "'If you are unable to start your experiment during your booking "
                "because another experiment is running, please contact us.'"
            )
        self._started = True
        self._intan = IntanSofware()
        self._trigger_gen = Trigger()

    def close(self):
        """Call this when done — mirrors the docs' required try/finally cleanup pattern."""
        if self._trigger_gen is not None:
            self._trigger_gen.close()
        if self._intan is not None:
            self._intan.close()
        if self._exp is not None and self._started:
            self._exp.stop()
        self._started = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def respond(self, stimulus, timestamp: float,
               post_stimulus_current: Optional[np.ndarray] = None,
               dt_ms: float = 1.0) -> _HardwareResponse:
        """
        Same two-phase contract as SyntheticOrganoid.respond():
          Phase 1 — deliver `stimulus` (per-electrode spike times), record
                    the culture's response for stimulus.duration_s.
          Phase 2 — if `post_stimulus_current` is given (reward/penalty
                    waveform), deliver it, record the response for its
                    duration too.
        """
        self._stimulate(stimulus)
        phase1 = self._read_spikes(duration_s=stimulus.duration_s)

        phase2 = []
        total_duration = stimulus.duration_s
        if post_stimulus_current is not None and len(post_stimulus_current) > 0:
            self._inject_current(post_stimulus_current, dt_ms=dt_ms)
            extra_duration = len(post_stimulus_current) * dt_ms / 1000.0
            phase2 = self._read_spikes(duration_s=extra_duration)
            phase2 = [t + stimulus.duration_s for t in phase2]
            total_duration += extra_duration

        combined = [np.concatenate([p1, p2]) if len(phase2) else p1
                   for p1, p2 in zip(phase1, phase2 or [np.array([])] * len(phase1))]

        return _HardwareResponse(
            spike_times=combined,
            n_neurons=len(combined),
            duration_s=total_duration,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------
    def _stimulate(self, stimulus) -> None:
        """
        CONFIRMED against np_usage.html — not a placeholder.

        Maps this project's per-electrode `stimulus.spike_times` (one
        array per electrode, each array the relative-second offsets
        within the stimulus window at which that electrode should fire)
        onto FinalSpark's StimParam + trigger model:

          - One StimParam per ACTIVE electrode (electrodes with >=1 spike
            time). All active electrodes share ONE trigger_key
            (STIM_TRIGGER_KEY) so a single trigger firing activates all
            of them in one call — sidestepping the 16-trigger-key limit,
            which this project's stimuli (up to 32 active electrodes)
            would otherwise exceed.
          - Each electrode's relative timing is encoded via that
            StimParam's `trigger_delay` field (in microseconds), so all
            electrodes fire at their correct relative offset from ONE
            simultaneous trigger event, rather than needing multiple
            trigger_gen.send() calls (which the docs note is LESS precise
            than pulse-train/delay settings — "using Python to time your
            stimulation...will always be less precise").
          - Only the FIRST spike time per electrode is used — this
            project's stimulus encoders (word_stimulus, expansion_stimulus,
            xy_stimulus) all currently emit exactly one spike per active
            electrode per trial.

        COST: intan.send_stimparam() takes 10 seconds and cannot be
        shortened (see module docstring). This happens every call since
        this project's stimuli change which electrodes are active from
        trial to trial.
        """
        from neuroplatform import StimParam, StimPolarity

        params = []
        for electrode_idx, spike_times in enumerate(stimulus.spike_times):
            if len(spike_times) == 0:
                continue
            p = StimParam()
            p.enable = True
            p.index = electrode_idx
            p.trigger_key = self.STIM_TRIGGER_KEY
            p.polarity = StimPolarity.NegativeFirst
            # charge-balanced default pulse — confirm against your specific
            # experiment's safe amplitude range before the first real session
            p.phase_duration1 = 100.0   # us
            p.phase_amplitude1 = 1.0    # uA
            p.phase_duration2 = 100.0   # us
            p.phase_amplitude2 = 1.0    # uA  (balanced: d1*a1 == d2*a2, per docs' recommendation)
            p.trigger_delay = float(spike_times[0] * 1_000_000)  # s -> us, relative to the trigger
            params.append(p)

        if not params:
            return  # nothing active this trial — nothing to send/trigger

        self._intan.send_stimparam(params)  # blocks ~10s, per docs

        trigger_array = np.zeros(16, dtype=np.uint8)
        trigger_array[self.STIM_TRIGGER_KEY] = 1
        self._trigger_gen.send(trigger_array)

        # disable these StimParams so the next trial's set doesn't collide
        # with leftover enabled params on the same indices
        for p in params:
            p.enable = False
        self._intan.send_stimparam(params)

    def _read_spikes(self, duration_s: float) -> List[np.ndarray]:
        """
        TODO — NOT YET CONFIRMED against the Database-tutorial page (this
        project has only confirmed the page exists and its general
        description: "active 24/7...will record spike events from the
        electrodes", threshold = 6x noise std). Fill in the exact method
        once that page's contents are available — expected shape, based
        on the general pattern of the rest of this API:

            events = self._exp.database.get_spike_events(  # method name UNCONFIRMED
                start=<now>, stop=<now + duration_s>,
            )
            # events likely indexed by electrode; convert each electrode's
            # event timestamps to seconds relative to this window's start
        """
        raise NotImplementedError(
            "FinalSparkOrganoid._read_spikes: the general Spike DB exists "
            "(confirmed in the docs) but the exact Python read method has "
            "not yet been confirmed against the Database-tutorial page. "
            "Fetch finalspark-np.github.io/np-docs and find that page "
            "before running this against live tissue."
        )

    def _inject_current(self, waveform: np.ndarray, dt_ms: float) -> None:
        """
        TODO — NOT YET CONFIRMED. FinalSpark's platform delivers
        reward/penalty via UV-uncaging of caged neurotransmitters (most
        notably dopamine) rather than direct current injection — this is
        architecturally different from the synthetic organoid's reward
        model (10 Hz theta / 200 Hz noise current), and will need its own
        translation layer, not just a method-name fill-in. Confirm the
        exact trigger method and timing against the docs' peripheral-
        control pages before implementing.
        """
        raise NotImplementedError(
            "FinalSparkOrganoid._inject_current: FinalSpark uses UV-uncaging "
            "for reward delivery (confirmed to exist), not current injection "
            "— this needs a translation layer, not just an API call. See "
            "docstring."
        )
