"""
oi_loop.py — organoid_oi_v3
=============================
Closed-loop experiment orchestrator — v3.

v3 changes:
    A. HebbianDecoder: local Hebbian rule with weight decay (replaces FrozenDecoder)
    B. Epoch-based homeostatic: applied every N trials, not every trial
    C. Blanking period: suppress spikes during stimulus artifact window

Fix A — Hebbian Downstream Decoder:
    Biologically motivated local learning rule for the readout layer.
    Weight update: dw = lr * (post_rate * pre_rate) - decay * w
    This is BCM-like Hebbian: strengthen connections that co-activate,
    with a decay term preventing runaway potentiation.

    Unlike the v2 frozen decoder, this allows the readout to adapt
    to whatever representation the organoid develops via STDP —
    while keeping learning local (no backpropagation).

    Unlike the v1 trained decoder, this does NOT use error signals
    or target labels directly — only firing rate co-activity.

Fix B — Epoch-based Homeostatic:
    homeostatic_scaling() called every config.homeostatic_epoch trials.
    Default: every 10 trials. Allows STDP updates to consolidate
    before normalization washes them out.

Fix C — Blanking Period:
    During the first blanking_ms of the stimulus window,
    organoid spikes are suppressed (set to empty).
    Simulates artifact rejection used in real MEA recordings —
    stimulus current bleeds into recording electrodes for ~5ms.
"""

import numpy as np
import time as time_module
from typing import List, Optional, Dict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from oi_types import (
    StimulusPattern, OrganoidResponse, RewardSignal,
    Trial, ExperimentConfig, STDPUpdate
)
from oi_stdp import compute_stdp_update, apply_update, homeostatic_scaling
from oi_reward import compute_reward, neuromodulator_factor


# ─────────────────────────────────────────────────────────────
# BLANKING PERIOD (fix C)
# ─────────────────────────────────────────────────────────────

def apply_blanking(
    spike_times: List[np.ndarray],
    blanking_ms: float,
    offset_s: float = 0.0,
) -> List[np.ndarray]:
    """
    Remove spikes within blanking_ms of stimulus onset.

    Simulates artifact rejection in real MEA recordings.
    Stimulus electrical pulse contaminates the recording electrode
    for ~2-10ms after delivery.

    Args:
        spike_times : list of spike time arrays
        blanking_ms : suppress spikes within this window from offset_s
        offset_s    : stimulus onset time (default 0)

    Returns:
        blanked_spike_times : spikes outside blanking window only
    """
    blanking_s = blanking_ms / 1000.0
    return [
        spikes[spikes > offset_s + blanking_s]
        for spikes in spike_times
    ]


# ─────────────────────────────────────────────────────────────
# HEBBIAN DECODER (fix A)
# ─────────────────────────────────────────────────────────────

class HebbianDecoder:
    """
    Local Hebbian readout layer — biologically motivated.

    Update rule (BCM-like):
        dw_cj = lr * r_c * r_j - decay * w_cj

    Where:
        r_c = category activation (1 if true label, 0 otherwise)
        r_j = neuron j firing rate
        decay = weight decay preventing runaway

    This is a supervised Hebb rule — uses the true label to
    compute r_c, but the update is local (no error backprop).
    Biologically: analogous to neuromodulator-gated Hebbian
    plasticity in cortical readout areas.

    Note on separability:
        Unlike frozen random projection, Hebbian decoder adapts
        to the organoid's emerging representation. As STDP
        reorganizes organoid weights, the decoder follows.
        This makes the system more robust to poor random initializations.
    """

    def __init__(
        self,
        categories: List[str],
        n_neurons: int,
        config: ExperimentConfig = None,
        rng: np.random.Generator = None,
    ):
        self.categories = categories
        self.n_cats = len(categories)
        self.n_neurons = n_neurons
        self.config = config or ExperimentConfig()
        self.rng = rng or np.random.default_rng(self.config.decoder_seed)

        # Small random initialization (not frozen — will be updated)
        self.weights = self.rng.normal(
            0, 0.01, size=(self.n_cats, n_neurons))

        # Track weight history for analysis
        self._update_count = 0

    def firing_rates(self, response: OrganoidResponse) -> np.ndarray:
        """Mean firing rate per neuron [Hz]."""
        rates = np.zeros(self.n_neurons)
        dur = max(response.duration_s, 1e-6)
        for j, spikes in enumerate(response.spike_times):
            if j >= self.n_neurons:
                break
            rates[j] = len(spikes) / dur
        return rates

    def decode(self, response: OrganoidResponse) -> tuple:
        """
        Decode category via linear readout + softmax.
        Returns (prediction, confidence).
        """
        rates = self.firing_rates(response)
        activations = self.weights @ rates      # [n_cats]
        exp_act = np.exp(activations - activations.max())
        probs = exp_act / (exp_act.sum() + 1e-10)
        best = int(np.argmax(probs))
        return self.categories[best], float(probs[best])

    def update(
        self,
        response: OrganoidResponse,
        true_label: str,
        learning_rate: float = None,
    ):
        """
        Hebbian weight update with decay.

        dw_cj = lr * r_c * r_j - decay * w_cj

        r_c = 1 for true category, 0 for others
        r_j = firing rate of neuron j
        """
        if learning_rate is None:
            learning_rate = self.config.decoder_lr

        rates = self.firing_rates(response)
        true_idx = self.categories.index(true_label)

        # Target activation: 1 for correct, 0 for others
        r_c = np.zeros(self.n_cats)
        r_c[true_idx] = 1.0

        # Hebbian: outer product of category activation and neuron rates
        dw = learning_rate * np.outer(r_c, rates)

        # Weight decay (prevents runaway)
        decay = self.config.decoder_hebbian_decay * self.weights

        self.weights += dw - decay
        self._update_count += 1

    def weight_stats(self) -> dict:
        return {
            'mean': float(self.weights.mean()),
            'std': float(self.weights.std()),
            'max_abs': float(np.abs(self.weights).max()),
            'updates': self._update_count,
        }


# ─────────────────────────────────────────────────────────────
# CLOSED-LOOP EXPERIMENT v3
# ─────────────────────────────────────────────────────────────

class ClosedLoopExperiment:
    """
    Full closed-loop OI experiment — v3.

    New in v3:
        A. HebbianDecoder (adaptive local readout)
        B. Epoch-based homeostatic (every N trials)
        C. Blanking period (artifact suppression)
        D. Vectorized STDP (from oi_stdp v3)
    """

    def __init__(
        self,
        organoid,
        config: ExperimentConfig = None,
        rng: np.random.Generator = None,
    ):
        self.organoid = organoid
        self.config = config or ExperimentConfig()
        self.rng = rng or np.random.default_rng(self.config.random_seed)

        # v3: Hebbian decoder (adaptive, local rule)
        self.decoder = HebbianDecoder(
            categories=self.config.categories,
            n_neurons=self.config.n_neurons,
            config=self.config,
            rng=self.rng,
        )

        self.trial_history: List[Trial] = []
        self._experiment_start = time_module.time()
        self._trial_count = 0  # for epoch-based homeostatic

    def elapsed_s(self) -> float:
        return time_module.time() - self._experiment_start

    def run_trial(
        self,
        stimulus: StimulusPattern,
        trial_id: int,
        verbose: bool = False,
    ) -> Trial:
        """
        One complete v3 closed-loop trial.

        Steps:
            1. Phase 1: organoid responds to stimulus
            2. Apply blanking period (artifact suppression)
            3. Decode blanked response → prediction
            4. Compute reward + current injection
            5. Phase 2: organoid responds to reward current
            6. Three-Factor STDP on (blanked Phase1 + Phase2)
            7. Apply STDP update (no homeostatic here)
            8. Update Hebbian decoder
            9. Epoch-based homeostatic (every N trials)
        """
        t = self.elapsed_s()

        # 1. Phase 1: stimulus
        response_p1 = self.organoid.respond(
            stimulus, timestamp=t, post_stimulus_current=None)

        # 2. Blanking period (fix C)
        if self.config.blanking_enabled:
            blanked_spikes = apply_blanking(
                response_p1.spike_times,
                blanking_ms=self.config.blanking_ms,
                offset_s=0.0,
            )
            response_blanked = OrganoidResponse(
                spike_times=blanked_spikes,
                n_neurons=response_p1.n_neurons,
                duration_s=response_p1.duration_s,
                timestamp=t,
            )
        else:
            response_blanked = response_p1

        # 3. Decode from blanked response
        prediction, confidence = self.decoder.decode(response_blanked)
        correct = (prediction == stimulus.label)

        # 4. Reward signal
        reward = compute_reward(
            prediction=prediction,
            ground_truth=stimulus.label,
            timestamp=self.elapsed_s(),
            config=self.config,
            rng=self.rng,
        )

        # 5. Phase 1 + Phase 2: full response with reward current
        response_full = self.organoid.respond(
            stimulus,
            timestamp=t,
            post_stimulus_current=reward.current_injection,
        )

        # Apply blanking to full response too
        if self.config.blanking_enabled:
            full_spikes = apply_blanking(
                response_full.spike_times,
                blanking_ms=self.config.blanking_ms,
            )
            response_full.spike_times = full_spikes

        response_full.prediction = prediction
        response_full.confidence = confidence

        # 6. Three-Factor STDP (vectorized)
        neuromod = neuromodulator_factor(reward, self.config)
        stdp = compute_stdp_update(
            pre_spike_times=stimulus.spike_times,
            post_spike_times=response_full.spike_times,
            config=self.config,
            neuromodulator=neuromod,
        )

        # 7. Apply STDP (no homeostatic here — epoch-based below)
        self.organoid.weights = apply_update(
            self.organoid.weights, stdp,
            learning_rate=self.config.learning_rate,
        )

        # 8. Hebbian decoder update (fix A)
        self.decoder.update(
            response_blanked,
            true_label=stimulus.label,
        )

        # 9. Epoch-based homeostatic (fix B)
        self._trial_count += 1
        if (self.config.homeostatic and
                self._trial_count % self.config.homeostatic_epoch == 0):
            target = self.config.target_weight_sum
            if target is None:
                target = self.config.n_electrodes * 0.5
            self.organoid.weights = homeostatic_scaling(
                self.organoid.weights,
                target_sum=target,
                rate=self.config.homeostatic_rate,
            )

        # Record
        trial = Trial(
            trial_id=trial_id,
            stimulus=stimulus,
            response=response_full,
            reward=reward,
            correct=correct,
            timestamp=t,
            response_p1=response_blanked,
        )
        self.trial_history.append(trial)

        if verbose:
            self._print_trial(trial, stdp)

        return trial

    def _print_trial(self, trial: Trial, stdp: STDPUpdate):
        correct_str = 'CORRECT' if trial.correct else 'wrong  '
        blanked = sum(len(s) for s in trial.response.spike_times)
        print(
            f"  Trial {trial.trial_id:4d} | "
            f"{trial.stimulus.label:8s} → {trial.response.prediction:8s} | "
            f"{correct_str} | "
            f"conf={trial.response.confidence:.2f} | "
            f"neuromod={stdp.neuromodulator_factor:+.2f} | "
            f"spikes={blanked:4d}"
        )

    def accuracy(self, last_n: int = None) -> float:
        if not self.trial_history:
            return 0.0
        trials = self.trial_history[-last_n:] if last_n else self.trial_history
        return sum(t.correct for t in trials) / len(trials)

    def per_category_accuracy(self) -> Dict[str, float]:
        result = {}
        for cat in self.config.categories:
            cat_trials = [t for t in self.trial_history if t.stimulus.label == cat]
            result[cat] = (sum(t.correct for t in cat_trials) / len(cat_trials)
                           if cat_trials else 0.0)
        return result

    def learning_curve(self, window: int = 20) -> np.ndarray:
        if not self.trial_history:
            return np.array([])
        correct = np.array([t.correct for t in self.trial_history], dtype=float)
        return np.convolve(correct, np.ones(window)/window, mode='same')

    def freeze_weights_test(
        self,
        stimuli: List[StimulusPattern],
        verbose: bool = False,
    ) -> float:
        """Freeze organoid weights, measure accuracy. Validates STDP as learning source."""
        saved = self.organoid.weights.copy()
        results = []
        for stim in stimuli:
            resp = self.organoid.respond(stim, timestamp=0.0)
            if self.config.blanking_enabled:
                resp.spike_times = apply_blanking(
                    resp.spike_times, self.config.blanking_ms)
            pred, _ = self.decoder.decode(resp)
            results.append(pred == stim.label)
        self.organoid.weights = saved
        acc = sum(results) / len(results) if results else 0.0
        if verbose:
            print(f"  Freeze-weights accuracy: {acc:.1%} "
                  f"(chance: {1/len(self.config.categories):.1%})")
        return acc


# Import for type hint
from core.oi_types import OrganoidResponse

