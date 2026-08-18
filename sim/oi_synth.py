"""
oi_synth.py — organoid_oi_v2
==============================
Synthetic LIF organoid with reward current injection.

v2 changes (peer review fix #1, #4):
    - respond() now accepts post_stimulus_current for reward injection
    - Reward delivered as actual input current AFTER stimulus window
    - homeostatic_scaling integrated into update_weights()
    - weight_history tracked for divergence analysis
"""

import numpy as np
from typing import List, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.oi_types import StimulusPattern, OrganoidResponse, RewardSignal, ExperimentConfig
from core.oi_stdp import homeostatic_scaling


def simulate_lif_neuron(
    input_current: np.ndarray,
    dt_s: float,
    v_rest_mv: float = -70.0,
    v_thresh_mv: float = -55.0,
    v_reset_mv: float = -75.0,
    tau_membrane_ms: float = 20.0,
    refractory_ms: float = 2.0,
    r_membrane_mohm: float = 10.0,
) -> np.ndarray:
    """
    Leaky Integrate-and-Fire neuron simulation.
    Returns spike mask (bool array).
    """
    n = len(input_current)
    v = v_rest_mv
    tau_s = tau_membrane_ms / 1000.0
    refrac_steps = int(refractory_ms / 1000.0 / dt_s)
    spikes = np.zeros(n, dtype=bool)
    refrac_count = 0

    for i in range(n):
        if refrac_count > 0:
            v = v_reset_mv
            refrac_count -= 1
            continue
        dv = (-(v - v_rest_mv) + r_membrane_mohm * input_current[i]) * (dt_s / tau_s)
        v += dv
        if v >= v_thresh_mv:
            spikes[i] = True
            v = v_reset_mv
            refrac_count = refrac_steps

    return spikes


def psp_kernel(t: np.ndarray, spike_time: float,
               tau_syn_ms: float = 5.0, amplitude: float = 1.0) -> np.ndarray:
    dt = t - spike_time
    return np.where(dt >= 0, amplitude * np.exp(-dt / (tau_syn_ms / 1000.0)), 0.0)


class SyntheticOrganoid:
    """
    Population of LIF neurons with plastic weights and reward current injection.

    v2 key change:
        respond() runs in TWO phases:
            Phase 1 (0 → stimulus_duration_s): stimulus-driven activity
            Phase 2 (0 → reward_duration_s): reward/penalty current injection

        Phase 2 response spikes are combined with Phase 1 for STDP computation.
        This means the reward signal shapes post-synaptic timing naturally,
        without any external manipulation of weight direction.
    """

    def __init__(self, config: ExperimentConfig = None, rng=None):
        if config is None:
            config = ExperimentConfig()
        self.config = config
        self.rng = rng or np.random.default_rng(config.random_seed)

        self.n_electrodes = config.n_electrodes
        self.n_neurons = config.n_neurons

        self.weights = (
            0.5 + self.rng.normal(0, 0.05, (self.n_electrodes, self.n_neurons))
        ).clip(0, 1)

        self._noise_current_na = 0.05
        # Noron basina esik (jitter=0 ise hepsi ayni, eski davranis korunur)
        self.v_thresh_por_neuron = self.config.v_thresh_mv + self.rng.normal(
            0, self.config.v_thresh_jitter_mv, self.n_neurons)
        self.weight_history = []   # track for divergence analysis

    def _build_stimulus_current(
        self,
        stimulus: StimulusPattern,
        neuron_idx: int,
        t: np.ndarray,
        dt_s: float,
    ) -> np.ndarray:
        """Build synaptic input current for one neuron from stimulus."""
        current = np.zeros(len(t))
        for i, spikes in enumerate(stimulus.spike_times):
            w = self.weights[i, neuron_idx]
            if w < 1e-6 or len(spikes) == 0:
                continue
            for t_spike in spikes:
                if t_spike >= stimulus.duration_s:
                    continue
                current += w * psp_kernel(t, t_spike, tau_syn_ms=self.config.tau_syn_ms)
        return current

    def respond(
        self,
        stimulus: StimulusPattern,
        timestamp: float,
        post_stimulus_current: Optional[np.ndarray] = None,
        dt_ms: float = 1.0,
    ) -> OrganoidResponse:
        """
        Simulate organoid response in two phases.

        Phase 1: Stimulus window (stimulus.duration_s)
            Driven by weighted synaptic input from stimulus spike times.

        Phase 2: Reward window (reward_duration_s if post_stimulus_current given)
            Driven by reward/penalty current injection.
            This phase produces post-synaptic spikes whose timing relative
            to the stimulus trace drives STDP naturally.

        The combined spike train (Phase 1 + Phase 2) is returned.
        Phase 2 spikes are offset by stimulus.duration_s in time.

        Args:
            stimulus              : StimulusPattern
            timestamp             : experiment time
            post_stimulus_current : current array from oi_reward (nA)
                                    None = skip Phase 2
            dt_ms                 : simulation time step
        """
        dt_s = dt_ms / 1000.0
        all_spike_times = []

        for j in range(self.n_neurons):
            # ── Phase 1: stimulus ──────────────────────────────────────
            n1 = int(stimulus.duration_s / dt_s)
            t1 = np.arange(n1) * dt_s

            current1 = self._build_stimulus_current(stimulus, j, t1, dt_s)
            current1 += self.rng.normal(0, self._noise_current_na, n1)

            spikes1 = simulate_lif_neuron(
                current1, dt_s,
                v_rest_mv=self.config.v_rest_mv,
                v_thresh_mv=self.v_thresh_por_neuron[j],
                v_reset_mv=self.config.v_reset_mv,
                tau_membrane_ms=self.config.tau_membrane_ms,
                refractory_ms=self.config.refractory_ms,
                r_membrane_mohm=self.config.r_membrane_mohm,
            )
            times1 = t1[spikes1]

            # ── Phase 2: reward current ────────────────────────────────
            times2 = np.array([])
            if post_stimulus_current is not None and len(post_stimulus_current) > 0:
                n2 = len(post_stimulus_current)
                # Add small noise to reward current
                current2 = post_stimulus_current + self.rng.normal(
                    0, self._noise_current_na, n2)

                spikes2 = simulate_lif_neuron(
                    current2, dt_s,
                    v_rest_mv=self.config.v_rest_mv,
                    v_thresh_mv=self.v_thresh_por_neuron[j],
                    v_reset_mv=self.config.v_reset_mv,
                    tau_membrane_ms=self.config.tau_membrane_ms,
                    refractory_ms=self.config.refractory_ms,
                    r_membrane_mohm=self.config.r_membrane_mohm,
                )
                t2 = np.arange(n2) * dt_s
                # Offset by stimulus duration
                times2 = t2[spikes2] + stimulus.duration_s

            all_spike_times.append(np.concatenate([times1, times2]))

        total_dur = stimulus.duration_s
        if post_stimulus_current is not None:
            total_dur += len(post_stimulus_current) * dt_s

        return OrganoidResponse(
            spike_times=all_spike_times,
            n_neurons=self.n_neurons,
            duration_s=total_dur,
            timestamp=timestamp,
        )

    def update_weights(
        self,
        dw: np.ndarray,
        learning_rate: float = 0.01,
    ):
        """
        Apply weight update with homeostatic scaling.

        Steps:
            1. w += lr * dw
            2. clip [0, 1]
            3. homeostatic normalization (if config.homeostatic)
        """
        if dw.shape != self.weights.shape:
            raise ValueError(f"dw shape {dw.shape} != weights {self.weights.shape}")

        self.weights = np.clip(self.weights + learning_rate * dw, 0.0, 1.0)

        if self.config.homeostatic:
            target = self.config.target_weight_sum
            if target is None:
                target = self.n_electrodes * 0.5
            self.weights = homeostatic_scaling(
                self.weights, target, self.config.homeostatic_rate)

    def snapshot_weights(self):
        """Save current weight matrix to history."""
        self.weight_history.append(self.weights.copy())

    def weight_stats(self) -> dict:
        w = self.weights
        col_sums = w.sum(axis=0)
        return {
            'mean': float(w.mean()),
            'std': float(w.std()),
            'min': float(w.min()),
            'max': float(w.max()),
            'mean_col_sum': float(col_sums.mean()),
            'fraction_saturated_high': float((w >= 0.99).mean()),
            'fraction_saturated_low': float((w <= 0.01).mean()),
        }

    def reset_weights(self):
        self.weights = (
            0.5 + self.rng.normal(0, 0.05, (self.n_electrodes, self.n_neurons))
        ).clip(0, 1)
        self.weight_history = []

