"""
oi_reward.py — organoid_oi_v2
==============================
Reward and penalty signal generation.

v2 changes (peer review fix #1):
    - Reward/penalty now also produced as current injection (nA)
      for direct delivery to LIF neurons AFTER stimulus window
    - current_injection field added to RewardSignal
    - Neuromodulator factor returned for Three-Factor STDP

Biological rationale:
    Reward (10 Hz theta): sub-threshold periodic current
        → drives regular, coherent post-synaptic activity
        → aligns post-synaptic spikes with stimulus trace
        → favors LTP for recently active synapses

    Penalty (200 Hz noise): suprathreshold chaotic current
        → drives random, uncorrelated post-synaptic firing
        → destroys temporal relationship with stimulus
        → naturally produces LTD via STDP timing mismatch
        → no external "direction forcing" needed

This replaces v1's mathematical hack (dw * reward_scale).
The reward signal now acts as a biological neuromodulator
that shapes STDP outcomes through timing, not external override.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from oi_types import RewardSignal, ExperimentConfig


# ─────────────────────────────────────────────────────────────
# WAVEFORM GENERATORS
# ─────────────────────────────────────────────────────────────

def reward_waveform(
    duration_s: float,
    frequency_hz: float = 10.0,
    amplitude_uv: float = 50.0,
    sr: int = 10000,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Regular sinusoidal reward signal (µV)."""
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    return amplitude_uv * np.sin(2 * np.pi * frequency_hz * t)


def penalty_waveform(
    duration_s: float,
    frequency_hz: float = 200.0,
    amplitude_uv: float = 100.0,
    sr: int = 10000,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """High-frequency noise burst penalty signal (µV)."""
    if rng is None:
        rng = np.random.default_rng()
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n, endpoint=False)
    carrier = np.sin(2 * np.pi * frequency_hz * t)
    noise = rng.standard_normal(n)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 20.0 * t + rng.uniform(0, 2*np.pi))
    waveform = amplitude_uv * carrier * envelope + 0.2 * amplitude_uv * noise
    rms = np.sqrt(np.mean(waveform**2))
    return waveform * (amplitude_uv / rms) if rms > 0 else waveform


def neutral_waveform(duration_s: float, sr: int = 10000) -> np.ndarray:
    """Silence."""
    return np.zeros(int(duration_s * sr))


# ─────────────────────────────────────────────────────────────
# CURRENT INJECTION (v2 addition)
# ─────────────────────────────────────────────────────────────

def reward_current(
    duration_s: float,
    current_na: float = 0.3,
    frequency_hz: float = 10.0,
    dt_s: float = 0.001,
) -> np.ndarray:
    """
    Reward current injection for LIF neurons (nA).

    Sub-threshold periodic current at theta frequency.
    Drives coherent post-synaptic activity that aligns
    with stimulus spike trace — naturally promotes LTP
    for recently potentiated synapses.
    """
    n = int(duration_s / dt_s)
    t = np.arange(n) * dt_s
    # Rectified sine — only positive phase (depolarizing)
    raw = np.sin(2 * np.pi * frequency_hz * t)
    return current_na * np.clip(raw, 0, None)


def penalty_current(
    duration_s: float,
    current_na: float = 1.5,
    dt_s: float = 0.001,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Penalty current injection for LIF neurons (nA).

    Suprathreshold random current — drives chaotic firing
    that is uncorrelated with stimulus. This randomizes
    post-synaptic timing relative to pre-synaptic stimulus,
    producing net LTD via STDP without any external override.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = int(duration_s / dt_s)
    # High-amplitude noise, always positive (depolarizing)
    noise = rng.exponential(scale=current_na, size=n)
    return noise


def neutral_current(duration_s: float, dt_s: float = 0.001) -> np.ndarray:
    """No current — silence."""
    return np.zeros(int(duration_s / dt_s))


# ─────────────────────────────────────────────────────────────
# REWARD DECISION
# ─────────────────────────────────────────────────────────────

def compute_reward(
    prediction: str,
    ground_truth: str,
    timestamp: float,
    config: ExperimentConfig = None,
    rng: np.random.Generator = None,
) -> RewardSignal:
    """
    Generate reward signal + current injection based on prediction.

    Returns RewardSignal with both waveform (for display/recording)
    and current_injection (for LIF neuron input in post-stimulus window).

    Also encodes the neuromodulator factor for Three-Factor STDP:
        reward  → config.reward_neuromod  (default +1.0)
        penalty → config.penalty_neuromod (default -0.3)
        neutral → 0.0
    """
    if config is None:
        config = ExperimentConfig()
    if rng is None:
        rng = np.random.default_rng()

    correct = (prediction == ground_truth) if prediction is not None else None

    if correct is True:
        signal_type = 'reward'
        waveform = reward_waveform(
            config.reward_duration_s, config.reward_freq_hz,
            config.reward_amp_uv, rng=rng)
        current = reward_current(
            config.reward_duration_s, config.reward_current_na,
            config.reward_freq_hz)
        freq = config.reward_freq_hz
        amp = config.reward_amp_uv

    elif correct is False:
        signal_type = 'penalty'
        waveform = penalty_waveform(
            config.reward_duration_s, config.penalty_freq_hz,
            config.penalty_amp_uv, rng=rng)
        current = penalty_current(
            config.reward_duration_s, config.penalty_current_na, rng=rng)
        freq = config.penalty_freq_hz
        amp = config.penalty_amp_uv

    else:
        signal_type = 'neutral'
        waveform = neutral_waveform(config.reward_duration_s)
        current = neutral_current(config.reward_duration_s)
        freq = 0.0
        amp = 0.0

    return RewardSignal(
        signal_type=signal_type,
        waveform=waveform,
        current_injection=current,
        duration_s=config.reward_duration_s,
        frequency_hz=freq,
        amplitude_uv=amp,
        timestamp=timestamp,
    )


def neuromodulator_factor(reward: RewardSignal, config: ExperimentConfig) -> float:
    """
    Extract neuromodulator scaling factor for Three-Factor STDP.

    reward  → +1.0 (potentiation favored)
    penalty → -0.3 (mild punishment, sign flip on STDP)
    neutral →  0.0 (no plasticity)
    """
    if reward.signal_type == 'reward':
        return config.reward_neuromod
    elif reward.signal_type == 'penalty':
        return config.penalty_neuromod
    return 0.0


def signal_stats(signal: RewardSignal) -> dict:
    """Descriptive statistics. No interpretation."""
    w = signal.waveform
    c = signal.current_injection
    return {
        'type': signal.signal_type,
        'duration_s': signal.duration_s,
        'frequency_hz': signal.frequency_hz,
        'rms_uv': float(np.sqrt(np.mean(w**2))),
        'peak_uv': float(np.abs(w).max()),
        'current_mean_na': float(c.mean()),
        'current_peak_na': float(c.max()),
    }

