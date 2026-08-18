"""
oi_types.py — organoid_oi_v3
=============================
Shared type definitions.

Changes from v2:
    - ExperimentConfig: epoch-based homeostatic, blanking period, decoder params
    - STDPUpdate: vectorized flag
    - BlankingWindow: new dataclass for stimulus artifact suppression
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np


@dataclass
class StimulusPattern:
    spike_times: List[np.ndarray]
    n_electrodes: int
    duration_s: float
    label: str
    meta: Dict = field(default_factory=dict)


@dataclass
class OrganoidResponse:
    spike_times: List[np.ndarray]
    n_neurons: int
    duration_s: float
    timestamp: float
    prediction: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class RewardSignal:
    signal_type: str
    waveform: np.ndarray
    current_injection: np.ndarray
    duration_s: float
    frequency_hz: float
    amplitude_uv: float
    timestamp: float


@dataclass
class STDPUpdate:
    dw: np.ndarray
    pre_times: List[np.ndarray]
    post_times: List[np.ndarray]
    n_ltp: int
    n_ltd: int
    neuromodulator_factor: float = 1.0
    vectorized: bool = True          # v3: always True


@dataclass
class Trial:
    trial_id: int
    stimulus: StimulusPattern
    response: OrganoidResponse
    reward: RewardSignal
    correct: bool
    timestamp: float
    # Faz 1 yaniti — odul/ceza akimi ENJEKTE EDILMEDEN once.
    # 'response' alani enjeksiyon SONRASI aktiviteyi icerir; odul dogru
    # tahminde verildigi icin o sinyal kategoriyle iliskilidir ve
    # cevrimdisi cozucuyu kirletir. Atif olcumleri bu alani kullanmalidir.
    response_p1: OrganoidResponse = None


@dataclass
class ExperimentConfig:
    # Categories
    # NOT: 'banana'/'apple'/'pear' sadece varsayilan yer tutucudur — gorev
    # katmani (core/oi_gorev.py) her zaman kendi etiketlerini (art/rat/tar)
    # acikca gecirir. Bu varsayilan sadece tests/test_oi_v3.py'nin mekanizma
    # testlerinde keyfi sabit deger olarak kullanildigi icin degistirilmedi.
    categories: List[str] = field(default_factory=lambda: ['banana', 'apple', 'pear'])

    # Electrodes / neurons
    n_electrodes: int = 64
    n_neurons: int = 100
    stimulus_duration_s: float = 0.5
    response_window_s: float = 0.5
    reward_duration_s: float = 0.3

    # Reward signal
    reward_freq_hz: float = 10.0
    penalty_freq_hz: float = 200.0
    reward_amp_uv: float = 50.0
    penalty_amp_uv: float = 100.0
    reward_current_na: float = 0.3
    penalty_current_na: float = 1.5

    # Three-Factor STDP
    tau_plus_ms: float = 20.0
    tau_minus_ms: float = 20.0
    a_plus: float = 0.01
    a_minus: float = 0.0105
    reward_neuromod: float = 1.0
    penalty_neuromod: float = -0.3

    # Training
    n_trials: int = 300
    learning_rate: float = 0.01

    # v3: Epoch-based homeostatic (fix B)
    homeostatic: bool = True
    homeostatic_epoch: int = 10      # apply every N trials (not every trial)
    target_weight_sum: float = None
    homeostatic_rate: float = 0.05   # stronger per-epoch since applied less often

    # v3: Downstream Hebbian decoder (fix A)
    decoder_type: str = 'hebbian'    # 'frozen' | 'hebbian'
    decoder_seed: int = 123
    decoder_lr: float = 0.005        # slow Hebbian, local rule only
    decoder_hebbian_decay: float = 0.001  # weight decay prevents runaway

    # v3: Blanking period (fix C)
    blanking_ms: float = 5.0        # suppress reading during stimulus artifact
    blanking_enabled: bool = True

    # IF neuron parameters
    v_rest_mv: float = -70.0
    v_thresh_mv: float = -55.0
    # Noronlar arasi esik cesitliligi (mV). 0 = eskisi gibi tam homojen.
    # >0: her noron kendi eşiğini alır (v_thresh_mv +- gauss jitter).
    # Amac: sessizlik/doygunluk arasindaki gecisin TUM populasyonda ayni
    # anda olmasini onlemek — geciste kademeli, gercekci bir bant yaratmak.
    v_thresh_jitter_mv: float = 0.0
    v_reset_mv: float = -75.0
    tau_membrane_ms: float = 20.0
    refractory_ms: float = 2.0
    r_membrane_mohm: float = 10.0
    tau_syn_ms: float = 5.0

    # Image encoding — DoG + rank-order
    image_size: int = 8
    dog_sigma1: float = 1.0
    dog_sigma2: float = 2.0
    use_rank_order: bool = True
    max_latency_ms: float = 50.0
    max_spike_rate_hz: float = 100.0

    # Output
    output_dir: str = 'oi_results_v3'
    random_seed: int = 42

