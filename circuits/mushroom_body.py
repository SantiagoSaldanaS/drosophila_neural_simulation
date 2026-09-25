"""
Mushroom body (MB) circuit model with dopaminergic plasticity.

Simulates Kenyon cell (KC) sparse expansion, dopaminergic reinforcement
channels (PAM appetitive vs PPL1 aversive), and three-factor synaptic
plasticity on mushroom body output neurons (MBONs).
"""

import numpy as np
from typing import Tuple, Dict


class MushroomBody:
    """Mushroom body associative learning circuit."""
    def __init__(self, 
                 num_inputs: int = 100,      # Downsampled visual/spectral feature dimension
                 num_kc: int = 1500,         # Kenyon cells (sparse visual expansion)
                 kc_sparsity: float = 0.06,  # 6% of KCs active per scene
                 learning_rate: float = 0.08):
        self.num_inputs = num_inputs
        self.num_kc = num_kc
        self.k_active = max(10, int(num_kc * kc_sparsity))
        self.lr = learning_rate

        # Fixed random projection from sensory features to Kenyon Cells (claw inputs)
        # In Drosophila, each KC receives random claws from ~5-7 projection neurons
        np.random.seed(42)
        self.input_weights = np.random.randn(num_kc, num_inputs).astype(np.float32)
        # Normalize weights
        self.input_weights /= np.linalg.norm(self.input_weights, axis=1, keepdims=True) + 1e-6

        # Kenyon cell activations
        self.kc_activity = np.zeros(num_kc, dtype=np.float32)

        # Mushroom Body Output Neurons (MBONs):
        # Index 0: Approach MBON (MBON_app) -> excites steering toward target / forward flight
        # Index 1: Avoidance MBON (MBON_av)  -> excites repulsive steering / evasive veer
        self.num_mbons = 2
        
        # Plastic synaptic weights from KCs to MBONs: shape (num_mbons, num_kc)
        # Initialized with balanced baseline weights (neutral valence = 0.5)
        self.w_kc_mbon = np.full((self.num_mbons, num_kc), 0.5, dtype=np.float32)

        # Synaptic eligibility traces E_ij (biochemical tag for delayed reinforcement)
        self.eligibility = np.zeros((self.num_mbons, num_kc), dtype=np.float32)
        self.tau_eligibility = 1.2  # 1.2 second eligibility window

        # Dopaminergic neuron activities
        self.ppl1_dopamine = 0.0   # Aversive/punishment
        self.pam_dopamine = 0.0    # Appetitive/reward

        # MBON output firing rates
        self.mbon_activity = np.zeros(self.num_mbons, dtype=np.float32)
        self.net_valence = 0.0     # >0: approach, <0: avoid

        # Cumulative learning stats
        self.total_plastic_updates = 0
        self.weight_shift_history = []

    def encode_sensory(self, visual_features: np.ndarray) -> np.ndarray:
        """
        Projects visual/spectral features into the high-dimensional sparse Kenyon cell layer.
        Implements lateral inhibition (winner-take-all / top-k) mimicking APL GABAergic interneuron.
        """
        # Ensure feature vector matches expected input dimensions
        if len(visual_features) != self.num_inputs:
            # Downsample or pad to match
            feats = np.interp(np.linspace(0, 1, self.num_inputs), 
                              np.linspace(0, 1, len(visual_features)), visual_features)
        else:
            feats = visual_features

        # Linear projection
        drive = np.dot(self.input_weights, feats)

        # Top-k sparse thresholding (APL inhibition)
        top_k_indices = np.argpartition(drive, -self.k_active)[-self.k_active:]
        self.kc_activity.fill(0.0)
        # Normalize active KC activations so sum is bounded
        self.kc_activity[top_k_indices] = np.maximum(0.0, drive[top_k_indices])
        sum_kc = np.sum(self.kc_activity)
        if sum_kc > 1e-6:
            self.kc_activity /= sum_kc  # Unit-sum sparse population vector

        return self.kc_activity

    def update(self, visual_features: np.ndarray, dt: float = 0.016) -> Dict[str, float]:
        """
        Computes MBON activation and decays eligibility traces.
        """
        # 1. Sparse KC encoding (normalized)
        self.encode_sensory(visual_features)

        # 2. Compute MBON activity: MBON_j = sum_i (W_ji * KC_i)
        # Since KC is unit-sum and W in [0.05, 1.5], MBON activity is smoothly in [0.05, 1.5]
        self.mbon_activity = np.dot(self.w_kc_mbon, self.kc_activity)

        # 3. Net behavioral valence:
        # If MBON_app > MBON_av: positive valence (approach)
        # If MBON_av > MBON_app: negative valence (avoid)
        self.net_valence = float(np.clip(self.mbon_activity[0] - self.mbon_activity[1], -1.5, 1.5))

        # 4. Update eligibility traces:
        # dE_ij/dt = -E_ij/tau_e + KC_i * MBON_j
        decay = np.exp(-dt / self.tau_eligibility)
        outer_product = np.outer(self.mbon_activity, self.kc_activity)
        self.eligibility = self.eligibility * decay + (1.0 - decay) * outer_product

        # Decay transient dopamine spikes
        self.ppl1_dopamine = max(0.0, self.ppl1_dopamine - dt * 3.0)
        self.pam_dopamine = max(0.0, self.pam_dopamine - dt * 3.0)

        return {
            "mbon_approach": float(self.mbon_activity[0]),
            "mbon_avoid": float(self.mbon_activity[1]),
            "net_valence": self.net_valence,
            "kc_active_count": int(np.count_nonzero(self.kc_activity)),
            "ppl1_dopamine": self.ppl1_dopamine,
            "pam_dopamine": self.pam_dopamine
        }

    def deliver_punishment(self, intensity: float = 1.0):
        """
        Fires PPL1 dopaminergic neurons upon taking damage or hitting obstacles.
        Depresses Approach synapses (LTD on MBON_app) and potentiates Avoidance synapses (LTP on MBON_av).
        """
        self.ppl1_dopamine = float(np.clip(intensity, 0.0, 5.0))
        
        # 3-Factor update:
        # Aversive dopamine depresses approach synapses that were recently eligible:
        dW_app = -self.lr * self.ppl1_dopamine * self.eligibility[0] * 5.0
        # And potentiates avoidance synapses:
        dW_av = +self.lr * self.ppl1_dopamine * self.eligibility[1] * 5.0

        self.w_kc_mbon[0] = np.clip(self.w_kc_mbon[0] + dW_app, 0.05, 1.5)
        self.w_kc_mbon[1] = np.clip(self.w_kc_mbon[1] + dW_av, 0.05, 1.5)

        self.total_plastic_updates += 1
        self.weight_shift_history.append(float(np.mean(np.abs(dW_app) + np.abs(dW_av))))

    def deliver_reward(self, intensity: float = 1.0):
        """
        Fires PAM dopaminergic neurons upon collecting energy or surviving a wave.
        Potentiates Approach synapses (LTP on MBON_app) and depresses Avoidance synapses (LTD on MBON_av).
        """
        self.pam_dopamine = float(np.clip(intensity, 0.0, 5.0))

        # 3-Factor update:
        dW_app = +self.lr * self.pam_dopamine * self.eligibility[0] * 5.0
        dW_av = -self.lr * self.pam_dopamine * self.eligibility[1] * 5.0

        self.w_kc_mbon[0] = np.clip(self.w_kc_mbon[0] + dW_app, 0.05, 1.5)
        self.w_kc_mbon[1] = np.clip(self.w_kc_mbon[1] + dW_av, 0.05, 1.5)

        self.total_plastic_updates += 1
        self.weight_shift_history.append(float(np.mean(np.abs(dW_app) + np.abs(dW_av))))
