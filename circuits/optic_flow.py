"""
Optic flow and looming detection circuits of Drosophila.

Simulates elementary motion detectors (Hassenstein-Reichardt correlators),
lobula plate tangential cells (LPTC HS/VS), and radial expansion detectors (LPLC2/Giant Fiber).
"""

import numpy as np
from typing import Tuple, Dict, Optional


class OpticFlowCircuit:
    """Visual motion and looming threat detection circuit."""
    def __init__(self, num_ommatidia: int, ommatidia_azimuths: np.ndarray, ommatidia_eyes: np.ndarray):
        self.num_ommatidia = num_ommatidia
        self.azimuths = ommatidia_azimuths
        self.eyes = ommatidia_eyes  # 0: left eye, 1: right eye

        # Low-pass filter states for delay in Hassenstein-Reichardt correlators
        self.tau_delay = 0.025  # 25 ms delay filter
        self.delayed_luminance = np.zeros(num_ommatidia, dtype=np.float32)

        # Build adjacent neighbor pairs for horizontal motion detection
        self.left_pairs, self.right_pairs = self._build_motion_pairs()

        # Output states:
        self.hs_left = 0.0   # Horizontal System cell (left eye yaw flow)
        self.hs_right = 0.0  # Horizontal System cell (right eye yaw flow)
        self.vs_flow = 0.0   # Vertical System cell (forward/downward flow)
        
        # Looming detector states (LPLC2 & Giant Fiber)
        self.looming_signal_left = 0.0
        self.looming_signal_right = 0.0
        self.looming_alert = False

        # Previous contrast profile for expansion rate calculation
        self.prev_contrast = np.zeros(num_ommatidia, dtype=np.float32)

    def _build_motion_pairs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Connects spatially adjacent ommatidia to form elementary motion detector (EMD) pairs.
        """
        left_indices = np.where(self.eyes == 0)[0]
        right_indices = np.where(self.eyes == 1)[0]

        left_pairs = np.column_stack([left_indices[:-1], left_indices[1:]])
        right_pairs = np.column_stack([right_indices[:-1], right_indices[1:]])

        return left_pairs, right_pairs

    def update(self, r1_r6_luminance: np.ndarray, r7_r8_spectral: Optional[np.ndarray] = None, dt: float = 0.016) -> Dict[str, float]:
        """
        Processes luminance inputs through EMDs and dual-channel looming circuits.
        Uses spectral chromatic weighting (R7/R8) to filter out appetitive stimuli (nectar)
        from triggering false-alarm Giant Fiber escapes.
        
        Args:
            r1_r6_luminance: Array of photoreceptor activations (size: num_ommatidia)
            r7_r8_spectral: Optional spectral contrast channel (distinguishes nectar vs hazards)
            dt: Timestep in seconds
        """
        # 1. Update temporal delay filter
        alpha = dt / (self.tau_delay + dt)
        self.delayed_luminance = (1.0 - alpha) * self.delayed_luminance + alpha * r1_r6_luminance

        # 2. Hassenstein-Reichardt Correlator (EMD)
        # Left eye motion
        s1_l = self.delayed_luminance[self.left_pairs[:, 0]]
        s2_l = r1_r6_luminance[self.left_pairs[:, 1]]
        s1_inst_l = r1_r6_luminance[self.left_pairs[:, 0]]
        s2_del_l = self.delayed_luminance[self.left_pairs[:, 1]]
        emd_left = (s1_l * s2_l) - (s1_inst_l * s2_del_l)

        # Right eye motion
        s1_r = self.delayed_luminance[self.right_pairs[:, 0]]
        s2_r = r1_r6_luminance[self.right_pairs[:, 1]]
        s1_inst_r = r1_r6_luminance[self.right_pairs[:, 0]]
        s2_del_r = self.delayed_luminance[self.right_pairs[:, 1]]
        emd_right = (s1_r * s2_r) - (s1_inst_r * s2_del_r)

        # 3. Lobula Plate Tangential Cells (LPTC HS cells)
        self.hs_left = float(np.mean(emd_left) * 100.0)
        self.hs_right = float(np.mean(emd_right) * 100.0)
        yaw_optomotor = self.hs_left - self.hs_right

        # 4. Dual-Channel Looming Detection (ON + OFF LPLC2 model)
        # In nature, looming detects rapid expansion of contrast against the background:
        contrast = np.abs(r1_r6_luminance - 0.5) * 2.0  # in [0, 1]

        if not hasattr(self, "initialized") or not self.initialized:
            self.prev_contrast = contrast.copy()
            self.initialized = True
            d_contrast = np.zeros_like(contrast)
        else:
            d_contrast = (contrast - self.prev_contrast) / max(dt, 0.001)
            self.prev_contrast = contrast.copy()

        # Expansion = contrast * positive_rate_of_expansion
        looming_local = contrast * np.clip(d_contrast, 0.0, 60.0)

        # Spectral weighting: In Drosophila, appetitive green/cyan targets inhibit Giant Fiber escape
        if r7_r8_spectral is not None:
            # Green/cyan has r7_r8 > 0.7 (threat_weight -> 0.0)
            # Red/orange hazards have r7_r8 < 0.3 (threat_weight -> 1.0)
            threat_weight = np.clip(1.2 - r7_r8_spectral * 1.5, 0.0, 1.0)
            looming_local = looming_local * threat_weight
        
        left_mask = self.eyes == 0
        right_mask = self.eyes == 1
        
        # Pool top receptive field activations for acute projectile looming sensitivity:
        left_vals = looming_local[left_mask]
        right_vals = looming_local[right_mask]

        # In Drosophila, active self-rotation (HS optomotor flow) suppresses false Giant Fiber alarms
        rot_suppression = 1.0 + 0.12 * abs(yaw_optomotor)

        self.looming_signal_left = (float(np.mean(np.sort(left_vals)[-3:])) / rot_suppression) if len(left_vals) >= 3 else 0.0
        self.looming_signal_right = (float(np.mean(np.sort(right_vals)[-3:])) / rot_suppression) if len(right_vals) >= 3 else 0.0

        # Calibrated biological collision alert threshold (high-velocity projectiles trigger > 4.5; stationary traps stay below 2.0)
        threshold = 3.2
        self.looming_alert = (self.looming_signal_left > threshold) or (self.looming_signal_right > threshold)

        # Giant Fiber escape direction: Bank sharply away from the expanding threat
        if self.looming_alert:
            if self.looming_signal_left > self.looming_signal_right:
                escape_bias = +1.0  # Threat on left -> Turn hard right
            else:
                escape_bias = -1.0  # Threat on right -> Turn hard left
        else:
            escape_bias = 0.0

        return {
            "hs_left": self.hs_left,
            "hs_right": self.hs_right,
            "yaw_optomotor": yaw_optomotor,
            "looming_left": self.looming_signal_left,
            "looming_right": self.looming_signal_right,
            "looming_alert": self.looming_alert,
            "escape_bias": escape_bias
        }
