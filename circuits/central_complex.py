"""
Central complex (CX) heading compass and steering circuit.

Simulates ellipsoid body (EB) E-PG ring attractor dynamics and
fan-shaped body (FB) vector steering.
"""

import numpy as np
from typing import Tuple, Dict


class CentralComplex:
    """Central complex ring attractor compass and steering model."""
    def __init__(self, num_wedges: int = 16):
        self.num_wedges = num_wedges
        self.wedge_angles = np.linspace(-np.pi, np.pi, num_wedges, endpoint=False)
        
        # E-PG compass neurons (membrane activity around the ring)
        self.epg_activity = np.zeros(num_wedges, dtype=np.float32)
        self.current_heading_estimate = 0.0

        # Ring attractor parameters
        self.bump_width = 0.6  # radians (~35 degrees)
        self._initialize_bump(0.0)

        # Goal heading (vector navigation)
        self.goal_heading = 0.0
        self.steering_signal = 0.0

    def _initialize_bump(self, heading: float):
        """
        Initializes Gaussian activity bump at a given heading angle.
        """
        # Circular angular distance: min(|theta1 - theta2|, 2pi - |theta1 - theta2|)
        d_theta = np.arctan2(np.sin(self.wedge_angles - heading), np.cos(self.wedge_angles - heading))
        self.epg_activity = np.exp(-0.5 * (d_theta / self.bump_width) ** 2).astype(np.float32)
        self.epg_activity /= np.sum(self.epg_activity)
        self.current_heading_estimate = heading

    def update(self, angular_velocity: float, visual_flow_yaw: float, dt: float = 0.016) -> Dict[str, float]:
        """
        Updates the internal compass bump using self-motion (proprioception/efference copy)
        and visual optomotor flow.
        
        Args:
            angular_velocity: Fly's own turning speed (rad/s)
            visual_flow_yaw: Optic flow yaw rotation signal
            dt: Timestep in seconds
        """
        # Integrate angular velocity: total yaw shift
        net_turn = angular_velocity + 0.3 * visual_flow_yaw
        
        # Shift heading estimate
        self.current_heading_estimate = (self.current_heading_estimate + net_turn * dt + np.pi) % (2 * np.pi) - np.pi

        # Update ring attractor activity bump
        d_theta = np.arctan2(np.sin(self.wedge_angles - self.current_heading_estimate), 
                             np.cos(self.wedge_angles - self.current_heading_estimate))
        self.epg_activity = np.exp(-0.5 * (d_theta / self.bump_width) ** 2).astype(np.float32)
        self.epg_activity /= np.sum(self.epg_activity)

        # Fan-shaped body steering computation:
        # Steering error = angular difference between current heading and goal
        heading_error = np.arctan2(np.sin(self.goal_heading - self.current_heading_estimate),
                                   np.cos(self.goal_heading - self.current_heading_estimate))
        
        # P-controller steering torque
        self.steering_signal = float(np.clip(heading_error * 1.5, -1.0, 1.0))

        return {
            "heading_estimate": self.current_heading_estimate,
            "steering_signal": self.steering_signal,
            "epg_bump_max_wedge": int(np.argmax(self.epg_activity))
        }

    def set_goal(self, goal_angle: float):
        """
        Sets internal navigation target angle (e.g. toward safe zone or away from threat).
        """
        self.goal_heading = (goal_angle + np.pi) % (2 * np.pi) - np.pi
