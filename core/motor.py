"""
Motor decoder for flight steering and thrust commands.

Integrates optomotor stabilization, central complex heading control,
mushroom body valence, obstacle clearance, and looming threat evasion.
"""

import math
import random
import numpy as np
from typing import Dict, Tuple, Optional


class MotorDecoder:
    """
    Decodes biological neural circuit activations into aerodynamic flight commands.
    """
    def __init__(self, mass: float = 1.0, max_speed: float = 380.0, max_yaw_rate: float = 14.0):
        self.mass = mass
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate

        # Baseline flight motor states (restored agile cruise speed)
        self.base_thrust = 280.0
        self.forward_thrust = 280.0
        self.yaw_torque = 0.0
        self.saccade_active = False
        self.saccade_timer = 0.0
        self.escape_target_turn = 0.0
        self.escape_accumulated_turn = 0.0
        self.escape_cooldown = 0.0

        # Discrete exploration saccade timer
        self.wander_turn_timer = random.uniform(1.5, 3.0)
        self.wander_turn_bias = 0.0

    def compute_action(self,
                       optomotor_data: Dict[str, float],
                       cx_data: Dict[str, float],
                       mb_data: Dict[str, float],
                       target_azimuth: float = 0.0,
                       target_dist: float = 999.0,
                       target_type: Optional[str] = None,
                       hazard_azimuth: float = 0.0,
                       hazard_dist: float = 999.0,
                       hazard_present: bool = False,
                       projectile_threat: bool = False,
                       projectile_azimuth: float = 0.0,
                       safe_escape_angle: Optional[float] = None,
                       fly_health: float = 100.0,
                       dt: float = 0.016) -> Dict[str, float]:
        """
        Integrates sensory and circuit outputs into flight steering and thrust commands.
        """
        if self.escape_cooldown > 0.0:
            self.escape_cooldown -= dt

        is_looming = optomotor_data.get("looming_alert", False) or projectile_threat

        # Emergency evasion trigger
        can_trigger = (not self.saccade_active and self.escape_cooldown <= 0.0) or (projectile_threat and not self.saccade_active)
        if is_looming and can_trigger:
            self.saccade_active = True
            self.saccade_timer = 0.22  # 220 ms maximum escape burst
            if safe_escape_angle is not None and abs(safe_escape_angle) > 0.01:
                self.escape_target_turn = safe_escape_angle
            elif projectile_threat:
                self.escape_target_turn = -1.35 if projectile_azimuth >= 0.0 else 1.35
            else:
                bias = optomotor_data.get("escape_bias", 0.0)
                self.escape_target_turn = bias * 1.35 if bias != 0.0 else random.choice([-1.35, 1.35])
            self.escape_accumulated_turn = 0.0
        elif projectile_threat and self.saccade_active and safe_escape_angle is not None:
            self.escape_target_turn = safe_escape_angle

        # Obstacle avoidance for static hazards
        net_valence = mb_data["net_valence"]  # in [-1.5, +1.5]
        hazard_steering = 0.0
        thrust_modulation = 0.0
        in_hazard_path = False

        if hazard_present and hazard_dist < 240.0:
            x_haz = hazard_dist * math.cos(hazard_azimuth)
            y_haz = hazard_dist * math.sin(hazard_azimuth)

            R_safe = 78.0
            lookahead = 240.0
            if (x_haz > -15.0) and (x_haz < lookahead) and (abs(y_haz) < R_safe):
                in_hazard_path = True
                delta = R_safe - abs(y_haz)
                w = (lookahead - max(0.0, x_haz)) / lookahead
                learned_aversion = 1.0 + max(0.0, -net_valence) * 0.4

                if abs(y_haz) < 2.0:
                    steer_dir = -1.0 if target_azimuth <= 0.0 else 1.0
                else:
                    steer_dir = -1.0 if y_haz >= 0.0 else 1.0

                hazard_steering = steer_dir * min(self.max_yaw_rate, (14.0 * (delta / R_safe) + 8.0 * w) * learned_aversion)

                # Moderate deceleration when obstacle is close ahead
                if x_haz < 110.0 and abs(y_haz) < 45.0:
                    thrust_modulation -= 60.0

        # Target guidance toward nectar
        target_steering = 0.0

        if target_type == "nectar":
            hunger_multiplier = 1.35 if fly_health < 70.0 else 1.15
            reward_boost = 1.0 + max(0.0, net_valence) * 0.25
            abs_tgt = abs(target_azimuth)

            target_steering = math.copysign(min(self.max_yaw_rate, (abs_tgt ** 1.15) * 11.0), target_azimuth)
            thrust_modulation += 50.0 * hunger_multiplier * reward_boost
            if target_dist < 90.0:
                thrust_modulation += 30.0

        # Process active escape saccade
        if self.saccade_active:
            self.saccade_timer -= dt
            if self.saccade_timer <= 0.0:
                self.saccade_active = False
                self.escape_cooldown = 0.04

            remaining_turn = self.escape_target_turn - self.escape_accumulated_turn
            if abs(remaining_turn) > 0.05:
                turn_rate = math.copysign(min(self.max_yaw_rate, abs(remaining_turn) * 16.0), remaining_turn)
                self.escape_accumulated_turn += turn_rate * dt
                yaw_cmd = turn_rate
            else:
                yaw_cmd = 0.0

            if in_hazard_path:
                yaw_cmd += hazard_steering * 0.85

            thrust_cmd = 440.0
            self.yaw_torque = float(np.clip(yaw_cmd, -self.max_yaw_rate, self.max_yaw_rate))
            self.forward_thrust = thrust_cmd
            return {
                "yaw_rate": self.yaw_torque,
                "thrust": self.forward_thrust,
                "saccade_active": True,
                "learned_component": float(hazard_steering),
                "optomotor_component": 0.0
            }

        # Composite steering for cruise and pursuit
        if in_hazard_path:
            if (target_steering * hazard_steering) > 0.0:
                yaw_command = hazard_steering + target_steering * 0.15
            else:
                yaw_command = hazard_steering
        elif target_type == "nectar":
            yaw_command = target_steering
        else:
            # Discrete exploration only when no nectar is in sight
            self.wander_turn_timer -= dt
            if self.wander_turn_timer <= 0.0:
                self.wander_turn_bias = -self.wander_turn_bias if self.wander_turn_bias != 0 else 2.0
                self.wander_turn_timer = random.uniform(2.0, 3.5)
            else:
                self.wander_turn_bias = max(0.0, abs(self.wander_turn_bias) - dt * 2.5) * (1.0 if self.wander_turn_bias > 0 else -1.0)
            
            cx_steering = cx_data["steering_signal"] * 0.2
            yaw_optomotor = optomotor_data["yaw_optomotor"] * 0.005
            yaw_command = self.wander_turn_bias + cx_steering - yaw_optomotor

        yaw_command = float(np.clip(yaw_command, -self.max_yaw_rate, self.max_yaw_rate))
        thrust_command = float(np.clip(self.base_thrust + thrust_modulation, 120.0, self.max_speed))

        self.yaw_torque = yaw_command
        self.forward_thrust = thrust_command

        return {
            "yaw_rate": self.yaw_torque,
            "thrust": self.forward_thrust,
            "saccade_active": False,
            "learned_component": float(hazard_steering),
            "optomotor_component": float(optomotor_data["yaw_optomotor"] * 0.005)
        }
