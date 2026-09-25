"""
Flight simulation environment for Drosophila bio-vision and navigation.

Features high-speed projectile looming evasion, optomotor flow background,
associative target conditioning (nectar vs toxic traps), and boundary dynamics.
"""

import math
import random
import numpy as np
import pygame
from typing import List, Tuple, Dict, Optional
from core.telemetry_logger import TelemetryLogger


class Projectile:
    """A fast-moving projectile that creates strong looming expansion."""
    def __init__(self, x: float, y: float, vx: float, vy: float, radius: float = 13.0, color: Tuple[int, int, int] = (255, 65, 65)):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.color = color
        self.alive = True

    def update(self, dt: float, bounds: Tuple[int, int]):
        self.x += self.vx * dt
        self.y += self.vy * dt
        w, h = bounds
        if self.x < -60 or self.x > w + 60 or self.y < -60 or self.y > h + 60:
            self.alive = False

    def draw(self, surface: pygame.Surface):
        pos = (int(self.x), int(self.y))
        r = int(self.radius)
        # Glowing outer corona
        pygame.draw.circle(surface, (120, 20, 20), pos, r + 4)
        # Main core
        pygame.draw.circle(surface, self.color, pos, r)
        # Inner bright specular point
        pygame.draw.circle(surface, (255, 200, 200), (pos[0] - r // 3, pos[1] - r // 3), max(2, r // 3))
        # High-contrast dark outline for looming edge detection
        pygame.draw.circle(surface, (10, 10, 15), pos, r, 2)


class AssociativeTarget:
    """
    An entity that the fly learns to classify via dopaminergic conditioning:
    - 'nectar': Rewarding energy source (+PAM dopamine)
    - 'toxic_trap': Damaging trap (+PPL1 aversive dopamine)
    """
    def __init__(self, x: float, y: float, target_type: str = "nectar"):
        self.x = x
        self.y = y
        self.type = target_type
        self.radius = 18.0
        self.pulse = random.uniform(0.0, math.pi * 2)
        self.alive = True

        if self.type == "nectar":
            # Cyan / emerald nectar orb (high blue/green contrast for R8 photoreceptors)
            self.base_color = (40, 245, 195)
            self.core_color = (210, 255, 240)
            self.glow_color = (20, 110, 90)
        else:
            # Red / crimson shock trap
            self.base_color = (245, 45, 65)
            self.core_color = (255, 180, 180)
            self.glow_color = (110, 15, 25)

    def update(self, dt: float):
        self.pulse += dt * 4.5

    def draw(self, surface: pygame.Surface):
        pos = (int(self.x), int(self.y))
        r = int(self.radius + math.sin(self.pulse) * 3.5)
        # Soft outer glow
        pygame.draw.circle(surface, self.glow_color, pos, r + 6)
        # Main orb
        pygame.draw.circle(surface, self.base_color, pos, r)
        # Bright pulsating center
        pygame.draw.circle(surface, self.core_color, pos, max(3, int(r * 0.45)))
        # Clean boundary rim
        pygame.draw.circle(surface, (15, 20, 25), pos, r, 2)


class ThreatResult(tuple):
    """Tuple of (azimuth, distance, is_threat) with optional safe_escape_angle attribute."""
    def __new__(cls, az: float, dist: float, is_threat: bool, safe_angle: float = 0.0):
        obj = super().__new__(cls, (az, dist, is_threat))
        obj.safe_escape_angle = safe_angle
        return obj


class BulletHellArena:
    """Visual flight arena with closed-loop physics and collision detection."""
    def __init__(self, width: int = 780, height: int = 850):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height))

        # Fly physics state
        self.fly_x = width / 2.0
        self.fly_y = height / 2.0
        self.fly_vx = 0.0
        self.fly_vy = 0.0
        self.fly_heading = random.uniform(-math.pi, math.pi)
        self.fly_angular_vel = 0.0
        self.fly_radius = 15.0
        self.fly_health = 100.0
        self.fly_alive = True
        self.score = 0
        self.respawn_timer = 0.0

        # Aerodynamic parameters (fluid, fast biological flight)
        self.drag_coeff = 1.6
        self.rotational_damping = 5.0

        # Run statistics for Leaderboards
        self.nectar_collected = 0
        self.projectiles_dodged = 0
        self.game_over_recorded = False
        self.last_run_summary = None

        # Entities
        self.projectiles: List[Projectile] = []
        self.targets: List[AssociativeTarget] = []
        self.particles: List[Dict] = []

        # Background visual grating (induces optomotor flow)
        self.grating_offset = 0.0
        self.grating_speed = 45.0  # pixels/sec drifting grating

        # Spawning timers
        self.projectile_spawn_timer = 0.0
        self.target_spawn_timer = 0.0
        self.time_elapsed = 0.0
        self.time_bot = 0.0
        self.time_human = 0.0

        # Fonts
        if not pygame.font.get_init():
            pygame.font.init()
        font_family = "Segoe UI" if "segoeui" in pygame.font.get_fonts() else "Arial"
        self.font_respawn = pygame.font.SysFont(font_family, 20, bold=True)

        # Event notifications for HUD
        self.recent_events = []

        # Comprehensive telemetry & debug logger
        self.telemetry = TelemetryLogger()
        self._prev_saccade_active = False

        # Initial target spawn
        self._spawn_target("nectar")
        self._spawn_target("toxic_trap")

    def set_dimensions(self, width: int, height: int):
        """Dynamically adjusts the arena surface dimensions if needed without resetting simulation entities."""
        if width == self.width and height == self.height:
            return
        self.width = max(400, width)
        self.height = max(400, height)
        self.surface = pygame.Surface((self.width, self.height))
        # Keep fly in bounds
        self.fly_x = float(np.clip(self.fly_x, 50, self.width - 50))
        self.fly_y = float(np.clip(self.fly_y, 50, self.height - 50))

    def log_event(self, text: str, color: Tuple[int, int, int] = (255, 255, 255)):
        self.recent_events.append({"text": text, "color": color, "time": 2.8})
        if len(self.recent_events) > 5:
            self.recent_events.pop(0)

    def _spawn_projectile(self):
        """Spawns a high-speed projectile aimed near the fly to test looming evasion."""
        speed = random.uniform(220.0, 360.0)

        # Ensure projectile spawns from an edge that gives fair flight distance (>= 220px)
        candidate_edges = []
        if self.fly_y >= 220.0:
            candidate_edges.append("top")
        if self.fly_y <= self.height - 220.0:
            candidate_edges.append("bottom")
        if self.fly_x >= 220.0:
            candidate_edges.append("left")
        if self.fly_x <= self.width - 220.0:
            candidate_edges.append("right")

        if not candidate_edges:
            candidate_edges = ["top", "bottom", "left", "right"]

        edge = random.choice(candidate_edges)
        if edge == "top":
            x = random.uniform(50, self.width - 50)
            y = 10
        elif edge == "bottom":
            x = random.uniform(50, self.width - 50)
            y = self.height - 10
        elif edge == "left":
            x = 10
            y = random.uniform(50, self.height - 50)
        else:
            x = self.width - 10
            y = random.uniform(50, self.height - 50)

        lead = random.uniform(-40.0, 40.0)
        target_x = self.fly_x + lead
        target_y = self.fly_y + lead
        dx = target_x - x
        dy = target_y - y
        dist = math.hypot(dx, dy) + 1e-6
        vx = (dx / dist) * speed
        vy = (dy / dist) * speed

        self.projectiles.append(Projectile(x, y, vx, vy, radius=random.uniform(12.0, 16.0)))

    def _spawn_target(self, target_type: Optional[str] = None):
        """Spawns an energy orb or a toxic trap away from the fly and away from other targets."""
        if target_type is None:
            has_nectar = any(t.alive and t.type == "nectar" for t in self.targets)
            if not has_nectar:
                target_type = "nectar"
            else:
                target_type = "nectar" if random.random() < 0.70 else "toxic_trap"
        
        for _ in range(25):
            x = random.uniform(90, self.width - 90)
            y = random.uniform(90, self.height - 90)
            # Minimum spacing from fly and existing targets
            dist_to_fly = math.hypot(x - self.fly_x, y - self.fly_y)
            too_close_to_other = any(math.hypot(x - ot.x, y - ot.y) < 110.0 for ot in self.targets if ot.alive)
            if dist_to_fly > 140.0 and not too_close_to_other:
                self.targets.append(AssociativeTarget(x, y, target_type))
                break

    def get_closest_target_info(self, fly_health: float = 100.0) -> Tuple[float, float, Optional[str]]:
        """
        Returns (azimuth_relative_to_heading, distance, type) of the most salient target.
        Favors targets in the frontal field of view over rear targets.
        """
        best_cost = 99999.0
        best_az = 0.0
        best_dist = 999.0
        best_type = None

        nectar_targets = [t for t in self.targets if t.alive and t.type == "nectar"]
        if not nectar_targets:
            return 0.0, 999.0, None

        for t in nectar_targets:
            d = math.hypot(t.x - self.fly_x, t.y - self.fly_y)
            world_angle = math.atan2(t.y - self.fly_y, t.x - self.fly_x)
            az = (world_angle - self.fly_heading + math.pi) % (2 * math.pi) - math.pi

            # Frontal FOV bias
            fov_penalty = 1.0 + 0.65 * (abs(az) / math.pi)
            cost = d * fov_penalty

            if cost < best_cost:
                best_cost = cost
                best_az = az
                best_dist = d
                best_type = t.type

        return best_az, best_dist, best_type

    def get_closest_hazard_info(self) -> Tuple[float, float, bool]:
        """
        Returns (azimuth, distance, is_present) of the closest static hazard in the flight path.
        """
        best_corridor_urgency = -1.0
        best_corridor_trap = None
        best_general_cost = 99999.0
        best_general_trap = None

        R_safe = 78.0
        lookahead = 240.0

        for t in self.targets:
            if t.alive and t.type == "toxic_trap":
                dx = t.x - self.fly_x
                dy = t.y - self.fly_y
                d = math.hypot(dx, dy)
                world_angle = math.atan2(dy, dx)
                az = (world_angle - self.fly_heading + math.pi) % (2 * math.pi) - math.pi
                abs_az = abs(az)

                # Coordinates in fly body frame (x forward, y left/right)
                x_haz = d * math.cos(az)
                y_haz = d * math.sin(az)

                # Check if in forward collision corridor
                if (-15.0 < x_haz < lookahead) and (abs(y_haz) < R_safe):
                    delta = R_safe - abs(y_haz)
                    urgency = (delta / R_safe) * (1.0 + 200.0 / max(20.0, x_haz))
                    if urgency > best_corridor_urgency:
                        best_corridor_urgency = urgency
                        best_corridor_trap = (az, d)

                # General awareness tracking for when no trap is in direct path
                if abs_az > 1.75:
                    cost = d * 6.0
                else:
                    cost = d * (1.0 + 0.35 * (abs_az / math.pi))
                if cost < best_general_cost:
                    best_general_cost = cost
                    best_general_trap = (az, d)

        if best_corridor_trap is not None:
            return best_corridor_trap[0], best_corridor_trap[1], True
        elif best_general_trap is not None:
            return best_general_trap[0], best_general_trap[1], True
        return 0.0, 999.0, False

    def get_safe_evasion_heading(self, threat_azimuth: float) -> float:
        """
        Evaluates candidate banked evasion options against static traps,
        walls, and threat velocity vectors. Returns safe relative escape angle.
        """
        # Candidate evasion bank angles (from gentle ~40 deg to wide ~115 deg)
        candidate_angles = [-2.0, -1.7, -1.35, -1.0, -0.7, 0.7, 1.0, 1.35, 1.7, 2.0]
        best_score = -999999.0
        best_angle = -1.35 if threat_azimuth >= 0.0 else +1.35

        for d_theta in candidate_angles:
            phi_cand = self.fly_heading + d_theta
            div = abs((d_theta - threat_azimuth + math.pi) % (2 * math.pi) - math.pi)

            # Bank away from incoming projectile
            if div < 0.8:  # < 45 degrees: banking directly toward threat trajectory
                score = -5000.0
            else:
                score = 50.0 * math.sin(div / 2.0)

            # Preference for standard bank angles (~70 deg) rather than extreme ~120 deg
            score -= 8.0 * (abs(d_theta) - 1.25) ** 2

            # Static toxic trap clearance check: project trajectory along candidate heading
            for t in self.targets:
                if t.alive and t.type == "toxic_trap":
                    dx = t.x - self.fly_x
                    dy = t.y - self.fly_y
                    dist = math.hypot(dx, dy)
                    t_az = (math.atan2(dy, dx) - phi_cand + math.pi) % (2 * math.pi) - math.pi
                    xt = dist * math.cos(t_az)
                    yt = dist * math.sin(t_az)

                    # Immediate proximity rejection
                    if dist < 55.0 and abs(t_az) < 1.3:
                        score -= 20000.0

                    # Projected corridor penetration check
                    if (xt > -15.0) and (xt < 240.0) and (abs(yt) < 80.0):
                        pen = ((80.0 - abs(yt)) / 80.0) * ((240.0 - max(0.0, xt)) / 240.0)
                        score -= 8000.0 * pen

            # Arena boundary clearance check: project 150px along candidate heading
            px = self.fly_x + 150.0 * math.cos(phi_cand)
            py = self.fly_y + 150.0 * math.sin(phi_cand)
            if px < 55.0 or px > self.width - 55.0 or py < 55.0 or py > self.height - 55.0:
                score -= 600.0

            # Swarm / Multi-projectile clearance check: project candidate flight path against all bullets
            v_cand_x = 440.0 * math.cos(phi_cand)
            v_cand_y = 440.0 * math.sin(phi_cand)
            for p in self.projectiles:
                if not p.alive:
                    continue
                prx = p.x - self.fly_x
                pry = p.y - self.fly_y
                prel_vx = p.vx - v_cand_x
                prel_vy = p.vy - v_cand_y
                pv_sq = prel_vx * prel_vx + prel_vy * prel_vy
                pdot = prx * prel_vx + pry * prel_vy
                if pdot < 0.0 and pv_sq > 1.0:
                    pt_cpa = -pdot / pv_sq
                    if 0.01 < pt_cpa < 0.55:
                        pcpa_x = prx + prel_vx * pt_cpa
                        pcpa_y = pry + prel_vy * pt_cpa
                        pd_cpa = math.hypot(pcpa_x, pcpa_y)
                        if pd_cpa < 55.0:
                            score -= 10000.0 * ((55.0 - pd_cpa) / 55.0)

            # Opportunistic nectar alignment bonus
            for t in self.targets:
                if t.alive and t.type == "nectar":
                    dx = t.x - self.fly_x
                    dy = t.y - self.fly_y
                    dist = math.hypot(dx, dy)
                    n_az = (math.atan2(dy, dx) - phi_cand + math.pi) % (2 * math.pi) - math.pi
                    if dist < 320.0 and abs(n_az) < 0.6:
                        score += 40.0 * ((320.0 - dist) / 320.0)

            if score > best_score:
                best_score = score
                best_angle = d_theta

        return best_angle

    def get_closest_projectile_threat(self) -> ThreatResult:
        """
        Antennal mechanosensory & visual looming detection (Johnston's organ + Giant Fiber):
        Uses Closest Point of Approach (CPA) kinematics to identify true collision-course
        projectiles up to 240px away, providing 0.35 - 0.50s advance warning.
        Returns a ThreatResult unpackable as (az, dist, is_threat) with .safe_escape_angle attribute.
        """
        best_t_cpa = 999.0
        threat_found = False
        threat_az = 0.0
        threat_dist = 999.0

        for p in self.projectiles:
            if not p.alive:
                continue
            dx = p.x - self.fly_x
            dy = p.y - self.fly_y
            dist = math.hypot(dx, dy)

            # Track projectiles within 240px
            if dist < 240.0:
                rel_vx = p.vx - self.fly_vx
                rel_vy = p.vy - self.fly_vy
                v_sq = rel_vx * rel_vx + rel_vy * rel_vy
                dot = dx * rel_vx + dy * rel_vy  # Negative means closing in

                if dot < 0.0 and v_sq > 100.0:
                    t_cpa = -dot / v_sq
                    # Imminent collision window: arrives within next 520 ms
                    if 0.01 < t_cpa < 0.52:
                        cpa_x = dx + rel_vx * t_cpa
                        cpa_y = dy + rel_vy * t_cpa
                        d_cpa = math.hypot(cpa_x, cpa_y)

                        # True collision course if closest approach passes within 52px
                        # (physical collision threshold is fly_radius 15 + p_radius 16 = 31px)
                        if d_cpa < 52.0:
                            if t_cpa < best_t_cpa:
                                best_t_cpa = t_cpa
                                threat_found = True
                                threat_dist = dist
                                world_angle = math.atan2(dy, dx)
                                threat_az = (world_angle - self.fly_heading + math.pi) % (2 * math.pi) - math.pi

        if threat_found:
            safe_escape_angle = self.get_safe_evasion_heading(threat_az)
            return ThreatResult(threat_az, threat_dist, True, safe_escape_angle)

        return ThreatResult(0.0, 999.0, False, 0.0)

    def get_boundary_repulsion(self) -> float:
        """
        Anticipatory boundary sensing:
        Returns a steering torque that steers the fly away from arena walls before hitting them.
        """
        margin = 85.0
        torque = 0.0

        # Vector towards arena center
        center_x = self.width / 2.0
        center_y = self.height / 2.0
        angle_to_center = math.atan2(center_y - self.fly_y, center_x - self.fly_x)
        heading_err = (angle_to_center - self.fly_heading + math.pi) % (2 * math.pi) - math.pi

        # Check proximity to borders
        dist_left = self.fly_x
        dist_right = self.width - self.fly_x
        dist_top = self.fly_y
        dist_bottom = self.height - self.fly_y

        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
        if min_dist < margin:
            # Strength scales as fly approaches border
            factor = (margin - min_dist) / margin
            torque = math.sin(heading_err) * 6.0 * factor

        return torque

    def step(self, motor_action: Dict[str, float], dt: float = 0.016, is_human: bool = False) -> Dict:
        """
        Updates arena physics, entity collisions, boundary bounces, and spawns.
        Accurately tracks active flight time spent under AI bot control vs human pilot override.
        """
        self.time_elapsed += dt
        if is_human:
            self.time_human += dt
        else:
            self.time_bot += dt

        # Handle respawn if dead
        if not self.fly_alive:
            self.respawn_timer -= dt
            if self.respawn_timer <= 0.0:
                # Respawn at arena center
                self.fly_x = self.width / 2.0
                self.fly_y = self.height / 2.0
                self.fly_vx = 0.0
                self.fly_vy = 0.0
                self.fly_heading = random.uniform(-math.pi, math.pi)
                self.fly_angular_vel = 0.0
                self.fly_health = 100.0
                self.fly_alive = True
                self.score = 0
                self.time_elapsed = 0.0
                self.time_bot = 0.0
                self.time_human = 0.0
                self.nectar_collected = 0
                self.projectiles_dodged = 0
                self.game_over_recorded = False
                self.log_event("Fly respawned", (80, 240, 160))
            return {
                "fly_pos": (self.fly_x, self.fly_y),
                "fly_heading": self.fly_heading,
                "fly_speed": 0.0,
                "fly_health": self.fly_health,
                "alive": self.fly_alive,
                "score": self.score,
                "punishment": 0.0,
                "reward": 0.0,
                "closest_target_azimuth": 0.0,
                "closest_target_dist": 999.0,
                "closest_target_type": None
            }

        self.score += int(dt * 12)

        # Telemetry continuous frame recording
        self.telemetry.record_step(self, motor_action, dt)
        curr_saccade = bool(motor_action.get("saccade_active", False))
        if curr_saccade and not self._prev_saccade_active:
            self.telemetry.log_incident("DODGE", f"Emergency banked evasion (thrust={motor_action.get('thrust', 0.0):.0f} px/s)", self)
        self._prev_saccade_active = curr_saccade

        # Update fly physics with boundary anticipation
        wall_torque = self.get_boundary_repulsion()
        yaw_rate = motor_action["yaw_rate"] + wall_torque
        thrust = motor_action["thrust"]

        # Saccade angular dynamics
        self.fly_angular_vel += 24.0 * (yaw_rate - self.fly_angular_vel) * dt
        self.fly_heading += self.fly_angular_vel * dt
        self.fly_heading = (self.fly_heading + math.pi) % (2 * math.pi) - math.pi

        # Aerodynamic velocity decomposition
        head_x = math.cos(self.fly_heading)
        head_y = math.sin(self.fly_heading)
        lat_x = -head_y
        lat_y = head_x

        v_fwd = self.fly_vx * head_x + self.fly_vy * head_y
        v_lat = self.fly_vx * lat_x + self.fly_vy * lat_y

        v_lat *= max(0.0, 1.0 - 24.0 * dt)

        a_fwd = (thrust - v_fwd) * 6.0
        v_fwd += a_fwd * dt

        self.fly_vx = v_fwd * head_x + v_lat * lat_x
        self.fly_vy = v_fwd * head_y + v_lat * lat_y

        prev_x = self.fly_x
        prev_y = self.fly_y

        self.fly_x += self.fly_vx * dt
        self.fly_y += self.fly_vy * dt

        # Arena boundary bounce and heading reflection
        margin = self.fly_radius + 8
        bounced = False

        if self.fly_x > self.width - margin:
            self.fly_x = self.width - margin
            self.fly_heading = math.pi - self.fly_heading
            self.fly_vx = -abs(self.fly_vx) * 0.7 - 50.0
            self.fly_angular_vel = random.uniform(-3.0, 3.0)
            bounced = True
        elif self.fly_x < margin:
            self.fly_x = margin
            self.fly_heading = math.pi - self.fly_heading
            self.fly_vx = abs(self.fly_vx) * 0.7 + 50.0
            self.fly_angular_vel = random.uniform(-3.0, 3.0)
            bounced = True

        if self.fly_y > self.height - margin:
            self.fly_y = self.height - margin
            self.fly_heading = -self.fly_heading
            self.fly_vy = -abs(self.fly_vy) * 0.7 - 50.0
            self.fly_angular_vel = random.uniform(-3.0, 3.0)
            bounced = True
        elif self.fly_y < margin:
            self.fly_y = margin
            self.fly_heading = -self.fly_heading
            self.fly_vy = abs(self.fly_vy) * 0.7 + 50.0
            self.fly_angular_vel = random.uniform(-3.0, 3.0)
            bounced = True

        if bounced:
            self.fly_heading = (self.fly_heading + math.pi) % (2 * math.pi) - math.pi

        # Update background visual grating
        self.grating_offset = (self.grating_offset + self.grating_speed * dt) % 64.0

        # Spawns
        self.projectile_spawn_timer += dt
        spawn_rate = max(0.45, 1.25 - (self.time_elapsed * 0.012))
        if self.projectile_spawn_timer > spawn_rate:
            self.projectile_spawn_timer = 0.0
            self._spawn_projectile()

        self.target_spawn_timer += dt
        has_nectar = any(t.alive and t.type == "nectar" for t in self.targets)
        if not has_nectar:
            self._spawn_target("nectar")
        elif len(self.targets) < 5 and self.target_spawn_timer > 1.6:
            self.target_spawn_timer = 0.0
            self._spawn_target()

        # Update projectiles and collisions
        punishment_event = 0.0
        reward_event = 0.0

        for p in self.projectiles:
            p.update(dt, (self.width, self.height))
            dist_p = math.hypot(p.x - self.fly_x, p.y - self.fly_y)
            if dist_p < (self.fly_radius + p.radius):
                p.alive = False
                self.fly_health -= 25.0
                punishment_event += 1.5
                self.log_event("Direct hit (-25 HP)", (255, 80, 80))
                self.telemetry.log_incident("COLLISION_PROJECTILE", f"Direct projectile hit (-25 HP, remaining={self.fly_health:.1f}%)", self)
                self._add_particles(self.fly_x, self.fly_y, (255, 90, 50), count=16)
            elif dist_p < 75.0 and not hasattr(p, "dodged"):
                p.dodged = True
                self.projectiles_dodged += 1

        self.projectiles = [p for p in self.projectiles if p.alive]

        # Target collisions with continuous detection
        seg_x = self.fly_x - prev_x
        seg_y = self.fly_y - prev_y
        seg_len_sq = seg_x * seg_x + seg_y * seg_y
        capture_radius = 58.0

        for t in self.targets:
            t.update(dt)
            
            if seg_len_sq > 1e-6:
                u = ((t.x - prev_x) * seg_x + (t.y - prev_y) * seg_y) / seg_len_sq
                u = max(0.0, min(1.0, u))
                closest_x = prev_x + u * seg_x
                closest_y = prev_y + u * seg_y
            else:
                closest_x = self.fly_x
                closest_y = self.fly_y

            d_min = math.hypot(t.x - closest_x, t.y - closest_y)
            curr_d = math.hypot(t.x - self.fly_x, t.y - self.fly_y)

            if t.type == "nectar" and curr_d < 40.0:
                pull = 200.0 * dt
                self.fly_vx += ((t.x - self.fly_x) / max(1.0, curr_d)) * pull
                self.fly_vy += ((t.y - self.fly_y) / max(1.0, curr_d)) * pull

            col_radius = 52.0 if t.type == "nectar" else (self.fly_radius + t.radius)
            if d_min < col_radius:
                t.alive = False
                if t.type == "nectar":
                    self.fly_health = min(100.0, self.fly_health + 32.0)
                    self.score += 400
                    self.nectar_collected += 1
                    reward_event += 2.8
                    self.log_event("Nectar collected (+32 HP, +PAM)", (50, 255, 190))
                    self.telemetry.log_incident("NECTAR", f"Consumed nectar orb (+32 HP, score={self.score})", self)
                    self._add_particles(t.x, t.y, (50, 255, 190), count=20)
                else:
                    self.fly_health -= 20.0
                    punishment_event += 2.0
                    self.log_event("Hazard hit (-20 HP, +PPL1)", (255, 55, 55))
                    self.telemetry.log_incident("COLLISION_TRAP", f"Shock from static toxic trap (-20 HP, remaining={self.fly_health:.1f}%)", self)
                    self._add_particles(t.x, t.y, (255, 55, 55), count=20)

        self.targets = [t for t in self.targets if t.alive]

        # Check death & record run stats for leaderboard
        if self.fly_health <= 0.0:
            if not self.game_over_recorded:
                self.game_over_recorded = True
                mostly_human = self.time_human > self.time_bot
                self.last_run_summary = {
                    "score": self.score,
                    "time_survived": self.time_elapsed,
                    "nectar_count": self.nectar_collected,
                    "dodges": self.projectiles_dodged,
                    "is_human": mostly_human,
                    "time_bot": round(self.time_bot, 1),
                    "time_human": round(self.time_human, 1)
                }
            self.fly_alive = False
            self.fly_health = 0.0
            self.respawn_timer = 2.0
            self.log_event("Fly destroyed. Respawning...", (255, 60, 60))
            self.telemetry.save(reason="FLY_DEATH")
            self._add_particles(self.fly_x, self.fly_y, (255, 50, 50), count=25)

        # Update particles
        for part in self.particles:
            part["x"] += part["vx"] * dt
            part["y"] += part["vy"] * dt
            part["life"] -= dt
        self.particles = [part for part in self.particles if part["life"] > 0]

        # Update event timers
        for ev in self.recent_events:
            ev["time"] -= dt
        self.recent_events = [ev for ev in self.recent_events if ev["time"] > 0]

        # Target azimuth & distance
        target_az, target_dist, target_type = self.get_closest_target_info()

        return {
            "fly_pos": (self.fly_x, self.fly_y),
            "fly_heading": self.fly_heading,
            "fly_speed": math.hypot(self.fly_vx, self.fly_vy),
            "fly_health": self.fly_health,
            "alive": self.fly_alive,
            "score": self.score,
            "punishment": punishment_event,
            "reward": reward_event,
            "closest_target_azimuth": target_az,
            "closest_target_dist": target_dist,
            "closest_target_type": target_type
        }

    def _add_particles(self, x: float, y: float, color: Tuple[int, int, int], count: int = 12):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(40.0, 200.0)
            self.particles.append({
                "x": x, "y": y,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "life": random.uniform(0.3, 0.65),
                "color": color
            })

    def render(self) -> pygame.Surface:
        """
        Renders the crisp visual arena that the fly's compound eyes sample.
        """
        # Dark modern background with moving vertical gratings
        self.surface.fill((16, 20, 26))

        # Optical flow vertical stripes
        stripe_width = 32
        for x in range(-stripe_width, self.width + stripe_width, stripe_width * 2):
            draw_x = int(x + self.grating_offset)
            pygame.draw.rect(self.surface, (24, 30, 38), (draw_x, 0, stripe_width, self.height))

        # Arena border boundary glow
        pygame.draw.rect(self.surface, (45, 60, 80), (0, 0, self.width, self.height), 4)

        # Draw targets
        for t in self.targets:
            t.draw(self.surface)

        # Draw projectiles
        for p in self.projectiles:
            p.draw(self.surface)

        # Draw particles
        for part in self.particles:
            pygame.draw.circle(self.surface, part["color"], (int(part["x"]), int(part["y"])), 3)

        # Draw the fly if alive
        if self.fly_alive:
            self._draw_fly()
        else:
            # Draw respawn indicator
            txt = self.font_respawn.render(f"RESPAWNING IN {self.respawn_timer:.1f}s", True, (255, 90, 90))
            self.surface.blit(txt, (int(self.fly_x - 110), int(self.fly_y - 12)))

        return self.surface

    def _draw_fly(self):
        px = int(self.fly_x)
        py = int(self.fly_y)
        h = self.fly_heading

        cos_h = math.cos(h)
        sin_h = math.sin(h)

        # Flight direction indicator line
        nose_x = px + cos_h * 24
        nose_y = py + sin_h * 24
        pygame.draw.line(self.surface, (60, 240, 210), (px, py), (int(nose_x), int(nose_y)), 2)

        # Abdomen
        ab_x = px - cos_h * 9
        ab_y = py - sin_h * 9
        pygame.draw.circle(self.surface, (100, 85, 65), (int(ab_x), int(ab_y)), 10)
        # Thorax
        pygame.draw.circle(self.surface, (155, 125, 80), (px, py), 9)
        # Head
        hd_x = px + cos_h * 9
        hd_y = py + sin_h * 9
        pygame.draw.circle(self.surface, (175, 140, 90), (int(hd_x), int(hd_y)), 8)

        # Vivid Red Compound Eyes (Drosophila anatomy)
        eye_offset = 6.0
        eye_l_x = hd_x - sin_h * eye_offset
        eye_l_y = hd_y + cos_h * eye_offset
        eye_r_x = hd_x + sin_h * eye_offset
        eye_r_y = hd_y - cos_h * eye_offset
        pygame.draw.circle(self.surface, (245, 35, 35), (int(eye_l_x), int(eye_l_y)), 4)
        pygame.draw.circle(self.surface, (245, 35, 35), (int(eye_r_x), int(eye_r_y)), 4)

        # Translucent wings
        wing_span = 20.0
        w_l_x = px - sin_h * wing_span - cos_h * 4
        w_l_y = py + cos_h * wing_span - sin_h * 4
        w_r_x = px + sin_h * wing_span - cos_h * 4
        w_r_y = py - cos_h * wing_span - sin_h * 4
        pygame.draw.line(self.surface, (215, 235, 255), (px, py), (int(w_l_x), int(w_l_y)), 2)
        pygame.draw.line(self.surface, (215, 235, 255), (px, py), (int(w_r_x), int(w_r_y)), 2)
