"""
Telemetry and flight logger for Drosophila simulation.

Records trajectory ring buffers, incident events (dodges, collisions,
food collection), and exports session summaries to JSON and text.
"""

import os
import json
import time
import math
from collections import deque
from typing import Dict, List, Optional, Any


class TelemetryLogger:
    def __init__(self, log_dir: Optional[str] = None, ring_buffer_size: int = 300):
        if log_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_dir = os.path.join(base_dir, "logs")
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.json_path = os.path.join(self.log_dir, "latest_game.json")
        self.summary_path = os.path.join(self.log_dir, "latest_game_summary.txt")

        self.ring_buffer_size = ring_buffer_size
        self.history = deque(maxlen=ring_buffer_size)
        self.incidents: List[Dict[str, Any]] = []

        # Run statistics
        self.start_time = time.time()
        self.total_steps = 0
        self.sim_time = 0.0
        self.dodges = 0
        self.nectar_collected = 0
        self.collisions_trap = 0
        self.collisions_projectile = 0
        self.peak_score = 0
        self.final_health = 100.0

    def reset(self):
        """Resets run statistics for a new game run."""
        self.history.clear()
        self.incidents.clear()
        self.start_time = time.time()
        self.total_steps = 0
        self.sim_time = 0.0
        self.dodges = 0
        self.nectar_collected = 0
        self.collisions_trap = 0
        self.collisions_projectile = 0
        self.peak_score = 0
        self.final_health = 100.0

    def record_step(self, arena: Any, motor_action: Dict[str, Any], dt: float = 0.016):
        """
        Records a single simulation timestep into the in-memory ring buffer.
        """
        self.total_steps += 1
        self.sim_time += dt
        self.peak_score = max(self.peak_score, arena.score)
        self.final_health = arena.fly_health

        # Snapshot active targets
        static_traps = []
        nectar_orbs = []
        for t in arena.targets:
            if t.alive:
                dist = math.hypot(t.x - arena.fly_x, t.y - arena.fly_y)
                item = {
                    "x": round(t.x, 1),
                    "y": round(t.y, 1),
                    "dist": round(dist, 1)
                }
                if t.type == "toxic_trap":
                    static_traps.append(item)
                else:
                    nectar_orbs.append(item)

        # Snapshot active projectiles
        projectiles = []
        for p in arena.projectiles:
            if p.alive:
                dist = math.hypot(p.x - arena.fly_x, p.y - arena.fly_y)
                projectiles.append({
                    "x": round(p.x, 1),
                    "y": round(p.y, 1),
                    "vx": round(p.vx, 1),
                    "vy": round(p.vy, 1),
                    "speed": round(math.hypot(p.vx, p.vy), 1),
                    "dist": round(dist, 1)
                })

        snapshot = {
            "step": self.total_steps,
            "sim_time": round(self.sim_time, 3),
            "fly": {
                "x": round(arena.fly_x, 1),
                "y": round(arena.fly_y, 1),
                "heading_deg": round(math.degrees(arena.fly_heading), 1),
                "speed": round(math.hypot(arena.fly_vx, arena.fly_vy), 1),
                "health": round(arena.fly_health, 1),
                "alive": arena.fly_alive
            },
            "motor": {
                "yaw_rate": round(motor_action.get("yaw_rate", 0.0), 3),
                "thrust": round(motor_action.get("thrust", 0.0), 1),
                "saccade_active": bool(motor_action.get("saccade_active", False))
            },
            "traps_count": len(static_traps),
            "nectar_count": len(nectar_orbs),
            "projectiles_count": len(projectiles),
            "closest_trap_dist": min((t["dist"] for t in static_traps), default=999.0),
            "closest_projectile_dist": min((p["dist"] for p in projectiles), default=999.0)
        }

        self.history.append(snapshot)

    def log_incident(self,
                     incident_type: str,
                     description: str,
                     arena: Any,
                     extra_data: Optional[Dict[str, Any]] = None):
        """
        Logs a key game event (DODGE, COLLISION_TRAP, COLLISION_PROJECTILE, NECTAR).
        """
        if incident_type == "DODGE":
            self.dodges += 1
        elif incident_type == "COLLISION_TRAP":
            self.collisions_trap += 1
        elif incident_type == "COLLISION_PROJECTILE":
            self.collisions_projectile += 1
        elif incident_type == "NECTAR":
            self.nectar_collected += 1

        traps_nearby = []
        for t in arena.targets:
            if t.alive and t.type == "toxic_trap":
                dist = math.hypot(t.x - arena.fly_x, t.y - arena.fly_y)
                if dist < 220.0:
                    traps_nearby.append({
                        "x": round(t.x, 1),
                        "y": round(t.y, 1),
                        "dist": round(dist, 1),
                        "azimuth_deg": round(math.degrees(
                            (math.atan2(t.y - arena.fly_y, t.x - arena.fly_x) - arena.fly_heading + math.pi) % (2 * math.pi) - math.pi
                        ), 1)
                    })

        projectiles_nearby = []
        for p in arena.projectiles:
            if p.alive:
                dist = math.hypot(p.x - arena.fly_x, p.y - arena.fly_y)
                if dist < 180.0:
                    projectiles_nearby.append({
                        "x": round(p.x, 1),
                        "y": round(p.y, 1),
                        "speed": round(math.hypot(p.vx, p.vy), 1),
                        "dist": round(dist, 1),
                        "azimuth_deg": round(math.degrees(
                            (math.atan2(p.y - arena.fly_y, p.x - arena.fly_x) - arena.fly_heading + math.pi) % (2 * math.pi) - math.pi
                        ), 1)
                    })

        incident = {
            "step": self.total_steps,
            "sim_time": round(self.sim_time, 2),
            "type": incident_type,
            "description": description,
            "fly": {
                "x": round(arena.fly_x, 1),
                "y": round(arena.fly_y, 1),
                "heading_deg": round(math.degrees(arena.fly_heading), 1),
                "speed": round(math.hypot(arena.fly_vx, arena.fly_vy), 1),
                "health": round(arena.fly_health, 1)
            },
            "nearby_traps": traps_nearby,
            "nearby_projectiles": projectiles_nearby,
            "extra": extra_data or {}
        }
        self.incidents.append(incident)

    def save(self, reason: str = "GAME_COMPLETE"):
        """
        Exports both logs/latest_game.json and logs/latest_game_summary.txt.
        """
        data = {
            "run_metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "reason": reason,
                "sim_duration_sec": round(self.sim_time, 2),
                "total_steps": self.total_steps,
                "peak_score": self.peak_score,
                "final_health": round(self.final_health, 1),
                "nectar_collected": self.nectar_collected,
                "dodges_executed": self.dodges,
                "trap_collisions": self.collisions_trap,
                "projectile_collisions": self.collisions_projectile
            },
            "incidents": self.incidents,
            "recent_trajectory": list(self.history)
        }

        # 1. Save JSON
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[TelemetryLogger] Failed to write JSON log: {e}")

        # 2. Save Human-Readable Text Summary
        try:
            with open(self.summary_path, "w", encoding="utf-8") as f:
                f.write("=" * 70 + "\n")
                f.write("DROSOPHILA BIO-VISION SIMULATION - LATEST GAME TELEMETRY REPORT\n")
                f.write("=" * 70 + "\n")
                f.write(f"Date / Time:           {data['run_metadata']['timestamp']}\n")
                f.write(f"Exit Reason:           {data['run_metadata']['reason']}\n")
                f.write(f"Survival Time:         {data['run_metadata']['sim_duration_sec']:.1f}s ({data['run_metadata']['total_steps']} steps)\n")
                f.write(f"Final Score:           {data['run_metadata']['peak_score']}\n")
                f.write(f"Final Health:          {data['run_metadata']['final_health']:.1f}%\n")
                f.write(f"Nectar Collected:      {data['run_metadata']['nectar_collected']} orbs\n")
                f.write(f"Looming Dodges:        {data['run_metadata']['dodges_executed']} successful evasions\n")
                f.write(f"Collisions (Trap):     {data['run_metadata']['trap_collisions']}\n")
                f.write(f"Collisions (Project):  {data['run_metadata']['projectile_collisions']}\n")
                f.write("-" * 70 + "\n")
                f.write("INCIDENT LOG (Key Dodges & Collisions):\n")
                if not self.incidents:
                    f.write("  (No collisions or high-urgency incidents occurred during this run.)\n")
                else:
                    for inc in self.incidents:
                        f.write(f"  [{inc['sim_time']:6.1f}s] {inc['type']:<20} {inc['description']}\n")
                        f.write(f"           Fly: pos=({inc['fly']['x']:.1f}, {inc['fly']['y']:.1f}) head={inc['fly']['heading_deg']:.1f} deg spd={inc['fly']['speed']:.1f}px/s hp={inc['fly']['health']:.1f}%\n")
                        if inc["nearby_traps"]:
                            traps_str = ", ".join(f"trap at ({t['x']:.0f},{t['y']:.0f}) d={t['dist']:.0f} az={t['azimuth_deg']:.0f} deg" for t in inc["nearby_traps"])
                            f.write(f"           Nearby Static Traps: {traps_str}\n")
                        if inc["nearby_projectiles"]:
                            projs_str = ", ".join(f"proj d={p['dist']:.0f} az={p['azimuth_deg']:.0f} deg spd={p['speed']:.0f}" for p in inc["nearby_projectiles"])
                            f.write(f"           Nearby Projectiles:  {projs_str}\n")
                        f.write("\n")
                f.write("=" * 70 + "\n")
        except Exception as e:
            print(f"[TelemetryLogger] Failed to write text summary: {e}")
