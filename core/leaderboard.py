"""
Leaderboard and ranking manager for Drosophila flight simulation.

Persists high scores, survival times, and ranks for autonomous
and human-controlled flights in leaderboard.json.
"""

import os
import json
import time
from typing import List, Dict, Tuple, Optional


class LeaderboardManager:
    """
    Manages persistent high scores and ranks for Fly Brain AI and Human Pilot modes.
    """
    def __init__(self, filepath: Optional[str] = None):
        if filepath is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            filepath = os.path.join(base_dir, "leaderboard.json")
        self.filepath = filepath

        self.fly_scores: List[Dict] = []
        self.human_scores: List[Dict] = []
        self.load()

    def get_rank_title(self, score: int) -> Tuple[str, str]:
        """Returns (rank_title, color_hex) based on score threshold."""
        if score >= 2500000:
            return "Eternal Chrono-Deity", "#FFD700"
        elif score >= 1000000:
            return "Quantum Connectome Titan", "#00FFFF"
        elif score >= 500000:
            return "Planetary Swarm Mind", "#E066FF"
        elif score >= 100000:
            return "Immortal Transcendent", "#FFE74C"
        elif score >= 60000:
            return "Apex Connectome Master", "#38EFD4"
        elif score >= 35000:
            return "Synaptic Titan", "#B57BFF"
        elif score >= 20000:
            return "Bullet Hell Maestro", "#FF5E7E"
        elif score >= 10000:
            return "Drosophila Veteran", "#55FF87"
        elif score >= 5000:
            return "Aerial Saccade Ace", "#FFB93C"
        elif score >= 2000:
            return "Fruit Fly Fledgling", "#4AA8FF"
        else:
            return "Larval Navigator", "#A0AEC0"

    def add_score(self, is_human: bool, score: int, time_survived: float, nectar_count: int, dodges: int = 0) -> bool:
        """
        Records a completed run score. Returns True if it is a new top-5 record.
        Maintains score history up to 1,000 entries.
        """
        if score <= 0:
            return False

        entry = {
            "score": int(score),
            "time_survived": round(float(time_survived), 1),
            "nectar_count": int(nectar_count),
            "dodges": int(dodges),
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "rank": self.get_rank_title(score)[0]
        }

        target_list = self.human_scores if is_human else self.fly_scores
        target_list.append(entry)
        target_list.sort(key=lambda x: x["score"], reverse=True)
        if len(target_list) > 1000:
            del target_list[1000:]

        self.save()
        return entry in target_list[:5]

    def get_top_fly_scores(self, limit: Optional[int] = None) -> List[Dict]:
        return self.fly_scores if limit is None else self.fly_scores[:limit]

    def get_top_human_scores(self, limit: Optional[int] = None) -> List[Dict]:
        return self.human_scores if limit is None else self.human_scores[:limit]

    def get_fly_high_score(self) -> int:
        return self.fly_scores[0]["score"] if self.fly_scores else 0

    def get_human_high_score(self) -> int:
        return self.human_scores[0]["score"] if self.human_scores else 0

    def load(self):
        """Loads leaderboard from disk and sorts entries."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.fly_scores = data.get("fly_scores", [])
                    self.human_scores = data.get("human_scores", [])
            except Exception as e:
                print(f"Error loading leaderboard: {e}")
                self._init_defaults()
        else:
            self._init_defaults()

        for entry in self.fly_scores + self.human_scores:
            entry["rank"] = self.get_rank_title(entry.get("score", 0))[0]

        self.fly_scores.sort(key=lambda x: x["score"], reverse=True)
        self.human_scores.sort(key=lambda x: x["score"], reverse=True)
        if len(self.fly_scores) > 1000:
            del self.fly_scores[1000:]
        if len(self.human_scores) > 1000:
            del self.human_scores[1000:]
        self.save()

    def _init_defaults(self):
        """Initial default leaderboard benchmarks."""
        self.fly_scores = [
            {"score": 1850, "time_survived": 45.2, "nectar_count": 4, "dodges": 28, "date": "Benchmark", "rank": "Aerial Saccade Ace"},
            {"score": 950, "time_survived": 28.1, "nectar_count": 2, "dodges": 16, "date": "Benchmark", "rank": "Fruit Fly Fledgling"},
            {"score": 450, "time_survived": 14.5, "nectar_count": 1, "dodges": 8, "date": "Benchmark", "rank": "Larval Navigator"}
        ]
        self.human_scores = [
            {"score": 2400, "time_survived": 52.0, "nectar_count": 5, "dodges": 34, "date": "Benchmark", "rank": "Aerial Saccade Ace"},
            {"score": 1200, "time_survived": 32.4, "nectar_count": 3, "dodges": 19, "date": "Benchmark", "rank": "Fruit Fly Fledgling"}
        ]
        self.save()

    def save(self):
        """Persists leaderboard to disk."""
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump({
                    "fly_scores": self.fly_scores,
                    "human_scores": self.human_scores
                }, f, indent=2)
        except Exception as e:
            print(f"Error saving leaderboard: {e}")
