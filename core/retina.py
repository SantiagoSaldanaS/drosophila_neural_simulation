"""
Compound eye retina simulation for Drosophila.

Maps an environment surface into discrete ommatidial receptive fields with
broad-spectrum (R1-R6) luminance and spectral (R7-R8) contrast channels.
"""

import numpy as np
import pygame
from typing import Tuple, List, Dict


class CompoundEye:
    """
    Simulates panoramic ommatidia array of Drosophila.
    """
    def __init__(self, num_ommatidia_per_eye: int = 375, fov_horizontal: float = 270.0, fov_vertical: float = 140.0):
        self.num_per_eye = num_ommatidia_per_eye
        self.total_ommatidia = num_ommatidia_per_eye * 2
        self.fov_h = np.radians(fov_horizontal)
        self.fov_v = np.radians(fov_vertical)
        self.acceptance_angle = np.radians(5.0)  # ~5 degree interommatidial resolution

        # Build ommatidia coordinates (azimuth, elevation)
        self.ommatidia = self._generate_ommatidia_grid()

        # Photoreceptor activations:
        self.r1_r6 = np.zeros(self.total_ommatidia, dtype=np.float32)
        self.r7_r8 = np.zeros(self.total_ommatidia, dtype=np.float32)
        self.raw_rgb = np.zeros((self.total_ommatidia, 3), dtype=np.float32)

        # Temporal filter state for photoreceptor adaptation
        self.adaptation_state = np.zeros(self.total_ommatidia, dtype=np.float32)
        self.adapt_tau = 0.05  # 50ms adaptation timescale

    def _generate_ommatidia_grid(self) -> Dict[str, np.ndarray]:
        """
        Generates hexagonal lattice of ommatidia across left and right eyes.
        """
        coords_az = []
        coords_el = []
        eye_id = []  # 0 for left, 1 for right

        # Golden spiral distribution mapped to cylindrical panoramic FOV
        for eye in [0, 1]:  # 0: left, 1: right
            az_center = -np.pi / 2 if eye == 0 else np.pi / 2
            az_span = self.fov_h / 2 + np.radians(15.0)  # include 15 deg binocular overlap

            indices = np.arange(0, self.num_per_eye, dtype=float) + 0.5
            phi = np.arccos(1 - 2 * indices / self.num_per_eye)  # elevation angle
            theta = np.pi * (1 + 5**0.5) * indices               # golden angle azimuth

            # Scale and center to eye's visual hemisphere
            norm_el = (phi - np.pi / 2) * (self.fov_v / np.pi)
            norm_az = az_center + (np.mod(theta, 2 * np.pi) - np.pi) * (az_span / np.pi)

            coords_az.extend(norm_az)
            coords_el.extend(norm_el)
            eye_id.extend([eye] * self.num_per_eye)

        az = np.array(coords_az, dtype=np.float32)
        el = np.array(coords_el, dtype=np.float32)
        eyes = np.array(eye_id, dtype=np.int32)

        # Sort left eye from back to front, right eye from front to back
        left_mask = eyes == 0
        right_mask = eyes == 1
        sort_left = np.lexsort((el[left_mask], az[left_mask]))
        sort_right = np.lexsort((el[right_mask], az[right_mask]))

        final_az = np.concatenate([az[left_mask][sort_left], az[right_mask][sort_right]])
        final_el = np.concatenate([el[left_mask][sort_left], el[right_mask][sort_right]])
        final_eyes = np.concatenate([eyes[left_mask][sort_left], eyes[right_mask][sort_right]])

        return {
            "azimuth": final_az,
            "elevation": final_el,
            "eye": final_eyes
        }

    def sample_arena(self, arena_surface: pygame.Surface, fly_pos: Tuple[float, float], fly_heading: float, arena_size: Tuple[int, int], dt: float = 0.016):
        """
        Samples the visual arena from the fly's egocentric perspective.
        """
        w, h = arena_size
        px, py = fly_pos
        az = self.ommatidia["azimuth"]
        world_angles = fly_heading + az
        
        sample_dists = [18.0, 38.0, 68.0, 115.0, 180.0]
        weights = [0.35, 0.25, 0.20, 0.12, 0.08]

        cos_angles = np.cos(world_angles)
        sin_angles = np.sin(world_angles)

        surf_w, surf_h = arena_surface.get_size()
        arr = pygame.surfarray.pixels3d(arena_surface)

        sampled_rgb = np.zeros((self.total_ommatidia, 3), dtype=np.float32)

        for d, weight in zip(sample_dists, weights):
            sample_x = np.clip(np.int32(px + cos_angles * d), 0, surf_w - 1)
            sample_y = np.clip(np.int32(py + sin_angles * d), 0, surf_h - 1)
            pts_rgb = arr[sample_x, sample_y].astype(np.float32) / 255.0
            sampled_rgb += pts_rgb * weight

        del arr

        self.raw_rgb = sampled_rgb

        # R1-R6 broad-spectrum luminance
        luminance = 0.299 * sampled_rgb[:, 0] + 0.587 * sampled_rgb[:, 1] + 0.114 * sampled_rgb[:, 2]

        # Temporal high-pass filtering
        alpha = dt / (self.adapt_tau + dt)
        self.adaptation_state = (1.0 - alpha) * self.adaptation_state + alpha * luminance
        self.r1_r6 = np.clip(luminance - self.adaptation_state + 0.5, 0.0, 1.0)

        # R7/R8 spectral contrast
        self.r7_r8 = np.clip((sampled_rgb[:, 1] + sampled_rgb[:, 2]) - sampled_rgb[:, 0] + 0.5, 0.0, 1.0)

        return self.r1_r6, self.r7_r8

    def get_layout_2d(self, canvas_width: int, canvas_height: int, margin: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns normalized (x, y, eye_id) coordinates centered within canvas bounds.
        """
        az = self.ommatidia["azimuth"]
        el = self.ommatidia["elevation"]
        eyes = self.ommatidia["eye"]

        min_az = np.min(az)
        max_az = np.max(az)
        norm_x = (az - min_az) / (max_az - min_az + 1e-6)

        min_el = np.min(el)
        max_el = np.max(el)
        norm_y = (el - min_el) / (max_el - min_el + 1e-6)

        usable_w = canvas_width - 2 * margin
        usable_h = canvas_height - 2 * margin

        x = norm_x * usable_w + margin
        y = norm_y * usable_h + margin
        
        return x.astype(np.int32), y.astype(np.int32), eyes
