"""
Neural telemetry dashboard and visualization HUD for Drosophila simulation.

Displays retina ommatidia activations, optic lobe flow and looming detection,
central complex heading compass, and mushroom body reinforcement channels.
"""

import math
import numpy as np
import pygame
from typing import Dict, Tuple, Optional


class NeuralDashboard:
    """Telemetry HUD displaying live circuit activations and fly state."""
    def __init__(self, hud_width: int = 720, hud_height: int = 850):
        self.width = hud_width
        self.height = hud_height
        self.surface = pygame.Surface((hud_width, hud_height))

        # Color palette
        self.bg_color = (13, 17, 23)
        self.panel_bg = (21, 27, 36)
        self.border_color = (48, 62, 82)
        self.cyan = (56, 239, 212)
        self.red = (255, 75, 85)
        self.amber = (255, 185, 60)
        self.green = (85, 255, 135)
        self.white = (245, 248, 252)
        self.gray = (145, 160, 180)

        self._recompute_layout()

    def set_dimensions(self, width: int, height: int):
        """Dynamically resizes HUD surface and recalculates proportional layout."""
        if width == self.width and height == self.height:
            return
        self.width = max(500, width)
        self.height = max(600, height)
        self.surface = pygame.Surface((self.width, self.height))
        self._recompute_layout()

    def _recompute_layout(self):
        """Recalculates proportional panel heights and font sizes to eliminate empty space."""
        pygame.font.init()
        font_main = "Segoe UI" if "segoeui" in pygame.font.get_fonts() else "Arial"
        font_mono = "Consolas" if "consolas" in pygame.font.get_fonts() else "Courier New"

        # Scale factor based on display resolution (reference 720x850)
        self.scale = max(0.9, min(self.width / 720.0, self.height / 850.0))

        self.font_header = pygame.font.SysFont(font_main, max(16, int(20 * self.scale)), bold=True)
        self.font_sub = pygame.font.SysFont(font_main, max(11, int(13 * self.scale)))
        self.font_section = pygame.font.SysFont(font_main, max(13, int(15 * self.scale)), bold=True)
        self.font_label = pygame.font.SysFont(font_main, max(11, int(13 * self.scale)), bold=True)
        self.font_mono = pygame.font.SysFont(font_mono, max(11, int(13 * self.scale)))
        self.font_mono_bold = pygame.font.SysFont(font_mono, max(12, int(14 * self.scale)), bold=True)
        self.font_alert = pygame.font.SysFont(font_main, max(13, int(16 * self.scale)), bold=True)

        # Vertical proportional allocation: 4 panels evenly filling available height
        self.header_h = int(48 * self.scale)
        self.gap = max(8, int(12 * self.scale))
        self.bottom_margin = max(10, int(14 * self.scale))
        
        avail_h = self.height - self.header_h - self.bottom_margin - (3 * self.gap)

        # Panel height weights: 21%, 21%, 21%, 37%
        self.p1_h = int(avail_h * 0.21)
        self.p2_h = int(avail_h * 0.21)
        self.p3_h = int(avail_h * 0.21)
        self.p4_h = avail_h - (self.p1_h + self.p2_h + self.p3_h)

        self.y1 = self.header_h + 4
        self.y2 = self.y1 + self.p1_h + self.gap
        self.y3 = self.y2 + self.p2_h + self.gap
        self.y4 = self.y3 + self.p3_h + self.gap

    def render(self,
               retina_layout: Tuple[np.ndarray, np.ndarray, np.ndarray],
               retina_rgb: np.ndarray,
               optomotor_data: Dict[str, float],
               cx_data: Dict[str, float],
               mb_data: Dict[str, float],
               motor_data: Dict[str, float],
               game_stats: Dict) -> pygame.Surface:
        """
        Renders the complete telemetry dashboard with balanced proportions.
        """
        self.surface.fill(self.bg_color)

        self._draw_header()
        self._draw_retina_panel(retina_layout, retina_rgb, self.y1, self.p1_h)
        self._draw_optic_lobe_panel(optomotor_data, motor_data, self.y2, self.p2_h)
        self._draw_central_complex_panel(cx_data, game_stats["fly_heading"], self.y3, self.p3_h)
        self._draw_mushroom_body_panel(mb_data, self.y4, self.p4_h)

        return self.surface

    def _draw_header(self):
        title = self.font_header.render("DROSOPHILA CNS NEURAL TELEMETRY", True, self.cyan)
        sub = self.font_sub.render("Live In-Silico Connectome & 3-Factor Plasticity Engine", True, self.gray)
        pad_x = max(16, int(24 * self.scale))
        self.surface.blit(title, (pad_x, 8))
        self.surface.blit(sub, (pad_x, 8 + int(22 * self.scale)))
        pygame.draw.line(self.surface, self.border_color, (14, self.header_h), (self.width - 14, self.header_h), 1)

    def _draw_retina_panel(self, layout: Tuple[np.ndarray, np.ndarray, np.ndarray], rgb: np.ndarray, y_offset: int, p_height: int):
        rect = pygame.Rect(16, y_offset, self.width - 32, p_height)
        pygame.draw.rect(self.surface, self.panel_bg, rect, border_radius=8)
        pygame.draw.rect(self.surface, self.border_color, rect, 1, border_radius=8)

        title = self.font_section.render("1. COMPOUND EYE (750 OMMATIDIA PANORAMA)", True, self.amber)
        self.surface.blit(title, (28, y_offset + 8))

        cx, cy, eyes = layout
        canvas_w = self.width - 64
        canvas_h = max(60, p_height - int(48 * self.scale))
        pad_x = 32
        pad_y = y_offset + int(32 * self.scale)

        # Dark eye hemisphere backdrop
        pygame.draw.rect(self.surface, (10, 13, 18), (pad_x, pad_y, canvas_w, canvas_h), border_radius=6)
        midline_x = pad_x + canvas_w // 2
        pygame.draw.line(self.surface, (45, 58, 75), (midline_x, pad_y), (midline_x, pad_y + canvas_h), 1)

        # Render ommatidia dots scaled to canvas bounds
        num_ommatidia = len(rgb)
        max_cx = max(1.0, np.max(cx))
        max_cy = max(1.0, np.max(cy))
        scaled_x = (cx / max_cx) * (canvas_w - 20) + pad_x + 10
        scaled_y = (cy / max_cy) * (canvas_h - 20) + pad_y + 10

        for i in range(num_ommatidia):
            color = (int(np.clip(rgb[i, 0] * 255 + 25, 0, 255)),
                     int(np.clip(rgb[i, 1] * 255 + 25, 0, 255)),
                     int(np.clip(rgb[i, 2] * 255 + 25, 0, 255)))
            pygame.draw.circle(self.surface, color, (int(scaled_x[i]), int(scaled_y[i])), 3)

        label_l = self.font_mono.render("[L] Left Eye (135 deg FOV)", True, self.gray)
        label_r = self.font_mono.render("[R] Right Eye (135 deg FOV)", True, self.gray)
        self.surface.blit(label_l, (pad_x + 10, pad_y + canvas_h - int(16 * self.scale)))
        self.surface.blit(label_r, (pad_x + canvas_w - int(180 * self.scale), pad_y + canvas_h - int(16 * self.scale)))

    def _draw_optic_lobe_panel(self, optomotor_data: Dict[str, float], motor_data: Dict[str, float], y_offset: int, p_height: int):
        rect = pygame.Rect(16, y_offset, self.width - 32, p_height)
        pygame.draw.rect(self.surface, self.panel_bg, rect, border_radius=8)
        pygame.draw.rect(self.surface, self.border_color, rect, 1, border_radius=8)

        title = self.font_section.render("2. OPTIC LOBE: OPTIC FLOW & LOOMING DETECTORS", True, self.amber)
        self.surface.blit(title, (28, y_offset + 8))

        hs_l = optomotor_data["hs_left"]
        hs_r = optomotor_data["hs_right"]
        looming_alert = optomotor_data["looming_alert"]
        saccade = motor_data["saccade_active"]

        # HS Flow readouts
        lbl_hs = self.font_mono_bold.render(f"LPTC HS Flow: L={hs_l:+.1f} | R={hs_r:+.1f}", True, self.white)
        self.surface.blit(lbl_hs, (28, y_offset + int(32 * self.scale)))

        # Flow bar
        bar_w = int((self.width - 64) * 0.44)
        center_x = 28 + bar_w // 2
        bar_y = y_offset + int(54 * self.scale)
        bar_h = max(14, int(16 * self.scale))
        pygame.draw.rect(self.surface, (28, 36, 50), (28, bar_y, bar_w, bar_h), border_radius=4)
        diff = np.clip((hs_l - hs_r) * 1.8, -bar_w // 2, bar_w // 2)
        if diff > 0:
            pygame.draw.rect(self.surface, self.cyan, (center_x, bar_y, int(diff), bar_h), border_radius=2)
        else:
            pygame.draw.rect(self.surface, self.amber, (center_x + int(diff), bar_y, int(-diff), bar_h), border_radius=2)
        pygame.draw.line(self.surface, self.white, (center_x, bar_y - 2), (center_x, bar_y + bar_h + 2), 2)

        # Looming Giant Fiber Box
        gf_box_x = 28 + bar_w + 24
        gf_box_w = self.width - gf_box_x - 32
        gf_box_h = max(70, p_height - int(45 * self.scale))
        gf_box = pygame.Rect(gf_box_x, y_offset + int(30 * self.scale), gf_box_w, gf_box_h)
        
        if looming_alert or saccade:
            pygame.draw.rect(self.surface, (95, 25, 32), gf_box, border_radius=8)
            pygame.draw.rect(self.surface, self.red, gf_box, 2, border_radius=8)
            alert_txt = self.font_alert.render("LOOMING THREAT IMMINENT!", True, self.red)
            sub_txt = self.font_sub.render("GIANT FIBER: Escape Jump Active", True, self.white)
            bias_txt = self.font_mono.render(f"Veer Direction: {motor_data.get('yaw_rate', 0.0):+.1f} rad/s", True, self.amber)
            self.surface.blit(alert_txt, (gf_box_x + 12, y_offset + int(38 * self.scale)))
            self.surface.blit(sub_txt, (gf_box_x + 12, y_offset + int(60 * self.scale)))
            self.surface.blit(bias_txt, (gf_box_x + 12, y_offset + int(82 * self.scale)))
        else:
            pygame.draw.rect(self.surface, (16, 22, 30), gf_box, border_radius=8)
            pygame.draw.rect(self.surface, self.border_color, gf_box, 1, border_radius=8)
            status_txt = self.font_label.render("GIANT FIBER: PASSIVE SCAN", True, self.green)
            loom_val = max(optomotor_data["looming_left"], optomotor_data["looming_right"])
            sub_txt = self.font_mono.render(f"Expansion: {loom_val:.2f} (Thresh: 0.45)", True, self.gray)
            status_txt2 = self.font_sub.render("No collision vectors detected", True, (120, 135, 155))
            self.surface.blit(status_txt, (gf_box_x + 12, y_offset + int(38 * self.scale)))
            self.surface.blit(sub_txt, (gf_box_x + 12, y_offset + int(60 * self.scale)))
            self.surface.blit(status_txt2, (gf_box_x + 12, y_offset + int(82 * self.scale)))

        status_line = self.font_sub.render(
            f"Optomotor Yaw Torque: {motor_data['optomotor_component']:+.2f} rad/s", True, self.gray)
        self.surface.blit(status_line, (28, y_offset + int(82 * self.scale)))

    def _draw_central_complex_panel(self, cx_data: Dict[str, float], actual_heading: float, y_offset: int, p_height: int):
        rect = pygame.Rect(16, y_offset, self.width - 32, p_height)
        pygame.draw.rect(self.surface, self.panel_bg, rect, border_radius=8)
        pygame.draw.rect(self.surface, self.border_color, rect, 1, border_radius=8)

        title = self.font_section.render("3. CENTRAL COMPLEX (CX): E-PG HEADING COMPASS", True, self.amber)
        self.surface.blit(title, (28, y_offset + 8))

        compass_cx = 90 + int(20 * self.scale)
        compass_cy = y_offset + p_height // 2 + 6
        ring_r = min(int(46 * self.scale), p_height // 3)

        pygame.draw.circle(self.surface, (28, 38, 52), (compass_cx, compass_cy), ring_r, 4)
        
        # 16 compass wedges
        wedge_idx = cx_data["epg_bump_max_wedge"]
        for w in range(16):
            angle = (w / 16.0) * 2 * math.pi - math.pi
            wx = compass_cx + math.cos(angle) * ring_r
            wy = compass_cy + math.sin(angle) * ring_r
            color = self.cyan if w == wedge_idx else (55, 72, 95)
            r = 5 if w == wedge_idx else 2
            pygame.draw.circle(self.surface, color, (int(wx), int(wy)), r)

        head_est = cx_data["heading_estimate"]
        nx = compass_cx + math.cos(head_est) * (ring_r - 8)
        ny = compass_cy + math.sin(head_est) * (ring_r - 8)
        pygame.draw.line(self.surface, self.cyan, (compass_cx, compass_cy), (int(nx), int(ny)), 3)
        pygame.draw.circle(self.surface, self.white, (compass_cx, compass_cy), 4)

        text_x = compass_cx + ring_r + 30
        txt_h = self.font_mono_bold.render(f"E-PG Heading Angle: {math.degrees(head_est):.1f} deg", True, self.white)
        txt_act = self.font_mono.render(f"True Body Heading: {math.degrees(actual_heading):.1f} deg", True, self.gray)
        txt_steer = self.font_sub.render(f"Course Vector Torque: {cx_data['steering_signal']:+.2f}", True, self.cyan)
        
        self.surface.blit(txt_h, (text_x, y_offset + int(32 * self.scale)))
        self.surface.blit(txt_act, (text_x, y_offset + int(56 * self.scale)))
        self.surface.blit(txt_steer, (text_x, y_offset + int(80 * self.scale)))

    def _draw_mushroom_body_panel(self, mb_data: Dict[str, float], y_offset: int, p_height: int):
        rect = pygame.Rect(16, y_offset, self.width - 32, p_height)
        pygame.draw.rect(self.surface, self.panel_bg, rect, border_radius=8)
        pygame.draw.rect(self.surface, self.border_color, rect, 1, border_radius=8)

        title = self.font_section.render("4. MUSHROOM BODY: 3-FACTOR PLASTICITY CIRCUIT", True, self.amber)
        sub = self.font_sub.render("Real-Time Associative Learning & Neuromodulation", True, self.gray)
        self.surface.blit(title, (28, y_offset + 8))
        self.surface.blit(sub, (28, y_offset + int(28 * self.scale)))

        kc_count = mb_data["kc_active_count"]
        kc_txt = self.font_mono.render(f"Kenyon Cells: {kc_count} / 1500 active (Sparse ~{kc_count/15:.1f}%)", True, self.white)
        self.surface.blit(kc_txt, (28, y_offset + int(50 * self.scale)))

        # Dopamine gauges
        ppl1 = mb_data["ppl1_dopamine"]
        pam = mb_data["pam_dopamine"]
        gauge_w = int((self.width - 90) / 2)

        # PAM (Reward)
        lbl_pam = self.font_mono_bold.render(f"PAM Dopamine (Nectar): {pam:.2f}", True, self.cyan)
        pam_y = y_offset + int(74 * self.scale)
        self.surface.blit(lbl_pam, (28, pam_y))
        pygame.draw.rect(self.surface, (22, 32, 45), (28, pam_y + int(20 * self.scale), gauge_w, 12), border_radius=3)
        pam_w = int(np.clip(pam / 3.0, 0.0, 1.0) * gauge_w)
        if pam_w > 0:
            pygame.draw.rect(self.surface, self.cyan, (28, pam_y + int(20 * self.scale), pam_w, 12), border_radius=3)

        # PPL1 (Punishment)
        ppl_x = 28 + gauge_w + 24
        lbl_ppl1 = self.font_mono_bold.render(f"PPL1 Dopamine (Damage): {ppl1:.2f}", True, self.red)
        self.surface.blit(lbl_ppl1, (ppl_x, pam_y))
        pygame.draw.rect(self.surface, (45, 22, 28), (ppl_x, pam_y + int(20 * self.scale), gauge_w, 12), border_radius=3)
        ppl1_w = int(np.clip(ppl1 / 3.0, 0.0, 1.0) * gauge_w)
        if ppl1_w > 0:
            pygame.draw.rect(self.surface, self.red, (ppl_x, pam_y + int(20 * self.scale), ppl1_w, 12), border_radius=3)

        # MBON Output Valence
        app = mb_data["mbon_approach"]
        avoid = mb_data["mbon_avoid"]
        net_val = mb_data["net_valence"]

        val_y = pam_y + int(42 * self.scale)
        val_title = self.font_mono_bold.render(
            f"MBON Valence: {net_val:+.2f}  [Approach: {app:.2f} | Avoid: {avoid:.2f}]", True, self.white)
        self.surface.blit(val_title, (28, val_y))

        # Valence balance slider (-1.5 to +1.5)
        slider_w = self.width - 64
        slider_y = val_y + int(22 * self.scale)
        slider_h = max(14, int(18 * self.scale))
        pygame.draw.rect(self.surface, (22, 30, 42), (28, slider_y, slider_w, slider_h), border_radius=4)
        mid_x = 28 + slider_w // 2
        pygame.draw.line(self.surface, self.gray, (mid_x, slider_y - 2), (mid_x, slider_y + slider_h + 2), 2)
        
        norm_val = np.clip(net_val / 1.5, -1.0, 1.0)
        slide_x = mid_x + int(norm_val * (slider_w // 2 - 12))
        indicator_color = self.green if net_val > 0.08 else (self.red if net_val < -0.08 else self.amber)
        pygame.draw.circle(self.surface, indicator_color, (slide_x, slider_y + slider_h // 2), 8)

        val_l = self.font_label.render("AVERSION (FLEE)", True, self.red)
        val_r = self.font_label.render("APPETITIVE (APPROACH)", True, self.green)
        self.surface.blit(val_l, (28, slider_y + slider_h + 4))
        self.surface.blit(val_r, (28 + slider_w - int(210 * self.scale), slider_y + slider_h + 4))

        # Scientific rule footer
        footer_y = p_height - int(38 * self.scale)
        footer_rule = self.font_mono.render(
            "Plastic Rule: dW_ij = eta * E_ij(t) * (PAM_dopamine - PPL1_dopamine)", True, self.cyan)
        self.surface.blit(footer_rule, (28, y_offset + footer_y))
