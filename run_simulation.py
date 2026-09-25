"""
Main simulation entry point for the Drosophila bio-vision and connectome model.

Runs closed-loop visual flight simulation with compound eye rendering,
optic flow calculation, central complex navigation, mushroom body reinforcement,
and interactive telemetry HUD.

Usage:
    python run_simulation.py                # Launch interactive simulation
    python run_simulation.py --headless     # Run automated benchmark
    python run_simulation.py --fullscreen   # Launch in fullscreen
    python run_simulation.py --human        # Human pilot override mode
"""

import sys
import os
import time
import argparse
import numpy as np
from typing import Tuple, Dict, Optional, List

# Enable Windows High-DPI awareness
if sys.platform == "win32":
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

import pygame

# Add current workspace to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.retina import CompoundEye
from core.dynamics import SparseNeuralEngine
from core.motor import MotorDecoder
from core.leaderboard import LeaderboardManager
from circuits.optic_flow import OpticFlowCircuit
from circuits.central_complex import CentralComplex
from circuits.mushroom_body import MushroomBody
from environments.bullet_hell import BulletHellArena
from visualization.dashboard import NeuralDashboard


def run_headless_benchmark(num_steps: int = 1000):
    """
    Runs a fast headless benchmark to evaluate biological looming survival vs control.
    """
    print("=" * 70)
    print("RUNNING HEADLESS BENCHMARK: Drosophila Bio-Vision System")
    print("=" * 70)

    eye = CompoundEye(num_ommatidia_per_eye=150)
    optic_circuit = OpticFlowCircuit(eye.total_ommatidia, eye.ommatidia["azimuth"], eye.ommatidia["eye"])
    cx = CentralComplex()
    mb = MushroomBody(num_inputs=eye.total_ommatidia, num_kc=800)
    motor = MotorDecoder()
    arena = BulletHellArena(width=800, height=850)

    start_time = time.time()
    escapes_triggered = 0
    rewards_earned = 0
    punishments_taken = 0

    print(f"Executing {num_steps} closed-loop neural steps...")
    
    dt = 0.016
    for step in range(num_steps):
        arena_surf = arena.render()
        r1_r6, r7_r8 = eye.sample_arena(arena_surf, (arena.fly_x, arena.fly_y), arena.fly_heading, (arena.width, arena.height), dt=dt)
        optomotor_data = optic_circuit.update(r1_r6, r7_r8_spectral=r7_r8, dt=dt)
        if optomotor_data["looming_alert"]:
            escapes_triggered += 1

        cx_data = cx.update(arena.fly_angular_vel, optomotor_data["yaw_optomotor"], dt=dt)
        visual_features = (r1_r6 + r7_r8) * 0.5
        mb_data = mb.update(visual_features, dt=dt)

        target_az, target_dist, target_type = arena.get_closest_target_info(fly_health=arena.fly_health)
        hazard_az, hazard_dist, hazard_present = arena.get_closest_hazard_info()
        threat_info = arena.get_closest_projectile_threat()
        proj_az, proj_dist, proj_threat = threat_info
        safe_escape_angle = getattr(threat_info, "safe_escape_angle", None)

        if target_type == "nectar":
            cx.set_goal(arena.fly_heading + target_az)
        motor_action = motor.compute_action(
            optomotor_data, cx_data, mb_data,
            target_azimuth=target_az, target_dist=target_dist, target_type=target_type,
            hazard_azimuth=hazard_az, hazard_dist=hazard_dist, hazard_present=hazard_present,
            projectile_threat=proj_threat, projectile_azimuth=proj_az,
            safe_escape_angle=safe_escape_angle,
            fly_health=arena.fly_health, dt=dt
        )

        game_stats = arena.step(motor_action, dt=dt)

        if game_stats["punishment"] > 0:
            mb.deliver_punishment(game_stats["punishment"])
            punishments_taken += 1
        if game_stats["reward"] > 0:
            mb.deliver_reward(game_stats["reward"])
            rewards_earned += 1

    arena.telemetry.save(reason="BENCHMARK_COMPLETE")

    elapsed = time.time() - start_time
    fps = num_steps / elapsed
    print("\nBenchmark Results:")
    print(f"Total Steps:            {num_steps}")
    print(f"Elapsed Time:           {elapsed:.2f} s ({fps:.1f} neural updates/sec)")
    print(f"Giant Fiber Escapes:    {escapes_triggered}")
    print(f"Appetitive Rewards:     {rewards_earned} (+PAM Dopamine)")
    print(f"Aversive Collisions:    {punishments_taken} (+PPL1 Punishment)")
    print(f"Plastic Synaptic Edits: {mb.total_plastic_updates} KC->MBON weight shifts")
    print(f"Final Survival Score:   {arena.score}")
    print("=" * 70)


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))


def draw_leaderboard_modal(screen: pygame.Surface, leaderboard: LeaderboardManager, win_w: int, win_h: int, scroll_offset: int = 0) -> int:
    """
    Renders leaderboard modal comparing autonomous and human pilot records.
    Returns clamped scroll offset.
    """
    # Semi-transparent dark backdrop
    overlay = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
    overlay.fill((10, 14, 20, 240))
    screen.blit(overlay, (0, 0))

    font_main = "Segoe UI" if "segoeui" in pygame.font.get_fonts() else "Arial"
    font_mono = "Consolas" if "consolas" in pygame.font.get_fonts() else "Courier New"

    f_title = pygame.font.SysFont(font_main, 24, bold=True)
    f_sub = pygame.font.SysFont(font_main, 14)
    f_col = pygame.font.SysFont(font_main, 16, bold=True)
    f_entry = pygame.font.SysFont(font_mono, 12)
    f_foot = pygame.font.SysFont(font_main, 13)

    modal_w = min(1280, int(win_w * 0.94))
    modal_h = min(780, int(win_h * 0.90))
    mx = (win_w - modal_w) // 2
    my = (win_h - modal_h) // 2

    # Modal container
    pygame.draw.rect(screen, (18, 24, 34), (mx, my, modal_w, modal_h), border_radius=12)
    pygame.draw.rect(screen, (56, 239, 212), (mx, my, modal_w, modal_h), 2, border_radius=12)

    # Title
    t_surf = f_title.render("LEADERBOARDS & BIOLOGICAL RANKINGS", True, (56, 239, 212))
    s_surf = f_sub.render("Autonomous Fly Brain (Connectome AI) vs Human Override Pilot Records", True, (145, 160, 180))
    screen.blit(t_surf, (mx + 30, my + 20))
    screen.blit(s_surf, (mx + 30, my + 52))
    pygame.draw.line(screen, (48, 62, 82), (mx + 20, my + 78), (mx + modal_w - 20, my + 78), 1)

    # Two columns: Left = Fly Brain AI, Right = Human Pilot
    col_w = (modal_w - 70) // 2
    col1_x = mx + 25
    col2_x = mx + 35 + col_w

    # Column 1 Header: Fly Brain
    c1_title = f_col.render("AUTONOMOUS FLY BRAIN (AI)", True, (85, 255, 135))
    screen.blit(c1_title, (col1_x, my + 92))
    hdr1 = f_entry.render("#    SCORE    TIME       NECTAR         DODGES         BIOLOGICAL RANK", True, (140, 155, 175))
    screen.blit(hdr1, (col1_x, my + 120))
    pygame.draw.line(screen, (38, 50, 68), (col1_x, my + 138), (col1_x + col_w, my + 138), 1)

    # Column 2 Header: Human Pilot
    c2_title = f_col.render("HUMAN OVERRIDE PILOT", True, (255, 185, 60))
    screen.blit(c2_title, (col2_x, my + 92))
    hdr2 = f_entry.render("#    SCORE    TIME       NECTAR         DODGES         PILOT RANK", True, (140, 155, 175))
    screen.blit(hdr2, (col2_x, my + 120))
    pygame.draw.line(screen, (38, 50, 68), (col2_x, my + 138), (col2_x + col_w, my + 138), 1)

    # Vertical dividing line between columns
    pygame.draw.line(screen, (48, 62, 82), (col1_x + col_w + 5, my + 92), (col1_x + col_w + 5, my + modal_h - 55), 1)

    # Footer separator
    footer_y = my + modal_h - 55
    pygame.draw.line(screen, (48, 62, 82), (mx + 20, footer_y), (mx + modal_w - 20, footer_y), 1)

    # Load scores
    fly_scores = leaderboard.get_top_fly_scores()
    human_scores = leaderboard.get_top_human_scores()

    content_y = my + 148
    content_h = footer_y - content_y - 6
    row_h = 26
    visible_rows = max(1, content_h // row_h)

    max_entries = max(len(fly_scores), len(human_scores), 1)
    max_scroll = max(0, max_entries - visible_rows)
    scroll_offset = max(0, min(scroll_offset, max_scroll))

    # Render Fly Brain entries
    for row_idx in range(visible_rows):
        data_idx = scroll_offset + row_idx
        if data_idx >= len(fly_scores):
            break
        sc = fly_scores[data_idx]
        rank_str, rank_hex = leaderboard.get_rank_title(sc['score'])
        rank_color = hex_to_rgb(rank_hex)
        y_pos = content_y + row_idx * row_h
        stat_txt = f"#{data_idx+1:<3} {sc['score']:<8} {sc['time_survived']:>7.1f}s   {sc['nectar_count']:>5} orbs    {sc['dodges']:>5} dodges    "
        row_color = (245, 248, 252) if data_idx > 0 else (56, 239, 212)
        r_stat = f_entry.render(stat_txt, True, row_color)
        r_rank = f_entry.render(rank_str, True, rank_color)
        screen.blit(r_stat, (col1_x, y_pos))
        screen.blit(r_rank, (col1_x + r_stat.get_width(), y_pos))

    # Render Human Pilot entries
    for row_idx in range(visible_rows):
        data_idx = scroll_offset + row_idx
        if data_idx >= len(human_scores):
            break
        sc = human_scores[data_idx]
        rank_str, rank_hex = leaderboard.get_rank_title(sc['score'])
        rank_color = hex_to_rgb(rank_hex)
        y_pos = content_y + row_idx * row_h
        stat_txt = f"#{data_idx+1:<3} {sc['score']:<8} {sc['time_survived']:>7.1f}s   {sc['nectar_count']:>5} orbs    {sc['dodges']:>5} dodges    "
        row_color = (245, 248, 252) if data_idx > 0 else (255, 185, 60)
        r_stat = f_entry.render(stat_txt, True, row_color)
        r_rank = f_entry.render(rank_str, True, rank_color)
        screen.blit(r_stat, (col2_x, y_pos))
        screen.blit(r_rank, (col2_x + r_stat.get_width(), y_pos))

    # Render interactive scrollbars if content exceeds view
    if len(fly_scores) > visible_rows:
        sb_x = col1_x + col_w - 6
        pygame.draw.rect(screen, (24, 32, 44), (sb_x, content_y, 4, content_h), border_radius=2)
        thumb_h = max(24, int(content_h * (visible_rows / len(fly_scores))))
        scroll_ratio = scroll_offset / max(1, len(fly_scores) - visible_rows)
        thumb_y = content_y + int((content_h - thumb_h) * min(1.0, scroll_ratio))
        pygame.draw.rect(screen, (56, 239, 212), (sb_x, thumb_y, 4, thumb_h), border_radius=2)

    if len(human_scores) > visible_rows:
        sb_x = col2_x + col_w - 6
        pygame.draw.rect(screen, (24, 32, 44), (sb_x, content_y, 4, content_h), border_radius=2)
        thumb_h = max(24, int(content_h * (visible_rows / len(human_scores))))
        scroll_ratio = scroll_offset / max(1, len(human_scores) - visible_rows)
        thumb_y = content_y + int((content_h - thumb_h) * min(1.0, scroll_ratio))
        pygame.draw.rect(screen, (255, 185, 60), (sb_x, thumb_y, 4, thumb_h), border_radius=2)

    # Footer instructions & records
    showing_end = min(scroll_offset + visible_rows, max_entries)
    scroll_info = f"Showing #{scroll_offset+1}-#{showing_end} of {max_entries} records | Scroll with [Mouse Wheel / Up/Down]"
    foot_txt = f_foot.render(f"[ESC/L] Close | [H] Pilot Mode | [R] Reset | {scroll_info}", True, (145, 160, 180))
    hs_txt = f_foot.render(f"All-Time High Score: Fly Brain {leaderboard.get_fly_high_score()} vs Human {leaderboard.get_human_high_score()}", True, (56, 239, 212))
    screen.blit(foot_txt, (mx + 25, footer_y + 16))
    screen.blit(hs_txt, (mx + modal_w - hs_txt.get_width() - 25, footer_y + 16))

    return scroll_offset


def run_gui(human_mode: bool = False, start_fullscreen: bool = False):
    """
    Launches the live interactive window with direct native rendering, zero blur, and Leaderboard tracking.
    """
    pygame.init()
    pygame.display.set_caption("Drosophila Bio-Vision & Connectome Game System")

    DEFAULT_W = 1500
    DEFAULT_H = 850
    BASE_ARENA_W = 810
    BASE_ARENA_H = 850

    is_fullscreen = start_fullscreen
    if is_fullscreen:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode((DEFAULT_W, DEFAULT_H), pygame.RESIZABLE)

    cur_w, cur_h = screen.get_size()
    arena_ratio = BASE_ARENA_W / BASE_ARENA_H
    arena_w = int(cur_h * arena_ratio)
    hud_w = cur_w - arena_w
    if hud_w < 640 and cur_w > 1000:
        hud_w = min(800, max(640, int(cur_w * 0.44)))
        arena_w = cur_w - hud_w

    clock = pygame.time.Clock()

    print("Initializing Drosophila Bio-Vision & Neural Circuits...")
    eye = CompoundEye(num_ommatidia_per_eye=375)
    
    optic_circuit = OpticFlowCircuit(eye.total_ommatidia, eye.ommatidia["azimuth"], eye.ommatidia["eye"])
    cx = CentralComplex()
    mb = MushroomBody(num_inputs=eye.total_ommatidia, num_kc=1500, learning_rate=0.18)
    motor = MotorDecoder()
    arena = BulletHellArena(width=BASE_ARENA_W, height=BASE_ARENA_H)
    hud = NeuralDashboard(hud_width=hud_w, hud_height=cur_h)
    leaderboard = LeaderboardManager()
    
    retina_layout = eye.get_layout_2d(canvas_width=hud_w - 64, canvas_height=hud.p1_h - 48, margin=14)

    print("System Ready! Running 60 FPS Closed-Loop Simulation...")
    print("Shortcuts:")
    print("  [L]: Toggle Leaderboards & Rankings")
    print("  [F] / [F11]: Toggle Fullscreen")
    print("  [H]: Toggle Human Pilot / Fly Brain Pilot")
    print("  [R]: Reset Game Arena")
    print("  [D]: Trigger Test Dopamine Pulse")
    print("  [SPACE]: Pause / Resume")
    print("  [ESC]: Quit")

    running = True
    paused = False
    show_leaderboard = False
    leaderboard_scroll = 0
    is_human = human_mode
    
    font_main = "Segoe UI" if "segoeui" in pygame.font.get_fonts() else "Arial"
    font_hud_title = pygame.font.SysFont(font_main, 15, bold=True)
    font_hud_sub = pygame.font.SysFont(font_main, 13)

    def handle_resize(new_w, new_h):
        nonlocal arena_w, hud_w, retina_layout, cur_w, cur_h
        cur_w, cur_h = new_w, new_h
        arena_ratio = BASE_ARENA_W / BASE_ARENA_H
        arena_w = int(new_h * arena_ratio)
        hud_w = new_w - arena_w
        if hud_w < 640 and new_w > 1000:
            hud_w = min(800, max(640, int(new_w * 0.44)))
            arena_w = new_w - hud_w
        hud.set_dimensions(hud_w, new_h)
        retina_layout = eye.get_layout_2d(canvas_width=hud_w - 64, canvas_height=hud.p1_h - 48, margin=14)

    handle_resize(cur_w, cur_h)

    while running:
        dt = clock.tick(60) / 1000.0
        dt = min(dt, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                if not is_fullscreen:
                    screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    handle_resize(event.w, event.h)
            elif event.type == pygame.MOUSEWHEEL:
                if show_leaderboard:
                    # event.y > 0 scrolls up (earlier records), event.y < 0 scrolls down
                    leaderboard_scroll = max(0, leaderboard_scroll - event.y * 2)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if show_leaderboard:
                    if event.button == 4:  # Wheel up
                        leaderboard_scroll = max(0, leaderboard_scroll - 2)
                    elif event.button == 5:  # Wheel down
                        leaderboard_scroll += 2
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if show_leaderboard:
                        show_leaderboard = False
                    elif is_fullscreen:
                        is_fullscreen = False
                        screen = pygame.display.set_mode((DEFAULT_W, DEFAULT_H), pygame.RESIZABLE)
                        handle_resize(DEFAULT_W, DEFAULT_H)
                    else:
                        running = False
                elif event.key == pygame.K_l:
                    show_leaderboard = not show_leaderboard
                    if show_leaderboard:
                        leaderboard_scroll = 0
                elif show_leaderboard and event.key in (pygame.K_UP, pygame.K_w):
                    leaderboard_scroll = max(0, leaderboard_scroll - 1)
                elif show_leaderboard and event.key in (pygame.K_DOWN, pygame.K_s):
                    leaderboard_scroll += 1
                elif show_leaderboard and event.key == pygame.K_PAGEUP:
                    leaderboard_scroll = max(0, leaderboard_scroll - 6)
                elif show_leaderboard and event.key == pygame.K_PAGEDOWN:
                    leaderboard_scroll += 6
                elif show_leaderboard and event.key == pygame.K_HOME:
                    leaderboard_scroll = 0
                elif event.key in (pygame.K_f, pygame.K_F11):
                    is_fullscreen = not is_fullscreen
                    if is_fullscreen:
                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                    else:
                        screen = pygame.display.set_mode((DEFAULT_W, DEFAULT_H), pygame.RESIZABLE)
                    w, h = screen.get_size()
                    handle_resize(w, h)
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    # Save run if game in progress to whichever pilot mode was active most
                    if arena.score > 200:
                        record_as_human = arena.time_human > arena.time_bot
                        leaderboard.add_score(record_as_human, arena.score, arena.time_elapsed, arena.nectar_collected, arena.projectiles_dodged)
                    arena = BulletHellArena(width=BASE_ARENA_W, height=BASE_ARENA_H)
                elif event.key == pygame.K_h:
                    is_human = not is_human
                    arena.log_event(f"PILOT CHANGED: {'HUMAN OVERRIDE' if is_human else 'AUTONOMOUS FLY BRAIN'}", (255, 205, 85))
                elif event.key == pygame.K_d:
                    if mb.net_valence < 0:
                        mb.deliver_reward(2.5)
                        arena.log_event("MANUAL PAM REWARD TRIGGERED (+Dopamine)", (50, 240, 200))
                    else:
                        mb.deliver_punishment(2.5)
                        arena.log_event("MANUAL PPL1 SHOCK TRIGGERED (-Dopamine)", (255, 60, 60))

        # Record completed run to leaderboard upon death
        if arena.last_run_summary:
            summary = arena.last_run_summary
            record_as_human = summary.get("is_human", False)
            is_top = leaderboard.add_score(
                is_human=record_as_human,
                score=summary["score"],
                time_survived=summary["time_survived"],
                nectar_count=summary["nectar_count"],
                dodges=summary["dodges"]
            )
            arena.last_run_summary = None
            if is_top:
                pilot_label = "Human Override" if record_as_human else "Fly Brain AI"
                color = (255, 185, 60) if record_as_human else (56, 239, 212)
                arena.log_event(f"New High Score: {summary['score']} ({leaderboard.get_rank_title(summary['score'])[0]})", color)

        if not paused:
            # Render arena
            arena_surf = arena.render()

            # Sample visual inputs
            r1_r6, r7_r8 = eye.sample_arena(
                arena_surf, (arena.fly_x, arena.fly_y), arena.fly_heading, (arena.width, arena.height), dt=dt
            )

            # Optic flow and looming detection
            optomotor_data = optic_circuit.update(r1_r6, r7_r8_spectral=r7_r8, dt=dt)

            # Central complex heading integration
            cx_data = cx.update(arena.fly_angular_vel, optomotor_data["yaw_optomotor"], dt=dt)

            # Mushroom body learning
            visual_features = (r1_r6 * 0.5) + (r7_r8 * 0.5)
            mb_data = mb.update(visual_features, dt=dt)

            # Motor decision
            target_az, target_dist, target_type = arena.get_closest_target_info(fly_health=arena.fly_health)
            hazard_az, hazard_dist, hazard_present = arena.get_closest_hazard_info()
            threat_info = arena.get_closest_projectile_threat()
            proj_az, proj_dist, proj_threat = threat_info
            safe_escape_angle = getattr(threat_info, "safe_escape_angle", None)

            if target_type == "nectar":
                cx.set_goal(cx.current_heading_estimate + target_az)

            if is_human:
                keys = pygame.key.get_pressed()
                yaw = 0.0
                thrust = 220.0
                if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    yaw = -6.5
                elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    yaw = +6.5
                if keys[pygame.K_UP] or keys[pygame.K_w]:
                    thrust = 350.0
                elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                    thrust = 70.0
                motor_action = {
                    "yaw_rate": yaw,
                    "thrust": thrust,
                    "saccade_active": False,
                    "learned_component": 0.0,
                    "optomotor_component": 0.0
                }
            else:
                motor_action = motor.compute_action(
                    optomotor_data, cx_data, mb_data,
                    target_azimuth=target_az, target_dist=target_dist, target_type=target_type,
                    hazard_azimuth=hazard_az, hazard_dist=hazard_dist, hazard_present=hazard_present,
                    projectile_threat=proj_threat, projectile_azimuth=proj_az,
                    safe_escape_angle=safe_escape_angle,
                    fly_health=arena.fly_health, dt=dt
                )

            # Step environment physics
            game_stats = arena.step(motor_action, dt=dt, is_human=is_human)

            # Reinforcement delivery
            if game_stats["punishment"] > 0:
                mb.deliver_punishment(game_stats["punishment"])
            if game_stats["reward"] > 0:
                mb.deliver_reward(game_stats["reward"])

        # Render to screen
        raw_arena_surf = arena.render()
        if arena_w == BASE_ARENA_W and cur_h == BASE_ARENA_H:
            screen.blit(raw_arena_surf, (0, 0))
        else:
            scaled_arena = pygame.transform.smoothscale(raw_arena_surf, (arena_w, cur_h))
            screen.blit(scaled_arena, (0, 0))

        # Status overlay
        hp = arena.fly_health
        hp_color = (65, 245, 130) if hp > 50 else ((255, 185, 45) if hp > 25 else (255, 60, 60))
        
        # Health bar
        bar_w = min(220, int(arena_w * 0.28))
        pygame.draw.rect(screen, (28, 34, 46), (24, 18, bar_w, 18), border_radius=4)
        hp_fill_w = int((hp / 100.0) * bar_w)
        if hp_fill_w > 0:
            pygame.draw.rect(screen, hp_color, (24, 18, hp_fill_w, 18), border_radius=4)
        pygame.draw.rect(screen, (180, 200, 220), (24, 18, bar_w, 18), 1, border_radius=4)
        
        # Stacked status labels
        lbl_hp = font_hud_title.render(f"HEALTH: {int(hp)}%", True, (255, 255, 255))
        
        current_rank, rank_color = leaderboard.get_rank_title(arena.score)
        hs = leaderboard.get_human_high_score() if is_human else leaderboard.get_fly_high_score()
        lbl_score = font_hud_title.render(f"SCORE: {arena.score}  [HIGH: {hs}]", True, (56, 239, 212))
        lbl_rank = font_hud_sub.render(f"RANK: {current_rank}", True, (255, 215, 80))
        
        pilot_mode = "PILOT: HUMAN OVERRIDE [H]" if is_human else "PILOT: AUTONOMOUS FLY BRAIN [H]"
        lbl_pilot = font_hud_sub.render(pilot_mode, True, (255, 205, 85) if is_human else (85, 245, 165))

        screen.blit(lbl_hp, (24 + bar_w + 14, 17))
        screen.blit(lbl_score, (24, 44))
        screen.blit(lbl_rank, (24, 68))
        screen.blit(lbl_pilot, (24, 92))

        # Control badges
        lbl_lb = font_hud_sub.render("[L] Leaderboard", True, (56, 239, 212))
        lbl_fs = font_hud_sub.render("[F] Fullscreen", True, (140, 155, 175))
        screen.blit(lbl_lb, (arena_w - 230, 18))
        screen.blit(lbl_fs, (arena_w - 110, 18))

        # Event log notifications
        y_ev = cur_h - 130
        for ev in arena.recent_events:
            ev_surf = font_hud_sub.render(f">> {ev['text']}", True, ev["color"])
            screen.blit(ev_surf, (24, y_ev))
            y_ev += 22

        # Draw HUD dashboard
        hud_surface = hud.render(
            retina_layout=retina_layout,
            retina_rgb=eye.raw_rgb,
            optomotor_data=optomotor_data,
            cx_data=cx_data,
            mb_data=mb_data,
            motor_data=motor_action,
            game_stats=game_stats
        )
        screen.blit(hud_surface, (arena_w, 0))

        # Vertical dividing line
        pygame.draw.line(screen, (48, 62, 82), (arena_w, 0), (arena_w, cur_h), 2)

        # Leaderboard modal overlay
        if show_leaderboard:
            leaderboard_scroll = draw_leaderboard_modal(screen, leaderboard, cur_w, cur_h, scroll_offset=leaderboard_scroll)

        pygame.display.flip()

    if arena.score > 200 and not arena.game_over_recorded:
        record_as_human = arena.time_human > arena.time_bot
        leaderboard.add_score(record_as_human, arena.score, arena.time_elapsed, arena.nectar_collected, arena.projectiles_dodged)
    arena.telemetry.save(reason="USER_EXIT")
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drosophila Connectome & Bio-Vision Game System")
    parser.add_argument("--headless", action="store_true", help="Run fast headless benchmark")
    parser.add_argument("--steps", type=int, default=1500, help="Steps for headless benchmark")
    parser.add_argument("--fullscreen", action="store_true", help="Start directly in fullscreen mode")
    parser.add_argument("--human", action="store_true", help="Start in human pilot mode")
    args = parser.parse_args()

    if args.headless:
        run_headless_benchmark(num_steps=args.steps)
    else:
        run_gui(human_mode=args.human, start_fullscreen=args.fullscreen)
