"""Unit tests for Drosophila neural circuits and learning models."""

import math
import numpy as np
import pytest
import pygame

from core.retina import CompoundEye
from core.dynamics import SparseNeuralEngine
from core.motor import MotorDecoder
from core.connectome import ConnectomeGraph
from circuits.optic_flow import OpticFlowCircuit
from circuits.central_complex import CentralComplex
from circuits.mushroom_body import MushroomBody


def test_retina_ommatidia_dimensions():
    """Verify compound eye creates correct panoramic ommatidia geometry."""
    eye = CompoundEye(num_ommatidia_per_eye=375)
    assert eye.total_ommatidia == 750
    assert len(eye.ommatidia["azimuth"]) == 750
    assert len(eye.ommatidia["elevation"]) == 750
    assert len(eye.ommatidia["eye"]) == 750

    # Half left eye (0), half right eye (1)
    assert np.sum(eye.ommatidia["eye"] == 0) == 375
    assert np.sum(eye.ommatidia["eye"] == 1) == 375


def test_retina_sampling():
    """Verify arena sampling extracts valid luminance and spectral channels."""
    pygame.init()
    surf = pygame.Surface((300, 300))
    surf.fill((200, 100, 50))  # test color

    eye = CompoundEye(num_ommatidia_per_eye=100)
    r1_r6, r7_r8 = eye.sample_arena(surf, (150, 150), 0.0, (300, 300), dt=0.016)

    assert r1_r6.shape == (200,)
    assert r7_r8.shape == (200,)
    assert np.all(r1_r6 >= 0.0) and np.all(r1_r6 <= 1.0)
    assert np.all(r7_r8 >= 0.0) and np.all(r7_r8 <= 1.0)


def test_looming_threat_detection():
    """Verify that rapid visual expansion triggers the Giant Fiber looming alert."""
    eye = CompoundEye(num_ommatidia_per_eye=100)
    circuit = OpticFlowCircuit(eye.total_ommatidia, eye.ommatidia["azimuth"], eye.ommatidia["eye"])

    steady = np.full(eye.total_ommatidia, 0.7, dtype=np.float32)
    res1 = circuit.update(steady, dt=0.016)
    assert not res1["looming_alert"]

    # Sudden rapid dark expansion on the left eye (approaching projectile)
    expanding_threat = steady.copy()
    left_indices = np.where(eye.ommatidia["eye"] == 0)[0]
    expanding_threat[left_indices[:40]] = 0.05

    res2 = circuit.update(expanding_threat, dt=0.016)
    assert res2["looming_alert"], "Looming circuit failed to trigger emergency alert on rapid expansion!"
    assert res2["escape_bias"] == +1.0, "Escape saccade should veer away from threat on the left"


def test_mushroom_body_plasticity():
    """Verify that behavioral valence updates via 3-factor dopaminergic plasticity."""
    mb = MushroomBody(num_inputs=50, num_kc=400, learning_rate=0.15)
    
    np.random.seed(99)
    pattern_trap = np.random.uniform(0.1, 0.9, size=50).astype(np.float32)

    # Initial state: Neutral valence
    res_initial = mb.update(pattern_trap, dt=0.016)
    initial_valence = res_initial["net_valence"]

    # Aversive punishment (PPL1 dopamine signal)
    mb.deliver_punishment(intensity=3.0)

    # Re-evaluate response to the pattern after punishment
    res_after_punishment = mb.update(pattern_trap, dt=0.016)
    punished_valence = res_after_punishment["net_valence"]

    assert punished_valence < initial_valence, (
        f"Plasticity update failed: Initial={initial_valence:.3f}, After={punished_valence:.3f}"
    )

    # Appetitive reward (PAM dopamine signal)
    mb.deliver_reward(intensity=5.0)
    res_after_reward = mb.update(pattern_trap, dt=0.016)
    rewarded_valence = res_after_reward["net_valence"]

    assert rewarded_valence > punished_valence, (
        f"Plasticity update failed: Punished={punished_valence:.3f}, Rewarded={rewarded_valence:.3f}"
    )


def test_central_complex_compass_integration():
    """Verify that E-PG compass bump shifts dynamically with yaw rotation."""
    cx = CentralComplex(num_wedges=16)
    initial_heading = cx.current_heading_estimate

    # Rotate with positive angular velocity for 10 timesteps
    dt = 0.05
    for _ in range(10):
        cx.update(angular_velocity=2.0, visual_flow_yaw=0.0, dt=dt)

    assert cx.current_heading_estimate > initial_heading, "Central Complex failed to integrate angular velocity!"
    assert 0 <= cx.epg_activity.shape[0] == 16


def test_sparse_neural_engine_dynamics():
    """Verify SparseNeuralEngine simulates graded and LIF neurons with PyTorch sparse tensors."""
    engine = SparseNeuralEngine(num_neurons=50, tau_m=0.02)
    
    # Connect neuron 0 to neuron 1
    src = np.array([0])
    tgt = np.array([1])
    weights = np.array([2.5])
    engine.set_connectivity(src, tgt, weights)

    # Step with current into neuron 0
    ext_i = np.zeros(50, dtype=np.float32)
    ext_i[0] = 3.0

    for _ in range(20):
        v, act = engine.step(ext_i, dt=0.005)

    # Neuron 0 and target neuron 1 should both have depolarized
    assert v[0] > 0.5
    assert v[1] > 0.1
