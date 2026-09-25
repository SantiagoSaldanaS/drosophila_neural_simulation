# Drosophila Bio-Vision & Neural Navigation Simulation

A real-time, closed-loop simulation of *Drosophila melanogaster* visual processing and motor control in an interactive 2D dynamic environment.

The system couples biophysically grounded neural circuits with a high-speed physics arena, allowing an autonomous simulated fly to navigate, avoid dynamic projectile hazards, bypass static obstacles, and forage for resources.

---

## Architecture Overview

The simulation integrates several core insect neuroscience circuits into a closed loop:

- **Panoramic Compound Eye (`core/retina.py`)**:
  Simulates 750 hexagonal ommatidia distributed across left and right eyes (270 deg field of view). Samples arena luminance and color channels at sub-pixel resolution.

- **Optic Flow & Looming Detection (`circuits/optic_flow.py`)**:
  - **Elementary Motion Detectors (EMD)**: Hassenstein-Reichardt correlation detectors computing directional wide-field visual flow for optomotor yaw stabilization.
  - **Looming Detection (LPLC2 / Giant Fiber)**: Detects rapid radial visual expansion with sub-30ms thresholding to trigger emergency escape saccades.

- **Central Complex Ring Attractor (`circuits/central_complex.py`)**:
  Models ellipsoid body E-PG compass neurons. Integrates angular velocity and visual landmark cues to maintain an internal heading representation.

- **Mushroom Body Plasticity (`circuits/mushroom_body.py`)**:
  Simulates sparse Kenyon cell encoding (400 KCs) and Mushroom Body Output Neurons (MBONs). Features three-factor dopaminergic synaptic plasticity:
  - **PAM cluster**: Reinforces appetitive approach behavior upon collecting nectar.
  - **PPL1 cluster**: Depresses synaptic weights and drives avoidance behavior upon collision damage.

- **Motor Decoding & Continuous Collision Detection (`core/motor.py`)**:
  Blends optomotor stabilization, looming escape vectors, ring attractor navigation, and predictive swept-circle raycasting to generate smooth thrust and yaw torque.

- **Telemetry & Leaderboard (`core/telemetry_logger.py`, `core/leaderboard.py`)**:
  Logs per-tick spatial telemetry (positions, velocities, distances to nearest threats) and maintains scrollable records partitioned by control mode (Neural Autopilot vs. Manual Human Flight).

---

## Installation

Ensure Python 3.10+ is installed, then clone the repository and install the dependencies:

```bash
git clone <repo-url>
cd Fly
pip install -r requirements.txt
```

---

## Usage

### Interactive GUI Mode

Run the main simulation with real-time neural visualization:

```bash
python run_simulation.py
```

### Headless Benchmark Mode

Run without opening a window for rapid testing or high-speed execution:

```bash
python run_simulation.py --headless --steps 1000
```

### Running Unit Tests

Run test suites covering circuit dynamics, retina geometry, and synaptic plasticity:

```bash
pytest tests/
```

---

## Controls

| Key / Input | Action |
| :--- | :--- |
| **Space** | Toggle between Autonomous Neural Fly and Manual Human Pilot |
| **A / D** or **Left / Right** | Steer yaw left / right (Manual mode) |
| **W / S** or **Up / Down** | Accelerate forward / reverse (Manual mode) |
| **F** | Toggle Fullscreen |
| **L** | Toggle Leaderboard view |
| **Mouse Wheel / Up / Down** | Scroll through Leaderboard entries (when Leaderboard is active) |
| **Esc** | Quit simulation |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
