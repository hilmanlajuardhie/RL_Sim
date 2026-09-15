# Reinforcement Learning: Development, Simulation, Deployment

A hands-on implementation of Reinforcement Learning (RL) control benchmarks using **Gymnasium**, **Stable-Baselines3**, and **MuJoCo**. This project demonstrates policy learning for discrete and continuous control tasks, complete with automated performance evaluations and environment diagnostics.

---

![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?logo=ubuntu&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20FP32-ee4c2c?logo=pytorch)
![MuJoCo](https://img.shields.io/badge/MuJoCo-Physics-black)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-v1.30.0-005CED?logo=onnx&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## Table of Contents

- [Domain](#domain)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Status](#status)
- [Contributing](#donate)
- [License](#license)

## Domain

* **Discrete Control (PPO):** Trains a Proximal Policy Optimization (PPO) agent on the `CartPole-v1` environment for 20,000 timesteps, evaluating score improvements before and after training.
* **Continuous Control & Live Visual Training (SAC):** Evaluates a Soft Actor-Critic (SAC) agent on `InvertedPendulum-v5` (MuJoCo dynamic physics engine) across a 3-stage lifecycle: Untrained baseline, Live visual training (1,000 timesteps), and Trained policy execution.
* **Stack Diagnostic Tool:** Automated system health check to verify Python dependencies, PyTorch CUDA hardware acceleration, MuJoCo C++ MJCF parsing, offscreen rendering backends (`MUJOCO_GL`), and deployment libraries.

---

### Tech Stack

| Package | Version | Description |
| :--- | :--- | :--- |
| **Python** | v3.12 | Core programming language environment |
| **PyTorch** | v2.13.0+cu126 | Deep learning tensor library (CUDA enabled) |
| **Pygame** | v2.5.8 | 2D rendering and window management |
| **Stable-Baselines3**| v2.9.0 | RL algorithm implementations |
| **Gymnasium** | v1.3.0 | RL environment API |
| **MuJoCo Physics** | v3.13.0 | Advanced physics simulation engine |
| **ONNX Runtime** | v1.30.0 | Cross-platform machine learning inference |

---

## Project Structure

```text
.
├── RL_ChartPole.py         # PPO algorithm benchmark on CartPole-v1
├── RL_Humanoid.py          # PPO algorithm benchmark on Humanoid-v5 (MuJoCo)
├── RL_InvertedPendulum.py  # SAC algorithm benchmark on InvertedPendulum-v5 (MuJoCo)
└── Sys_Check.py            # Environment & hardware diagnostic script
```

---

## Installation

```bash
git clone https://github.com/hilmanlajuardhie/RL_Sim.git
cd RL_Sim/
```

## Usage

Create and activate the virtual environtment (.venv)
```bash
python3 -m venv .venv
Source .venv/bin/Activate
```

Run the System Check
```bash
python3 src/Sys_Check.py
```

---

## Status

- Creating mechanical model for Mujoco
- Developing RL based on PPO algorithm
- Deploy global format model on .onnx

## Donate

If you find this project useful for your work or research, consider supporting its ongoing development!

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/hilmanlajuardhie)

*Every coffee or donation helps keep experimental hardware and vision projects going. Thank you!*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Made with ❤ by [JIAR](https://github.com/hilmanlajuardhie)