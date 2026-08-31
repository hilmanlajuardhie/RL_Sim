# Reinforcement Learning Control Benchmarks

A hands-on implementation of Reinforcement Learning (RL) control benchmarks using **Gymnasium**, **Stable-Baselines3**, and **MuJoCo**[cite: 7, 8, 9]. This project demonstrates policy learning for discrete and continuous control tasks, complete with automated performance evaluations and environment diagnostics[cite: 7, 8, 9].

---

![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20FP16-ee4c2c?logo=pytorch)
![YOLO11](https://img.shields.io/badge/YOLO-v11m-00FFFF)
![OpenCV](https://img.shields.io/badge/OpenCV-GStreamer%20Backend-5C3EE8?logo=opencv)
![GStreamer](https://img.shields.io/badge/GStreamer-RTP%2FUDP-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Status](#status)
- [Contributing](#donate)
- [License](#license)

## Features

* **Discrete Control (PPO):** Trains a Proximal Policy Optimization (PPO) agent on the `CartPole-v1` environment for 20,000 timesteps, evaluating score improvements before and after training.
* **Continuous Control & Live Visual Training (SAC):** Evaluates a Soft Actor-Critic (SAC) agent on `InvertedPendulum-v5` (MuJoCo dynamic physics engine) across a 3-stage lifecycle: Untrained baseline, Live visual training (1,000 timesteps), and Trained policy execution.
* **Stack Diagnostic Tool:** Automated system health check to verify Python dependencies, PyTorch CUDA hardware acceleration, MuJoCo C++ MJCF parsing, offscreen rendering backends (`MUJOCO_GL`), and deployment libraries.

---

### Tech Stack

| Command | Description |
| --- | --- |
| Language | Python 3.12 |
| Computer Vision | OpenCV (compiled with **GStreamer & GTK backends**) |
| Deep Learning | PyTorch, Ultralytics YOLO v11 (yolo11m) |
| Streaming Framework | GStreamer 1.0 (RTP/JPEG over UDP) |
| Target OS | Linux (Ubuntu / Debian environment) |

## Project Structure

```text
.
├── RL_ChartPole.py          # PPO algorithm benchmark on CartPole-v1
├── RL_InvertedPendulum.py    # SAC algorithm benchmark on InvertedPendulum-v5 (MuJoCo)
└── Sys_Check.py             # Environment & hardware diagnostic script
```

## Installation

```bash
git clone https://github.com/PAPA/remote_vision:-real-time-edge-vision-&-streaming-pipeline.git
cd remote_vision:-real-time-edge-vision-&-streaming-pipeline
```

## Usage

Activate the Virtual Environtment (.venv)
```bash
Source .venv/bin/Activate
```

Run the System Check
```bash
python3 Sys_Check.py
```

Run the program in **Terminal 1**
```bash

```

Run the program in **Terminal 2**
```bash

```

## Status

- 
- 
- 

## Donate

If you find this project useful for your work or research, consider supporting its ongoing development!

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/hilmanlajuardhie)

*Every coffee or donation helps keep experimental hardware and vision projects going. Thank you!*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Made with ❤ by [JIAR](https://github.com/hilmanlajuardhie)