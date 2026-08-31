import importlib
import os
import sys

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def print_status(status, message):
    if status == "OK":
        print(f"[{GREEN} OK {RESET}] {message}")
    elif status == "FAIL":
        print(f"[{RED}FAIL{RESET}] {message}")
    elif status == "WARN":
        print(f"[{YELLOW}WARN{RESET}] {message}")


def check_robotics_dependencies():
    print("========================================")
    print("   Robotics RL Stack Diagnostic Check   ")
    print("========================================\n")

    # 1. Python Version
    print("--- 1. Python Environment ---")
    py_ver = sys.version_info
    py_str = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    if py_ver >= (3, 10):
        print_status("OK", f"Python version is {py_str}.")
    else:
        print_status("WARN", f"Python version is {py_str} (3.10+ recommended).")

    # 2. Python Packages
    print("\n--- 2. Core Packages ---")
    packages = {
        "torch": "PyTorch",
        "gymnasium": "Gymnasium",
        "stable_baselines3": "Stable-Baselines3",
        "mujoco": "MuJoCo Physics",
        "onnxruntime": "ONNX Runtime (Deployment)",
        "pygame": "Pygame (2D Rendering)",
        "matplotlib": "Matplotlib (Plotting)",
        "numpy": "NumPy",
    }

    installed = {}
    for module, name in packages.items():
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "Installed")
            print_status("OK", f"{name} (v{version})")
            installed[module] = True
        except ImportError:
            print_status("FAIL", f"{name} is missing! (pip install {module})")
            installed[module] = False

    # 3. Hardware & CUDA
    print("\n--- 3. Hardware Acceleration ---")
    if installed.get("torch"):
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print_status("OK", f"PyTorch CUDA active. GPU: {gpu_name}")
        else:
            print_status(
                "WARN", "CUDA unavailable. Training will execute on CPU."
            )

    # 4. Low-Level MuJoCo Engine & GL Check
    print("\n--- 4. MuJoCo Dynamic Engine & Rendering ---")
    if installed.get("mujoco"):
        import mujoco

        # Check C++ MJCF Parser
        test_xml = '<mujoco><worldbody><body name="b"><freejoint/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
        try:
            m = mujoco.MjModel.from_xml_string(test_xml)
            d = mujoco.MjData(m)
            mujoco.mj_step(m, d)
            print_status("OK", "MuJoCo C++ engine & MJCF string parser active.")
        except Exception as e:
            print_status("FAIL", f"MuJoCo engine failure: {e}")

        # Check Offscreen Rendering Engine
        try:
            renderer = mujoco.Renderer(m, height=240, width=320)
            renderer.update_scene(d)
            _ = renderer.render()
            gl_backend = os.environ.get("MUJOCO_GL", "default")
            print_status(
                "OK", f"MuJoCo Offscreen Renderer functional (Backend: {gl_backend})."
            )
            renderer.close()
        except Exception as e:
            print_status(
                "WARN",
                f"MuJoCo rendering warning: {e}. Try setting 'export MUJOCO_GL=egl' or 'glfw'.",
            )

    # 5. Continuous Gymnasium & PPO Framework
    print("\n--- 5. SB3 & Gym Integration ---")
    if installed.get("gymnasium") and installed.get("stable_baselines3"):
        import gymnasium as gym
        from stable_baselines3 import SAC

        try:
            env = gym.make("InvertedPendulum-v5")
            model = SAC("MlpPolicy", env, device="cpu", verbose=0)
            print_status(
                "OK",
                "Continuous control (SAC + InvertedPendulum-v5) initialized.",
            )
            env.close()
        except Exception as e:
            print_status(
                "FAIL", f"Continuous environment initialization failed: {e}"
            )

    print("\n========================================")
    print("Diagnosis Complete.")


if __name__ == "__main__":
    check_robotics_dependencies()