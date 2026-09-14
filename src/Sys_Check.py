import os
import sys
import importlib

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
    print("----- 1. Python Environment -----")
    py_ver = sys.version_info
    py_str = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    if py_ver >= (3, 10):
        print_status("OK", f"Python version is {py_str}.")
    else:
        print_status("WARN", f"Python version is {py_str} (3.10+ recommended).")

    # 2. Python Packages
    print("\n----- 2. Core Packages -----")
    packages = {
        "numpy": "NumPy\t\t",
        "matplotlib": "Matplotlib\t",
        "torch": "PyTorch\t\t",
        "pygame": "Pygame\t\t",
        "gymnasium": "Gymnasium\t",
        "stable_baselines3": "Stable-Baselines3",
        "mujoco": "MuJoCo Physics\t",
        "onnxruntime": "ONNX Runtime\t",
    }
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

    installed = {}
    for module, name in packages.items():
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "Installed")
            print_status("OK", f"{name}: v{version}")
            installed[module] = True
        except ImportError:
            print_status("FAIL", f"{name} is missing! (pip install {module})")
            installed[module] = False

    # 3. Hardware & CUDA
    print("\n----- 3. Hardware Acceleration -----")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print_status("OK", f"PyTorch CUDA active.\n       GPU: {gpu_name}")
        else:
            print_status("WARN", "PyTorch CUDA is NOT available. YOLO inference will run on CPU.")
    except ImportError:
        print_status("FAIL", "Skipping CUDA check because PyTorch is not installed.")

    # 4. Continuous Gymnasium & PPO Framework
    print("\n----- 8. SB3 & Gym Integration -----")
    if installed.get("gymnasium") and installed.get("stable_baselines3"):
        import gymnasium as gym
        from stable_baselines3 import PPO

        try:
            env_name = "Humanoid-v5"
            env = gym.make(env_name)
            model = PPO("MlpPolicy", env, device="cpu", verbose=0)
            print_status(
                "OK",
                f"Environment initialized: {env_name} + {model.__class__.__name__}.",
            )
            env.close()
        except Exception as e:
            print_status(
                "FAIL", f"environment initialization failed: {e}"
            )

    # 5. Low-Level MuJoCo Engine & GL Check
    print("\n----- 5. MuJoCo Dynamic Engine & Rendering -----")
    if installed.get("mujoco"):
        import mujoco

        # Check C++ MJCF Parser
        test_xml = '<mujoco><worldbody><body name="b"><freejoint/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
        try:
            m = mujoco.MjModel.from_xml_string(test_xml)
            d = mujoco.MjData(m)
            mujoco.mj_step(m, d)
            print_status("OK", "C++ engine & MJCF active.")
        except Exception as e:
            print_status("FAIL", f"MuJoCo engine failure: {e}")

        # Check Offscreen Rendering Engine
        try:
            renderer = mujoco.Renderer(m, height=240, width=320)
            renderer.update_scene(d)
            _ = renderer.render()
            gl_backend = os.environ.get("MUJOCO_GL", "default")
            print_status(
                "OK", f"Offscreen Renderer functional (Backend: {gl_backend})."
            )
            renderer.close()
        except Exception as e:
            print_status(
                "WARN",
                f"MuJoCo rendering warning: {e}. Try setting 'export MUJOCO_GL=egl' or 'glfw'.",
            )

    # 6. ONNX & ONNX Runtime Inference Engine Check
    print("\n----- 6. ONNX Engine & Inference Session -----")
    if installed.get("onnxruntime"):
        import numpy as np
        import onnxruntime as ort

        try:
            providers = ort.get_available_providers()

            # End-to-end forward pass using an in-memory minimal graph if 'onnx' is available
            if installed.get("onnx"):
                import onnx
                from onnx import TensorProto, helper

                # Create dummy graph with Humanoid observation vector space (1x47)
                x = helper.make_tensor_value_info(
                    "observation", TensorProto.FLOAT, [1, 47]
                )
                y = helper.make_tensor_value_info(
                    "action", TensorProto.FLOAT, [1, 47]
                )
                node = helper.make_node(
                    "Identity", ["observation"], ["action"]
                )
                graph = helper.make_graph([node], "onnx_diag_test", [x], [y])
                test_model = helper.make_model(graph)

                session = ort.InferenceSession(
                    test_model.SerializeToString(), providers=providers
                )
                dummy_input = np.zeros((1, 47), dtype=np.float32)
                _ = session.run(None, {"observation": dummy_input})
                print_status(
                    "OK",
                    f"ONNX Runtime session & forward pass active\n        Providers: {', '.join(providers)}.",
                )
            else:
                print_status(
                    "OK",
                    f"ONNX Runtime ready\n       Providers: {', '.join(providers)}.",
                )
        except Exception as e:
            print_status("FAIL", f"ONNX Runtime session execution failed: {e}")

    print("\n========================================")
    print("Diagnosis Complete.")


if __name__ == "__main__":
    check_robotics_dependencies()