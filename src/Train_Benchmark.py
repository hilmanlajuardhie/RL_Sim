import os
import time
import numpy as np
import gymnasium as gym
import onnxruntime as ort

model_path = os.path.join("Model", "humanoid_ppo_trained-2M.onnx")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at '{model_path}'. Ensure the export step completed successfully.")

# Initialize ONNX Runtime Session (CUDA with CPU fallback)
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession(model_path, providers=providers)

# Get dynamic input/output node names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"ONNX Policy Loaded: {model_path}")
print(f"Execution Provider: {session.get_providers()[0]}")
print(f"Input Node Name: '{input_name}' | Output Node Name: '{output_name}'\n")

# Setup Gymnasium Humanoid Environment
env = gym.make("Humanoid-v5", render_mode="human")
episodes = 10
frame_delay = 0.02

# Real-Time Inference Loop
for episode in range(1, episodes + 1):
    obs, info = env.reset()
    terminated = False
    truncated = False
    episode_reward = 0.0

    while not (terminated or truncated):
        # Format raw observation (47,) into batched float32 array (1, 47)
        obs_tensor = np.expand_dims(obs.astype(np.float32), axis=0)

        # Execute ONNX forward pass
        action_tensor = session.run([output_name], {input_name: obs_tensor})[0]

        # Extract continuous motor action vector (17,)
        action = action_tensor[0]

        # Step physical simulation
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        time.sleep(frame_delay)

    print(f"Episode {episode} Total Reward: {episode_reward:.2f}")
    time.sleep(1)

env.close()
print("\nInference benchmarking complete.")