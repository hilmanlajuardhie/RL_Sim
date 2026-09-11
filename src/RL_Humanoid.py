import os
import time
import torch
import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3 import PPO

print(f"Gym version: {gym.__version__}")
print(f"SB3 version: {sb3.__version__}")

environment_name = "Humanoid-v5"
episodes = 10
frame_delay = 0.02 #The Playback FPS

train_env = gym.make(environment_name)
model = PPO("MlpPolicy", train_env, device="cuda", verbose=1)

print(f"Environtment: {environment_name}")
print(f"Model: {model.__class__.__name__}")
# print(f"Model Epochs: {model.n_epochs}")
print(f"Episode: {episodes}")

# 1. STATUS BEFORE TRAINING (Untrained Model)
print("\n--- STATUS 1: BEFORE TRAINING ---")
test_env_before = gym.make(environment_name, render_mode="human")
before_scores = []

for eps in range(1, episodes + 1):
    obs, info = test_env_before.reset()
    terminated = False
    truncated = False
    score = 0

    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env_before.step(action)
        score += reward
        time.sleep(frame_delay)

    before_scores.append(score)
    print(f"Episode {eps} Score: {score:.1f}")
    time.sleep(1.0)

test_env_before.close()

# 2. TRAINING PHASE
print("\nTraining Agent for 1.000.000 timesteps...")
model.learn(total_timesteps=10_000)
train_env.close()

# 3. STATUS AFTER TRAINING (Trained Model)
print("\n--- STATUS 2: AFTER TRAINING ---")
test_env_after = gym.make(environment_name, render_mode="human")
after_scores = []

for eps in range(1, episodes + 1):
    obs, info = test_env_after.reset()
    terminated = False
    truncated = False
    score = 0

    while not (terminated or truncated):
        # Predict using trained weights
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env_after.step(action)
        score += reward
        time.sleep(frame_delay)

    after_scores.append(score)
    print(f"Episode {eps} Score: {score:.1F}")
    time.sleep(1.5)

test_env_after.close()

# 4. PERFORMANCE COMPARISON
avg_before = sum(before_scores) / len(before_scores)
avg_after = sum(after_scores) / len(after_scores)

print("\n==========================================")
print("             FINAL COMPARISON             ")
print("==========================================")
print(f"Average Score Before Training: {avg_before:.1f} / 500.0")
print(f"Average Score After Training:  {avg_after:.1f} / 500.0")
print(f"Differents:                    {avg_after - avg_before:.1f} points")
print("==========================================")


# ==========================================
# 5. EXPORT TRAINED MODEL TO FOLDER "Model"
# ==========================================
print("\n--- EXPORTING TRAINED POLICY TO ONNX ---")

# Ensure the target directory exists
save_dir = "Model"
os.makedirs(save_dir, exist_ok=True)
onnx_filename = os.path.join(save_dir, "humanoid_ppo_policy_trained.onnx")

class OnnxablePolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # Returns deterministic continuous action vector (a_t) for input state
        return self.policy._predict(observation, deterministic=True)

# 1. Move policy to CPU for clean, device-agnostic ONNX serialization
model.policy.to("cpu")
onnx_policy = OnnxablePolicy(model.policy)

# 2. Generate dummy input tensor matching observation space (1, obs_dim)
obs_dim = model.observation_space.shape[0]
dummy_input = torch.randn(1, obs_dim, dtype=torch.float32)

# 3. Export policy computation graph to .onnx file inside "Model" folder
torch.onnx.export(
    onnx_policy,
    dummy_input,
    onnx_filename,
    opset_version=17,
    input_names=["observation"],
    output_names=["action"],
    dynamic_axes={
        "observation": {0: "batch_size"},
        "action": {0: "batch_size"}
    },
    dynamo=False
)

print(f"Policy successfully exported to: {onnx_filename}")