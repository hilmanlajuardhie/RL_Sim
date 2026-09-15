import os
import torch
import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3 import PPO


print(f"Gym version: {gym.__version__}")
print(f"SB3 version: {sb3.__version__}")

environment_name = "Humanoid-v5"
train_env = gym.make(environment_name)
model = PPO("MlpPolicy", train_env, device="cpu", verbose=1)

print(f"Environtment: {environment_name}")
print(f"Model: {model.__class__.__name__}")
print(f"Model Epochs: {model.n_epochs}")

# TRAINING PHASE
print("\nTraining Agent for 2.000.000 timesteps...")
model.learn(total_timesteps=2_000_000)
train_env.close()


# ==========================================
# 5. EXPORT TRAINED MODEL TO FOLDER "Model"
# ==========================================
print("\n--- EXPORTING TRAINED POLICY TO ONNX ---")

# Ensure the target directory exists
os.makedirs("Model", exist_ok=True)
onnx_filename = os.path.join("Model", "humanoid_ppo_trained-2M.onnx")

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