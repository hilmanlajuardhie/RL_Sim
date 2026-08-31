import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3 import PPO

print(f"Gym version: {gym.__version__}")
print(f"SB3 version: {sb3.__version__}")

environment_name = "CartPole-v1"
episodes = 5

# Set device="cpu" to silence the CUDA warning and speed up vector processing
train_env = gym.make(environment_name)
model = PPO("MlpPolicy", train_env, verbose=1)

print(f"Environtment: {environment_name}")
print(f"Episode: {episodes}")
print(f"Model: {model}")
print(f"Model Epochs: {model.n_epochs}")
print(f"Model Learn: {model.learn}")

# ==========================================
# 1. STATUS BEFORE TRAINING (Untrained Model)
# ==========================================
print("\n--- STATUS 1: BEFORE TRAINING ---")
test_env_before = gym.make(environment_name, render_mode="human")
before_scores = []

for eps in range(1, episodes + 1):
    obs, info = test_env_before.reset()
    terminated = False
    truncated = False
    score = 0

    while not (terminated or truncated):
        # Predict using untrained initial weights
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env_before.step(action)
        score += reward

    before_scores.append(score)
    print(f"Episode {eps} Score: {score}")

test_env_before.close()

# ==========================================
# 2. TRAINING PHASE
# ==========================================
print("\nTraining PPO Agent for 20,000 timesteps...")
model.learn(total_timesteps=20_000)

train_env.close()

# ==========================================
# 3. STATUS AFTER TRAINING (Trained Model)
# ==========================================
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

    after_scores.append(score)
    print(f"Episode {eps} Score: {score}")

test_env_after.close()

# ==========================================
# 4. PERFORMANCE COMPARISON
# ==========================================
avg_before = sum(before_scores) / len(before_scores)
avg_after = sum(after_scores) / len(after_scores)

print("\n==========================================")
print("             FINAL COMPARISON             ")
print("==========================================")
print(f"Average Score Before Training: {avg_before:.1f} / 500.0")
print(f"Average Score After Training:  {avg_after:.1f} / 500.0")
print(f"Improvement:                   +{avg_after - avg_before:.1f} points")
print("==========================================")