import time
import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3 import SAC

print(f"Gym version: {gym.__version__}")
print(f"SB3 version: {sb3.__version__}")

environment_name = "InvertedPendulum-v5"
episodes = 5
fps_delay = 0.02

print(f"Environment: {environment_name}")
print(f"Episodes for Evaluation: {episodes}")

# ==========================================
# STAGE 1: UNTRAINED MODEL (Visualizing Baseline)
# ==========================================
print("\n--- STAGE 1: UNTRAINED MODEL ---")
test_env_before = gym.make(environment_name, render_mode="human")
model = SAC("MlpPolicy", test_env_before, device="cpu", verbose=1)
print(f"Model Architecture: {model.__class__.__name__}")

before_scores = []

for eps in range(1, episodes + 1):
    obs, info = test_env_before.reset()
    terminated = False
    truncated = False
    score = 0

    while not (terminated or truncated):
        # Predict continuous actions using initial random weights
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = test_env_before.step(action)
        score += reward
        time.sleep(fps_delay)

    before_scores.append(score)
    print(f"Episode {eps} Score: {score:.1f}")
    time.sleep(1.0)

test_env_before.close()

# ==========================================
# STAGE 2: MODEL ON TRAINING (Live Visual Training)
# ==========================================
print("\n--- STAGE 2: MODEL ON TRAINING ---")
print("Training SAC Agent for 1,000 timesteps with real-time rendering...")

# Instantiate training environment with human rendering active
train_env = gym.make(environment_name, render_mode="human")
model.set_env(train_env)

# Watch the agent interact with physics live while filling replay buffer & updating Q-values
model.learn(total_timesteps=1_000)

train_env.close()

# ==========================================
# STAGE 3: TRAINED MODEL (Visualizing Final Policy)
# ==========================================
print("\n--- STAGE 3: TRAINED MODEL ---")
test_env_after = gym.make(environment_name, render_mode="human")
after_scores = []

for eps in range(1, episodes + 1):
    obs, info = test_env_after.reset()
    terminated = False
    truncated = False
    score = 0

    while not (terminated or truncated):
        # Predict continuous actions deterministically from learned policy
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env_after.step(action)
        score += reward
        time.sleep(fps_delay)

    after_scores.append(score)
    print(f"Episode {eps} Score: {score:.1f}")
    time.sleep(1.0)

test_env_after.close()

# ==========================================
# PERFORMANCE COMPARISON
# ==========================================
avg_before = sum(before_scores) / len(before_scores)
avg_after = sum(after_scores) / len(after_scores)

print("\n==========================================")
print("             FINAL COMPARISON             ")
print("==========================================")
print(f"Average Score Before Training: {avg_before:.1f} steps")
print(f"Average Score After Training:  {avg_after:.1f} steps")
print(f"Improvement:                   +{avg_after - avg_before:.1f} steps")
print("==========================================")