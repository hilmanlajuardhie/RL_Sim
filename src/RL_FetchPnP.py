import time
import gymnasium as gym
import gymnasium_robotics 
from stable_baselines3 import PPO

# Environment setup
gym.register_envs(gymnasium_robotics)
environment_name = "FetchPickAndPlace-v4"
episodes = 5
frame_delay = 0.02
target_timesteps = 200_000  # Increase to 1,000,000+ for full convergence

# 1. INITIALIZE TRAINING ENVIRONMENT & MODEL
# MultiInputPolicy is required for Dict observation spaces (GoalEnvs)
train_env = gym.make(environment_name)
model = PPO("MultiInputPolicy", train_env, device="cuda", verbose=1)

print(f"Environment: {environment_name}")
print(f"Policy Type: {model.policy.__class__.__name__}")
print(f"Evaluation Episodes: {episodes}\n")

# 2. STATUS 1: UNTRAINED MODEL EVALUATION
print("--- STATUS 1: BEFORE TRAINING ---")
test_env_before = gym.make(environment_name, render_mode="human")
before_scores = []

for eps in range(1, episodes + 1):
    obs, info = test_env_before.reset()
    terminated = False
    truncated = False
    score = 0.0

    while not (terminated or truncated):
        # Predict action directly from Dict observation
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env_before.step(action)
        score += reward
        time.sleep(frame_delay)

    before_scores.append(score)
    print(f"Episode {eps} Score: {score:.2f}")

test_env_before.close()

# 3. TRAINING PHASE
print(f"\nTraining Agent for {target_timesteps:,} timesteps...")
model.learn(total_timesteps=target_timesteps)
train_env.close()

# 4. STATUS 2: TRAINED MODEL EVALUATION
print("\n--- STATUS 2: AFTER TRAINING ---")
test_env_after = gym.make(environment_name, render_mode="human")
after_scores = []

for eps in range(1, episodes + 1):
    obs, info = test_env_after.reset()
    terminated = False
    truncated = False
    score = 0.0

    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env_after.step(action)
        score += reward
        time.sleep(frame_delay)

    after_scores.append(score)
    print(f"Episode {eps} Score: {score:.2f}")

test_env_after.close()

# 5. SCORE COMPARISON
avg_before = sum(before_scores) / len(before_scores)
avg_after = sum(after_scores) / len(after_scores)

print("\n==========================================")
print("             FINAL COMPARISON             ")
print("==========================================")
print(f"Average Score Before Training: {avg_before:.2f}")
print(f"Average Score After Training:  {avg_after:.2f}")
print(f"Difference:                    {avg_after - avg_before:+.2f} points")
print("==========================================")