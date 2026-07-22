"""
Watch the trained PPO agent play LunarLander in a live window.
Run this AFTER train_lunarlander.py has produced models/ppo_lunar.zip

Usage:
    python watch_agent.py
"""

import gymnasium as gym
from stable_baselines3 import PPO

MODEL_PATH = "models/ppo_lunar"
EPISODES = 5


def main():
    env = gym.make("LunarLander-v3", continuous=False, render_mode="human")
    model = PPO.load(MODEL_PATH, env=env)

    for ep in range(EPISODES):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {ep + 1}: reward = {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
