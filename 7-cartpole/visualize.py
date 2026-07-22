"""
Visualize a trained PPO CartPole agent.
========================================
Run this AFTER ppo_cartpole.py has trained and saved actor_model.keras.

Usage:
    python visualize.py            # opens a live pop-up window
    python visualize.py --video    # saves an .mp4 instead (headless-friendly)
"""

import argparse
import glob
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import keras
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


def watch_live(model_path="actor_model.keras", episodes=3):
    actor = keras.models.load_model(model_path)
    env = gym.make("CartPole-v1", render_mode="human")

    for ep in range(1, episodes + 1):
        observation, _ = env.reset()
        done = False
        steps = 0
        while not done:
            logits = actor(observation.reshape(1, -1))
            action = int(np.argmax(logits.numpy()[0]))  # greedy action
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        print(f"Episode {ep}: balanced the pole for {steps} steps.")

    env.close()


def save_video(model_path="actor_model.keras", episodes=1, video_folder="./video"):
    actor = keras.models.load_model(model_path)
    base_env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(base_env, video_folder=video_folder, name_prefix="cartpole-agent", disable_logger=True)

    for ep in range(1, episodes + 1):
        observation, _ = env.reset()
        done = False
        steps = 0
        while not done:
            logits = actor(observation.reshape(1, -1))
            action = int(np.argmax(logits.numpy()[0]))
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        print(f"Episode {ep}: balanced the pole for {steps} steps.")

    env.close()
    video_file = sorted(glob.glob(f"{video_folder}/cartpole-agent*.mp4"))[-1]
    print(f"Saved video to: {video_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="store_true", help="Save an .mp4 instead of opening a live window")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    args = parser.parse_args()

    if args.video:
        save_video(episodes=args.episodes)
    else:
        watch_live(episodes=args.episodes)
