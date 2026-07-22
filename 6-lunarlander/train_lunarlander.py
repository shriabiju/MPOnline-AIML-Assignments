"""
LunarLander-v3 PPO Agent
-------------------------
Trains a PPO agent (Stable-Baselines3) to solve Gymnasium's LunarLander-v3.
Adapted to run locally (VS Code / plain Python), no Colab-specific setup needed.

Usage:
    python train_lunarlander.py
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def main():
    # ---- 1. Create the environment ----
    env = gym.make("LunarLander-v3", continuous=False)
    env.reset()

    print("_____OBSERVATION SPACE_____")
    print("Shape:", env.observation_space.shape)
    print("Sample:", env.observation_space.sample())

    print("\n_____ACTION SPACE_____")
    print("Shape:", env.action_space.n)
    print("Sample:", env.action_space.sample())

    # ---- 2. Define the PPO model ----
    # device="cpu" because MlpPolicy PPO usually runs faster on CPU than GPU
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1,
        device="cpu",
    )

    # ---- 3. Train ----
    # Note: 1,000,000 timesteps takes a while on CPU (~20-40 min depending on
    # your machine). Lower this for a quick test run, e.g. 200_000.
    TOTAL_TIMESTEPS = 1_000_000
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # ---- 4. Save the model ----
    model.save("models/ppo_lunar")
    print("\nModel saved to models/ppo_lunar.zip")

    # ---- 5. Evaluate ----
    eval_env = Monitor(gym.make("LunarLander-v3", continuous=False))
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"\nmean_reward={mean_reward:.2f} +/- {std_reward}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
