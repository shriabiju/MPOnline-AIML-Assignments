"""
PPO (Proximal Policy Optimization) on CartPole-v1
===================================================
A from-scratch actor-critic PPO agent, built in clear steps.

Install first:
    pip install gymnasium keras tensorflow scipy

Run:
    python ppo_cartpole.py
"""

# ──────────────────────────────────────────────────────────────
# STEP 1: Imports & backend setup
# ──────────────────────────────────────────────────────────────
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers
import numpy as np
import tensorflow as tf
import gymnasium as gym
import scipy.signal


# ──────────────────────────────────────────────────────────────
# STEP 2: Helper function — discounted cumulative sums
# Used to compute both the discounted returns and the
# GAE (Generalized Advantage Estimation) advantages.
# ──────────────────────────────────────────────────────────────
def discounted_cumulative_sums(x, discount):
    # Runs a discounted sum backwards through a sequence
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


# ──────────────────────────────────────────────────────────────
# STEP 3: Buffer class — stores one epoch's worth of experience
# and computes advantages/returns once the epoch is done.
# ──────────────────────────────────────────────────────────────
class Buffer:
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        self.observation_buffer = np.zeros((size, observation_dimensions), dtype=np.float32)
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Called at the end of an episode (or when the buffer is full)
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        # TD residuals (deltas) → GAE advantage
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantage_buffer[path_slice] = discounted_cumulative_sums(deltas, self.gamma * self.lam)

        # Discounted rewards-to-go → used as the value function's training target
        self.return_buffer[path_slice] = discounted_cumulative_sums(rewards, self.gamma)[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Called once per epoch — resets pointers and normalizes advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = np.mean(self.advantage_buffer), np.std(self.advantage_buffer)
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )


# ──────────────────────────────────────────────────────────────
# STEP 4: Build the actor and critic networks (simple MLPs)
# ──────────────────────────────────────────────────────────────
def mlp(x, sizes, activation=keras.activations.tanh, output_activation=None):
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=int(sizes[-1]), activation=output_activation)(x)


def logprobabilities(logits, a, num_actions):
    # Log-probability of taking action `a`, given the actor's logits
    logprobabilities_all = keras.ops.log_softmax(logits)
    logprobability = keras.ops.sum(
        keras.ops.one_hot(a, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability


# ──────────────────────────────────────────────────────────────
# STEP 5: Hyperparameters
# ──────────────────────────────────────────────────────────────
steps_per_epoch = 4000
epochs = 20
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
train_policy_iterations = 80
train_value_iterations = 80
lam = 0.97
target_kl = 0.01
hidden_sizes = (64, 64)


# ──────────────────────────────────────────────────────────────
# STEP 6: Environment + models + optimizers
# ──────────────────────────────────────────────────────────────
env = gym.make("CartPole-v1")
observation_dimensions = env.observation_space.shape[0]
num_actions = env.action_space.n

buffer = Buffer(observation_dimensions, steps_per_epoch, gamma, lam)

observation_input = keras.Input(shape=(observation_dimensions,), dtype="float32")
logits = mlp(observation_input, list(hidden_sizes) + [num_actions])
actor = keras.Model(inputs=observation_input, outputs=logits)

value = keras.ops.squeeze(mlp(observation_input, list(hidden_sizes) + [1]), axis=1)
critic = keras.Model(inputs=observation_input, outputs=value)

policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)

seed_generator = keras.random.SeedGenerator(1337)


# ──────────────────────────────────────────────────────────────
# STEP 7: Action sampling + training step functions
# ──────────────────────────────────────────────────────────────
@tf.function
def sample_action(observation):
    logits = actor(observation)
    action = keras.ops.squeeze(
        keras.random.categorical(logits, 1, seed=seed_generator), axis=1
    )
    return logits, action


@tf.function
def train_policy(observation_buffer, action_buffer, logprobability_buffer, advantage_buffer):
    with tf.GradientTape() as tape:
        # PPO clipped surrogate objective
        ratio = keras.ops.exp(
            logprobabilities(actor(observation_buffer), action_buffer, num_actions)
            - logprobability_buffer
        )
        min_advantage = keras.ops.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )
        policy_loss = -keras.ops.mean(
            keras.ops.minimum(ratio * advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    # KL divergence — used to trigger early stopping if the policy moves too far
    kl = keras.ops.mean(
        logprobability_buffer
        - logprobabilities(actor(observation_buffer), action_buffer, num_actions)
    )
    return keras.ops.sum(kl)


@tf.function
def train_value_function(observation_buffer, return_buffer):
    with tf.GradientTape() as tape:
        value_loss = keras.ops.mean((return_buffer - critic(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))


# ──────────────────────────────────────────────────────────────
# STEP 8: Main training loop
# ──────────────────────────────────────────────────────────────
def train():
    observation, _ = env.reset()
    episode_return, episode_length = 0, 0

    for epoch in range(epochs):
        sum_return = 0
        sum_length = 0
        num_episodes = 0

        for t in range(steps_per_epoch):
            observation_reshaped = observation.reshape(1, -1)
            logits, action = sample_action(observation_reshaped)
            observation_new, reward, terminated, truncated, _ = env.step(action[0].numpy())
            episode_return += reward
            episode_length += 1

            value_t = critic(observation_reshaped)
            logprobability_t = logprobabilities(logits, action, num_actions)

            buffer.store(
                observation,
                int(action[0].numpy()),
                reward,
                float(value_t[0].numpy()),
                float(logprobability_t[0].numpy()),
            )
            observation = observation_new

            terminal = terminated or truncated
            if terminal or (t == steps_per_epoch - 1):
                # Bootstrap with the critic's value estimate unless the episode
                # ended in a true terminal state (pole fell) — a truncation
                # (hit the 500-step time limit) is not a "true" end of episode.
                last_value = 0 if terminated else float(critic(observation.reshape(1, -1))[0].numpy())
                buffer.finish_trajectory(last_value)
                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                observation, _ = env.reset()
                episode_return, episode_length = 0, 0

        # Pull the epoch's data out of the buffer
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = buffer.get()

        # Update the policy (with early stopping on KL divergence)
        for _ in range(train_policy_iterations):
            kl = train_policy(observation_buffer, action_buffer, logprobability_buffer, advantage_buffer)
            if kl > 1.5 * target_kl:
                break

        # Update the value function
        for _ in range(train_value_iterations):
            train_value_function(observation_buffer, return_buffer)

        print(f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. "
              f"Mean Length: {sum_length / num_episodes}")

    # Save the trained actor so it can be reused later without retraining
    actor.save("actor_model.keras")
    print("Saved trained actor to actor_model.keras")


# ──────────────────────────────────────────────────────────────
# STEP 9: Evaluate the trained agent (no video, just a run count)
# ──────────────────────────────────────────────────────────────
def evaluate():
    test_env = gym.make("CartPole-v1")
    observation, _ = test_env.reset()
    done = False
    steps_survived = 0

    while not done:
        logits = actor(observation.reshape(1, -1))
        action = np.argmax(logits.numpy()[0])  # greedy action at test time
        observation, reward, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated
        steps_survived += 1

    print(f"Evaluation: the agent balanced the pole for {steps_survived} steps.")
    test_env.close()


if __name__ == "__main__":
    train()
    evaluate()
