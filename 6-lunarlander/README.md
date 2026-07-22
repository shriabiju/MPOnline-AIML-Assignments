# LunarLander-v3 PPO Agent

Trains a PPO (Proximal Policy Optimization) agent using Stable-Baselines3 to solve
Gymnasium's `LunarLander-v3` environment. Adapted from a Colab notebook to run
locally in VS Code / plain Python — no `pyvirtualdisplay` or Colab-specific setup
required.

## Setup

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

If the `box2d` dependency fails to install, try installing `swig` first:

```bash
pip install swig
pip install -r requirements.txt
```

## Usage

**1. Train the agent** (takes ~20-40 min on CPU for the full 1M timesteps —
lower `TOTAL_TIMESTEPS` in `train_lunarlander.py` for a quicker test run):

```bash
python train_lunarlander.py
```

This prints observation/action space info, trains the PPO model, saves it to
`models/ppo_lunar.zip`, and evaluates it over 10 episodes.

**2. Watch the trained agent play** (opens a live rendering window):

```bash
python watch_agent.py
```

## Files

- `train_lunarlander.py` — trains and evaluates the PPO agent
- `watch_agent.py` — loads the saved model and renders it playing live
- `requirements.txt` — dependencies
- `models/` — where the trained model gets saved

## Notes

- PPO with `MlpPolicy` (non-CNN) typically runs *faster on CPU* than GPU, so
  the model is set to `device="cpu"`.
- A `mean_reward` above ~200 is generally considered "solved" for LunarLander.
- Environment: 8-dim observation space (position, velocity, angle, leg contact),
  4 discrete actions (do nothing, fire left/main/right engine).
