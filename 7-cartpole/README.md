# PPO CartPole

A from-scratch PPO (Proximal Policy Optimization) agent trained on `CartPole-v1`, built with Keras/TensorFlow.

## Setup (VS Code)

1. Open this folder in VS Code (`File > Open Folder...`).
2. Open a terminal in VS Code (`` Ctrl+` ``) and create a virtual environment:

   ```bash
   python -m venv venv
   ```

3. Activate it:
   - **Windows**: `venv\Scripts\activate`
   - **macOS/Linux**: `source venv/bin/activate`

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. In VS Code, select the venv as your Python interpreter: `Ctrl+Shift+P` → `Python: Select Interpreter` → pick the one inside `venv`.

## Run

```bash
python ppo_cartpole.py
```

You'll see per-epoch output like:

```
 Epoch: 1.  Mean Return: 20.8
 Epoch: 5.  Mean Return: 69.0
 Epoch: 10. Mean Return: 266.7
 Epoch: 15. Mean Return: 500.0
 Epoch: 20. Mean Return: 500.0
Evaluation: the agent balanced the pole for 500 steps.
```

The agent typically reaches the max episode length (500) by epoch ~15 and stays there — CartPole "solved."

After training, `evaluate()` runs one greedy episode and prints how many steps the agent survived. The trained actor is also saved to `actor_model.keras`.

## Visualize the trained agent

Once training has finished and `actor_model.keras` exists, run:

```bash
python visualize.py              # opens a live pop-up window (3 episodes)
python visualize.py --video       # saves an .mp4 to ./video instead
python visualize.py --episodes 5  # change the number of episodes
```

`--video` mode doesn't need a display and works well over SSH/headless setups; the live window mode needs a local display (works fine on a normal Windows/Mac/Linux desktop).

## Project structure

```
ppo_cartpole/
├── ppo_cartpole.py     # main script (env, model, PPO training loop)
├── visualize.py         # watch or record video of the trained agent
├── requirements.txt    # dependencies
├── .vscode/
│   └── launch.json     # F5 debug config
└── README.md
```

## How it works

This is a PPO (Proximal Policy Optimization) actor-critic agent:

- **Actor** — a small MLP (64→64) outputting action logits (left/right)
- **Critic** — a small MLP (64→64) estimating the value of a state
- **Buffer** — collects 4000 steps of experience per epoch, then computes advantages via GAE (Generalized Advantage Estimation) and discounted returns
- **Policy update** — the PPO clipped surrogate objective, capped by early stopping if KL divergence gets too large
- **Value update** — simple MSE regression toward the discounted returns

Trained for 20 epochs on `CartPole-v1`, reaching the maximum episode length (500 steps) consistently by around epoch 15.

## Notes

- Training takes a few minutes on CPU; no GPU required for CartPole.
- `CartPole-v1` episodes end either when the pole falls (`terminated`) or after 500 steps (`truncated`). Both are handled correctly in the training loop — truncated episodes bootstrap the return using the critic's value estimate rather than treating the cutoff as a true failure.