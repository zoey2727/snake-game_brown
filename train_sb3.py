# snake/train_sb3.py
"""
Trains a small DQN agent on the Snake environment and
saves it as dqn_snake.zip in the parent folder.
Run with:
    python -m snake.train_sb3
from the directory that contains the `snake/` package.
"""

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from .snake_env import SnakeEnv

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Wrap the env so SB3 can batch automatically (even if batch size = 1)
env = DummyVecEnv([lambda: SnakeEnv()])

model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=50_000,
    learning_starts=1_000,
    batch_size=64,
    gamma=0.99,
    target_update_interval=1_000,
    exploration_fraction=0.10,
    device=DEVICE,
    verbose=1,
)

print("▶ Training…  (≈10–40 min depending on CPU/GPU)")
model.learn(total_timesteps=200_000)
model.save("dqn_snake")
print("✓ Done. Model saved as dqn_snake.zip")
