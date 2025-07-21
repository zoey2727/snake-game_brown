# snake/play_sb3.py
"""
Loads the trained DQN model (dqn_snake.zip) and plays one game
with rendering so you can watch the score climb.
Run from the parent folder with:
    python -m snake.play_sb3
"""

from stable_baselines3 import DQN
from .snake_env import SnakeEnv   # relative import inside the package

# Create a renderable environment
env = SnakeEnv(render_mode="human")

# Load the model saved by train_sb3.py
model = DQN.load("dqn_snake", device="cpu")   # CPU is fine for inference

# Play one episode
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, info = env.step(action)

print("\nFinal score:", info["score"])
"/Users/zoeytzhori/whatever i want/.venv/bin/python" "/Users/zoeytzhori/whatever i want/snake/train_sb3.py"

