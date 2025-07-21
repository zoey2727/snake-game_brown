# snake_env.py
from . import snake        # ← relative import, matches snake/snake.py

import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, w=10, h=10, render_mode=None):
        self.w, self.h = w, h
        self.render_mode = render_mode
        self.game = None

        # flat 10×10 grid: 0 empty, 1 snake, 2 food
        self.observation_space = spaces.Box(0, 2,
                                            shape=(w * h,), dtype=np.int8)
        self.action_space = spaces.Discrete(4)  # 0=UP 1=DOWN 2=LEFT 3=RIGHT
        self._dir = ["UP", "DOWN", "LEFT", "RIGHT"]

    def _obs(self):
        g = np.zeros((self.h, self.w), np.int8)
        fx, fy = self.game.food
        g[fy, fx] = 2
        for x, y in self.game.snake:
            g[y, x] = 1
        return g.flatten()

    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game = snake.SnakeGame(self.w, self.h)
        return self._obs(), {}

    def step(self, action: int):
        self.game.step(self._dir[action])

        reward = 0.01                     # keep‑alive bonus
        terminated = not self.game.alive
        if terminated:
            reward = -1.0
        elif self.game.snake[0] == self.game.food:
            reward = +1.0

        return self._obs(), reward, terminated, False, {"score": self.game.score}

    def render(self):
        if self.render_mode == "human":
            self.game.draw()
