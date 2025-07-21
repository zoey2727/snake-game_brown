import gymnasium as gym
import numpy as np
from gymnasium import spaces
from . import snake  

# --- helper tables (add once, near the top of the file) -----------------
DIR_VEC  = {"UP": (0, -1), "DOWN": (0, 1),
            "LEFT": (-1, 0), "RIGHT": (1, 0)}
OPPOSITE = {"UP": "DOWN", "DOWN": "UP",
            "LEFT": "RIGHT", "RIGHT": "LEFT"}

# ------------------------------- Bot ----------------------------------- #
class SnakeBot:
    """
    Greedy baseline:
    – move horizontally toward the food first, then vertically
    – never request a 180° turn
    """
    def next_move(self, state):
        hx, hy  = state["snake"][0]      # head
        fx, fy  = state["food"]          # food
        cur_dir = state["direction"]

        # align X first
        if hx < fx and cur_dir != "LEFT":   return "RIGHT"
        if hx > fx and cur_dir != "RIGHT":  return "LEFT"

        # then align Y
        if hy < fy and cur_dir != "UP":     return "DOWN"
        if hy > fy and cur_dir != "DOWN":   return "UP"

        return cur_dir                      # already lined up

