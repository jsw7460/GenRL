from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

GymEnv = gym.Env
PolicyOutput = Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]
