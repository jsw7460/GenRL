import collections.abc
from types import MappingProxyType
from typing import Dict

import jax

from genrl.utils.common.type_aliases import GymEnv
from genrl.utils.jax_utils.general import str_to_flax_activation


class GenRLBaseModule:
    def __init__(self, seed: int, cfg: Dict, env: GymEnv, init_build_model: bool):
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)

        self.cfg = cfg
        self.env = env

        if init_build_model:
            self._str_to_activation()

        self.cfg = MappingProxyType(cfg)  # Freeze
        self.n_update = 0

    def _str_to_activation(self):
        def str_to_activation(data: collections.abc.Mapping):
            for key, value in data.items():
                if isinstance(value, collections.abc.Mapping):
                    str_to_activation(value)
                else:
                    if key == "activation_fn":
                        activation = str_to_flax_activation(value)
                        data[key] = activation

        str_to_activation(data=self.cfg)
