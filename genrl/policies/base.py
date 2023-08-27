import collections.abc
import random
from types import MappingProxyType
from typing import Dict, Union

import jax
import numpy as np

from genrl.utils.jax_utils.general import str_to_flax_activation


class GenRLBaseModule:
    def __init__(self, seed: int, cfg: Dict, init_build_model: bool):
        random.seed(seed)
        np.random.seed(seed)

        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)

        self.cfg = cfg

        if init_build_model:
            self._str_to_activation()

        self.cfg = MappingProxyType(cfg)  # Freeze
        self.n_update = 0

    def _str_to_activation(self):
        def str_to_activation(data: Union[collections.abc.Mapping, Dict]):
            for key, value in data.items():
                if isinstance(value, collections.abc.Mapping):
                    str_to_activation(value)
                else:
                    if key == "activation_fn":
                        activation = str_to_flax_activation(value)
                        data[key] = activation

        str_to_activation(data=self.cfg)
