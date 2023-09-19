import collections.abc
import random
from copy import deepcopy
from types import MappingProxyType
from typing import Dict, Union, Tuple

import jax
import numpy as np
import optax
from jax import numpy as jnp

from genrl.utils.common.type_aliases import GenRLPolicyInput, GenRLPolicyOutput
from genrl.utils.jax_utils.general import get_basic_rngs
from genrl.utils.jax_utils.general import str_to_flax_activation
from genrl.utils.jax_utils.model import Model


class GenRLBaseModule:

    def __init__(self, seed: int, cfg: Dict, init_build_model: bool):
        random.seed(seed)
        np.random.seed(seed)

        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)

        self.cfg = deepcopy(dict(cfg))

        if init_build_model:
            self._str_to_activation()

        self.cfg = MappingProxyType(self.cfg)  # Freeze
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


class PolicyNNWrapper:

    def __init__(self, seed: int, cfg: Dict):
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)
        self.cfg = cfg

        self.observation_dim = cfg["observation_dim"]
        self.action_dim = cfg["action_dim"]
        self.subseq_len = cfg["subseq_len"]

        self.policy_nn = None  # type: Model
        self.optimizer_class = None
        self.nn_class = None

    def get_nn_class(self):
        raise NotImplementedError()

    def predict(self, x: GenRLPolicyInput, *args, **kwargs) -> GenRLPolicyOutput:
        self.rng, _ = jax.random.split(self.rng)
        return self._predict(x, *args, **kwargs)

    def _predict(self, x: GenRLPolicyInput, deterministic: bool = True, *args, **kwargs) -> GenRLPolicyOutput:
        raise NotImplementedError()

    def get_init_arrays(self) -> Tuple[jnp.ndarray, ...]:
        """This should be overloaded by child class.
        The output of this method is a tuple of arrays,
        WHOSE ORDER MUST BE MATCHED WITH THE CORRESPONDING NN_MODULE'S FORWARD FUNCTION.
        """
        raise NotImplementedError()

    def build(self):
        nn_class = self.get_nn_class()
        self.policy_nn = nn_class(**self.cfg["policy"])
        init_arr = self.get_init_arrays()

        self.optimizer_class = getattr(optax, self.cfg["optimizer_class"])

        tx = self.optimizer_class(learning_rate=self.cfg["lr"], **self.cfg["optimizer_kwargs"])
        self.rng, rngs = get_basic_rngs(self.rng)

        self.policy_nn = Model.create(
            apply_fn=self.policy_nn.apply,
            params=self.policy_nn.init(rngs, *init_arr),
            tx=tx
        )