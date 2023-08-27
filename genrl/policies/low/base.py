from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import optax

from genrl.policies.base import GenRLBaseModule
from genrl.utils.interfaces import JaxSavable, Trainable
from genrl.utils.jax_utils.model import Model
from genrl.utils.jax_utils.type_aliases import Params
from genrl.utils.common.type_aliases import PolicyOutput


class BaseLowPolicy(GenRLBaseModule, JaxSavable, Trainable):
    PARAM_NAMES = ["policy_nn"]

    def __init__(self, seed: int, cfg: Dict, init_build_model: bool):
        super(BaseLowPolicy, self).__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)

        self.observation_dim = cfg["observation_dim"]
        self.action_dim = cfg["action_dim"]

        self.optimizer_class = None
        self.policy_nn = None  # type: Model

        if init_build_model:
            self.build()

    def build(self):
        optimizer = getattr(optax, self.cfg["optimizer_class"])
        self.optimizer_class = optimizer

    def _excluded_save_params(self) -> List:
        return BaseLowPolicy.PARAM_NAMES

    def _get_save_params(self) -> Dict[str, Params]:
        params_dict = {}
        for str_component in BaseLowPolicy.PARAM_NAMES:
            component = getattr(self, str_component)
            params_dict[str_component] = component.params
        return params_dict

    def _get_load_params(self) -> List[str]:
        return BaseLowPolicy.PARAM_NAMES

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> PolicyOutput:
        """
            Get the policy action from an observation (and optional hidden state).
            Includes sugar-coating to handle different observations (e.g. normalizing images).

            :param observation: the input observation
            :param state: The last hidden states (can be None, used in recurrent policies)
            :param episode_start: The last masks (can be None, used in recurrent policies)
                this correspond to beginning of episodes,
                where the hidden states of the RNN must be reset.
            :param deterministic: Whether or not to return deterministic actions.
            :return: the model's action and the next hidden state
                (used in recurrent policies)
        """
        return self._predict(
            observation=observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic
        )

    def _predict(self, *args, **kwargs) -> PolicyOutput:
        raise NotImplementedError()

    def update(self, *args, **kwargs) -> Dict:
        self.n_update += 1
        return self._update(*args, **kwargs)

    def _update(self, *args, **kwargs):
        raise NotImplementedError()

    def evaluate(self, *args, **kwargs) -> Dict:
        pass
