from functools import partial
from typing import Callable, Dict, Tuple, List

import jax
from flax import linen as nn
from jax import numpy as jnp

from genrl.utils.common.base import PolicyNNWrapper
from genrl.utils.common.type_aliases import nnOutput, GenRLPolicyInput, GenRLPolicyOutput
from genrl.utils.jax_utils.general import create_mlp
from genrl.utils.jax_utils.model import Model
from genrl.utils.jax_utils.scaler import Scaler
from genrl.utils.jax_utils.type_aliases import PRNGKey


class MLPBehaviorCloneNN(nn.Module):
    net_arch: List[int]

    activation_fn: Callable
    act_scale: float
    output_dim: int
    dropout: float

    pred_act = None

    def setup(self) -> None:
        pred_act = create_mlp(
            output_dim=self.output_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            dropout=self.dropout,
            squash_output=True
        )

        self.pred_act = Scaler(base_model=pred_act, scale=self.act_scale)

    def __call__(self, observations: jnp.ndarray, **kwargs) -> nnOutput:
        return self.forward(observations=observations)

    def forward(self, *, observations: jnp.ndarray) -> nnOutput:
        action_preds = self.pred_act(observations)
        return {"pred": action_preds}


class MLPBehaviorClone(PolicyNNWrapper):

    def __init__(self, seed: int, cfg: Dict):
        super(MLPBehaviorClone, self).__init__(seed=seed, cfg=cfg)

    def get_nn_class(self):
        return MLPBehaviorCloneNN

    def get_init_arrays(self) -> Tuple[jnp.ndarray, ...]:
        obs = jnp.zeros((1, 1, self.observation_dim))
        return obs,

    def _predict(self, x: GenRLPolicyInput, deterministic: bool = True, *args, **kwargs) -> GenRLPolicyOutput:
        pred = self.nn_forward(rng=self.rng, policy_nn=self.policy_nn, observations=x.observations)
        pred_action = pred.pop("pred")
        return GenRLPolicyOutput(pred_action=pred_action, info=pred)

    @staticmethod
    @partial(jax.jit, static_argnames=("deterministic",))
    def nn_forward(
        rng: PRNGKey,
        policy_nn: Model,
        observations: jnp.ndarray,
        deterministic: bool = False
    ):
        rng, dropout_key = jax.random.split(rng)
        action_pred = policy_nn.apply_fn(
            policy_nn.params,
            observations=observations,
            deterministic=deterministic,
            rngs={"dropout": dropout_key}
        )
        return action_pred
