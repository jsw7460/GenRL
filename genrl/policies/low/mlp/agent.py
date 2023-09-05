from typing import Dict, Tuple

import jax
from hydra.utils import get_class
from jax import numpy as jnp

from genrl.policies.base import BasePolicy
from genrl.rl.buffers.type_aliases import GenRLBufferSample
from genrl.utils.common.type_aliases import GenRLPolicyInput, GenRLPolicyOutput
from genrl.utils.jax_utils.model import Model
from genrl.utils.jax_utils.type_aliases import Params


class MLPBCAgent(BasePolicy):

    def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
        super(MLPBCAgent, self).__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)

    def build(self) -> None:
        super(MLPBCAgent, self).build()

        policy_cls = get_class(self.cfg["policy"]["target"])
        policy_cfg = self.cfg["policy"]["cfg"]
        self.policy = policy_cls(seed=self.seed, cfg=policy_cfg)
        self.policy.build()

    def _update(self, replay_data: GenRLBufferSample) -> Dict:
        new_policy, info = self.update_mlp(
            rng=self.rng,
            policy=self.policy_nn,
            observations=replay_data.observations,
            actions=replay_data.actions,
            maskings=replay_data.masks
        )
        self.policy_nn = new_policy
        return info

    @staticmethod
    @jax.jit
    def update_mlp(
        rng: jnp.ndarray,
        policy: Model,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        maskings: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Dict]:
        rng, dropout_key = jax.random.split(rng)

        batch_size = observations.shape[0]
        subseq_len = observations.shape[1]

        action_targ = actions.reshape(batch_size * subseq_len, -1) * maskings.reshape(-1, 1)

        def _loss(params: Params) -> Tuple[jnp.ndarray, Dict]:
            pred = policy.apply_fn(params, observations=observations, rngs={"dropout": dropout_key})

            action_pred = pred["pred"].reshape(batch_size * subseq_len, -1) * maskings.reshape(-1, 1)

            mse_loss = jnp.sum(jnp.mean((action_pred - action_targ) ** 2, axis=-1)) / jnp.sum(maskings)
            _info = {"mse_loss": mse_loss}

            return mse_loss, _info

        grads, info = jax.grad(_loss, has_aux=True)(policy.params)
        new_policy = policy.apply_gradients(grads=grads)

        return new_policy, info

    def _predict(self, x: GenRLPolicyInput) -> GenRLPolicyOutput:
        return self.policy.predict(x=x)
