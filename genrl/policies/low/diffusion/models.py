from functools import partial
from typing import Dict, Tuple

import jax
from jax import numpy as jnp

from genrl.utils import AttrDict
from genrl.policies.low.diffusion.ddpm_schedule import ddpm_linear_schedule
from genrl.policies.low.base import BaseLowPolicy
from genrl.policies.low.diffusion.arch.mlpdiffusion import MLPDiffusionNN  # Todo: Change to transformer
from genrl.rl.buffers.type_aliases import GenRLBufferSample
from genrl.utils.common.type_aliases import GymEnv
from genrl.utils.jax_utils.general import get_basic_rngs
from genrl.utils.jax_utils.model import Model
from genrl.utils.jax_utils.type_aliases import PRNGKey, Params


class DiffusionLowPolicy(BaseLowPolicy):

    def __init__(self, seed: int, env: GymEnv, cfg: Dict, init_build_model: bool = True):
        super(DiffusionLowPolicy, self).__init__(seed=seed, env=env, cfg=cfg, init_build_model=init_build_model)

        self.n_denoise = cfg["n_denoise"]
        self.noise_dim = cfg["noise_dim"]
        self.ddpm_schedule = ddpm_linear_schedule(**self.cfg["ddpm_schedule"])

    def build(self):
        nn = MLPDiffusionNN(**self.cfg["nn_cfg"])
        obs = jnp.array([self.env.observation_space.sample()])
        act = jnp.array([self.env.action_space.sample()])
        time = jnp.array([0])
        self.rng, rngs = get_basic_rngs(self.rng)

        tx = self.optimizer_class(learning_rate=self.cfg["lr"], **self.cfg["optimizer_kwargs"])
        self.policy = Model.create(
            apply_fn=nn.apply,
            params=nn.init(self.rng, obs, act, time),
            tx=tx
        )

    def update(self, replay_data: GenRLBufferSample) -> Dict:
        self.rng, _ = jax.random.split(self.rng)
        new_policy, info = self.update_diffusion(
            rng=self.rng,
            policy=self.policy,
            observations=replay_data.observations,
            actions=replay_data.actions,
            masks=replay_data.masks,

            ddpm_schedule=self.ddpm_schedule,
            noise_dim=self.noise_dim,
            n_denoise=self.n_denoise
        )

        print("Run!!" * 999)


    @staticmethod
    @partial(jax.jit, static_argnames=("noise_dime", "n_denoise"))
    def update_diffusion(
        rng: PRNGKey,
        policy: Model,
        observations: jnp.ndarray,  # [b, l, d]
        actions: jnp.ndarray,   # [b, l, d]
        masks: jnp.ndarray, # [b, l]

        ddpm_schedule: AttrDict,
        noise_dim: int,
        n_denoise: int,
    ) -> Tuple[Model, Dict]:
        batch_size = observations.shape[0]
        broadcast_shape = (batch_size,) + (1,) * (actions.ndim - 1)

        # sample noise
        noise = jax.random.normal(rng, shape=(*actions.shape[:-1], noise_dim))

        # add noise to clean target actions
        _ts = jax.random.randint(rng, shape=(batch_size,), minval=1, maxval=n_denoise)
        sqrtab = ddpm_schedule.sqrtab[_ts].reshape(broadcast_shape)
        sqrtmab = ddpm_schedule.sqrtmab[_ts].reshape(broadcast_shape)

        y_t = sqrtab * actions + sqrtmab * noise

        # use diffusion model to predict noise
        def _loss(params: Params) -> Tuple[jnp.ndarray, Dict]:
            noise_pred = policy.apply_fn(
                {"params": params},
                x=observations,
                y=y_t,
                t=_ts
            )
            mse_loss = jnp.mean(jnp.sum((noise_pred - noise) ** 2, axis=-1) * masks)

            _info = {
                "mse_loss": mse_loss,
                "__noise_pred": noise_pred
            }

            return mse_loss, _info

        grads, info = jax.grad(_loss, has_aux=True)(policy.params)
        new_policy = policy.apply_gradients(grads=grads)

        return new_policy, info