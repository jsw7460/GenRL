from functools import partial
from typing import Dict, Tuple

import jax
from hydra.utils import get_class
from jax import numpy as jnp

from genrl.policies.base import BasePolicy
from genrl.policies.low.diffusion.ddpm_schedule import ddpm_linear_schedule
from genrl.rl.buffers.type_aliases import GenRLBufferSample
from genrl.utils.jax_utils.model import Model
from genrl.utils.jax_utils.type_aliases import PRNGKey, Params
from genrl.utils.common.type_aliases import GenRLPolicyInput, GenRLPolicyOutput
from genrl.utils.common.base import PolicyNNWrapper


class DiffusionLowPolicy(BasePolicy):

    def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
        self.total_denoise_steps = cfg["total_denoise_steps"]
        self.noise_dim = cfg["noise_dim"]
        self.ddpm_schedule = ddpm_linear_schedule(**cfg["ddpm_schedule"])
        super(DiffusionLowPolicy, self).__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)

    def build(self):
        super(DiffusionLowPolicy, self).build()

        policy_cls = get_class(self.cfg["policy"]["target"])
        policy_cfg = self.cfg["policy"]["cfg"]
        policy_cfg.update({"ddpm_schedule": self.ddpm_schedule})
        self.policy = policy_cls(seed=self.seed, cfg=policy_cfg)    # type: PolicyNNWrapper
        self.policy.build()

    @property
    def policy_nn(self):
        return self.policy.policy_nn

    @policy_nn.setter
    def policy_nn(self, nn):
        self.policy.policy_nn = nn

    def _update(self, replay_data: GenRLBufferSample) -> Dict:
        new_policy, info = self.update_diffusion(
            rng=self.rng,
            policy=self.policy_nn,
            observations=replay_data.observations,
            actions=replay_data.actions,
            masks=replay_data.masks,

            ddpm_schedule=self.ddpm_schedule,
            noise_dim=self.noise_dim,
            total_denoise_steps=self.total_denoise_steps
        )

        self.policy_nn = new_policy
        return info

    @staticmethod
    @partial(jax.jit, static_argnames=("noise_dim", "total_denoise_steps"))
    def update_diffusion(
        rng: PRNGKey,
        policy: Model,
        observations: jnp.ndarray,  # [b, l, d]
        actions: jnp.ndarray,  # [b, l, d]
        masks: jnp.ndarray,  # [b, l]

        ddpm_schedule: Dict,
        noise_dim: int,
        total_denoise_steps: int,
    ) -> Tuple[Model, Dict]:
        rng, dropout_key = jax.random.split(rng)
        batch_size = observations.shape[0]
        subseq_len = observations.shape[1]

        broadcast_shape = (batch_size,) + (1,) * (actions.ndim - 1)

        # sample noise
        noise = jax.random.normal(rng, shape=(*actions.shape[:-1], noise_dim))

        # add noise to clean target actions
        _ts = jax.random.randint(rng, shape=(batch_size,), minval=1, maxval=total_denoise_steps)

        sqrtab = ddpm_schedule["sqrtab"][_ts].reshape(broadcast_shape)
        sqrtmab = ddpm_schedule["sqrtmab"][_ts].reshape(broadcast_shape)

        y_t = sqrtab * actions + sqrtmab * noise

        # use diffusion model to predict noise
        def _loss(params: Params) -> Tuple[jnp.ndarray, Dict]:
            pred = policy.apply_fn(
                params,
                x=observations,
                y=y_t,
                t=jnp.repeat(_ts[..., jnp.newaxis], repeats=subseq_len, axis=-1),
                maskings=masks,
                rngs={"dropout": dropout_key}
            )
            noise_pred = pred["pred"]

            mse_loss = jnp.sum(jnp.mean((noise_pred - noise) ** 2, axis=-1)) / jnp.sum(masks)

            # If key startswith __ (double underbar), then it is not printed.
            _info = {
                "mse_loss": mse_loss,
                "__noise_pred": noise_pred,
                "__noise_targ": noise
            }

            return mse_loss, _info

        grads, info = jax.grad(_loss, has_aux=True)(policy.params)
        new_policy = policy.apply_gradients(grads=grads)

        return new_policy, info

    def _predict(self, x: GenRLPolicyInput) -> GenRLPolicyOutput:

        observation = x.observations
        broadcast_shape = observation.shape[: -1]

        # sample initial noise, y_T ~ Normal(0, 1)
        y_t = jax.random.normal(self.rng, shape=(*observation.shape[: -1], self.noise_dim))
        # denoising chain
        for t in range(self.total_denoise_steps, 0, -1):
            self.rng, _ = jax.random.split(self.rng)
            denoise_step = t + jnp.zeros(shape=broadcast_shape, dtype="i4")
            z = jax.random.normal(self.rng, shape=(*observation.shape[: -1], self.noise_dim)) if t > 0 else 0

            pred = self.policy.predict(x=x, denoise_step=denoise_step, deterministic=True)
            eps = pred.pred_action

            oneover_sqrta = self.ddpm_schedule["oneover_sqrta"]
            ma_over_sqrtmab_inv = self.ddpm_schedule["ma_over_sqrtmab_inv"]
            sqrt_beta_t = self.ddpm_schedule["sqrt_beta_t"]

            y_t = (oneover_sqrta[t] * (y_t - ma_over_sqrtmab_inv[t] * eps)) + (sqrt_beta_t[t] * z)

        self.rng, _ = jax.random.split(self.rng)

        output = GenRLPolicyOutput(pred_action=y_t, info={})
        return output
