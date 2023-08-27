from functools import partial
from typing import Dict, Tuple, Union, Optional

import jax
import numpy as np
from jax import numpy as jnp

from genrl.policies.low.base import BaseLowPolicy
from genrl.policies.low.diffusion.arch.mlpdiffusion import MLPDiffusionNN  # Todo: Change to transformer
from genrl.policies.low.diffusion.ddpm_schedule import ddpm_linear_schedule
from genrl.rl.buffers.type_aliases import GenRLBufferSample
from genrl.utils.jax_utils.general import get_basic_rngs
from genrl.utils.jax_utils.model import Model
from genrl.utils.jax_utils.type_aliases import PRNGKey, Params
from genrl.utils.common.type_aliases import PolicyOutput


class DiffusionLowPolicy(BaseLowPolicy):

    def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
        super(DiffusionLowPolicy, self).__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)

        self.total_denoise_steps = cfg["total_denoise_steps"]
        self.noise_dim = cfg["noise_dim"]
        self.ddpm_schedule = ddpm_linear_schedule(**self.cfg["ddpm_schedule"])

    def build(self):
        super(DiffusionLowPolicy, self).build()
        nn_class = MLPDiffusionNN

        nn = nn_class(**self.cfg["nn_cfg"])
        obs = jnp.zeros((1, self.observation_dim))
        act = jnp.zeros((1, self.action_dim))
        time = jnp.array([0], dtype="i4")

        tx = self.optimizer_class(learning_rate=self.cfg["lr"], **self.cfg["optimizer_kwargs"])
        self.rng, rngs = get_basic_rngs(self.rng)

        self.policy_nn = Model.create(
            apply_fn=nn.apply,
            params=nn.init(self.rng, obs, act, time, False),
            tx=tx
        )

    @staticmethod
    @jax.jit
    def policy_forward(
        policy: Model,
        observations: jnp.ndarray,
        y_t: jnp.ndarray,
        denoise_step: jnp.ndarray,
        deterministic: bool = True
    ) -> jnp.ndarray:
        """
        reverse process of diffusion
        :return y_{t-1}
        """
        noise_pred = policy.apply_fn(
            policy.params,
            x=observations,
            y=y_t,
            t=denoise_step,
            deterministic=deterministic
        )

        return noise_pred

    def _update(self, replay_data: GenRLBufferSample) -> Dict:
        self.rng, _ = jax.random.split(self.rng)

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
            noise_pred = policy.apply_fn(
                params,
                x=observations,
                y=y_t,
                t=jnp.repeat(_ts[..., jnp.newaxis], repeats=subseq_len, axis=-1)
            )

            mse_loss = jnp.mean(jnp.mean((noise_pred - noise) ** 2, axis=-1) * masks)

            # If key startswith __ (double underbar), then it is not printed.
            _info = {
                "mse_loss": mse_loss,
                "__noise_pred": noise_pred
            }

            return mse_loss, _info

        grads, info = jax.grad(_loss, has_aux=True)(policy.params)
        new_policy = policy.apply_gradients(grads=grads)

        return new_policy, info

    def _predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> PolicyOutput:
        broadcast_shape = observation.shape[: -1]

        # sample initial noise, y_T ~ Normal(0, 1)
        y_t = jax.random.normal(self.rng, shape=(*observation.shape[: -1], self.noise_dim))
        # denoising chain
        for t in range(self.total_denoise_steps, 0, -1):
            denoise_steps = t + jnp.zeros(shape=broadcast_shape, dtype="i4")
            z = jax.random.normal(self.rng, shape=(*observation.shape[: -1], self.noise_dim)) if t > 0 else 0

            eps = DiffusionLowPolicy.policy_forward(
                policy=self.policy_nn,
                observations=observation,
                y_t=y_t,
                denoise_step=denoise_steps,
                deterministic=True
            )

            oneover_sqrta = self.ddpm_schedule["oneover_sqrta"]
            ma_over_sqrtmab_inv = self.ddpm_schedule["ma_over_sqrtmab_inv"]
            sqrt_beta_t = self.ddpm_schedule["sqrt_beta_t"]

            y_t = (oneover_sqrta[t] * (y_t - ma_over_sqrtmab_inv[t] * eps)) + (sqrt_beta_t[t] * z)

        self.rng, _ = jax.random.split(self.rng)
        return y_t, None
