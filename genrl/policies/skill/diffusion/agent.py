from functools import partial
from typing import Dict, Tuple

import jax
from hydra.utils import get_class
from jax import numpy as jnp

from dataclasses import asdict

from genrl.policies.base import BasePolicy
from genrl.policies.low.diffusion.ddpm_schedule import DiffusionBetaScheduler
from genrl.rl.buffers.type_aliases import GenRLBufferSample
from genrl.utils.jax_utils.model import Model
from genrl.utils.jax_utils.type_aliases import PRNGKey, Params
from genrl.utils.common.type_aliases import GenRLPolicyInput, GenRLPolicyOutput
from genrl.utils.common.base import PolicyNNWrapper


class DiffusionSkillPolicy(BasePolicy):

    def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
        self.total_denoise_steps = cfg["total_denoise_steps"]
        self.noise_dim = cfg["noise_dim"]
        self.ddpm_scheduler = DiffusionBetaScheduler(**cfg["ddpm_schedule"])
        self.ddpm_schedule = self.ddpm_scheduler.schedule()
        super(DiffusionSkillPolicy, self).__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)

        self.skill_dim = cfg["skill_dim"]

    def build(self) -> None:
        super(DiffusionSkillPolicy, self).build()

        policy_cls = get_class(self.cfg["policy"]["target"])
        policy_cfg = self.cfg["policy"]["cfg"]
        policy_cfg.update({"ddpm_schedule": asdict(self.ddpm_schedule)})

        self.policy = policy_cls(seed=self.seed, cfg=policy_cfg)    # type: PolicyNNWrapper
        self.policy.build()

    def _update(self, replay_data: GenRLBufferSample) -> Dict:
        new_policy, info = self.update_diffusion(
            rng=self.rng,
            policy=self.policy_nn,
            observations=replay_data.observations,
            actions=replay_data.actions,
            skills=replay_data.sem_skills,
            masks=replay_data.masks,
            ddpm_schedule=asdict(self.ddpm_schedule),
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
        skills: jnp.ndarray,
        masks: jnp.ndarray,  # [b, l]

        ddpm_schedule: Dict,
        noise_dim: int,
        total_denoise_steps: int,
    ) -> Tuple[Model, Dict]:
        rng, dropout_key = jax.random.split(rng)
        batch_size = observations.shape[0]
        subseq_len = observations.shape[1]

        # sample noise
        noise = jax.random.normal(rng, shape=(batch_size, subseq_len, noise_dim))   # [b, l, d]
        # noise = jnp.repeat(noise[:, jnp.newaxis, ...], repeats=subseq_len, axis=1)  # [b, l, d]

        # add noise to clean target actions
        _ts = jax.random.randint(rng, shape=(batch_size, subseq_len), minval=1, maxval=total_denoise_steps + 1)

        sqrtab = ddpm_schedule["sqrtab"][_ts][..., jnp.newaxis]      # [b, 1, 1]
        sqrtmab = ddpm_schedule["sqrtmab"][_ts][..., jnp.newaxis]    # [b, 1, 1]

        y_t = sqrtab * actions + sqrtmab * noise

        # use diffusion model to predict noise
        def _loss(params: Params) -> Tuple[jnp.ndarray, Dict]:
            pred = policy.apply_fn(
                params,
                x=observations,
                y=y_t,
                sk=skills,
                t=_ts,  # [b, l]
                maskings=masks,
                rngs={"dropout": dropout_key}
            )
            noise_pred = pred["pred"]

            noise_pred = noise_pred.reshape(-1, noise_dim) * masks.reshape(-1, 1)
            noise_targ = noise.reshape(-1, noise_dim) * masks.reshape(-1, 1)

            mse_loss = jnp.sum(jnp.mean((noise_pred - noise_targ) ** 2, axis=-1)) / jnp.sum(masks)

            # If key startswith __ (double underbar), then it is not printed.
            _info = {
                "denoise_mse_loss": mse_loss,
                "__noise_pred": noise_pred,
                "__noise_targ": noise_targ,
                "__ts": _ts,
                "__sqrtab": sqrtab,
                "__sqrtmab": sqrtmab,
            }

            return mse_loss, _info

        grads, info = jax.grad(_loss, has_aux=True)(policy.params)
        new_policy = policy.apply_gradients(grads=grads)

        return new_policy, info

    def _predict(self, x: GenRLPolicyInput) -> GenRLPolicyOutput:

        observation = x.observations
        batch_size = observation.shape[0]
        subseq_len = observation.shape[1]
        broadcast_shape = observation.shape[: -1]

        oneover_sqrta = self.ddpm_schedule.oneover_sqrta
        ma_over_sqrtmab_inv = self.ddpm_schedule.ma_over_sqrtmab_inv
        sqrt_beta_t = self.ddpm_schedule.sqrt_beta_t

        # sample initial noise, y_T ~ Normal(0, 1)
        y_t = jax.random.normal(self.rng, shape=(batch_size, subseq_len, self.noise_dim))
        # y_t = jnp.repeat(y_t[:, jnp.newaxis, ...], repeats=subseq_len, axis=1)  # [b, l, d]

        # denoising chain
        for t in range(self.total_denoise_steps, 0, -1):
            self.rng, _ = jax.random.split(self.rng)
            denoise_step = t + jnp.zeros(shape=broadcast_shape, dtype="i4")
            z = jax.random.normal(self.rng, shape=(*observation.shape[: -1], self.noise_dim)) if t > 0 else 0
            
            # z = jax.random.normal(self.rng, shape=(*observation.shape[: -1], self.noise_dim))
            policy_input = GenRLPolicyInput(
                observations=x.observations,
                actions=y_t,
                sem_skills=x.sem_skills,
                timesteps=x.timesteps,
                masks=x.masks,
                rewards=x.rewards,
                terminations=x.terminations
            )

            pred = self.policy.predict(
                x=policy_input,
                denoise_step=denoise_step,
                deterministic=True
            )
            eps = pred.pred_action

            y_t = oneover_sqrta[t] * (y_t - ma_over_sqrtmab_inv[t] * eps) + (sqrt_beta_t[t] * z)
            y_t = jnp.clip(y_t, -2.0, 2.0)

        output = GenRLPolicyOutput(pred_action=y_t, info={})
        return output
