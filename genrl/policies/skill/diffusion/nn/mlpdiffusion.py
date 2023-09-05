from functools import partial
from typing import Callable, Type
from typing import Dict, Tuple

import jax
from flax import linen as nn
from jax import numpy as jnp

from genrl.utils.common.base import PolicyNNWrapper
from genrl.utils.common.type_aliases import nnOutput, GenRLPolicyInput, GenRLPolicyOutput
from genrl.utils.jax_utils.general import create_mlp
from genrl.utils.jax_utils.model import Model
from genrl.utils.jax_utils.type_aliases import PRNGKey


class MLPSkillDiffusionNN(nn.Module):
    embed_dim: int
    hidden_dim: int
    output_dim: int
    dropout: float
    activation_fn: Callable
    total_denoise_steps: int
    layernorm: bool = False

    emb_x = None
    emb_y = None
    emb_t = None
    emb_sk = None
    out1 = None
    out2 = None
    out3 = None
    out4 = None

    def setup(self) -> None:
        self.emb_x = create_mlp(
            output_dim=self.embed_dim,
            net_arch=[self.embed_dim],
            activation_fn=self.activation_fn,
            layernorm=True,
            dropout=self.dropout
        )
        self.emb_y = create_mlp(
            output_dim=self.embed_dim,
            net_arch=[self.embed_dim],
            activation_fn=self.activation_fn,
            layernorm=True,
            dropout=self.dropout
        )

        self.emb_sk = create_mlp(
            output_dim=self.embed_dim,
            net_arch=[self.embed_dim],
            activation_fn=self.activation_fn,
            layernorm=True,
            dropout=self.dropout
        )

        self.emb_t = nn.Sequential([
            nn.Dense(self.embed_dim),
            jnp.sin,
            nn.Dense(self.embed_dim)
        ])
        
        self.out1 = create_mlp(
            output_dim=self.hidden_dim,
            net_arch=[self.embed_dim, ],
            layernorm=True,
            activation_fn=nn.gelu
        )
        self.out2 = create_mlp(
            output_dim=self.hidden_dim,
            net_arch=[self.embed_dim, ],
            layernorm=True,
            activation_fn=nn.gelu
        )
        self.out3 = create_mlp(
            output_dim=self.hidden_dim,
            net_arch=[self.embed_dim, ],
            layernorm=True,
            activation_fn=nn.gelu
        )
        self.out4 = create_mlp(
            output_dim=self.output_dim,
            net_arch=[self.hidden_dim],
            layernorm=True,
            activation_fn=nn.gelu
        )

    def __call__(
        self,
        x: jnp.ndarray,  # [b, l, d]
        y: jnp.ndarray,  # [b, l, d]
        sk: jnp.ndarray, # [b, l, d]
        t: jnp.ndarray,  # [b]
        deterministic: bool = False,
        *args,
        **kwargs
    ) -> nnOutput:
        return self.forward(x=x, y=y, sk=sk, t=t, deterministic=deterministic)

    def forward(
        self,
        *,
        x: jnp.ndarray,
        y: jnp.ndarray,
        sk: jnp.ndarray,
        t: jnp.ndarray,
        deterministic: bool = False
    ) -> nnOutput:
        """
        :param x: observation   [b, l, d]
        :param y: action    [b, l, d]
        :param sk: skill    [b, l, d]
        :param t: denoising step    [b, l]
        :param deterministic
        :return:
        """
        t = t[..., jnp.newaxis] / self.total_denoise_steps

        sk = sk / 10

        emb_x = self.emb_x(x, deterministic=deterministic)
        emb_y = self.emb_y(y, deterministic=deterministic)
        emb_sk = self.emb_sk(sk, deterministic=deterministic)
        emb_t = self.emb_t(t)

        in1 = jnp.concatenate((emb_y, emb_x, emb_sk, emb_t), axis=-1)
        out1 = self.out1(in1, deterministic=deterministic)

        in2 = jnp.concatenate((out1 / 1.414, emb_y, emb_t), axis=-1)
        out2 = self.out2(in2, deterministic=deterministic)
        out2 = out2 + out1 / 1.414

        in3 = jnp.concatenate((out2 / 1.414, emb_y, emb_t), axis=-1)
        out3 = self.out3(in3, deterministic=deterministic)
        out3 = out3 + out2 / 1.414

        in4 = jnp.concatenate((out3, emb_y, emb_sk, emb_t), axis=-1)
        out4 = self.out4(in4)

        return {"pred": out4, "emb_sk": emb_sk, "emb_x": emb_x, "emb_y": emb_y, "emb_t": emb_t}


class MLPSkillDiffusion(PolicyNNWrapper):

    def __init__(self, seed: int, cfg: Dict):
        super(MLPSkillDiffusion, self).__init__(seed=seed, cfg=cfg)
        self.ddpm_schedule = cfg["ddpm_schedule"]
        self.skill_dim = cfg["skill_dim"]

    def get_nn_class(self) -> Type[nn.Module]:
        return MLPSkillDiffusionNN

    def _predict(
        self,
        x: GenRLPolicyInput,
        denoise_step: jnp.ndarray = None,
        deterministic: bool = True,
        *args,
        **kwargs
    ) -> GenRLPolicyOutput:
        pred = self.nn_forward(
            rng=self.rng,
            policy_nn=self.policy_nn,
            observations=x.observations,
            y_t=x.actions,
            skill=x.sem_skills,
            denoise_step=denoise_step,
            deterministic=deterministic,
        )
        pred_action = pred.pop("pred")
        return GenRLPolicyOutput(pred_action=pred_action, info=pred)

    def get_init_arrays(self) -> Tuple[jnp.ndarray, ...]:
        obs = jnp.zeros((1, 1, self.observation_dim))
        act = jnp.zeros((1, 1, self.action_dim))
        skill = jnp.zeros((1, 1, self.skill_dim))
        time = jnp.zeros((1, 1), dtype="i4")
        return obs, act, skill, time

    @staticmethod
    @partial(jax.jit, static_argnames=("deterministic",))
    def nn_forward(
        rng: PRNGKey,
        policy_nn: Model,
        observations: jnp.ndarray,
        y_t: jnp.ndarray,
        skill: jnp.ndarray,
        denoise_step: jnp.ndarray,
        deterministic: bool = True,
    ) -> nnOutput:
        rng, dropout_key = jax.random.split(rng)

        noise_pred = policy_nn.apply_fn(
            policy_nn.params,
            x=observations,
            y=y_t,
            sk=skill,
            t=denoise_step,
            deterministic=deterministic,
            rngs={"dropout": dropout_key}
        )
        return noise_pred
