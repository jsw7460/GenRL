from functools import partial
from typing import Callable, Dict, Type

import einops
import jax
import transformers
from flax import linen as nn
from jax import numpy as jnp

from genrl.policies.low.dt.nn.third_party import FlaxGPT2ModuleWoTimePosEmb
from genrl.utils.common.base import PolicyNNWrapper
from genrl.utils.common.type_aliases import nnOutput, GenRLPolicyInput, GenRLPolicyOutput
from genrl.utils.jax_utils.general import create_mlp
from genrl.utils.jax_utils.model import Model
from genrl.utils.jax_utils.type_aliases import PRNGKey


class TransformerDiffusionNN(nn.Module):
    gpt2_config: Dict
    embed_dim: int
    hidden_dim: int
    output_dim: int
    dropout: float
    activation_fn: Callable
    total_denoise_steps: int

    emb_x = None
    emb_y = None
    emb_t = None
    emb_ln = None
    pos_embed = None
    transformer = None
    pred_noise = None

    def setup(self) -> None:
        self.emb_x = create_mlp(
            output_dim=self.embed_dim,
            net_arch=[self.hidden_dim, self.hidden_dim],
            activation_fn=self.activation_fn,
            dropout=self.dropout
        )
        self.emb_y = create_mlp(
            output_dim=self.embed_dim,
            net_arch=[self.hidden_dim, self.hidden_dim],
            activation_fn=self.activation_fn,
            dropout=self.dropout
        )
        self.emb_t = nn.Sequential([
            nn.Dense(self.embed_dim),
            jnp.sin,
            nn.Dense(self.embed_dim)
        ])
        # self.emb_t = nn.Sequential([
        #     nn.Embed(self.total_denoise_steps + 1, self.embed_dim),
        #     jnp.sin,
        #     nn.Dense(self.embed_dim)
        # ])

        self.pos_embed = nn.Sequential([
            nn.Dense(self.embed_dim),
            jnp.sin,
            nn.Dense(self.embed_dim)
        ])

        self.emb_ln = nn.LayerNorm(self.embed_dim)
        gpt2_config = transformers.GPT2Config(**self.gpt2_config, n_embd=self.embed_dim)
        self.transformer = FlaxGPT2ModuleWoTimePosEmb(gpt2_config, dtype=jnp.float32)

        self.pred_noise = create_mlp(
            output_dim=self.output_dim,
            net_arch=[self.hidden_dim, self.hidden_dim],
            activation_fn=nn.gelu,
            dropout=self.dropout
        )

    def __call__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        t: jnp.ndarray,
        maskings: jnp.ndarray,
        deterministic: bool = False,
        **kwargs
    ) -> nnOutput:
        return self.forward(x=x, y=y, t=t, maskings=maskings, deterministic=deterministic, **kwargs)

    def forward(
        self,
        *,
        x: jnp.ndarray,
        y: jnp.ndarray,
        t: jnp.ndarray,  # This is denoise step. Not a trajectory timestep.
        maskings: jnp.ndarray,
        deterministic: bool = False
    ) -> nnOutput:
        """
        :param x:   [b, l, d]
        :param y:   [b, l, d]
        :param t:   [b, l]
        :param maskings:
        :param deterministic:
        :return:
        """

        batch_size = x.shape[0]
        subseq_len = x.shape[1]

        t = t[..., jnp.newaxis] / self.total_denoise_steps

        emb_x = self.emb_x(x, deterministic=deterministic)  # [b, l, d]
        emb_y = self.emb_y(y, deterministic=deterministic)  # [b, l, d]
        emb_t = self.emb_t(t)  # [b, l, d]

        emb_x += (self.pos_embed(jnp.zeros((batch_size, subseq_len, 1)) + 1.0) + emb_t)
        emb_y += (self.pos_embed(jnp.zeros((batch_size, subseq_len, 1)) + 2.0) + emb_t)
        emb_t += (self.pos_embed(jnp.zeros((batch_size, subseq_len, 1)) + 3.0) + emb_t)

        # stacked_inputs = jnp.stack((emb_x, emb_t, emb_y), axis=1)  # [b, 3, l, d]
        # stacked_inputs = jnp.stack((emb_x, emb_y, emb_t), axis=1)  # [b, 3, l, d]
        stacked_inputs = jnp.stack((emb_x, emb_y, emb_t), axis=1)  # [b, 3, l, d]

        stacked_inputs = einops.rearrange(stacked_inputs, "b c l d -> b l c d")
        stacked_inputs = stacked_inputs.reshape(batch_size, 3 * subseq_len, self.embed_dim)  # [b, 3 * l, d]
        stacked_inputs = self.emb_ln(stacked_inputs)

        # y_maskings = jnp.copy(maskings)
        # t_maskings = jnp.copy(maskings)

        stacked_masks = jnp.stack((maskings, maskings, maskings), axis=1)  # [b, 3, l]
        stacked_masks = einops.rearrange(stacked_masks, "b c l -> b l c")
        stacked_masks = stacked_masks.reshape(batch_size, 3 * subseq_len)

        transformer_outputs = self.transformer(
            hidden_states=stacked_inputs,
            attention_mask=stacked_masks,
            deterministic=deterministic
        )

        x = transformer_outputs["last_hidden_state"]  # [b, 3 * l, d]
        x = x.reshape(batch_size, subseq_len, 3, self.embed_dim)  # [b, l, c, d]
        x = einops.rearrange(x, "b l c d -> b c l d")

        pred = self.pred_noise(x[:, 2])
        return {"pred": pred}


class TransformerDiffusion(PolicyNNWrapper):

    def __init__(self, seed: int, cfg: Dict):
        super(TransformerDiffusion, self).__init__(seed=seed, cfg=cfg)
        self.ddpm_schedule = cfg["ddpm_schedule"]

    def get_nn_class(self) -> Type[nn.Module]:
        return TransformerDiffusionNN

    def get_init_arrays(self):
        obs = jnp.zeros((1, 1, self.observation_dim))
        act = jnp.zeros((1, 1, self.action_dim))
        time = jnp.zeros((1, 1), dtype="i4")
        mask = jnp.zeros((1, 1), dtype="i4")
        return obs, act, time, mask

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
            denoise_step=denoise_step,
            maskings=x.masks,
            deterministic=deterministic,
        )
        pred_action = pred.pop("pred")
        return GenRLPolicyOutput(pred_action=pred_action, info=pred)

    @staticmethod
    @partial(jax.jit, static_argnames=("deterministic",))
    def nn_forward(
        rng: PRNGKey,
        policy_nn: Model,
        observations: jnp.ndarray,
        y_t: jnp.ndarray,
        denoise_step: jnp.ndarray,
        maskings: jnp.ndarray,
        deterministic: bool = True,
    ) -> nnOutput:
        rng, dropout_key = jax.random.split(rng)

        noise_pred = policy_nn.apply_fn(
            policy_nn.params,
            x=observations,
            y=y_t,
            t=denoise_step,
            maskings=maskings,
            deterministic=deterministic,
            rngs={"dropout": dropout_key}
        )
        return noise_pred
