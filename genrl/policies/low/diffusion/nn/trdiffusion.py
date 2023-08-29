from typing import Callable, Dict

import transformers
from flax import linen as nn
from jax import numpy as jnp

import einops

from genrl.policies.low.dt.nn.third_party import FlaxGPT2ModuleWoTimePosEmb
from genrl.utils.jax_utils.general import create_mlp
from genrl.utils.common.type_aliases import nnOutput


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
            nn.Embed(num_embeddings=self.total_denoise_steps + 99, features=self.embed_dim),
            jnp.sin,
            nn.Dense(self.embed_dim)
        ])

        self.pos_embed = nn.Sequential([
            nn.Dense(self.embed_dim),
            jnp.sin,
            nn.Dense(self.embed_dim)
        ])

        self.emb_ln = nn.LayerNorm(self.embed_dim)
        gpt2_config = transformers.GPT2Config(**self.gpt2_config, n_embd=self.embed_dim)
        self.transformer = FlaxGPT2ModuleWoTimePosEmb(gpt2_config, dtype=jnp.float32)

        self.pred_noise = nn.Dense(self.output_dim)

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
        t: jnp.ndarray,
        maskings: jnp.ndarray,
        deterministic: bool = False
    ) -> nnOutput:

        batch_size = x.shape[0]
        subseq_len = x.shape[1]

        emb_x = self.emb_x(x, deterministic=deterministic)  # [b, l, d]
        emb_y = self.emb_y(y, deterministic=deterministic)  # [b, l, d]
        emb_t = self.emb_t(t)  # [b, l, d]

        emb_x += self.pos_embed(jnp.zeros((batch_size, subseq_len, 1)) + 1.0)
        emb_y += self.pos_embed(jnp.zeros((batch_size, subseq_len, 1)) + 2.0)
        emb_t += self.pos_embed(jnp.zeros((batch_size, subseq_len, 1)) + 3.0)

        stacked_inputs = jnp.stack((emb_x, emb_t, emb_y), axis=1)  # [b, 3, l, d]
        stacked_inputs = einops.rearrange(stacked_inputs, "b c l d -> b l c d")
        stacked_inputs = stacked_inputs.reshape(batch_size, 3 * subseq_len, self.embed_dim)    # [b, 3 * l, d]
        # stacked_inputs = self.emb_ln(stacked_inputs)

        stacked_masks = jnp.stack((maskings, maskings, maskings), axis=1)    # [b, 3, l]
        stacked_masks = einops.rearrange(stacked_masks, "b c l -> b l c")
        stacked_masks = stacked_masks.reshape(batch_size, 3 * subseq_len)

        transformer_outputs = self.transformer(
            hidden_states=stacked_inputs,
            attention_mask=stacked_masks,
            deterministic=deterministic
        )

        x = transformer_outputs["last_hidden_state"]        # [b, 3 * l, d]
        x = x.reshape(batch_size, subseq_len, 3, self.embed_dim)    # [b, l, c, d]
        x = einops.rearrange(x, "b l c d -> b c l d")

        pred = self.pred_noise(x[:, 2])

        return {"pred": pred}
