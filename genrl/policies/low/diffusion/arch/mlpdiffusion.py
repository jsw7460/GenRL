from typing import Callable

from flax import linen as nn
from jax import numpy as jnp

from genrl.utils.jax_utils.general import create_mlp


class MLPDiffusionNN(nn.Module):
    embed_dim: int
    hidden_dim: int
    output_dim: int
    dropout: float
    activation_fn: Callable
    total_denoise_steps: int

    emb_x = None
    emb_y = None
    emb_t = None
    out1 = None
    out2 = None
    out3 = None
    out4 = None

    def setup(self) -> None:
        self.emb_x = create_mlp(
            output_dim=self.embed_dim,
            net_arch=[self.embed_dim,],
            activation_fn=self.activation_fn,
            dropout=self.dropout
        )
        self.emb_y = create_mlp(
            output_dim=self.embed_dim,
            net_arch=[self.embed_dim,],
            activation_fn=self.activation_fn,
            dropout=self.dropout
        )
        self.emb_t = nn.Sequential([
            nn.Embed(num_embeddings=self.total_denoise_steps, features=self.embed_dim),
            jnp.sin,
            nn.Dense(self.embed_dim)
        ])

        self.out1 = create_mlp(
            output_dim=self.hidden_dim,
            net_arch=[self.hidden_dim,],
            activation_fn=self.activation_fn
        )
        self.out2 = create_mlp(
            output_dim=self.hidden_dim,
            net_arch=[self.hidden_dim,],
            activation_fn=self.activation_fn
        )
        self.out3 = create_mlp(
            output_dim=self.hidden_dim,
            net_arch=[self.hidden_dim,],
            activation_fn=self.activation_fn
        )
        self.out4 = create_mlp(
            output_dim=self.output_dim,
            net_arch=[self.hidden_dim,],
            activation_fn=self.activation_fn
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: jnp.ndarray, y: jnp.ndarray, t: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        """
        :param x: observation
        :param y: action
        :param t: denoising step
        :param deterministic
        :return:
        """
        emb_x = self.emb_x(x, deterministic=deterministic)
        emb_y = self.emb_y(y, deterministic=deterministic)
        emb_t = self.emb_t(t)

        in1 = jnp.concatenate((emb_x, emb_y, emb_t), axis=-1)
        out1 = self.out1(in1, deterministic=deterministic)

        in2 = jnp.concatenate((out1 / 1.414, emb_y, emb_t), axis=-1)
        out2 = self.out2(in2, deterministic=deterministic)
        out2 = out2 + out1 / 1.414

        in3 = jnp.concatenate((out2 / 1.414, emb_y, emb_t), axis=-1)
        out3 = self.out3(in3, deterministic=deterministic)
        out3 = out3 + out2 / 1.414

        in4 = jnp.concatenate((out3 / 1.414, y, emb_t), axis=-1)
        out4 = self.out4(in4, deterministic=deterministic)

        return out4
