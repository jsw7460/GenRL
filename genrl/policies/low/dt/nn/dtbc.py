from functools import partial
from typing import Callable, Dict, Type, Tuple

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
from genrl.utils.jax_utils.scaler import Scaler


class DecisionTransformerNN(nn.Module):
    gpt2_config: Dict
    embed_dim: int
    hidden_dim: int
    output_dim: int
    dropout: float
    activation_fn: Callable
    act_scale: float
    max_ep_len: int

    emb_time = None
    emb_obs = None
    emb_rew = None
    emb_act = None
    emb_ln = None
    transformer = None
    pred_act = None

    def setup(self) -> None:
        self.emb_time = nn.Embed(self.max_ep_len, self.hidden_dim)
        self.emb_obs = nn.Dense(self.hidden_dim)
        self.emb_rew = nn.Dense(self.hidden_dim)
        self.emb_act = nn.Dense(self.hidden_dim)
        self.emb_ln = nn.LayerNorm(self.hidden_dim)

        gpt2_config = transformers.GPT2Config(**self.gpt2_config, n_embd=self.hidden_dim)
        self.transformer = FlaxGPT2ModuleWoTimePosEmb(gpt2_config, dtype=jnp.float32)

        pred_act = create_mlp(
            output_dim=self.output_dim,
            net_arch=[],
            squash_output=True
        )
        self.pred_act = Scaler(base_model=pred_act, scale=self.act_scale)

    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        timesteps: jnp.ndarray,
        maskings: jnp.ndarray,
        deterministic: bool = False,
        **kwargs
    ) -> nnOutput:
        return self.forward(
            observations=observations,
            actions=actions,
            rewards=rewards,
            timesteps=timesteps,
            maskings=maskings,
            deterministic=deterministic,
        )

    def forward(
        self,
        *,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        timesteps: jnp.ndarray,
        maskings: jnp.ndarray,
        deterministic: bool = False,
    ) -> nnOutput:

        batch_size = observations.shape[0]
        subseq_len = observations.shape[1]

        observations_emb = self.emb_obs(observations)
        actions_emb = self.emb_act(actions)

        rewards = rewards[..., jnp.newaxis]
        rewards_emb = self.emb_rew(rewards)
        timesteps_emb = self.emb_time(timesteps)

        observations_emb = observations_emb + timesteps_emb
        actions_emb = actions_emb + timesteps_emb
        rewards_emb = rewards_emb + timesteps_emb

        # jax.debug.print("1 {x}", x=observations_emb.shape)
        # jax.debug.print("2 {x}", x=actions_emb.shape)
        # jax.debug.print("3 {x}", x=rewards_emb.shape)

        # this makes the sequence look like (s_1, r_1, a_1, s_2, r_2, a_2, ...)
        # which works nice in an autoregressive sense since observations predict actions
        stacked_inputs = jnp.stack((observations_emb, rewards_emb, actions_emb), axis=1)  # [b, 3, l, d]

        # stacked_inputs = jnp.stack((observations_emb, observations_emb, observations_emb), axis=1)  # [b, 3, l, d]

        stacked_inputs = einops.rearrange(stacked_inputs, "b c l d -> b l c d")  # [b, l, 3, d]
        stacked_inputs = stacked_inputs.reshape(batch_size, 3 * subseq_len, self.hidden_dim)  # [b, 3 * l, d]
        stacked_inputs = self.emb_ln(stacked_inputs)

        stacked_masks = jnp.stack((maskings, maskings, maskings), axis=1)  # [b, 3, l]
        stacked_masks = einops.rearrange(stacked_masks, "b c l -> b l c")
        stacked_masks = stacked_masks.reshape(batch_size, 3 * subseq_len)

        transformer_outputs = self.transformer(
            hidden_states=stacked_inputs,
            attention_mask=stacked_masks,
            deterministic=deterministic
        )
        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        # observations (0), rewards(1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, subseq_len, 3, self.hidden_dim)
        x = einops.rearrange(x, "b l c d -> b c l d")

        action_preds = self.pred_act(x[:, 1])

        return {"pred": action_preds}


class DecisionTransformer(PolicyNNWrapper):

    def __init__(self, seed: int, cfg: Dict):
        super(DecisionTransformer, self).__init__(seed=seed, cfg=cfg)

    def get_nn_class(self):
        return DecisionTransformerNN

    def get_init_arrays(self) -> Tuple[jnp.ndarray, ...]:
        obs = jnp.zeros((1, 1, self.observation_dim))
        act = jnp.zeros((1, 1, self.action_dim))
        rewards = jnp.zeros((1, 1))
        time = jnp.zeros((1, 1), dtype="i4")
        mask = jnp.zeros((1, 1), dtype="i4")
        return obs, act, rewards, time, mask

    def _predict(self, x: GenRLPolicyInput, deterministic: bool = True, *args, **kwargs) -> GenRLPolicyOutput:
        pred = self.nn_forward(
            rng=self.rng,
            policy_nn=self.policy_nn,
            observations=x.observations,
            actions=x.actions,
            rewards=x.rewards,
            timesteps=x.timesteps,
            maskings=x.masks,
            deterministic=deterministic
        )
        # pred_action = pred.pop("pred")[:, -1, ...]      # Use only last actions
        pred_action = pred.pop("pred")  # Use only last actions
        return GenRLPolicyOutput(pred_action=pred_action, info=pred)

    @staticmethod
    @partial(jax.jit, static_argnames=("deterministic",))
    def nn_forward(
        rng: PRNGKey,
        policy_nn: Model,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        timesteps: jnp.ndarray,
        maskings: jnp.ndarray,
        deterministic: bool = False
    ):
        rng, dropout_key = jax.random.split(rng)

        action_pred = policy_nn.apply_fn(
            policy_nn.params,
            observations=observations,
            actions=actions,
            rewards=rewards,
            timesteps=timesteps,
            maskings=maskings,
            deterministic=deterministic,
            rngs={"dropout": dropout_key}
        )

        return action_pred