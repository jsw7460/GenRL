from jax import numpy as jnp

from genrl.utils import AttrDict


def ddpm_linear_schedule(beta1: float, beta2: float, total_denoise_step: int) -> AttrDict:
    beta_t = (beta2 - beta1) * jnp.arange(-1, total_denoise_step, dtype=jnp.float32) / (total_denoise_step - 1) + beta1
    beta_t = beta_t.at[0].set(beta1)

    sqrt_beta_t = jnp.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = jnp.log(alpha_t)
    alpha_bar_t = jnp.exp(jnp.cumsum(log_alpha_t, axis=0))

    sqrtab = jnp.sqrt(alpha_bar_t)
    oneover_sqrta = 1 / jnp.sqrt(alpha_t)

    sqrtmab = jnp.sqrt(1 - alpha_bar_t)
    mab_over_sqrtmab_inv = (1 - alpha_bar_t) / sqrtmab

    # shape = (total_denoise_step, )
    return AttrDict(
        alpha_t=alpha_t,
        oneover_sqrta=oneover_sqrta,
        sqrt_beta_t=sqrt_beta_t,
        alpha_bar_t=alpha_bar_t,
        sqrtab=sqrtab,
        sqrtmab=sqrtmab,
        mab_over_sqrtmab_inv=mab_over_sqrtmab_inv
    )
