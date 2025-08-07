from functools import partial
import jax
from jax.typing import ArrayLike
from jax import Array
import jax.numpy as jnp
from typing import Callable, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import em_config, em_params


@partial(jax.vmap, in_axes=(0, None, None, None))
def posterior(
    y: ArrayLike,
    params: "em_params",
    config: "em_config",
    log_prob: Callable[[ArrayLike, "em_params", "em_config"], Tuple[Array, Array]],
) -> Array:
    log_prob_norm, weighted_log_prob = log_prob(y, params, config)
    log_resp = weighted_log_prob - log_prob_norm
    return jnp.exp(log_resp)
