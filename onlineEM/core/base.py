from typing import NamedTuple, Tuple, Callable
from jax.typing import ArrayLike
from jax import Array
import jax.numpy as jnp


class em_config(NamedTuple):
    n_components: int
    num_features: int

    num_epochs: int
    batch_size: int
    reduction: Array = jnp.asarray([])
    n_first: int = 20000


class em_params(NamedTuple):
    pass


class em_stats(NamedTuple):
    pass


class EM:
    def init(self, X_init: ArrayLike, config: em_config) -> Tuple[em_config, em_params, em_stats]:
        pass

    def burnin(
        self,
        batch: ArrayLike,
        step: int,
        em_config: em_config,
        params: em_params,
        stats: em_stats,
        schedule: Callable[[int], float],
    ) -> em_stats:
        pass

    def update(
        self,
        batch: ArrayLike,
        step: int,
        params: em_params,
        stats: em_stats,
        config: em_config,
        schedule: Callable[[int], float],
    ) -> Tuple[em_params, em_stats]:
        pass

    def batch_log_prob(self, batch: ArrayLike, params: em_params, config: em_config) -> Array:
        pass

    def weighted_log_prob(self, y: ArrayLike, params: em_params, config: em_config) -> Array:
        pass

    def get_blank_states(self) -> Tuple[em_params, em_stats]:
        pass
