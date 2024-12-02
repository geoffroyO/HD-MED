from functools import partial
import sys
from tqdm import tqdm, trange
from typing import Callable, Iterator, NamedTuple, Tuple

import jax
from jax import Array

import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike


class em_config(NamedTuple):
    n_components: int
    num_features: int

    num_epochs: int
    mini_batch_size: int
    reduction: Tuple[int] = ()
    n_first: int = 20000


class em_params(NamedTuple):
    pass


class em_stats(NamedTuple):
    pass


def stochastic_step(
    curr_value: ArrayLike, next_value: ArrayLike, step_size: DTypeLike
) -> Array:
    return step_size * next_value + (1 - step_size) * curr_value


def step_size(k: DTypeLike) -> float:
    return (1 - 10e-10) * (k + 2) ** (-6 / 10)


def inv_step_size(step: DTypeLike) -> int:
    return (((1 - 10e-10) / step) ** (10 / 6) - 2).astype(int)


@partial(jax.vmap, in_axes=(0, None, None, None))
def posterior(
    y: ArrayLike, params: em_params, config: em_config, log_prob: Callable
) -> Array:
    log_prob_norm, weighted_log_prob = log_prob(y, params, config, weighted=True)
    log_resp = weighted_log_prob - log_prob_norm
    return jnp.exp(log_resp)


def bic(X: ArrayLike, params: em_params, config: em_config, model: NamedTuple) -> float:
    def size(x):
        if isinstance(x, (tuple, list)):
            return sum(size(y) for y in x)
        else:
            return x.size

    n = X.shape[0]
    log_prob_norm = model.log_prob(X, params, config, False).sum()

    num_params = sum(size(array) for array in params)
    return -2 * log_prob_norm + num_params * jnp.log(n)


@partial(jax.jit, static_argnames=("config", "log_prob", "update_stats"))
def burnin_step(
    Y: ArrayLike,
    step: DTypeLike,
    params: em_params,
    stats: em_stats,
    config: em_config,
    log_prob: Callable,
    update_stats: Callable,
):
    return update_stats(Y, step, params, stats, config, log_prob)


def burnin(
    X: Iterator[ArrayLike],
    config: em_config,
    initialization: Callable,
    log_prob: Callable,
    update_stats: Callable,
) -> Tuple[em_params, em_stats]:
    X_init = jnp.concat(
        [e for k, e in enumerate(X) if k <= config.n_first // config.mini_batch_size]
    )
    config, params, stats = initialization(X_init, config)
    del X_init

    for step in trange(
        2 * config.num_features,
        desc="Burn-in",
        colour="green",
        file=sys.stdout,
        ncols=100,
    ):
        Y = next(iter(X))
        stats = burnin_step(
            Y, step_size(step), params, stats, config, log_prob, update_stats
        )

    return config, params, stats


@partial(
    jax.jit, static_argnames=("config", "log_prob", "update_params", "update_stats")
)
def online_step(
    Y: ArrayLike,
    step: DTypeLike,
    params: em_params,
    stats: em_stats,
    config: em_config,
    log_prob: Callable,
    update_params: Callable,
    update_stats: Callable,
) -> Tuple[em_params, em_stats]:
    stats = update_stats(Y, step, params, stats, config, log_prob)
    params = update_params(stats, config, params)
    return params, stats


def compute_factor_matrix(config, params):
    identities = list(jax.tree_map(lambda x: jnp.eye(x), config.reduction))
    W_prec = jax.tree_map(lambda x, y: x * y, list(params.b), identities)
    return jax.tree_map(
        lambda x, y, z: z @ jnp.sqrt(jnp.diag(x) - y), params.A, W_prec, params.D_tilde
    )


def online_epochs(
    X: Iterator[ArrayLike],
    params: em_params,
    stats: em_stats,
    config: em_config,
    log_prob: Callable,
    update_params: Callable,
    update_stats: Callable,
) -> Tuple[em_params, em_stats]:
    step = 0

    for epoch in range(config.num_epochs):
        for Y in tqdm(
            X,
            desc=f"Epoch {epoch+1} / {config.num_epochs}",
            colour="green",
            file=sys.stdout,
            ncols=100,
        ):
            params, stats = online_step(
                Y,
                step_size(step),
                params,
                stats,
                config,
                log_prob,
                update_params,
                update_stats,
            )
            step += 1

    return params, stats
