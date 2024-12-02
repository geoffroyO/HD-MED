from functools import partial
from typing import Callable, NamedTuple, Tuple, Union

import numpy as np

import jax
from jax import Array, vmap

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike, DTypeLike

from ..em import (
    burnin,
    em_config,
    online_epochs,
    online_step,
    posterior,
    stochastic_step,
    inv_step_size,
)
from ..utils import fill_diagonal, trimkmeans


class gmm_stats(NamedTuple):
    s0: ArrayLike
    s1: ArrayLike
    S2: ArrayLike
    S2_inv: ArrayLike
    log_det_S2_inv: ArrayLike


class gmm_params(NamedTuple):
    pi: ArrayLike
    mu: ArrayLike
    covariances: ArrayLike
    precisions: ArrayLike
    log_det_precisions: ArrayLike


def log_prob(
    y: ArrayLike, params: gmm_params, config: em_config, weighted: bool = False
) -> Union[Array, Tuple[Array, Array]]:
    norm = 0.5 * (params.log_det_precisions - config.num_features * jnp.log(2 * jnp.pi))
    diff_tmp = y - params.mu

    tmp = jnp.einsum("kij,kj->ki", params.precisions, diff_tmp)
    log_prob = jnp.einsum("ki,ki->k", diff_tmp, tmp)
    weighted_log_prob = norm - 0.5 * log_prob + jnp.log(params.pi)

    log_prob_norm = logsumexp(weighted_log_prob)

    if weighted:
        return log_prob_norm, weighted_log_prob

    else:
        return log_prob_norm


@partial(vmap, in_axes=(0, 0))
def update_s1(y: ArrayLike, t: ArrayLike) -> Array:
    return jnp.einsum("i,k->ik", t, y)


def update_S2(
    Y: ArrayLike, T: ArrayLike, stats: gmm_stats, config: em_config
) -> ArrayLike:
    def body(carry, x):
        y, post_y = x
        yyT = jnp.einsum("i,j->ij", y, y)
        return carry + jnp.einsum("k,ij->kij", post_y, yyT), None

    init = 0 * stats.S2
    return jax.lax.scan(body, init, (Y, T))[0] / config.mini_batch_size


def _smw_update_log_det_S2_inv(
    Y: ArrayLike,
    T: ArrayLike,
    step_size: DTypeLike,
    S2_inv: ArrayLike,
    log_det_S2_inv: ArrayLike,
    config: em_config,
) -> Tuple[Array]:
    alpha = step_size * T / config.mini_batch_size

    def body_scan(val_curr, params_y):
        log_det_curr, S2_inv_curr = val_curr
        y, alpha_y = params_y

        yyT = jnp.einsum("i,j->ij", y, y)

        alpha_yyT = jnp.einsum("k,ij->kij", alpha_y, yyT)

        th1 = jnp.einsum("kij,kjm->kmi", S2_inv_curr, alpha_yyT)
        th1 = jnp.einsum("kij,kjm->kmi", S2_inv_curr, th1)

        tmp = jnp.einsum("k,i->ki", alpha_y, y)
        th2 = jnp.einsum("kji,kj->ki", S2_inv_curr, tmp)
        th2 = 1 + jnp.einsum("ki,i->k", th2, y)
        tmp = jnp.einsum("kij,k->kij", th1, 1 / th2)
        S2_inv_next = S2_inv_curr - tmp

        tmp = jnp.einsum("k,i->ki", alpha_y, y)
        tmp = jnp.einsum("kji,kj->ki", S2_inv_curr, tmp)
        tmp = -jnp.log1p(jnp.einsum("ki,i->k", tmp, y))
        log_det_next = log_det_curr + tmp

        return (log_det_next, S2_inv_next), None

    S2_inv_0, log_det_0 = (
        S2_inv / (1 - step_size),
        log_det_S2_inv - config.num_features * jnp.log1p(-step_size),
    )
    return jax.lax.scan(body_scan, (log_det_0, S2_inv_0), (Y, alpha))[0]


def update_log_det_S2_inv(
    Y: ArrayLike,
    T: ArrayLike,
    step_size: DTypeLike,
    S2: ArrayLike,
    S2_inv: ArrayLike,
    log_det_S2_inv: ArrayLike,
    config: em_config,
) -> Array:
    return jax.lax.cond(
        inv_step_size(step_size) % 5 != 0,
        lambda _: _smw_update_log_det_S2_inv(
            Y, T, step_size, S2_inv, log_det_S2_inv, config
        ),
        lambda _: (-jnp.linalg.slogdet(S2)[1], jnp.linalg.inv(S2)),
        None,
    )


def update_stats(
    Y: Array,
    step_size: DTypeLike,
    params: gmm_params,
    stats: gmm_stats,
    config: em_config,
    log_prob: Callable,
) -> gmm_stats:
    T = posterior(Y, params, config, log_prob)

    s0 = stochastic_step(stats.s0, T.mean(axis=0), step_size)
    s1 = stochastic_step(stats.s1, update_s1(Y, T).mean(axis=0), step_size)
    S2 = stochastic_step(stats.S2, update_S2(Y, T, stats, config), step_size)
    log_det_S2_inv, S2_inv = update_log_det_S2_inv(
        Y, T, step_size, S2, stats.S2_inv, stats.log_det_S2_inv, config
    )

    return gmm_stats(s0, s1, S2, S2_inv, log_det_S2_inv)


def update_pi(stats: gmm_stats) -> Array:
    return stats.s0 / stats.s0.sum()


def update_mu(stats: gmm_stats) -> Array:
    return stats.s1


def update_covariances(stats: gmm_stats) -> Array:
    tmp = stats.S2 - jnp.einsum("ki,kj->kij", stats.s1, stats.s1)
    diag = jnp.diagonal(tmp, axis1=-2, axis2=-1) + 1e-6
    return fill_diagonal(tmp, diag)


def update_precisions(stats: gmm_stats) -> Array:
    th1 = jnp.einsum("ki,kj->kij", stats.s1, stats.s1)
    th1 = jnp.einsum("kij,kjm->kmi", stats.S2_inv, th1)
    th1 = jnp.einsum("kij,kjm->kmi", stats.S2_inv, th1)

    th2 = jnp.einsum("kji,kj->ki", stats.S2_inv, stats.s1)
    th2 = 1 - jnp.einsum("ki,ki->k", th2, stats.s1)
    th = jnp.einsum("kij,k->kij", th1, 1 / th2)

    return stats.S2_inv + th


def update_log_det_precisions(stats: gmm_stats) -> Array:
    tmp = jnp.einsum("kji,kj->ki", stats.S2_inv, stats.s1)
    tmp = -jnp.einsum("ki,ki->k", tmp, stats.s1)
    tmp = -jnp.log1p(tmp)
    return stats.log_det_S2_inv + tmp


def update_params(stats: gmm_stats, config: em_config, *kwargs) -> gmm_params:
    s0 = stats.s0
    s1 = jnp.einsum("ki,k->ki", stats.s1, 1 / stats.s0)
    S2 = jnp.einsum("kij,k->kij", stats.S2, 1 / stats.s0)
    S2_inv = jnp.einsum("kij,k->kij", stats.S2_inv, stats.s0)
    log_det_S2_inv = stats.log_det_S2_inv + config.num_features * jnp.log(stats.s0)
    stats_mix = gmm_stats(s0, s1, S2, S2_inv, log_det_S2_inv)

    weights = update_pi(stats_mix)
    means = update_mu(stats_mix)
    covariances = update_covariances(stats_mix)
    precisions = update_precisions(stats_mix)
    log_det_precisions = update_log_det_precisions(stats_mix)

    return gmm_params(weights, means, covariances, precisions, log_det_precisions)


def initialization(
    X_init: ArrayLike, config: em_config
) -> Tuple[em_config, gmm_params, gmm_stats]:
    clusters, means = trimkmeans(X_init, config.n_components)

    weights = jnp.array(
        [
            (clusters == k).sum() / (clusters != 0).sum()
            for k in range(1, config.n_components + 1)
        ]
    )
    means = jnp.array(means)

    covariances, precisions, log_det_precisions = [], [], []
    for k in range(config.n_components):
        X_cluster = X_init[clusters == k + 1]
        cov = np.cov(X_cluster, rowvar=False)

        covariances.append(cov)
        precisions.append(np.linalg.inv(cov))
        log_det_precisions.append(np.linalg.slogdet(cov)[1])

    covariances = jnp.asarray(covariances)
    precisions = jnp.asarray(precisions)
    log_det_precisions = jnp.asarray(log_det_precisions)

    params = gmm_params(weights, means, covariances, precisions, log_det_precisions)

    s0 = jnp.zeros(config.n_components)
    s1 = jnp.zeros(
        (
            config.n_components,
            config.num_features,
        )
    )
    S2 = jnp.zeros(
        (
            config.n_components,
            config.num_features,
            config.num_features,
        )
    )
    S2_inv = jnp.zeros(
        (
            config.n_components,
            config.num_features,
            config.num_features,
        )
    )
    log_det_S2_inv = jnp.zeros(config.n_components)

    stats = gmm_stats(s0, s1, S2, S2_inv, log_det_S2_inv)

    return config, params, stats


class gmm(NamedTuple):
    burnin: Callable = partial(
        burnin,
        initialization=initialization,
        log_prob=log_prob,
        update_stats=update_stats,
    )
    online_step: Callable = partial(
        online_step,
        log_prob=log_prob,
        update_params=update_params,
        update_stats=update_stats,
    )
    online_epochs: Callable = partial(
        online_epochs,
        log_prob=log_prob,
        update_params=update_params,
        update_stats=update_stats,
    )
    log_prob: Callable = vmap(log_prob, in_axes=(0, None, None, None))
