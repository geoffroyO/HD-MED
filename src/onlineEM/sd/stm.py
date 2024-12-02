from functools import partial
from typing import Callable, NamedTuple, Tuple, Union

import numpy as np

import jax
from jax import Array, vmap

import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln, logsumexp
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
from ..utils import Bisection, trimkmeans


class stm_stats(NamedTuple):
    s0: ArrayLike
    s1: ArrayLike
    S2: ArrayLike
    S2_inv: ArrayLike
    log_det_S2_inv: ArrayLike
    s3: ArrayLike
    s4: ArrayLike


class stm_params(NamedTuple):
    pi: ArrayLike
    mu: ArrayLike
    sigma: ArrayLike
    inv_sigma: ArrayLike
    log_det_inv_sigma: ArrayLike
    nu: ArrayLike


def log_prob(
    y: ArrayLike, params: stm_params, config: em_config, weighted: bool = False
) -> Union[Array, Tuple[Array, Array]]:
    y_mu = y - params.mu

    norm = (
        gammaln((params.nu + config.num_features) / 2)
        + 0.5 * params.log_det_inv_sigma
        - 0.5 * config.num_features * jnp.log(jnp.pi * params.nu)
        - gammaln(params.nu / 2)
    )

    tmp = jnp.einsum("kij,kj->ki", params.inv_sigma, y_mu)
    tmp = jnp.einsum("ki,ki->k", y_mu, tmp)
    tmp = jnp.log1p(tmp / params.nu)
    tmp *= -0.5 * (params.nu + config.num_features)

    log_prob = norm + tmp
    weighted_log_prob = log_prob + jnp.log(params.pi)

    log_prob_norm = logsumexp(weighted_log_prob, axis=0)

    if weighted:
        return log_prob_norm, weighted_log_prob

    else:
        return log_prob_norm


@partial(vmap, in_axes=(0, None, None))
def _compute_alpha_beta(
    y: ArrayLike, params: stm_params, config: em_config
) -> Tuple[Array, Array]:
    y_mu = y - params.mu

    tmp = params.nu / 2
    alpha = tmp + config.num_features / 2

    dist = jnp.einsum("kij,kj->ki", params.inv_sigma, y_mu)
    dist = jnp.einsum("ki,ki->k", y_mu, dist)
    beta = tmp + 0.5 * dist

    return alpha, beta


def _u(alpha: ArrayLike, beta: ArrayLike) -> Array:
    return alpha / beta


def _u_tilde(alpha: ArrayLike, beta: ArrayLike) -> Array:
    return digamma(alpha) - jnp.log(beta)


@partial(vmap, in_axes=(0, 0, 0))
def update_s1(y: ArrayLike, t: ArrayLike, u: ArrayLike) -> Array:
    t_u = t * u
    return jnp.einsum("k,i->ki", t_u, y)


def update_S2(
    Y: ArrayLike, T: ArrayLike, U: ArrayLike, stats: stm_stats, config: em_config
) -> Array:
    def body(carry, x):
        y, post_y, u_y = x
        post_u_y = post_y * u_y
        yyT = jnp.einsum("i,j->ij", y, y)
        return carry + jnp.einsum("k,ij->kij", post_u_y, yyT), None

    init = 0 * stats.S2
    return jax.lax.scan(body, init, (Y, T, U))[0] / config.mini_batch_size


def _smw_update_log_det_S2_inv(
    Y: ArrayLike,
    T: ArrayLike,
    U: ArrayLike,
    step_size: DTypeLike,
    S2_inv: ArrayLike,
    log_det_S2_inv: ArrayLike,
    config: em_config,
) -> Tuple[ArrayLike]:
    alpha = step_size * T / config.mini_batch_size
    alpha = jnp.einsum("ki,ki->ki", alpha, U)

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
    U: ArrayLike,
    step_size: DTypeLike,
    S2: ArrayLike,
    S2_inv: ArrayLike,
    log_det_S2_inv: ArrayLike,
    config: em_config,
) -> jnp.ndarray:
    return jax.lax.cond(
        inv_step_size(step_size) % 5 != 0,
        lambda _: _smw_update_log_det_S2_inv(
            Y, T, U, step_size, S2_inv, log_det_S2_inv, config
        ),
        lambda _: (-jnp.linalg.slogdet(S2)[1], jnp.linalg.inv(S2)),
        None,
    )


@partial(vmap, in_axes=(0, 0))
def update_s3(t: ArrayLike, u: ArrayLike) -> Array:
    return t * u


@partial(vmap, in_axes=(0, 0))
def update_s4(t: ArrayLike, u_tilde: ArrayLike) -> Array:
    return t * u_tilde


def update_stats(
    Y: ArrayLike,
    step_size: DTypeLike,
    params: stm_params,
    stats: stm_stats,
    config: em_config,
    log_prob: Callable,
) -> stm_stats:
    T = posterior(Y, params, config, log_prob)

    alpha, beta = _compute_alpha_beta(Y, params, config)
    U, U_tilde = _u(alpha, beta), _u_tilde(alpha, beta)

    s0 = stochastic_step(stats.s0, T.mean(axis=0), step_size)
    s1 = stochastic_step(stats.s1, update_s1(Y, T, U).mean(axis=0), step_size)
    S2 = stochastic_step(stats.S2, update_S2(Y, T, U, stats, config), step_size)
    log_det_S2_inv, S2_inv = update_log_det_S2_inv(
        Y, T, U, step_size, S2, stats.S2_inv, stats.log_det_S2_inv, config
    )
    s3 = stochastic_step(stats.s3, update_s3(T, U).mean(axis=0), step_size)
    s4 = stochastic_step(stats.s4, update_s4(T, U_tilde).mean(axis=0), step_size)

    return stm_stats(s0, s1, S2, S2_inv, log_det_S2_inv, s3, s4)


def update_pi(stats: stm_stats) -> Array:
    return stats.s0 / stats.s0.sum()


def update_mu(stats: stm_stats) -> Array:
    return jnp.einsum("ki,k->ki", stats.s1, 1 / stats.s3)


def update_sigma(stats: stm_stats) -> Array:
    tmp = jnp.einsum("ki,kj->kij", stats.s1, stats.s1)
    tmp = jnp.einsum("kij,k->kij", tmp, 1 / stats.s3)
    return stats.S2 - tmp


def update_inv_sigma(stats: stm_stats) -> Array:
    tmp = jnp.einsum("ki,k->ki", stats.s1, 1 / stats.s3)

    th1 = jnp.einsum("ki,kj->kij", stats.s1, tmp)
    th1 = jnp.einsum("kij,kjm->kmi", stats.S2_inv, th1)
    th1 = jnp.einsum("kij,kjm->kmi", stats.S2_inv, th1)

    th2 = jnp.einsum("kji,kj->ki", stats.S2_inv, tmp)
    th2 = 1 - jnp.einsum("ki,ki->k", th2, stats.s1)
    th = jnp.einsum("kij,k->kij", th1, 1 / th2)

    return stats.S2_inv + th


def update_log_det_inv_sigma(stats: stm_stats) -> Array:
    tmp = jnp.einsum("ki,k->ki", stats.s1, 1 / stats.s3)
    tmp = jnp.einsum("kji,kj->ki", stats.S2_inv, tmp)
    tmp = -jnp.einsum("ki,ki->k", tmp, stats.s1)
    tmp = -jnp.log1p(tmp)
    return stats.log_det_S2_inv + tmp


@partial(
    vmap,
    in_axes=(
        0,
        0,
    ),
)
def update_nu(s3: ArrayLike, s4: ArrayLike) -> Array:
    return Bisection(
        fun=lambda x: s4 - s3 - digamma(x / 2) + jnp.log(x / 2) + 1,
        lower=0.001,
        upper=100,
    ).run()


def update_params(stats: stm_stats, config: em_config, *kwargs) -> stm_params:
    s0 = stats.s0
    s1 = jnp.einsum("ki,k->ki", stats.s1, 1 / stats.s0)
    S2 = jnp.einsum("kij,k->kij", stats.S2, 1 / stats.s0)
    S2_inv = jnp.einsum("kij,k->kij", stats.S2_inv, stats.s0)
    log_det_S2_inv = stats.log_det_S2_inv + config.num_features * jnp.log(stats.s0)
    s3 = jnp.einsum("k,k->k", stats.s3, 1 / stats.s0)
    s4 = jnp.einsum("k,k->k", stats.s4, 1 / stats.s0)
    stats_mix = stm_stats(s0, s1, S2, S2_inv, log_det_S2_inv, s3, s4)

    pi = update_pi(stats_mix)
    mu = update_mu(stats_mix)
    sigma = update_sigma(stats_mix)
    inv_sigma = update_inv_sigma(stats_mix)
    log_det_inv_sigma = update_log_det_inv_sigma(stats_mix)
    nu = update_nu(stats_mix.s3, stats_mix.s4)

    return stm_params(pi, mu, sigma, inv_sigma, log_det_inv_sigma, nu)


def initialization(
    X_init: ArrayLike, config: em_config
) -> Tuple[em_config, stm_params, stm_stats]:
    clusters, means = trimkmeans(X_init, config.n_components)

    weights = jnp.array(
        [
            (clusters == k).sum() / (clusters != 0).sum()
            for k in range(1, config.n_components + 1)
        ]
    )
    means = jnp.array(means)

    sigma, inv_sigma, log_det_inv_sigma = [], [], []
    for k in range(config.n_components):
        cov = jnp.cov(X_init[clusters == k + 1], rowvar=False)
        sigma.append(cov)
        inv_sigma.append(np.linalg.inv(cov))
        log_det_inv_sigma.append(np.linalg.slogdet(cov)[1])

    sigma = jnp.asarray(sigma)
    inv_sigma = jnp.asarray(inv_sigma)
    log_det_inv_sigma = jnp.asarray(log_det_inv_sigma)

    nu = 10.0 * jnp.ones((config.n_components,))

    params = stm_params(weights, means, sigma, inv_sigma, log_det_inv_sigma, nu)

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
    log_det_S2_inv = jnp.zeros((config.n_components,))
    s3 = jnp.zeros(config.n_components)
    s4 = jnp.zeros(config.n_components)

    stats = stm_stats(s0, s1, S2, S2_inv, log_det_S2_inv, s3, s4)
    return config, params, stats


class stm(NamedTuple):
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
