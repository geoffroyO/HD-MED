from functools import partial
from typing import Callable, NamedTuple, List, Tuple, Union

from kneed import KneeLocator

import numpy as np

import jax
from jax import Array, grad, vmap

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike, DTypeLike

from netket.jax import vmap_chunked

from scipy.linalg import eig
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from .hdem import norm_proj, project, inv_project, classif_fun
from ..em import (
    burnin,
    em_config,
    online_epochs,
    online_step,
    posterior,
    stochastic_step,
)
from ..utils import OptStiefel, permute_Ab


class hdgmm_stats(NamedTuple):
    s0: ArrayLike
    s1: ArrayLike
    S2: ArrayLike
    s3: ArrayLike


class hdgmm_params(NamedTuple):
    pi: ArrayLike
    mu: ArrayLike
    A: List[ArrayLike]
    b: ArrayLike
    D_tilde: List[ArrayLike]


def log_prob(
    y: ArrayLike, params: hdgmm_params, config: em_config, weighted: bool = False
) -> Union[Array, Tuple[Array, Array]]:
    y_mu = y - params.mu
    sq_norm = norm_proj(y, params)

    log_prob = config.num_features * jnp.log(2 * jnp.pi)

    tmp = jax.tree.map(jnp.log, params.A)
    tmp = jax.tree.map(jnp.sum, tmp)
    tmp = jnp.asarray(tmp)

    log_prob += tmp + (config.num_features - jnp.asarray(config.reduction)) * jnp.log(
        params.b
    )

    tmp = jax.tree.map(
        lambda x, y, z: jnp.einsum("ij,i->j", x, y) ** 2 / z,
        params.D_tilde,
        list(y_mu),
        params.A,
    )
    tmp = jax.tree.map(jnp.sum, tmp)
    tmp = jnp.asarray(tmp)

    log_prob += tmp + sq_norm / params.b
    log_prob *= -0.5

    weighted_log_prob = log_prob + jnp.log(params.pi)
    log_prob_norm = logsumexp(weighted_log_prob)

    if weighted:
        return log_prob_norm, weighted_log_prob

    else:
        return log_prob_norm


@partial(jax.vmap, in_axes=(0, 0))
def update_s1(y: ArrayLike, t: ArrayLike) -> Array:
    return jnp.einsum("i,k->ik", t, y)


def update_S2(
    Y: ArrayLike, T: ArrayLike, stats: hdgmm_stats, config: em_config
) -> Array:
    def body(carry, x):
        y, post_y = x
        yyT = jnp.einsum("i,j->ij", y, y)
        return carry + jnp.einsum("k,ij->kij", post_y, yyT), None

    return jax.lax.scan(body, 0 * stats.S2, (Y, T))[0] / config.mini_batch_size


@partial(jax.vmap, in_axes=(0, 0))
def update_s3(y: ArrayLike, t: ArrayLike) -> Array:
    return t * jnp.dot(y, y)


def update_stats(
    Y: Array,
    step_size: DTypeLike,
    params: hdgmm_params,
    stats: hdgmm_stats,
    config: em_config,
    log_prob: Callable,
) -> hdgmm_stats:
    T = posterior(Y, params, config, log_prob)

    s0 = stochastic_step(stats.s0, T.mean(axis=0), step_size)
    s1 = stochastic_step(stats.s1, update_s1(Y, T).mean(axis=0), step_size)
    S2 = stochastic_step(stats.S2, update_S2(Y, T, stats, config), step_size)
    s3 = stochastic_step(stats.s3, update_s3(Y, T).mean(axis=0), step_size)
    return hdgmm_stats(s0, s1, S2, s3)


def update_pi(stats: hdgmm_stats) -> Array:
    return stats.s0 / stats.s0.sum()


def update_mu(stats: hdgmm_stats) -> Array:
    return stats.s1


def update_A(params: hdgmm_params, stats: hdgmm_stats) -> List[Array]:
    mu_mu = jnp.einsum("ki,kj->kij", params.mu, params.mu)
    tmp = stats.S2 + mu_mu - 2 * jnp.einsum("ki,kj->kij", stats.s1, params.mu)
    tmp = jax.tree.map(
        lambda x, y: jnp.einsum("ij,jm->mi", x, y), list(tmp), params.D_tilde
    )
    return jax.tree.map(
        lambda x, y: jnp.einsum("mi,im->m", x, y) + 1e-6, tmp, params.D_tilde
    )


def update_b(params: hdgmm_params, stats: hdgmm_stats, config: em_config) -> Array:
    mu_mu = jnp.einsum("ki,kj->kij", params.mu, params.mu)
    tmp = stats.S2 + mu_mu - 2 * jnp.einsum("ki,kj->kij", stats.s1, params.mu)
    tmp = jax.tree.map(
        lambda x, y: jnp.einsum("ij,jm->mi", x, y), list(tmp), params.D_tilde
    )
    tmp = jnp.asarray(
        jax.tree.map(lambda x, y: -jnp.einsum("mi,im->", x, y), tmp, params.D_tilde)
    )

    tmp += (
        stats.s3
        + jnp.einsum("ki,ki->k", params.mu, params.mu)
        - 2 * jnp.einsum("ki,ki->k", stats.s1, params.mu)
    )
    return tmp / (config.num_features - jnp.asarray(config.reduction)) + 1e-6


def cost_D_tilde(
    D_tilde: ArrayLike,
    mu: ArrayLike,
    A: ArrayLike,
    b: ArrayLike,
    s1: ArrayLike,
    S2: ArrayLike,
) -> Array:
    mu_mu = jnp.einsum("i,j->ij", mu, mu)
    tmp = 2 * jnp.einsum("i,j->ij", mu, s1) - S2 - mu_mu
    tmp = jnp.einsum("m,ij->mij", 1 / A - 1 / b, tmp)
    tmp = jnp.einsum("mij,jm->mi", tmp, D_tilde)
    cost = jnp.einsum("mi,im->", tmp, D_tilde)
    return -cost


def update_D_tilde(
    D_tilde: List[ArrayLike],
    mu: List[ArrayLike],
    A: List[ArrayLike],
    b: List[ArrayLike],
    s1: List[ArrayLike],
    S2: List[ArrayLike],
) -> List[Array]:
    _cost_fun = partial(cost_D_tilde, mu=mu, A=A, b=b, s1=s1, S2=S2)
    _grad_fun = grad(_cost_fun)
    return OptStiefel(D_tilde, _cost_fun, _grad_fun).run()


def update_params(
    stats: hdgmm_stats, config: em_config, params: hdgmm_params, *kwargs
) -> hdgmm_params:
    s0 = stats.s0
    s1 = jnp.einsum("ki,k->ki", stats.s1, 1 / stats.s0)
    S2 = jnp.einsum("kij,k->kij", stats.S2, 1 / stats.s0)
    s3 = jnp.einsum("k,k->k", stats.s3, 1 / stats.s0)
    stats_mix = hdgmm_stats(s0, s1, S2, s3)

    pi = update_pi(stats_mix)
    mu = update_mu(stats_mix)
    A = update_A(params, stats_mix)
    b = update_b(params, stats_mix, config)

    A, b = permute_Ab(A, b)

    D_tilde = jax.tree.map(
        update_D_tilde,
        params.D_tilde,
        list(mu),
        A,
        list(b),
        list(stats_mix.s1),
        list(stats_mix.S2),
    )

    return hdgmm_params(pi, mu, A, b, D_tilde)


def initialization(
    X_init: ArrayLike, config: em_config
) -> Tuple[em_config, hdgmm_params, hdgmm_stats]:
    em_model = GaussianMixture(n_components=config.n_components, covariance_type="full")
    em_model.fit(X_init)
    clusters = em_model.predict(X_init)

    weights = jnp.array(em_model.weights_)
    means = jnp.array(em_model.means_)

    A, D, b, reductions = [], [], [], []

    for k in range(config.n_components):
        X_cluster = X_init[clusters == k]
        cov = em_model.covariances_[k]

        # Compute eigenvalues and eigenvectors
        eig_val, eig_vec = eig(cov)
        eig_val, eig_vec = np.real(eig_val), np.real(eig_vec)

        # Sort eigenvalues and eigenvectors in descending order
        argsort_l = np.argsort(-eig_val)
        eig_val, eig_vec = eig_val[argsort_l], eig_vec[:, argsort_l]

        # Estimated reduction
        if config.reduction == ():
            pca = PCA()
            pca.fit(X_cluster)
            explained_ratio = pca.explained_variance_ratio_
            kneedle = KneeLocator(
                range(len(explained_ratio)),
                explained_ratio,
                S=0,
                curve="convex",
                direction="decreasing",
                online=True,
            )
            reduction = kneedle.knee
        else:
            reduction = config.reduction

        eig_val, eig_vec = eig_val[:reduction], eig_vec[:, :reduction]
        reductions.append(reduction)

        D.append(eig_vec)
        A.append(eig_val)
        b.append((jnp.trace(cov) - eig_val.sum()) / (config.num_features - reduction))

    b = jnp.asarray(b)
    config = config._replace(reduction=tuple(reductions))
    params = hdgmm_params(weights, means, A, b, D)

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
    s3 = jnp.zeros(config.n_components)

    stats = hdgmm_stats(s0, s1, S2, s3)

    return config, params, stats


class hdgmm(NamedTuple):
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

    classif: Callable = jax.jit(
        vmap_chunked(
            partial(classif_fun, log_prob=log_prob),
            in_axes=(0, None, None),
            chunk_size=256,
        )
    )
    project: Callable = partial(project, classif_vmaped=classif)
    inv_project: Callable = inv_project

    log_prob: Callable = vmap(log_prob, in_axes=(0, None, None, None))
