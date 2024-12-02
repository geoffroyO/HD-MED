from functools import partial
from typing import Callable, List, NamedTuple, Tuple, Union

from kneed import KneeLocator

import numpy as np

import jax
from jax import Array
from jax import grad, vmap
from jax.lax import scan
import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln, logsumexp
from jax.typing import ArrayLike, DTypeLike

from netket.jax import vmap_chunked

from scipy.linalg import eig
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from .hdem import classif_fun, norm_proj, project, inv_project
from ..em import (
    burnin,
    em_config,
    online_epochs,
    online_step,
    posterior,
    stochastic_step,
)
from ..utils import Bisection, OptStiefel, permute_Ab


class hdstm_stats(NamedTuple):
    s0: ArrayLike
    s1: ArrayLike
    S2: ArrayLike
    s3: ArrayLike
    s4: ArrayLike
    s5: ArrayLike


class hdstm_params(NamedTuple):
    pi: ArrayLike
    mu: ArrayLike
    A: List[ArrayLike]
    b: ArrayLike
    D_tilde: List[ArrayLike]
    nu: ArrayLike


def log_prob(
    y: ArrayLike, params: hdstm_params, config: em_config, weighted: bool = False
) -> Union[Array, Tuple[Array, Array]]:
    y_mu = y - params.mu
    sq_norm = norm_proj(y, params)

    norm = gammaln((params.nu + config.num_features) / 2)

    tmp = jax.tree_map(jnp.log, params.A)
    tmp = jax.tree_map(jnp.sum, tmp)
    tmp = jnp.asarray(tmp)

    norm -= 0.5 * (
        tmp + (config.num_features - jnp.asarray(config.reduction)) * jnp.log(params.b)
    )

    norm -= 0.5 * config.num_features * jnp.log(jnp.pi * params.nu) - gammaln(
        params.nu / 2
    )

    tmp = jax.tree_map(
        lambda x, y, z: jnp.einsum("ij,i->j", x, y) ** 2 / z,
        params.D_tilde,
        list(y_mu),
        params.A,
    )

    tmp = jax.tree_map(jnp.sum, tmp)
    tmp = jnp.asarray(tmp)

    tmp = jnp.log1p(tmp / params.nu + sq_norm / (params.nu * params.b))
    tmp *= -0.5 * (params.nu + config.num_features)

    log_prob = norm + tmp
    weighted_log_prob = log_prob + jnp.log(params.pi)

    log_prob_norm = logsumexp(weighted_log_prob, axis=0)

    if weighted:
        return log_prob_norm, weighted_log_prob
    else:
        return log_prob_norm


@partial(vmap, in_axes=(0, None, None))
def compute_alpha_beta(
    y: ArrayLike, params: hdstm_params, config: em_config
) -> Tuple[Array, Array]:
    y_mu = y - params.mu

    tmp_A = jax.tree_map(
        lambda x, y, z: jnp.einsum("ij,i->j", x, y) ** 2 / z,
        params.D_tilde,
        list(y_mu),
        params.A,
    )
    tmp_A = jax.tree_map(jnp.sum, tmp_A)
    tmp_A = jnp.asarray(tmp_A)

    tmp = params.nu / 2
    alpha = tmp + config.num_features / 2
    beta = tmp + 0.5 * (tmp_A + norm_proj(y, params) / params.b)

    return alpha, beta


def compute_u(alpha: ArrayLike, beta: ArrayLike) -> Array:
    return alpha / beta


def compute_u_tilde(alpha: ArrayLike, beta: ArrayLike) -> Array:
    return digamma(alpha) - jnp.log(beta)


@partial(vmap, in_axes=(0, 0, 0))
def update_s1(y: ArrayLike, t: ArrayLike, u: ArrayLike) -> Array:
    t_u = t * u
    return jnp.einsum("k,i->ki", t_u, y)


def update_S2(
    Y: ArrayLike, T: ArrayLike, U: ArrayLike, stats: hdstm_stats, config: em_config
) -> Array:
    def body(carry, x):
        y, post_y, u_y = x
        post_u_y = post_y * u_y
        yyT = jnp.einsum("i,j->ij", y, y)
        return carry + jnp.einsum("k,ij->kij", post_u_y, yyT), None

    init = 0 * stats.S2
    return scan(body, init, (Y, T, U))[0] / config.mini_batch_size


@partial(vmap, in_axes=(0, 0, 0))
def update_s3(y: ArrayLike, t: ArrayLike, u: ArrayLike) -> Array:
    yTy = jnp.dot(y, y)
    t_u = t * u
    return t_u * yTy


@partial(vmap, in_axes=(0, 0))
def update_s4(t: ArrayLike, u: ArrayLike) -> Array:
    return t * u


@partial(vmap, in_axes=(0, 0))
def update_s5(t: ArrayLike, u_tilde: ArrayLike) -> Array:
    return t * u_tilde


def update_stats(
    Y: ArrayLike,
    step_size: DTypeLike,
    params: hdstm_params,
    stats: hdstm_stats,
    config: em_config,
    log_prob: Callable,
) -> hdstm_stats:
    T = posterior(Y, params, config, log_prob)

    alpha, beta = compute_alpha_beta(Y, params, config)
    U, U_tilde = compute_u(alpha, beta), compute_u_tilde(alpha, beta)
    s0 = stochastic_step(stats.s0, T.mean(axis=0), step_size)
    s1 = stochastic_step(stats.s1, update_s1(Y, T, U).mean(axis=0), step_size)
    S2 = stochastic_step(stats.S2, update_S2(Y, T, U, stats, config), step_size)
    s3 = stochastic_step(stats.s3, update_s3(Y, T, U).mean(axis=0), step_size)
    s4 = stochastic_step(stats.s4, update_s4(T, U).mean(axis=0), step_size)
    s5 = stochastic_step(stats.s5, update_s5(T, U_tilde).mean(axis=0), step_size)
    return hdstm_stats(s0, s1, S2, s3, s4, s5)


def update_pi(stats: hdstm_stats) -> Array:
    return stats.s0 / stats.s0.sum()


def update_mu(stats: hdstm_stats) -> Array:
    return jnp.einsum("ki,k->ki", stats.s1, 1 / stats.s4)


def update_A(params: hdstm_params, stats: hdstm_stats) -> List[Array]:
    mu_mu = jnp.einsum("ki,kj->kij", params.mu, params.mu)
    s4_mu_mu = jnp.einsum("k,kij->kij", stats.s4, mu_mu)
    tmp = stats.S2 + s4_mu_mu - 2 * jnp.einsum("ki,kj->kij", stats.s1, params.mu)
    tmp = jax.tree.map(
        lambda x, y: jnp.einsum("ij,jm->mi", x, y), list(tmp), params.D_tilde
    )
    return jax.tree.map(
        lambda x, y: jnp.einsum("mi,im->m", x, y) + 1e-6, tmp, params.D_tilde
    )


def update_b(params: hdstm_params, stats: hdstm_stats, config: em_config) -> Array:
    mu_mu = jnp.einsum("ki,kj->kij", params.mu, params.mu)
    s4_mu_mu = jnp.einsum("k,kij->kij", stats.s4, mu_mu)
    tmp = stats.S2 + s4_mu_mu - 2 * jnp.einsum("ki,kj->kij", stats.s1, params.mu)
    tmp = jax.tree.map(
        lambda x, y: jnp.einsum("ij,jm->mi", x, y), list(tmp), params.D_tilde
    )
    tmp = jnp.asarray(
        jax.tree.map(lambda x, y: -jnp.einsum("mi,im->", x, y), tmp, params.D_tilde)
    )

    tmp += (
        stats.s3
        + stats.s4 * jnp.einsum("ki,ki->k", params.mu, params.mu)
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
    s4: ArrayLike,
) -> Array:
    mu_mu = jnp.einsum("i,j->ij", mu, mu)
    tmp = 2 * jnp.einsum("i,j->ij", mu, s1) - S2 - s4 * mu_mu
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
    s4: List[ArrayLike],
) -> List[Array]:
    _cost_fun = partial(cost_D_tilde, mu=mu, A=A, b=b, s1=s1, S2=S2, s4=s4)
    _grad_fun = grad(_cost_fun)
    return OptStiefel(D_tilde, _cost_fun, _grad_fun).run()


@partial(
    vmap,
    in_axes=(0, 0),
)
def update_nu(s4: ArrayLike, s5: ArrayLike) -> Array:
    return Bisection(
        fun=lambda x: s5 - s4 - digamma(x / 2) + jnp.log(x / 2) + 1,
        lower=0.001,
        upper=100,
    ).run()


def update_params(
    stats: hdstm_stats, config: em_config, params: hdstm_params, *kwargs
) -> hdstm_params:
    s0 = stats.s0
    s1 = jnp.einsum("ki,k->ki", stats.s1, 1 / stats.s0)
    S2 = jnp.einsum("kij,k->kij", stats.S2, 1 / stats.s0)
    s3 = jnp.einsum("k,k->k", stats.s3, 1 / stats.s0)
    s4 = jnp.einsum("k,k->k", stats.s4, 1 / stats.s0)
    s5 = jnp.einsum("k,k->k", stats.s5, 1 / stats.s0)
    stats_mix = hdstm_stats(s0, s1, S2, s3, s4, s5)

    pi = update_pi(stats_mix)
    mu = update_mu(stats_mix)
    A = update_A(params, stats_mix)
    b = update_b(params, stats_mix, config)

    A, b = permute_Ab(A, b)
    D_tilde = jax.tree_map(
        update_D_tilde,
        params.D_tilde,
        list(mu),
        A,
        list(b),
        list(stats_mix.s1),
        list(stats_mix.S2),
        list(stats_mix.s4),
    )

    nu = update_nu(stats_mix.s4, stats_mix.s5)

    return hdstm_params(pi, mu, A, b, D_tilde, nu)


def initialization(
    X_init: ArrayLike, config: em_config
) -> Tuple[em_config, hdstm_params, hdstm_stats]:
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
    nu = 10.0 * jnp.ones((config.n_components,))

    config = config._replace(reduction=tuple(reductions))
    params = hdstm_params(weights, means, A, b, D, nu)

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
    s4 = jnp.zeros(config.n_components)
    s5 = jnp.zeros(config.n_components)

    stats = hdstm_stats(s0, s1, S2, s3, s4, s5)

    return config, params, stats


class hdstm(NamedTuple):
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
