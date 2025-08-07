from functools import partial
from typing import Callable, List, NamedTuple, Tuple

import jax
from jax import Array
import jax.numpy as jnp
import jax.scipy.special as jsp
from jax.typing import ArrayLike

from kneed import KneeLocator

import numpy as np

import optax

from scipy.linalg import eig
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


from .hdem import norm_proj
from .utils import permute_Ab
from ...em import posterior
from ...core import EM
from ...utils import OptStiefel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...core import em_config


class hdgmm_stats(NamedTuple):
    s0: Array
    s1: Array
    S2: Array
    s3: Array


class hdgmm_params(NamedTuple):
    pi: Array
    mu: Array
    A: List[Array]
    b: Array
    D_tilde: List[Array]


@jax.jit
def log_prob(y: ArrayLike, params: hdgmm_params, config: "em_config") -> Tuple[Array, Array]:
    sq_norm = norm_proj(y, params)

    log_prob = config.num_features * jnp.log(2 * jnp.pi)
    tmp = jnp.asarray(jax.tree.map(lambda x: jnp.sum(jnp.log(x)), params.A))
    log_prob += tmp + (config.num_features - config.reduction) * jnp.log(params.b)

    tmp = jax.tree.map(
        lambda x, y, z: jnp.einsum("ij,i->j", x, y) ** 2 / z, params.D_tilde, list(y - params.mu), params.A
    )
    tmp = jnp.asarray(jax.tree.map(jnp.sum, tmp))

    # Safe division by b parameter
    b_safe = jnp.maximum(params.b, 1e-8)
    log_prob += tmp + sq_norm / b_safe
    log_prob *= -0.5

    # Safe log of pi to prevent log(0)
    pi_safe = jnp.maximum(params.pi, 1e-8)
    weighted_log_prob = log_prob + jnp.log(pi_safe)
    log_prob_norm = jsp.logsumexp(weighted_log_prob)

    return log_prob_norm, weighted_log_prob


@partial(jax.vmap, in_axes=(0, 0))
def update_s1(y: ArrayLike, t: ArrayLike) -> Array:
    return jnp.einsum("i,k->ik", t, y)


def update_S2(batch: ArrayLike, T: ArrayLike, stats: hdgmm_stats, config: "em_config") -> Array:
    def body(carry, x):
        y, post_y = x
        yyT = jnp.einsum("i,j->ij", y, y)
        return carry + jnp.einsum("k,ij->kij", post_y, yyT), None

    return jax.lax.scan(body, 0 * stats.S2, (batch, T))[0] / config.batch_size


@partial(jax.vmap, in_axes=(0, 0))
def update_s3(y: ArrayLike, t: ArrayLike) -> Array:
    return t * jnp.dot(y, y)


def update_pi(stats: hdgmm_stats) -> Array:
    # Add small epsilon to prevent division by zero
    s0_sum = jnp.maximum(stats.s0.sum(), 1e-8)
    return stats.s0 / s0_sum


def update_mu(stats: hdgmm_stats) -> Array:
    return stats.s1


def update_A(params: hdgmm_params, stats: hdgmm_stats) -> List[Array]:
    mu_mu = jnp.einsum("ki,kj->kij", params.mu, params.mu)
    tmp = stats.S2 + mu_mu - 2 * jnp.einsum("ki,kj->kij", stats.s1, params.mu)
    tmp = jax.tree.map(lambda x, y: jnp.einsum("ij,jm->mi", x, y), list(tmp), params.D_tilde)
    return jax.tree.map(lambda x, y: jnp.einsum("mi,im->m", x, y) + 1e-4, tmp, params.D_tilde)


def update_b(params: hdgmm_params, stats: hdgmm_stats, config: "em_config") -> Array:
    mu_mu = jnp.einsum("ki,kj->kij", params.mu, params.mu)
    tmp = stats.S2 + mu_mu - 2 * jnp.einsum("ki,kj->kij", stats.s1, params.mu)
    tmp = jax.tree.map(lambda x, y: jnp.einsum("ij,jm->mi", x, y), list(tmp), params.D_tilde)
    tmp = jnp.asarray(jax.tree.map(lambda x, y: -jnp.einsum("mi,im->", x, y), tmp, params.D_tilde))

    tmp += stats.s3 + jnp.einsum("ki,ki->k", params.mu, params.mu) - 2 * jnp.einsum("ki,ki->k", stats.s1, params.mu)

    return tmp / (config.num_features - config.reduction) + 1e-4


def cost_D_tilde(D_tilde: ArrayLike, mu: ArrayLike, A: ArrayLike, b: ArrayLike, s1: ArrayLike, S2: ArrayLike) -> Array:
    mu_mu = jnp.einsum("i,j->ij", mu, mu)
    tmp = 2 * jnp.einsum("i,j->ij", mu, s1) - S2 - mu_mu
    # Safe division to prevent NaN
    A_safe = jnp.maximum(A, 1e-8)
    b_safe = jnp.maximum(b, 1e-8)
    tmp = jnp.einsum("m,ij->mij", 1 / A_safe - 1 / b_safe, tmp)
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
    _grad_fun = jax.grad(_cost_fun)

    return OptStiefel(D_tilde, _cost_fun, _grad_fun, max_iter=100).run()


def update_stats(
    batch: ArrayLike,
    step: int,
    params: hdgmm_params,
    stats: hdgmm_stats,
    config: "em_config",
    schedule: Callable[[int], float],
) -> hdgmm_stats:
    T = posterior(batch, params, config, log_prob)
    new_s0 = T.mean(axis=0)
    new_s1 = update_s1(batch, T).mean(axis=0)
    new_S2 = update_S2(batch, T, stats, config)
    new_s3 = update_s3(batch, T).mean(axis=0)

    new_stats = hdgmm_stats(new_s0, new_s1, new_S2, new_s3)
    step_size = schedule(step)

    return optax.incremental_update(stats, new_stats, step_size)


def update_params(params: hdgmm_params, stats: hdgmm_stats, config: "em_config") -> hdgmm_params:
    # Safe division to prevent NaN when stats.s0 approaches zero
    eps = 1e-8
    s0 = stats.s0
    s0_safe = jnp.maximum(stats.s0, eps)
    s1 = jnp.einsum("ki,k->ki", stats.s1, 1 / s0_safe)
    S2 = jnp.einsum("kij,k->kij", stats.S2, 1 / s0_safe)
    s3 = jnp.einsum("k,k->k", stats.s3, 1 / s0_safe)
    stats_mix = hdgmm_stats(s0, s1, S2, s3)

    pi = update_pi(stats_mix)
    mu = update_mu(stats_mix)
    A = update_A(params, stats_mix)
    b = update_b(params, stats_mix, config)

    A, b = permute_Ab(A, b)

    D_tilde = jax.tree.map(update_D_tilde, params.D_tilde, list(mu), A, list(b), list(stats_mix.s1), list(stats_mix.S2))

    return hdgmm_params(pi, mu, A, b, D_tilde)


class HDgmm(EM):
    def init(self, X_init: ArrayLike, config: "em_config") -> Tuple["em_config", hdgmm_params, hdgmm_stats]:
        em_model = GaussianMixture(n_components=config.n_components, covariance_type="full", max_iter=1)
        em_model.fit(X_init)
        clusters = em_model.predict(X_init)

        weights = jnp.asarray(em_model.weights_)
        means = jnp.asarray(em_model.means_)

        A, D, b, reductions = [], [], [], []

        for k in range(config.n_components):
            X_cluster = X_init[clusters == k]
            cov = em_model.covariances_[k]

            # Compute eigenvalues and eigenvectors
            eig_val, eig_vec = eig(cov)
            eig_val, eig_vec = np.real(eig_val), np.real(eig_vec)
            # Ensure positive eigenvalues for numerical stability
            eig_val = np.maximum(eig_val, 1e-6)

            # Sort eigenvalues and eigenvectors in descending order
            argsort_l = np.argsort(-eig_val)
            eig_val, eig_vec = eig_val[argsort_l], eig_vec[:, argsort_l]

            # Estimated reduction
            if len(X_cluster) <= 100:
                reduction = 20
            elif config.reduction.size == 0:
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
                reduction = int(kneedle.knee)
            else:
                reduction = config.reduction

            eig_val, eig_vec = eig_val[:reduction], eig_vec[:, :reduction]
            reductions.append(reduction)

            D.append(eig_vec)
            A.append(eig_val)
            # Safe b parameter computation to prevent negative values
            b_val = (jnp.trace(cov) - eig_val.sum()) / (config.num_features - reduction)
            b.append(jnp.maximum(b_val, 1e-6))

        b = jnp.asarray(b)
        config = config._replace(reduction=jnp.asarray(reductions).astype(jnp.float32))
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

    def burnin(
        self,
        batch: ArrayLike,
        step: int,
        params: hdgmm_params,
        stats: hdgmm_stats,
        em_config: "em_config",
        schedule: Callable[[int], float],
    ) -> hdgmm_stats:
        return update_stats(batch, step, params, stats, em_config, schedule)

    def update(
        self,
        batch: ArrayLike,
        step: int,
        params: hdgmm_params,
        stats: hdgmm_stats,
        em_config: "em_config",
        schedule: Callable[[int], float],
    ) -> Tuple[hdgmm_params, hdgmm_stats]:
        stats = update_stats(batch, step, params, stats, em_config, schedule)
        params = update_params(params, stats, em_config)
        return params, stats

    def batch_log_prob(self, batch: ArrayLike, params: hdgmm_params, config: "em_config") -> Array:
        return jax.vmap(log_prob, in_axes=(0, None, None))(batch, params, config)[1]

    def weighted_log_prob(self, y: ArrayLike, params: hdgmm_params, config: "em_config") -> Array:
        return log_prob(y, params, config)

    def get_blank_states(self) -> Tuple[hdgmm_params, hdgmm_stats]:
        return hdgmm_params, hdgmm_stats
