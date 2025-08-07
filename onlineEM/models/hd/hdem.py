from typing import List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from typing import Callable
from ...core import em_config
from functools import partial
import numpy as np


class hd_params(NamedTuple):
    mu: Array
    D_tilde: List[Array]


def norm_proj(y: ArrayLike, params: hd_params) -> Array:
    y_mu = y - params.mu
    tmp = jnp.asarray(jax.tree.map(lambda x: jnp.einsum("ij,mj->mi", x, x), params.D_tilde))
    return jnp.linalg.norm(y_mu - jnp.einsum("kmi,ki->km", tmp, y_mu), axis=1) ** 2


@partial(jax.jit, static_argnames=("log_prob",))
def classif_fun(
    y: ArrayLike,
    params: hd_params,
    config: em_config,
    log_prob: Callable[[ArrayLike, hd_params, em_config], Tuple[Array, Array]],
) -> Array:
    log_prob_norm, weighted_log_prob = log_prob(y, params, config)
    log_resp = weighted_log_prob - log_prob_norm
    return jnp.argmax(log_resp, axis=0)


@jax.jit
@partial(jax.vmap, in_axes=(0, None, None))
def project_clf(y: ArrayLike, D_tilde: ArrayLike, mu: ArrayLike) -> Array:
    return D_tilde.T @ (y - mu)


def project(
    signals: ArrayLike, classif: ArrayLike, params: hd_params, config: em_config
) -> Tuple[List[Array], List[Array]]:
    Y_proj_clf, idx_clf_list = [], []

    for clf in range(config.n_components):
        idx_clf = np.asarray(jnp.where(classif == clf)[0])
        Y_clf = signals[idx_clf]
        Y_proj_clf.append(project_clf(Y_clf, params.D_tilde[clf], params.mu[clf]))
        idx_clf_list.append(jnp.asarray(idx_clf))

    return Y_proj_clf, idx_clf_list
