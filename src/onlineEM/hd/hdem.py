from typing import Callable, NamedTuple, Tuple, List

import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike

from netket.jax import vmap_chunked

from ..em import em_config

from functools import partial


class hd_params(NamedTuple):
    mu: Array
    D_tilde: Array
    pass


def norm_proj(y: ArrayLike, params: hd_params) -> Array:
    y_mu = y - params.mu
    tmp = jax.tree_map(lambda x: jnp.einsum("ij,mj->mi", x, x), params.D_tilde)
    tmp = jnp.asarray(tmp)
    return jnp.linalg.norm(y_mu - jnp.einsum("kmi,ki->km", tmp, y_mu), axis=1) ** 2


def classif_fun(
    y: ArrayLike, params: hd_params, config: em_config, log_prob: Callable
) -> Array:
    log_prob_norm, weighted_log_prob = log_prob(y, params, config, weighted=True)
    log_resp = weighted_log_prob - log_prob_norm
    return jnp.argmax(log_resp)


@jax.jit
@partial(vmap_chunked, in_axes=(0, None, None), chunk_size=256)
def project_clf(y: ArrayLike, D_tilde: ArrayLike, mu: ArrayLike) -> Array:
    return D_tilde.T @ (y - mu)


def project(
    Y: ArrayLike, params: hd_params, config: em_config, classif_vmaped: Callable
) -> Tuple[List[Array], List[Array]]:
    classif = classif_vmaped(Y, params, config)

    Y_proj, idx_clf_lst = [], []

    for clf in range(config.n_components):
        idx_clf = jnp.where(classif == clf)[0]
        Y_clf = Y[idx_clf]

        Y_proj.append(project_clf(Y_clf, params.D_tilde[clf], params.mu[clf]))
        idx_clf_lst.append(idx_clf)
    return Y_proj, idx_clf_lst


@jax.jit
@partial(vmap_chunked, in_axes=(0, None, None), chunk_size=256)
def inv_project_clf(y: ArrayLike, D_tilde: ArrayLike, mu: ArrayLike) -> Array:
    return D_tilde @ y + mu


def inv_project(
    Y_proj: List[Array], idx_clf_lst: List[Array], params: hd_params, config: em_config
) -> Array:
    Y_inv_proj = jnp.zeros((sum(map(len, Y_proj)), config.num_features))
    for clf, (Y_proj_clf, idx_clf) in enumerate(zip(Y_proj, idx_clf_lst)):
        Y_inv_proj = Y_inv_proj.at[idx_clf].set(
            inv_project_clf(Y_proj_clf, params.D_tilde[clf], params.mu[clf])
        )
    return Y_inv_proj
