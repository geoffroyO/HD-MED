from typing import Tuple, List

import jax
from jax import Array, vmap
import jax.numpy as jnp
from jax.typing import ArrayLike

fill_diagonal = vmap(
    lambda x, y: jnp.fill_diagonal(x, y, inplace=False), in_axes=(0, 0)
)
batch_diag = vmap(jnp.diag, in_axes=(0))

vmap_eig = vmap(jnp.linalg.eig, in_axes=(0))


def permute_Ab(A: List[ArrayLike], b: ArrayLike) -> Tuple[Array, Array]:
    def tmp(A, b):
        min_A = A.min()
        A_bool = min_A >= A
        bool_vec = min_A < b

        A = bool_vec * (A_bool * b + (1 - A_bool) * A) + (1 - bool_vec) * A
        b = bool_vec * min_A + (1 - bool_vec) * b
        return A, b

    tmp_res = jax.tree_map(tmp, A, list(b))

    A = [e[0] for e in tmp_res]
    b = jnp.asarray([e[1] for e in tmp_res])

    return A, b
