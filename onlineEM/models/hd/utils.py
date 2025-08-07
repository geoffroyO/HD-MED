from typing import List, Tuple

import jax
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp


def permute_Ab(A: List[ArrayLike], b: ArrayLike) -> Tuple[Array, Array]:
    def tmp(A: ArrayLike, b: ArrayLike) -> Tuple[Array, Array]:
        min_A = A.min()
        A_bool = min_A >= A
        bool_vec = min_A < b

        A = bool_vec * (A_bool * b + (1 - A_bool) * A) + (1 - bool_vec) * A
        b = bool_vec * min_A + (1 - bool_vec) * b
        return A, b

    tmp_res = jax.tree.map(tmp, A, list(b))

    A = [e[0] for e in tmp_res]
    b = jnp.asarray([e[1] for e in tmp_res])

    return A, b
