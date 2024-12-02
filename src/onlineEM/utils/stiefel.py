from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike


def norm(tangent_vector: ArrayLike) -> float:
    return jnp.linalg.norm(tangent_vector)


def retraction(point: ArrayLike, tangent_vector: ArrayLike) -> Array:
    a = point + tangent_vector
    q, r = jnp.linalg.qr(a)
    s = jnp.diagonal(r, axis1=-2, axis2=-1)
    s = s + jnp.where(s == 0, 1, 0)
    s = s / jnp.abs(s)
    q = q * s.T
    return q


def inner_product(tangent_vector_a: ArrayLike, tangent_vector_b: ArrayLike) -> Array:
    return jnp.trace(tangent_vector_a @ tangent_vector_b.T)


def projection(point: ArrayLike, tangent_vector: ArrayLike) -> Array:
    tmp = point.T @ tangent_vector
    return tangent_vector - point @ (tmp + tmp.T) / 2


def transport(point: ArrayLike, tangent_vector: ArrayLike) -> Array:
    return projection(point, tangent_vector)


def riemannian_gradient(point: ArrayLike, euclidian_gradient: ArrayLike) -> Array:
    return projection(point, euclidian_gradient)


def beta_polak_ribiere(
    newgrad: ArrayLike, gradgrad: ArrayLike, oldgrad: ArrayLike
) -> Array:
    ip_diff = inner_product(newgrad, newgrad - oldgrad)
    return jnp.where(ip_diff / gradgrad > 0, ip_diff / gradgrad, 0)
