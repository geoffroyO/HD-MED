from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike


def norm(tangent_vector: ArrayLike) -> float:
    return jnp.linalg.norm(tangent_vector)


def retraction(point: ArrayLike, tangent_vector: ArrayLike) -> Array:
    """QR-based retraction with improved numerical stability."""
    a = point + tangent_vector
    q, r = jnp.linalg.qr(a)

    # Sign correction for deterministic results
    d = jnp.diagonal(r, axis1=-2, axis2=-1)
    sign_correction = jnp.sign(d + jnp.where(d == 0, 1, 0))

    # Apply sign correction
    if len(q.shape) == 2:
        q = q * sign_correction[None, :]
    else:
        q = q * sign_correction.T

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


def beta_polak_ribiere_plus(newgrad: ArrayLike, gradgrad: ArrayLike, oldgrad: ArrayLike) -> Array:
    """Polak-Ribière+ conjugate gradient coefficient with automatic restart."""
    ip_diff = inner_product(newgrad, newgrad - oldgrad)
    beta = ip_diff / gradgrad
    return jnp.maximum(0.0, beta)


def should_restart_cg(newgrad: ArrayLike, oldgrad: ArrayLike, threshold: float = 0.1) -> Array:
    """Check if CG should restart based on orthogonality condition."""
    newgrad_norm = norm(newgrad)
    oldgrad_norm = norm(oldgrad)
    cos_angle = jnp.abs(inner_product(newgrad, oldgrad)) / (newgrad_norm * oldgrad_norm + 1e-12)
    return cos_angle > threshold
