"""Polyak averaging utilities for online EM training."""

from functools import partial
from typing import Any, List, NamedTuple
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def polyak_average_euclidean(avg_param: ArrayLike, new_param: ArrayLike, step: int) -> Array:
    """Apply Polyak averaging to Euclidean parameters.

    Args:
        avg_param: Current averaged parameter
        new_param: New parameter to average in
        step: Current step number (must be >= 1)

    Returns:
        Updated averaged parameter using formula: (t-1)/t * avg + 1/t * new
    """
    weight = (step - 1) / step
    return weight * avg_param + (1 - weight) * new_param


def polyak_average_stiefel(avg_D: ArrayLike, new_D: ArrayLike, step: int) -> Array:
    """Apply Polyak averaging to Stiefel manifold parameters with QR projection.

    Args:
        avg_D: Current averaged matrix on Stiefel manifold
        new_D: New matrix to average in
        step: Current step number (must be >= 1)

    Returns:
        Updated averaged matrix projected back to Stiefel manifold via QR decomposition
    """
    weight = (step - 1) / step
    interpolated = weight * avg_D + (1 - weight) * new_D
    Q, _ = jnp.linalg.qr(interpolated, mode="reduced")
    return Q


@jax.jit
def polyak_average_list_euclidean(avg_params: List[Array], new_params: List[Array], step: int) -> List[Array]:
    """Apply Polyak averaging to a list of Euclidean parameters.

    Args:
        avg_params: List of current averaged parameters
        new_params: List of new parameters to average in
        step: Current step number (must be >= 1)

    Returns:
        List of updated averaged parameters
    """
    return jax.tree.map(lambda avg, new: polyak_average_euclidean(avg, new, step), avg_params, new_params)


@jax.jit
def polyak_average_list_stiefel(avg_D_list: List[Array], new_D_list: List[Array], step: int) -> List[Array]:
    """Apply Polyak averaging to a list of Stiefel manifold parameters.

    Args:
        avg_D_list: List of current averaged matrices on Stiefel manifold
        new_D_list: List of new matrices to average in
        step: Current step number (must be >= 1)

    Returns:
        List of updated averaged matrices projected to Stiefel manifold
    """
    return jax.tree.map(lambda avg, new: polyak_average_stiefel(avg, new, step), avg_D_list, new_D_list)


class PolyakAveragerState(NamedTuple):
    """State for Polyak averaging containing averaged parameters and step counter."""

    params_polyak: NamedTuple  # Generic parameter structure (works with any model)
    polyak_step: int


class PolyakAverager:
    """Generic Polyak averager for model parameters with configurable update frequency.

    Works with any parameter structure (HDGMM, HDSTM, etc.) using JAX tree operations.
    Automatically handles Stiefel manifold parameters (identified by 'D_tilde' name pattern).
    """

    def __init__(self, update_frequency: int = 1):
        """Initialize Polyak averager.

        Args:
            update_frequency: Update averaged parameters every X steps
        """
        self.update_frequency = update_frequency

    def init_state(self, initial_params: Any) -> PolyakAveragerState:
        """Initialize Polyak averaging state with initial parameters.

        Args:
            initial_params: Initial model parameters (any structure)

        Returns:
            Initial Polyak averager state
        """
        return PolyakAveragerState(params_polyak=initial_params, polyak_step=0)

    def should_update(self, step: int) -> bool:
        """Check if parameters should be updated at current step.

        Args:
            step: Current training step

        Returns:
            True if parameters should be updated
        """
        return (step + 1) % self.update_frequency == 0

    @partial(jax.jit, static_argnames=["self"])
    def update_state(self, polyak_state: PolyakAveragerState, new_params: Any) -> PolyakAveragerState:
        """Update Polyak averaged parameters using generic JAX tree operations.

        Args:
            polyak_state: Current Polyak averaging state
            new_params: New parameters to average in (any structure)

        Returns:
            Updated Polyak averaging state
        """
        new_step = polyak_state.polyak_step + 1

        # Handle both HDGMM and HDSTM parameters generically using field names
        updated_fields = {}
        for field_name in polyak_state.params_polyak._fields:
            avg_param = getattr(polyak_state.params_polyak, field_name)
            new_param = getattr(new_params, field_name)

            if field_name == "D_tilde":
                # Stiefel manifold parameters
                updated_fields[field_name] = polyak_average_list_stiefel(avg_param, new_param, new_step)
            elif field_name == "A":
                # List of Euclidean parameters
                updated_fields[field_name] = polyak_average_list_euclidean(avg_param, new_param, new_step)
            else:
                # Regular Euclidean parameters (pi, mu, b, nu, etc.)
                updated_fields[field_name] = polyak_average_euclidean(avg_param, new_param, new_step)

        # Reconstruct the parameter object with the same type
        updated_params = type(polyak_state.params_polyak)(**updated_fields)

        return PolyakAveragerState(params_polyak=updated_params, polyak_step=new_step)

    def update_if_needed(self, step: int, polyak_state: PolyakAveragerState, new_params: Any) -> PolyakAveragerState:
        """Conditionally update Polyak averaged parameters.

        Args:
            step: Current training step
            polyak_state: Current Polyak averaging state
            new_params: New parameters to potentially average in (any structure)

        Returns:
            Updated or unchanged Polyak averaging state
        """
        if self.should_update(step):
            return self.update_state(polyak_state, new_params)
        else:
            return polyak_state
