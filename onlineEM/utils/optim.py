from typing import Callable, NamedTuple, Tuple


import jax
from jax import jit
from jax.lax import while_loop
import jax.lax
import jax.numpy as jnp

from .stiefel import (
    beta_polak_ribiere_plus,
    inner_product,
    norm,
    retraction,
    riemannian_gradient,
    should_restart_cg,
    transport,
)


class line_search_params(NamedTuple):
    alpha: float
    x: jnp.ndarray
    cost: float
    cost_evaluations: int = 1


class opt_params(NamedTuple):
    x: jnp.ndarray
    cost: float
    grad: jnp.ndarray
    gradient_norm: float
    gradgrad: float
    descent_direction: jnp.ndarray
    oldalpha: float
    cost_evaluations: int = 0
    cg_iterations_since_restart: int = 0


@jax.tree_util.register_pytree_node_class
class LineSearch:
    def __init__(
        self,
        x: jnp.ndarray,
        objective: Callable[[jnp.ndarray], jnp.ndarray],
        d: jnp.ndarray,
        f0: jnp.ndarray,
        df0: jnp.ndarray,
        oldalpha: jnp.ndarray,
    ) -> None:
        self.x = x
        self.objective = objective
        self.d = d
        self.norm_d = norm(d)
        self.f0 = f0
        self.df0 = df0

        initial_alpha = jnp.where(oldalpha == -1, jnp.minimum(1.0, 1.0 / self.norm_d), jnp.clip(oldalpha, 1e-8, 10.0))
        alpha = initial_alpha
        newx = retraction(x, alpha * d)
        newf = objective(newx)

        self.params = line_search_params(alpha, newx, newf)

    def tree_flatten(self):
        children = (self.params.alpha,)
        aux_data = {
            "x": self.x,
            "objective": self.objective,
            "d": self.d,
            "f0": self.f0,
            "df0": self.df0,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def search(self, params: "line_search_params") -> "line_search_params":
        reduction_factor = jnp.where(params.cost_evaluations < 3, 0.5, 0.1)
        alpha = params.alpha * reduction_factor
        newx = retraction(self.x, alpha * self.d)
        newf = self.objective(newx)
        cost_evaluations = params.cost_evaluations + 1
        return line_search_params(alpha, newx, newf, cost_evaluations)

    def cond(self, params: "line_search_params") -> bool:
        sufficient_decrease = params.cost > self.f0 + 1e-4 * params.alpha * self.df0
        max_evals_reached = params.cost_evaluations >= 25
        min_step_size = params.alpha < 1e-16
        return sufficient_decrease * (~max_evals_reached) * (~min_step_size)

    def run(self) -> Tuple[jnp.ndarray, float]:
        params = while_loop(self.cond, self.search, self.params)
        alpha = jnp.where(params.cost > self.f0, 0.0, params.alpha)
        newx = jnp.where(params.cost > self.f0, self.x, params.x)
        oldalpha = jnp.where(
            params.cost_evaluations <= 2, jnp.minimum(2.0 * alpha, 1.0), jnp.maximum(0.1 * alpha, 1e-8)
        )
        return newx, oldalpha


@jax.tree_util.register_pytree_node_class
class OptStiefel:
    def __init__(
        self,
        x: jnp.ndarray,
        cost_fun: Callable[[jnp.ndarray], jnp.ndarray],
        grad_fun: Callable[[jnp.ndarray], jnp.ndarray],
        max_iter: int = 300,
        grad_tol: float = None,
        cg_restart_freq: int = 50,
    ) -> None:
        self.cost_fun = cost_fun
        self.grad_fun = grad_fun
        self.max_iter = max_iter
        self.cg_restart_freq = cg_restart_freq

        cost = cost_fun(x)
        grad = riemannian_gradient(x, grad_fun(x))
        gradient_norm = norm(grad)

        if grad_tol is None:
            self.grad_tol = 1e-6
        else:
            self.grad_tol = grad_tol

        gradgrad = inner_product(grad, grad)
        oldalpha = -1
        descent_direction = -grad

        self.params = opt_params(x, cost, grad, gradient_norm, gradgrad, descent_direction, oldalpha, 1, 0)

    def tree_flatten(self):
        children = (self.params.x,)
        aux_data = {
            "cost_fun": self.cost_fun,
            "grad_fun": self.grad_fun,
            "max_iter": self.max_iter,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def step(self, params: "opt_params") -> "opt_params":
        df0 = inner_product(params.grad, params.descent_direction)
        descent_direction = jnp.where(df0 >= 0, -params.grad, params.descent_direction)
        df0 = jnp.where(df0 >= 0, -params.gradgrad, df0)

        newx, oldalpha = LineSearch(
            params.x,
            self.cost_fun,
            descent_direction,
            params.cost,
            df0,
            params.oldalpha,
        ).run()

        newcost = self.cost_fun(newx)
        newgrad = riemannian_gradient(newx, self.grad_fun(newx))
        newgradient_norm = norm(newgrad)
        newgradnewgrad = inner_product(newgrad, newgrad)

        oldgrad = transport(newx, params.grad)
        new_descent_direction = transport(newx, descent_direction)

        should_restart = should_restart_cg(newgrad, params.grad)
        cg_restart_due = params.cg_iterations_since_restart >= self.cg_restart_freq
        restart_cg = should_restart | cg_restart_due

        beta = jnp.where(restart_cg, 0.0, beta_polak_ribiere_plus(newgrad, params.gradgrad, oldgrad))
        new_descent_direction = -newgrad + beta * new_descent_direction

        cg_iterations_since_restart = jnp.where(restart_cg, 0, params.cg_iterations_since_restart + 1)

        cost_evaluations = params.cost_evaluations + 1

        return opt_params(
            newx,
            newcost,
            newgrad,
            newgradient_norm,
            newgradnewgrad,
            new_descent_direction,
            oldalpha,
            cost_evaluations,
            cg_iterations_since_restart,
        )

    def cond(self, params: "opt_params") -> bool:
        grad_criterion = params.gradient_norm >= self.grad_tol
        iter_criterion = params.cost_evaluations < self.max_iter
        return grad_criterion & iter_criterion

    @jit
    def run(self) -> jnp.ndarray:
        self.params = while_loop(self.cond, self.step, self.params)
        return self.params.x


class bissec_state(NamedTuple):
    lower: float
    upper: float
    sign: float
    params: float = 0
    err: float = 1
    niter: int = 0
    interval_size: float = 1
    converged: bool = False


@jax.tree_util.register_pytree_node_class
class Bisection:
    def __init__(self, fun, lower, upper, maxiter=200, tol=1e-5, rtol=1e-8):
        self.fun = fun
        self.maxiter = maxiter
        self.tol = tol
        self.rtol = rtol

        lower_value = self.fun(lower)
        upper_value = self.fun(upper)

        self.init_lower = jnp.asarray(lower, float)
        self.init_upper = jnp.asarray(upper, float)

        # More robust bracketing check
        self.init_lower_value = jnp.asarray(lower_value, float)
        self.init_upper_value = jnp.asarray(upper_value, float)

        # Check if we have a valid bracket
        has_bracket = (lower_value * upper_value) < 0

        sign = jnp.where(has_bracket & (lower_value < 0), 1.0, jnp.where(has_bracket & (lower_value > 0), -1.0, 0.0))
        self.init_sign = jnp.asarray(sign)
        self.has_valid_bracket = jnp.asarray(has_bracket, bool)

    def tree_flatten(self):
        return (
            (self.fun, self.maxiter, self.tol, self.rtol, self.init_lower, self.init_upper, self.has_valid_bracket),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def update(self, state):
        params = 0.5 * (state.upper + state.lower)
        value = self.fun(params)

        # Update interval based on sign
        too_large = state.sign * value > 0
        upper = jnp.where(too_large, params, state.upper)
        lower = jnp.where(too_large, state.lower, params)

        # Better error metrics
        interval_size = upper - lower
        abs_error = jnp.abs(value)
        rel_error = interval_size / (jnp.abs(params) + 1e-16)

        # Combined error criterion
        err = jnp.maximum(abs_error, rel_error)

        # Check convergence
        abs_converged = abs_error <= self.tol
        rel_converged = rel_error <= self.rtol
        interval_converged = interval_size <= self.tol
        converged = abs_converged | rel_converged | interval_converged

        niter = state.niter + 1

        new_state = bissec_state(
            lower=lower,
            upper=upper,
            sign=state.sign,
            params=params,
            err=err,
            niter=niter,
            interval_size=interval_size,
            converged=converged,
        )
        return new_state

    def cond(self, state):
        continue_iter = state.niter < self.maxiter
        not_converged = ~state.converged
        return continue_iter & not_converged

    def run(self):
        init_interval_size = jnp.abs(self.init_upper - self.init_lower)
        init_state = bissec_state(
            lower=self.init_lower, upper=self.init_upper, sign=self.init_sign, interval_size=init_interval_size
        )

        # Run bisection if we have a valid bracket, otherwise return midpoint
        def run_bisection():
            return while_loop(self.cond, self.update, init_state)

        def return_midpoint():
            midpoint = 0.5 * (self.init_lower + self.init_upper)
            return init_state._replace(params=midpoint, converged=False, err=1.0)

        state = jax.lax.cond(self.has_valid_bracket, run_bisection, return_midpoint)

        return state.params
