from typing import Callable, NamedTuple, Tuple

import numpy as np

import jax
from jax import jit
from jax.lax import while_loop
import jax.numpy as jnp

from .stiefel import (
    beta_polak_ribiere,
    inner_product,
    norm,
    retraction,
    riemannian_gradient,
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

        alpha = jnp.where(oldalpha == -1, 1 / self.norm_d, oldalpha)
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
        alpha = params.alpha * 0.5
        newx = retraction(self.x, alpha * self.d)
        newf = self.objective(newx)
        cost_evaluations = params.cost_evaluations + 1
        return line_search_params(alpha, newx, newf, cost_evaluations)

    def cond(self, params: "line_search_params") -> bool:
        return (params.cost > self.f0 + 0.5 * params.alpha * self.df0) * (
            params.cost_evaluations <= 10
        )

    def run(self) -> Tuple[jnp.ndarray, float]:
        params = while_loop(self.cond, self.search, self.params)
        alpha = jnp.where(params.cost > self.f0, 0.0, params.alpha)
        newx = jnp.where(params.cost > self.f0, self.x, params.x)
        oldalpha = jnp.where(params.cost_evaluations == 2, alpha, 2 * alpha)
        return newx, oldalpha


@jax.tree_util.register_pytree_node_class
class OptStiefel:
    def __init__(
        self,
        x: jnp.ndarray,
        cost_fun: Callable[[jnp.ndarray], jnp.ndarray],
        grad_fun: Callable[[jnp.ndarray], jnp.ndarray],
        max_iter: int = 300,
    ) -> None:
        self.cost_fun = cost_fun
        self.grad_fun = grad_fun
        self.max_iter = max_iter

        cost = cost_fun(x)
        grad = riemannian_gradient(x, grad_fun(x))
        gradient_norm = norm(grad)
        gradgrad = inner_product(grad, grad)
        oldalpha = -1
        descent_direction = -grad

        self.params = opt_params(
            x, cost, grad, gradient_norm, gradgrad, descent_direction, oldalpha
        )

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

        beta = beta_polak_ribiere(newgrad, params.gradgrad, oldgrad)
        new_descent_direction = -newgrad + beta * new_descent_direction

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
        )

    def cond(self, params: "opt_params") -> bool:
        return (params.gradient_norm >= 1e-6) * (
            params.cost_evaluations <= self.max_iter
        )

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


@jax.tree_util.register_pytree_node_class
class Bisection:
    def __init__(self, fun, lower, upper, maxiter=200, tol=1e-5):
        self.fun = fun
        self.maxiter = maxiter
        self.tol = tol

        lower_value = self.fun(lower)
        upper_value = self.fun(upper)

        self.init_lower = jnp.asarray(lower, float)
        self.init_upper = jnp.asarray(upper, float)

        sign = jnp.where(
            (lower_value < 0) & (upper_value >= 0),
            1,
            jnp.where((lower_value > 0) & (upper_value <= 0), -1, 0),
        )
        self.init_sign = jnp.asarray(sign)

    def tree_flatten(self):
        return (
            (self.fun, self.maxiter, self.tol, self.init_lower, self.init_upper),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def update(self, state):
        params = 0.5 * (state.upper + state.lower)
        value = self.fun(params)

        too_large = state.sign * value > 0
        upper = jnp.where(too_large, params, state.upper)
        lower = jnp.where(too_large, state.lower, params)

        err = jnp.sqrt(value**2)
        niter = state.niter + 1

        new_state = bissec_state(
            lower=lower,
            upper=upper,
            sign=state.sign,
            params=params,
            err=err,
            niter=niter,
        )
        return new_state

    def cond(self, state):
        return (state.niter < self.maxiter) * (self.tol <= state.err)

    def run(self):
        init_state = bissec_state(
            lower=self.init_lower, upper=self.init_upper, sign=self.init_sign
        )
        state = while_loop(self.cond, self.update, init_state)
        return state.params


def trimkmeans(X, k, trim=0.1, runs=100):
    X = np.array(X, dtype=np.float64)
    n = len(X)
    maxit = 2 * n
    nin = int(np.ceil((1 - trim) * n))
    crit = np.inf
    oldclass = np.zeros(n, dtype=int)
    iclass = np.zeros(n, dtype=int)
    optclass = np.zeros(n, dtype=int)

    disttom = np.zeros(n, dtype=np.float64)

    for _ in range(runs):
        means = X[np.random.choice(n, k, replace=False)]
        wend = False
        itcounter = 0

        while not wend:
            itcounter += 1
            for j in range(n):
                dj = np.sum((X[j] - means) ** 2, axis=-1)

                iclass[j] = np.argmin(dj) + 1
                disttom[j] = dj[iclass[j] - 1]

            iclass[np.argsort(disttom)[nin:]] = 0
            if (itcounter >= maxit) or np.array_equal(oldclass, iclass):
                wend = True
            else:
                for p in range(1, k + 1):
                    if np.sum(iclass == p) == 0:
                        means[p - 1] = X[iclass == 0][0]
                    else:
                        if np.sum(iclass == p) > 1:
                            means[p - 1] = np.mean(X[iclass == p], axis=0)
                        else:
                            means[p - 1] = X[iclass == p]

                oldclass = iclass.copy()

        newcrit = np.sum(disttom[iclass > 0])

        if newcrit <= crit:
            optclass = iclass.copy()
            crit = newcrit
            optmeans = means.copy()

    return optclass, optmeans
