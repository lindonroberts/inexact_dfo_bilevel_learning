#!/usr/bin/env python3

"""
A variety of lower-level solvers (i.e. methods for unconstrained strongly convex optimization).

Main feature is to select either a fixed number of iterations, or to a given tolerance
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from math import ceil, log, sqrt
import numpy as np

__all__ = ['strongly_convex_solve', 'ALGORITHMS']

MIN_XTOL = 1e-15
ALGORITHMS = ['gd', 'fista']


def strongly_convex_solve(gradf, callback, x0, L, mu, x0tol=None, niters=None, xtol=None, algorithm='gd'):
    """
    Run strongly convex solver. The convergence of these methods are:
        ||xk-x*||^2 <= w^k ||x0-x*||^2
    where 0 < w < 1 depends on the condition number of the problem (mu/L).
    Run either for 'niters' iterations, or enough to achieve final tolerance ||xk-x*||^2 <= xtol

    Allowable algorithms are:
    - 'gd' (gradient descent with fixed step size)
    - 'fista' (FISTA)

    :param gradf: gradient of objective function (np.array -> np.array)
    :param callback: callback function for each iteration (takes np.array, returns nothing)
    :param x0: starting point for algorithm (np.array)
    :param L: Lipschitz constant of gradf
    :param mu: strong convexity parameter
    :param x0tol: estimated error to true solution, ||x0-x*||^2
    :param niters: number of iterations to run
    :param xtol: desired tolerance on result, ||xk-x*||^2
    :param algorithm: name of solver to use (must be 'gd' or 'fista')
    :return: approximate solution xk, number of iterations, estimated error ||xk-x*||^2
    """
    assert algorithm.lower() in ALGORITHMS, "Unknown algorithm: %s" % algorithm
    smooth_switcher = {'gd': gradient_descent, 'fista': fista}  # lookup of solvers
    xtol_to_use = None if xtol is None else max(xtol, MIN_XTOL)  # floor xtol, if given
    if algorithm.lower() in smooth_switcher:
        return smooth_switcher[algorithm.lower()](gradf, callback, x0, L, mu, x0tol=x0tol, niters=niters, xtol=xtol_to_use)
    else:
        raise RuntimeError("Algorithm %s not listed as smooth or nonsmooth solver" % algorithm.lower())


def gradient_descent(gradf, callback, x0, L, mu, x0tol=None, niters=None, xtol=None):
    """
    Run gradient descent with fixed step length 2/(L+mu).

    The usual convergence result is:
        ||xk-x*||^2 <= w^k * ||x0-x*||^2
    where w := (1 - mu/L)^2.

    However, here, we also use the bound
        ||xk-x*|| <= ||gradf(xk)|| / mu
    and terminate if ||xk-x*||^2 <= xtol, since this tends to be a better bound in practice.

    :param gradf: gradient of objective function (np.array -> np.array)
    :param callback: callback function for each iteration (takes np.array, returns nothing)
    :param x0: starting point for algorithm (np.array)
    :param L: Lipschitz constant of gradf
    :param mu: strong convexity parameter
    :param x0tol: estimated error to true solution, ||x0-x*||^2
    :param niters: number of iterations to run
    :param xtol: desired tolerance on result, ||xk-x*||^2
    :return: approximate solution xk, number of iterations, estimated error ||xk-x*||^2
    """
    step_size = 2.0 / (L + mu)
    w = (1.0 - mu / L) ** 2

    if niters is not None:
        # Run for fixed number of iterations
        assert xtol is None, "If niters is specified, xtol must be None"
        assert niters >= 0, "niters must be >= 0"
        max_iters = niters
        terminate_on_small_gradient = False
    else:
        # Terminate when ||xk-x*||^2 <= xtol
        assert xtol is not None, "If niters is not specified, xtol must be given"
        assert xtol > 0.0, "xtol must be strictly positive"

        if x0tol is not None:
            # Have initial error estimate ||x0-x*||^2
            max_iters = max(ceil(log(xtol / x0tol, w)), 0)
            terminate_on_small_gradient = True
        else:
            max_iters = None
            terminate_on_small_gradient = True

    # We know ||xk-x*||^2 <= ||gradf(xk)||^2 / mu^2
    small_gradient = lambda gradient: terminate_on_small_gradient and np.linalg.norm(gradient)**2 / (mu**2) <= xtol

    x = x0.copy()
    g = gradf(x)

    # Check for termination before loop starts
    if (max_iters is not None and max_iters <= 0) or small_gradient(g):
        if x0tol is not None:
            x_sqerror_bound = min(x0tol, np.linalg.norm(g) ** 2 / (mu ** 2))
        else:
            x_sqerror_bound = np.linalg.norm(g) ** 2 / (mu ** 2)
        return x, 0, x_sqerror_bound

    # Main loop
    k = 0
    while True:
        x -= step_size * g
        g = gradf(x)
        if callback is not None:
            callback(x)
        k += 1
        # Check termination on xtol
        if (max_iters is not None and k >= max_iters) or small_gradient(g):
            break

    # Return: solution, number iterations, error bound ||xk-x*||^2
    if x0tol is not None:
        x_sqerror_bound = min((w ** k) * x0tol, np.linalg.norm(g)**2 / (mu**2))
    else:
        x_sqerror_bound = np.linalg.norm(g) ** 2 / (mu ** 2)
    return x, k, x_sqerror_bound


def fista(gradf, callback, x0, L, mu, x0tol=None, niters=None, xtol=None):
    """
    Run FISTA.

    The convergence result is:
        ||xk-x*||^2 <= w^k * (L/mu) * (1+sqrt(mu/L)) * ||x0-x*||^2
    where w := 1 - sqrt(mu/L).

    However, here, we also use the bound
        ||xk-x*|| <= ||gradf(xk)|| / mu
    and terminate if ||xk-x*||^2 <= xtol, since this tends to be a better bound in practice.

    This is a modified FISTA for strongly-convex objectives; see Algorithm 1 of
        L. Calatroni and A. Chambolle (2019), Backtracking Strategies for Accelerated Descent Methods
        with Smooth Composite Objectives, SIAM J. Optim. 29(3), pp. 1772-1798.
    or Algorithm 5 of
        A. Chambolle and T. Pock (2016), An introduction to continuous optimization for imaging,
        Acta Numerica 25, pp. 161-319.
    Compared to that version, here we have no regularization term, so have no prox operator and mu_f=mu, mu_g=0

    :param gradf: gradient of objective function (np.array -> np.array)
    :param callback: callback function for each iteration (takes np.array, returns nothing)
    :param x0: starting point for algorithm (np.array)
    :param L: Lipschitz constant of gradf
    :param mu: strong convexity parameter
    :param x0tol: estimated error to true solution, ||x0-x*||^2
    :param niters: number of iterations to run
    :param xtol: desired tolerance on result, ||xk-x*||^2
    :return: approximate solution xk, number of iterations, estimated error ||xk-x*||^2
    """
    step_size = 1.0 / L
    q = step_size * mu
    w = 1.0 - sqrt(q)

    if niters is not None:
        # Run for fixed number of iterations
        assert xtol is None, "If niters is specified, xtol must be None"
        assert niters >= 0, "niters must be >= 0"
        max_iters = niters
        terminate_on_small_gradient = False
    else:
        # Terminate when ||xk-x*||^2 <= xtol
        assert xtol is not None, "If niters is not specified, xtol must be given"
        assert xtol > 0.0, "xtol must be strictly positive"

        if x0tol is not None:
            # Have initial error estimate ||x0-x*||^2
            max_iters = max(ceil(log(xtol / ((L / mu) * (1 + sqrt(q)) * x0tol), w)), 0)
            terminate_on_small_gradient = True
        else:
            max_iters = None
            terminate_on_small_gradient = True

    # We know ||xk-x*||^2 <= ||gradf(xk)||^2 / mu^2
    small_gradient = lambda gradient: terminate_on_small_gradient and np.linalg.norm(gradient)**2 / (mu**2) <= xtol

    x = x0.copy()
    g = gradf(x)

    # Check for termination before loop starts
    if (max_iters is not None and max_iters <= 0) or small_gradient(g):
        if x0tol is not None:
            x_sqerror_bound = min(x0tol, np.linalg.norm(g) ** 2 / (mu ** 2))
        else:
            x_sqerror_bound = np.linalg.norm(g) ** 2 / (mu ** 2)
        return x, 0, x_sqerror_bound

    # Main loop
    k = 0
    tk = 0.0  # initial value t0
    xkm1 = x.copy()  # x_{-1}=x0 initially
    while True:
        tksq = tk**2
        tkp1 = (1 - q*tksq + np.sqrt((1-q*tksq)**2 + 4*tksq)) / 2
        beta_k = (tk-1) * (1-q*tkp1) / (tkp1*(1-q))
        yk = x + beta_k * (x - xkm1)
        yk -= step_size * gradf(yk)
        # Update x, xkm1
        xkm1[:] = x
        x[:] = yk
        # Update t
        tk = tkp1
        # Check termination
        if callback is not None:
            callback(x)
        k += 1
        g = gradf(x)
        # Check termination on xtol
        if (max_iters is not None and k >= max_iters) or small_gradient(g):
            break

    # Return: solution, number iterations, error bound ||xk-x*||^2
    if x0tol is not None:
        x_sqerror_bound = min((w ** k) * (L / mu) * (1 + sqrt(q)) * x0tol, np.linalg.norm(g) ** 2 / (mu ** 2))
    else:
        x_sqerror_bound = np.linalg.norm(g) ** 2 / (mu ** 2)
    return x, k, x_sqerror_bound
