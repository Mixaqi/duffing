from __future__ import annotations

import autograd.numpy as np
from autograd import jacobian


def func(x, h, i):
    return (
        (x[i - 1] - 2 * x[i] + x[i + 1]) / h**2
        + 0.05 * (x[i + 1] - x[i - 1]) / h
        + x[i]
        + x[i] ** 3
        - 0.5 * np.cos(i * h)
    )


def equations(x, h):
    n = len(x)
    F = np.zeros(n)
    for i in range(1, n - 1):
        F[i] = func(x, h, i)
    return F


def solve_equations(n, h):
    x_guess = np.zeros(n)
    x_guess[0] = x_guess[-1] = 0
    v_guess = np.zeros(n)
    initial_guess = np.concatenate((x_guess, v_guess))
    result = custom_newton(equations, initial_guess, h)
    return result[:n]


def custom_newton(F, x0, h, tol=1e-5, max_iter=100):
    x = np.asarray(x0)
    for i in range(max_iter):
        Fx = F(x, h)
        Jx = jacobian(F)(x)
        dx = np.linalg.solve(Jx, -Fx)
        x = x + dx
        if np.linalg.norm(dx) < tol:
            return x
    raise Exception(f"Failed to converge after {max_iter} iterations")


if __name__ == "__main__":
    n = 5
    h = 2 * np.pi / n
    solution = solve_equations(n, h)
    print("Solution:", solution)
