from __future__ import annotations

import autograd.numpy as np
from autograd import jacobian


def newton_system(
    F: callable,
    x0: np.ndarray,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> np.ndarray:
    x = np.asarray(x0)
    for i in range(max_iter):
        Fx = F(x)
        Jx = jacobian(F)(x)
        dx = np.linalg.solve(Jx, -Fx)
        x = x + dx
        if np.linalg.norm(dx) < tol:
            return x, i + 1
    raise Exception(f"Failed to converge after {max_iter} iterations")


def newton_system_full_predictor(
    F: callable,
    x0: np.ndarray,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> np.ndarray:
    x = np.asarray(x0)
    for i in range(max_iter):
        Fx = F(x)
        Jx = jacobian(F)(x)
        dx = np.linalg.solve(Jx, -Fx)
        x_pred = x + dx
        Fx_pred = F(x_pred)
        dx_corr = np.linalg.solve(Jx, -Fx_pred)
        x = x_pred + dx_corr
        if np.linalg.norm(dx_corr) < tol:
            return x, i + 1
    raise Exception(f"Failed to converge after {max_iter} iterations")


def newton_system_partial_predictor_corrector(
    F: callable,
    x0: np.ndarray,
    alpha: float = 0.25,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> np.ndarray:
    x = np.asarray(x0)
    for i in range(max_iter):
        Fx = F(x)
        Jx = jacobian(F)(x)
        dx = np.linalg.solve(Jx, -Fx)
        x_pred = x + alpha * dx
        Fx_pred = F(x_pred)
        dx_corr = np.linalg.solve(Jx, -Fx_pred)
        x = x_pred + (1 - alpha) * dx_corr
        if np.linalg.norm(dx_corr) < tol:
            return x, i + 1
    raise Exception(f"Failed to converge after {max_iter} iterations")


def F(x: np.ndarray) -> np.ndarray:
    return np.array(
        [
            x[0] ** 3 + 0.5 * x[0] + 0.28 * x[1] + 0.23 * x[3] - 0.5,
            0.23 * x[0] + x[1] ** 3 + 0.5 * x[1] + 0.28 * x[2] + 0.25,
            0.23 * x[1] + x[2] ** 3 + 0.5 * x[2] + 0.28 * x[3] + 0.25,
            0.28 * x[0] + 0.23 * x[2] + x[3] ** 3 + 0.5 * x[3] - 0.5,
            # x[0] + 2*x[1] - 2,
            # x[0]**2 + 4*x[1]**2 - 4,
        ],
    )


def gradient_descent(F, x0, alpha=0.01, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        grad = F(x)
        x_new = x - alpha * grad
        if np.linalg.norm(x_new - x) < tol:
            return x_new, i+1
        x = x_new
    raise Exception("Gradient descent did not converge")

x0 = [0.5, 0.5, 0.5,0.5]

# Применение метода спуска
solution, iterations = gradient_descent(F, x0)
print("Решение системы:", solution)
print("Число итераций:", iterations)
