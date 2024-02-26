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
            x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 10,
            x[0] * x[1] - 5 * x[1] + x[2] ** 3 + 1,
            x[0] * x[2] + x[1] * x[2] - 2 * x[2] - 1,
        ],
    )


# x0 = np.array([1.0, 1.0, 1.0])
# start_time = time.time()
# x, n_iterations = newton_system(F, x0)
# end_time = time.time()

# print(f"Solution: {x}")
# print(f"Iterations: {n_iterations}")
# print(f"Execution time: {end_time - start_time} seconds")


# x0 = np.array([1.0, 1.0, 1.0])
# start_time = time.time()
# x, n_iterations = newton_system_full_predictor(F, x0)
# end_time = time.time()

# print(f"Solution: {x}")
# print(f"Iterations: {n_iterations}")
# print(f"Execution time: {end_time - start_time} seconds")


# x0 = np.array([1.0, 1.0, 1.0])
# start_time = time.time()
# x, n_iterations = newton_system_partial_predictor_corrector(F, x0)
# end_time = time.time()

# print(f"Solution: {x}")
# print(f"Iterations: {n_iterations}")
# print(f"Execution time: {end_time - start_time} seconds")
