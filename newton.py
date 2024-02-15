import autograd.numpy as np
from autograd import jacobian
import time

def newton_system(F: callable, x0: np.ndarray, tol: float = 1e-5, max_iter: int = 100) -> np.ndarray:
    """
    Implementation of Newton's method for solving a system of nonlinear equations.

    Parameters:
        F (callable): A function that takes a vector x as input and returns a vector F(x).
        x0 (numpy.ndarray): An initial guess for the solution.
        tol (float): The tolerance for the stopping criterion.
        max_iter (int): The maximum number of iterations.

    Returns:
        numpy.ndarray: The approximate solution of the system of nonlinear equations.
    """
    x = np.asarray(x0)
    for i in range(max_iter):
        Fx = F(x)
        Jx = jacobian(F)(x)
        dx = np.linalg.solve(Jx, -Fx)
        x = x + dx
        if np.linalg.norm(dx) < tol:
            return x, i+1
    raise Exception("Failed to converge after {} iterations".format(max_iter))

def newton_system_full_predictor(F: callable, x0: np.ndarray, tol: float = 1e-5, max_iter: int = 100) -> np.ndarray:
    """
    Implementation of Newton's method with full predictor step for solving a system of nonlinear equations.

    Parameters:
        F (callable): A function that takes a vector x as input and returns a vector F(x).
        x0 (numpy.ndarray): An initial guess for the solution.
        tol (float): The tolerance for the stopping criterion.
        max_iter (int): The maximum number of iterations.

    Returns:
        numpy.ndarray: The approximate solution of the system of nonlinear equations.
    """
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
            return x, i+1
    raise Exception("Failed to converge after {} iterations".format(max_iter))


def newton_system_partial_predictor_corrector(F: callable, x0: np.ndarray, alpha: float = 0.25,
                                              tol: float = 1e-5, max_iter: int = 100) -> np.ndarray:
    """
    Implementation of Newton's method with partial predictor-corrector step for solving a system of nonlinear equations.

    Parameters:
        F (callable): A function that takes a vector x as input and returns a vector F(x).
        x0 (numpy.ndarray): An initial guess for the solution.
        alpha (float): The fraction of the full predictor step to use in the partial predictor step.
        tol (float): The tolerance for the stopping criterion.
        max_iter (int): The maximum number of iterations.

    Returns:
        numpy.ndarray: The approximate solution of the system of nonlinear equations.
    """
    x = np.asarray(x0)
    for i in range(max_iter):
        Fx = F(x)
        Jx = jacobian(F)(x)
        dx = np.linalg.solve(Jx, -Fx)
        x_pred = x + alpha*dx
        Fx_pred = F(x_pred)
        dx_corr = np.linalg.solve(Jx, -Fx_pred)
        x = x_pred + (1-alpha)*dx_corr
        if np.linalg.norm(dx_corr) < tol:
            return x, i+1
    raise Exception("Failed to converge after {} iterations".format(max_iter))



def F(x: np.ndarray) -> np.ndarray:
    return np.array([
        x[0]**2 + x[1]**2 + x[2]**2 - 10,
        x[0]*x[1] - 5*x[1] + x[2]**3 + 1,
        x[0]*x[2] + x[1]*x[2] - 2*x[2] - 1
    ])

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