from __future__ import annotations

import math
from test import create_equations, initialize_x

import numpy as np
import sympy as sy
from scipy.integrate import quad
from scipy.optimize import newton

from methods.fourier import (
    calculate_approximation_function,
    calculate_coefficients,
    f,
)


def main() -> None:
    delta = 0.1
    alpha = 1.0
    beta = 1.0
    gamma = 0.5
    omega = 1
    n = 3
    h = 2 * math.pi / n
    t = sy.Symbol("t")
    x = sy.Symbol("x")
    harmonics_number = 5
    interval_start = 0
    interval_end = 2 * math.pi
    period = 2 * math.pi

    x = initialize_x(n)  # calculate equatuions number
    equations = create_equations(
        x, n=n, delta=delta, alpha=alpha, gamma=gamma, omega=omega, beta=beta,
    )
    equations = np.array(equations)
    x0 = np.zeros(len(equations))
    solution = newton(lambda x: create_equations(x, n, delta, alpha, gamma, omega, beta), x0)
    print(solution)
    a0, ai_list, bi_list = calculate_coefficients(f, period, interval_start, interval_end, harmonics_number, h)
    print(a0, ai_list, bi_list)
    approximation_function = calculate_approximation_function(harmonics_number, a0, ai_list, bi_list)
    print(approximation_function)
    # norm_value = calculate_norm(solution, approximation_function, interval_start, interval_end)



def calculate_norm(f_values: np.ndarray, approximation_function, interval_start: float, interval_end: float) -> float:
    integrand = lambda x: (f_values - approximation_function(x)) ** 2
    result, _ = quad(integrand, interval_start, interval_end)
    return np.sqrt(result)

if __name__ == "__main__":
    main()
