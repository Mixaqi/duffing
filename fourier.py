from __future__ import annotations

import math
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
from scipy.integrate import quad

x = sy.Symbol("x")
harmonics_number = 2
interval_start = -math.pi
interval_end = math.pi


# Исходная функция
def f(x: sy.Symbol) -> sy.Expr:
    return sy.cos(2*x)


def calculate_coefficients(f: sy.Expr) -> tuple[float, list[float], list[float]]:
    x = sy.Symbol("x")
    ai_list = []
    bi_list = []
    period = math.pi
    a0 = (1 / period) * sy.integrate(f(x), (x, interval_start, interval_end))

    for i in range(1, harmonics_number + 1):
        ai = (1 / period) * sy.integrate(
            f(x) * sy.cos(i * x),
            (x, interval_start, interval_end),
        )
        ai_list.append(ai)

        bi = (1 / period) * sy.integrate(
            f(x) * sy.sin(i * x),
            (x, interval_start, interval_end),
        )
        bi_list.append(bi)

    return a0, ai_list, bi_list


def calculate_approximation_function(
    harmonics_number: int,
) -> Callable([[sy.Symbol], sy.Expr]):
    x = sy.Symbol("x")
    a0, ai_list, bi_list = calculate_coefficients(f)
    approximation_function = a0 / 2
    for i in range(1, harmonics_number + 1):
        approximation_function += ai_list[i - 1] * sy.cos(i * x) + bi_list[
            i - 1
        ] * sy.sin(i * x)
    return approximation_function


def plot_function(f: sy.Expr, start: float, end: float) -> None:
    x_vals = np.linspace(start, end, 1000)
    y_vals = [f(val) for val in x_vals]
    plt.plot(x_vals, y_vals, label="Original Function")
    plt.title("Original Function")
    plt.legend()
    plt.show()


def plot_function_and_approximation(
    f: Callable[[float], float],
    approximation_function: Callable[[float], float],
    interval_start: float,
    interval_end: float,
) -> None:
    x_vals = np.linspace(interval_start, interval_end, 1000)
    y_vals_original = [f(val) for val in x_vals]
    y_vals_approximation = [approximation_function.subs(x, val) for val in x_vals]
    plt.plot(x_vals, y_vals_original, label="Original Function")
    plt.plot(x_vals, y_vals_approximation, label="Approximation Function")
    plt.title("Original and Approximation Functions")
    plt.legend()
    plt.show()


def calculate_norm(
    f: Callable[[float], sy.Expr],
    approximation_function: Callable[[float], sy.Expr],
    start: float,
    end: float,
) -> float:
    integrand = lambda x: (f(x) - approximation_function.subs("x", x)) ** 2
    result, _ = quad(integrand, start, end)
    return np.sqrt(result)


print(calculate_coefficients(f))
approximation_function = calculate_approximation_function(harmonics_number)
norm_value = calculate_norm(f, approximation_function, interval_start, interval_end)
print(f"L2 Norm: {norm_value}")
plot_function_and_approximation(f, approximation_function, interval_start, interval_end)

# plot_function(f, -math.pi, math.pi)
