from __future__ import annotations

import math
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
from scipy.integrate import quad

t = sy.Symbol("t")
x = sy.Symbol("x")
harmonics_number = 5
interval_start = -math.pi
interval_end = math.pi


# Исходная функция
def f(t: sy.Symbol) -> sy.Expr:
    return sy.cos(2 * t)


def calculate_coefficients(f: sy.Expr) -> tuple[float, list[float], list[float]]:
    t = sy.Symbol("t")
    ai_list = []
    bi_list = []
    period = math.pi
    a0 = (1 / period) * sy.integrate(f(t), (t, interval_start, interval_end))

    for i in range(1, harmonics_number + 1):
        ai = (1 / period) * sy.integrate(
            f(t) * sy.cos(i * t),
            (t, interval_start, interval_end),
        )
        ai_list.append(ai)

        bi = (1 / period) * sy.integrate(
            f(t) * sy.sin(i * t),
            (t, interval_start, interval_end),
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


# print(calculate_coefficients(f))

# print(calculate_coefficients(f))
approximation_function = calculate_approximation_function(harmonics_number)
# print(approximation_function)
