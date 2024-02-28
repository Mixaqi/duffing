from __future__ import annotations
from scipy.optimize import newton
import numpy as np
from sympy import cos, symbols

from methods.newton import *

delta = 0.1
alpha = 1.0
beta = 1.0
gamma = 0.5
omega = 1
n = 3
h = 2 * np.pi / n


def initialize_x(n):
    x = [symbols(f"x[{i}]") for i in range(n + 1)]
    return x


# Пример использования
# x = initialize_x(n)
# print("Массив x:", x)


def create_equations(x, n):
    equations = []

    # Вычисление значения h
    h = 2 * np.pi / n

    # Добавление уравнений для производной первого порядка x'
    for i in range(n + 1):
        idx_plus_1 = (i + 1) % (n + 1)
        equation_x = ((x[idx_plus_1] - x[i]) / h) * delta
        equation_x_cubed = beta * x[i] ** 3  # ЗДЕСЬ ЕЩЕ ДОБАВИЛИ БЕТА
        equation = (
            equation_x + equation_x_cubed + alpha * x[i] - gamma * cos(omega * (i * h))
        )
        equations.append(equation)

    # Добавление уравнений для производной второго порядка x''
    for i in range(n + 1):
        idx_plus_1 = (i + 1) % (n + 1)
        idx_minus_1 = (i - 1) % (n + 1)
        equation = (x[idx_plus_1] - 2 * x[i] + x[idx_minus_1]) / h**2
        equations.append(equation)

    # Группировка уравнений
    num_equations = len(equations)
    half_point = num_equations // 2
    grouped_equations = [
        equations[i] + equations[i + half_point] for i in range(half_point)
    ]
    # print(grouped_equations)
    return grouped_equations



x = initialize_x(n)
equations = create_equations(x, n)
numerical_values = [eq.evalf() for eq in equations]

# Преобразование списка в массив ndarray
equations_array = np.array(numerical_values)
for value in equations_array:
    print(type(value))


for eq in equations:
    print(eq)

def equations_function(x: np.ndarray) -> np.ndarray:
    return create_equations(x, n)

x0 = np.zeros(len(equations_array))

solution = newton(equations_function, x0)

print(solution)