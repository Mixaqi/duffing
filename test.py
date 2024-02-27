from __future__ import annotations

import numpy as np
from sympy import symbols
from sympy import pi

# Parameters
delta = 0.1
alpha = 1.0
beta = 1.0
gamma = 0.2
omega = 1.4
n = 3
h = 2 * np.pi / n

# def create_symbols(n, prefix="x"):
#     return symbols(" ".join([f"{prefix}{i}" for i in range(n)]))

# def convert_to_array_indices(symbols, prefix="x"):
#     array_indices = []
#     for sym in symbols:
#         index = int(sym.name[len(prefix):])
#         array_indices.append(f"x[{index}]")
#     return array_indices

# def F(x):
#     return [
#         x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 10,
#         x[0] * x[1] - 5 * x[1] + x[2] ** 3 + 1,
#         x[0] * x[2] + x[1] * x[2] - 2 * x[2] - 1,
#     ]

# # Пример использования
# x_symbols = create_symbols(3, prefix="x")
# array_indices = convert_to_array_indices(x_symbols, prefix="x")
# print("Символьные переменные:", x_symbols)
# print("Индексы массива x:", array_indices)

# x_values = [1, 2, 3]  # Пример значений переменных x
# F_values = F(x_values)
# print("Значения функции F:", F_values)



def initialize_x(n):
    x = [symbols(f"x[{i}]") for i in range(n + 1)]  ###Здесь мы резервируем максимально возможное число этих переменных
    return x

# Пример использования
x = initialize_x(n)
print("Массив x:", x)


def create_equations(x, n):
    equations = []

    # Вычисление значения h
    h = 2 * pi / n

    # Определение количества переменных
    num_vars = len(x)

    # Добавление уравнений для производной первого порядка x'
    for i in range(num_vars):
        # Рассчитываем индексы для xi+1 и xi-1 с учетом особых случаев
        idx_plus_1 = (i + 1) % num_vars
        idx_minus_1 = (i - 1) % num_vars

        # Создаем уравнение для x'
        equation = (x[idx_plus_1] - x[i]) / h
        equations.append(equation)

    # Добавление уравнений для производной второго порядка x''
    for i in range(num_vars):
        # Рассчитываем индексы для xi+1, xi и xi-1 с учетом особых случаев
        idx_plus_1 = (i + 1) % num_vars
        idx_minus_1 = (i - 1) % num_vars

        # Создаем уравнение для x''
        equation = (x[idx_plus_1] - 2*x[i] + x[idx_minus_1]) / h**2
        equations.append(equation)

    return equations

x = initialize_x(n)
equations = create_equations(x, n)
for eq in equations:
    print(eq)