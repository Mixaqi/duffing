import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Задаем периодическую функцию, которую хотим аппроксимировать
def target_function(x: np.ndarray) -> np.ndarray:
    return np.sin(x) + 0.5 * np.sin(2 * x) + 0.2 * np.sin(3 * x)

# Функция для вычисления коэффициентов Фурье
def compute_fourier_coefficients(x_values: np.ndarray, num_harmonics: int) -> List[Tuple[float, float]]:
    coefficients = []
    for n in range(num_harmonics):
        an = (2 / np.pi) * np.trapz(target_function(x_values) * np.cos(n * x_values), x_values)
        bn = (2 / np.pi) * np.trapz(target_function(x_values) * np.sin(n * x_values), x_values)
        coefficients.append((an, bn))
    return coefficients

# Функция для аппроксимации функции с использованием коэффициентов Фурье
def approximate_with_fourier(x_values: np.ndarray, coefficients: List[Tuple[float, float]]) -> np.ndarray:
    approximated_function = np.zeros_like(x_values)
    for n in range(len(coefficients)):
        an, bn = coefficients[n]
        approximated_function += an * np.cos(n * x_values) + bn * np.sin(n * x_values)
    return approximated_function

# Функция для построения графиков
def plot_functions(x_values: np.ndarray, original_function: np.ndarray, approximated_function: np.ndarray, num_harmonics: int) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, original_function, label='Исходная функция', linewidth=2)
    plt.plot(x_values, approximated_function, label=f'Аппроксимация Фурье ({num_harmonics} гармоник)', linestyle='--', linewidth=2)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Аппроксимация функции с {num_harmonics} гармониками Фурье')
    plt.grid(True)
    plt.show()

# Основная функция
# def main() -> None:
#     x_values = np.linspace(0, 2 * np.pi, 1000)
#     num_harmonics = 10

#     fourier_coefficients = compute_fourier_coefficients(x_values, num_harmonics)
#     approximated_function = approximate_with_fourier(x_values, fourier_coefficients)

#     plot_functions(x_values, target_function(x_values), approximated_function, num_harmonics)