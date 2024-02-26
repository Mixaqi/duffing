from __future__ import annotations

import numpy as np
from scipy.optimize import newton


def func(x, h, i):
    n = len(x)
    return (x[(i-1) % n] - 2*x[i] + x[(i+1) % n])/h**2 + 0.05*(x[(i+1) % n] - x[(i-1) % n])/h + x[i] + x[i]**3 - 0.5*np.cos(i*h)


def equations(x, h):
    n = len(x)
    F = np.zeros(n)
    for i in range(1, n-1):
        F[i] = func(x, h, i)
    return F

def solve_equations(n, h):
    x_guess = np.zeros(n)
    x_guess[0] = x_guess[-1] = 0  # Граничные условия x(0) = x(2*pi) = 0
    v_guess = np.zeros(n)         # Предполагаем, что производные в начале и конце равны 0
    initial_guess = np.concatenate((x_guess, v_guess))
    result = newton(equations, initial_guess, args=(h,))
    return result[:n]

if __name__ == "__main__":
    n = 3
    h = 2*np.pi/n
    solution = solve_equations(n, h)
    print("Solution:", solution)
