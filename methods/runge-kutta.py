from __future__ import annotations

from typing import List, Tuple

import numpy as np


def f(
    t: float,
    u: List[float],
    delta: float,
    alpha: float,
    beta: float,
    gamma: float,
    omega: float,
) -> List[float]:
    x, v = u
    return [v, gamma * np.cos(omega * t) - delta * v - alpha * x - beta * x**3]


def runge_kutta(
    f,
    u0: List[float],
    t0: float,
    tn: float,
    delta: float,
    alpha: float,
    beta: float,
    gamma: float,
    omega: float,
    h: float,
) -> Tuple[np.ndarray, np.ndarray]:
    ts = [t0]
    us = [u0]
    t = t0
    u = u0

    while t < tn:
        k1 = h * np.array(f(t, u, delta, alpha, beta, gamma, omega))
        k2 = h * np.array(
            f(t + 0.5 * h, u + 0.5 * k1, delta, alpha, beta, gamma, omega)
        )
        k3 = h * np.array(
            f(t + 0.5 * h, u + 0.5 * k2, delta, alpha, beta, gamma, omega)
        )
        k4 = h * np.array(f(t + h, u + k3, delta, alpha, beta, gamma, omega))

        u = u + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += h

        us.append(u)
        ts.append(t)

    return np.array(ts), np.array(us)


t0: float = 0
x0: float = 0
v0: float = 0
u0: List[float] = [x0, v0]

# Параметры уравнения
delta: float = 0.1
alpha: float = 1.0
beta: float = 1.0
gamma: float = 0.5
omega: float = 1.0

# Временной интервал и шаг
tn: float = 10
h: float = 0.01

# Решение уравнения методом Рунге-Кутты
ts, us = runge_kutta(f, u0, t0, tn, delta, alpha, beta, gamma, omega, h)

# Шаг, на котором будем выводить результаты
output_step: int = 100  # Например, каждые 100 временных шагов

# Вывод результатов
print("t \t x \t v")
for i in range(0, len(ts), output_step):
    print(f"{ts[i]:.2f} \t {us[i, 0]:.4f} \t {us[i, 1]:.4f}")
