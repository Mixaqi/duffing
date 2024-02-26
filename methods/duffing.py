from __future__ import annotations

import sympy as sy

delta = 0.1
alpha = 1
beta = 1
gamma = 0.5
omega = 1


def duffing_equation(
    delta: float,
    alpha: float,
    beta: float,
    gamma: float,
    omega: float,
) -> sy.Expr:
    t = sy.Symbol("t")
    x = sy.Function("x")(t)
    equation = (
        sy.diff(x, t, 2)
        + delta * sy.diff(x, t)
        + alpha * x
        + beta * sy.diff(x, t, 3)
        - gamma * sy.cos(omega * t)
    )
    return equation
