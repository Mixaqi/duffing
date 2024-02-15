from __future__ import annotations

import sympy as sy


def duffing_equation(delta: float, alpha: float, beta: float, gamma: float, omega: float) -> sy.Expr:
    t = sy.Symbol("t")
    x = sy.Function("x")(t)  #
