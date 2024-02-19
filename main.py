from __future__ import annotations

import sympy as sy

from methods.duffing import duffing_equation
from methods.fourier import calculate_coefficients, f


def main() -> None:
    omega = 1
    equation = duffing_equation(delta=0.1, alpha=1, beta=1, gamma=0.5, omega=omega)
    coswt = f("t", omega=omega)
    print(equation)
    print(coswt)

if __name__ == "__main__":
    main()
