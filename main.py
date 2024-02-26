from __future__ import annotations

from methods.duffing import alpha, beta, delta, duffing_equation, gamma, omega
from methods.fourier import calculate_coefficients, f, calculate_approximation_function, harmonics_number


def main() -> None:
    equation = duffing_equation(
        delta=delta,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        omega=omega,
    )
    # print(equation) 
    approximation_function = calculate_approximation_function(harmonics_number)
    right_side = approximation_function * gamma
    print(approximation_function)
    print(right_side)


if __name__ == "__main__":
    main()
