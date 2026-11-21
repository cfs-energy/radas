import numpy as np

from radas.time_evolution import calculate_derivative, calculate_jacobian


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(0)
    ionisation, recombination = rng.uniform(1e-16, 1e-14, (2, 6))
    ne, ne_tau = 1e19, 1e17
    fraction = rng.uniform(0.0, 1.0, 6)

    jacobian = calculate_jacobian(ionisation, recombination, ne, ne_tau)

    # The derivative is linear in the fractions, so central differences are exact up to rounding.
    for i in range(6):
        step = np.zeros(6)
        step[i] = 1e-3
        column = (
            calculate_derivative(0.0, fraction + step, ionisation, recombination, ne, ne_tau)
            - calculate_derivative(0.0, fraction - step, ionisation, recombination, ne, ne_tau)
        ) / (2e-3)
        np.testing.assert_allclose(jacobian[:, i], column, rtol=1e-6, atol=1e-9 * abs(jacobian).max())
