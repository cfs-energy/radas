"""Tests for the charge-state time evolution."""

import numpy as np

from tests.rate_equation_rhs import derivative_fsum, derivative_linear


def test_linear_rhs_matches_fsum_away_from_equilibrium():
    """J @ y + b agrees with a math.fsum term sum when cancellation is mild."""
    rng = np.random.default_rng(1)
    ionisation, recombination = rng.uniform(1e-16, 1e-14, (2, 8))
    ionisation[-1] = 0.0
    ne, ne_tau = 1e19, 1e17
    fraction = rng.dirichlet(np.ones(8))

    linear = derivative_linear(0.0, fraction, ionisation, recombination, ne, ne_tau)
    precise = derivative_fsum(0.0, fraction, ionisation, recombination, ne, ne_tau)
    np.testing.assert_allclose(linear, precise, rtol=1e-12, atol=0.0)
