"""Shared right-hand-side helpers for time-evolution / steady-state tests."""

import math

import numpy as np

from radas.time_evolution import calculate_jacobian, calculate_source


def rate_equation_terms(y, ionisation, recombination, ne_tau):
    """The individual contributions to dy/dt / ne (before multiplying by electron density)."""
    n = len(y)
    terms = [[] for _ in range(n)]
    for i in range(n):
        terms[i].append(-ionisation[i] * y[i])
        terms[i].append(-y[i] / ne_tau)
        if i > 0:
            terms[i].append(ionisation[i - 1] * y[i - 1])
            terms[i].append(-recombination[i - 1] * y[i])
        if i < n - 1:
            terms[i].append(recombination[i] * y[i + 1])
        if i == 0:
            terms[i].append(1.0 / ne_tau)
    return terms


def derivative_fsum(y, ionisation, recombination, electron_density, ne_tau):
    """High-precision term sum using math.fsum (order-independent correctly rounded sum)."""
    return electron_density * np.array([math.fsum(terms) for terms in rate_equation_terms(y, ionisation, recombination, ne_tau)])


def derivative_linear(y, ionisation, recombination, electron_density, ne_tau):
    """Fast right-hand side: J @ y + b."""
    jacobian = calculate_jacobian(ionisation, recombination, electron_density, ne_tau)
    return jacobian @ y + calculate_source(electron_density, ne_tau, len(y))


def cancellation_scale(y, ionisation, recombination, ne_tau):
    """Per-charge-state sum of absolute rate terms (before multiplying by ne)."""
    return np.array([sum(abs(t) for t in terms) for terms in rate_equation_terms(y, ionisation, recombination, ne_tau)])
