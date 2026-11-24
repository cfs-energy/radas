# ruff: noqa: D103
"""Tests for the direct steady-state charge-state solver."""

from fractions import Fraction

import numpy as np
import pytest
import xarray as xr

from radas import Quantity, run_radas_computation
from radas.cli import run_radas_cli
from radas.read_rate_coeffs import write_global_attributes
from radas.steady_state import calculate_steady_state_fractional_abundances, steady_state
from radas.time_evolution import calculate_time_evolution
from tests.rate_equation_rhs import cancellation_scale, derivative_fsum


def exact_steady_state(ionisation, recombination, ne_tau):
    """Exact rational solution of d/dt = 0, by the plain Thomas algorithm."""
    S = [Fraction(float(x)) for x in ionisation]
    R = [Fraction(float(x)) for x in recombination]
    n = len(S)
    if np.isinf(ne_tau):
        ratio, fraction = [S[i] / R[i] for i in range(n - 1)], [Fraction(1)]
    else:
        loss = 1 / Fraction(float(ne_tau))
        diagonal = [S[i] + (R[i - 1] if i else 0) + loss for i in range(n)]
        ratio, reduced = [None] * (n - 1), diagonal[-1]
        for i in range(n - 2, -1, -1):
            ratio[i] = S[i] / reduced
            reduced = diagonal[i] - R[i] * ratio[i]
        fraction = [loss / reduced]
    for r in ratio:
        fraction.append(fraction[-1] * r)
    # The refuelled solution is not normalised by hand, so the test also checks particle conservation.
    total = sum(fraction) if np.isinf(ne_tau) else 1
    return [f / total for f in fraction]


def random_rates(rng, n):
    ionisation = 10 ** rng.uniform(-90, -12, n)
    ionisation[-1] = 0.0
    return ionisation, 10 ** rng.uniform(-25, -14, n)


@pytest.mark.parametrize("ne_tau", [5e15, 5e17, 1e21, 1e30, np.inf])
def test_steady_state_matches_exact_solution(ne_tau):
    rng = np.random.default_rng(1)
    for n in [2, 5, 19, 75]:
        for _ in range(5):
            ionisation, recombination = random_rates(rng, n)
            fraction = steady_state(ionisation, recombination, ne_tau)
            exact = exact_steady_state(ionisation, recombination, ne_tau)
            peak = max(exact)
            for computed, reference in zip(fraction, exact, strict=True):
                if reference > 1e-10 * peak:
                    assert abs(Fraction(float(computed)) - reference) / reference < 1e-12


def test_infinite_ne_tau_is_detailed_balance():
    rng = np.random.default_rng(2)
    ionisation, recombination = random_rates(rng, 6)
    fraction = steady_state(ionisation, recombination, np.inf)
    np.testing.assert_allclose(fraction[1:] / fraction[:-1], ionisation[:-1] / recombination[:-1], rtol=1e-13)
    np.testing.assert_allclose(steady_state(ionisation, recombination, 1e40), fraction, rtol=1e-12)


def synthetic_dataset(n=4, wide_rates=False, evolution_stop=1e2, ne_tau=(0.5e16, 0.5e17, 0.5e18)):
    """A small rate dataset with the variables run_radas_computation needs.

    wide_rates spans three decades of rate in each of ionisation and recombination, instead of one.
    """
    rng = np.random.default_rng(3)
    dims = ("dim_charge_state", "dim_electron_temp", "dim_electron_density")
    shape = (n, 3, 2)
    if wide_rates:
        ionisation = 10 ** rng.uniform(-16, -13, shape)
        recombination = 10 ** rng.uniform(-17, -14, shape)
    else:
        ionisation = rng.uniform(1e-15, 1e-14, shape)
        recombination = rng.uniform(1e-16, 1e-15, shape)
    ionisation[-1] = 0.0

    def rate(values, units):
        return xr.DataArray(Quantity(values, units), dims=dims)

    dataset = xr.Dataset(
        dict(
            effective_ionisation=rate(ionisation, "m**3/s"),
            effective_recombination=rate(recombination, "m**3/s"),
            line_emission_from_excitation=rate(rng.uniform(1e-33, 1e-32, shape), "W m**3"),
            recombination_and_bremsstrahlung=rate(rng.uniform(1e-34, 1e-33, shape), "W m**3"),
            electron_density=xr.DataArray(Quantity([1e19, 1e20], "m**-3"), dims="dim_electron_density"),
        ),
        coords=dict(dim_charge_state=np.arange(n)),
        attrs=dict(atomic_number=n - 1, species_name="synthetic"),
    )
    return write_global_attributes(
        dataset,
        dict(
            evolution_start=dict(value=1e-8, units="s"),
            evolution_stop=dict(value=evolution_stop, units="s"),
            ne_tau=dict(value=list(ne_tau), units="m^-3 s"),
        ),
    )


def test_time_evolution_is_opt_in_and_converges_to_steady_state(tmp_path):
    run_radas_computation(synthetic_dataset(), tmp_path / "default", 0)
    run_radas_computation(synthetic_dataset(), tmp_path / "evolved", 0, time_evolution=True)
    default = xr.open_dataset(tmp_path / "default" / "synthetic.nc")
    evolved = xr.open_dataset(tmp_path / "evolved" / "synthetic.nc")

    assert "charge_state_evolution" not in default and "dim_time" not in default.dims
    assert np.all(np.isfinite(default.equilibrium_charge_state_fraction))
    np.testing.assert_allclose(
        evolved.charge_state_evolution.isel(dim_time=-1),
        default.equilibrium_charge_state_fraction,
        rtol=1e-2,
        atol=1e-8,
    )
    np.testing.assert_allclose(default.equilibrium_charge_state_fraction.sum("dim_charge_state"), 1.0, rtol=1e-12)
    xr.testing.assert_identical(default.equilibrium_Lz, evolved.equilibrium_Lz)


@pytest.mark.parametrize("ne_tau", [5e15, 5e17, 1e19, 1e21, np.inf])
def test_steady_state_has_zero_time_evolution_derivative(ne_tau):
    """The time-evolution right-hand side vanishes at the steady state, relative to the size of its terms."""
    rng = np.random.default_rng(4)
    for n in [3, 8, 19]:
        for _ in range(5):
            ionisation, recombination = random_rates(rng, n)
            fraction = steady_state(ionisation, recombination, ne_tau)
            derivative = derivative_fsum(fraction, ionisation, recombination, 1.0, ne_tau)
            scale = cancellation_scale(fraction, ionisation, recombination, ne_tau)
            assert np.all(np.abs(derivative) <= 1e-10 * np.maximum(scale, 1e-300))


@pytest.mark.parametrize("n", [2, 5, 12])
def test_steady_state_matches_converged_time_evolution(n):
    # The slowest timescale is far below evolution_stop, so the time evolution has converged.
    dataset = synthetic_dataset(n, wide_rates=True, evolution_stop=1e3, ne_tau=(1e16, 1e18, 1e19))

    evolved = calculate_time_evolution(dataset).isel(dim_time=-1)
    direct = calculate_steady_state_fractional_abundances(dataset)

    evolved = evolved.transpose(*direct.dims).pint.dequantify().values
    direct = direct.pint.dequantify().values

    # assert_allclose treats NaN as equal to NaN, so check the inputs are sane first.
    assert np.all(np.isfinite(evolved)) and np.all(np.isfinite(direct))
    np.testing.assert_allclose(evolved.sum(axis=-1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(direct.sum(axis=-1), 1.0, rtol=1e-12)
    # The solver tolerance is rtol=1e-3 per step, but the converged state agrees far better; the floor is for rare states.
    np.testing.assert_allclose(evolved, direct, rtol=1e-6, atol=1e-12)


def test_cli_exposes_time_evolution_flag():
    from click.testing import CliRunner

    assert "--time-evolution" in CliRunner().invoke(run_radas_cli, ["--help"]).output
