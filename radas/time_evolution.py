import numpy as np
import xarray as xr
from scipy.integrate import solve_ivp
from .unit_handling import ureg, magnitude_in_units


def calculate_time_evolution(dataset: xr.Dataset) -> xr.Dataset:
    """Evolve the system over time, and record the impurity charge-state fractions as a function of time.

    The equations are stiff, so we need to use "BDF", "Radau" or "LSODA" as the solver method. The rates are
    constant in time, so the exact (tridiagonal) Jacobian is supplied; LSODA with this Jacobian is much faster
    than Radau, at the same accuracy. The system is linear, so the right-hand side is J @ y + b.
    """
    evaluation_times = np.logspace(
        np.log10(magnitude_in_units(dataset.evolution_start, ureg.s)).item(),
        np.log10(magnitude_in_units(dataset.evolution_stop, ureg.s)).item(),
    )

    def _time_evolve(
        effective_ionisation,
        effective_recombination,
        electron_density,
        ne_tau,
    ):
        charge_state_fraction = np.zeros_like(effective_ionisation)
        charge_state_fraction[0] = 1.0

        jacobian = calculate_jacobian(
            effective_ionisation, effective_recombination, electron_density, ne_tau
        )
        source = np.zeros_like(charge_state_fraction)
        source[0] = electron_density / ne_tau

        result = solve_ivp(
            lambda _t, y: jacobian @ y + source,
            y0=charge_state_fraction,
            t_span=[evaluation_times[0], evaluation_times[-1]],
            t_eval=evaluation_times,
            method="LSODA",
            jac=lambda *_: jacobian,
            rtol=1e-3,
            atol=1e-12,
        )

        return result.y

    charge_state_fraction = xr.apply_ufunc(
        _time_evolve,
        magnitude_in_units(dataset.effective_ionisation, ureg.m**3 / ureg.s),
        magnitude_in_units(
            dataset.effective_recombination.roll(dim_charge_state=-1),
            ureg.m**3 / ureg.s,
        ),
        magnitude_in_units(dataset.electron_density, ureg.m**-3),
        magnitude_in_units(dataset.ne_tau, ureg.m**-3 * ureg.s),
        vectorize=True,
        input_core_dims=[("dim_charge_state",), ("dim_charge_state",), (), ()],
        output_core_dims=[("dim_charge_state", "dim_time")],
    ).assign_coords(dim_time=evaluation_times).pint.quantify("")

    return charge_state_fraction


def calculate_jacobian(
    effective_ionisation,
    effective_recombination,
    electron_density,
    ne_tau=np.inf,
):
    """Jacobian of dy/dt = J @ y + b for the charge-state rate equations.

    The system is linear, so this is constant in time and tridiagonal. Index k is the
    charge state: ionisation connects k -> k+1, recombination connects k+1 -> k.
    """
    diagonal = -effective_ionisation - 1.0 / ne_tau
    diagonal[1:] -= effective_recombination[:-1]

    return electron_density * (
        np.diag(diagonal)
        + np.diag(effective_ionisation[:-1], -1)
        + np.diag(effective_recombination[:-1], +1)
    )


def calculate_source(electron_density, ne_tau, n_charge_states):
    """Inhomogeneous term b in dy/dt = J @ y + b (ground-state refuelling)."""
    source = np.zeros(n_charge_states)
    source[0] = electron_density / ne_tau
    return source
