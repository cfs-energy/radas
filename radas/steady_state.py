"""Direct solve for the equilibrium charge-state fractions."""

import math

import numpy as np
import xarray as xr
from .unit_handling import ureg, magnitude_in_units


def calculate_steady_state_fractional_abundances(dataset: xr.Dataset) -> xr.DataArray:
    """Charge-state fractions at t -> infinity of the refuelled system in time_evolution.py, solved directly.

    ne_tau = inf gives the non-refuelled (coronal) limit.
    """
    return xr.apply_ufunc(
        steady_state,
        magnitude_in_units(dataset.effective_ionisation, ureg.m**3 / ureg.s),
        magnitude_in_units(
            dataset.effective_recombination.roll(dim_charge_state=-1),
            ureg.m**3 / ureg.s,
        ),
        magnitude_in_units(dataset.ne_tau, ureg.m**-3 * ureg.s),
        vectorize=True,
        input_core_dims=[("dim_charge_state",), ("dim_charge_state",), ()],
        output_core_dims=[("dim_charge_state",)],
    ).pint.quantify("")


def steady_state(effective_ionisation, effective_recombination, ne_tau=np.inf):
    """Solve d/dt = 0 for the rate equations in time_evolution.py (dy/dt = J @ y + b), assuming the fully stripped state cannot ionise.

    The tridiagonal system is eliminated from the top charge state down without subtractions (the
    Grassmann-Taksar-Heyman form of Gaussian elimination), since the plain Thomas algorithm cancels
    catastrophically when ne_tau is large. Ratios of neighbouring states are combined in log space, so
    fractions spanning more than the double-precision range do not overflow. Particles are conserved, so
    normalising the fractions to sum to one fixes the refuelling source.
    """
    loss = 1.0 / ne_tau
    # outflow = rate out of state i+1 other than recombination to i, per unit fraction of state i+1
    outflow = loss
    rate_down = np.zeros(effective_ionisation.size - 1)
    for i in range(effective_ionisation.size - 2, -1, -1):
        rate_down[i] = effective_recombination[i] + outflow
        outflow = loss + effective_ionisation[i] * outflow / rate_down[i]
    log_ratio = (np.log(effective_ionisation[:-1]) - np.log(rate_down)).tolist()
    # math.fsum gives correctly rounded sums, so accuracy does not degrade with the number of charge states.
    log_fraction = np.array([0.0] + [math.fsum(log_ratio[: i + 1]) for i in range(len(log_ratio))])
    fraction = np.exp(log_fraction - log_fraction.max())
    return fraction / math.fsum(fraction.tolist())
