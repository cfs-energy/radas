import numpy as np
import xarray as xr
from scipy.integrate import solve_ivp

from .kahan_summation import kahan_babushka_neumaier_sum

def shift(arr, num, fill_value=0.0):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def calculate_derivative(_, charge_state_fraction: np.ndarray, ionisation_rate_coeff, recombination_rate_coeff, electron_density, ne_tau=np.inf):
    """Calculate the the partial derivative of the impurity density w.r.t. time.

    A compensated sum is used to reduce the risk of truncation. If you think this is affecting performance, you can replace
    the last lines with
    dydt = np.sum([-ionisation_to_above, ionisation_from_below, recombination_from_above, -recombination_to_below], axis=0)

    The default option is for a non-refuelled impurity. If you set a ne_tau, it is assumed that the ground state is
    constantly refuelled at a rate of 1 / ne_tau and that the excited states
    are lost at a rate proportional to their concentration.
    """
    # Prevent negative fractions (can happen due to numerical errors)
    # charge_state_fraction = np.maximum(charge_state_fraction, 0.0)

    ionisation_to_above = ionisation_rate_coeff * charge_state_fraction
    ionisation_from_below = shift(ionisation_rate_coeff * charge_state_fraction, +1)

    recombination_from_above = recombination_rate_coeff * shift(charge_state_fraction, -1)
    recombination_to_below = shift(recombination_rate_coeff, +1) * charge_state_fraction

    change_in_charge_state_fraction = np.zeros_like(charge_state_fraction)
    for i in range(len(change_in_charge_state_fraction)):
        change_in_charge_state_fraction[i] = kahan_babushka_neumaier_sum([-ionisation_to_above[i], ionisation_from_below[i], recombination_from_above[i], -recombination_to_below[i]])
    
    change_in_charge_state_fraction -= charge_state_fraction/ne_tau
    change_in_charge_state_fraction[0] += 1.0/ne_tau

    return change_in_charge_state_fraction * electron_density

def calculate_time_evolution(dataset: xr.Dataset) -> xr.Dataset:
    """Evolve the system over time, and record the impurity charge-state fractions as a function of time.
    
    The equations are stiff, so we need to use "BDF", "Radau" or "LSODA" as the solver method. Radau was
    found to give a good balance of accuracy and speed.
    """
    evaluation_times = np.logspace(np.log10(dataset.evolution_start), np.log10(dataset.evolution_stop))

    def _time_evolve(ionisation_rate_coeff, recombination_rate_coeff, electron_density, ne_tau):
        charge_state_fraction = np.zeros_like(ionisation_rate_coeff)
        charge_state_fraction[0] = 1.0

        result = solve_ivp(
            calculate_derivative,
            y0 = charge_state_fraction,
            t_span=[evaluation_times[0], evaluation_times[-1]],
            t_eval=evaluation_times,
            args = (ionisation_rate_coeff, recombination_rate_coeff, electron_density, ne_tau),
            method="Radau",
            rtol=1E-3,
            atol=1E-12,
        )

        return result.y

    charge_state_fraction = xr.apply_ufunc(
        _time_evolve,
        dataset.ionisation_rate_coeff,
        dataset.recombination_rate_coeff.roll(dim_charge_state=-1),
        dataset.electron_density,
        dataset.ne_tau,
        vectorize=True,
        input_core_dims=[("dim_charge_state",), ("dim_charge_state",), (), ()],
        output_core_dims=[("dim_charge_state", "dim_time")],
    ).assign_coords(dim_time=evaluation_times)

    return charge_state_fraction
