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

def calculate_derivative(_, y: np.ndarray, ionisation_rate_coeff, recombination_rate_coeff, electron_density, refuelling_time=np.inf):
    """Calculate the the partial derivative of the impurity density w.r.t. time.

    A compensated sum is used to reduce the risk of truncation. If you think this is affecting performance, you can replace
    the last lines with
    dydt = np.sum([-ionisation_to_above, ionisation_from_below, recombination_from_above, -recombination_to_below], axis=0)

    The default option is for a non-refuelled impurity. If you set a refuelling_time, it is assumed that the ground state is
    constantly refuelled at a rate of impurity_density / (refuelling_time * electron_density) and that the excited states
    are lost at a rate proportional to their concentration.
    """
    ionisation_rate_coeff = np.nan_to_num(ionisation_rate_coeff)
    recombination_rate_coeff = np.nan_to_num(recombination_rate_coeff)

    ionisation_to_above = ionisation_rate_coeff * y
    ionisation_from_below = shift(ionisation_rate_coeff * y, +1)

    recombination_from_above = recombination_rate_coeff * shift(y, -1)
    recombination_to_below = shift(recombination_rate_coeff, +1) * y

    dydt = np.zeros_like(y)
    for i in range(len(dydt)):
        dydt[i] = kahan_babushka_neumaier_sum([-ionisation_to_above[i], ionisation_from_below[i], recombination_from_above[i], -recombination_to_below[i]])
    
    ne_tau = electron_density * refuelling_time
    dydt -= y/ne_tau
    dydt[0] += np.sum(y)/ne_tau

    return dydt * electron_density

def calculate_time_evolution(dataset: xr.Dataset) -> xr.Dataset:
    """Evolve the system over time, and record the impurity density as a function of time."""
    evaluation_times = np.logspace(np.log10(dataset.evolution_start), np.log10(dataset.evolution_stop))

    def _time_evolve(ionisation_rate_coeff, recombination_rate_coeff, electron_density, impurity_density, refuelling_rate):
        y = np.zeros_like(ionisation_rate_coeff)
        y[0] = impurity_density

        result = solve_ivp(
            calculate_derivative,
            y0 = y,
            t_span=[evaluation_times[0], evaluation_times[-1]],
            t_eval=evaluation_times,
            args = (ionisation_rate_coeff, recombination_rate_coeff, electron_density, refuelling_rate),
            method="BDF",
        )

        return result.y

    impurity_density = xr.apply_ufunc(
        _time_evolve,
        dataset.ionisation_rate_coeff,
        dataset.recombination_rate_coeff,
        dataset.electron_density,
        dataset.impurity_density,
        dataset.refuelling_rate,
        vectorize=True,
        input_core_dims=[("dim_charge_state",), ("dim_charge_state",), (), (), ()],
        output_core_dims=[("dim_charge_state", "dim_time")],
    ).assign_coords(dim_time=evaluation_times)

    return impurity_density
