import numpy as np
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

def calculate_derivative(_, y: np.ndarray, ionisation_rate_coeff, recombination_rate_coeff, electron_density):
    """Calculate the the partial derivative of the impurity density w.r.t. time.

    A compensated sum is used to reduce the risk of truncation. If you think this is affecting performance, you can replace
    the last lines with
    dydt = np.sum([-ionisation_to_above, ionisation_from_below, recombination_from_above, -recombination_to_below], axis=0)
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

    return dydt * electron_density

