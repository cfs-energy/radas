import xarray as xr
from .unit_handling import dimensionless_magnitude


def calculate_coronal_fractional_abundances(dataset: xr.Dataset) -> xr.DataArray:
    """Calculate the fractional abundances of different charge states, assuming coronal equilibrium."""

    ratio_of_ionisation_to_recombination = (
        dataset.effective_ionisation
        / dataset.effective_recombination.roll(dim_charge_state=-1)
    )
    charge_state_fraction = xr.zeros_like(ratio_of_ionisation_to_recombination)
    ratio_of_ionisation_to_recombination = dimensionless_magnitude(
        ratio_of_ionisation_to_recombination
    )

    charge_state_fraction[0] = 1.0

    for charge_state in range(dataset.atomic_number):
        charge_state_fraction[charge_state + 1] = (
            charge_state_fraction[charge_state]
            * ratio_of_ionisation_to_recombination[charge_state]
        )

    charge_state_fraction /= charge_state_fraction.sum(dim="dim_charge_state")

    return charge_state_fraction
