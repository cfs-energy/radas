import xarray as xr

def calculate_coronal_states(dataset: xr.Dataset) -> xr.DataArray:
    """Calculate the fractional abundances of different charge states, assuming coronal equilibrium."""

    ratio_of_ionisation_to_recombination = dataset.ionisation_rate_coeff / dataset.recombination_rate_coeff
    fractional_abundances = xr.zeros_like(ratio_of_ionisation_to_recombination)

    fractional_abundances[0] = 1.0

    for charge_state in range(dataset.atomic_number):
        fractional_abundances.loc[dict(dim_charge_state=charge_state+1)] = \
            fractional_abundances.sel(dim_charge_state=charge_state) * ratio_of_ionisation_to_recombination.sel(dim_charge_state=charge_state)

    fractional_abundances /= fractional_abundances.sum(dim="dim_charge_state")

    return fractional_abundances
