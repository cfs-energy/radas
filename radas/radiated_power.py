import xarray as xr
from radas.unit_handling import convert_units, ureg


def calculate_Lz(
    dataset: xr.Dataset, charge_state_fraction: xr.DataArray
) -> xr.DataArray:
    """Calculate the radiated power prefactor due to electron-impurity interactions.

    You can calculate P_rad_electron = n_e * n_z * electron_emission
    where n_z = impurity density summed over charge states
    """

    electron_emission_per_charge_state = (
        dataset.line_emission_from_excitation + dataset.recombination_and_bremsstrahlung
    ) * charge_state_fraction
    electron_emission = electron_emission_per_charge_state.sum(dim="dim_charge_state")

    return convert_units(electron_emission, ureg.W * ureg.m**3)
