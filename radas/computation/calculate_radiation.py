import xarray as xr
from ..unit_handling import ureg

def calculate_electron_emission_prefactor(dataset: xr.Dataset) -> xr.DataArray:
    """Calculate the radiation rate prefactor due to electron-impurity interactions.
    
    You can calculate P_rad_electron = n_e * n_z * electron_emission
    where n_z = impurity density summed over charge states
    """

    electron_emission_per_charge_state = ((dataset.line_emission_coeff + dataset.continuum_emission_coeff) * dataset.fractional_abundances)
    electron_emission = electron_emission_per_charge_state.sum(dim="dim_charge_state")

    return electron_emission.pint.to(ureg.W * ureg.m**3)
