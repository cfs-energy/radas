from scipy.interpolate import RectBivariateSpline
import numpy as np
import xarray as xr

def _interpolate_rate_coefficient(electron_density_grid, electron_temperature_grid, rate_coeff_for_charge_state, electron_density_point, electron_temperature_point):
    """Interpolate a rate coefficient to the values specified.

    The interpolation is performed on logarithmic quantities, but this conversion is performed internally so you should pass
    non-logarithmic values in.
    """
    interpolator = RectBivariateSpline(np.log10(electron_density_grid), np.log10(electron_temperature_grid), np.log10(rate_coeff_for_charge_state))

    return np.power(10, interpolator(np.log10(electron_density_point), np.log10(electron_temperature_point), grid=True))

def interpolate_rate_coefficient(rate_coefficient_dataset: xr.Dataset, electron_density: xr.DataArray, electron_temperature: xr.DataArray) -> xr.DataArray:
    """Interpolate a rate coefficient to a grid of the electron_density and electron_temperature values specified.
    """
    # Retrieve the reference density and electron temperature from the dataset
    reference_electron_density = rate_coefficient_dataset.reference_electron_density
    reference_electron_temperature = rate_coefficient_dataset.reference_electron_temperature

    # Temporarily rename the dimensions in the arrays to lookup for
    electron_density = electron_density.rename(dim_electron_density="dim_electron_density_lookup")
    electron_temperature = electron_temperature.rename(dim_electron_temperature="dim_electron_temperature_lookup")

    # Perform the interpolation, automatically looping over any dimension which isn't listed
    interpolated_rate_coefficient = xr.apply_ufunc(
        _interpolate_rate_coefficient,
        rate_coefficient_dataset.electron_density / reference_electron_density,
        rate_coefficient_dataset.electron_temperature / reference_electron_temperature,
        rate_coefficient_dataset.rate_coefficient,
        electron_density / reference_electron_density,
        electron_temperature / reference_electron_temperature,
        vectorize=True,
        input_core_dims=[
            ("dim_electron_density",),
            ("dim_electron_temperature",),
            ("dim_electron_density", "dim_electron_temperature"),
            (*electron_density.dims,),
            (*electron_temperature.dims,),
        ],
        output_core_dims=[
            (*electron_density.dims, *electron_temperature.dims),
        ]
    ).rename(dim_electron_density_lookup="dim_electron_density", dim_electron_temperature_lookup="dim_electron_temperature")

    return interpolated_rate_coefficient.pint.quantify(rate_coefficient_dataset.rate_coefficient.pint.units)
