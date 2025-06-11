"""Routines to interpolate a dataset of rate coefficients to higher resolution."""
import xarray as xr
import numpy as np
from scipy.interpolate import RectBivariateSpline
from numpy.typing import NDArray

def interpolate_array(array: xr.DataArray, new_electron_density: NDArray[np.floating], new_electron_temp: NDArray[np.floating]) -> xr.DataArray:
    """Interpolate array onto new values for the electron density and electron temp.
    
    The interpolation is performed for logarithmic values.
    """
    units = array.pint.units
    array = array.pint.dequantify().squeeze()

    if np.allclose(array, 0.0, atol=0.0, rtol=1e-6):
        # If all values of the array are zero, return a zero array.
        return xr.DataArray(np.zeros((np.size(new_electron_temp), np.size(new_electron_density))),
            coords=dict(dim_electron_temp=new_electron_temp, dim_electron_density=new_electron_density)
        ) * units
    elif np.any(array <= 0.0):
        # If only some of the values of the array are zero, raise an error.
        raise NotImplementedError("Cannot handle zero-valued entries in non-zero rate coefficients.")
    
    x = np.log10(array.dim_electron_density)
    y = np.log10(array.dim_electron_temp)
    z = np.log10(array.transpose("dim_electron_density", "dim_electron_temp").pint.magnitude)

    x_interp = np.log10(new_electron_density)
    y_interp = np.log10(new_electron_temp)
    z_interp = np.power(10, RectBivariateSpline(x, y, z)(x_interp, y_interp, grid=True).T)

    return xr.DataArray(z_interp,
        coords=dict(dim_electron_temp=new_electron_temp, dim_electron_density=new_electron_density)
    ) * units

def interpolate_dataset(dataset: xr.Dataset, electron_density_resolution: int, electron_temp_resolution: int) -> xr.Dataset:
    """Interpolate all rate coefficients in a dataset."""
    new_electron_density = np.logspace(
        np.log10(dataset["dim_electron_density"].min().item()),
        np.log10(dataset["dim_electron_density"].max().item()),
        num = electron_density_resolution
    )

    new_electron_temp = np.logspace(
        np.log10(dataset["dim_electron_temp"].min().item()),
        np.log10(dataset["dim_electron_temp"].max().item()),
        num = electron_temp_resolution
    )

    new_dataset = xr.Dataset().assign_attrs(dataset.attrs)

    for key, array in dataset.items():
        if key in [
            "electron_density",
            "electron_temp",
        ]:
            # Don't copy in the coordinate arrays which we'll interpolate
            continue
        elif key in [
            "ne_tau"
        ]:
            # Directly copy in the coordinate arrays which we'll leave unchanged
            new_dataset[key] = array
        elif array.ndim == 0:
            # Directly copy in scalar arrays
            new_dataset[key] = array
        elif (("dim_electron_density" in array.coords)
              and ("dim_electron_temp" in array.coords)
              and ("dim_charge_state" in array.coords)):
            # For each charge state, interpolate the rate coefficient
            new_dataset[key] = array.groupby("dim_charge_state").map(interpolate_array, args=(new_electron_density, new_electron_temp))
        else:
            raise NotImplementedError(f"Could not process array '{key}' with coords {array.coords}")
    
    new_dataset["electron_density"] = xr.DataArray(new_electron_density, dims="dim_electron_density") * dataset["electron_density"].pint.units
    new_dataset["electron_temp"] = xr.DataArray(new_electron_temp, dims="dim_electron_temp") * dataset["electron_temp"].pint.units
    
    return new_dataset