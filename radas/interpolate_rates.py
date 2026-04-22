"""Routines to interpolate a dataset of rate coefficients to higher resolution."""
import xarray as xr
import numpy as np
from scipy.interpolate import RectBivariateSpline
from numpy.typing import NDArray

def interpolate_array(array: xr.DataArray, new_electron_density: NDArray[np.floating], new_electron_temp: NDArray[np.floating]) -> xr.DataArray:
    """Interpolate array onto new values for the electron density and electron temp.
    
    The interpolation is performed for logarithmic values.

    Nearest-neighbour extrapolation is used to fill any off-grid points.
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

    # Clip the interpolation to exclude off-grid points. This is equivalent to using nearest-neighbour
    # extrapolation for off-grid values.
    x_interp_clipped = np.clip(x_interp, x.min().item(), x.max().item())
    y_interp_clipped = np.clip(y_interp, y.min().item(), y.max().item())

    z_interp = np.power(10, RectBivariateSpline(x, y, z)(x_interp_clipped, y_interp_clipped, grid=True).T)

    return xr.DataArray(z_interp,
        coords=dict(dim_electron_temp=new_electron_temp, dim_electron_density=new_electron_density)
    ) * units
