"""Routines for log-log interpolation of rate coefficients with boundary clipping."""
import xarray as xr
import numpy as np
from scipy.interpolate import RectBivariateSpline
from numpy.typing import NDArray
import warnings

def is_significantly_below(requested, limit):
    return requested < limit and not np.isclose(requested, limit)

def is_significantly_above(requested, limit):
    return requested > limit and not np.isclose(requested, limit)

def interpolate_array(
    array: xr.DataArray, 
    new_electron_density: NDArray[np.floating], 
    new_electron_temp: NDArray[np.floating]
) -> xr.DataArray:
    """
    Interpolate rate coefficients onto a new density/temperature grid in log-log space.
    
    Uses nearest-neighbor extrapolation by clipping out-of-bounds coordinates to 
    the original grid edges.
    """
    units = array.pint.units
    array = array.pint.dequantify().squeeze()

    # Handle zero-value edge cases (log of zero is undefined)
    if np.allclose(array, 0.0, atol=0.0, rtol=1e-6):
        return xr.DataArray(
            np.zeros((np.size(new_electron_temp), np.size(new_electron_density))),
            coords=dict(dim_electron_temp=new_electron_temp, dim_electron_density=new_electron_density)
        ) * units
    
    if np.any(array <= 0.0):
        raise NotImplementedError("Cannot log-interpolate rate coefficients containing zeros.")
    
    # Check if extrapolation is needed and raise a warning if this is the case.
    out_of_bounds_msg = []

    req_dens_min, req_dens_max = new_electron_density.min(), new_electron_density.max()
    grid_dens_min, grid_dens_max = array.dim_electron_density.min(), array.dim_electron_density.max()

    if is_significantly_below(req_dens_min, grid_dens_min) or is_significantly_above(req_dens_max, grid_dens_max):
        out_of_bounds_msg.append(
            f"Density requested [{req_dens_min:.2e}, {req_dens_max:.2e}] "
            f"exceeds grid [{grid_dens_min:.2e}, {grid_dens_max:.2e}]."
        )
    
    # Check Temperature Bounds
    req_temp_min, req_temp_max = new_electron_temp.min(), new_electron_temp.max()
    grid_temp_min, grid_temp_max = array.dim_electron_temp.min(), array.dim_electron_temp.max()

    if is_significantly_below(req_temp_min, grid_temp_min) or is_significantly_above(req_temp_max, grid_temp_max):
        out_of_bounds_msg.append(
            f"Temperature requested [{req_temp_min:.2e}, {req_temp_max:.2e}] "
            f"exceeds grid [{grid_temp_min:.2e}, {grid_temp_max:.2e}]."
        )

    if out_of_bounds_msg:
        full_msg = "Nearest-neighbour extrapolation used for off-grid values: " + " ".join(out_of_bounds_msg)
        warnings.warn(full_msg, RuntimeWarning)
    
    # ------------------------------------
    
    # Prepare original grid and data in log10 space
    x = np.log10(array.dim_electron_density)
    y = np.log10(array.dim_electron_temp)
    z = np.log10(array.transpose("dim_electron_density", "dim_electron_temp").pint.magnitude)

    # Transform target coordinates to log10
    x_interp = np.log10(new_electron_density)
    y_interp = np.log10(new_electron_temp)

    # Force nearest-neighbor extrapolation by clipping points to the grid domain
    x_clipped = np.clip(x_interp, x.min().item(), x.max().item())
    y_clipped = np.clip(y_interp, y.min().item(), y.max().item())

    # Perform spline interpolation and revert from log space
    z_interp_log = RectBivariateSpline(x, y, z)(x_clipped, y_clipped, grid=True)
    z_interp = np.power(10, z_interp_log.T)

    return xr.DataArray(
        z_interp,
        coords=dict(dim_electron_temp=new_electron_temp, dim_electron_density=new_electron_density)
    ) * units