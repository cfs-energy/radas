"""Set up the pint library for unit handling."""

from typing import Any, overload

import numpy as np
import numpy.typing as npt
import pint
import pint_xarray
import xarray as xr
from pint import UnitStrippedWarning, DimensionalityError

ureg = pint_xarray.setup_registry(
    pint.UnitRegistry(
        force_ndarray_like=True,
    )
)
pint.set_application_registry(ureg)

Quantity = ureg.Quantity


@overload
def convert_units(array: xr.DataArray, units: Any) -> xr.DataArray: ...
@overload
def convert_units(array: pint.Quantity, units: Any) -> pint.Quantity: ...
def convert_units(array: xr.DataArray | pint.Quantity, units: Any) -> xr.DataArray | pint.Quantity:
    """Convert an array to specified units, handling both Quantities and xr.DataArrays."""
    if isinstance(array, xr.DataArray):
        if not hasattr(array.pint, "units") or array.pint.units is None:
            array = array.pint.quantify(ureg.dimensionless)

        return array.pint.to(units)  # type: ignore[no-any-return]
    elif isinstance(array, Quantity):
        return array.to(units)  # type:ignore[no-any-return]
    else:
        raise NotImplementedError(f"No implementation for 'convert_units' with an array of type {type(array)} ({array})")


@overload
def magnitude(array: xr.DataArray) -> xr.DataArray: ...
@overload
def magnitude(array: pint.Quantity) -> npt.NDArray[np.float32] | float: ...
def magnitude(array: xr.DataArray | pint.Quantity) -> xr.DataArray | npt.NDArray[np.float32] | float:
    """Return the magnitude of an array, handling both Quantities and xr.DataArrays."""
    if isinstance(array, xr.DataArray):
        return array.pint.dequantify()
    elif isinstance(array, Quantity):
        return array.magnitude  # type: ignore[no-any-return]
    else:
        raise NotImplementedError(f"No implementation for 'magnitude' with an array of type {type(array)} ({array})")


@overload
def dimensionless_magnitude(array: xr.DataArray) -> xr.DataArray: ...
@overload
def dimensionless_magnitude(array: pint.Quantity) -> npt.NDArray[np.float32] | float: ...
def dimensionless_magnitude(array: xr.DataArray | pint.Quantity) -> xr.DataArray | npt.NDArray[np.float32] | float:
    """Converts the array to dimensionless and returns the magnitude."""
    return magnitude(convert_units(array, ureg.dimensionless))


@overload
def magnitude_in_units(array: xr.DataArray, units: Any) -> xr.DataArray: ...
@overload
def magnitude_in_units(array: pint.Quantity, units: Any) -> npt.NDArray[np.float32] | float: ...
def magnitude_in_units(array: xr.DataArray | pint.Quantity, units: Any) -> xr.DataArray | npt.NDArray[np.float32] | float:
    """Convert the array to the specified units and then return the magnitude."""
    return magnitude(convert_units(array, units))


__all__ = [
    "DimensionalityError",
    "Quantity",
    "UnitStrippedWarning",
    "convert_units",
    "dimensionless_magnitude",
    "magnitude",
    "ureg",
]
