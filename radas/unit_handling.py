"""Set up the pint library for unit handling."""

import warnings
from functools import wraps
from typing import Any, Union

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


def suppress_downcast_warning(func):
    """Suppresses a common warning about downcasting quantities to arrays."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The unit of the quantity is stripped when downcasting to ndarray.",
            )
            return func(*args, **kwargs)

    return wrapper


def convert_units(
    array: Union[xr.DataArray, pint.Quantity], units: Any
) -> Union[xr.DataArray, pint.Quantity]:
    """Convert an array to specified units, handling both Quantities and xr.DataArrays."""
    if isinstance(array, xr.DataArray):
        if not hasattr(array.pint, "units") or array.pint.units is None:
            array = array.pint.quantify(ureg.dimensionless)

        return array.pint.to(units)  # type: ignore[no-any-return]
    elif isinstance(array, Quantity):
        return array.to(units)  # type:ignore[no-any-return]
    else:
        raise NotImplementedError(
            f"No implementation for 'convert_units' with an array of type {type(array)} ({array})"
        )


@suppress_downcast_warning
def magnitude(
    array: Union[xr.DataArray, pint.Quantity]
) -> Union[npt.NDArray[np.float32], float]:
    """Return the magnitude of an array, handling both Quantities and xr.DataArrays."""
    if isinstance(array, xr.DataArray):
        return array.pint.dequantify()
    elif isinstance(array, Quantity):
        return array.magnitude  # type: ignore[no-any-return]
    else:
        raise NotImplementedError(
            f"No implementation for 'magnitude' with an array of type {type(array)} ({array})"
        )


def dimensionless_magnitude(
    array: Union[xr.DataArray, pint.Quantity]
) -> Union[npt.NDArray[np.float32], float]:
    """Converts the array to dimensionless and returns the magnitude."""
    return magnitude(convert_units(array, ureg.dimensionless))


__all__ = [
    "DimensionalityError",
    "UnitStrippedWarning",
    "ureg",
    "Quantity",
    "convert_units",
    "magnitude",
    "suppress_downcast_warning",
    "dimensionless_magnitude",
]
