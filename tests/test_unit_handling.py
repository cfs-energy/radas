"""Simple tests to make sure that the unit handling is working as expected."""

import pytest
import numpy as np
import xarray as xr

from radas.unit_handling import (
    DimensionalityError,
    ureg,
    Quantity,
    convert_units,
    magnitude,
    dimensionless_magnitude,
)


@pytest.mark.filterwarnings("error")
def test_simple_conversion():
    assert np.isclose(Quantity(1.2, ureg.m).to(ureg.cm).magnitude, 120.0)


@pytest.mark.filterwarnings("error")
def test_invalid_conversion():
    with pytest.raises(DimensionalityError):
        Quantity(1.2, ureg.m).to(ureg.W)


@pytest.mark.filterwarnings(
    "ignore:The unit of the quantity is stripped when downcasting to ndarray."
)
def test_convert_units():

    values = Quantity([1.2, 2.4], ureg.m)
    assert np.allclose(
        convert_units(values, ureg.cm).magnitude, values.magnitude * 100.0
    )

    values = xr.DataArray(Quantity([1.2, 2.4], ureg.m))
    assert np.allclose(convert_units(values, ureg.cm).values, values.values * 100.0)

    values = xr.DataArray([1.2, 2.4])
    assert np.allclose(
        convert_units(values, ureg.percent).values, values.values * 100.0
    )

    with pytest.raises(NotImplementedError):
        values = [1.2, 2.4]
        convert_units(values, ureg.percent)


@pytest.mark.filterwarnings("error")
def test_magnitude():
    values = Quantity([1.2, 2.4], ureg.m)
    assert np.allclose(
        magnitude(convert_units(values, ureg.cm)), magnitude(values) * 100.0
    )

    values = xr.DataArray(Quantity([1.2, 2.4], ureg.m))
    assert np.allclose(
        magnitude(convert_units(values, ureg.cm)), magnitude(values) * 100.0
    )

    values = xr.DataArray([1.2, 2.4])
    assert np.allclose(
        magnitude(convert_units(values, ureg.percent)), magnitude(values) * 100.0
    )

    with pytest.raises(NotImplementedError):
        values = [1.2, 2.4]
        magnitude(values)


@pytest.mark.filterwarnings("error")
def test_dimensionless_magnitude():

    values1 = Quantity([1.2, 2.4], ureg.m)
    values2 = Quantity([240.0, 240.0], ureg.cm)

    assert np.allclose(dimensionless_magnitude(values1 / values2), [0.5, 1.0])

    with pytest.raises(DimensionalityError):
        dimensionless_magnitude(values1)
