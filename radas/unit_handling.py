"""Use the pint library for unit handling."""
import numpy as np
import pint_xarray
import pint
import warnings

ureg = pint_xarray.setup_registry(
    pint.UnitRegistry(
        force_ndarray_like=True,
    )
)

Quantity = ureg.Quantity
# Define custom units for density as n_19 or n_20
ureg.define("_1e19_per_cubic_metre = 1e19 m^-3 = n19")
ureg.define("_1e20_per_cubic_metre = 1e20 m^-3 = n20")
ureg.define("percent = 0.01")

nodim = ureg.dimensionless

# Needed for serialization/deserialization
pint.set_application_registry(ureg)

warnings.filterwarnings("ignore", message="The unit of the quantity is stripped when downcasting to ndarray.")