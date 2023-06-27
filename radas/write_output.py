import xarray as xr
import numpy as np
import sys
from enum import Enum
from .unit_handling import Quantity
from .directories import cases_directory

def write_output(dataset: xr.Dataset, file_output: dict):

    write_netcdf = file_output.pop("netcdf")
    if write_netcdf:
        write_dataset_to_netcdf(dataset)

def write_dataset_to_netcdf(dataset: xr.Dataset):
    """Write the dataset to a netcdf file for later analysis."""

    serializable_dataset = xr.Dataset()

    attrs = dict()
    for key, val in dataset.attrs.items():
        if isinstance(val, bool):
            attrs[key] = int(val)
        elif isinstance(val, (float, np.ndarray, int, str, tuple, list)):
            attrs[key] = val
        elif isinstance(val, Quantity):
            attrs[key] = str(val)
        elif isinstance(val, Enum):
            attrs[key] = str(val)
        elif callable(val) or val is None:
            # Can't serialize these types
            pass
        else:
            print(f"Could not serialize {key} of type {type(val)}")

    serializable_dataset = serializable_dataset.assign_attrs(**attrs)

    for key in dataset.keys():
        serializable_dataset[key] = dataset[key].pint.dequantify()
    
    if not "pytest" in sys.modules: #skip saving output if running tests
        serializable_dataset.to_netcdf(cases_directory / dataset.case / "output" / f"{dataset.case}.nc")
