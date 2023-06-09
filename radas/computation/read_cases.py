from pathlib import Path
from typing import Any, Optional
import yaml
import xarray as xr
from numbers import Number
import numpy as np
from ..unit_handling import Quantity, ureg

from ..directories import cases_directory

from ..named_options import AtomicSpecies

def _list_directory(directory):
    """
    List all paths in a directory, including all files in subdirectories
    """
    files = [path for path in directory.iterdir()]

    for subdirectory in [path for path in directory.iterdir() if path.is_dir()]:
        files += _list_directory(subdirectory)
    
    return files

def list_cases() -> list[str]:
    """List all of the valid cases in the "cases" folder.
    
    For a case to be listed, it must be a folder with a "input.yaml" file in it.
    """
    return [
        str(path.relative_to(cases_directory))
        for path in _list_directory(cases_directory)
        if (path.is_dir() and (path / "input.yaml").exists())
    ]

def read_case(case: str, parameters: dict=None) -> xr.Dataset:
    """Read parameters from input.yaml for a specified case and return as a Dataset.
    
    Add a new element "case" which stores the name of the case.

    You can optional pass in a dictionary of parameters. If you don't, the routine
    will look for a subfolder in the cases directory matching "case"
    """
    if parameters is None:
        with open(cases_directory / case / "input.yaml") as file:
            parameters = yaml.load(file, Loader=yaml.FullLoader)
    
    plotting = parameters.pop("plotting")
    file_output = parameters.pop("file_output")

    ds = xr.Dataset().assign_attrs(case=case)

    units = dict(
        electron_density="m**-3",
        neutral_density="m**-3",
        electron_temperature="eV",
        ne_tau="m**-3 * s",
    )

    attrs = dict()
    for key, value in parameters.items():
        if key in ["electron_density", "neutral_density", "electron_temperature", "ne_tau"]:
            value = Quantity(np.atleast_1d(value), units[key])
            ds[key] = xr.DataArray(value, coords={f"dim_{key}": value.magnitude})
        
        elif isinstance(value, Number):
            ds[key] = xr.DataArray(float(value))

        elif isinstance(value, list):
            ds[key] = xr.DataArray(value, coords={f"dim_{key}": value})
            
        else:
            attrs[key] = value
        
    ds = ds.assign_attrs(**attrs)

    return ds, plotting, file_output

def convert_enums_for_parameters(dataset: xr.Dataset) -> xr.Dataset:
    """Convert strings to enumerators for the input parameters."""
    attrs = dict(
        species = AtomicSpecies[dataset.species],
    )

    return dataset.assign_attrs(**attrs)
