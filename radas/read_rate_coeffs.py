from .unit_handling import Quantity, ureg, convert_units, dimensionless_magnitude
from .adas_interface.determine_adas_dataset_type import (
    determine_reader_class_and_config,
)
from .shared import get_git_revision_short_hash
from importlib.metadata import version, PackageNotFoundError
import datetime
import xarray as xr
import numpy as np

reference_electron_density = Quantity(1.0, ureg.m**-3)
reference_electron_temp = Quantity(1.0, ureg.eV)

def read_rate_coeff(data_file_dir, species_name, config):
    """Builds a rate_dataset combining all of the raw data available for a given species."""
    config_for_species = config["species"][species_name]

    try:
        radas_version=version("radas")
    except PackageNotFoundError:
        radas_version="UNDEFINED"
    
    rate_coefficients = dict()

    for dataset_type in config_for_species["data_files"].keys():
        reader_key, dataset_config = determine_reader_class_and_config(
            config["data_file_config"], dataset_type
        )

        if reader_key == "adf11":
            rate_dataset = build_adf11_rate_dataset(
                data_file_dir,
                species_name,
                dataset_type,
                dataset_config,
            )

        else:
            raise NotImplementedError(
                f"No implementation for reading {reader_key} files."
            )

        rate_coefficients[dataset_type] = rate_dataset.rate_coefficient
    
    dataset = xr.merge([v.rename(k) for k, v in rate_coefficients.items()],
                       join=config_for_species.get("join_method", "exact"))
    dataset["electron_density"] = dataset["dim_electron_density"] * reference_electron_density
    dataset["electron_temp"] = dataset["dim_electron_temp"] * reference_electron_temp

    dataset = align_rates_on_charge_states(dataset)

    dataset = dataset.assign_attrs(
        atomic_number=config_for_species["atomic_number"],
        species_name=species_name,
        git_hash=get_git_revision_short_hash(),
        radas_version=radas_version,
        created=datetime.date.today().strftime("%Y-%b-%d"),
    )

    dataset = write_global_attributes(dataset, config["globals"])

    return dataset


def write_global_attributes(dataset: xr.Dataset, globals: dict) -> xr.Dataset:
    for attribute, value in globals.items():
        if isinstance(value, dict):
            if np.ndim(value["value"]) >= 1:
                dataset[attribute] = xr.DataArray(
                    Quantity(value["value"], value["units"]),
                    coords={f"dim_{attribute}": value["value"]},
                )
            else:
                dataset[attribute] = Quantity(value["value"], value["units"])
        else:
            dataset[attribute] = value
    
    return dataset


def build_adf11_rate_dataset(
    data_file_dir, species_name, dataset_type, dataset_config
):
    from .adas_interface.read_adf11_file import read_adf11_file

    data = read_adf11_file(data_file_dir, species_name, dataset_type)

    ds = xr.Dataset()

    ds["species"] = species_name
    ds["dataset"] = dataset_type
    ds["charge"] = data["IZMAX"]

    electron_density = convert_units(
        Quantity(10 ** data["DDENSD"][: data["IDMAXD"]], ureg.cm**-3), ureg.m**-3
    )
    electron_temp = Quantity(10 ** data["DTEVD"][: data["ITMAXD"]], ureg.eV)

    coefficient = data["DRCOFD"][: data["IZMAX"], : data["ITMAXD"], : data["IDMAXD"]]
    if dataset_config["code"] <= 9:
        coefficient = 10**coefficient

    input_units = dataset_config["stored_units"]

    dim_electron_density = dimensionless_magnitude(electron_density / reference_electron_density)
    dim_electron_temp = dimensionless_magnitude(electron_temp / reference_electron_temp)
    dim_charge_state = np.arange(data["IZMAX"])

    rate_coefficient = xr.DataArray(coefficient, coords=dict(
        dim_charge_state = dim_charge_state,
        dim_electron_temp = dim_electron_temp,
        dim_electron_density = dim_electron_density,
    )).pint.quantify(input_units)

    ds["rate_coefficient"] = convert_units(rate_coefficient, dataset_config["desired_units"])

    return ds


def align_rates_on_charge_states(dataset: xr.Dataset) -> xr.Dataset:
    """For rates which are for k+1->k reactions, we shift these by one position
    in the dim_charge_state dimension so that the kth position of a rate always
    refers to the reactant species."""

    dataset = dataset.pad(
        pad_width=dict(dim_charge_state=(0, 1)), mode="constant", constant_values=0.0
    )
    dataset = dataset.assign_coords(
        dim_charge_state=np.arange(dataset.sizes["dim_charge_state"])
    )

    for key in [
        "effective_recombination",
        "charge_exchange_cross_coupling",
        "recombination_and_bremsstrahlung",
        "charge_exchange_emission",
    ]:
        dataset[key] = dataset[key].roll(dim_charge_state=+1)

    return dataset
