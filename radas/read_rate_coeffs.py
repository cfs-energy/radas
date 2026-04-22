from .unit_handling import Quantity, ureg, convert_units, dimensionless_magnitude
from .adas_interface.determine_adas_dataset_type import (
    determine_reader_class_and_config,
)
from .shared import get_git_revision_short_hash
from importlib.metadata import version, PackageNotFoundError
import datetime
import xarray as xr
import numpy as np
from .interpolate_rates import interpolate_array

reference_electron_density = Quantity(1.0, ureg.m**-3)
reference_electron_temp = Quantity(1.0, ureg.eV)

def read_rate_coeff(data_file_dir, species_name, config):
    """Builds a rate_dataset combining all of the raw data available for a given species."""

    try:
        radas_version=version("radas")
    except PackageNotFoundError:
        radas_version="UNDEFINED"
    
    rate_coefficients = build_sorted_dictionary_of_rate_coefficients(config, species_name, data_file_dir)
    rate_coefficients = interpolate_rates_onto_matching_grids(config, rate_coefficients)
    
    try:
        dataset = xr.merge([v.rename(k) for k, v in rate_coefficients.items()], join="exact")
    except xr.AlignmentError as e:
        raise xr.AlignmentError(f"Alignment failed for {species_name} with join='exact'. Error was {e}")

    if dataset.sizes["dim_electron_density"] <= 2 or dataset.sizes["dim_electron_temp"] <= 2:
        raise xr.AlignmentError(f"Alignment resulted in dataset sizes {dataset.sizes} for {species_name}.")

    dataset["electron_density"] = dataset["dim_electron_density"] * reference_electron_density
    dataset["electron_temp"] = dataset["dim_electron_temp"] * reference_electron_temp

    dataset = align_rates_on_charge_states(dataset)

    dataset = dataset.assign_attrs(
        atomic_number=config["species"][species_name]["atomic_number"],
        species_name=species_name,
        git_hash=get_git_revision_short_hash(),
        radas_version=radas_version,
        created=datetime.date.today().strftime("%Y-%b-%d"),
    )

    dataset = write_global_attributes(dataset, config["globals"])

    return dataset

def build_sorted_dictionary_of_rate_coefficients(config, species_name, data_file_dir):
    """Parses configuration data to build a collection of rate coefficients sorted by year (most-recent first)."""
    rate_coefficients = dict()
    years = dict()

    for dataset_type, file_to_read in config["species"][species_name]["data_files"].items():
        reader_key, dataset_config = determine_reader_class_and_config(
            config["data_file_config"], dataset_type
        )

        if isinstance(file_to_read, int):
            years[dataset_type] = file_to_read
        elif isinstance(file_to_read, list) and len(file_to_read) == 2:
            years[dataset_type] = file_to_read[1]
        else:
            raise NotImplementedError(f"Could not process entry: {file_to_read} for {species_name} {dataset_type}")

        if reader_key == "adf11":
            rate_dataset = build_adf11_rate_dataset(
                data_file_dir,
                species_name,
                years[dataset_type],
                dataset_type,
                dataset_config,
            )

        else:
            raise NotImplementedError(
                f"No implementation for reading {reader_key} files."
            )

        rate_coefficients[dataset_type] = rate_dataset.rate_coefficient
    
    # Sort the datasets so that the most recent data comes first
    sorted_by_year = dict(sorted(years.items(), key=lambda item: item[1], reverse=True))

    return {k: rate_coefficients[k] for k in sorted_by_year.keys()}

def interpolate_rates_onto_matching_grids(config, rate_coefficients):

    # Since the rate coefficients are sorted by year, taking the first value
    # gives the most recent data.
    most_recent_rate_coeff = list(rate_coefficients.values())[0]

    new_electron_density = np.logspace(
        np.log10(most_recent_rate_coeff["dim_electron_density"].min().item()),
        np.log10(most_recent_rate_coeff["dim_electron_density"].max().item()),
        num = config["globals"]["electron_density_resolution"]
    )

    new_electron_temp = np.logspace(
        np.log10(most_recent_rate_coeff["dim_electron_temp"].min().item()),
        np.log10(most_recent_rate_coeff["dim_electron_temp"].max().item()),
        num = config["globals"]["electron_temp_resolution"]
    )

    interpolated_rate_coefficients = dict()

    for key, value in rate_coefficients.items():
        interpolated_rate_coefficients[key] = value.groupby("dim_charge_state").map(interpolate_array, args=(new_electron_density, new_electron_temp))
    
    return interpolated_rate_coefficients


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
    data_file_dir, species_name, year, dataset_type, dataset_config
):
    from .adas_interface.read_adf11_file import read_adf11_file

    data = read_adf11_file(data_file_dir, species_name, year, dataset_type)

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
