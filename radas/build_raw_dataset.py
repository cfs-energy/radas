from .unit_handling import Quantity, dimensionless_magnitude
from .adas_interface.download_species_data import determine_reader_class_and_config
from .shared import data_file_directory
import xarray as xr
import numpy as np


def build_raw_dataset(species_name, config):
    """Builds a dataset combining all of the raw data available for a given species."""
    from .readers import read_adf11_file

    species_config = config["species"][species_name]
    data_file_config = config["data_file_config"]

    raw_dataset = xr.Dataset().assign_attrs(
        atomic_number=species_config["atomic_number"], species_name=species_name
    )
    for attribute, value in config["globals"].items():

        if isinstance(value, dict):
            raw_dataset[attribute] = xr.DataArray(
                Quantity(value["value"], value["units"]),
                coords={f"dim_{attribute}": value["value"]},
            )
        else:
            raw_dataset[attribute] = value

    for i, dataset_type in enumerate(species_config["data_files"].keys()):
        reader_key, dataset_config = determine_reader_class_and_config(
            data_file_config, dataset_type
        )
        if reader_key == "adf11":
            dataset = read_adf11_file(
                data_file_directory, species_name, dataset_type, dataset_config
            )
        else:
            raise NotImplementedError(
                f"No implementation for reading {reader_key} files."
            )

        if i == 0:
            raw_dataset["electron_density"] = dataset["electron_density"]
            raw_dataset["electron_temp"] = dataset["electron_temp"]
            raw_dataset["reference_electron_density"] = (
                dataset.reference_electron_density
            )
            raw_dataset["reference_electron_temp"] = dataset.reference_electron_temp
        else:
            # Ensure that all files have a common grid
            np.testing.assert_allclose(
                dimensionless_magnitude(
                    (raw_dataset["electron_density"] - dataset["electron_density"])
                    / raw_dataset.reference_electron_density
                ),
                0.0,
            )
            np.testing.assert_allclose(
                dimensionless_magnitude(
                    (raw_dataset["electron_temp"] - dataset["electron_temp"])
                    / raw_dataset.reference_electron_temp
                ),
                0.0,
            )

        raw_dataset[dataset_type] = dataset.rate_coefficient

    raw_dataset = raw_dataset.pad(
        pad_width=dict(dim_charge_state=(0, 1)), mode="constant", constant_values=0.0
    )
    raw_dataset = raw_dataset.assign_coords(
        dim_charge_state=np.arange(raw_dataset.sizes["dim_charge_state"])
    )

    for key in [
        "effective_recombination_coeff",
        "charge_exchange_cross_coupling_coeff",
        "recombination_and_bremsstrahlung",
        "charge_exchange_emission",
    ]:
        raw_dataset[key] = raw_dataset[key].roll(dim_charge_state=+1)

    return raw_dataset
