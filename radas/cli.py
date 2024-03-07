import click
import xarray as xr

from .shared import open_config_file, output_directory
# from .adas_interface import prepare_adas_fortran_interface, download_species_data
from .adas_interface.prepare_adas_readers import prepare_adas_fortran_interface
from .adas_interface.download_adas_datasets import download_species_data
from .build_raw_dataset import build_raw_dataset

from .coronal_equilibrium import calculate_coronal_fractional_abundances
from .radiation import calculate_Lz
from .time_evolution import calculate_time_evolution
from .unit_handling import convert_units, ureg


@click.command()
@click.option("--config", type=click.Path(exists=True), default=None)
@click.option("--species", type=str, default="all")
def run_radas_cli(config: str | None, species: str="all"):
    """
    """
    configuration = open_config_file(config)

    download_data_from_adas(configuration)
    datasets = build_raw_datasets(configuration)

    if species == "all":
        for species, dataset in datasets.items():
            print(f"Running computation for {species}")
            run_radas_computation(dataset)
    else:
        run_radas_computation(dataset[species])

def download_data_from_adas(configuration: dict):
    """Connect to OpenADAS and download the datasets requested in the configuration file.

    config: if not given, uses the config.yaml file in the radas folder.
            You can optionally pass the path to a modified file, for instance
            if you want to change the datasets downloaded.
    """
    prepare_adas_fortran_interface(configuration["data_file_config"])

    for species_name, species_config in configuration["species"].items():
        if "data_files" in species_config:
            download_species_data(
                species_name,
                species_config,
                configuration["data_file_config"],
            )

def build_raw_datasets(configuration: dict):
    datasets = dict()
    output_directory.mkdir(exist_ok=True, parents=True)

    for species_name, species_config in configuration["species"].items():
        if "data_files" in species_config:
            datasets[species_name] = build_raw_dataset(species_name, configuration)
    
    return datasets

def run_radas_computation(dataset: xr.Dataset):
    
    dataset["coronal_charge_state_fraction"] = calculate_coronal_fractional_abundances(
        dataset
    )
    dataset["coronal_mean_charge_state"] = (
        dataset.coronal_charge_state_fraction * dataset.dim_charge_state
    ).sum(dim="dim_charge_state")
    dataset["coronal_Lz"] = calculate_Lz(dataset, dataset.coronal_charge_state_fraction)
    dataset["residence_time"] = convert_units(
        dataset.ne_tau / dataset.electron_density, ureg.s
    )
    dataset["charge_state_fraction_evolution"] = calculate_time_evolution(dataset)
    dataset["charge_state_fraction_at_equilibrium"] = (
        dataset.charge_state_fraction_evolution.isel(dim_time=-1)
    )
    dataset["noncoronal_mean_charge_state"] = (
        dataset.charge_state_fraction_at_equilibrium * dataset.dim_charge_state
    ).sum(dim="dim_charge_state")
    dataset["noncoronal_Lz"] = calculate_Lz(
        dataset, dataset.charge_state_fraction_at_equilibrium
    )

    dataset.pint.dequantify().to_netcdf(
        output_directory / f"{dataset.species_name}.nc"
    )
