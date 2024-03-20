import click
import xarray as xr
import multiprocessing as mp
from ipdb import launch_ipdb_on_exception
from pathlib import Path

from .shared import open_config_file, output_directory, data_file_directory, module_directory
from .adas_interface import prepare_adas_fortran_interface, download_species_data
from .read_rate_coeffs import read_rate_coeff

from .coronal_equilibrium import calculate_coronal_fractional_abundances
from .radiated_power import calculate_Lz
from .time_evolution import calculate_time_evolution
from .unit_handling import convert_units, ureg
from .mavrin_reference import compare_radas_to_mavrin


@click.command()
@click.option("--config", type=click.Path(exists=True), default=None)
@click.option("--species", type=str, default="all")
@click.option("--plot/--no-plot", type=bool, default=False)
def run_radas_cli(config: str | None, species: str = "all", plot: bool = False):
    """Runs the radas program.

    If config is given, it must point to a config.yaml file. Otherwise, a
    default config.yaml (stored in the module directory) is used.

    If species is given, it must be a valid species name (i.e. 'hydrogen').
    Otherwise, all valid species in the config.yaml file are evaluated.
    """
    with launch_ipdb_on_exception():

        if species == "none":
            print("Skipping computation.")
        else:
            print(f"Opening config file at {module_directory / 'config.yaml' if config is None else Path(config)}")
            configuration = open_config_file(config)

            print(f"Downloading data from OpenADAS to {data_file_directory.absolute()}")
            download_data_from_adas(configuration)

            print(f"Reading rate coefficients")
            datasets = read_rate_coefficients(configuration)

            if species == "all":
                print(f"Processing all species and saving output to {output_directory}")
                with mp.Pool() as pool:
                    pool.map(run_radas_computation, [ds for ds in datasets.values()])
            else:
                print(f"Processing {species} and saving output to {output_directory}")
                run_radas_computation(datasets[species])

        if plot:
            print(f"Generating plots and saving output to {output_directory}")
            compare_radas_to_mavrin()
        
        print("Done")


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


def read_rate_coefficients(configuration: dict):
    """Builds datasets containing the rate coefficients for each species."""
    datasets = dict()
    output_directory.mkdir(exist_ok=True, parents=True)

    for species_name, species_config in configuration["species"].items():
        if "data_files" in species_config:
            datasets[species_name] = read_rate_coeff(species_name, configuration)

    return datasets


def run_radas_computation(dataset: xr.Dataset):
    """Calculate several dependent quantities based on the atomic rates, and store
    the result as a NetCDF file.
    """

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
    dataset["charge_state_evolution"] = calculate_time_evolution(dataset)
    dataset["equilibrium_charge_state_fraction"] = dataset.charge_state_evolution.isel(
        dim_time=-1
    )
    dataset["equilibrium_mean_charge_state"] = (
        dataset.equilibrium_charge_state_fraction * dataset.dim_charge_state
    ).sum(dim="dim_charge_state")
    dataset["equilibrium_Lz"] = calculate_Lz(
        dataset, dataset.equilibrium_charge_state_fraction
    )

    output_directory.mkdir(exist_ok=True)
    dataset.pint.dequantify().to_netcdf(output_directory / f"{dataset.species_name}.nc")
