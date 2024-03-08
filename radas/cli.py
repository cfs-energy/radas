import click
import xarray as xr
import multiprocessing as mp

from .shared import open_config_file, output_directory
from .adas_interface import prepare_adas_fortran_interface, download_species_data
from .read_rate_coefficients import read_rate_coefficients

from .coronal_equilibrium import calculate_coronal_fractional_abundances
from .radiation import calculate_Lz
from .time_evolution import calculate_time_evolution
from .unit_handling import convert_units, ureg


@click.command()
@click.option("--config", type=click.Path(exists=True), default=None)
@click.option("--species", type=str, default="all")
def run_radas_cli(config: str | None, species: str = "all"):
    """Runs the radas program.
    
    If config is given, it must point to a config.yaml file. Otherwise, a
    default config.yaml (stored in the module directory) is used.

    If species is given, it must be a valid species name (i.e. 'hydrogen').
    Otherwise, all valid species in the config.yaml file are evaluated.
    """
    configuration = open_config_file(config)

    download_data_from_adas(configuration)
    datasets = read_rate_coefficientss(configuration)

    if species == "all":
        with mp.Pool() as pool:
            pool.map(run_radas_computation, [ds for ds in datasets.values()])
    else:
        run_radas_computation(datasets[species])


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


def read_rate_coefficientss(configuration: dict):
    """Builds datasets containing the rate coefficients for each species."""
    datasets = dict()
    output_directory.mkdir(exist_ok=True, parents=True)

    for species_name, species_config in configuration["species"].items():
        if "data_files" in species_config:
            datasets[species_name] = read_rate_coefficients(species_name, configuration)

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
    dataset["equilibrium_charge_state_fraction"] = (
        dataset.charge_state_evolution.isel(dim_time=-1)
    )
    dataset["equilibrium_mean_charge_state"] = (
        dataset.equilibrium_charge_state_fraction * dataset.dim_charge_state
    ).sum(dim="dim_charge_state")
    dataset["equilibrium_Lz"] = calculate_Lz(
        dataset, dataset.equilibrium_charge_state_fraction
    )

    dataset.pint.dequantify().to_netcdf(output_directory / f"{dataset.species_name}.nc")
