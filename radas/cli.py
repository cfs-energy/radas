import numpy as np
import xarray as xr
import yaml
from pathlib import Path
import click
import multiprocessing as mp

from .adas_interface import (
    module_directory,
    prepare_adas_fortran_interface,
    download_species_data,
    build_raw_dataset
)
from .coronal_equilibrium import calculate_coronal_fractional_abundances
from .radiation import calculate_Lz
from .time_evolution import calculate_time_evolution
from .unit_handling import convert_units, ureg

@click.command()
@click.option("--config", type=click.Path(exists=True), default=None)
def download_data_from_adas(config: str | None):

    configuration = open_config_file(config)
    prepare_adas_fortran_interface(configuration["data_file_config"])

    for species_name, species_config in configuration["species"].items():
        if "data_files" in species_config:
            download_species_data(species_name, species_config, configuration["data_file_config"], quiet=True)

@click.command()
@click.option("--config", type=click.Path(exists=True), default=None)
@click.option("--species", type=str, default="all")
def run_computation(config: str | None, species: str="all"):
    configuration = open_config_file(config)

    (module_directory.parent / "output").mkdir(exist_ok=True, parents=True)

    datasets = dict()
    for species_name, species_config in configuration["species"].items():
        if "data_files" in species_config:
            datasets[species_name] = build_raw_dataset(species_name, configuration)

def open_config_file(config: str | None) -> dict:
    config_file = module_directory / "config.yaml" if config is None else Path(config)
    
    with open(config_file, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)

def run_computation_on_dataset(dataset: xr.Dataset):

    dataset["coronal_charge_state_fraction"] = calculate_coronal_fractional_abundances(dataset)
    dataset["coronal_mean_charge_state"] = (dataset.coronal_charge_state_fraction * dataset.dim_charge_state).sum(dim="dim_charge_state")
    dataset["coronal_Lz"] = calculate_Lz(dataset, dataset.coronal_charge_state_fraction)
    dataset["residence_time"] = convert_units(dataset.ne_tau / dataset.electron_density, ureg.s)
    dataset["charge_state_fraction_evolution"] = calculate_time_evolution(dataset)
    dataset["charge_state_fraction_at_equilibrium"] = dataset.charge_state_fraction_evolution.isel(dim_time=-1)
    dataset["noncoronal_mean_charge_state"] = (dataset.charge_state_fraction_at_equilibrium * dataset.dim_charge_state).sum(dim="dim_charge_state")
    dataset["noncoronal_Lz"] = calculate_Lz(dataset, dataset.charge_state_fraction_at_equilibrium)

    dataset.pint.dequantify().to_netcdf(
        module_directory.parent / "output" / 
        f"{dataset.species_name}.nc"
    )
