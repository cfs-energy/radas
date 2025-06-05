import click
import xarray as xr
import multiprocessing as mp
from pathlib import Path
from functools import partial
from typing import Optional

from .shared import open_yaml_file, default_config_file
from .adas_interface.download_adas_datasets import download_species_data
from .read_rate_coeffs import read_rate_coeff

from .coronal_equilibrium import calculate_coronal_fractional_abundances
from .radiated_power import calculate_Lz
from .time_evolution import calculate_time_evolution
from .unit_handling import convert_units, ureg
from .mavrin_reference import compare_radas_to_mavrin
from .interpolate_rates import interpolate_dataset


@click.command()
@click.option(
    "-d",
    "--directory",
    type=click.Path(),
    default=Path("./radas_dir").absolute(),
    help="Directory for storing work files and outputs. DEFAULT: ./radas_dir",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Path to a yaml file for configuring radas. DEFAULT: RADAS_DIR/radas/config.yaml",
)
@click.option(
    "-s",
    "--species",
    default=("all",),
    multiple=True,
    help="Species to perform analysis for ('species_name'|'all'|'none'). DEFAULT: all",
)
@click.option(
    "-v", "--verbose", count=True, help="Write additional output to the command line."
)
@click.option(
    "--debug",
    is_flag=True,
    help="Flag to enable debug mode (disables multiprocessing).",
)
def run_radas_cli(
    directory: Path,
    config: Optional[str],
    species: list[str],
    verbose: int,
    debug: bool,
):
    """Runs the radas program.

    If config is given, it must point to a config.yaml file. Otherwise, a
    default config.yaml (stored in the module directory) is used.

    If species is given, it must be a valid species name (i.e. 'hydrogen').
    Otherwise, all valid species in the config.yaml file are evaluated.
    """
    kwargs = dict(
        directory=directory,
        config=config,
        species=species,
        verbose=verbose,
        debug=debug,
    )
    try:
        from ipdb import launch_ipdb_on_exception

        with launch_ipdb_on_exception():
            run_radas(**kwargs)
    except ModuleNotFoundError:
        run_radas(**kwargs)


def run_radas(
    directory: Path,
    config: Optional[str],
    species: list[str],
    verbose: int,
    debug: bool,
):

    radas_dir = Path(directory)
    if verbose:
        print(f"Running radas in {radas_dir.absolute()}")
    data_file_dir = radas_dir / "data_files"
    output_dir = radas_dir / "output"

    for path in [radas_dir, data_file_dir, output_dir]:
        path.mkdir(exist_ok=True)

    if species == ("none",):
        if verbose:
            print("Skipping computation.")
    else:
        config_file = default_config_file if config is None else Path(config).absolute()
        if verbose:
            print(f"Opening config file at {config_file}")
        configuration = open_yaml_file(config_file)

        if verbose:
            print(f"Downloading data from OpenADAS to {data_file_dir.absolute()}")
        for species_name, species_config in configuration["species"].items():
            if "data_files" in species_config and (
                (species_name in species) or (species == ("all",))
            ):
                download_species_data(
                    data_file_dir,
                    species_name,
                    species_config,
                    configuration["data_file_config"],
                    verbose=verbose,
                )

        if verbose:
            print("Reading rate coefficients")
        datasets = dict()
        for species_name, species_config in configuration["species"].items():
            if "data_files" in species_config and (
                (species_name in species) or (species == ("all",))
            ):
                datasets[species_name] = read_rate_coeff(
                    data_file_dir, species_name, configuration
                )
        
        if ("electron_density_resolution" in configuration["globals"]) or ("electron_temp_resolution") in configuration["globals"]:
            if verbose:
                print("Interpolating rate coefficients")
            
            new_datasets = dict()
            for species_name, dataset in datasets.items():
                electron_density_resolution = configuration["globals"].get("electron_density_resolution", dataset.sizes["dim_electron_density"])
                electron_temp_resolution = configuration["globals"].get("electron_temp_resolution", dataset.sizes["dim_electron_temp"])
                new_datasets[species_name] = interpolate_dataset(dataset,
                                                                 electron_density_resolution = electron_density_resolution,
                                                                 electron_temp_resolution = electron_temp_resolution)
            datasets = new_datasets

        output_dir.mkdir(exist_ok=True, parents=True)
        if not debug:
            with mp.Pool() as pool:
                if species != ("all",):
                    datasets = {
                        species_name: datasets[species_name] for species_name in species
                    }

                pool.map(
                    partial(
                        run_radas_computation, output_dir=output_dir, verbose=verbose
                    ),
                    [(ds) for ds in datasets.values()],
                )
        else:
            for ds in datasets.values():
                run_radas_computation(ds, output_dir=output_dir, verbose=verbose)

    if verbose:
        print(f"Generating plots and saving output to {output_dir}")
    compare_radas_to_mavrin(output_dir)

    if verbose:
        print("Done")


def run_radas_computation(dataset: xr.Dataset, output_dir: Path, verbose: int):
    """Calculate several dependent quantities based on the atomic rates, and store
    the result as a NetCDF file.
    """
    if verbose:
        print(f"Running computation for {dataset.species_name}")

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

    output_dir.mkdir(exist_ok=True)
    dataset.pint.dequantify().to_netcdf(output_dir / f"{dataset.species_name}.nc")


@click.command()
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=Path("./config.yaml").absolute(),
    help="Output path for a config file. DEFAULT: ./config.yaml",
)
def write_config_template(output: Path):
    output = Path(output).absolute()
    print(f"Copying {default_config_file} to {output}")
    output.write_text(default_config_file.read_text())
    print("Done")
