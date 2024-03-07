from pathlib import Path
import urllib.request
import shutil
import subprocess
from .unit_handling import dimensionless_magnitude, Quantity
import numpy as np
import xarray as xr

url_base = "https://open.adas.ac.uk"

def compile_with_f2py(files_to_compile: list[str], module_name: str, output_folder: Path):
    """Compiles a list of fortran files into a named module."""
    compile_reader = lambda quiet: subprocess.run(
        [
            "python3",
            "-m",
            "numpy.f2py",
            "-c",
        ]
        + files_to_compile
        + [
            "-m",
            # Will build in the current working directory
            module_name,
        ],
        capture_output=quiet,
        check=not quiet,
    )

    result = compile_reader(True)

    if not result.returncode == 0:
        # If the first compile attempt fails, do it again
        # and this time print output
        compile_reader(False)

    for file in Path().iterdir():
        if file.name.startswith(module_name):
            file.rename(output_folder / file.name)


def build_adas_file_reader(reader_name: str):
    """Builds an ADAS file reader by wrapping fortran code using f2py."""
    assert reader_name.startswith("adf"), f"Reader should be of the format adfXX where XX is an integer"
    reader_int = int(reader_name.lstrip("adf"))
    output_folder = module_directory / "readers" / reader_name

    archive_file = f"xxdata_{reader_int}.tar.gz"
    query_path = f"{url_base}/code/{archive_file}"
    urllib.request.urlretrieve(query_path, output_folder / archive_file)
    shutil.unpack_archive(output_folder / archive_file, output_folder)

    fortran_files = [
        str(file) for file in (output_folder / f"xxdata_{reader_int}").iterdir() if (file.suffix == ".for") and not (file.stem == "test")
    ]

    fortran_files = fortran_files + [str(output_folder / f"xxdata_{reader_int}.pyf")]

    compile_with_f2py(files_to_compile=fortran_files, module_name=str(f"{reader_name}_reader"), output_folder=output_folder)

def determine_reader_class_and_config(data_file_config, dataset_type):
    """Examines the data_file_config to determine which reader class to use to reader a specific dataset_type."""
    for reader_key, reader_config in data_file_config.items():
        for dataset_key, dataset_config in reader_config.items():
            if dataset_key == dataset_type:
                return reader_key, dataset_config
    raise NotImplementedError(f"Cannot identify reader for {dataset_type}.")


def download_species_data(species_name, species_config, data_file_config, quiet):
    """Downloads all of the data files for a specific species."""
    data_file_directory.mkdir(exist_ok=True, parents=True)

    for dataset_type, year in species_config["data_files"].items():

        reader_class, dataset_config = determine_reader_class_and_config(data_file_config, dataset_type)

        year_key = f"{year}"[-2:]
        dataset_prefix = dataset_config["prefix"].lower()
        species_key = species_config["atomic_symbol"].lower()

        output_filename = data_file_directory / f"{species_name}_{dataset_type}.dat"
        query_path = f"{url_base}/download/{reader_class}/{dataset_prefix}{year_key}/{dataset_prefix}{year_key}_{species_key}.dat"

        if output_filename.exists():
            if not quiet:
                print(f"{output_filename} already exists. Skipping download of {query_path}")
        else:
            if not quiet:
                print(f"Downloading {query_path} and saving as {output_filename}")
            urllib.request.urlretrieve(query_path, output_filename)

        if "OPEN-ADAS Error" in output_filename.read_text():
            output_filename.unlink()
            print(f"Failed to download the {year} {dataset_prefix} for {species_name}")


def build_raw_dataset(species_name, config):
    """Builds a dataset combining all of the raw data available for a given species."""
    from .readers import read_adf11_file

    species_config = config["species"][species_name]
    data_file_config = config["data_file_config"]

    raw_dataset = xr.Dataset().assign_attrs(
        atomic_number=species_config["atomic_number"],
        species_name=species_name
    )
    for attribute, value in config["globals"].items():

        if isinstance(value, dict):
            raw_dataset[attribute] = xr.DataArray(Quantity(value["value"], value["units"]), coords={f"dim_{attribute}": value["value"]})
        else:
            raw_dataset[attribute] = value

    for i, dataset_type in enumerate(species_config["data_files"].keys()):
        reader_key, dataset_config = determine_reader_class_and_config(data_file_config, dataset_type)
        if reader_key == "adf11":
            dataset = read_adf11_file(data_file_directory, species_name, dataset_type, dataset_config)
        else:
            raise NotImplementedError(f"No implementation for reading {reader_key} files.")

        if i == 0:
            raw_dataset["electron_density"] = dataset["electron_density"]
            raw_dataset["electron_temp"] = dataset["electron_temp"]
            raw_dataset["reference_electron_density"] = dataset.reference_electron_density
            raw_dataset["reference_electron_temp"] = dataset.reference_electron_temp
        else:
            # Ensure that all files have a common grid
            np.testing.assert_allclose(
                dimensionless_magnitude((raw_dataset["electron_density"] - dataset["electron_density"]) / raw_dataset.reference_electron_density), 0.0
            )
            np.testing.assert_allclose(
                dimensionless_magnitude((raw_dataset["electron_temp"] - dataset["electron_temp"]) / raw_dataset.reference_electron_temp), 0.0
            )

        raw_dataset[dataset_type] = dataset.rate_coefficient

    raw_dataset = raw_dataset.pad(pad_width=dict(dim_charge_state=(0, 1)), mode="constant", constant_values=0.0)
    raw_dataset = raw_dataset.assign_coords(dim_charge_state=np.arange(raw_dataset.sizes["dim_charge_state"]))

    for key in [
        "effective_recombination_coeff",
        "charge_exchange_cross_coupling_coeff",
        "recombination_and_bremsstrahlung",
        "charge_exchange_emission",
    ]:
        raw_dataset[key] = raw_dataset[key].roll(dim_charge_state=+1)
    
    return raw_dataset

def prepare_adas_fortran_interface(data_file_config: dict):
    compile_with_f2py(
        files_to_compile=[module_directory / "readers" / "fortran_file_handling.f90"],
        module_name="fortran_file_handling",
        output_folder=module_directory / "readers",
    )

    for reader_name in data_file_config.keys():
        reader_found = False
        for file in (module_directory / "readers" / reader_name).iterdir():
            if file.name.startswith(f"{reader_name}_reader"):
                reader_found = True

        if not reader_found:
            build_adas_file_reader(reader_name)
