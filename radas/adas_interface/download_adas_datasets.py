import urllib.request
from ..shared import data_file_directory
from .determine_adas_dataset_type import determine_reader_class_and_config


def download_species_data(
    species_name: str,
    species_config: dict,
    data_file_config: dict,
    url_base: str = "https://open.adas.ac.uk",
):
    """Downloads all of the data files for a specific species."""
    data_file_directory.mkdir(exist_ok=True, parents=True)

    for dataset_type, year in species_config["data_files"].items():

        reader_class, dataset_config = determine_reader_class_and_config(
            data_file_config, dataset_type
        )

        year_key = f"{year}"[-2:]
        dataset_prefix = dataset_config["prefix"].lower()
        species_key = species_config["atomic_symbol"].lower()

        output_filename = data_file_directory / f"{species_name}_{dataset_type}.dat"
        query_path = f"{url_base}/download/{reader_class}/{dataset_prefix}{year_key}/{dataset_prefix}{year_key}_{species_key}.dat"

        if not output_filename.exists():
            urllib.request.urlretrieve(query_path, output_filename)

        if "OPEN-ADAS Error" in output_filename.read_text():
            output_filename.unlink()
            print(f"Failed to download the {year} {dataset_prefix} for {species_name}")
