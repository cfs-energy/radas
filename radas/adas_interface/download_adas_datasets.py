from pathlib import Path
import urllib.request
from .determine_adas_dataset_type import determine_reader_class_and_config


def download_species_data(
    data_file_dir: Path,
    species_name: str,
    species_config: dict,
    data_file_config: dict,
    verbose: int,
    url_base: str = "https://open.adas.ac.uk",
):
    """Downloads all of the data files for a specific species."""
    data_file_dir.mkdir(exist_ok=True, parents=True)

    for dataset_type, year in species_config["data_files"].items():

        reader_class, dataset_config = determine_reader_class_and_config(
            data_file_config, dataset_type
        )

        year_key = f"{year}"[-2:]
        dataset_prefix = dataset_config["prefix"].lower()
        species_key = species_config["atomic_symbol"].lower()

        output_filename = data_file_dir / f"{species_name}_{dataset_type}.dat"
        query_path = f"{url_base}/download/{reader_class}/{dataset_prefix}{year_key}/{dataset_prefix}{year_key}_{species_key}.dat"

        if not output_filename.exists():
            if verbose >= 2:
                print(f"Downloading {query_path} to {output_filename}")
            urllib.request.urlretrieve(query_path, output_filename)
        else:
            if verbose >= 2:
                print(f"Reusing {query_path} ({output_filename} already exists)")

        if "OPEN-ADAS Error" in output_filename.read_text():
            output_filename.unlink()
            if verbose:
                print(
                    f"Failed to download the {year} {dataset_prefix} for {species_name}"
                )
