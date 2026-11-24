"""Download OpenADAS data files."""

from pathlib import Path
import urllib.request
from .determine_adas_dataset_type import determine_reader_class_and_config


def is_complete(filename: Path) -> bool:
    """Whether an ADAS file ends with the footer OPEN-ADAS appends, which a truncated download lacks."""
    text = filename.read_text(errors="replace").rstrip()
    return "downloaded from OPEN-ADAS" in text and text.rpartition("\n")[2].startswith("C---")


def download_species_data(  # noqa: PLR0917 - verbose and url_base are part of the public positional signature
    data_file_dir: Path,
    species_name: str,
    species_config: dict,
    data_file_config: dict,
    verbose: int,
    url_base: str = "https://open.adas.ac.uk",
):
    """Downloads all of the data files for a specific species."""
    data_file_dir.mkdir(exist_ok=True, parents=True)

    for dataset_type, file_to_download in species_config["data_files"].items():
        reader_class, dataset_config = determine_reader_class_and_config(data_file_config, dataset_type)

        if isinstance(file_to_download, int):
            species_key = species_config["atomic_symbol"].lower()
            year = file_to_download
        elif isinstance(file_to_download, list) and len(file_to_download) == 2:
            species_key = file_to_download[0].lower()
            year = file_to_download[1]
        else:
            raise NotImplementedError(f"Could not process entry: {file_to_download} for {species_name} {dataset_type}")

        year_key = f"{year}"[-2:]
        dataset_prefix = dataset_config["prefix"].lower()

        output_filename = data_file_dir / f"{species_name}_{dataset_type}_{year_key}.dat"
        query_path = f"{url_base}/download/{reader_class}/{dataset_prefix}{year_key}/{dataset_prefix}{year_key}_{species_key}.dat"

        if output_filename.exists() and not is_complete(output_filename):  # left by an interrupted download (or an error page)
            output_filename.unlink()

        if not output_filename.exists():
            if verbose >= 2:
                print(f"Downloading {query_path} to {output_filename}")
            urllib.request.urlretrieve(query_path, output_filename)
        elif verbose >= 2:
            print(f"Reusing {query_path} ({output_filename} already exists)")

        if "OPEN-ADAS Error" in output_filename.read_text():
            output_filename.unlink()
            if verbose:
                print(f"Failed to download the {year} {dataset_prefix} for {species_name}")
        elif not is_complete(output_filename):
            output_filename.unlink()
            raise RuntimeError(f"Downloaded {query_path} is incomplete; removed it. Try again.")
