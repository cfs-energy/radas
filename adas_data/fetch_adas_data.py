#!.venv/bin/python
from pathlib import Path
import urllib.request
import datetime
import yaml
import shutil
import subprocess
import sys

from radas.named_options.adf11_dataset import ADF11Dataset
from radas.named_options.atomic_species import AtomicSpecies
from radas.directories import module_directory, environment_directory

here = Path(__file__).parent
url_base = "https://open.adas.ac.uk"

def download_adas_file(dataset: ADF11Dataset, species: AtomicSpecies, year: int) -> bool:
    """Download an ADAS ADF11 file corresponding to a specific dataset and specific species.

    Returns a bool indicating if the download was successful.
    """

    year_key = f"{year}"[-2:]
    dataset_key = f"{dataset.value.lower()}"
    species_key = f"{species.value[0].lower()}"

    output_filename = here / "dat_files" / f"{species.name}_{dataset.name}.dat"

    query_path = f"{url_base}/download/adf11/{dataset_key}{year_key}/{dataset_key}{year_key}_{species_key}.dat"
    urllib.request.urlretrieve(query_path, output_filename)

    if "OPEN-ADAS Error" in output_filename.read_text():
        output_filename.unlink()
        return False
    else:
        return True

def download_adas_file_latest(dataset: ADF11Dataset, species: AtomicSpecies, earliest_year: int=1950):
    """Downloads an ADAS file corresponding to the latest available year."""
    now = datetime.date.today().year

    for year in range(now, earliest_year-1, -1):
        success = download_adas_file(dataset, species, year)
        if success: break

def download_adas_reader(reader: str) -> bool:
    assert reader.startswith("ADF"), f"Reader should be of the format ADFXX where XX is an integer"
    format_int = int(reader.lstrip("ADF"))
    archive_filename = here / "source_files" / f"{reader}.tar.gz"

    query_path = f"{url_base}/code/xxdata_{format_int}.tar.gz"

    try:
        urllib.request.urlretrieve(query_path, archive_filename)
    except urllib.error.HTTPError:
        return False
    
    # Extract reader from the tar file
    shutil.unpack_archive(archive_filename, here / "source_files")

    dest_path = here / "source_files" / reader
    if dest_path.exists():
        shutil.rmtree(dest_path)
    
    (here / "source_files" / f"xxdata_{format_int}").replace(dest_path)
    return True

def download_and_compile_adas_reader(reader: str) -> bool:
    success = download_adas_reader(reader)
    format_int = int(reader.lstrip("ADF"))

    if not success:
        print(f"Failed to download the {reader} reader.")
        return False

    foldername = here / "source_files" / f"{reader}"

    # Compile the reader using f2py

    fortran_files = [str(file) for file in foldername.iterdir() if (file.suffix == ".for") and not (file.stem == "test")]

    compile_reader = lambda quiet: subprocess.run(
        [
            str(environment_directory / "bin" / "python"),
            "-m",
            "numpy.f2py",
            "-c",
            str(here / "headers" / f"xxdata_{format_int}.pyf")
        ] + fortran_files + [
            "-m",
            # Will build in the current working directory
            str(f"{reader}_reader"),
        ],
        capture_output = quiet,
        check = not quiet
    )

    result = compile_reader(True)
    
    if not result.returncode == 0:
        # If the first compile attempt fails, do it again
        # and this time print output
        compile_reader(False)
    
    for file in Path().iterdir():
        if file.name.startswith(f"{reader}_reader"):
            file.rename(module_directory / "adas_file_readers" / file.name)
    
    return True

def compile_file_handling_helper():
    compile_file_handling = lambda quiet: subprocess.run(
        [
            str(environment_directory / "bin" / "python"),
            "-m",
            "numpy.f2py",
            "-c",
            str(here / "fortran_file_handling.f90"),
            "-m",
            # Will build in the current working directory
            str(f"fortran_file_handling"),
        ],
        capture_output = quiet,
        check = not quiet
    )

    result = compile_file_handling(True)
    
    if not result.returncode == 0:
        # If the first compile attempt fails, do it again
        # and this time print output
        compile_file_handling(False)
    
    for file in Path().iterdir():
        if file.name.startswith(f"fortran_file_handling"):
            file.replace(module_directory / "adas_file_readers" / file.name)

if __name__=="__main__":
    input_file = here / "data_to_fetch.yaml"
    with open(input_file) as file:
        data_to_fetch = yaml.load(file, Loader=yaml.FullLoader)
    
    (here / "dat_files").mkdir(exist_ok=True, parents=True)
    (here / "source_files").mkdir(exist_ok=True, parents=True)

    print(f"Building file I/O helper")
    compile_file_handling_helper()

    for reader in data_to_fetch["readers"]:
        print(f"Building {reader} reader")
        success = download_and_compile_adas_reader(reader)
    
    for species_key, species_datasets in data_to_fetch["dat_files"].items():
        print(f"Downloading ADAS files for {species_key}")
        for dataset_key, year in species_datasets.items():
            success = download_adas_file(ADF11Dataset[dataset_key], AtomicSpecies[species_key], year)
            if not success: print(f"Failed to download the {year} {dataset_key} for {species_key}")
        
        if "pytest" in sys.modules: break # Only download a single species if testing
    
    print("Done")