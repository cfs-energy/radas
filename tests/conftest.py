import pytest
import shutil
import sys
from pathlib import Path
from radas import module_directory, shared

@pytest.fixture(scope="session")
def temp_repository_directory(tmpdir_factory):
    temp_repo_dir = Path(tmpdir_factory.mktemp("test_radas"))
    print(f"Temporary repository for testing is at: {temp_repo_dir}")
    return temp_repo_dir

@pytest.fixture(scope="session")
def temp_module_directory(temp_repository_directory):
    temp_module_directory = temp_repository_directory / "radas"
    temp_module_directory.mkdir(exist_ok = True)

    extensions_to_copy = [".py", ".pyf"]
    files_to_copy = ["config.yaml", "fortran_file_handling.f90"]

    for file in module_directory.rglob("*"):

        if file.suffix in extensions_to_copy or file.name in files_to_copy:
            dst = temp_module_directory / (file.relative_to(module_directory))
            dst.parent.mkdir(exist_ok=True, parents=True)
            
            # if str(dst.parent) not in on_path:
            #     sys.path.insert(0, str(dst.parent))
            #     on_path += [str(dst.parent)]

            shutil.copy(file, dst)
    
    return temp_module_directory

@pytest.fixture(scope="session")
def temp_data_file_directory(temp_module_directory):
    return temp_module_directory / ".data_files"

@pytest.fixture(scope="session")
def temp_output_directory(temp_repository_directory):
    return temp_repository_directory / "output"

@pytest.fixture(scope="session")
def selected_species():
    return "helium"

@pytest.fixture(scope="session")
def configuration(temp_module_directory, selected_species):
    config = shared.open_config_file(temp_module_directory / "config.yaml")

    # Drop everything aside from one species
    config["species"] = {selected_species: config["species"][selected_species]}

    return config