"""Sets up some shared resources for testing.

Since radas builds the fortran interface inside the library, we
make a separate copy of the radas library to test it cleanly.
"""

import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def radas_dir(tmpdir_factory):
    "Make a temporary working directory."
    temp_dir = Path(tmpdir_factory.mktemp("radas_dir"))
    print(f"Temporary repository for testing is at: {temp_dir}")
    return temp_dir

@pytest.fixture(scope="session")
def selected_species():
    "Select a single species to run the analysis for, instead of running for all species."
    return "helium"

@pytest.fixture(scope="session")
def data_file_dir(radas_dir):
    (radas_dir / "data_files").mkdir(exist_ok=False)
    return radas_dir / "data_files"

@pytest.fixture(scope="session")
def reader_dir(radas_dir):
    (radas_dir / "readers").mkdir(exist_ok=False)
    return radas_dir / "readers"

@pytest.fixture(scope="session")
def output_dir(radas_dir):
    (radas_dir / "output").mkdir(exist_ok=False)
    return radas_dir / "output"

@pytest.fixture()
def verbose():
    return 10