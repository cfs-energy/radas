import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def test_directory():
    return Path(__file__).parent

@pytest.fixture(scope="session")
def repository_directory(test_directory):
    return test_directory.parent

@pytest.fixture(scope="session")
def adas_data_directory(repository_directory):
    return repository_directory / "adas_data"

