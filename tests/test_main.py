"""Run the main integrated radas program."""

import pytest
import sys
import importlib.util
from radas.cli import download_data_from_adas, read_rate_coefficientss, run_radas_computation


@pytest.fixture()
def datasets(
    monkeypatch,
    temp_module_directory,
    temp_data_file_directory,
    selected_species,
    configuration,
):
    # Download ADAS data to our temporary data file directory
    monkeypatch.setattr(
        "radas.adas_interface.download_adas_datasets.data_file_directory",
        temp_data_file_directory,
    )
    # Build the readers in our temporary module
    monkeypatch.setattr(
        "radas.adas_interface.prepare_adas_readers.module_directory",
        temp_module_directory,
    )

    download_data_from_adas(configuration)

    assert (
        temp_data_file_directory / f"{selected_species}_effective_ionisation.dat"
    ).exists()
    assert (temp_module_directory / "readers" / "adf11" / "xxdata_11").exists()

    def mock_read_adf11_file(*args, **kwargs):
        """Tell 'read_rate_coefficientss' to use the adf11 reader in our temporary directory,
        instead of the one in the current module.
        """
        spec = importlib.util.spec_from_file_location(
            "read_adf11_file", temp_module_directory / "readers" / "read_adf11_file.py"
        )
        adf11_module = importlib.util.module_from_spec(spec)
        sys.modules["read_adf11_file"] = adf11_module
        spec.loader.exec_module(adf11_module)

        return adf11_module.read_adf11_file(*args, **kwargs)

    monkeypatch.setattr(
        "radas.read_rate_coefficients.read_data_from_adf11_file", mock_read_adf11_file
    )

    datasets = read_rate_coefficientss(configuration)

    return datasets


@pytest.mark.filterwarnings("error")
def test_download_data_from_adas_and_read_rate_coefficientss(datasets):
    """This test makes sure that the 'datasets' fixture builds correctly.

    'datasets' is a test as well as a fixture, but there doesn't seem to be a good way to
    make dependent tests.
    """
    pass


@pytest.mark.filterwarnings("error")
def test_run_radas_computation(datasets, selected_species):
    run_radas_computation(datasets[selected_species])
