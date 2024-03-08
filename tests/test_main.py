"""Run the main integrated radas program."""

import pytest
import sys
import xarray as xr
import numpy as np
import importlib.util
from radas.cli import download_data_from_adas, read_rate_coefficients, run_radas_computation
from radas.mavrin_reference import compare_radas_to_mavrin, read_mavrin_data, compute_Mavrin_polynomial_fit
from radas.unit_handling import ureg

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
        """Tell 'read_rate_coeff' to use the adf11 reader in our temporary directory,
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
        "radas.read_rate_coeffs.read_data_from_adf11_file", mock_read_adf11_file
    )

    datasets = read_rate_coefficients(configuration)

    return datasets

@pytest.mark.order(1)
@pytest.mark.filterwarnings("error")
def test_download_data_from_adas_and_read_rate_coefficients(datasets):
    """This test makes sure that the 'datasets' fixture builds correctly.

    'datasets' is a test as well as a fixture, but there doesn't seem to be a good way to
    make dependent tests.
    """
    pass

@pytest.mark.order(2)
@pytest.mark.filterwarnings("error")
def test_run_radas_computation(monkeypatch, temp_output_directory, datasets, selected_species):
    monkeypatch.setattr(
        "radas.cli.output_directory",
        temp_output_directory,
    )
    run_radas_computation(datasets[selected_species])

@pytest.mark.order(3)
@pytest.mark.filterwarnings("error")
def test_compare_radas_to_mavrin(monkeypatch, temp_output_directory):
    monkeypatch.setattr(
        "radas.cli.output_directory",
        temp_output_directory,
    )
    compare_radas_to_mavrin()

@pytest.mark.order(4)
@pytest.mark.filterwarnings("error")
def test_compare_to_mavrin(temp_output_directory, selected_species):

    mavrin_data = read_mavrin_data()

    ds = xr.open_dataset(temp_output_directory / f"{selected_species}.nc").pint.quantify()
    ds = ds.sel(dim_electron_density=1e20, method="nearest")

    Te = ds["electron_temp"]
    ne_tau = ds["ne_tau"]

    Lz_coeffs = mavrin_data[f"{selected_species}_Lz"]
    mean_charge_coeffs = mavrin_data[f"{selected_species}_mean_charge"]

    Lz_mavrin = compute_Mavrin_polynomial_fit(Te, ne_tau, coeff=Lz_coeffs).squeeze().pint.quantify(ureg.W * ureg.m**3)
    mean_charge_mavrin = compute_Mavrin_polynomial_fit(Te, ne_tau, coeff=mean_charge_coeffs).squeeze()

    Lz_radas = ds["equilibrium_Lz"]
    mean_charge_radas = ds["equilibrium_mean_charge_state"]

    # The Mavrin curves are fitted to the data and have some occasional spikes. They agree
    # overall pretty well with the radas results, but there are some (seemingly spurious)
    # features which are not in the radas results. As such, our comparison has a very
    # loose tolerance.

    # Make sure that 90% of the Mavrin and radas Lz curves are within 50% of each other.
    assert (np.abs(1.0 - Lz_mavrin / Lz_radas)).quantile(0.9) < 0.5

    # Make sure that 90% of the Mavrin and radas <Z> curves are within half of a charge
    # state of each other
    assert (np.abs(mean_charge_mavrin - mean_charge_radas)).quantile(0.9) < 0.5
