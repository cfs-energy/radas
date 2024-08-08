"""Run the main integrated radas program."""

import pytest
import xarray as xr
import numpy as np


@pytest.fixture()
def configuration():
    from radas.shared import default_config_file, open_yaml_file

    config_file = default_config_file
    return open_yaml_file(config_file)


@pytest.mark.order(1)
@pytest.mark.filterwarnings("error")
def test_read_yaml(configuration):
    pass


@pytest.mark.order(2)
@pytest.mark.filterwarnings("error")
def test_download_species_data(data_file_dir, selected_species, configuration, verbose):
    from radas.adas_interface import download_species_data

    species_config = configuration["species"][selected_species]

    # Call twice to check if reuse works
    for i in range(2):
        download_species_data(
            data_file_dir,
            selected_species,
            species_config,
            configuration["data_file_config"],
            verbose=verbose,
        )


@pytest.fixture()
def datasets(reader_dir, data_file_dir, selected_species, configuration, verbose):
    from radas import read_rate_coeff

    datasets = dict()

    datasets[selected_species] = read_rate_coeff(
        reader_dir, data_file_dir, selected_species, configuration
    )

    return datasets


@pytest.mark.order(3)
@pytest.mark.filterwarnings("error")
def test_datasets(datasets):
    pass


@pytest.mark.order(4)
@pytest.mark.filterwarnings("error")
def test_radas_computation(datasets, selected_species, output_dir, verbose):
    from radas import run_radas_computation

    run_radas_computation(datasets[selected_species], output_dir, verbose)


@pytest.mark.order(5)
@pytest.mark.filterwarnings("error")
def test_compare_radas_to_mavrin(output_dir):
    from radas.mavrin_reference import compare_radas_to_mavrin

    compare_radas_to_mavrin(output_dir)


@pytest.mark.order(6)
@pytest.mark.filterwarnings("error")
def test_compare_results_to_mavrin_reference(output_dir, selected_species):
    from radas.mavrin_reference import read_mavrin_data, compute_Mavrin_polynomial_fit
    from radas.unit_handling import ureg

    mavrin_data = read_mavrin_data()

    ds = xr.open_dataset(output_dir / f"{selected_species}.nc").pint.quantify()
    ds = ds.sel(dim_electron_density=1e20, method="nearest")

    Te = ds["electron_temp"]
    ne_tau = ds["ne_tau"]

    Lz_coeffs = mavrin_data[f"{selected_species}_Lz"]
    mean_charge_coeffs = mavrin_data[f"{selected_species}_mean_charge"]

    Lz_mavrin = (
        compute_Mavrin_polynomial_fit(Te, ne_tau, coeff=Lz_coeffs)
        .squeeze()
        .pint.quantify(ureg.W * ureg.m**3)
    )
    mean_charge_mavrin = compute_Mavrin_polynomial_fit(
        Te, ne_tau, coeff=mean_charge_coeffs
    ).squeeze()

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


@pytest.mark.order(7)
@pytest.mark.filterwarnings("error")
def test_cli(radas_dir, selected_species):
    from click.testing import CliRunner
    from radas.cli import run_radas_cli

    runner = CliRunner()
    result = runner.invoke(
        run_radas_cli,
        ["-d", str(radas_dir), "-s", selected_species, "-s", "hydrogen", "-vv"],
    )
    assert result.exit_code == 0
