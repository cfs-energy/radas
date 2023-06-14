import pytest
import numpy as np
import xarray as xr
import warnings
import yaml

from radas.computation import (
    read_cases,
    build_rate_coefficients,
    calculate_coronal_states,
    calculate_radiation,
    calculate_derivatives,
)
from radas.unit_handling import ureg
from radas.mavrin_reference import read_mavrin_data, mavrin_species, compute_Mavrin_polynomial_fit

@pytest.mark.parametrize("species", mavrin_species(), ids=mavrin_species())
def test_against_mavrin_coronal(species, tmp_path):

    input_file = tmp_path / "input.yaml"
    mavrin_data = read_mavrin_data()

    parameters = {
        "species": species,
        "electron_density": 1e+19,
        "neutral_density": 0.0,
        "electron_temperature": np.logspace(0, 4).tolist(),
        "evolution_start": 1e-08,
        "evolution_stop": 100.0,
        "residence_time": 1.0,
        "file_output": {"netcdf": False},
        "plotting": {},
    }

    # Turn the dictionary into temporary file so that we can read it using the standard methods
    with open(input_file, "w") as file:
        yaml.dump(parameters, file)
    
    # Read in the file we just wrote
    dataset, _, _ = read_cases.read_case(species, input_file=input_file)

    dataset = read_cases.convert_enums_for_parameters(dataset)
    dataset = build_rate_coefficients.build_rate_coefficients(dataset)

    dataset["coronal_charge_state_fraction"] = calculate_coronal_states.calculate_coronal_states(dataset)
    dataset["mean_charge_state"] = (dataset.coronal_charge_state_fraction * dataset.dim_charge_state).sum(dim="dim_charge_state")
    dataset["coronal_electron_emission_prefactor"] = calculate_radiation.calculate_electron_emission_prefactor(dataset, dataset.coronal_charge_state_fraction)
    dataset["ne_tau"] = (dataset.electron_density * dataset.residence_time).pint.to(ureg.m**-3 * ureg.s)

    dataset = dataset.squeeze()

    Lz_radas = dataset["coronal_electron_emission_prefactor"].values
    mean_charge_radas = dataset["mean_charge_state"].values
    
    Te = dataset["electron_temperature"]
    ne_tau = dataset["ne_tau"]

    Lz_mavrin = compute_Mavrin_polynomial_fit(Te, ne_tau, coeff=mavrin_data[f"{species}_Lz"]).squeeze()
    mean_charge_mavrin = compute_Mavrin_polynomial_fit(Te, ne_tau, coeff=mavrin_data[f"{species}_mean_charge"]).squeeze()

    # Make sure the Lz curves come out within a factor of "tolerance_ratio" of each other
    # We take the mean, since there are some fine features in the curves which don't agree.
    tolerance_ratio = 1.25
    assert 1/tolerance_ratio < np.mean(Lz_mavrin / Lz_radas) < tolerance_ratio

    # Make sure that the mean charge state is always within 1.0
    assert np.max(np.abs((mean_charge_mavrin - mean_charge_radas))) < 1.0
