import pytest
import numpy as np
from radas.mavrin_reference import read_mavrin_data, mavrin_species, compute_Mavrin_polynomial_fit
from radas.mavrin_reference.compare_against_mavrin import make_parameters, run_case

@pytest.fixture()
def Lz_tolerance():
    return 1.5

@pytest.fixture()
def mean_charge_tolerance():
    return 1.0

@pytest.mark.parametrize("species", mavrin_species(), ids=mavrin_species())
def test_against_mavrin_coronal(species, Lz_tolerance, mean_charge_tolerance):

    mavrin_data = read_mavrin_data()

    parameters = make_parameters(species)
    parameters["residence_time"] = 1.0
    dataset = run_case(species, parameters)

    Lz_radas = dataset["coronal_electron_emission_prefactor"]
    mean_charge_radas = dataset["coronal_mean_charge_state"]
    
    Te = dataset["electron_temperature"]
    ne_tau = dataset["ne_tau"]

    Lz_mavrin = compute_Mavrin_polynomial_fit(Te, ne_tau, coeff=mavrin_data[f"{species}_Lz"]).squeeze()
    mean_charge_mavrin = compute_Mavrin_polynomial_fit(Te, ne_tau, coeff=mavrin_data[f"{species}_mean_charge"]).squeeze()

    check_result(Lz_mavrin, Lz_radas, mean_charge_mavrin, mean_charge_radas, species, Te, ne_tau, Lz_tolerance, mean_charge_tolerance)

@pytest.mark.parametrize("species", mavrin_species(), ids=mavrin_species())
def test_against_mavrin_noncoronal(species, Lz_tolerance, mean_charge_tolerance):

    mavrin_data = read_mavrin_data()

    parameters = make_parameters(species)
    dataset = run_case(species, parameters)

    Lz_radas = dataset["noncoronal_electron_emission_prefactor"]
    mean_charge_radas = dataset["noncoronal_mean_charge_state"]
    
    ne = dataset["electron_density"]
    Te = dataset["electron_temperature"]
    ne_tau = dataset["ne_tau"]

    Lz_mavrin = compute_Mavrin_polynomial_fit(Te, ne_tau, coeff=mavrin_data[f"{species}_Lz"]).squeeze()
    mean_charge_mavrin = compute_Mavrin_polynomial_fit(Te, ne_tau, coeff=mavrin_data[f"{species}_mean_charge"]).squeeze()

    for i, ne_tau_value in enumerate(ne_tau.values):
        tau = ne_tau_value / ne
        check_result(Lz_mavrin.sel(dim_residence_time=tau), Lz_radas.sel(dim_residence_time=tau),
                     mean_charge_mavrin.sel(dim_residence_time=tau), mean_charge_radas.sel(dim_residence_time=tau),
                     species, Te, ne_tau_value, Lz_tolerance, mean_charge_tolerance)

def check_result(Lz_mavrin, Lz_radas, mean_charge_mavrin, mean_charge_radas, species, Te, ne_tau, Lz_tolerance, mean_charge_tolerance):

    test_failed = not (
        # We take the mean, since there are some fine features in the curves which don't agree.
        (1/Lz_tolerance < np.mean(Lz_mavrin / Lz_radas.values) < Lz_tolerance)
        and
        (np.max(np.abs((mean_charge_mavrin - mean_charge_radas.values))) < mean_charge_tolerance)
    )

    if test_failed:
        raise AssertionError(f"Test failed. Lz was not {1/Lz_tolerance} < {np.mean(Lz_mavrin / Lz_radas.values).values} < {Lz_tolerance} "
                             +f"or mean charge was not {np.max(np.abs((mean_charge_mavrin - mean_charge_radas.values))).values} < {mean_charge_tolerance}")
