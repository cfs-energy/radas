#!.venv/bin/python
# Run this script from the repository directory.

import numpy as np
import yaml
import matplotlib.pyplot as plt

from radas.computation import (
    read_cases,
    build_rate_coefficients,
    calculate_coronal_states,
    calculate_radiation,
    calculate_derivatives,
)
from radas.unit_handling import ureg
from radas.mavrin_reference import read_mavrin_data, mavrin_species, compute_Mavrin_polynomial_fit

def make_parameters(species):
    return {
        "species": species,
        "electron_density": 1e+19,
        "neutral_density": 0.0,
        "electron_temperature": np.logspace(0, 4).tolist(),
        "evolution_start": 1e-08,
        "evolution_stop": 100.0,
        "residence_time": (np.logspace(16, 19, num=5) / 1e+19).tolist(),
        "file_output": {"netcdf": False},
        "plotting": {},
    }

def run_case(species, parameters):
    # Read in the file we just wrote
    dataset, _, _ = read_cases.read_case(species, parameters=parameters)

    dataset = read_cases.convert_enums_for_parameters(dataset)
    dataset = build_rate_coefficients.build_rate_coefficients(dataset)

    dataset["coronal_charge_state_fraction"] = calculate_coronal_states.calculate_coronal_states(dataset)
    dataset["coronal_mean_charge_state"] = (dataset.coronal_charge_state_fraction * dataset.dim_charge_state).sum(dim="dim_charge_state")
    dataset["coronal_electron_emission_prefactor"] = calculate_radiation.calculate_electron_emission_prefactor(dataset, dataset.coronal_charge_state_fraction)
    
    dataset["ne_tau"] = (dataset.electron_density * dataset.residence_time).pint.to(ureg.m**-3 * ureg.s)
    dataset["charge_state_fraction_evolution"] = calculate_derivatives.calculate_time_evolution(dataset)
    dataset["charge_state_fraction_at_equilibrium"] = dataset.charge_state_fraction_evolution.isel(dim_time=-1)
    dataset["noncoronal_mean_charge_state"] = (dataset.charge_state_fraction_at_equilibrium * dataset.dim_charge_state).sum(dim="dim_charge_state")
    dataset["noncoronal_electron_emission_prefactor"] = calculate_radiation.calculate_electron_emission_prefactor(dataset, dataset.charge_state_fraction_at_equilibrium)

    return dataset.squeeze()

if __name__=="__main__":

    mavrin_data = read_mavrin_data()

    for species in mavrin_species():

        parameters = make_parameters(species)
        dataset = run_case(species, parameters)

        ne = dataset["electron_density"]
        Te = dataset["electron_temperature"]
        ne_tau = dataset["ne_tau"]

        noncoronal_Lz_radas = dataset["noncoronal_electron_emission_prefactor"]
        noncoronal_mean_charge_radas = dataset["noncoronal_mean_charge_state"]

        noncoronal_Lz_mavrin = compute_Mavrin_polynomial_fit(Te, ne_tau, coeff=mavrin_data[f"{species}_Lz"]).squeeze()
        noncoronal_mean_charge_mavrin = compute_Mavrin_polynomial_fit(Te, ne_tau, coeff=mavrin_data[f"{species}_mean_charge"]).squeeze()

        fig, axs = plt.subplots(ncols=2)
        axs[0].loglog(Te, dataset["coronal_electron_emission_prefactor"], "k-")
        axs[1].semilogx(Te, dataset["coronal_mean_charge_state"], "k-", label="coronal")

        for i, ne_tau_value in enumerate(ne_tau.values):
            tau = ne_tau_value / ne
            axs[0].loglog(Te, noncoronal_Lz_mavrin.sel(dim_residence_time=tau), f"C{i}--")
            axs[0].loglog(Te, noncoronal_Lz_radas.sel(dim_residence_time=tau), f"C{i}-")

            axs[1].semilogx(Te, noncoronal_mean_charge_mavrin.sel(dim_residence_time=tau), f"C{i}--")
            axs[1].semilogx(Te, noncoronal_mean_charge_radas.sel(dim_residence_time=tau), f"C{i}-", label=f"{ne_tau_value:3.2e}")
        
        axs[0].set_ylabel("$L_z$")
        axs[1].set_ylabel("$\\langle Z \\rangle$")
        axs[1].legend()

        fig.suptitle(f"{species}")
        
        plt.tight_layout()
        plt.savefig(species, dpi=300)