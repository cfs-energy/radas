#!.venv/bin/python
# Run this script from the repository directory.
import click
import matplotlib.pyplot as plt

from radas.computation import (
    read_cases,
    build_rate_coefficients,
    calculate_coronal_states,
    calculate_radiation,
    calculate_derivatives,
)
from radas import plotting

@click.command()
@click.argument("case", type=click.Choice(read_cases.list_cases()))
@click.option("--show", is_flag=True, help="Display an interactive figure of the result")
def run_radas(case: str, show: bool):
    dataset = read_cases.read_case(case)

    dataset = read_cases.convert_enums_for_parameters(dataset)
    dataset = build_rate_coefficients.build_rate_coefficients(dataset)

    dataset["coronal_charge_state_fraction"] = calculate_coronal_states.calculate_coronal_states(dataset)
    dataset["mean_charge_state"] = (dataset.coronal_charge_state_fraction * dataset.dim_charge_state).sum(dim="dim_charge_state")
    dataset["coronal_electron_emission_prefactor"] = calculate_radiation.calculate_electron_emission_prefactor(dataset, dataset.coronal_charge_state_fraction)

    dataset["charge_state_fraction_evolution"] = calculate_derivatives.calculate_time_evolution(dataset)
    dataset["charge_state_fraction_at_equilibrium"] = dataset.charge_state_fraction_evolution.isel(dim_time=-1)
    dataset["noncoronal_electron_emission_prefactor"] = calculate_radiation.calculate_electron_emission_prefactor(dataset, dataset.charge_state_fraction_at_equilibrium)

    # plotting.plot_time_evolution(dataset)

    # plotting.plot_charge_state_fraction(dataset)
    # plotting.plot_charge_states(dataset)
    # plotting.plot_mean_charge_state(dataset)
    plotting.plot_electron_emission_prefactor(dataset.coronal_electron_emission_prefactor)
    plotting.plot_electron_emission_prefactor(dataset.noncoronal_electron_emission_prefactor)

    if show:
        plt.show()

if __name__=="__main__":
    run_radas()
