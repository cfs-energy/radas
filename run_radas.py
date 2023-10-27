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
from radas import make_plots, write_output
from radas.directories import cases_directory
from radas.unit_handling import ureg
from radas.get_git_hash import get_git_revision_short_hash

@click.command()
@click.argument("case", type=click.Choice(read_cases.list_cases() + ["all"]))
@click.option("--show", is_flag=True, help="Display an interactive figure of the result")
def run_radas(case: str, show: bool):

    if case == "all":
        for case in read_cases.list_cases():
            print(f"Running {case}")
            run_radas_for_case(case)
    else:
        run_radas_for_case(case)

    if show:
        plt.show()

def run_radas_for_case(case: str):
    dataset, plots, file_output = read_cases.read_case(case)
    dataset.assign_attrs(radas_git_hash = get_git_revision_short_hash())

    dataset = read_cases.convert_enums_for_parameters(dataset)
    dataset = build_rate_coefficients.build_rate_coefficients(dataset)

    dataset["coronal_charge_state_fraction"] = calculate_coronal_states.calculate_coronal_states(dataset)
    dataset["coronal_mean_charge_state"] = (dataset.coronal_charge_state_fraction * dataset.dim_charge_state).sum(dim="dim_charge_state")
    dataset["coronal_electron_emission_prefactor"] = calculate_radiation.calculate_electron_emission_prefactor(dataset, dataset.coronal_charge_state_fraction)
    
    dataset["residence_time"] = (dataset.ne_tau / dataset.electron_density).pint.to(ureg.s)

    dataset["charge_state_fraction_evolution"] = calculate_derivatives.calculate_time_evolution(dataset)
    dataset["charge_state_fraction_at_equilibrium"] = dataset.charge_state_fraction_evolution.isel(dim_time=-1)
    dataset["noncoronal_mean_charge_state"] = (dataset.charge_state_fraction_at_equilibrium * dataset.dim_charge_state).sum(dim="dim_charge_state")
    dataset["noncoronal_electron_emission_prefactor"] = calculate_radiation.calculate_electron_emission_prefactor(dataset, dataset.charge_state_fraction_at_equilibrium)
    
    (cases_directory / dataset.case / "output").mkdir(exist_ok=True)

    write_output(dataset, file_output)

    # Plot the results
    figsize, show_dpi, save_dpi = plots.pop("figsize"), plots.pop("show_dpi"), plots.pop("save_dpi")
    for key, plot in plots.items():
        make_plots(dataset, key, plot, figsize, show_dpi, save_dpi)
    
if __name__=="__main__":
    run_radas()
