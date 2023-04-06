#!.venv/bin/python
# Run this script from the repository directory.
import click

from radas.computation import (
    read_cases,
    build_rate_coefficients,
    calculate_coronal_states,
    calculate_radiation,
)

@click.command()
@click.argument("case", type=click.Choice(read_cases.list_cases()))
@click.option("--show", is_flag=True, help="Display an interactive figure of the result")
def run_radas(case: str, show: bool):
    dataset = read_cases.read_case(case)

    dataset = read_cases.convert_enums_for_parameters(dataset)
    dataset = build_rate_coefficients.build_rate_coefficients(dataset)

    dataset["fractional_abundances"] = calculate_coronal_states.calculate_coronal_states(dataset)
    dataset["electron_emission_prefactor"] = calculate_radiation.calculate_electron_emission_prefactor(dataset)


    

if __name__=="__main__":
    run_radas()
