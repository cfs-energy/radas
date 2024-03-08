import xarray as xr
import matplotlib.pyplot as plt
import warnings
from .read_mavrin_data import (
    read_mavrin_data,
    compute_Mavrin_polynomial_fit,
    mavrin_species,
)
from ..shared import output_directory
from ..unit_handling import ureg

def compare_radas_to_mavrin():
    for species in mavrin_species():
        if (output_directory / f"{species}.nc").exists():
            compare_radas_to_mavrin_per_species(species)
        else:
            print(f"Dataset not available for {species}. Skipping")

def compare_radas_to_mavrin_per_species(species: str):
    mavrin_data = read_mavrin_data()

    ds = xr.open_dataset(f"output/{species}.nc").pint.quantify()
    ds = ds.sel(dim_electron_density=1e20, method="nearest")

    Te = ds["electron_temp"]
    ne_tau = ds["ne_tau"]

    Lz_coeffs = mavrin_data[f"{species}_Lz"]
    mean_charge_coeffs = mavrin_data[f"{species}_mean_charge"]

    Lz_mavrin = compute_Mavrin_polynomial_fit(Te, ne_tau, coeff=Lz_coeffs).squeeze().pint.quantify(ureg.W * ureg.m**3)
    mean_charge_mavrin = compute_Mavrin_polynomial_fit(Te, ne_tau, coeff=mean_charge_coeffs).squeeze()

    Lz_radas = ds["equilibrium_Lz"]
    mean_charge_radas = ds["equilibrium_mean_charge_state"]

    fig, axs = plt.subplots(ncols=2)

    for i in range(ds.sizes["dim_ne_tau"]):
        ne_tau = ds.ne_tau.isel(dim_ne_tau=i).item()

        Lz_radas.isel(dim_ne_tau=i).plot(ax=axs[0], label=f"{ne_tau:~P}", color=f"C{i}")
        Lz_mavrin.isel(dim_ne_tau=i).plot(ax=axs[0], color=f"C{i}", linestyle="--")

        mean_charge_radas.isel(dim_ne_tau=i).plot(ax=axs[1], color=f"C{i}")
        mean_charge_mavrin.isel(dim_ne_tau=i).plot(ax=axs[1], color=f"C{i}", linestyle="--")

    axs[0].legend()
    axs[0].set_yscale("log")
    axs[0].set_ylim(*Lz_coeffs["ylims"])
    axs[1].set_ylim(*mean_charge_coeffs["ylims"])

    axs[0].set_title("$L_z$")
    axs[1].set_title("$<Z>$")

    for ax in axs.flatten():
        ax.set_xscale("log")
        ax.set_xlabel("$T_e$ [$eV$]")

    plt.suptitle(species)

    plt.savefig(output_directory / f"{species}.png")
