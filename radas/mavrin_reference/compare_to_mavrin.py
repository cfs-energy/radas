import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from .read_mavrin_data import (
    read_mavrin_data,
    compute_Mavrin_polynomial_fit,
)
from ..unit_handling import ureg


def compare_radas_to_mavrin(output_dir: Path):

    for output_file in output_dir.iterdir():
        if output_file.suffix == ".nc":
            species = output_file.stem
            compare_radas_to_mavrin_per_species(output_dir, species)


def compare_radas_to_mavrin_per_species(output_dir: Path, species: str):
    mavrin_data = read_mavrin_data()

    ds = xr.open_dataset(output_dir / f"{species}.nc").pint.quantify()
    ds = ds.sel(dim_electron_density=1e20, method="nearest")

    Te = ds["electron_temp"]
    ne_tau = ds["ne_tau"]

    if f"{species}_Lz" in mavrin_data:
        Lz_coeffs = mavrin_data.get(f"{species}_Lz")
        Lz_mavrin = (
            compute_Mavrin_polynomial_fit(Te, ne_tau, coeff=Lz_coeffs)
            .squeeze()
            .pint.quantify(ureg.W * ureg.m**3)
        )
    else:
        Lz_mavrin = None

    if f"{species}_mean_charge" in mavrin_data:
        mean_charge_coeffs = mavrin_data.get(f"{species}_mean_charge")
        mean_charge_mavrin = compute_Mavrin_polynomial_fit(
            Te, ne_tau, coeff=mean_charge_coeffs
        ).squeeze()
    else:
        mean_charge_mavrin = None

    Lz_radas = ds["equilibrium_Lz"].pint.to(ureg.W * ureg.m**3)
    mean_charge_radas = ds["equilibrium_mean_charge_state"].pint.to(ureg.dimensionless)

    fig, axs = plt.subplots(ncols=2)

    for i in range(ds.sizes["dim_ne_tau"]):
        ne_tau = ds.ne_tau.isel(dim_ne_tau=i).item()

        Lz_radas.isel(dim_ne_tau=i).plot(ax=axs[0], label=f"{ne_tau:~P}", color=f"C{i}")
        if Lz_mavrin is not None:
            Lz_mavrin.isel(dim_ne_tau=i).plot(ax=axs[0], color=f"C{i}", linestyle="--")

        mean_charge_radas.isel(dim_ne_tau=i).plot(ax=axs[1], color=f"C{i}")
        if mean_charge_mavrin is not None:
            mean_charge_mavrin.isel(dim_ne_tau=i).plot(
                ax=axs[1], color=f"C{i}", linestyle="--"
            )

    axs[0].legend()
    axs[0].set_yscale("log")
    if Lz_mavrin is not None:
        axs[0].set_ylim(*Lz_coeffs["ylims"])
    if mean_charge_mavrin is not None:
        axs[1].set_ylim(*mean_charge_coeffs["ylims"])

    axs[0].set_title("$L_z$ $[W m^3]$")
    axs[1].set_title("$<Z>$")

    for ax in axs.flatten():
        ax.set_xscale("log")
        ax.set_xlabel("$T_e$ [$eV$]")
        ax.set_ylabel("")

    plt.suptitle(species)

    plt.savefig(output_dir / f"{species}.png")
