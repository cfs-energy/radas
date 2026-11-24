"""Compare radas results to the Mavrin polynomial fits."""

import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from .read_mavrin_data import (
    read_mavrin_data,
    compute_Mavrin_polynomial_fit,
)
from ..unit_handling import ureg, magnitude_in_units


def compare_radas_to_mavrin(output_dir: Path):
    """Compare radas results to the Mavrin fits for every species with a NetCDF file in output_dir."""
    for output_file in output_dir.iterdir():
        if output_file.suffix == ".nc":
            species = output_file.stem
            compare_radas_to_mavrin_per_species(output_dir, species)


def _mavrin_fit_or_none(mavrin_data: dict, key: str, Te, ne_tau, units=None):
    """Evaluate the Mavrin fit for key, or return None if Mavrin provides no fit for it."""
    if key not in mavrin_data:
        return None
    fit = compute_Mavrin_polynomial_fit(Te, ne_tau, coeff=mavrin_data[key])
    return fit if units is None else fit.pint.quantify(units)


def compare_radas_to_mavrin_per_species(output_dir: Path, species: str, max_decades: int = 4, show: bool = False):
    """Plot the radas equilibrium Lz and mean charge state against the Mavrin fits for one species."""
    mavrin_data = read_mavrin_data()

    ds = xr.open_dataset(output_dir / f"{species}.nc").pint.quantify()
    ds = ds.sel(dim_electron_density=1e20, method="nearest")

    Te = ds["electron_temp"]
    ne_tau = ds["ne_tau"]

    Lz_mavrin = _mavrin_fit_or_none(mavrin_data, f"{species}_Lz", Te, ne_tau, units=ureg.W * ureg.m**3)
    mean_charge_mavrin = _mavrin_fit_or_none(mavrin_data, f"{species}_mean_charge", Te, ne_tau)

    Lz_radas = ds["equilibrium_Lz"].pint.to(ureg.W * ureg.m**3)
    mean_charge_radas = ds["equilibrium_mean_charge_state"].pint.to(ureg.dimensionless)

    _, axs = plt.subplots(ncols=2, nrows=2, sharex="all", sharey="row")

    for i in range(ds.sizes["dim_ne_tau"]):
        ne_tau = ds.ne_tau.isel(dim_ne_tau=i).item()

        Lz_radas.isel(dim_ne_tau=i).plot(ax=axs[0][0], color=f"C{i}")
        if Lz_mavrin is not None:
            Lz_mavrin.isel(dim_ne_tau=i).plot(ax=axs[0][1], color=f"C{i}", label=f"{ne_tau:~.1P}")
        else:
            axs[0][1].plot([], [], color=f"C{i}", label=f"{ne_tau:~.1P}")

        mean_charge_radas.isel(dim_ne_tau=i).plot(ax=axs[1][0], color=f"C{i}")
        if mean_charge_mavrin is not None:
            mean_charge_mavrin.isel(dim_ne_tau=i).plot(ax=axs[1][1], color=f"C{i}")

    ds["coronal_Lz"].pint.to(ureg.W * ureg.m**3).plot(ax=axs[0][0], color="k")
    axs[0][1].plot([], [], color="k", label="coronal")
    ds["coronal_mean_charge_state"].pint.to(ureg.dimensionless).plot(ax=axs[1][0], color="k")

    axs[0][1].legend()

    Lz_radas_mag = magnitude_in_units(Lz_radas, ureg.W * ureg.m**3)
    Lz_coronal_mag = magnitude_in_units(ds["coronal_Lz"], ureg.W * ureg.m**3)
    Lz_min = min(Lz_radas_mag.min(), Lz_coronal_mag.min())
    Lz_max = max(Lz_radas_mag.max(), Lz_coronal_mag.max())

    for ax in axs.flatten():
        ax.set_title("")
        ax.set_ylabel("")

    axs[0][0].set_yscale("log")
    axs[0][0].set_ylim(max(Lz_min, Lz_max / 10**max_decades) / 2, Lz_max * 2)
    axs[1][0].set_ylim(0, ds.atomic_number * 1.2)

    axs[0][0].set_ylabel("$L_z$ $[W m^3]$")
    axs[1][0].set_ylabel("$<Z>$")

    axs[0][0].set_title("radas")
    axs[0][1].set_title("Mavrin")

    for ax in axs.flatten():
        ax.set_xscale("log")
        ax.set_xlabel("")

    for ax in axs[-1][:].flatten():
        ax.set_xlabel("$T_e$ [$eV$]")

    plt.suptitle(species)

    if show:
        plt.show()

    plt.savefig(output_dir / f"{species}.png", dpi=300)
