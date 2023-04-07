import matplotlib.pyplot as plt
import xarray as xr

def plot_charge_state_fraction(dataset: xr.Dataset):
    fig, ax = plt.subplots()
    dataset.charge_state_fraction.plot(ax=ax)
    ax.set_xscale("log")

def plot_charge_states(dataset: xr.Dataset):
    fig, ax = plt.subplots()

    for Z in range(dataset.atomic_number+1):
        dataset.charge_state_fraction.sel(dim_charge_state=Z).plot(ax=ax, label=Z)

    ax.set_xscale("log")
    ax.legend()
    ax.grid()

def plot_mean_charge_state(dataset: xr.Dataset):
    fig, ax = plt.subplots()
    dataset.mean_charge_state.plot(ax=ax)
    ax.set_xscale("log")

def plot_electron_emission_prefactor(electron_emission_prefactor: xr.DataArray):
    
    fig, ax = plt.subplots()
    if "dim_refuelling_time" in electron_emission_prefactor.dims:
        for i in range(electron_emission_prefactor.sizes["dim_refuelling_time"]):
            electron_emission_prefactor.isel(dim_refuelling_time=i).plot(ax=ax)
    else:
        electron_emission_prefactor.plot(ax=ax)

    ax.set_yscale("log")
    ax.set_xscale("log")

def plot_time_evolution(dataset: xr.Dataset):
    
    point = dataset.sel(dim_electron_temperature=50.0, dim_electron_density=1e19, method="nearest")

    fig, ax = plt.subplots()
    for z in range(dataset.atomic_number+1):
        label=f"${dataset.species.value[0]}^{{{z}{'+' if z>0 else ''}}}$"
        point.charge_state_fraction_evolution.sel(dim_charge_state=z).plot(ax=ax, label=label)
    ax.set_xscale("log")

    ax.legend()
    ax.set_title(f"Charge state ({dataset.species.name}, $T_e=50eV, n_e=10^{{19}}m^{{-3}}$)")
    ax.set_xlabel("Time after injection [$s$]")
