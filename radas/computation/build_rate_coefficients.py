import numpy as np
import xarray as xr

from ..named_options import ADF11Dataset, AtomicSpecies
from ..adas_file_readers import read_adf11_file
from .interpolate_rate_coefficient import interpolate_rate_coefficient

def build_rate_coefficients(dataset: xr.Dataset) -> xr.Dataset:
    """Read the ADAS rates and assign them to the dataset."""
    kwargs = dict(
        species=dataset.species,
        electron_density=dataset.electron_density,
        electron_temperature=dataset.electron_temperature,
    )
    dataset = dataset.assign_attrs(atomic_number=dataset.species.value[1])
    dataset = dataset.assign_coords(dim_charge_state=np.arange(dataset.atomic_number+1))

    dataset["ionisation_rate_coeff"] = read_and_interpolate_rates(ADF11Dataset.EffectiveIonisationCoefficients, **kwargs)
    dataset["recombination_rate_coeff"] = read_and_interpolate_rates(ADF11Dataset.EffectiveRecombinationCoefficients, **kwargs)
    dataset["charge_exchange_rate_coeff"] = read_and_interpolate_rates(ADF11Dataset.CXCrossCouplingCoefficients, **kwargs)

    dataset["line_emission_coeff"] = read_and_interpolate_rates(ADF11Dataset.LineEmissionFromExcitation, **kwargs)
    dataset["continuum_emission_coeff"] = read_and_interpolate_rates(ADF11Dataset.RecombinationAndBremsstrahlung, **kwargs)
    dataset["charge_exchange_emission_coeff"] = read_and_interpolate_rates(ADF11Dataset.ChargeExchangeEmission, **kwargs)

    dataset["mean_ionisation_potential"] = read_and_interpolate_rates(ADF11Dataset.MeanIonisationPotential, **kwargs)

    return dataset

def read_and_interpolate_rates(adf11_dataset: ADF11Dataset, species: AtomicSpecies, electron_density: xr.DataArray, electron_temperature: xr.DataArray) -> xr.DataArray:
    """Reads an ADAS rate from file and interpolates it to the region of interest (defined by electron_density and electron_temperature)."""
    rate_coefficient_dataset = read_adf11_file(dataset=adf11_dataset, species=species)
    
    return interpolate_rate_coefficient(rate_coefficient_dataset, electron_density, electron_temperature)