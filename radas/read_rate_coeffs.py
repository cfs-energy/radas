from .unit_handling import Quantity, ureg, convert_units, dimensionless_magnitude
from .adas_interface.determine_adas_dataset_type import (
    determine_reader_class_and_config,
)
from importlib.metadata import version, PackageNotFoundError
import datetime
import xarray as xr
import numpy as np
from .interpolate_rates import interpolate_array

# Reference units for non-dimensionalizing coordinates
reference_electron_density = Quantity(1.0, ureg.m**-3)
reference_electron_temp = Quantity(1.0, ureg.eV)

def read_rate_coeff(data_file_dir, species_name, config):
    """
    Main pipeline to assemble an atomic rate dataset for a specific species.
    
    Reads raw ADAS files, standardizes their grids, aligns charge states, 
    and attaches metadata.
    """
    try:
        radas_version = version("radas")
    except PackageNotFoundError:
        radas_version = "UNDEFINED"
    
    # 1. Collect and sort data by year
    rate_coefficients = build_sorted_dictionary_of_rate_coefficients(config, species_name, data_file_dir)
    
    # 2. Resample all datasets to a common resolution
    rate_coefficients = interpolate_rates_onto_matching_grids(config, rate_coefficients)
    
    # 3. Merge individual datasets (e.g., recombination, ionization) into one
    try:
        dataset = xr.merge([v.rename(k) for k, v in rate_coefficients.items()], join="exact")
    except xr.AlignmentError as e:
        raise xr.AlignmentError(f"Alignment failed for {species_name}: {e}")

    if dataset.sizes["dim_electron_density"] <= 2 or dataset.sizes["dim_electron_temp"] <= 2:
        raise xr.AlignmentError(f"Alignment resulted in dataset sizes {dataset.sizes} for {species_name}.")

    # Convert dimensionless coordinates back to physical quantities
    dataset["electron_density"] = dataset["dim_electron_density"] * reference_electron_density
    dataset["electron_temp"] = dataset["dim_electron_temp"] * reference_electron_temp
    dataset["reference_electron_density"] = reference_electron_density
    dataset["reference_electron_temp"] = reference_electron_temp

    # 4. Standardize charge state indexing and attach global attributes
    dataset = align_rates_on_charge_states(dataset)
    dataset = dataset.assign_attrs(
        atomic_number=config["species"][species_name]["atomic_number"],
        species_name=species_name,
        radas_version=radas_version,
        created=datetime.date.today().strftime("%Y-%b-%d"),
    )

    return write_global_attributes(dataset, config["globals"])

def build_sorted_dictionary_of_rate_coefficients(config, species_name, data_file_dir):
    """Make a dictionary of rate coefficient datasets, ordered most-recent first."""
    rate_coefficients = dict()
    years = dict()

    for dataset_type, file_to_read in config["species"][species_name]["data_files"].items():
        reader_key, dataset_config = determine_reader_class_and_config(
            config["data_file_config"], dataset_type
        )

        # Extract year for sorting; entries can be a simple int or [file, year] list
        if isinstance(file_to_read, int):
            years[dataset_type] = file_to_read
        elif isinstance(file_to_read, list) and len(file_to_read) == 2:
            years[dataset_type] = file_to_read[1]
        else:
            raise NotImplementedError(f"Unsupported config format for {species_name} {dataset_type}")

        if reader_key == "adf11":
            rate_dataset = build_adf11_rate_dataset(
                data_file_dir, species_name, years[dataset_type], dataset_type, dataset_config,
            )
        else:
            raise NotImplementedError(f"No implementation for reader: {reader_key}")

        rate_coefficients[dataset_type] = rate_dataset.rate_coefficient
    
    # Sort keys by year descending
    sorted_keys = sorted(years, key=years.get, reverse=True)
    return {k: rate_coefficients[k] for k in sorted_keys}

def interpolate_rates_onto_matching_grids(config, rate_coefficients):
    """Resample all rate coefficients to a uniform log-grid defined by the newest dataset."""
    
    # Use the range of the most recent dataset to define the master grid
    most_recent_rate_coeff = list(rate_coefficients.values())[0]

    new_electron_density = np.logspace(
        np.log10(most_recent_rate_coeff["dim_electron_density"].min().item()),
        np.log10(most_recent_rate_coeff["dim_electron_density"].max().item()),
        num = config["globals"]["electron_density_resolution"]
    )

    new_electron_temp = np.logspace(
        np.log10(most_recent_rate_coeff["dim_electron_temp"].min().item()),
        np.log10(most_recent_rate_coeff["dim_electron_temp"].max().item()),
        num = config["globals"]["electron_temp_resolution"]
    )

    interpolated_rate_coefficients = dict()
    for key, value in rate_coefficients.items():
        # Map interpolation across charge states
        interpolated_rate_coefficients[key] = value.groupby("dim_charge_state").map(
            interpolate_array, args=(new_electron_density, new_electron_temp)
        )
    
    return interpolated_rate_coefficients

def write_global_attributes(dataset: xr.Dataset, globals: dict) -> xr.Dataset:
    """Attach global configuration parameters to the dataset as attributes or DataArrays."""
    for attribute, value in globals.items():
        if isinstance(value, dict):
            # Complex attributes (with units) are added as coordinates/DataArrays
            if np.ndim(value["value"]) >= 1:
                dataset[attribute] = xr.DataArray(
                    Quantity(value["value"], value["units"]),
                    coords={f"dim_{attribute}": value["value"]},
                )
            else:
                dataset[attribute] = Quantity(value["value"], value["units"])
        else:
            # Simple metadata (strings/ints) added as attributes
            dataset[attribute] = value
    return dataset

def build_adf11_rate_dataset(data_file_dir, species_name, year, dataset_type, dataset_config):
    """Read a specific ADF11 file and format it as a quantified xarray Dataset."""
    from .adas_interface.read_adf11_file import read_adf11_file

    data = read_adf11_file(data_file_dir, species_name, year, dataset_type)
    ds = xr.Dataset()

    # Log values stored in ADAS files are converted to linear scale if required
    electron_density = convert_units(Quantity(10**data["DDENSD"][:data["IDMAXD"]], ureg.cm**-3), ureg.m**-3)
    electron_temp = Quantity(10**data["DTEVD"][:data["ITMAXD"]], ureg.eV)

    coefficient = data["DRCOFD"][:data["IZMAX"], :data["ITMAXD"], :data["IDMAXD"]]
    if dataset_config["code"] <= 9:
        coefficient = 10**coefficient

    # Create dimensionless indices for internal processing
    dim_electron_density = dimensionless_magnitude(electron_density / reference_electron_density)
    dim_electron_temp = dimensionless_magnitude(electron_temp / reference_electron_temp)
    
    rate_coefficient = xr.DataArray(coefficient, coords=dict(
        dim_charge_state = np.arange(data["IZMAX"]),
        dim_electron_temp = dim_electron_temp,
        dim_electron_density = dim_electron_density,
    )).pint.quantify(dataset_config["stored_units"])

    ds["rate_coefficient"] = convert_units(rate_coefficient, dataset_config["desired_units"])
    return ds

def align_rates_on_charge_states(dataset: xr.Dataset) -> xr.Dataset:
    """
    Standardize charge state mapping so index 'k' always refers to the reactant.
    
    For k+1 -> k reactions (e.g. recombination), the rates are shifted so that 
    index k represents the species being recombined.
    """
    # Pad to accommodate the N+1 charge state after shifting
    dataset = dataset.pad(pad_width=dict(dim_charge_state=(0, 1)), mode="constant", constant_values=0.0)
    dataset = dataset.assign_coords(dim_charge_state=np.arange(dataset.sizes["dim_charge_state"]))

    # Shift k+1 -> k processes
    keys_to_shift = [
        "effective_recombination",
        "charge_exchange_cross_coupling",
        "recombination_and_bremsstrahlung",
        "charge_exchange_emission",
    ]
    
    for key in [k for k in keys_to_shift if k in dataset]:
        dataset[key] = dataset[key].roll(dim_charge_state=+1)

    return dataset