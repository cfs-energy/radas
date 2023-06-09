from pathlib import Path
import yaml
import xarray as xr
import numpy as np
import warnings

def read_mavrin_data():
    input_file = Path(__file__).parent / "mavrin_data.yml"
    with open(input_file) as file:
        mavrin_data = yaml.load(file, Loader=yaml.FullLoader)
    
    return mavrin_data

def mavrin_species():
    mavrin_data = read_mavrin_data()

    keys = mavrin_data.keys()

    species = []
    for key in keys:
        species_key = key.removesuffix("_Lz").removesuffix("_mean_charge")
        if not species_key in species: species.append(species_key)
    
    return species

def compute_Mavrin_polynomial_fit(Te_eV, ne_tau_s_per_m3, coeff):
    """Compute Lz or mean_charge curves from Mavrin, J. Fus. Eng., 2017."""
    return xr.apply_ufunc(
        compute_Mavrin_polynomial_fit_single,
        Te_eV, ne_tau_s_per_m3,
        kwargs=dict(coeff=coeff),
        vectorize=True,
    )

def compute_Mavrin_polynomial_fit_single(Te_eV, ne_tau_s_per_m3, coeff):
    """Inner loop for computing the Lz or mean_charge polynomial fit from Mavrin, J. Fus. Eng., 2017."""
    Tmin_eV = coeff["Tmin_eV"]
    Tmax_eV = coeff["Tmax_eV"]

    if not Tmin_eV[0] <= Te_eV <= Tmax_eV[-1]:
        warnings.warn(f"{Te_eV}eV outside fitted range {Tmin_eV[0]}eV to {Tmax_eV[-1]}eV")
        return np.nan
    if ne_tau_s_per_m3 < 1e15:
        warnings.warn(f"{ne_tau_s_per_m3} outside fitted range above 1e16 m^-3 s")
        return np.nan

    X = np.log10(Te_eV)
    Y = np.log10(ne_tau_s_per_m3 / 1e19)
    if Y > 0.0: warnings.warn(f"Warning: treating points with ne_tau_s_per_m3 > 1e19 m^-3 s as coronal.")
    Y = np.minimum(Y, 0.0)

    N_bins = len(Tmin_eV)
    assert len(Tmax_eV) == N_bins

    for i in range(N_bins):
        if Tmin_eV[i] <= Te_eV <= Tmax_eV[i]:
            T_bin = i

    A = np.zeros(10)
    for i in range(10):
        A[i] = coeff[f"A{i}"][T_bin]

    F = A[0] + A[1]*X + A[2]*Y + A[3]*X**2 + A[4]*X*Y + A[5]*Y**2 + A[6]*X**3 + A[7]*X**2*Y + A[8]*X*Y**2 + A[9]*Y**3

    return np.power(10, F)
