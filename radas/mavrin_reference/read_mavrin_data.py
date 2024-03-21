import xarray as xr
import numpy as np
import warnings
from ..unit_handling import ureg, convert_units, magnitude


def read_mavrin_data():
    from ..shared import open_yaml_file, mavrin_data_file

    "Opens the Mavrin dataset from a yaml file."
    return open_yaml_file(mavrin_data_file)


def compute_Mavrin_polynomial_fit(Te_eV, ne_tau_s_per_m3, coeff):
    """Compute Lz or mean_charge curves from Mavrin, J. Fus. Eng., 2017."""
    return xr.apply_ufunc(
        compute_Mavrin_polynomial_fit_single,
        magnitude(convert_units(Te_eV, ureg.eV)),
        magnitude(convert_units(ne_tau_s_per_m3, ureg.s / ureg.m**3)),
        kwargs=dict(coeff=coeff),
        vectorize=True,
    )


def compute_Mavrin_polynomial_fit_single(
    Te_eV, ne_tau_s_per_m3, coeff, warn: bool = False
):
    """Inner loop for computing the Lz or mean_charge polynomial fit from Mavrin, J. Fus. Eng., 2017."""
    Tmin_eV = coeff["Tmin_eV"]
    Tmax_eV = coeff["Tmax_eV"]

    if not Tmin_eV[0] <= Te_eV <= Tmax_eV[-1]:
        if warn:
            warnings.warn(
                f"{Te_eV}eV outside fitted range {Tmin_eV[0]}eV to {Tmax_eV[-1]}eV"
            )
        return np.nan
    if ne_tau_s_per_m3 < 1e15:
        if warn:
            warnings.warn(f"{ne_tau_s_per_m3} outside fitted range above 1e16 m^-3 s")
        return np.nan

    X = np.log10(Te_eV)
    Y = np.log10(ne_tau_s_per_m3 / 1e19)
    if warn and (Y > 0.0):
        warnings.warn(
            f"Warning: treating points with ne_tau_s_per_m3 > 1e19 m^-3 s as coronal."
        )
    Y = np.minimum(Y, 0.0)

    N_bins = len(Tmin_eV)
    assert len(Tmax_eV) == N_bins

    for i in range(N_bins):
        if Tmin_eV[i] <= Te_eV <= Tmax_eV[i]:
            T_bin = i

    A = np.zeros(10)
    for i in range(10):
        A[i] = coeff[f"A{i}"][T_bin]

    F = (
        A[0]
        + A[1] * X
        + A[2] * Y
        + A[3] * X**2
        + A[4] * X * Y
        + A[5] * Y**2
        + A[6] * X**3
        + A[7] * X**2 * Y
        + A[8] * X * Y**2
        + A[9] * Y**3
    )

    return np.power(10, F)
