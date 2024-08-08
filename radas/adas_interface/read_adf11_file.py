from pathlib import Path
import importlib.util
from .fortran_reader import FReader, read_single_values, read_1d_array, read_2d_array
import numpy as np


def load_library(library_name: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(library_name, filepath)

    library = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(library)

    return library


def read_adf11_file(
    data_file_dir, species_name, dataset_type
) -> dict:
    """Open and read an ADF11 OpenADAS file.
    
    Uses the format specification from https://www.adas.ac.uk/man/appxa-11.pdf
    """


    filename = data_file_dir / f"{species_name}_{dataset_type}.dat"
    if not filename.exists():
        raise FileNotFoundError(f"{filename} does not exist.")

    f = filename.read_text().split("\n")

    IZMAX, IDMAXD, ITMAXD, IZ1MIN, IZ1MAX = FReader("5i5").read(f[0])

    # Initialise the value reader
    
    # IMPORTANT: do not reinitialise the reader, since otherwise we'll loose the position of the reader
    # Note that line_numbers will be reported from the read-offset
    value_reader = read_single_values(f, start_at_line=2, format_spec="8f10.5")

    DDENSD, line_number, _ = read_1d_array(value_reader, number_of_values=IDMAXD)
    DTEVD, line_number, _ = read_1d_array(value_reader, number_of_values=ITMAXD)

    DRCOFD = np.zeros((IZMAX, ITMAXD, IDMAXD))

    for IZ1 in range(IZMAX):
        value_reader = read_single_values(f, start_at_line=line_number + 2, format_spec="8f10.5")
        DRCOFD[IZ1, :, :], line_number, _ = read_2d_array(value_reader, rows=IDMAXD, columns=ITMAXD, fortran_order_arrays=False)

    data = dict(
        IZMAX = IZMAX,
        IDMAXD = IDMAXD,
        ITMAXD = ITMAXD,
        IZ1MIN = IZ1MIN,
        IZ1MAX = IZ1MAX,
        DDENSD = DDENSD,
        DTEVD = DTEVD,
        DRCOFD = DRCOFD,
    )

    return data
