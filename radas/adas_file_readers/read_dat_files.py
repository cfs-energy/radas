import numpy as np
import xarray as xr
from ..named_options.adf11_dataset import ADF11Dataset, ADF11Codes, ADF11RateCoeffStoredUnits, ADF11RateCoeffDesiredUnits
from ..named_options.atomic_species import AtomicSpecies
from ..directories import dat_files_directory
from ..unit_handling import Quantity, ureg

try:
    from . import fortran_file_handling
    from . import ADF11_reader
except ModuleNotFoundError:
    print("Error: could not find the compiled f2py OpenADAS readers. Have you run 'python adas_data/fetch_adas_data.py'?")


def read_adf11_file(dataset: ADF11Dataset, species: AtomicSpecies):
    """Open and read an ADF11 OpenADAS file.

    Key to outputs (from xxdata_11.pdf)
    type   | name       | description
    (i*4)  | iz0        | nuclear charge
    (i*4)  | is1min     | minimum ion charge + 1
           |            | (generalised to connection vector index)
    (i*4)  | is1max     | maximum ion charge + 1
           |            | (note excludes the bare nucleus)
           |            | (generalised to connection vector index and excludes
           |            | last one which always remains the bare nucleus)
    (i*4)  | nptnl      | number of partition levels in block
    (i*4)  | nptn()     | number of partitions in partition level
           |            | 1st dim: partition level
    (i*4)  | nptnc(,)   | number of components in partition
           |            | 1st dim: partition level
           |            | 2nd dim: member partition in partition level
    (i*4)  | iptnla()   | partition level label (0=resolved root,1=
           |            | unresolved root)
           |            | 1st dim: partition level index
    (i*4)  | iptna(,)   | partition member label (labelling starts at 0)
           |            | 1st dim: partition level index
           |            | 2nd dim: member partition index in partition level
    (i*4)  | iptnca(,,) | component label (labelling starts at 0)
           |            | 1st dim: partition level index
           |            | 2nd dim: member partition index in partition level
           |            | 3rd dim: component index of member partition
    (i*4)  | ncnct      | number of elements in connection vector
    (i*4)  | icnctv()   | connection vector of number of partitions
           |            | of each superstage in resolved case
           |            | including the bare nucleus
           |            | 1st dim: connection vector index
    (i*4)  | iblmx      | number of (sstage, parent, base)
           |            | blocks in isonuclear master file
    (i*4)  | ismax      | number of charge states
           |            | in isonuclear master file
           |            | (generalises to number of elements in
           |            |  connection vector)
    (c*12) | dnr_ele    | CX donor element name for iclass = 3 or 5
           |            | (blank if unset)
    (r*8)  | dnr_ams    | CX donor element mass for iclass = 3 or 5
           |            | (0.0d0 if unset)
    (i*4)  | isppr()    | 1st (parent) index for each partition block
           |            | 1st dim: index of (sstage, parent, base)
           |            |          block in isonuclear master file
    (i*4)  | ispbr()    | 2nd (base) index for each partition block
           |            | 1st dim: index of (sstage, parent, base)
           |            |          block in isonuclear master file
    (i*4)  | isstgr()   | s1 for each resolved data block
           |            | (generalises to connection vector index)
           |            | 1st dim: index of (sstage, parent, base)
           |            |          block in isonuclear master file
    (i*4)  | idmax      | number of dens values in
           |            | isonuclear master files
    (i*4)  | itmax      | number of temp values in
           |            | isonuclear master files
    (r*8)  | ddens()    | log10(electron density(cm-3)) from adf11
    (r*8)  | dtev()     | log10(electron temperature (eV) from adf11
    (r*8)  | drcof(,,)  | if(iclass <=9):
           |            | 	log10(coll.-rad. coefft.) from
           |            | 	isonuclear master file
           |            | if(iclass >=10):
           |            | 	coll.-rad. coefft. from
           |            | 	isonuclear master file
           |            | 1st dim: index of (sstage, parent, base)
           |            | 		 block in isonuclear master file
           |            | 2nd dim: electron temperature index
           |            | 3rd dim: electron density index
    (l*4)  | lres       | = .true. => partial file
           |            | = .false. => not partial file
    (l*4)  | lstan      | = .true. => standard file
           |            | = .false. => not standard file
    (l*4)  | lptn       | = .true. => partition block present
           |            | = .false. => partition block not present
    """
    filename = dat_files_directory / f"{species.name}_{dataset.name}.dat"
    
    if not filename.exists(): raise FileNotFoundError(f"{filename} does not exist.")

    file_unit = 10

    fortran_file_handling.open_file(str(filename), file_unit)

    (
        iz0,
        is1min,
        is1max,
        nptnl,
        nptn,
        nptnc,
        iptnla,
        iptna,
        iptnca,
        ncnct,
        icnctv,
        iblmx,
        ismax,
        dnr_ele,
        dnr_ams,
        isppr,
        ispbr,
        isstgr,
        idmax,
        itmax,
        ddens,
        dtev,
        drcof,
        lres,
        lstan,
        lptn
    ) = ADF11_reader.xxdata_11(
        # unit to which input file is allocated
        iunit  = file_unit,
        # class of data (numerical code)
        iclass = ADF11Codes[dataset.name].value,
        # 
        # Hard-coded values for ADF11 files
        # 
        # maximum number of (sstage, parent, base) blocks in isonuclear master files
        isdimd = 200,
        # maximum number of dens values in isonuclear master files
        iddimd = 40,
        # maximum number of temp values in isonuclear master files
        itdimd = 50,
        # maximum level of partitions
        ndptnl = 4,
        # maximum no. of partitions in one level
        ndptn  = 128,
        # maximum no. of components in a partition
        ndptnc = 256,
        # maximum number of elements in connection vector
        ndcnct = 100
    )

    fortran_file_handling.close_file(file_unit)

    ds = xr.Dataset()

    ds["species"] = species.name
    ds["dataset"] = dataset.name
    ds["charge"] = iz0

    electron_density = Quantity(10**ddens[:idmax], ureg.cm**-3).to(ureg.m**-3)
    electron_temperature = Quantity(10**dtev[:itmax], ureg.eV)
    
    # Use logarithmic quantities to define the coordinates, so that we can interpolate over logarithmic quantities.
    ds["electron_density"] = \
       xr.DataArray(electron_density, coords=dict(dim_electron_density=electron_density.magnitude))
    ds["electron_temperature"] = \
       xr.DataArray(electron_temperature, coords=dict(dim_electron_temperature=electron_temperature.magnitude))

    ds = ds.assign_attrs(
        reference_electron_density = Quantity(1.0, ureg.m**-3),
        reference_electron_temperature = Quantity(1.0, ureg.eV),
    )
    
    ds['number_of_charge_states'] = ismax
    charge_state = np.arange(ismax)
    ds["charge_state"] = xr.DataArray(charge_state, coords=dict(dim_charge_state=charge_state))
    
    coefficient = drcof[:ismax, :itmax, :idmax]
    if ADF11Codes[dataset.name].value <= 9:
        coefficient = 10**coefficient
    
    input_units = ADF11RateCoeffStoredUnits[dataset.name].value
    output_units = ADF11RateCoeffDesiredUnits[dataset.name].value
    ds["rate_coefficient"] = xr.DataArray(
       coefficient, dims=("dim_charge_state", "dim_electron_temperature", "dim_electron_density")
    ).pint.quantify(input_units).pint.to(output_units)
    
    return ds
