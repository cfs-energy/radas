from pathlib import Path
import importlib.util


def load_library(library_name: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(library_name, filepath)

    library = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(library)

    return library


def read_adf11_file(
    reader_dir, data_file_dir, species_name, dataset_type, dataset_config
) -> dict:
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
    fortran_file_handling = load_library(
        "fortran_file_handling", reader_dir / "fortran_file_handling.so"
    )
    adf11_reader = load_library(
        "adf11_reader", reader_dir / "adf11" / "adf11_reader.so"
    )

    filename = data_file_dir / f"{species_name}_{dataset_type}.dat"

    if not filename.exists():
        raise FileNotFoundError(f"{filename} does not exist.")

    file_unit = 10

    data = dict()

    fortran_file_handling.open_file(str(filename), file_unit)
    (
        data["iz0"],
        data["is1min"],
        data["is1max"],
        data["nptnl"],
        data["nptn"],
        data["nptnc"],
        data["iptnla"],
        data["iptna"],
        data["iptnca"],
        data["ncnct"],
        data["icnctv"],
        data["iblmx"],
        data["ismax"],
        data["dnr_ele"],
        data["dnr_ams"],
        data["isppr"],
        data["ispbr"],
        data["isstgr"],
        data["idmax"],
        data["itmax"],
        data["ddens"],
        data["dtev"],
        data["drcof"],
        data["lres"],
        data["lstan"],
        data["lptn"],
    ) = adf11_reader.xxdata_11(
        # unit to which input file is allocated
        iunit=file_unit,
        # class of data (numerical code)
        iclass=dataset_config["code"],
        #
        # Hard-coded values for ADF11 files
        #
        # maximum number of (sstage, parent, base) blocks in isonuclear master files
        isdimd=200,
        # maximum number of dens values in isonuclear master files
        iddimd=40,
        # maximum number of temp values in isonuclear master files
        itdimd=50,
        # maximum level of partitions
        ndptnl=4,
        # maximum no. of partitions in one level
        ndptn=128,
        # maximum no. of components in a partition
        ndptnc=256,
        # maximum number of elements in connection vector
        ndcnct=100,
    )

    fortran_file_handling.close_file(file_unit)

    return data
