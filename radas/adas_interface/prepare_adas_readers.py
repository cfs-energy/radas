import urllib.request
import shutil
from pathlib import Path
from .compile_with_f2py import compile_with_f2py


def prepare_adas_fortran_interface(reader_dir: Path, config: dict, verbose: int):
    from ..shared import (
        fortran_file_handling_source,
        library_extensions,
    )

    fortran_file_handling_library = None
    for file in reader_dir.iterdir():
        if (
            file.name.startswith("fortran_file_handling")
            and file.suffix in library_extensions
        ):
            fortran_file_handling_library = file

    if fortran_file_handling_library is None:
        if verbose:
            print(f"Compiling {reader_dir / 'fortran_file_handling.f90'}")
        (reader_dir / "fortran_file_handling.f90").write_text(
            fortran_file_handling_source.read_text()
        )

        fortran_file_handling_library = compile_with_f2py(
            files_to_compile=[reader_dir / "fortran_file_handling.f90"],
            module_name="fortran_file_handling",
            output_folder=reader_dir,
        )
    elif verbose:
        print(f"Reusing {fortran_file_handling_library}")

    adf_file_reader = dict()
    for reader_name in config.keys():
        adf_file_reader[reader_name] = None

        (reader_dir / reader_name).mkdir(exist_ok=True, parents=True)
        for file in (reader_dir / reader_name).iterdir():
            if file.name.startswith(reader_name) and file.suffix in library_extensions:
                adf_file_reader[reader_name] = file

        if adf_file_reader[reader_name] is None:
            build_adas_file_reader(reader_dir, reader_name, verbose=verbose)
        elif verbose:
            print(f"Reusing {reader_name}_reader")


def build_adas_file_reader(
    reader_dir: Path,
    reader_name: str,
    verbose: int,
    url_base: str = "https://open.adas.ac.uk",
):
    """Builds an ADAS file reader by wrapping fortran code using f2py."""
    from ..shared import reader_pyf_source

    assert reader_name.startswith(
        "adf"
    ), f"Reader should be of the format adfXX where XX is an integer"

    reader_int = int(reader_name.lstrip("adf"))
    output_folder = reader_dir / reader_name
    (output_folder / f"xxdata_{reader_int}.pyf").write_text(
        reader_pyf_source[reader_name].read_text()
    )

    archive_file = f"xxdata_{reader_int}.tar.gz"
    query_path = f"{url_base}/code/{archive_file}"
    output_filename = output_folder / archive_file

    if not output_filename.exists():
        if verbose:
            print(f"Downloading {query_path} to {output_filename}")
        urllib.request.urlretrieve(query_path, output_filename)
        if verbose:
            print(f"Unpacking {output_filename} into {output_folder}")
        shutil.unpack_archive(output_filename, output_folder)
    else:
        if verbose:
            print(f"Reusing {query_path} ({output_filename} already exists)")

    fortran_files = [
        str(file)
        for file in (output_folder / f"xxdata_{reader_int}").iterdir()
        if (file.suffix == ".for") and not (file.stem == "test")
    ]

    fortran_files = fortran_files + [str(output_folder / f"xxdata_{reader_int}.pyf")]

    if verbose:
        print(f"Compiling {reader_name}")
    compile_with_f2py(
        files_to_compile=fortran_files,
        module_name=f"{reader_name}_reader",
        output_folder=output_folder,
    )
