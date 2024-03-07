import urllib.request
import shutil
from .compile_with_f2py import compile_with_f2py
from ..shared import module_directory


def prepare_adas_fortran_interface(data_file_config: dict):
    compile_with_f2py(
        files_to_compile=[module_directory / "readers" / "fortran_file_handling.f90"],
        module_name="fortran_file_handling",
        output_folder=module_directory / "readers",
    )

    for reader_name in data_file_config.keys():
        reader_found = False
        for file in (module_directory / "readers" / reader_name).iterdir():
            if file.name.startswith(f"{reader_name}_reader"):
                reader_found = True

        if not reader_found:
            build_adas_file_reader(reader_name)


def build_adas_file_reader(reader_name: str, url_base: str = "https://open.adas.ac.uk"):
    """Builds an ADAS file reader by wrapping fortran code using f2py."""
    assert reader_name.startswith(
        "adf"
    ), f"Reader should be of the format adfXX where XX is an integer"
    reader_int = int(reader_name.lstrip("adf"))
    output_folder = module_directory / "readers" / reader_name

    archive_file = f"xxdata_{reader_int}.tar.gz"
    query_path = f"{url_base}/code/{archive_file}"
    urllib.request.urlretrieve(query_path, output_folder / archive_file)
    shutil.unpack_archive(output_folder / archive_file, output_folder)

    fortran_files = [
        str(file)
        for file in (output_folder / f"xxdata_{reader_int}").iterdir()
        if (file.suffix == ".for") and not (file.stem == "test")
    ]

    fortran_files = fortran_files + [str(output_folder / f"xxdata_{reader_int}.pyf")]

    compile_with_f2py(
        files_to_compile=fortran_files,
        module_name=str(f"{reader_name}_reader"),
        output_folder=output_folder,
    )
