from pathlib import Path
import subprocess


def compile_with_f2py(
    files_to_compile: list[str],
    module_name: str,
    output_folder: Path,
) -> Path:
    """Compiles a list of fortran files into a module."""
    from ..shared import library_extensions

    def compile_fortran_files(quiet: bool):
        command = (
            ["python3", "-m", "numpy.f2py", "-c"]
            + files_to_compile
            + ["-m", module_name]
        )

        return subprocess.run(command, capture_output=quiet, check=not quiet)

    result = compile_fortran_files(True)

    if not result.returncode == 0:
        # If the first compile attempt fails, do it again
        # and this time print output
        compile_fortran_files(False)

    library_found = []
    for file in Path(".").iterdir():
        if file.name.startswith(module_name) and file.suffix in library_extensions:
            output_name = output_folder / f"{module_name}{file.suffix}"
            library_found.append(output_name)
            file.rename(output_name)

    assert len(library_found) > 0, f"Could not find compiled library for {module_name}"
    assert len(library_found) == 1, f"Found multiple compiled objects for {module_name}"
    return library_found[0]
