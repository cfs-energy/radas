from pathlib import Path
import subprocess

def compile_with_f2py(
    files_to_compile: list[str], module_name: str, output_folder: Path
):
    """Compiles a list of fortran files into a module."""

    def compile_fortran_files(quiet: bool):
        command = [
            "python3", "-m", "numpy.f2py", "-c"
        ] + files_to_compile + ["-m", module_name]

        return subprocess.run(command, capture_output=quiet, check=not quiet)

    result = compile_fortran_files(True)

    if not result.returncode == 0:
        # If the first compile attempt fails, do it again
        # and this time print output
        compile_fortran_files(False)

    for file in Path().iterdir():
        if file.name.startswith(module_name):
            file.rename(output_folder / file.name)
