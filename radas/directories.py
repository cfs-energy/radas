from pathlib import Path

module_directory = Path(__file__).parent
repository_directory = module_directory.parent
adas_data_directory = repository_directory / "adas_data"
dat_files_directory = adas_data_directory / "dat_files"
cases_directory = repository_directory / "cases"
