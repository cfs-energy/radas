"""Bits and pieces shared by many other parts of radas."""
from pathlib import Path
import subprocess
import yaml

module_directory = Path(__file__).parent
data_file_directory = module_directory / ".data_files"
repository_directory = module_directory.parent
output_directory = repository_directory / "output"

def get_git_revision_short_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except:
        # If git isn't available (sometimes the case in tests), return a blank
        return ""

def open_config_file(config: str | None) -> dict:
    config_file = module_directory / "config.yaml" if config is None else Path(config)

    with open(config_file, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)
