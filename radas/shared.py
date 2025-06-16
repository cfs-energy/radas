"""Bits and pieces shared by many other parts of radas."""

from pathlib import Path
from importlib.resources import files
import subprocess
import yaml

default_config_file = files("radas").joinpath("config.yaml")
mavrin_data_file = files("radas.mavrin_reference").joinpath("mavrin_data.yaml")

library_extensions = [".a", ".so"]


def get_git_revision_short_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except: # noqa:E722
        # If git isn't available (sometimes the case in tests), return a blank
        return "UNDEFINED"


def open_yaml_file(yaml_file: Path) -> dict:
    with open(yaml_file, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)
