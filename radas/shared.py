"""Bits and pieces shared by many other parts of radas."""

from pathlib import Path
from importlib.resources import files
import yaml

default_config_file = files("radas").joinpath("config.yaml")
mavrin_data_file = files("radas.mavrin_reference").joinpath("mavrin_data.yaml")

library_extensions = [".a", ".so"]


def open_yaml_file(yaml_file: Path) -> dict:
    with open(yaml_file, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)
