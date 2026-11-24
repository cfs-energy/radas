"""Bits and pieces shared by many other parts of radas."""

from pathlib import Path
from importlib.resources import files
from importlib.resources.abc import Traversable
import yaml

default_config_file = files("radas").joinpath("config.yaml")
mavrin_data_file = files("radas.mavrin_reference").joinpath("mavrin_data.yaml")

library_extensions = [".a", ".so"]


def open_yaml_file(yaml_file: Path | Traversable) -> dict:
    """Load a yaml file into a dictionary."""
    with yaml_file.open() as file:
        return yaml.load(file, Loader=yaml.FullLoader)
