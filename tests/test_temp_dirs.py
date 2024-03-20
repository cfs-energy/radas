from radas import shared


def test_moddir_monkeypatch(monkeypatch, temp_module_directory):
    "Make sure that we've set up the module directory correctly, and copied the config file into it."
    monkeypatch.setattr(shared, "module_directory", temp_module_directory)
    assert shared.module_directory.resolve() == temp_module_directory.resolve()
    assert (temp_module_directory / "config.yaml").exists()


def test_open_yaml_file(monkeypatch, temp_module_directory):
    monkeypatch.setattr(shared, "module_directory", temp_module_directory)
    configuration = shared.open_yaml_file()

    assert "globals" in configuration
    assert "data_file_config" in configuration
    assert "species" in configuration


def test_open_yaml_file_from_specified_path(temp_module_directory):
    configuration = shared.open_yaml_file(temp_module_directory / "config.yaml")

    assert "globals" in configuration
    assert "data_file_config" in configuration
    assert "species" in configuration
